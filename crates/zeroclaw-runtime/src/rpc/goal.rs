//! Process-local ZeroCode ownership for Goal execution workers.
//!
//! Durable lifecycle and accounting state remain in the control plane. This
//! registry only retains the exact join handles needed to drain a session's
//! fenced execution epoch when an RPC lifecycle operation replaces or removes
//! that session.

use std::{collections::HashMap, sync::Arc};

use anyhow::{Context as _, Result, bail, ensure};
use async_trait::async_trait;
use tokio::sync::Mutex;
use zeroclaw_api::{
    jsonrpc::RpcOutbound,
    model_provider::{ChatMessage, ChatRequest},
};

use crate::{
    agent::agent::{Agent, IsolatedTranscriptSource, build_session_model_provider},
    agent::cost::build_type_level_model_provider_pricing,
    control_plane::control_plane,
    cost::CostTracker,
    goal_mode::{
        GoalExecutionNotice, GoalExecutionScope, GoalHostSettings, GoalIngressContext,
        GoalIngressPrincipal, GoalOperationScope, GoalParentTurn, GoalParentTurnKind,
        GoalParentTurnResult, GoalResponse, GoalRuntime, GoalSessionBinding, GoalSessionDriver,
        GoalSessionExecutionLease, GoalSessionKey, GoalSessionLease, GoalSurface, GoalVerifierTurn,
        dispose_unowned_session_goal,
    },
    rpc::context::RpcContext,
};

use crate::goal_mode::GoalExecutionSupervisor;
use zeroclaw_commands::goal::GoalCommand;

const GOAL_UPDATE_METHOD: &str = "session/goal_update";

/// The immutable, trusted ZeroCode binding for one Goal admission.
pub struct ZeroCodeGoalSessionDriver {
    context: Arc<RpcContext>,
    outbound: Arc<RpcOutbound>,
    // Held by the worker-owned driver so the RPC generation cannot finish
    // draining while a Goal it admitted is still unwinding.
    connection_activity: Option<crate::rpc::ConnectionActivity>,
    session_key: GoalSessionKey,
    agent_alias: String,
    session_generation: u64,
    tui_id: String,
}

impl ZeroCodeGoalSessionDriver {
    /// Capture the live session facts that command text cannot control.
    pub async fn new(
        context: Arc<RpcContext>,
        outbound: Arc<RpcOutbound>,
        connection_activity: Option<crate::rpc::ConnectionActivity>,
        session_id: String,
        tui_id: String,
    ) -> Result<Self> {
        let session_key = GoalSessionKey::zero_code(session_id)?;
        let raw_session_id = match &session_key {
            GoalSessionKey::ZeroCode { raw_session_id } => raw_session_id,
            GoalSessionKey::Matrix { .. } => unreachable!("ZeroCode key constructor is typed"),
        };
        let snapshot = context
            .sessions
            .goal_session_snapshot(raw_session_id)
            .await
            .context("ZeroCode Goal session is absent")?;
        ensure!(
            snapshot.owner_tui_id.as_deref() == Some(tui_id.as_str()),
            "ZeroCode Goal caller does not own the session"
        );
        ensure!(
            matches!(snapshot.chat_mode, crate::rpc::types::ChatMode::Chat),
            "Goal Mode is unavailable for ACP sessions"
        );
        Ok(Self {
            context,
            outbound,
            connection_activity,
            session_key,
            agent_alias: snapshot.agent_alias,
            session_generation: snapshot.generation,
            tui_id,
        })
    }

    fn raw_session_id(&self) -> &str {
        match &self.session_key {
            GoalSessionKey::ZeroCode { raw_session_id } => raw_session_id,
            GoalSessionKey::Matrix { .. } => unreachable!("ZeroCode driver has a typed key"),
        }
    }

    pub fn agent_alias(&self) -> &str {
        &self.agent_alias
    }

    fn assert_ingress(&self, ingress: &GoalIngressContext) -> Result<()> {
        ensure!(
            ingress.surface() == GoalSurface::ZeroCode,
            "Goal ingress is not ZeroCode"
        );
        ensure!(
            ingress.session_key() == &self.session_key,
            "Goal ingress session is stale"
        );
        ensure!(
            ingress.agent() == self.agent_alias,
            "Goal ingress agent is stale"
        );
        ensure!(
            ingress.route() == format!("zerocode:{}", self.raw_session_id()),
            "Goal ingress route is stale"
        );
        match ingress.principal() {
            GoalIngressPrincipal::ZeroCode { tui_id } if tui_id == &self.tui_id => Ok(()),
            _ => bail!("Goal ingress principal is stale"),
        }
    }

    async fn revalidate_session(&self) -> Result<Arc<Mutex<Agent>>> {
        let snapshot = self
            .context
            .sessions
            .goal_session_snapshot(self.raw_session_id())
            .await
            .context("ZeroCode Goal session is absent")?;
        ensure!(
            snapshot.generation == self.session_generation,
            "ZeroCode Goal session was replaced"
        );
        ensure!(
            snapshot.agent_alias == self.agent_alias,
            "ZeroCode Goal session agent changed"
        );
        ensure!(
            snapshot.owner_tui_id.as_deref() == Some(self.tui_id.as_str()),
            "ZeroCode Goal caller no longer owns the session"
        );
        ensure!(
            matches!(snapshot.chat_mode, crate::rpc::types::ChatMode::Chat),
            "Goal Mode is unavailable for ACP sessions"
        );
        Ok(snapshot.agent)
    }
}

#[async_trait]
impl GoalSessionDriver for ZeroCodeGoalSessionDriver {
    fn surface(&self) -> GoalSurface {
        GoalSurface::ZeroCode
    }

    fn session_key(&self) -> &GoalSessionKey {
        &self.session_key
    }

    async fn bind(&self, ingress: &GoalIngressContext) -> Result<GoalSessionLease> {
        // This is deliberately a liveness token rather than a cancellation
        // signal. Ordinary EOF must retain the durable Goal, while reload must
        // wait until the worker has released the driver.
        let _connection_generation = self.connection_activity.as_ref();
        self.assert_ingress(ingress)?;
        self.revalidate_session().await?;
        let lock = self
            .context
            .goal_runtime
            .command_lock(self.raw_session_id())
            .await;
        let guard = lock.lock_owned().await;
        self.revalidate_session().await?;
        Ok(GoalSessionLease::new(
            GoalSessionBinding::new(self.session_key.clone()),
            guard,
        ))
    }

    async fn acquire_execution(
        &self,
        ingress: &GoalIngressContext,
        scope: &GoalExecutionScope,
    ) -> Result<Box<dyn GoalSessionExecutionLease>> {
        self.assert_ingress(ingress)?;
        ensure!(
            scope.session_id() == self.session_key.durable_id(),
            "Goal execution session is stale"
        );
        let queue_guard = self
            .context
            .sessions
            .session_queue
            .acquire(self.raw_session_id())
            .await
            .context("ZeroCode Goal session is busy")?;
        let agent = self.revalidate_session().await?;
        let canonical_history = agent.lock().await.isolated_canonical_prefix()?;
        Ok(Box::new(ZeroCodeGoalExecutionLease {
            _queue_guard: queue_guard,
            agent,
            canonical_history,
            outbound: Arc::clone(&self.outbound),
            context: Arc::clone(&self.context),
            session_id: self.raw_session_id().to_owned(),
        }))
    }
}

struct ZeroCodeGoalExecutionLease {
    _queue_guard: zeroclaw_infra::session_queue::SessionGuard,
    agent: Arc<Mutex<Agent>>,
    canonical_history: Vec<ChatMessage>,
    outbound: Arc<RpcOutbound>,
    context: Arc<RpcContext>,
    session_id: String,
}

fn goal_parent_directive(turn: &GoalParentTurn) -> ChatMessage {
    let kind = match turn.kind {
        GoalParentTurnKind::Start => "start",
        GoalParentTurnKind::Resume => "resume",
        GoalParentTurnKind::Continue => "continue",
    };
    ChatMessage::system(format!(
        "Goal success criterion (trusted runtime directive): {}\nTurn kind: {kind}",
        turn.objective
    ))
}

#[async_trait]
impl GoalSessionExecutionLease for ZeroCodeGoalExecutionLease {
    fn canonical_history(&self) -> Result<Vec<ChatMessage>> {
        Ok(self.canonical_history.clone())
    }

    async fn run_parent_turn(
        &mut self,
        _operation: &GoalOperationScope,
        turn: GoalParentTurn,
    ) -> Result<GoalParentTurnResult> {
        let directive = goal_parent_directive(&turn);
        let source = match turn.kind {
            GoalParentTurnKind::Start | GoalParentTurnKind::Resume => {
                IsolatedTranscriptSource::Canonical {
                    prefix: turn.working_history,
                }
            }
            GoalParentTurnKind::Continue => {
                IsolatedTranscriptSource::Continuation(turn.working_history)
            }
        };
        let outcome = self
            .agent
            .lock()
            .await
            .run_isolated_turn(source, directive)
            .await?;
        Ok(GoalParentTurnResult {
            candidate: outcome.response,
            working_history: outcome.working_history,
        })
    }

    async fn run_verifier(
        &mut self,
        _operation: &GoalOperationScope,
        turn: GoalVerifierTurn,
    ) -> Result<String> {
        let config = self.context.config.read().clone();
        let (provider, provider_name, model) = build_session_model_provider(
            &config,
            &config.goal.verifier.model_provider,
            config.goal.verifier.model.as_deref(),
        )?;
        let messages = vec![
            ChatMessage::system(
                "Return only strict JSON: {\"decision\":\"complete|continue|blocked\",\"reason\":\"...\",\"blockers\":[...]}.",
            ),
            ChatMessage::user(format!(
                "Objective:\n{}\n\nCandidate:\n{}",
                turn.objective, turn.candidate
            )),
        ];
        let response = crate::agent::loop_::ResolvedModelAccess {
            model_provider: provider.as_ref(),
            provider_name: &provider_name,
            model: &model,
            temperature: None,
        }
        .run_model_query(ChatRequest {
            messages: &messages,
            tools: None,
            thinking: None,
        })
        .await?;
        response.text.context("Goal verifier returned no text")
    }

    async fn append_verified_candidate(&mut self, candidate: String) -> Result<()> {
        self.agent
            .lock()
            .await
            .seed_history(&[ChatMessage::assistant(&candidate)]);
        self.outbound
            .notify(
                GOAL_UPDATE_METHOD,
                serde_json::to_value(crate::rpc::types::SessionGoalUpdate::VerifiedCandidate {
                    session_id: self.session_id.clone(),
                    candidate,
                })?,
            )
            .await;
        Ok(())
    }

    async fn publish_goal_notice(&mut self, notice: GoalExecutionNotice) -> Result<()> {
        let update = match notice {
            GoalExecutionNotice::Completed => crate::rpc::types::SessionGoalUpdate::Completed {
                session_id: self.session_id.clone(),
            },
            GoalExecutionNotice::PausedForBlocker => {
                crate::rpc::types::SessionGoalUpdate::PausedForBlocker {
                    session_id: self.session_id.clone(),
                }
            }
        };
        self.outbound
            .notify(GOAL_UPDATE_METHOD, serde_json::to_value(update)?)
            .await;
        Ok(())
    }
}

/// ZeroCode's process-local Goal worker and command-lock registry.
#[derive(Default)]
pub struct RpcGoalRuntime {
    supervisors: Mutex<HashMap<String, Arc<GoalExecutionSupervisor>>>,
    command_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl RpcGoalRuntime {
    /// Fence and dispose the current Goal before its RPC session disappears.
    ///
    /// This must run before the caller waits on `SessionActorQueue`: an active
    /// Goal owns that queue until its admitted operation has settled.
    pub async fn dispose_session(&self, session_id: &str) -> Result<()> {
        if let Some(supervisor) = self.supervisor(session_id).await {
            supervisor
                .dispose_session(&GoalSessionKey::zero_code(session_id)?.durable_id())
                .await?;
            self.remove_supervisor(session_id).await;
        } else if let Some(control_plane) = control_plane() {
            let registry = control_plane.goal_store()?;
            let durable_session_id = GoalSessionKey::zero_code(session_id)?.durable_id();
            let _ = dispose_unowned_session_goal(registry.as_ref(), &durable_session_id).await?;
        }
        Ok(())
    }

    /// Submit one parsed Goal command through the current ZeroCode session.
    pub async fn submit(
        &self,
        context: Arc<RpcContext>,
        driver: Arc<ZeroCodeGoalSessionDriver>,
        command: GoalCommand,
    ) -> Result<GoalResponse> {
        let config = context.config.read().clone();
        if !config.goal.enabled {
            return Ok(GoalResponse::Disabled);
        }
        let control_plane = control_plane().context("Goal control plane is unavailable")?;
        let registry = control_plane.goal_store()?;
        let restart_coordinator = control_plane.goal_execution_restart();
        let limits = config.goal.effective_limits().map_err(|error| {
            anyhow::Error::msg(format!("Goal configuration is invalid: {error:?}"))
        })?;
        let settings = GoalHostSettings::new(
            config.goal.enabled,
            zeroclaw_commands::goal::GoalBudgetLimits {
                token_limit: limits.token_limit,
                cost_limit_usd: limits.cost_limit_usd,
            },
            std::process::id(),
            control_plane.boot_id.clone(),
        )?;
        let session_id = driver.raw_session_id().to_owned();
        let supervisor = match self.supervisor(&session_id).await {
            Some(supervisor) => supervisor,
            None => {
                let runtime = GoalRuntime::new(Arc::clone(&registry));
                let tracker = CostTracker::get_or_init_global_required(
                    config.cost.clone(),
                    &config.data_dir,
                )?;
                let engine = Arc::new(runtime.execution_engine(
                    tracker,
                    driver.agent_alias().to_owned(),
                    Arc::new(build_type_level_model_provider_pricing(&config)),
                )?);
                let supervisor = restart_coordinator.new_supervisor(engine).await;
                self.install_supervisor(session_id.clone(), Arc::clone(&supervisor))
                    .await;
                supervisor
            }
        };
        let ingress = GoalIngressContext::trusted(
            driver.session_key().clone(),
            driver.agent_alias().to_owned(),
            format!("zerocode:{session_id}"),
            GoalIngressPrincipal::ZeroCode {
                tui_id: driver.tui_id.clone(),
            },
        )?;
        let response = supervisor
            .submit(settings, ingress, driver, command)
            .await?;
        if matches!(
            response,
            GoalResponse::Paused(_)
                | GoalResponse::AlreadyPaused(_)
                | GoalResponse::Cancelled(_)
                | GoalResponse::AlreadyCancelled(_)
                | GoalResponse::Terminal(_)
        ) {
            self.remove_supervisor(&session_id).await;
        }
        Ok(response)
    }
    /// Return the command lease for one raw RPC session identifier.
    pub async fn command_lock(&self, session_id: &str) -> Arc<Mutex<()>> {
        let mut locks = self.command_locks.lock().await;
        Arc::clone(
            locks
                .entry(session_id.to_owned())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        )
    }

    /// Return a resident supervisor for this session, if it has one.
    pub async fn supervisor(&self, session_id: &str) -> Option<Arc<GoalExecutionSupervisor>> {
        self.supervisors.lock().await.get(session_id).cloned()
    }

    /// Install the sole supervisor for one session.
    pub async fn install_supervisor(
        &self,
        session_id: String,
        supervisor: Arc<GoalExecutionSupervisor>,
    ) {
        self.supervisors.lock().await.insert(session_id, supervisor);
    }

    /// Remove the resident supervisor after its worker has drained.
    pub async fn remove_supervisor(&self, session_id: &str) {
        self.supervisors.lock().await.remove(session_id);
    }
}
