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
    goal_mode::{
        GoalExecutionNotice, GoalExecutionScope, GoalIngressContext, GoalIngressPrincipal,
        GoalOperationScope, GoalParentTurn, GoalParentTurnKind, GoalParentTurnResult,
        GoalSessionBinding, GoalSessionDriver, GoalSessionExecutionLease, GoalSessionKey,
        GoalSessionLease, GoalSurface, GoalVerifierTurn,
    },
    rpc::context::RpcContext,
};

use crate::goal_mode::GoalExecutionSupervisor;

const GOAL_UPDATE_METHOD: &str = "session/goal_update";

/// The immutable, trusted ZeroCode binding for one Goal admission.
pub struct ZeroCodeGoalSessionDriver {
    context: Arc<RpcContext>,
    outbound: Arc<RpcOutbound>,
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
        session_id: String,
        tui_id: String,
    ) -> Result<Self> {
        let session_key = GoalSessionKey::zero_code(session_id)?;
        let raw_session_id = match &session_key {
            GoalSessionKey::ZeroCode { raw_session_id } => raw_session_id,
            GoalSessionKey::Matrix { .. } => unreachable!("ZeroCode key constructor is typed"),
        };
        let session_generation = context
            .sessions
            .get_generation(raw_session_id)
            .await
            .context("ZeroCode Goal session is absent")?;
        let agent_alias = context
            .sessions
            .get_agent_alias(raw_session_id)
            .await
            .context("ZeroCode Goal session has no agent")?;
        ensure!(
            matches!(
                context.sessions.chat_mode(raw_session_id).await,
                Some(crate::rpc::types::ChatMode::Chat)
            ),
            "Goal Mode is unavailable for ACP sessions"
        );
        Ok(Self {
            context,
            outbound,
            session_key,
            agent_alias,
            session_generation,
            tui_id,
        })
    }

    fn raw_session_id(&self) -> &str {
        match &self.session_key {
            GoalSessionKey::ZeroCode { raw_session_id } => raw_session_id,
            GoalSessionKey::Matrix { .. } => unreachable!("ZeroCode driver has a typed key"),
        }
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
        ensure!(
            self.context
                .sessions
                .get_generation(self.raw_session_id())
                .await
                == Some(self.session_generation),
            "ZeroCode Goal session was replaced"
        );
        let agent = self
            .context
            .sessions
            .get_agent(self.raw_session_id())
            .await
            .context("ZeroCode Goal session is absent")?;
        ensure!(
            self.context
                .sessions
                .get_agent_alias(self.raw_session_id())
                .await
                .as_deref()
                == Some(self.agent_alias.as_str()),
            "ZeroCode Goal session agent changed"
        );
        Ok(agent)
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
            GoalSessionBinding::new(
                self.session_key.clone(),
                self.session_generation.to_string(),
            )?,
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
