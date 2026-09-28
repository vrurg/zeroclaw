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
    goal_accounting::{ModelProviderGoalUsagePricer, RuntimeGoalOperationAccountingFactory},
    rpc::context::RpcContext,
};

use zeroclaw_commands::goal::GoalCommand;
use zeroclaw_goal::control::{
    GoalExecutionNotice, GoalExecutionScope, GoalExecutionSupervisor, GoalHostSettings,
    GoalIngressContext, GoalIngressPrincipal, GoalOperationScope, GoalParentTurn,
    GoalParentTurnKind, GoalParentTurnResult, GoalResponse, GoalRuntime, GoalSessionBinding,
    GoalSessionDriver, GoalSessionExecutionLease, GoalSessionKey, GoalSessionLease, GoalSurface,
    GoalVerifierAttribution, GoalVerifierResult, GoalVerifierTurn, dispose_unowned_session_goal,
};

const GOAL_UPDATE_METHOD: &str = "session/goal_update";

/// The immutable, trusted ZeroCode binding for one Goal admission.
pub struct ZeroCodeGoalSessionDriver {
    context: Arc<RpcContext>,
    authorization: super::dispatch::GoalSessionAuthorization,
    outbound: Arc<RpcOutbound>,
    // Held by the worker-owned driver so the RPC generation cannot finish
    // draining while a Goal it admitted is still unwinding.
    connection_activity: Option<crate::rpc::ConnectionActivity>,
    session_key: GoalSessionKey,
    agent_alias: String,
    session_generation: u64,
    tui_id: String,
    command_lock: Arc<Mutex<()>>,
}

impl ZeroCodeGoalSessionDriver {
    /// Capture the live session facts that command text cannot control.
    pub(super) async fn new(
        context: Arc<RpcContext>,
        outbound: Arc<RpcOutbound>,
        connection_activity: Option<crate::rpc::ConnectionActivity>,
        session_id: String,
        tui_id: String,
        authorization: super::dispatch::GoalSessionAuthorization,
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
            authorization,
            outbound,
            connection_activity,
            session_key,
            agent_alias: snapshot.agent_alias,
            session_generation: snapshot.generation,
            tui_id,
            command_lock: snapshot.goal_command,
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
            GoalIngressPrincipal::ZeroCode { tui_id }
                if tui_id.as_str() == self.tui_id.as_str() =>
            {
                Ok(())
            }
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
        let guard = self.command_lock.clone().lock_owned().await;
        self.revalidate_session().await?;
        self.authorization.revalidate().await?;
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
            history_has_trim_breadcrumb: false,
            outbound: Arc::clone(&self.outbound),
            context: Arc::clone(&self.context),
            session_key: self.session_key.clone(),
        }))
    }
}

struct ZeroCodeGoalExecutionLease {
    _queue_guard: zeroclaw_infra::session_queue::SessionGuard,
    agent: Arc<Mutex<Agent>>,
    canonical_history: Vec<ChatMessage>,
    history_has_trim_breadcrumb: bool,
    outbound: Arc<RpcOutbound>,
    context: Arc<RpcContext>,
    session_key: GoalSessionKey,
}

impl ZeroCodeGoalExecutionLease {
    fn raw_session_id(&self) -> Result<&str> {
        match &self.session_key {
            GoalSessionKey::ZeroCode { raw_session_id } => Ok(raw_session_id),
            GoalSessionKey::Matrix { .. } => bail!("ZeroCode Goal lease has a Matrix session key"),
        }
    }

    async fn publish_parent_turn_failure(&self, error: String) {
        let Ok(session_id) = self.raw_session_id() else {
            return;
        };
        super::dispatch::forward_turn_completion(
            &self.outbound,
            session_id,
            crate::rpc::types::TurnCompletionOutcome::Failed,
            format!("turn failed: Agent turn failed: {error}"),
            None,
            None,
        )
        .await;
    }
}

fn verifier_attribution_from_attempts<'a>(
    attempts: impl IntoIterator<Item = (bool, &'a str, &'a str)>,
) -> Option<GoalVerifierAttribution> {
    attempts
        .into_iter()
        .filter(|(accepted, _, _)| *accepted)
        .map(|(_, provider_ref, model)| GoalVerifierAttribution {
            provider_ref: provider_ref.to_owned(),
            model: model.to_owned(),
        })
        .last()
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
    fn session_key(&self) -> &GoalSessionKey {
        &self.session_key
    }

    fn canonical_history(&self) -> Result<Vec<ChatMessage>> {
        Ok(self.canonical_history.clone())
    }

    async fn append_resume_response(&mut self, response: String) -> Result<()> {
        let session_key = format!("rpc_{}", self.raw_session_id()?);
        let message = ChatMessage::user(&response);
        // Persist before changing the live agent. A successful resume must
        // never be visible only to this process and then vanish when the
        // same ZeroCode session is reconstructed after daemon restart.
        let backend = self
            .context
            .session_backend
            .as_ref()
            .context("cannot resume Goal response: ZeroCode session persistence is disabled")?;
        backend
            .append(&session_key, &message)
            .context("persist ZeroCode Goal resume response")?;
        self.agent.lock().await.seed_history(&[message]);
        Ok(())
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
            GoalParentTurnKind::Continue => IsolatedTranscriptSource::Continuation {
                history: turn.working_history,
                history_has_trim_breadcrumb: self.history_has_trim_breadcrumb,
            },
        };
        let session_id = self.raw_session_id()?.to_owned();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(64);
        let agent = Arc::clone(&self.agent);
        let run = async move {
            agent
                .lock()
                .await
                .run_isolated_turn_with_events(source, directive, event_tx)
                .await
        };
        tokio::pin!(run);

        let mut event_rx_open = true;
        let result = loop {
            tokio::select! {
                outcome = &mut run => {
                    while let Some(event) = event_rx.recv().await {
                        super::dispatch::forward_turn_event(&self.outbound, &session_id, &event).await;
                    }
                    break outcome;
                }
                event = event_rx.recv(), if event_rx_open => {
                    match event {
                        Some(event) => {
                            super::dispatch::forward_turn_event(&self.outbound, &session_id, &event).await;
                        }
                        None => event_rx_open = false,
                    }
                }
            }
        };
        let outcome = match result {
            Ok(outcome) => outcome,
            Err(error) => {
                self.publish_parent_turn_failure(error.to_string()).await;
                return Err(error);
            }
        };
        self.history_has_trim_breadcrumb = outcome.history_has_trim_breadcrumb;
        GoalParentTurnResult::from_completed_agent_turn(outcome.response, outcome.working_history)
    }

    async fn run_verifier(
        &mut self,
        _operation: &GoalOperationScope,
        turn: GoalVerifierTurn,
    ) -> Result<GoalVerifierResult> {
        let (provider, model_selector, route_resolver) = {
            let config = self.context.config.read();
            let (provider, _provider_ref, model_selector, route_resolver) =
                build_session_model_provider(
                    &config,
                    &config.goal.verifier.model_provider,
                    config.goal.verifier.model.as_deref(),
                )?;
            (provider, model_selector, route_resolver)
        };
        let route = route_resolver.resolve(&model_selector);
        let messages = turn.to_chat_messages();
        let mut settled_attempts = Vec::new();
        let response = crate::agent::loop_::ResolvedModelAccess {
            model_provider: provider.as_ref(),
            provider_name: &route.provider_name,
            model: &route.model,
            dispatch_model: &model_selector,
            temperature: None,
        }
        .run_model_query(
            ChatRequest {
                messages: &messages,
                tools: None,
                thinking: None,
            },
            &mut settled_attempts,
        )
        .await;
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                self.publish_parent_turn_failure(error.to_string()).await;
                return Err(error);
            }
        };
        let response = response.text.context("Goal verifier returned no text")?;
        let attribution =
            verifier_attribution_from_attempts(settled_attempts.iter().map(|attempt| {
                (
                    attempt.accepted,
                    attempt.provider_ref.as_str(),
                    attempt.model.as_str(),
                )
            }));
        Ok(GoalVerifierResult {
            response,
            attribution,
        })
    }

    async fn append_parent_candidate(&mut self, candidate: String) -> Result<()> {
        let session_key = format!("rpc_{}", self.raw_session_id()?);
        let message = ChatMessage::assistant(&candidate);
        let backend = self.context.session_backend.as_ref().context(
            "cannot publish Goal parent response: ZeroCode session persistence is disabled",
        )?;
        if let Err(error) = backend.append(&session_key, &message) {
            self.publish_parent_turn_failure(error.to_string()).await;
            return Err(error).context("persist ZeroCode Goal parent response");
        }
        self.agent.lock().await.seed_history(&[message]);
        self.outbound
            .notify(
                GOAL_UPDATE_METHOD,
                serde_json::to_value(crate::rpc::types::SessionGoalUpdate::VerifiedCandidate {
                    session_id: self.raw_session_id()?.to_owned(),
                    candidate,
                })?,
            )
            .await;
        Ok(())
    }

    async fn publish_goal_notice(&mut self, notice: GoalExecutionNotice) -> Result<()> {
        let update = match notice {
            GoalExecutionNotice::Completed => crate::rpc::types::SessionGoalUpdate::Completed {
                session_id: self.raw_session_id()?.to_owned(),
            },
            GoalExecutionNotice::PausedForBlocker => {
                crate::rpc::types::SessionGoalUpdate::PausedForBlocker {
                    session_id: self.raw_session_id()?.to_owned(),
                }
            }
        };
        self.outbound
            .notify(GOAL_UPDATE_METHOD, serde_json::to_value(update)?)
            .await;
        Ok(())
    }
}

#[cfg(test)]
mod verifier_attribution_tests {
    use super::{GoalVerifierAttribution, verifier_attribution_from_attempts};

    #[test]
    fn reports_the_provider_and_model_of_the_accepted_attempt() {
        let attribution = verifier_attribution_from_attempts([
            (false, "openai.primary", "gpt-primary"),
            (true, "anthropic.fallback", "claude-fallback"),
        ]);

        assert_eq!(
            attribution,
            Some(GoalVerifierAttribution {
                provider_ref: "anthropic.fallback".to_owned(),
                model: "claude-fallback".to_owned(),
            })
        );
    }

    #[test]
    fn leaves_attribution_absent_when_no_attempt_was_accepted() {
        let attribution = verifier_attribution_from_attempts([
            (false, "openai.primary", "gpt-primary"),
            (false, "anthropic.fallback", "claude-fallback"),
        ]);

        assert_eq!(attribution, None);
    }
}

/// ZeroCode's process-local Goal worker registry.
#[derive(Default)]
pub struct RpcGoalRuntime {
    supervisors: Mutex<HashMap<String, Arc<GoalExecutionSupervisor>>>,
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
            self.remove_supervisor_if_current(session_id, &supervisor)
                .await;
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
        let settings = GoalHostSettings::from_config(
            &config.goal,
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
                let pricer = Arc::new(ModelProviderGoalUsagePricer::new(Arc::new(
                    build_type_level_model_provider_pricing(&config),
                )));
                let accounting = Arc::new(RuntimeGoalOperationAccountingFactory::new(
                    Arc::clone(&registry),
                    tracker,
                    driver.agent_alias().to_owned(),
                    pricer,
                )?);
                let engine = Arc::new(runtime.execution_engine(accounting));
                let candidate = restart_coordinator.new_supervisor(engine).await;
                self.install_supervisor_if_absent(session_id.clone(), candidate)
                    .await
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
        let submission = supervisor
            .submit(settings, ingress, driver, command)
            .await?;
        if matches!(
            submission.response(),
            GoalResponse::Paused(_)
                | GoalResponse::AlreadyPaused(_)
                | GoalResponse::Cancelled(_)
                | GoalResponse::AlreadyCancelled(_)
                | GoalResponse::Terminal(_)
        ) {
            self.remove_supervisor_if_current(&session_id, &supervisor)
                .await;
        }
        Ok(submission.into_response())
    }
    /// Return a resident supervisor for this session, if it has one.
    pub async fn supervisor(&self, session_id: &str) -> Option<Arc<GoalExecutionSupervisor>> {
        self.supervisors.lock().await.get(session_id).cloned()
    }

    /// Atomically retain one supervisor for a session and return the resident
    /// instance. A concurrent candidate was never used to launch work, so it
    /// safely drops after losing this insertion race.
    pub async fn install_supervisor_if_absent(
        &self,
        session_id: String,
        supervisor: Arc<GoalExecutionSupervisor>,
    ) -> Arc<GoalExecutionSupervisor> {
        Arc::clone(
            self.supervisors
                .lock()
                .await
                .entry(session_id)
                .or_insert(supervisor),
        )
    }

    /// Remove a supervisor only if it is still the instance that completed the
    /// current lifecycle command.
    pub async fn remove_supervisor_if_current(
        &self,
        session_id: &str,
        supervisor: &Arc<GoalExecutionSupervisor>,
    ) {
        let mut supervisors = self.supervisors.lock().await;
        if supervisors
            .get(session_id)
            .is_some_and(|current| Arc::ptr_eq(current, supervisor))
        {
            supervisors.remove(session_id);
        }
    }
}
