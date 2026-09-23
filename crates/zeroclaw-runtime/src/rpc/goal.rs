//! Process-local ZeroCode ownership for Goal execution workers.
//!
//! Durable lifecycle and accounting state remain in the control plane. This
//! registry only retains the exact join handles needed to drain a session's
//! fenced execution epoch when an RPC lifecycle operation replaces or removes
//! that session.

use std::{collections::HashMap, sync::Arc};

use anyhow::{Context as _, Result, bail, ensure};
use async_trait::async_trait;
use tokio::sync::{Mutex, mpsc};
use zeroclaw_api::{
    jsonrpc::RpcOutbound,
    model_provider::{ChatMessage, ChatRequest},
};

use crate::{
    agent::agent::{Agent, IsolatedTranscriptSource, TurnEvent, build_session_model_provider},
    agent::cost::build_type_level_model_provider_pricing,
    control_plane::control_plane,
    cost::CostTracker,
    goal_mode::{
        GoalExecutionNotice, GoalExecutionScope, GoalHostSettings, GoalIngressContext,
        GoalIngressPrincipal, GoalOperationScope, GoalParentTurn, GoalParentTurnResult,
        GoalResponse, GoalRuntime, GoalSessionBinding, GoalSessionDriver,
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
    command_lock: Arc<Mutex<()>>,
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
            session_key: self.session_key.clone(),
            agent_alias: self.agent_alias.clone(),
            cancellation: None,
        }))
    }
}

struct ZeroCodeGoalExecutionLease {
    _queue_guard: zeroclaw_infra::session_queue::SessionGuard,
    agent: Arc<Mutex<Agent>>,
    canonical_history: Vec<ChatMessage>,
    outbound: Arc<RpcOutbound>,
    context: Arc<RpcContext>,
    session_key: GoalSessionKey,
    agent_alias: String,
    cancellation: Option<tokio_util::sync::CancellationToken>,
}

impl ZeroCodeGoalExecutionLease {
    fn raw_session_id(&self) -> Result<&str> {
        match &self.session_key {
            GoalSessionKey::ZeroCode { raw_session_id } => Ok(raw_session_id),
            GoalSessionKey::Matrix { .. } => bail!("ZeroCode Goal lease has a Matrix session key"),
        }
    }

    /// Refresh the durable attachment tail for every Goal parent operation.
    /// An RPC `Agent` is long-lived, so its previous ordinary-turn snapshot is
    /// not authority for a later Goal start, resume, or continuation.
    async fn refresh_session_prompt_attachments(&self) -> Result<()> {
        let enabled = self.context.config.read().channels.session_prompts_enabled;
        let attachments = if enabled {
            let backend = self.context.session_backend.as_ref().context(
                "persistent session prompts are enabled but the chat session backend is unavailable",
            )?;
            let session_key = format!("rpc_{}", self.raw_session_id()?);
            let prompts = backend
                .list_session_prompts(&session_key)
                .context("load persistent session prompts for ZeroCode Goal")?;
            zeroclaw_infra::session_prompts::render_session_prompts(&prompts)
        } else {
            String::new()
        };
        self.agent
            .lock()
            .await
            .set_session_prompt_attachments(attachments);
        Ok(())
    }
}

#[async_trait]
impl GoalSessionExecutionLease for ZeroCodeGoalExecutionLease {
    fn session_key(&self) -> &GoalSessionKey {
        &self.session_key
    }

    fn canonical_history(&self) -> Result<Vec<ChatMessage>> {
        Ok(self.canonical_history.clone())
    }

    fn set_execution_cancellation(&mut self, cancellation: tokio_util::sync::CancellationToken) {
        self.cancellation = Some(cancellation);
    }

    fn execution_cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
    }

    async fn run_parent_turn(
        &mut self,
        _operation: &GoalOperationScope,
        turn: GoalParentTurn,
    ) -> Result<GoalParentTurnResult> {
        self.refresh_session_prompt_attachments().await?;
        let directive = crate::goal_mode::goal_parent_directive(&turn);
        let source = isolated_source_for_goal_turn(turn);
        let (event_tx, mut event_rx) = mpsc::channel::<TurnEvent>(64);
        let event_outbound = Arc::clone(&self.outbound);
        let event_context = Arc::clone(&self.context);
        let event_session_id = self.raw_session_id()?.to_owned();
        let event_max_context_tokens = {
            let config = self.context.config.read();
            crate::rpc::dispatch::context_usage_max_tokens(&config, &self.agent_alias)
        };
        let event_relay = zeroclaw_spawn::spawn!(async move {
            while let Some(event) = event_rx.recv().await {
                forward_zerocode_goal_event(
                    event_context.as_ref(),
                    &event_outbound,
                    &event_session_id,
                    event_max_context_tokens,
                    event,
                )
                .await;
            }
        });
        let outcome = self
            .agent
            .lock()
            .await
            .run_isolated_turn_with_cancellation(
                source,
                directive,
                Some(event_tx),
                self.cancellation.clone(),
            )
            .await;
        event_relay.await.map_err(|error| {
            anyhow::Error::msg(format!("ZeroCode Goal event relay panicked: {error}"))
        })?;
        let outcome = outcome?;
        Ok(GoalParentTurnResult {
            candidate: outcome.response,
            working_history: outcome.working_history,
            interruption: outcome.interruption,
        })
    }

    async fn run_verifier(
        &mut self,
        _operation: &GoalOperationScope,
        turn: GoalVerifierTurn,
    ) -> Result<String> {
        let (provider, provider_name, model) = {
            let config = self.context.config.read();
            build_session_model_provider(
                &config,
                &config.goal.verifier.model_provider,
                config.goal.verifier.model.as_deref(),
            )?
        };
        let messages = crate::goal_mode::goal_verifier_messages(&turn);
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

    async fn finish_parent_turn_presentation(&mut self) -> Result<()> {
        self.outbound
            .notify(
                GOAL_UPDATE_METHOD,
                serde_json::to_value(crate::rpc::types::SessionGoalUpdate::ParentTurnFinished {
                    session_id: self.raw_session_id()?.to_owned(),
                })?,
            )
            .await;
        Ok(())
    }

    async fn present_core_error(&mut self, error: &anyhow::Error) -> Result<()> {
        let safe_error = zeroclaw_providers::sanitize_api_error(&error.to_string());
        self.outbound
            .notify(
                GOAL_UPDATE_METHOD,
                serde_json::to_value(crate::rpc::types::SessionGoalUpdate::ParentError {
                    session_id: self.raw_session_id()?.to_owned(),
                    error: safe_error,
                })?,
            )
            .await;
        Ok(())
    }

    async fn record_presented_parent_candidate(&mut self, candidate: String) -> Result<()> {
        self.agent
            .lock()
            .await
            .seed_history(&[ChatMessage::assistant(&candidate)]);
        // `run_parent_turn` has already forwarded the exact agent response via
        // ordinary `session/update` events. Sending the candidate again through
        // a Goal-specific notification would duplicate it for every RPC client.
        Ok(())
    }

    async fn publish_goal_notice(&mut self, notice: GoalExecutionNotice) -> Result<()> {
        let update = match notice {
            GoalExecutionNotice::Completed => crate::rpc::types::SessionGoalUpdate::Completed {
                session_id: self.raw_session_id()?.to_owned(),
            },
            GoalExecutionNotice::PausedForBlocker { pause_reason } => {
                crate::rpc::types::SessionGoalUpdate::PausedForBlocker {
                    session_id: self.raw_session_id()?.to_owned(),
                    pause_reason: Some(pause_reason),
                }
            }
            GoalExecutionNotice::PausedForInterruption => {
                crate::rpc::types::SessionGoalUpdate::PausedForInterruption {
                    session_id: self.raw_session_id()?.to_owned(),
                    message: crate::i18n::get_required_cli_string(
                        "goal-mode-paused-core-interruption-detail",
                    ),
                }
            }
            GoalExecutionNotice::Failed {
                terminal_reason,
                terminal_provider,
                terminal_detail,
            } => crate::rpc::types::SessionGoalUpdate::Failed {
                session_id: self.raw_session_id()?.to_owned(),
                terminal_reason: Some(terminal_reason),
                terminal_provider,
                terminal_detail,
            },
        };
        self.outbound
            .notify(GOAL_UPDATE_METHOD, serde_json::to_value(update)?)
            .await;
        Ok(())
    }
}

/// Forward one Goal-owned agent event through ZeroCode's ordinary session
/// presentation protocol. Lifecycle updates remain separate Goal metadata.
async fn forward_zerocode_goal_event(
    context: &RpcContext,
    outbound: &RpcOutbound,
    session_id: &str,
    max_context_tokens: u64,
    event: TurnEvent,
) {
    // The Goal driver only admits `Chat` sessions. ACP durable plan storage
    // belongs exclusively to ACP sessions, so a daemon-wide ACP store must
    // not cause a Goal event to attempt a write for this chat session.
    crate::rpc::dispatch::persist_plan_if_any(context.sessions.as_ref(), None, session_id, &event)
        .await;
    if let Some(notification) = crate::rpc::dispatch::notification_for_turn_event(
        session_id,
        &event,
        Some(max_context_tokens),
    ) {
        let _ = outbound.send_raw(notification).await;
    }
}

fn isolated_source_for_goal_turn(turn: GoalParentTurn) -> IsolatedTranscriptSource {
    let mut working_history = turn.working_history;
    match turn.history_source {
        crate::goal_mode::GoalParentHistorySource::Canonical => {
            IsolatedTranscriptSource::Canonical {
                prefix: working_history,
                trailing_user: turn.resume_response.map(ChatMessage::user),
            }
        }
        crate::goal_mode::GoalParentHistorySource::Continuation => {
            let append_execution_request = turn.resume_response.is_none();
            if let Some(response) = turn.resume_response {
                // The response is ordinary untrusted session input. Keeping it
                // in the transient isolated transcript makes a resident resume
                // continuous without adding it to canonical history.
                working_history.push(ChatMessage::user(response));
            }
            IsolatedTranscriptSource::Continuation {
                history: working_history,
                append_execution_request,
            }
        }
    }
}

/// ZeroCode's process-local Goal worker registry.
#[derive(Default)]
pub struct RpcGoalRuntime {
    supervisors: Mutex<HashMap<String, Arc<GoalExecutionSupervisor>>>,
}

impl RpcGoalRuntime {
    /// Apply the Goal lifecycle consequence of an externally cancelled RPC
    /// session before the ordinary session cancellation token is signalled.
    ///
    /// A resident Goal owns the session queue while a model operation is in
    /// flight.  Its durable pause and accounting settlement therefore finish
    /// first; signalling the ordinary token beforehand would interrupt the
    /// operation the Goal controller is required to account for.
    pub async fn pause_for_external_cancellation(&self, session_id: &str) -> Result<bool> {
        let Some(supervisor) = self.supervisor(session_id).await else {
            return Ok(false);
        };
        let durable_session_id = GoalSessionKey::zero_code(session_id)?.durable_id();
        let result = supervisor
            .pause_for_external_cancellation(&durable_session_id)
            .await?;
        if matches!(result, crate::control_plane::GoalTransitionResult::Applied) {
            self.remove_supervisor_if_current(session_id, &supervisor)
                .await;
            return Ok(true);
        }
        Ok(false)
    }

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
        if matches!(command, GoalCommand::Help) {
            return Ok(GoalResponse::Help);
        }
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
                // Match the resident RPC turn path. Reload can replace the
                // global singleton while this context still owns the tracker
                // serving its active sessions; Goal Mode must use that same
                // canonical ledger rather than fail admission on the stale
                // configuration snapshot.
                let tracker = context.cost_tracker.clone().map(Ok).unwrap_or_else(|| {
                    CostTracker::get_or_init_global_required(config.cost.clone(), &config.data_dir)
                })?;
                let engine = Arc::new(runtime.execution_engine(
                    tracker,
                    driver.agent_alias().to_owned(),
                    Arc::new(build_type_level_model_provider_pricing(&config)),
                )?);
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
        if matches!(&command, GoalCommand::PauseNow | GoalCommand::Cancel) {
            supervisor
                .interrupt_and_drain_session(
                    &driver.session_key().durable_id(),
                    matches!(command, GoalCommand::Cancel),
                )
                .await?;
        }
        let acknowledgement_outbound = Arc::clone(&driver.outbound);
        let acknowledgement_session_id = session_id.clone();
        let submission = supervisor
            .submit_with_before_launch(settings, ingress, driver, command, move |response| {
                let outbound = Arc::clone(&acknowledgement_outbound);
                let session_id = acknowledgement_session_id.clone();
                let response = response.clone();
                async move {
                    let update = crate::rpc::types::SessionGoalUpdate::Acknowledged {
                        session_id,
                        response,
                    };
                    let payload = serde_json::to_value(update)?;
                    let notification = zeroclaw_api::jsonrpc::JsonRpcNotification::new(
                        GOAL_UPDATE_METHOD,
                        payload,
                    );
                    let encoded = serde_json::to_string(&notification)?;
                    if !outbound.send_raw(encoded).await {
                        bail!("ZeroCode Goal acknowledgement could not be delivered");
                    }
                    Ok(())
                }
            })
            .await?;
        if submission.response().retires_resident_supervisor() {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::sync::mpsc;
    use zeroclaw_api::jsonrpc::RpcOutbound;
    use zeroclaw_config::schema::Config;
    use zeroclaw_infra::session_queue::SessionActorQueue;

    use super::*;
    use crate::goal_mode::GoalParentTurnKind;
    use crate::rpc::session::SessionStore;

    #[test]
    fn retained_goal_resume_uses_the_isolated_continuation_source() {
        let source = isolated_source_for_goal_turn(GoalParentTurn {
            kind: GoalParentTurnKind::Resume,
            objective: "finish the task".to_owned(),
            resume_response: Some("use the authentication module".to_owned()),
            paused_request: None,
            history_source: crate::goal_mode::GoalParentHistorySource::Continuation,
            working_history: vec![
                ChatMessage::system("previous Goal system prompt"),
                ChatMessage::assistant("Which module should I change?"),
            ],
        });

        match source {
            IsolatedTranscriptSource::Continuation {
                history,
                append_execution_request,
            } => {
                assert_eq!(history[0].role, "system");
                assert_eq!(history[1].content, "Which module should I change?");
                assert_eq!(history[2].role, "user");
                assert_eq!(history[2].content, "use the authentication module");
                assert!(!append_execution_request);
            }
            IsolatedTranscriptSource::Canonical { .. } => {
                panic!("a retained Goal transcript must not be treated as canonical history")
            }
        }
    }

    #[test]
    fn recovery_goal_resume_places_the_response_after_the_execution_request() {
        let source = isolated_source_for_goal_turn(GoalParentTurn {
            kind: GoalParentTurnKind::Resume,
            objective: "finish the task".to_owned(),
            resume_response: Some("the blocker is resolved".to_owned()),
            paused_request: None,
            history_source: crate::goal_mode::GoalParentHistorySource::Canonical,
            working_history: vec![ChatMessage::user("original task")],
        });

        match source {
            IsolatedTranscriptSource::Canonical {
                prefix,
                trailing_user: Some(response),
            } => {
                assert_eq!(
                    prefix.last().map(|message| message.content.as_str()),
                    Some("original task")
                );
                assert_eq!(response.role, "user");
                assert_eq!(response.content, "the blocker is resolved");
            }
            other => panic!("expected canonical recovery source, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn goal_event_uses_the_ordinary_zerocode_session_update_protocol() {
        use zeroclaw_infra::session_queue::SessionActorQueue;

        let (outbound_tx, mut outbound_rx) = mpsc::channel(1);
        let outbound = RpcOutbound::new(outbound_tx);
        let context = RpcContext::minimal(
            Config::default(),
            Arc::new(SessionStore::new(
                1,
                Arc::new(SessionActorQueue::new(1, 1, 1)),
            )),
        );

        forward_zerocode_goal_event(
            context.as_ref(),
            &outbound,
            "goal-session",
            128_000,
            TurnEvent::Chunk {
                delta: "agent-visible progress".to_owned(),
            },
        )
        .await;

        let notification: serde_json::Value = serde_json::from_str(
            &outbound_rx
                .recv()
                .await
                .expect("Goal event should reach the ordinary session surface"),
        )
        .expect("ordinary session notification should be JSON");
        assert_eq!(notification["method"], "session/update");
        assert_eq!(notification["params"]["type"], "agent_message_chunk");
        assert_eq!(notification["params"]["session_id"], "goal-session");
        assert_eq!(notification["params"]["text"], "agent-visible progress");
    }

    #[tokio::test]
    async fn help_is_available_while_goal_mode_is_disabled() {
        let config = Config::default();
        assert!(
            !config.goal.enabled,
            "test requires Goal Mode to be disabled"
        );
        let sessions = Arc::new(SessionStore::new(
            1,
            Arc::new(SessionActorQueue::new(1, 1, 1)),
        ));
        let context = RpcContext::minimal(config, sessions);
        let (outbound_tx, _outbound_rx) = mpsc::channel(1);
        let driver = Arc::new(ZeroCodeGoalSessionDriver {
            context: Arc::clone(&context),
            outbound: Arc::new(RpcOutbound::new(outbound_tx)),
            connection_activity: None,
            session_key: GoalSessionKey::zero_code("disabled-goal-help").unwrap(),
            agent_alias: "test-agent".to_owned(),
            session_generation: 0,
            tui_id: "test-tui".to_owned(),
            command_lock: Arc::new(Mutex::new(())),
        });

        let response = RpcGoalRuntime::default()
            .submit(context, driver, GoalCommand::Help)
            .await
            .unwrap();

        assert!(matches!(response, GoalResponse::Help));
    }
}
