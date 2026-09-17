//! Matrix implementation of the transport-neutral Goal session driver.
//!
//! This module owns only live Matrix session mechanics. Durable lifecycle,
//! accounting, and execution fencing remain in `zeroclaw-runtime`.

use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result, bail, ensure};
use async_trait::async_trait;
use zeroclaw_api::{
    channel::ChannelMessage,
    model_provider::{ChatMessage, ChatRequest},
};
use zeroclaw_runtime::{
    agent::cost::build_type_level_model_provider_pricing,
    agent::loop_::{LoopKnobs, ToolLoop, run_tool_call_loop, scope_session_key, scope_thread_id},
    control_plane::control_plane,
    cost::CostTracker,
    goal_mode::{
        GoalExecutionNotice, GoalExecutionScope, GoalIngressContext, GoalOperationScope,
        GoalParentTurn, GoalParentTurnKind, GoalParentTurnResult, GoalSessionBinding,
        GoalSessionDriver, GoalSessionExecutionLease, GoalSessionKey, GoalSessionLease,
        GoalSurface, GoalVerifierTurn, dispose_unowned_session_goal, goal_parent_directive,
        goal_parent_execution_request, goal_parent_system_message, goal_verifier_messages,
        scope_goal_parent_turn,
    },
};

use super::foreground::{foreground_lock, goal_command_lock, goal_supervisor_slot};
use super::{
    ChannelRouteSelection, ChannelRuntimeContext, append_sender_turn,
    build_channel_system_prompt_for_message_with_signal,
    callable_protocol_exposed_for_channel_turn, find_channel_for_message, get_or_create_provider,
    get_route_selection, model_provider_entry_for_ref, outbound_content_format_for_channel,
    persist_session_routing_context, runtime_defaults_from_config, runtime_defaults_snapshot,
    sanitize_channel_response_for_format_with_leak_detection, system_prompt_for_channel_turn,
    turn_execution::resolved_channel_execution,
};

/// Immutable Matrix facts captured before mutable inbound hooks run.
#[derive(Clone)]
pub(super) struct MatrixGoalSessionDriver {
    context: Arc<ChannelRuntimeContext>,
    session_key: GoalSessionKey,
    raw_mxid: String,
    route: String,
    message: ChannelMessage,
    /// The per-session lifecycle boundary is acquired before selecting a
    /// resident supervisor. `bind` transfers it to the runtime submission so
    /// selection, durable transition, worker launch, and retirement remain
    /// one serialized operation.
    command_lease: Arc<Mutex<Option<tokio::sync::OwnedMutexGuard<()>>>>,
}

impl MatrixGoalSessionDriver {
    pub(super) fn new(
        context: Arc<ChannelRuntimeContext>,
        history_key: String,
        raw_mxid: String,
        route: String,
        message: ChannelMessage,
        command_lease: tokio::sync::OwnedMutexGuard<()>,
    ) -> Result<Self> {
        ensure!(
            message.channel.eq_ignore_ascii_case("matrix"),
            "Goal driver is not Matrix"
        );
        ensure!(
            message.sender == raw_mxid,
            "Matrix Goal raw MXID does not match message sender"
        );
        Ok(Self {
            context,
            session_key: GoalSessionKey::matrix(history_key)?,
            raw_mxid,
            route,
            message,
            command_lease: Arc::new(Mutex::new(Some(command_lease))),
        })
    }

    fn assert_ingress(&self, ingress: &GoalIngressContext) -> Result<()> {
        ensure!(
            ingress.surface() == GoalSurface::Matrix,
            "Goal ingress is not Matrix"
        );
        ensure!(
            ingress.session_key() == &self.session_key,
            "Goal ingress session is stale"
        );
        ensure!(
            ingress.agent() == self.context.agent_alias.as_str(),
            "Goal ingress agent is stale"
        );
        ensure!(ingress.route() == self.route, "Goal ingress route is stale");
        match ingress.principal() {
            zeroclaw_runtime::goal_mode::GoalIngressPrincipal::Matrix { raw_mxid }
                if raw_mxid.as_str() == self.raw_mxid.as_str() =>
            {
                Ok(())
            }
            _ => bail!("Goal ingress principal is stale"),
        }
    }

    fn persist_execution_routing_context(&self, session_id: &str) -> Result<()> {
        let Some(store) = self.context.session_store.as_ref() else {
            return Ok(());
        };
        persist_session_routing_context(store.as_ref(), &self.message, session_id)
            .context("persist Matrix Goal execution routing context")
    }
}

/// Submit an already parsed Goal command using immutable Matrix ingress facts.
pub(super) async fn submit_matrix_goal(
    context: Arc<ChannelRuntimeContext>,
    history_key: String,
    original: ChannelMessage,
    command: zeroclaw_commands::goal::GoalCommand,
) -> Result<zeroclaw_runtime::goal_mode::GoalResponse> {
    // Help is local grammar. It must not require Matrix driver validation,
    // runtime configuration, or a live control plane.
    if matches!(command, zeroclaw_commands::goal::GoalCommand::Help) {
        return Ok(zeroclaw_runtime::goal_mode::GoalResponse::Help);
    }
    let defaults = runtime_defaults_snapshot(context.as_ref());
    if !defaults.config.goal.enabled {
        return Ok(zeroclaw_runtime::goal_mode::GoalResponse::Disabled);
    }
    // Acquire this before reading durable state or selecting a supervisor.
    // In particular, a terminal predecessor must be drained by the same
    // serialized lifecycle operation before another command can install a
    // successor supervisor for this session.
    let command_lock = goal_command_lock(&context.persist_locks, &history_key);
    let command_lease = command_lock.lock_owned().await;
    let route = format!(
        "matrix:{}:{}",
        original.channel_alias.as_deref().unwrap_or_default(),
        original.reply_target
    );
    let driver = Arc::new(MatrixGoalSessionDriver::new(
        Arc::clone(&context),
        history_key.clone(),
        original.sender.clone(),
        route.clone(),
        original,
        command_lease,
    )?);
    let control_plane = control_plane().context("Goal control plane is unavailable")?;
    let registry = control_plane.goal_store()?;
    let restart_coordinator = control_plane.goal_execution_restart();
    let settings = zeroclaw_runtime::goal_mode::GoalHostSettings::from_config(
        &defaults.config.goal,
        std::process::id(),
        control_plane.boot_id.clone(),
    )?;
    let current = registry.current_goal_for_session(&history_key).await?;
    let supervisor_slot = goal_supervisor_slot(&context.persist_locks, &history_key);
    let retired_scope = current
        .as_ref()
        .filter(|task| task.status.is_terminal())
        .map(|task| {
            GoalExecutionScope::new(task.id.clone(), history_key.clone(), task.execution_epoch)
        })
        .transpose()?;
    let retired_supervisor = if retired_scope.is_some() {
        supervisor_slot.lock().await.take()
    } else {
        None
    };
    if let (Some(retired_supervisor), Some(scope)) = (retired_supervisor, retired_scope)
        && retired_supervisor.owns_scope(&scope).await
    {
        retired_supervisor
            .drain(&scope)
            .await
            .context("drain terminal Matrix Goal worker before replacement")?;
    }
    let supervisor = {
        let mut slot = supervisor_slot.lock().await;
        if let Some(supervisor) = slot.as_ref() {
            Arc::clone(supervisor)
        } else {
            let runtime = zeroclaw_runtime::goal_mode::GoalRuntime::new(Arc::clone(&registry));
            let tracker = CostTracker::get_or_init_global_required(
                defaults.config.cost.clone(),
                &defaults.config.data_dir,
            )?;
            let engine = Arc::new(runtime.execution_engine(
                tracker,
                context.agent_alias.to_string(),
                Arc::new(build_type_level_model_provider_pricing(
                    defaults.config.as_ref(),
                )),
            )?);
            let supervisor = restart_coordinator.new_supervisor(engine).await;
            *slot = Some(Arc::clone(&supervisor));
            supervisor
        }
    };
    let ingress = GoalIngressContext::trusted(
        GoalSessionKey::matrix(history_key)?,
        context.agent_alias.to_string(),
        route,
        zeroclaw_runtime::goal_mode::GoalIngressPrincipal::Matrix {
            raw_mxid: driver.raw_mxid.clone(),
        },
    )?;
    let submission = supervisor
        .submit(settings, ingress, driver, command)
        .await?;
    if matches!(
        submission.response(),
        zeroclaw_runtime::goal_mode::GoalResponse::Paused(_)
            | zeroclaw_runtime::goal_mode::GoalResponse::AlreadyPaused(_)
            | zeroclaw_runtime::goal_mode::GoalResponse::Cancelled(_)
            | zeroclaw_runtime::goal_mode::GoalResponse::AlreadyCancelled(_)
            | zeroclaw_runtime::goal_mode::GoalResponse::Terminal(_)
    ) {
        // Pause and cancellation drain the old worker before the supervisor
        // returns. Rebuilding on an explicit resume picks up live policy and
        // pricing instead of keeping a stale configuration snapshot.
        clear_supervisor_slot_if_current(&supervisor_slot, &supervisor).await;
    }
    Ok(submission.into_response())
}

/// Dispose a Matrix session's Goal before the channel resets its history.
///
/// A resident supervisor performs the required fence-and-drain ordering. If
/// this process has no worker (for example after restart), the durable row is
/// already the sole live authority and can be fenced, classified, and deleted
/// directly without recreating a worker merely to dispose it.
pub(super) async fn dispose_matrix_goal(
    context: &ChannelRuntimeContext,
    history_key: &str,
) -> Result<()> {
    // `/new` shares the same lifecycle boundary as Goal commands. It must not
    // clear a supervisor installed by a concurrent start while disposal was
    // awaiting a durable fence or worker drain.
    let command_lock = goal_command_lock(&context.persist_locks, history_key);
    let _command_lease = command_lock.lock_owned().await;
    let slot = goal_supervisor_slot(&context.persist_locks, history_key);
    if let Some(supervisor) = slot.lock().await.as_ref().cloned() {
        supervisor.dispose_session(history_key).await?;
        clear_supervisor_slot_if_current(&slot, &supervisor).await;
        return Ok(());
    }

    let Some(control_plane) = control_plane() else {
        // Without the canonical control plane no Goal could have been
        // admitted, so ordinary Matrix `/new` keeps its historical behavior.
        return Ok(());
    };
    let registry = control_plane.goal_store()?;
    let _ = dispose_unowned_session_goal(registry.as_ref(), history_key).await?;
    Ok(())
}

/// Clear a session's resident supervisor only if it is still the one that
/// performed the lifecycle transition.
///
/// The submission lease remains held through this cleanup, so another Goal
/// command cannot reuse or replace the supervisor while its predecessor is
/// retiring. Pointer comparison additionally protects disposal and other
/// lifecycle paths that do not own that command lease.
async fn clear_supervisor_slot_if_current(
    slot: &Arc<
        tokio::sync::Mutex<Option<Arc<zeroclaw_runtime::goal_mode::GoalExecutionSupervisor>>>,
    >,
    supervisor: &Arc<zeroclaw_runtime::goal_mode::GoalExecutionSupervisor>,
) {
    let mut slot = slot.lock().await;
    if slot
        .as_ref()
        .is_some_and(|current| Arc::ptr_eq(current, supervisor))
    {
        *slot = None;
    }
}

#[async_trait]
impl GoalSessionDriver for MatrixGoalSessionDriver {
    fn session_key(&self) -> &GoalSessionKey {
        &self.session_key
    }

    async fn bind(&self, ingress: &GoalIngressContext) -> Result<GoalSessionLease> {
        self.assert_ingress(ingress)?;
        let guard = self
            .command_lease
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
            .context("Matrix Goal command lease was already consumed")?;
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
        let session_id = self.session_key.durable_id();
        ensure!(
            scope.session_id() == session_id,
            "Goal execution session is stale"
        );
        let foreground = foreground_lock(&self.context.persist_locks, &session_id)
            .lock_owned()
            .await;
        // A Goal can append canonical history only after this lease is
        // acquired. Persist the route strictly at the same boundary so an
        // execution that cannot be discovered after restart never begins.
        self.persist_execution_routing_context(&session_id)?;
        Ok(Box::new(MatrixGoalExecutionLease {
            _foreground: foreground,
            context: Arc::clone(&self.context),
            session_key: self.session_key.clone(),
            message: self.message.clone(),
        }))
    }
}

struct MatrixGoalExecutionLease {
    _foreground: tokio::sync::OwnedMutexGuard<()>,
    context: Arc<ChannelRuntimeContext>,
    session_key: GoalSessionKey,
    message: ChannelMessage,
}

impl MatrixGoalExecutionLease {
    fn history(&self) -> Vec<ChatMessage> {
        self.context
            .conversation_histories
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .peek(&self.session_key.durable_id())
            .cloned()
            .unwrap_or_default()
    }

    fn goal_system_prompt(
        &self,
        provider: &dyn zeroclaw_providers::ModelProvider,
        route: &ChannelRouteSelection,
    ) -> String {
        let excluded_tools: &[String] =
            if self.context.autonomy_level == zeroclaw_config::autonomy::AutonomyLevel::Full {
                &[]
            } else {
                self.context.non_cli_excluded_tools.as_ref()
            };
        let native_tool_specs_present =
            zeroclaw_runtime::agent::loop_::native_tool_specs_present_for_turn(
                provider,
                route.model.as_str(),
                self.context.tools_registry.as_ref(),
                excluded_tools,
                self.context.activated_tools.as_ref(),
            )
            .unwrap_or(false);
        let callable_protocol_exposed = callable_protocol_exposed_for_channel_turn(
            native_tool_specs_present,
            self.context.agent_cfg.resolved.strict_tool_parsing,
            self.context.system_prompt.as_str(),
        );
        let base_system_prompt = system_prompt_for_channel_turn(
            self.context.as_ref(),
            self.context.system_prompt.as_str(),
            true,
            callable_protocol_exposed,
            excluded_tools,
        );
        let target_channel =
            find_channel_for_message(&self.context.channels_by_name, &self.message);
        build_channel_system_prompt_for_message_with_signal(
            &base_system_prompt,
            &self.message,
            target_channel,
            native_tool_specs_present,
        )
    }

    fn initial_working_history(
        &self,
        provider: &dyn zeroclaw_providers::ModelProvider,
        route: &ChannelRouteSelection,
        directive: ChatMessage,
        canonical_history: Vec<ChatMessage>,
    ) -> Vec<ChatMessage> {
        goal_start_history(
            self.goal_system_prompt(provider, route),
            directive,
            canonical_history,
        )
    }
}

fn goal_start_history(
    system_prompt: String,
    directive: ChatMessage,
    canonical_history: Vec<ChatMessage>,
) -> Vec<ChatMessage> {
    let mut history = Vec::with_capacity(canonical_history.len() + 2);
    history.push(goal_parent_system_message(system_prompt, directive.content));
    history.extend(canonical_history);
    history.push(goal_parent_execution_request());
    history
}

fn goal_continue_history(
    system_prompt: String,
    directive: ChatMessage,
    mut history: Vec<ChatMessage>,
) -> Result<Vec<ChatMessage>> {
    let first = history
        .first_mut()
        .context("Goal continuation lost its system prompt")?;
    ensure!(
        first.role == "system",
        "Goal continuation lost its system prompt"
    );
    *first = goal_parent_system_message(system_prompt, directive.content);
    history.push(goal_parent_execution_request());
    Ok(history)
}
#[async_trait]
impl GoalSessionExecutionLease for MatrixGoalExecutionLease {
    fn session_key(&self) -> &GoalSessionKey {
        &self.session_key
    }

    fn canonical_history(&self) -> Result<Vec<ChatMessage>> {
        Ok(self.history())
    }

    async fn run_parent_turn(
        &mut self,
        operation: &GoalOperationScope,
        turn: GoalParentTurn,
    ) -> Result<GoalParentTurnResult> {
        let defaults = runtime_defaults_snapshot(self.context.as_ref());
        let route = get_route_selection(
            self.context.as_ref(),
            &self.message,
            &self.session_key.durable_id(),
            &defaults,
        );
        let provider = get_or_create_provider(
            self.context.as_ref(),
            &route.model_provider,
            route.api_key.as_deref(),
            &defaults,
        )
        .await?;
        let directive = goal_parent_directive(&turn);
        let mut history = match turn.kind {
            GoalParentTurnKind::Start | GoalParentTurnKind::Resume => self.initial_working_history(
                provider.as_ref(),
                &route,
                directive,
                turn.working_history,
            ),
            GoalParentTurnKind::Continue => goal_continue_history(
                self.goal_system_prompt(provider.as_ref(), &route),
                directive,
                turn.working_history,
            )?,
        };
        let turn_id = uuid::Uuid::new_v4().to_string();
        let loop_knobs = LoopKnobs::default();
        let tool_loop = run_tool_call_loop(ToolLoop {
            exec: resolved_channel_execution(
                self.context.as_ref(),
                provider.as_ref(),
                &route,
                self.context.observer.as_ref(),
                &loop_knobs,
                self.context.non_cli_excluded_tools.as_ref(),
                defaults.defaults.temperature,
            ),
            history: &mut history,
            channel_name: "matrix",
            channel_reply_target: Some(self.message.reply_target.as_str()),
            cancellation_token: None,
            on_delta: None,
            shared_budget: None,
            channel: None,
            collected_receipts: None,
            event_tx: None,
            steering: None,
            new_messages_out: None,
            image_cache: None,
            ingress: zeroclaw_api::ingress::IngressContext::channel(),
            memory: None,
            agent_alias: Some(self.context.agent_alias.as_str()),
            parent_agent_alias: None,
            turn_id: &turn_id,
            sop_reassembly: None,
        });
        let candidate = scope_goal_parent_turn(scope_session_key(
            Some(self.session_key.durable_id()),
            async { scope_thread_id(Some(self.message.id.clone()), tool_loop).await },
        ))
        .await?;
        let _ = operation;
        Ok(GoalParentTurnResult {
            candidate,
            working_history: history,
        })
    }

    async fn run_verifier(
        &mut self,
        _operation: &GoalOperationScope,
        turn: GoalVerifierTurn,
    ) -> Result<String> {
        let defaults = runtime_defaults_snapshot(self.context.as_ref());
        let provider_ref = defaults.config.goal.verifier.model_provider.as_str();
        let (provider_name, entry) =
            model_provider_entry_for_ref(defaults.config.as_ref(), provider_ref)?;
        let model = defaults.config.goal.verifier.model.as_deref().unwrap_or(
            entry
                .model
                .as_deref()
                .context("Goal verifier has no model")?,
        );
        let verifier_defaults =
            runtime_defaults_from_config(defaults.config.as_ref(), &provider_name)?;
        let provider = get_or_create_provider(
            self.context.as_ref(),
            &provider_name,
            entry.api_key.as_deref(),
            &defaults,
        )
        .await?;
        let messages = goal_verifier_messages(&turn);
        let response = resolved_channel_execution(
            self.context.as_ref(),
            provider.as_ref(),
            &ChannelRouteSelection {
                model_provider: provider_name,
                model: model.to_owned(),
                api_key: entry.api_key.clone(),
            },
            self.context.observer.as_ref(),
            &LoopKnobs::default(),
            &[],
            verifier_defaults.temperature,
        )
        .model_access
        .run_model_query(ChatRequest {
            messages: &messages,
            tools: None,
            thinking: None,
        })
        .await?;
        response.text.context("Goal verifier returned no text")
    }

    async fn append_verified_candidate(&mut self, candidate: String) -> Result<()> {
        let delivered = sanitize_channel_response_for_format_with_leak_detection(
            &candidate,
            self.context.tools_registry.as_ref(),
            &self.context.prompt_config.security.leak_detection,
            outbound_content_format_for_channel(&self.message.channel),
        );
        let channel = find_channel_for_message(&self.context.channels_by_name, &self.message)
            .context("Matrix Goal channel is no longer available")?;
        channel
            .send(&zeroclaw_api::channel::SendMessage::reply_to(
                &self.message,
                &delivered,
            ))
            .await
            .context("deliver verified Matrix Goal candidate")?;
        append_sender_turn(
            self.context.as_ref(),
            &self.session_key.durable_id(),
            ChatMessage::assistant(&delivered),
        );
        Ok(())
    }

    async fn publish_goal_notice(&mut self, notice: GoalExecutionNotice) -> Result<()> {
        let channel = find_channel_for_message(&self.context.channels_by_name, &self.message)
            .context("Matrix Goal channel is no longer available")?;
        channel
            .send(&zeroclaw_api::channel::SendMessage::reply_to(
                &self.message,
                goal_notice_message(notice),
            ))
            .await
            .context("deliver Matrix Goal lifecycle notice")
    }
}

fn goal_notice_message(notice: GoalExecutionNotice) -> String {
    match notice {
        GoalExecutionNotice::Completed => {
            zeroclaw_runtime::i18n::get_required_cli_string("goal-mode-completed")
        }
        GoalExecutionNotice::PausedForBlocker { blocker_messages } => {
            let mut message =
                zeroclaw_runtime::i18n::get_required_cli_string("goal-mode-paused-blocked");
            for blocker in blocker_messages {
                message.push('\n');
                message.push_str(&zeroclaw_runtime::i18n::get_required_cli_string_with_args(
                    "goal-mode-blocker",
                    &[("blocker", blocker.as_str())],
                ));
            }
            message.push('\n');
            message.push_str(&zeroclaw_runtime::i18n::get_required_cli_string(
                "goal-mode-paused-blocked-guidance",
            ));
            message
        }
        GoalExecutionNotice::Failed => {
            zeroclaw_runtime::i18n::get_required_cli_string("goal-mode-failed")
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_notice_includes_the_verifier_blocker() {
        let rendered = goal_notice_message(GoalExecutionNotice::PausedForBlocker {
            blocker_messages: vec!["Provide the task packet reference.".to_owned()],
        });

        assert!(rendered.starts_with("⏸️ Goal paused."));
        assert!(rendered.contains("Blocker: Provide the task packet reference."));
        assert!(!rendered.contains("verifier requires resolution"));
        assert!(rendered.contains("`/goal resume`"));
    }

    #[test]
    fn start_history_combines_goal_directive_into_the_system_prompt() {
        let history = goal_start_history(
            "system prompt".to_owned(),
            ChatMessage::system("Goal directive"),
            vec![ChatMessage::assistant("earlier assistant response")],
        );

        assert_eq!(history.len(), 3);
        assert_eq!(history[0].role, "system");
        assert!(history[0].content.contains("system prompt"));
        assert!(history[0].content.contains("Goal directive"));
        assert_eq!(history[1].role, "assistant");
        assert_eq!(history[1].content, "earlier assistant response");
        assert_eq!(history[2].role, "user");
        assert!(history[2].content.contains("Proceed with the Goal work"));
    }

    #[test]
    fn continuation_replaces_the_goal_directive_in_the_only_system_prompt() {
        let history = goal_continue_history(
            "rebuilt system prompt".to_owned(),
            ChatMessage::system("continue Goal directive"),
            vec![
                ChatMessage::system("old system prompt\n\nstart Goal directive"),
                ChatMessage::assistant("earlier assistant response"),
            ],
        )
        .expect("continuation with a system prompt should build");

        assert_eq!(history[0].role, "system");
        assert!(history[0].content.contains("rebuilt system prompt"));
        assert!(history[0].content.contains("continue Goal directive"));
        assert!(!history[0].content.contains("start Goal directive"));
        assert_eq!(
            history
                .iter()
                .filter(|message| message.role == "system")
                .count(),
            1
        );
        assert_eq!(
            history.last().map(|message| message.role.as_str()),
            Some("user")
        );
    }

    #[test]
    fn continuation_rejects_a_transcript_without_a_system_prompt() {
        let error = goal_continue_history(
            "rebuilt system prompt".to_owned(),
            ChatMessage::system("continue Goal directive"),
            vec![ChatMessage::assistant("earlier assistant response")],
        )
        .expect_err("continuation must retain the system prompt at history index zero");

        assert!(error.to_string().contains("lost its system prompt"));
    }
}
