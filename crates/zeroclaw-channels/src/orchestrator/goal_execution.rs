//! Matrix implementation of the transport-neutral Goal session driver.
//!
//! This module owns only live Matrix session mechanics. Durable lifecycle,
//! accounting, and execution fencing remain in `zeroclaw-runtime`.

use std::sync::Arc;

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
        GoalSurface, GoalVerifierTurn,
    },
};

use super::foreground::{foreground_lock, goal_command_lock, goal_supervisor_slot};
use super::{
    ChannelRouteSelection, ChannelRuntimeContext, append_sender_turn,
    build_channel_system_prompt_for_message_with_signal,
    callable_protocol_exposed_for_channel_turn, find_channel_for_message, get_or_create_provider,
    get_route_selection, model_provider_entry_for_ref, outbound_content_format_for_channel,
    runtime_defaults_from_config, runtime_defaults_snapshot,
    sanitize_channel_response_for_format_with_leak_detection, system_prompt_for_channel_turn,
    turn_execution::resolved_channel_execution,
};

/// Immutable Matrix facts captured before mutable inbound hooks run.
#[derive(Clone)]
struct MatrixGoalSessionDriver {
    context: Arc<ChannelRuntimeContext>,
    session_key: GoalSessionKey,
    raw_mxid: String,
    route: String,
    message: ChannelMessage,
}

impl MatrixGoalSessionDriver {
    fn new(
        context: Arc<ChannelRuntimeContext>,
        history_key: String,
        raw_mxid: String,
        route: String,
        message: ChannelMessage,
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
                if raw_mxid == &self.raw_mxid =>
            {
                Ok(())
            }
            _ => bail!("Goal ingress principal is stale"),
        }
    }
}

/// Submit an already parsed Goal command using immutable Matrix ingress facts.
pub(super) async fn submit_matrix_goal(
    context: Arc<ChannelRuntimeContext>,
    history_key: String,
    original: ChannelMessage,
    command: zeroclaw_commands::goal::GoalCommand,
) -> Result<zeroclaw_runtime::goal_mode::GoalResponse> {
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
    )?);
    let control_plane = control_plane().context("Goal control plane is unavailable")?;
    let registry = control_plane.goal_store()?;
    let restart_coordinator = control_plane.goal_execution_restart();
    let defaults = runtime_defaults_snapshot(context.as_ref());
    let configured_limits =
        defaults.config.goal.effective_limits().map_err(|error| {
            anyhow::Error::msg(format!("Goal configuration is invalid: {error:?}"))
        })?;
    let settings = zeroclaw_runtime::goal_mode::GoalHostSettings::new(
        defaults.config.goal.enabled,
        zeroclaw_commands::goal::GoalBudgetLimits {
            token_limit: configured_limits.token_limit,
            cost_limit_usd: configured_limits.cost_limit_usd,
        },
        std::process::id(),
        control_plane.boot_id.clone(),
    )?;
    let current = registry.current_goal_for_session(&history_key).await?;
    let supervisor_slot = goal_supervisor_slot(&context.persist_locks, &history_key);
    let supervisor = {
        let mut slot = supervisor_slot.lock().await;
        if current
            .as_ref()
            .is_some_and(|task| task.status.is_terminal())
        {
            // A completed worker remains in the supervisor until drained. A
            // successor must instead receive a fresh engine resolved from the
            // current runtime configuration.
            *slot = None;
        }
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
    let response = supervisor
        .submit(settings, ingress, driver, command)
        .await?;
    if matches!(
        response,
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
    Ok(response)
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
    let Some(current) = registry.current_goal_for_session(history_key).await? else {
        return Ok(());
    };
    if !current.status.is_terminal() {
        let _ = registry
            .finish_session_goal(
                &current.id,
                history_key,
                current.execution_epoch,
                zeroclaw_runtime::control_plane::TaskStatus::Cancelled,
                Some("session_disposed".to_owned()),
            )
            .await?;
    }
    let Some(reloaded) = registry.current_goal_for_session(history_key).await? else {
        return Ok(());
    };
    if let Some(goal) = registry.get_goal_task(&reloaded.id).await?
        && let Some((pending_id, pending_epoch)) =
            goal.pending_call_id.as_deref().zip(goal.pending_call_epoch)
    {
        let _ = registry
            .settle_pending_operation(
                &reloaded.id,
                history_key,
                pending_epoch,
                pending_id,
                zeroclaw_runtime::control_plane::GoalAccountingState::OutcomeUnknown,
            )
            .await?;
    }
    let _ = registry
        .delete_session_goal(&reloaded.id, history_key, reloaded.execution_epoch)
        .await?;
    Ok(())
}

/// Clear a session's resident supervisor only if it is still the one that
/// performed the lifecycle transition.
///
/// Goal commands serialize their durable transition under the driver's command
/// lease, but that lease is released before the transport receives the typed
/// response. A subsequent command can therefore install a fresh supervisor
/// before this function reacquires the process-local slot. Clearing the slot
/// unconditionally would then orphan the successor's join handle.
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
    fn surface(&self) -> GoalSurface {
        GoalSurface::Matrix
    }

    fn session_key(&self) -> &GoalSessionKey {
        &self.session_key
    }

    async fn bind(&self, ingress: &GoalIngressContext) -> Result<GoalSessionLease> {
        self.assert_ingress(ingress)?;
        let guard = goal_command_lock(&self.context.persist_locks, &self.session_key.durable_id())
            .lock_owned()
            .await;
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
        let foreground =
            foreground_lock(&self.context.persist_locks, &self.session_key.durable_id())
                .lock_owned()
                .await;
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

    fn initial_working_history(
        &self,
        provider: &dyn zeroclaw_providers::ModelProvider,
        route: &ChannelRouteSelection,
        canonical_history: Vec<ChatMessage>,
    ) -> Vec<ChatMessage> {
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
        let system_prompt = build_channel_system_prompt_for_message_with_signal(
            &base_system_prompt,
            &self.message,
            target_channel,
            native_tool_specs_present,
        );
        let mut history = vec![ChatMessage::system(system_prompt)];
        history.extend(canonical_history);
        history
    }
}

fn goal_parent_directive(turn: &GoalParentTurn) -> ChatMessage {
    let turn_kind = match turn.kind {
        GoalParentTurnKind::Start => "start",
        GoalParentTurnKind::Resume => "resume",
        GoalParentTurnKind::Continue => "continue",
    };
    ChatMessage::system(format!(
        "Goal success criterion (trusted runtime directive): {}\nTurn kind: {turn_kind}",
        turn.objective
    ))
}

#[async_trait]
impl GoalSessionExecutionLease for MatrixGoalExecutionLease {
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
            GoalParentTurnKind::Start | GoalParentTurnKind::Resume => {
                let mut history =
                    self.initial_working_history(provider.as_ref(), &route, turn.working_history);
                history.insert(1, directive);
                history
            }
            GoalParentTurnKind::Continue => {
                let mut history = turn.working_history;
                history.push(directive);
                history
            }
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
        let candidate = scope_session_key(Some(self.session_key.durable_id()), async {
            scope_thread_id(Some(self.message.id.clone()), tool_loop).await
        })
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
        let messages = vec![
            ChatMessage::system(
                "Return only strict JSON: {\\\"decision\\\":\\\"complete|continue|blocked\\\",\\\"reason\\\":\\\"...\\\",\\\"blockers\\\":[...]}.",
            ),
            ChatMessage::user(format!(
                "Objective:\n{}\n\nCandidate:\n{}",
                turn.objective, turn.candidate
            )),
        ];
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
        let key = match notice {
            GoalExecutionNotice::Completed => "goal-mode-completed",
            GoalExecutionNotice::PausedForBlocker => "goal-mode-paused-blocked",
        };
        let channel = find_channel_for_message(&self.context.channels_by_name, &self.message)
            .context("Matrix Goal channel is no longer available")?;
        channel
            .send(&zeroclaw_api::channel::SendMessage::reply_to(
                &self.message,
                zeroclaw_runtime::i18n::get_required_cli_string(key),
            ))
            .await
            .context("deliver Matrix Goal lifecycle notice")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goal_parent_directive_keeps_lifecycle_phase_out_of_user_control() {
        for (kind, expected) in [
            (GoalParentTurnKind::Start, "start"),
            (GoalParentTurnKind::Resume, "resume"),
            (GoalParentTurnKind::Continue, "continue"),
        ] {
            let directive = goal_parent_directive(&GoalParentTurn {
                kind,
                objective: "the trusted success criterion".to_owned(),
                working_history: Vec::new(),
            });
            assert_eq!(directive.role, "system");
            assert!(directive.content.contains("the trusted success criterion"));
            assert!(
                directive
                    .content
                    .contains(&format!("Turn kind: {expected}"))
            );
        }
    }
}
