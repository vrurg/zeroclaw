//! Matrix implementation of the transport-neutral Goal session driver.
//!
//! This module owns only live Matrix session mechanics. Durable lifecycle,
//! accounting, and execution fencing remain in `zeroclaw-runtime`.

use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result, bail, ensure};
use async_trait::async_trait;
use zeroclaw_api::{
    channel::{Channel, ChannelMessage, SendMessage},
    model_provider::{ChatMessage, ChatRequest},
};
use zeroclaw_commands::goal::GoalBudgetLimits;
use zeroclaw_goal::control::{dispose_unowned_session_goal, scope_goal_parent_turn};
use zeroclaw_goal::{
    GoalExecutionNotice, GoalExecutionScope, GoalExecutionSubmission, GoalExecutionSupervisor,
    GoalFailureReason, GoalHostSettings, GoalIngressContext, GoalIngressPrincipal,
    GoalOperationScope, GoalParentTurn, GoalParentTurnKind, GoalParentTurnResult, GoalPauseReason,
    GoalResponse, GoalRuntime, GoalSessionBinding, GoalSessionDriver, GoalSessionExecutionLease,
    GoalSessionKey, GoalSessionLease, GoalSurface, GoalVerifierAttribution, GoalVerifierResult,
    GoalVerifierTurn, checkpoint::GoalParentTurnCheckpoint, control::goal_parent_turn_input,
};
use zeroclaw_runtime::{
    agent::cost::build_type_level_model_provider_pricing,
    agent::loop_::{
        LoopKnobs, ProgressEvent, StreamDelta, ToolLoop, run_tool_call_loop, scope_session_key,
        scope_thread_id,
    },
    control_plane::control_plane,
    cost::CostTracker,
    goal_accounting::{ModelProviderGoalUsagePricer, RuntimeGoalOperationAccountingFactory},
};

use super::foreground::{foreground_lock, goal_command_lock, goal_supervisor_slot};
use super::{
    ChannelRouteSelection, ChannelRuntimeContext, append_sender_turn_durably,
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
    /// The per-session lifecycle boundary is acquired before selecting a
    /// resident supervisor. `bind` transfers it to the runtime submission so
    /// selection, durable transition, worker launch, and retirement remain
    /// one serialized operation.
    command_lease: Arc<Mutex<Option<tokio::sync::OwnedMutexGuard<()>>>>,
}

impl MatrixGoalSessionDriver {
    fn new(
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
            GoalIngressPrincipal::Matrix { raw_mxid }
                if raw_mxid.as_str() == self.raw_mxid.as_str() =>
            {
                Ok(())
            }
            _ => bail!("Goal ingress principal is stale"),
        }
    }
}

async fn acquire_goal_foreground_lease(
    foreground_lock: Arc<tokio::sync::Mutex<()>>,
    checkpoint: &GoalParentTurnCheckpoint,
) -> Result<tokio::sync::OwnedMutexGuard<()>> {
    tokio::select! {
        biased;
        _ = checkpoint.wait_closed() => bail!("Goal execution was fenced before foreground admission"),
        foreground = foreground_lock.lock_owned() => Ok(foreground),
    }
}

/// Holds a Matrix Goal response together with the worker that must not start
/// until that response has been delivered.
pub(super) struct MatrixGoalCommandSubmission {
    supervisor: Arc<GoalExecutionSupervisor>,
    supervisor_slot: Arc<tokio::sync::Mutex<Option<Arc<GoalExecutionSupervisor>>>>,
    settings: GoalHostSettings,
    submission: GoalExecutionSubmission,
}

impl MatrixGoalCommandSubmission {
    pub(super) fn response(&self) -> &GoalResponse {
        self.submission.response()
    }

    /// Release the admitted worker after Matrix confirms lifecycle delivery.
    pub(super) async fn after_response(&mut self) -> Result<Option<GoalResponse>> {
        self.supervisor
            .launch_after_response(self.settings.clone(), &mut self.submission)
            .await
    }

    /// Retire a supervisor only after the corresponding final lifecycle
    /// response has been delivered by Matrix.
    pub(super) async fn final_response_delivered(&self, late_response: Option<&GoalResponse>) {
        let initial_is_final = matches!(
            self.submission.response(),
            GoalResponse::Paused(_)
                | GoalResponse::AlreadyPaused(_)
                | GoalResponse::Cancelled(_)
                | GoalResponse::AlreadyCancelled(_)
                | GoalResponse::Terminal(_)
        );
        let pending_was_finalized = matches!(
            self.submission.response(),
            GoalResponse::TransitionPending(_)
        ) && late_response.is_some();
        if initial_is_final || pending_was_finalized {
            clear_supervisor_slot_if_current(&self.supervisor_slot, &self.supervisor).await;
        }
    }

    /// Keep an unseen Goal from starting when Matrix cannot deliver its
    /// lifecycle acknowledgement.
    pub(super) async fn response_delivery_failed(mut self) -> Result<()> {
        self.supervisor
            .pause_before_launch(&mut self.submission, GoalPauseReason::ExternalDependency)
            .await?;
        clear_supervisor_slot_if_current(&self.supervisor_slot, &self.supervisor).await;
        Ok(())
    }
}

/// Deliver `/stop`'s Goal-pause lifecycle response before draining the
/// already-admitted parent operation. This mirrors the ordinary Matrix Goal
/// command ordering so the user sees the pause acknowledgement promptly.
pub(super) async fn deliver_matrix_goal_stop(
    mut submission: MatrixGoalCommandSubmission,
    channel: Option<Arc<dyn Channel>>,
    message: ChannelMessage,
) {
    let Some(channel) = channel else {
        if let Err(error) = submission.response_delivery_failed().await {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(serde_json::json!({
                        "error": zeroclaw_providers::sanitize_api_error(&error.to_string()),
                    })),
                "Goal stop acknowledgement could not be delivered or settled"
            );
        }
        return;
    };

    let response = super::render_goal_response(submission.response());
    if let Err(error) = channel
        .send(&SendMessage::reply_to(&message, response))
        .await
    {
        if let Err(pause_error) = submission.response_delivery_failed().await {
            ::zeroclaw_log::record!(
                ERROR,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(serde_json::json!({
                        "error": zeroclaw_providers::sanitize_api_error(&pause_error.to_string()),
                    })),
                "Goal could not be safely settled after /stop acknowledgement failure"
            );
        }
        ::zeroclaw_log::record!(
            WARN,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                .with_attrs(serde_json::json!({
                    "error": zeroclaw_providers::sanitize_api_error(&error.to_string()),
                })),
            "Goal /stop acknowledgement could not be delivered"
        );
        return;
    }

    let late_response = match submission.after_response().await {
        Ok(response) => response,
        Err(error) => {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(serde_json::json!({
                        "error": zeroclaw_providers::sanitize_api_error(&error.to_string()),
                    })),
                "Goal /stop pause could not be finalized after acknowledgement"
            );
            return;
        }
    };
    if let Some(response) = late_response.as_ref()
        && let Err(error) = channel
            .send(&SendMessage::reply_to(
                &message,
                super::render_goal_response(response),
            ))
            .await
    {
        ::zeroclaw_log::record!(
            WARN,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                .with_attrs(serde_json::json!({
                    "error": zeroclaw_providers::sanitize_api_error(&error.to_string()),
                })),
            "Final Goal pause response after /stop could not be delivered"
        );
    }
    submission
        .final_response_delivered(late_response.as_ref())
        .await;
}

/// Submit an already parsed Goal command using immutable Matrix ingress facts.
pub(super) async fn submit_matrix_goal(
    context: Arc<ChannelRuntimeContext>,
    history_key: String,
    original: ChannelMessage,
    command: zeroclaw_commands::goal::GoalCommand,
) -> Result<MatrixGoalCommandSubmission> {
    // Acquire this before reading durable state or selecting a supervisor.
    // In particular, a terminal predecessor must be drained by the same
    // serialized lifecycle operation before another command can install a
    // successor supervisor for this session.
    let command_lock = goal_command_lock(&context.persist_locks, &history_key);
    let command_lease = command_lock.lock_owned().await;
    submit_matrix_goal_with_command_lease(context, history_key, original, command, command_lease)
        .await
}

async fn submit_matrix_goal_with_command_lease(
    context: Arc<ChannelRuntimeContext>,
    history_key: String,
    original: ChannelMessage,
    command: zeroclaw_commands::goal::GoalCommand,
    command_lease: tokio::sync::OwnedMutexGuard<()>,
) -> Result<MatrixGoalCommandSubmission> {
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
    let defaults = runtime_defaults_snapshot(context.as_ref());
    let effective_limits =
        defaults.config.goal.effective_limits().map_err(|error| {
            anyhow::Error::msg(format!("invalid Goal default limits: {error:?}"))
        })?;
    let settings = GoalHostSettings::from_validated_limits(
        defaults.config.goal.enabled,
        GoalBudgetLimits {
            token_limit: effective_limits.token_limit,
            cost_limit_usd: effective_limits.cost_limit_usd,
        },
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
            let runtime = GoalRuntime::new(Arc::clone(&registry));
            let tracker = CostTracker::get_or_init_global_required(
                defaults.config.cost.clone(),
                &defaults.config.data_dir,
            )?;
            let pricer = Arc::new(ModelProviderGoalUsagePricer::new(Arc::new(
                build_type_level_model_provider_pricing(defaults.config.as_ref()),
            )));
            let accounting = Arc::new(RuntimeGoalOperationAccountingFactory::new(
                Arc::clone(&registry),
                tracker,
                context.agent_alias.to_string(),
                pricer,
            )?);
            let engine = Arc::new(runtime.execution_engine(accounting));
            let supervisor = restart_coordinator.new_supervisor(engine).await;
            *slot = Some(Arc::clone(&supervisor));
            supervisor
        }
    };
    let ingress = GoalIngressContext::trusted(
        GoalSessionKey::matrix(history_key)?,
        context.agent_alias.to_string(),
        route,
        GoalIngressPrincipal::Matrix {
            raw_mxid: driver.raw_mxid.clone(),
        },
    )?;
    let submission = supervisor
        .submit(settings.clone(), ingress, driver, command)
        .await?;
    Ok(MatrixGoalCommandSubmission {
        supervisor,
        supervisor_slot,
        settings,
        submission,
    })
}

/// Route Matrix `/stop` through the ordinary immediate Goal-pause command
/// when the sender's canonical session currently owns a running Goal.
///
/// The durable task row is the authority for whether a Goal is running. The
/// process-local supervisor slot alone is not sufficient: it may be absent
/// after recovery or retained briefly while a terminal transition is being
/// delivered.
pub(super) async fn submit_matrix_goal_stop(
    context: Arc<ChannelRuntimeContext>,
    message: ChannelMessage,
) -> Result<Option<MatrixGoalCommandSubmission>> {
    if !message.channel.eq_ignore_ascii_case("matrix") {
        return Ok(None);
    }

    let history_key = super::runtime_conversation_history_key(context.as_ref(), &message);
    // Serialize the durable Running check with starts and other lifecycle
    // commands. Otherwise a concurrent start could commit after the read and
    // leave this `/stop` following the ordinary no-task path.
    let command_lease = goal_command_lock(&context.persist_locks, &history_key)
        .lock_owned()
        .await;
    let Some(control_plane) = zeroclaw_runtime::control_plane::control_plane() else {
        // Goal admission itself requires this control plane, so no live Goal
        // can be running when it is unavailable.
        return Ok(None);
    };
    let registry = control_plane.goal_store()?;
    let Some(task) = registry.current_goal_for_session(&history_key).await? else {
        return Ok(None);
    };
    let Some(command) = goal_stop_command_for_status(task.status) else {
        return Ok(None);
    };

    submit_matrix_goal_with_command_lease(context, history_key, message, command, command_lease)
        .await
        .map(Some)
}

fn goal_stop_command_for_status(
    status: zeroclaw_runtime::control_plane::TaskStatus,
) -> Option<zeroclaw_commands::goal::GoalCommand> {
    (status == zeroclaw_runtime::control_plane::TaskStatus::Running)
        .then_some(zeroclaw_commands::goal::GoalCommand::Pause { immediate: true })
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

struct MatrixGoalParentDelivery {
    channel: Arc<dyn Channel>,
    draft_id: Option<String>,
    streamed_text: Arc<Mutex<String>>,
}

struct MatrixGoalDraftStream {
    channel: Arc<dyn Channel>,
    delta_tx: Option<tokio::sync::mpsc::Sender<StreamDelta>>,
    draft_id: Option<String>,
    updater: Option<tokio::task::JoinHandle<()>>,
    streamed_text: Arc<Mutex<String>>,
    single_message: bool,
    is_partial_draft: bool,
}

async fn start_matrix_goal_draft_stream(
    context: &ChannelRuntimeContext,
    message: &ChannelMessage,
    channel: Arc<dyn Channel>,
) -> MatrixGoalDraftStream {
    let single_message = super::matrix_single_message_streaming_enabled(context, message);
    let use_draft_streaming = channel.supports_draft_updates();
    let is_partial_draft =
        (use_draft_streaming && !channel.supports_multi_message_streaming()) || single_message;
    let mut typing_scope = if single_message && use_draft_streaming {
        Some(super::start_matrix_single_message_typing_scope(
            Arc::clone(&channel),
            message.reply_target.clone(),
        ))
    } else {
        None
    };

    let (delta_tx, delta_rx) = if use_draft_streaming {
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamDelta>(64);
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };
    let draft_id = if use_draft_streaming {
        if let Some(scope) = typing_scope.take() {
            super::stop_matrix_single_message_typing_scope(scope).await;
        }
        match channel
            .send_draft(&SendMessage::reply_to(
                message,
                zeroclaw_runtime::agent::loop_::DRAFT_PLACEHOLDER,
            ))
            .await
        {
            Ok(id) => id,
            Err(error) => {
                ::zeroclaw_log::record!(
                    DEBUG,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                        .with_attrs(serde_json::json!({"error": format!("{error}")})),
                    &format!("Failed to start Matrix Goal draft on {}", channel.name())
                );
                None
            }
        }
    } else {
        None
    };

    let streamed_text = Arc::new(Mutex::new(String::new()));
    let updater = match (delta_rx, draft_id.as_deref()) {
        (Some(rx), Some(draft_id)) if single_message => {
            let channel = Arc::clone(&channel);
            let reply_target = message.reply_target.clone();
            let draft_id = draft_id.to_owned();
            let interval_ms = super::matrix_draft_update_interval_ms(context, message);
            let stream_draft_lines = super::matrix_stream_draft_lines(context, message);
            let matrix_config = Arc::clone(&context.prompt_config);
            let matrix_alias = message.channel_alias.clone().unwrap_or_default();
            Some(zeroclaw_spawn::spawn!(async move {
                super::run_matrix_single_message_draft_updater(
                    rx,
                    channel,
                    reply_target,
                    draft_id,
                    interval_ms,
                    stream_draft_lines,
                    matrix_config,
                    matrix_alias,
                )
                .await;
            }))
        }
        (Some(rx), Some(draft_id)) => {
            let known_tool_names = context
                .tools_registry
                .iter()
                .map(|tool| tool.name().to_ascii_lowercase())
                .collect();
            let reply_target = message.reply_target.clone();
            let draft_id = draft_id.to_owned();
            let channel = Arc::clone(&channel);
            let streamed_text = Arc::clone(&streamed_text);
            let leak_detection = context.prompt_config.security.leak_detection.clone();
            let content_format = super::outbound_content_format_for_channel(&message.channel);
            let turn_flush_narration = channel.supports_turn_flush_narration();
            let outbound_hooks = context.hooks.clone();
            let outbound_channel = message.channel.clone();
            Some(zeroclaw_spawn::spawn!(async move {
                super::run_draft_updater_with_leak_detection(
                    channel,
                    reply_target,
                    draft_id,
                    known_tool_names,
                    streamed_text,
                    leak_detection,
                    content_format,
                    turn_flush_narration,
                    outbound_hooks,
                    outbound_channel,
                    rx,
                )
                .await;
            }))
        }
        (Some(_), None) | (None, _) => None,
    };

    if let Some(tx) = delta_tx.as_ref() {
        let _ = tx
            .send(StreamDelta::Lifecycle(ProgressEvent::Received))
            .await;
        let _ = tx
            .send(StreamDelta::Lifecycle(ProgressEvent::Planning))
            .await;
    }

    MatrixGoalDraftStream {
        channel,
        delta_tx,
        draft_id,
        updater,
        streamed_text,
        single_message,
        is_partial_draft,
    }
}

async fn finish_matrix_goal_draft_stream(
    mut stream: MatrixGoalDraftStream,
    message: &ChannelMessage,
    completed: bool,
) -> MatrixGoalParentDelivery {
    if completed && let Some(tx) = stream.delta_tx.as_ref() {
        let _ = tx
            .send(StreamDelta::Lifecycle(ProgressEvent::FinalizingResponse))
            .await;
    }
    drop(stream.delta_tx.take());
    if let Some(updater) = stream.updater.take() {
        let _ = updater.await;
    }
    if !completed
        && let Some(draft_id) = stream.draft_id.as_deref()
        && let Err(error) = stream
            .channel
            .cancel_draft(&message.reply_target, draft_id)
            .await
    {
        ::zeroclaw_log::record!(
            DEBUG,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                .with_attrs(serde_json::json!({"error": format!("{error}")})),
            &format!(
                "Failed to cancel Matrix Goal draft on {}",
                stream.channel.name()
            )
        );
    }
    MatrixGoalParentDelivery {
        channel: stream.channel,
        draft_id: stream.draft_id,
        streamed_text: stream.streamed_text,
    }
}

fn matrix_goal_core_error_text(error: &anyhow::Error) -> Option<String> {
    if zeroclaw_runtime::agent::loop_::is_tool_loop_cancelled(error) {
        return None;
    }
    let safe_error = zeroclaw_providers::sanitize_api_error(&error.to_string());
    Some(super::channel_user_error_message(error, &safe_error))
}

async fn publish_matrix_goal_core_error(
    channel: &dyn Channel,
    message: &ChannelMessage,
    error: &anyhow::Error,
) {
    let Some(error_text) = matrix_goal_core_error_text(error) else {
        return;
    };
    // Match the ordinary channel-turn error path: render the safe core error
    // as a plain, non-voice reply rather than hiding it in Goal status.
    let _ = channel
        .send(&SendMessage::reply_to(message, &error_text).suppress_voice())
        .await;
}

/// Clear a session's resident supervisor only if it is still the one that
/// performed the lifecycle transition.
///
/// The submission lease remains held through this cleanup, so another Goal
/// command cannot reuse or replace the supervisor while its predecessor is
/// retiring. Pointer comparison additionally protects disposal and other
/// lifecycle paths that do not own that command lease.
async fn clear_supervisor_slot_if_current(
    slot: &Arc<tokio::sync::Mutex<Option<Arc<GoalExecutionSupervisor>>>>,
    supervisor: &Arc<GoalExecutionSupervisor>,
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
        checkpoint: &GoalParentTurnCheckpoint,
    ) -> Result<Box<dyn GoalSessionExecutionLease>> {
        self.assert_ingress(ingress)?;
        ensure!(
            scope.session_id() == self.session_key.durable_id(),
            "Goal execution session is stale"
        );
        let foreground_lock =
            foreground_lock(&self.context.persist_locks, &self.session_key.durable_id());
        let foreground = acquire_goal_foreground_lease(foreground_lock, checkpoint).await?;
        Ok(Box::new(MatrixGoalExecutionLease {
            _foreground: foreground,
            context: Arc::clone(&self.context),
            session_key: self.session_key.clone(),
            message: self.message.clone(),
            pending_delivery: None,
            pending_receipts: None,
            history_has_trim_breadcrumb: false,
        }))
    }
}

struct MatrixGoalExecutionLease {
    _foreground: tokio::sync::OwnedMutexGuard<()>,
    context: Arc<ChannelRuntimeContext>,
    session_key: GoalSessionKey,
    message: ChannelMessage,
    pending_delivery: Option<MatrixGoalParentDelivery>,
    pending_receipts: Option<String>,
    history_has_trim_breadcrumb: bool,
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

    fn base_system_prompt(
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
        let target_channel =
            find_channel_for_message(&self.context.channels_by_name, &self.message);
        let room_purpose = target_channel
            .and_then(|channel| channel.room_context(&self.message.reply_target))
            .and_then(|room| room.purpose);
        let base_system_prompt = system_prompt_for_channel_turn(
            self.context.as_ref(),
            self.context.system_prompt.as_str(),
            true,
            callable_protocol_exposed,
            excluded_tools,
            room_purpose.as_deref(),
        );
        build_channel_system_prompt_for_message_with_signal(
            &base_system_prompt,
            &self.message,
            target_channel,
            native_tool_specs_present,
        )
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

#[async_trait]
impl GoalSessionExecutionLease for MatrixGoalExecutionLease {
    fn session_key(&self) -> &GoalSessionKey {
        &self.session_key
    }

    fn canonical_history(&self) -> Result<Vec<ChatMessage>> {
        Ok(self.history())
    }

    async fn append_resume_response(&mut self, response: String) -> Result<()> {
        // `/goal resume <response>` is a user turn, not Goal metadata.  Use
        // the ordinary Matrix canonical-history append path so the exact
        // multiline response survives a daemon restart and reaches the next
        // resumed parent turn just like any other user message.
        let appended = append_sender_turn_durably(
            self.context.as_ref(),
            &self.session_key.durable_id(),
            ChatMessage::user(&response),
        );
        ensure!(
            appended.is_some(),
            "cannot resume Goal because the Matrix session transcript could not be persisted"
        );
        Ok(())
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
        let input = goal_parent_turn_input(&turn);
        let base_system_prompt = self.base_system_prompt(provider.as_ref(), &route);
        let mut history = match turn.kind {
            GoalParentTurnKind::Start | GoalParentTurnKind::Resume => {
                input.goal_start_transcript(&base_system_prompt, turn.working_history)?
            }
            GoalParentTurnKind::Continue => {
                let mut history = turn.working_history;
                input.refresh_goal_continuation(&base_system_prompt, &mut history)?;
                history
            }
        };
        ensure!(
            self.pending_delivery.is_none() && self.pending_receipts.is_none(),
            "Matrix Goal parent delivery was not settled before the next turn"
        );
        let channel = find_channel_for_message(&self.context.channels_by_name, &self.message)
            .context("Matrix Goal channel is no longer available")?;
        let draft_stream = start_matrix_goal_draft_stream(
            self.context.as_ref(),
            &self.message,
            Arc::clone(channel),
        )
        .await;
        let (notify_tx, notify_task) = if draft_stream.single_message {
            (None, None)
        } else {
            let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(128);
            let notify_channel = Arc::clone(channel);
            let reply_target = self.message.reply_target.clone();
            let thread = super::followup_thread_id(&self.message);
            let notify_task = if !self.context.show_tool_calls || draft_stream.is_partial_draft {
                Some(zeroclaw_spawn::spawn!(async move {
                    while rx.recv().await.is_some() {}
                }))
            } else {
                Some(zeroclaw_spawn::spawn!(async move {
                    while let Some(text) = rx.recv().await {
                        let _ = notify_channel
                            .send(
                                &SendMessage::new(&text, &reply_target)
                                    .in_thread(thread.clone())
                                    .suppress_voice(),
                            )
                            .await;
                    }
                }))
            };
            (Some(tx), notify_task)
        };
        let notify_observer = Arc::new(super::ChannelNotifyObserver {
            inner: Arc::clone(&self.context.observer),
            tx: notify_tx,
            tools_used: std::sync::atomic::AtomicBool::new(false),
        });
        let typing_controller = if draft_stream.is_partial_draft {
            None
        } else {
            Some(Arc::new(super::ScopedTypingController::new(
                Arc::clone(channel),
                self.message.reply_target.clone(),
            )))
        };
        if let Some(typing) = typing_controller.as_ref() {
            typing.resume().await;
        }
        let approval_origin: Option<Arc<dyn Channel>> = match typing_controller.as_ref() {
            Some(typing) => Some(Arc::new(super::ApprovalTypingChannel::new(
                Arc::clone(channel),
                Arc::clone(typing),
            ))),
            None => Some(Arc::clone(channel)),
        };
        let (approval_manager, approval_channel) = super::channel_turn_approval(
            &self.context.approval_manager,
            self.context
                .prompt_config
                .risk_profile_for_agent(self.context.agent_alias.as_str()),
            self.context.channels_by_name.as_ref(),
            approval_origin,
        );
        let turn_id = uuid::Uuid::new_v4().to_string();
        let mut turn_guard = zeroclaw_runtime::observability::AgentTurnGuard::start(
            self.context.observer.as_ref(),
            route.model_provider.clone(),
            route.model.clone(),
            Some("matrix".to_owned()),
            Some(self.context.agent_alias.to_string()),
            Some(turn_id.clone()),
        );
        let mut loop_knobs = LoopKnobs::default();
        if draft_stream.single_message {
            loop_knobs.draft_reasoning =
                super::matrix_stream_reasoning(self.context.as_ref(), &self.message);
        }
        let history_key = self.session_key.durable_id();
        let mut memory_sessions: Vec<Option<String>> =
            super::sender_memory_session_ids(&self.message, &history_key)
                .into_iter()
                .map(Some)
                .collect();
        if super::is_group_reply_target(&self.message.reply_target) {
            memory_sessions.push(Some(history_key));
        }
        let receipt_collector: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let receipt_scope = self.context.receipt_generator.as_ref().map(|generator| {
            zeroclaw_runtime::agent::tool_receipts::ReceiptScope {
                generator: generator.clone(),
                collector: Arc::clone(&receipt_collector),
            }
        });
        let mut injected_memory_preamble = None;
        let tool_loop = run_tool_call_loop(ToolLoop {
            exec: resolved_channel_execution(
                self.context.as_ref(),
                provider.as_ref(),
                &route,
                notify_observer.as_ref(),
                &approval_manager,
                &loop_knobs,
                self.context.non_cli_excluded_tools.as_ref(),
                defaults.defaults.temperature,
            ),
            history: &mut history,
            history_has_trim_breadcrumb: &mut self.history_has_trim_breadcrumb,
            injected_memory_preamble: &mut injected_memory_preamble,
            channel_name: "matrix",
            channel_reply_target: Some(self.message.reply_target.as_str()),
            cancellation_token: None,
            on_delta: draft_stream.delta_tx.clone(),
            shared_budget: None,
            channel: approval_channel.as_deref(),
            collected_receipts: self
                .context
                .receipt_generator
                .as_ref()
                .map(|_| receipt_collector.as_ref()),
            event_tx: None,
            steering: None,
            new_messages_out: None,
            image_cache: None,
            ingress: zeroclaw_api::ingress::IngressContext::channel(),
            memory: Some(zeroclaw_runtime::agent::memory_inject::TurnMemory {
                handle: self.context.memory.as_ref(),
                query: self.message.content.clone(),
                sessions: memory_sessions,
                suppress: false,
                cfg: zeroclaw_runtime::agent::memory_inject::MemoryInjectConfig {
                    min_relevance_score: self.context.min_relevance_score,
                    ..zeroclaw_runtime::agent::memory_inject::MemoryInjectConfig::from_memory_config(
                        &self.context.prompt_config.memory,
                        zeroclaw_runtime::agent::memory_inject::DEFAULT_RECALL_LIMIT,
                    )
                },
            }),
            agent_alias: Some(self.context.agent_alias.as_str()),
            parent_agent_alias: None,
            turn_id: &turn_id,
            served_route_sink: None,
            sop_reassembly: Some(zeroclaw_runtime::agent::loop_::SopStepReassembly {
                config: self.context.prompt_config.as_ref(),
                live_config: Some(Arc::clone(&self.context.live_config)),
            }),
        });
        let candidate = scope_goal_parent_turn(scope_session_key(
            Some(self.session_key.durable_id()),
            async {
                let tool_loop = zeroclaw_runtime::agent::tool_receipts::TOOL_LOOP_RECEIPT_CONTEXT
                    .scope(receipt_scope, tool_loop);
                scope_thread_id(Some(self.message.id.clone()), tool_loop).await
            },
        ))
        .await;
        let candidate = match candidate {
            Ok(candidate) => candidate,
            Err(error) => {
                turn_guard.finish();
                if let Some(typing) = typing_controller.as_ref() {
                    typing.pause().await;
                }
                let _ = finish_matrix_goal_draft_stream(draft_stream, &self.message, false).await;
                drop(notify_observer);
                if let Some(task) = notify_task {
                    let _ = task.await;
                }
                publish_matrix_goal_core_error(channel.as_ref(), &self.message, &error).await;
                return Err(error);
            }
        };
        turn_guard.finish();
        if let Some(typing) = typing_controller.as_ref() {
            typing.pause().await;
        }
        let candidate_is_nonempty = !candidate.trim().is_empty();
        self.pending_receipts = if candidate_is_nonempty && self.context.show_receipts_in_response {
            let receipts = receipt_collector
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            zeroclaw_runtime::agent::tool_receipts::render_receipts_block(&receipts)
        } else {
            None
        };
        let delivery =
            finish_matrix_goal_draft_stream(draft_stream, &self.message, candidate_is_nonempty)
                .await;
        drop(notify_observer);
        if let Some(task) = notify_task {
            let _ = task.await;
        }
        if candidate_is_nonempty {
            self.pending_delivery = Some(delivery);
        }
        let _ = operation;
        GoalParentTurnResult::from_completed_agent_turn(candidate, history)
    }

    async fn run_verifier(
        &mut self,
        _operation: &GoalOperationScope,
        turn: GoalVerifierTurn,
    ) -> Result<GoalVerifierResult> {
        let result: Result<GoalVerifierResult> = async {
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
            let messages = turn.to_chat_messages();
            let mut settled_attempts = Vec::new();
            let approval_manager = self.context.approval_manager.for_new_turn();
            let response = resolved_channel_execution(
                self.context.as_ref(),
                provider.as_ref(),
                &ChannelRouteSelection {
                    model_provider: provider_name,
                    model: model.to_owned(),
                    api_key: entry.api_key.clone(),
                },
                self.context.observer.as_ref(),
                &approval_manager,
                &LoopKnobs::default(),
                &[],
                verifier_defaults.temperature,
            )
            .model_access
            .run_model_query(
                ChatRequest {
                    messages: &messages,
                    tools: None,
                    thinking: None,
                },
                &mut settled_attempts,
            )
            .await?;
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
        .await;
        if let Err(error) = &result
            && let Some(channel) =
                find_channel_for_message(&self.context.channels_by_name, &self.message)
        {
            publish_matrix_goal_core_error(channel.as_ref(), &self.message, error).await;
        }
        result
    }

    async fn append_parent_candidate(&mut self, candidate: String) -> Result<()> {
        let delivery = self
            .pending_delivery
            .take()
            .context("Matrix Goal parent output has no matching delivery session")?;
        let channel = delivery.channel;
        let mut outbound_response = candidate;
        if let Some(hooks) = self.context.hooks.as_ref() {
            match hooks
                .run_on_message_sending(
                    self.message.channel.clone(),
                    self.message.reply_target.clone(),
                    outbound_response.clone(),
                )
                .await
            {
                zeroclaw_runtime::hooks::HookResult::Cancel(reason) => {
                    ::zeroclaw_log::record!(
                        INFO,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note,)
                            .with_attrs(serde_json::json!({"reason": reason.to_string()})),
                        "Matrix Goal parent output suppressed by hook"
                    );
                    if let Some(draft_id) = delivery.draft_id.as_deref() {
                        let _ = channel
                            .cancel_draft(&self.message.reply_target, draft_id)
                            .await;
                    }
                    self.pending_receipts.take();
                    return Ok(());
                }
                zeroclaw_runtime::hooks::HookResult::Continue((
                    hook_channel,
                    hook_recipient,
                    mut modified_content,
                )) => {
                    if hook_channel != self.message.channel
                        || hook_recipient != self.message.reply_target
                    {
                        ::zeroclaw_log::record!(
                            WARN,
                            ::zeroclaw_log::Event::new(
                                module_path!(),
                                ::zeroclaw_log::Action::Note,
                            )
                            .with_outcome(::zeroclaw_log::EventOutcome::Unknown)
                            .with_attrs(serde_json::json!({
                                "from_channel": self.message.channel,
                                "from_recipient": self.message.reply_target,
                                "to_channel": hook_channel,
                                "to_recipient": hook_recipient,
                            })),
                            "on_message_sending attempted to rewrite Goal routing; only content mutation is applied"
                        );
                    }
                    let modified_len = modified_content.chars().count();
                    if modified_len > super::CHANNEL_HOOK_MAX_OUTBOUND_CHARS {
                        modified_content = super::truncate_with_ellipsis(
                            &modified_content,
                            super::CHANNEL_HOOK_MAX_OUTBOUND_CHARS,
                        );
                    }
                    outbound_response = modified_content;
                }
            }
        }
        let delivered = sanitize_channel_response_for_format_with_leak_detection(
            &outbound_response,
            self.context.tools_registry.as_ref(),
            &self.context.prompt_config.security.leak_detection,
            outbound_content_format_for_channel(&self.message.channel),
        );
        let delivered = if delivered.is_empty() && !outbound_response.trim().is_empty() {
            zeroclaw_runtime::i18n::get_required_cli_string("channel-runtime-malformed-tool-output")
        } else {
            delivered
        };
        let delivered = super::ensure_nonempty_channel_reply(
            delivered,
            &outbound_response,
            &self.message.channel,
            &self.message.reply_target,
        );
        let transcript_appended = append_sender_turn_durably(
            self.context.as_ref(),
            &self.session_key.durable_id(),
            ChatMessage::assistant(&delivered),
        );

        let (suppress_voice, force_voice) = super::voice_override_from_sender_verdict(
            super::sender_prefers_voice(self.context.as_ref(), &self.message),
        );
        let reply_delivered = if let Some(draft_id) = delivery.draft_id.as_deref() {
            let final_text = if channel.supports_multi_message_streaming() {
                let confirmed_offset = channel
                    .multi_message_confirmed_offset(&self.message.reply_target, draft_id)
                    .await;
                let streamed_text = delivery
                    .streamed_text
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                super::cumulative_multi_message_final_text(
                    &streamed_text,
                    &delivered,
                    confirmed_offset,
                )
            } else {
                delivered.clone()
            };
            match channel
                .finalize_draft(
                    &self.message.reply_target,
                    draft_id,
                    &final_text,
                    suppress_voice.unwrap_or(false),
                )
                .await
            {
                Ok(()) => true,
                Err(error)
                    if error
                        .downcast_ref::<zeroclaw_api::channel::FinalizePartialDelivery>()
                        .is_some() =>
                {
                    true
                }
                Err(error) => {
                    ::zeroclaw_log::record!(
                        WARN,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note,)
                            .with_outcome(::zeroclaw_log::EventOutcome::Unknown)
                            .with_attrs(serde_json::json!({"error": format!("{error}")})),
                        "Failed to finalize Matrix Goal draft; sending final candidate separately"
                    );
                    let mut fallback = SendMessage::reply_to(&self.message, &delivered);
                    if suppress_voice == Some(true) {
                        fallback = fallback.suppress_voice();
                    } else if force_voice {
                        fallback = fallback.force_voice();
                    }
                    channel.send_final(&fallback).await.is_ok()
                }
            }
        } else {
            let mut reply = SendMessage::reply_to(&self.message, &delivered);
            if suppress_voice == Some(true) {
                reply = reply.suppress_voice();
            } else if force_voice {
                reply = reply.force_voice();
            }
            channel.send_final(&reply).await.is_ok()
        };
        if reply_delivered && let Some(hooks) = self.context.hooks.as_ref() {
            hooks
                .fire_message_sent(
                    &self.message.channel,
                    &self.message.reply_target,
                    &delivered,
                )
                .await;
        }
        let receipts_block = self.pending_receipts.take();
        if reply_delivered
            && let Some(block) = receipts_block.as_deref()
            && let Err(error) = channel
                .send(
                    &SendMessage::new(block, &self.message.reply_target)
                        .in_thread(self.message.thread_ts.clone())
                        .suppress_voice(),
                )
                .await
        {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                    .with_outcome(::zeroclaw_log::EventOutcome::Unknown)
                    .with_attrs(serde_json::json!({"error": format!("{error}")})),
                "failed to send Matrix Goal tool receipts block"
            );
        }
        if !reply_delivered {
            let error = anyhow::Error::msg("Matrix Goal parent candidate could not be delivered");
            publish_matrix_goal_core_error(channel.as_ref(), &self.message, &error).await;
            return Err(error);
        }
        if transcript_appended.is_none() {
            let error = anyhow::Error::msg(
                "Matrix Goal parent candidate could not be durably appended to the session transcript",
            );
            publish_matrix_goal_core_error(channel.as_ref(), &self.message, &error).await;
            return Err(error);
        }
        Ok(())
    }

    async fn publish_goal_notice(&mut self, notice: GoalExecutionNotice) -> Result<()> {
        let key = match notice {
            GoalExecutionNotice::Completed => "goal-mode-completed",
            GoalExecutionNotice::PausedForBlocker => "goal-mode-paused-blocked",
            GoalExecutionNotice::Failed(reason) => matrix_goal_failure_key(reason),
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

fn matrix_goal_failure_key(reason: GoalFailureReason) -> &'static str {
    match reason {
        GoalFailureReason::AgentOperationFailed => "goal-mode-failure-agent-operation",
        GoalFailureReason::ToolResultsUnpaired => "goal-mode-failure-tool-results-unpaired",
        GoalFailureReason::EmptyAgentResponse => "goal-mode-failure-empty-agent-response",
        GoalFailureReason::VerifierOperationFailed => "goal-mode-failure-verifier-operation",
        GoalFailureReason::VerifierResponseInvalid => "goal-mode-failure-verifier-response",
        GoalFailureReason::AccountingIncomplete => "goal-mode-failure-accounting",
        GoalFailureReason::PricingIncomplete => "goal-mode-failure-pricing",
        GoalFailureReason::OperationOutcomeUnknown => "goal-mode-failure-outcome-unknown",
        GoalFailureReason::ExecutorStartFailed => "goal-mode-failure-executor-start",
        GoalFailureReason::ExecutorFailed => "goal-mode-failure-executor",
        GoalFailureReason::Other => "goal-mode-failure-other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_goal_failure_reason_uses_a_fluent_key() {
        for reason in [
            GoalFailureReason::AgentOperationFailed,
            GoalFailureReason::ToolResultsUnpaired,
            GoalFailureReason::EmptyAgentResponse,
            GoalFailureReason::VerifierOperationFailed,
            GoalFailureReason::VerifierResponseInvalid,
            GoalFailureReason::AccountingIncomplete,
            GoalFailureReason::PricingIncomplete,
            GoalFailureReason::OperationOutcomeUnknown,
            GoalFailureReason::ExecutorStartFailed,
            GoalFailureReason::ExecutorFailed,
            GoalFailureReason::Other,
        ] {
            assert!(matrix_goal_failure_key(reason).starts_with("goal-mode-failure-"));
        }
    }

    #[test]
    fn matrix_stop_only_maps_a_running_goal_to_immediate_pause() {
        assert_eq!(
            goal_stop_command_for_status(zeroclaw_runtime::control_plane::TaskStatus::Running),
            Some(zeroclaw_commands::goal::GoalCommand::Pause { immediate: true })
        );
        for status in [
            zeroclaw_runtime::control_plane::TaskStatus::Paused,
            zeroclaw_runtime::control_plane::TaskStatus::Completed,
            zeroclaw_runtime::control_plane::TaskStatus::Failed,
            zeroclaw_runtime::control_plane::TaskStatus::Cancelled,
            zeroclaw_runtime::control_plane::TaskStatus::Lost,
            zeroclaw_runtime::control_plane::TaskStatus::TimedOut,
        ] {
            assert_eq!(goal_stop_command_for_status(status), None);
        }
    }

    #[derive(Default)]
    struct MatrixGoalErrorRecordingChannel {
        messages: tokio::sync::Mutex<Vec<String>>,
    }

    impl zeroclaw_api::attribution::Attributable for MatrixGoalErrorRecordingChannel {
        fn role(&self) -> zeroclaw_api::attribution::Role {
            zeroclaw_api::attribution::Role::Channel(zeroclaw_api::attribution::ChannelKind::Matrix)
        }

        fn alias(&self) -> &str {
            "goal-error-test"
        }
    }

    #[async_trait::async_trait]
    impl Channel for MatrixGoalErrorRecordingChannel {
        fn name(&self) -> &str {
            "matrix"
        }

        async fn send(&self, message: &SendMessage) -> anyhow::Result<()> {
            self.messages.lock().await.push(message.content.clone());
            Ok(())
        }

        async fn listen(
            &self,
            _tx: tokio::sync::mpsc::Sender<ChannelMessage>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    struct FailingAppendSessionBackend;

    impl zeroclaw_infra::session_backend::SessionBackend for FailingAppendSessionBackend {
        fn load(&self, _session_key: &str) -> Vec<ChatMessage> {
            Vec::new()
        }

        fn append(&self, _session_key: &str, _message: &ChatMessage) -> std::io::Result<()> {
            Err(std::io::Error::other("simulated append failure"))
        }

        fn remove_last(&self, _session_key: &str) -> std::io::Result<bool> {
            Ok(false)
        }

        fn list_sessions(&self) -> Vec<String> {
            Vec::new()
        }
    }

    #[test]
    fn matrix_goal_does_not_advance_canonical_cache_when_durable_append_fails() {
        let context =
            super::super::test_channel_ctx_with_backend(Arc::new(FailingAppendSessionBackend));
        let key = "goal-session";

        assert!(
            append_sender_turn_durably(&context, key, ChatMessage::user("resume answer"),)
                .is_none()
        );
        assert!(
            context
                .conversation_histories
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .peek(key)
                .is_none()
        );
    }

    #[tokio::test]
    async fn matrix_goal_fence_cancels_only_a_waiting_foreground_admission() {
        let foreground_lock = Arc::new(tokio::sync::Mutex::new(()));
        let held = foreground_lock.clone().lock_owned().await;
        let checkpoint = GoalParentTurnCheckpoint::new();
        let waiter_checkpoint = checkpoint.clone();
        let mut waiter = zeroclaw_spawn::spawn!(async move {
            acquire_goal_foreground_lease(foreground_lock, &waiter_checkpoint).await
        });

        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), &mut waiter)
                .await
                .is_err(),
            "foreground admission should remain pending while another turn owns the lease"
        );
        checkpoint.close();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(500), waiter)
                .await
                .expect("checkpoint closure must wake the lease waiter")
                .expect("lease waiter must not panic")
                .is_err(),
            "a fenced Goal must not acquire the foreground lease"
        );
        drop(held);
    }

    #[tokio::test]
    async fn matrix_goal_surfaces_core_error_through_channel_error_renderer() {
        let channel = MatrixGoalErrorRecordingChannel::default();
        let message = ChannelMessage::new("event", "alice", "!room", "/goal start", "matrix", 1);
        let error = anyhow::Error::msg("provider rejected the request");
        let expected = super::super::channel_user_error_message(
            &error,
            &zeroclaw_providers::sanitize_api_error(&error.to_string()),
        );

        publish_matrix_goal_core_error(&channel, &message, &error).await;

        assert_eq!(channel.messages.lock().await.as_slice(), [expected]);
    }

    #[tokio::test]
    async fn matrix_goal_does_not_surface_controlled_tool_loop_cancellation_as_error() {
        let channel = MatrixGoalErrorRecordingChannel::default();
        let message = ChannelMessage::new("event", "alice", "!room", "/goal start", "matrix", 1);
        let error = anyhow::Error::new(zeroclaw_runtime::agent::loop_::ToolLoopCancelled);

        publish_matrix_goal_core_error(&channel, &message, &error).await;

        assert!(channel.messages.lock().await.is_empty());
    }

    #[test]
    fn verifier_attribution_uses_the_provider_and_model_of_the_accepted_attempt() {
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
    fn verifier_attribution_is_absent_when_no_attempt_was_accepted() {
        let attribution = verifier_attribution_from_attempts([
            (false, "openai.primary", "gpt-primary"),
            (false, "anthropic.fallback", "claude-fallback"),
        ]);

        assert_eq!(attribution, None);
    }

    #[test]
    fn goal_parent_input_keeps_lifecycle_phase_out_of_user_control() {
        for (kind, expected) in [
            (GoalParentTurnKind::Start, "start"),
            (GoalParentTurnKind::Resume, "resume"),
            (GoalParentTurnKind::Continue, "continue"),
        ] {
            let input = goal_parent_turn_input(&GoalParentTurn {
                kind,
                objective: "the trusted success criterion".to_owned(),
                working_history: Vec::new(),
            });
            let directive = input.compose_goal_system_row("Ordinary policy");
            assert_eq!(directive.role, "system");
            assert!(directive.content.contains("the trusted success criterion"));
            assert!(
                directive
                    .content
                    .contains(&format!("Turn kind (trusted runtime fact): {expected}"))
            );
        }
    }

    #[test]
    fn start_history_preserves_the_leading_canonical_user_slot() {
        let input = goal_parent_turn_input(&GoalParentTurn {
            kind: GoalParentTurnKind::Start,
            objective: "Goal objective".to_owned(),
            working_history: Vec::new(),
        });
        let history = input
            .goal_start_transcript(
                "system prompt",
                vec![ChatMessage::user("earlier user message")],
            )
            .unwrap();

        assert_eq!(history.len(), 3);
        assert_eq!(history[0].role, "system");
        assert_eq!(history[1].content, "earlier user message");
        assert_eq!(history[2].content, "Goal objective");
    }

    #[derive(Default)]
    struct RecoveryRecordingProvider {
        requests: Mutex<Vec<Vec<ChatMessage>>>,
    }

    impl zeroclaw_api::attribution::Attributable for RecoveryRecordingProvider {
        fn role(&self) -> zeroclaw_api::attribution::Role {
            zeroclaw_api::attribution::Role::Provider(
                zeroclaw_api::attribution::ProviderKind::Model(
                    zeroclaw_api::attribution::ModelProviderKind::Custom,
                ),
            )
        }

        fn alias(&self) -> &str {
            "goal-recovery-fixture"
        }
    }

    #[async_trait]
    impl zeroclaw_providers::ModelProvider for RecoveryRecordingProvider {
        async fn chat_with_system(
            &self,
            _system: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: Option<f64>,
        ) -> Result<String> {
            bail!("fixture requires the history boundary")
        }

        async fn chat_with_history(
            &self,
            messages: &[ChatMessage],
            _model: &str,
            _temperature: Option<f64>,
        ) -> Result<String> {
            let mut requests = self.requests.lock().unwrap();
            requests.push(messages.to_vec());
            if requests.len() % 2 == 1 {
                bail!("maximum context length exceeded")
            }
            Ok("candidate after recovery".to_owned())
        }
    }

    #[tokio::test]
    async fn matrix_goal_keeps_one_breadcrumb_across_two_continuation_recoveries() {
        let provider = Arc::new(RecoveryRecordingProvider::default());
        let channel = Arc::new(MatrixGoalErrorRecordingChannel::default());
        let context = super::super::test_channel_ctx_with_backend_channel_and_provider(
            Arc::new(FailingAppendSessionBackend),
            channel,
            provider.clone(),
            0,
        );
        let session_key = GoalSessionKey::matrix("matrix_goal_recovery_fixture").unwrap();
        let operation = GoalOperationScope::new(
            GoalExecutionScope::new("goal-recovery-fixture", session_key.durable_id(), 1).unwrap(),
        );
        let mut lease = MatrixGoalExecutionLease {
            _foreground: Arc::new(tokio::sync::Mutex::new(())).lock_owned().await,
            context,
            session_key,
            message: ChannelMessage::new("event", "user", "room", "work", "matrix", 1),
            pending_delivery: None,
            pending_receipts: None,
            history_has_trim_breadcrumb: false,
        };
        let mut history = vec![
            ChatMessage::system("previous Goal prompt"),
            ChatMessage::user("old ".repeat(20_000)),
            ChatMessage::assistant("old response"),
            ChatMessage::user("recent request"),
            ChatMessage::assistant("recent response"),
            ChatMessage::user("continue the repair"),
        ];
        let breadcrumb = zeroclaw_runtime::agent::history_trim::breadcrumb();
        for turn in 0..2 {
            if turn == 1 {
                assert!(lease.history_has_trim_breadcrumb);
                // Make another whole old turn reclaimable without replacing
                // the lease-owned breadcrumb or recognizing it by content.
                history.splice(
                    2..2,
                    [
                        ChatMessage::user("older ".repeat(20_000)),
                        ChatMessage::assistant("older response"),
                    ],
                );
                history.push(ChatMessage::user("continue with the remaining criterion"));
            }
            let result = lease
                .run_parent_turn(
                    &operation,
                    GoalParentTurn {
                        kind: GoalParentTurnKind::Continue,
                        objective: "bounded repair".to_owned(),
                        working_history: history,
                    },
                )
                .await
                .expect("ordinary context recovery must finish the parent turn");
            history = result.working_history().to_vec();
            assert!(lease.history_has_trim_breadcrumb);
            assert_eq!(history[1].content, breadcrumb.content);
            assert_eq!(
                history
                    .iter()
                    .filter(|row| row.content == breadcrumb.content)
                    .count(),
                1,
                "each recovery must retain exactly one owner-tracked breadcrumb"
            );
            // This fixture probes assembly and recovery, not final delivery.
            lease.pending_delivery.take();
            lease.pending_receipts.take();
        }
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        for request in [&requests[1], &requests[3]] {
            assert_eq!(request[1].content, breadcrumb.content);
            assert_eq!(
                request
                    .iter()
                    .filter(|row| row.content == breadcrumb.content)
                    .count(),
                1
            );
        }
    }
}
