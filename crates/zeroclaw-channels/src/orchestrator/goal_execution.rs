//! Matrix implementation of the transport-neutral Goal session driver.
//!
//! This module owns only live Matrix session mechanics. Durable lifecycle,
//! accounting, and execution fencing remain in `zeroclaw-runtime`.

use std::sync::{Arc, Mutex};

use super::foreground::{foreground_lock, goal_command_lock, goal_supervisor_slot};
use super::{
    ChannelRouteSelection, ChannelRuntimeContext, append_sender_turn_durably,
    build_channel_system_prompt_for_message_with_signal,
    callable_protocol_exposed_for_channel_turn, channel_ingress_context, find_channel_for_message,
    get_or_create_provider, get_route_selection, model_provider_entry_for_ref,
    outbound_content_format_for_channel, read_goal_canonical_history, runtime_defaults_snapshot,
    sanitize_channel_response_for_format_with_leak_detection, system_prompt_for_channel_turn,
    turn_execution::resolved_channel_execution,
};
use anyhow::{Context as _, Result, bail, ensure};
use async_trait::async_trait;
use zeroclaw_api::{
    channel::{Channel, ChannelMessage, SendMessage},
    model_provider::{ChatMessage, ChatRequest, ModelProvider},
};
use zeroclaw_commands::goal::GoalBudgetLimits;
use zeroclaw_config::schema::Config;
use zeroclaw_goal::control::dispose_unowned_session_goal;
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
    agent::execution_tree_budget::ExecutionTreeBudget,
    agent::loop_::{
        LoopKnobs, ProgressEvent, StreamDelta, ToolLoop, run_tool_call_loop, scope_session_key,
        scope_thread_id,
    },
    control_plane::control_plane,
    cost::CostTracker,
    goal_accounting::{
        ModelProviderGoalUsagePricer, RuntimeGoalOperationAccountingFactory,
        RuntimeGoalUsageTotalsReader,
    },
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
    /// resident supervisor. The first `bind` transfers it to submission;
    /// subsequent bindings reacquire the same canonical lock after delivery
    /// releases submission serialization for the launch handoff.
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

/// Represents a Matrix Goal response and any lifecycle work that must wait
/// until the response has been delivered.
pub(super) enum MatrixGoalCommandSubmission {
    /// A grammar-only response that requires no control-plane access.
    Immediate(GoalResponse),
    /// A lifecycle response whose admitted worker is gated on delivery.
    Deferred {
        supervisor: Arc<GoalExecutionSupervisor>,
        supervisor_slot: Arc<tokio::sync::Mutex<Option<Arc<GoalExecutionSupervisor>>>>,
        settings: GoalHostSettings,
        submission: Box<GoalExecutionSubmission>,
    },
}

impl MatrixGoalCommandSubmission {
    pub(super) fn response(&self) -> &GoalResponse {
        match self {
            Self::Immediate(response) => response,
            Self::Deferred { submission, .. } => submission.response(),
        }
    }

    /// Release the admitted worker after Matrix confirms lifecycle delivery.
    pub(super) async fn after_response(&mut self) -> Result<Option<GoalResponse>> {
        match self {
            Self::Immediate(_) => Ok(None),
            Self::Deferred {
                supervisor,
                settings,
                submission,
                ..
            } => {
                supervisor
                    .launch_after_response(settings.clone(), submission)
                    .await
            }
        }
    }

    /// Retire a supervisor only after the corresponding final lifecycle
    /// response has been delivered by Matrix.
    pub(super) async fn final_response_delivered(&self, late_response: Option<&GoalResponse>) {
        let Self::Deferred {
            supervisor,
            supervisor_slot,
            submission,
            ..
        } = self
        else {
            return;
        };
        let initial_is_final = matches!(
            submission.response(),
            GoalResponse::Paused(_)
                | GoalResponse::AlreadyPaused(_)
                | GoalResponse::Cancelled(_)
                | GoalResponse::AlreadyCancelled(_)
                | GoalResponse::Terminal(_)
        );
        let pending_was_finalized =
            matches!(submission.response(), GoalResponse::TransitionPending(_))
                && late_response.is_some();
        if initial_is_final || pending_was_finalized {
            clear_supervisor_slot_if_current(supervisor_slot, supervisor).await;
        }
    }

    /// Keep an unseen Goal from starting when Matrix cannot deliver its
    /// lifecycle acknowledgement.
    pub(super) async fn response_delivery_failed(self) -> Result<()> {
        match self {
            Self::Immediate(_) => Ok(()),
            Self::Deferred {
                supervisor,
                supervisor_slot,
                mut submission,
                ..
            } => {
                supervisor
                    .pause_before_launch(&mut submission, GoalPauseReason::ExternalDependency)
                    .await?;
                clear_supervisor_slot_if_current(&supervisor_slot, &supervisor).await;
                Ok(())
            }
        }
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
    // Help is local grammar and must remain available when Goal execution is
    // disabled or its control plane is unavailable.
    if matches!(&command, zeroclaw_commands::goal::GoalCommand::Help) {
        return Ok(MatrixGoalCommandSubmission::Immediate(GoalResponse::Help));
    }
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
            let tracker = CostTracker::get_or_init_global_required(
                defaults.config.cost.clone(),
                &defaults.config.data_dir,
            )?;
            let runtime = GoalRuntime::new(Arc::clone(&registry)).with_usage_totals_reader(
                Arc::new(RuntimeGoalUsageTotalsReader::new(Arc::clone(&tracker))),
            );
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
    Ok(MatrixGoalCommandSubmission::Deferred {
        supervisor,
        supervisor_slot,
        settings,
        submission: Box::new(submission),
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
        let admitted = self
            .command_lease
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        let guard = match admitted {
            Some(guard) => guard,
            None => {
                goal_command_lock(&self.context.persist_locks, &self.session_key.durable_id())
                    .lock_owned()
                    .await
            }
        };
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
            // Placeholder until take_canonical_history captures the prefix and
            // provenance together. A resume answer may hydrate or trim history
            // after lease construction, so resolving provenance here is too early.
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
    fn excluded_tools<'a>(
        context: &'a ChannelRuntimeContext,
        message: &ChannelMessage,
    ) -> &'a [String] {
        if message.channel == "cli"
            || context.autonomy_level == zeroclaw_config::autonomy::AutonomyLevel::Full
        {
            &[]
        } else {
            context.non_cli_excluded_tools.as_ref()
        }
    }

    fn history(&self) -> Result<(Vec<ChatMessage>, bool)> {
        read_goal_canonical_history(self.context.as_ref(), &self.session_key.durable_id())
    }

    fn base_system_prompt(
        &self,
        provider: &dyn zeroclaw_providers::ModelProvider,
        route: &ChannelRouteSelection,
    ) -> String {
        let excluded_tools = Self::excluded_tools(self.context.as_ref(), &self.message);
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
        Ok(self.history()?.0)
    }

    async fn publish_goal_core_error(&mut self, error: &anyhow::Error) -> Result<()> {
        if zeroclaw_runtime::agent::loop_::is_tool_loop_cancelled(error) {
            return Ok(());
        }
        // These engine-handed errors arise outside provider calls. Preserve
        // the ordinary safe error text, without a provider-transient hint
        // that could hide the actual session or accounting failure.
        let error_text = zeroclaw_runtime::agent::terminal_completion_error_message(error, None)
            .map(|message| format!("⚠️ Error: {message}"))
            .unwrap_or_else(|| {
                format!(
                    "⚠️ Error: {}",
                    zeroclaw_providers::sanitize_api_error(&error.to_string())
                )
            });
        let channel = find_channel_for_message(&self.context.channels_by_name, &self.message)
            .context("Matrix Goal channel is no longer available")?;
        channel
            .send(&SendMessage::reply_to(&self.message, &error_text).suppress_voice())
            .await
            .context("publish Matrix Goal original execution error")
    }

    fn take_canonical_history(&mut self) -> Result<Vec<ChatMessage>> {
        let (history, history_has_trim_breadcrumb) = self.history()?;
        self.history_has_trim_breadcrumb = history_has_trim_breadcrumb;
        Ok(history)
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
        let preparation = async {
            let provider = get_or_create_provider(
                self.context.as_ref(),
                &route.model_provider,
                route.api_key.as_deref(),
                &defaults,
            )
            .await?;
            let input = goal_parent_turn_input(&turn);
            let base_system_prompt = self.base_system_prompt(provider.as_ref(), &route);
            let history = match turn.kind {
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
            Ok::<_, anyhow::Error>((provider, history, Arc::clone(channel)))
        }
        .await;
        let (provider, mut history, prepared_channel) = match preparation {
            Ok(prepared) => prepared,
            Err(error) => {
                if let Some(channel) =
                    find_channel_for_message(&self.context.channels_by_name, &self.message)
                {
                    publish_matrix_goal_core_error(channel.as_ref(), &self.message, &error).await;
                }
                return Err(error);
            }
        };
        let channel = &prepared_channel;
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
        let execution_tree_budget = ExecutionTreeBudget::from_limit(
            self.context
                .prompt_config
                .effective_max_execution_tree_iterations(&self.context.agent_alias),
        );
        let tool_loop = run_tool_call_loop(ToolLoop {
            exec: resolved_channel_execution(
                self.context.as_ref(),
                provider.as_ref(),
                &route,
                notify_observer.as_ref(),
                &approval_manager,
                &loop_knobs,
                Self::excluded_tools(self.context.as_ref(), &self.message),
                defaults.defaults.temperature,
            ),
            history: &mut history,
            history_has_trim_breadcrumb: &mut self.history_has_trim_breadcrumb,
            injected_memory_preamble: &mut injected_memory_preamble,
            channel_name: "matrix",
            channel_reply_target: Some(self.message.reply_target.as_str()),
            cancellation_token: None,
            on_delta: draft_stream.delta_tx.clone(),
            shared_budget: execution_tree_budget,
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
            ingress: channel_ingress_context(&self.message),
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
        let candidate = scope_session_key(Some(self.session_key.durable_id()), async {
            let tool_loop = zeroclaw_runtime::agent::tool_receipts::TOOL_LOOP_RECEIPT_CONTEXT
                .scope(receipt_scope, tool_loop);
            scope_thread_id(Some(self.message.id.clone()), tool_loop).await
        })
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
        if let Some(preamble) = injected_memory_preamble.take() {
            preamble.strip_from_history(&mut history);
        }
        GoalParentTurnResult::from_completed_agent_turn(candidate, history)
    }

    async fn run_verifier(
        &mut self,
        _operation: &GoalOperationScope,
        turn: GoalVerifierTurn,
    ) -> Result<GoalVerifierResult> {
        let result: Result<GoalVerifierResult> = async {
            let defaults = runtime_defaults_snapshot(self.context.as_ref());
            let (provider, route, temperature) =
                matrix_goal_verifier_provider(defaults.config).await?;
            let messages = turn.to_chat_messages();
            let mut settled_attempts = Vec::new();
            let approval_manager = self.context.approval_manager.for_new_turn();
            let response = resolved_channel_execution(
                self.context.as_ref(),
                provider.as_ref(),
                &route,
                self.context.observer.as_ref(),
                &approval_manager,
                &LoopKnobs::default(),
                &[],
                temperature,
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
        let delivery = match self
            .pending_delivery
            .take()
            .context("Matrix Goal parent output has no matching delivery session")
        {
            Ok(delivery) => delivery,
            Err(error) => {
                if let Some(channel) =
                    find_channel_for_message(&self.context.channels_by_name, &self.message)
                {
                    publish_matrix_goal_core_error(channel.as_ref(), &self.message, &error).await;
                }
                return Err(error);
            }
        };
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

async fn matrix_goal_verifier_provider(
    config: Arc<Config>,
) -> Result<(Box<dyn ModelProvider>, ChannelRouteSelection, Option<f64>)> {
    let (provider_ref, entry) =
        model_provider_entry_for_ref(&config, &config.goal.verifier.model_provider)?;
    let api_key = entry.api_key.clone();
    let temperature = entry.temperature;
    // Reliable pins its primary model during construction. The ordinary worker
    // cache cannot serve a verifier override on the same provider alias.
    let (provider, model_provider, model, _resolver) = tokio::task::spawn_blocking(move || {
        zeroclaw_runtime::agent::agent::build_session_model_provider(
            &config,
            &provider_ref,
            config.goal.verifier.model.as_deref(),
        )
    })
    .await
    .context("Goal verifier provider construction task failed")??;
    Ok((
        provider,
        ChannelRouteSelection {
            model_provider,
            model,
            api_key,
        },
        temperature,
    ))
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

    #[tokio::test]
    async fn matrix_goal_restores_cold_canonical_history_without_installing_cache() {
        use zeroclaw_infra::session_backend::SessionBackend as _;
        let tmp = tempfile::TempDir::new().unwrap();
        let store = Arc::new(zeroclaw_infra::session_store::SessionStore::new(tmp.path()).unwrap());
        let session_key = GoalSessionKey::matrix("matrix_goal_cold_fixture").unwrap();
        let key = session_key.durable_id();
        let rows = vec![
            ChatMessage::assistant("Please choose an option."),
            ChatMessage::user("Choose the first option."),
        ];
        for row in &rows {
            store.append(&key, row).unwrap();
        }
        store.set_session_trim_breadcrumb(&key, false).unwrap();
        let context = super::super::test_channel_ctx_with_backend(store.clone());
        let lease = MatrixGoalExecutionLease {
            _foreground: Arc::new(tokio::sync::Mutex::new(())).lock_owned().await,
            context: context.clone(),
            session_key,
            message: ChannelMessage::new("event", "user", "room", "work", "matrix", 1),
            pending_delivery: None,
            pending_receipts: None,
            history_has_trim_breadcrumb: false,
        };
        assert_eq!(
            serde_json::to_value(lease.canonical_history().unwrap()).unwrap(),
            serde_json::to_value(&rows).unwrap()
        );
        assert!(
            context
                .conversation_histories
                .lock()
                .unwrap()
                .peek(&key)
                .is_none()
        );
        assert!(
            context
                .history_crumb_flags
                .lock()
                .unwrap()
                .peek(&key)
                .is_none()
        );
        assert_eq!(
            serde_json::to_value(store.try_load(&key).unwrap()).unwrap(),
            serde_json::to_value(rows).unwrap()
        );
        assert_eq!(
            store.get_session_trim_breadcrumb(&key).unwrap(),
            Some(false)
        );
    }

    #[tokio::test]
    async fn matrix_goal_captures_provenance_after_answered_resume() {
        use zeroclaw_infra::session_backend::SessionBackend as _;
        for (cap, initial_owned, duplicate, expected_owned) in [
            (3, true, false, true),
            (2, true, false, false),
            (3, false, false, false),
            (3, true, true, false),
            (4, true, true, true),
        ] {
            let tmp = tempfile::TempDir::new().unwrap();
            let store =
                Arc::new(zeroclaw_infra::session_store::SessionStore::new(tmp.path()).unwrap());
            let session_key = GoalSessionKey::matrix("matrix_goal_capture_fixture").unwrap();
            let key = session_key.durable_id();
            let mut rows = vec![
                ChatMessage::user(
                    zeroclaw_runtime::agent::history::HISTORY_TRIM_BREADCRUMB_CANONICAL,
                ),
                ChatMessage::assistant("Please choose an option."),
            ];
            if duplicate {
                rows.insert(
                    1,
                    ChatMessage::user(
                        zeroclaw_runtime::agent::history::HISTORY_TRIM_BREADCRUMB_CANONICAL,
                    ),
                );
            }
            for row in &rows {
                store.append(&key, row).unwrap();
            }
            store
                .set_session_trim_breadcrumb(&key, initial_owned)
                .unwrap();
            let mut context = super::super::test_channel_ctx_with_backend(store.clone());
            let mut config = zeroclaw_config::schema::AliasedAgentConfig::default();
            config.resolved.max_history_messages = cap;
            Arc::get_mut(&mut context).unwrap().agent_cfg = Arc::new(config);
            let mut lease = MatrixGoalExecutionLease {
                _foreground: Arc::new(tokio::sync::Mutex::new(())).lock_owned().await,
                context,
                session_key,
                message: ChannelMessage::new("event", "user", "room", "work", "matrix", 1),
                pending_delivery: None,
                pending_receipts: None,
                history_has_trim_breadcrumb: false,
            };
            let response = "Choose the first option.\nKeep this second line.";
            lease.append_resume_response(response.into()).await.unwrap();
            rows.push(ChatMessage::user(response));
            let captured = lease.take_canonical_history().unwrap();
            assert_eq!(
                serde_json::to_value(captured).unwrap(),
                serde_json::to_value(&rows[rows.len().saturating_sub(cap)..]).unwrap()
            );
            assert_eq!(
                lease.history_has_trim_breadcrumb, expected_owned,
                "provenance must describe the captured post-append prefix"
            );
            assert_eq!(
                serde_json::to_value(store.try_load(&key).unwrap()).unwrap(),
                serde_json::to_value(&rows).unwrap()
            );
            assert_eq!(
                store.get_session_trim_breadcrumb(&key).unwrap(),
                Some(initial_owned),
                "cache eviction must not rewrite durable provenance"
            );

            // A later ordinary append on the same key may evict the owned
            // first row. The next capture must replace the previous flag,
            // including when an identical user row is now at the front.
            lease
                .append_resume_response("A later answer.".into())
                .await
                .unwrap();
            rows.push(ChatMessage::user("A later answer."));
            let captured = lease.take_canonical_history().unwrap();
            assert_eq!(
                serde_json::to_value(captured).unwrap(),
                serde_json::to_value(&rows[rows.len().saturating_sub(cap)..]).unwrap()
            );
            assert!(!lease.history_has_trim_breadcrumb);
            assert_eq!(
                serde_json::to_value(store.try_load(&key).unwrap()).unwrap(),
                serde_json::to_value(rows).unwrap()
            );
        }
    }

    async fn assert_matrix_goal_verifier_wire_model(
        alias_model: Option<&str>,
        verifier_model: Option<&str>,
        expected_model: &str,
    ) {
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{body_partial_json, method},
        };
        use zeroclaw_config::schema::WireApi;
        use zeroclaw_providers::dispatch::ProviderDispatch;

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(body_partial_json(serde_json::json!({"model": expected_model})))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "verdict"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
            })))
            .expect(1)
            .mount(&server)
            .await;
        let mut config = Config::default();
        config.reliability.provider_retries = 0;
        config.goal.verifier.model_provider = "openai.verifier-test".into();
        config.goal.verifier.model = verifier_model.map(str::to_owned);
        let slot = config
            .providers
            .models
            .ensure("openai", "verifier-test")
            .unwrap();
        slot.api_key = Some("test-key".into());
        slot.uri = Some(server.uri());
        slot.kind = Some("openai-compatible".into());
        slot.wire_api = Some(WireApi::ChatCompletions);
        slot.model = alias_model.map(str::to_owned);
        slot.temperature = Some(0.2);
        slot.live_pricing = false;
        let (provider, route, temperature) = matrix_goal_verifier_provider(Arc::new(config))
            .await
            .unwrap();
        assert_eq!(route.model_provider, "openai.verifier-test");
        assert_eq!(route.model, expected_model);
        assert_eq!(temperature, Some(0.2));
        let messages = [ChatMessage::user("Verify the candidate.")];
        let response = ProviderDispatch::from_ref(provider.as_ref())
            .chat(
                ChatRequest {
                    messages: &messages,
                    tools: None,
                    thinking: None,
                },
                &route.model,
                temperature,
            )
            .await
            .unwrap();
        assert_eq!(response.text.as_deref(), Some("verdict"));
        server.verify().await;
    }

    #[tokio::test]
    async fn matrix_goal_verifier_constructs_the_explicit_model_override() {
        assert_matrix_goal_verifier_wire_model(
            Some("goal-worker"),
            Some("goal-verifier"),
            "goal-verifier",
        )
        .await;
    }

    #[tokio::test]
    async fn matrix_goal_verifier_uses_the_alias_default_without_override() {
        assert_matrix_goal_verifier_wire_model(Some("goal-worker"), None, "goal-worker").await;
    }

    #[tokio::test]
    async fn matrix_goal_verifier_override_does_not_require_an_alias_default() {
        assert_matrix_goal_verifier_wire_model(None, Some("goal-verifier"), "goal-verifier").await;
    }

    #[tokio::test]
    async fn matrix_goal_verifier_rejects_a_missing_model_without_override() {
        let mut config = Config::default();
        config.goal.verifier.model_provider = "openai.verifier-test".into();
        config.goal.verifier.model = None;
        config
            .providers
            .models
            .ensure("openai", "verifier-test")
            .unwrap()
            .model = None;
        let error = match matrix_goal_verifier_provider(Arc::new(config)).await {
            Ok(_) => panic!("a missing verifier model must fail construction"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("no `model` configured"),
            "{error:#}"
        );
    }

    #[tokio::test]
    async fn matrix_goal_rebinds_the_canonical_command_lock_after_delivery() {
        let context =
            super::super::test_channel_ctx_with_backend(Arc::new(FailingAppendSessionBackend));
        let key = "matrix_goal_binding_test";
        let lock = goal_command_lock(&context.persist_locks, key);
        let message = ChannelMessage::new("event", "sender", "room", "/goal start", "matrix", 1);
        let driver = MatrixGoalSessionDriver::new(
            context,
            key.into(),
            "sender".into(),
            "matrix:test:room".into(),
            message,
            lock.clone().lock_owned().await,
        )
        .unwrap();
        let ingress = GoalIngressContext::trusted(
            driver.session_key.clone(),
            "test",
            "matrix:test:room",
            GoalIngressPrincipal::Matrix {
                raw_mxid: "sender".into(),
            },
        )
        .unwrap();
        let first = driver.bind(&ingress).await.unwrap();
        assert!(lock.try_lock().is_err());
        drop(first);
        let intervening = lock.clone().lock_owned().await;
        let mut next = Box::pin(driver.bind(&ingress));
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), &mut next)
                .await
                .is_err(),
            "the launch binding must wait for competing command serialization"
        );
        drop(intervening);
        let rebound = tokio::time::timeout(std::time::Duration::from_secs(1), next)
            .await
            .unwrap()
            .unwrap();
        assert!(lock.try_lock().is_err());
        drop(rebound);
        assert!(lock.try_lock().is_ok());
        let stale = GoalIngressContext::trusted(
            driver.session_key.clone(),
            "test",
            "matrix:test:room",
            GoalIngressPrincipal::Matrix {
                raw_mxid: "other-sender".into(),
            },
        )
        .unwrap();
        assert!(driver.bind(&stale).await.is_err());
    }

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
            let key = matrix_goal_failure_key(reason);
            assert!(key.starts_with("goal-mode-failure-"));
            assert_ne!(
                zeroclaw_runtime::i18n::get_required_cli_string(key),
                format!("{{{key}}}"),
                "missing Fluent message for {key}"
            );
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
        fail_sends: bool,
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
            if self.fail_sends {
                bail!("synthetic Matrix delivery failure");
            }
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

    async fn matrix_original_error_lease(
        channel: Arc<MatrixGoalErrorRecordingChannel>,
    ) -> MatrixGoalExecutionLease {
        let mut context =
            super::super::test_channel_ctx_with_backend(Arc::new(FailingAppendSessionBackend));
        Arc::get_mut(&mut context).unwrap().channels_by_name = Arc::new(
            std::collections::HashMap::from([("matrix".into(), channel as Arc<dyn Channel>)]),
        );
        MatrixGoalExecutionLease {
            _foreground: Arc::new(tokio::sync::Mutex::new(())).lock_owned().await,
            context,
            session_key: GoalSessionKey::matrix("matrix_goal_error_fixture").unwrap(),
            message: ChannelMessage::new(
                "fixture-event",
                "fixture-user",
                "fixture-room",
                "work",
                "matrix",
                1,
            ),
            pending_delivery: None,
            pending_receipts: None,
            history_has_trim_breadcrumb: false,
        }
    }

    #[tokio::test]
    async fn matrix_goal_original_unavailable_error_is_not_a_provider_hint() {
        let channel = Arc::new(MatrixGoalErrorRecordingChannel::default());
        let mut lease = matrix_original_error_lease(channel.clone()).await;
        let error = anyhow::Error::msg("Goal canonical transcript is unavailable");
        lease.publish_goal_core_error(&error).await.unwrap();
        assert_eq!(
            channel.messages.lock().await.as_slice(),
            ["⚠️ Error: Goal canonical transcript is unavailable",]
        );
    }

    #[tokio::test]
    async fn matrix_goal_original_error_uses_existing_secret_scrub() {
        let channel = Arc::new(MatrixGoalErrorRecordingChannel::default());
        let mut lease = matrix_original_error_lease(channel.clone()).await;
        let secret = "sk-fixturecredential012345678901234567890123456789";
        let error = anyhow::Error::msg(format!("Goal persistence failed with credential {secret}"));
        let safe = zeroclaw_providers::sanitize_api_error(&error.to_string());
        assert!(
            !safe.contains(secret),
            "fixture must exercise an actual scrub rule"
        );
        lease.publish_goal_core_error(&error).await.unwrap();
        assert_eq!(
            channel.messages.lock().await.as_slice(),
            [format!("⚠️ Error: {safe}")]
        );
    }

    #[tokio::test]
    async fn matrix_goal_original_tool_loop_cancellation_is_not_sent() {
        let channel = Arc::new(MatrixGoalErrorRecordingChannel::default());
        let mut lease = matrix_original_error_lease(channel.clone()).await;
        let error = anyhow::Error::new(zeroclaw_runtime::agent::loop_::ToolLoopCancelled);
        lease.publish_goal_core_error(&error).await.unwrap();
        assert!(channel.messages.lock().await.is_empty());
    }

    #[tokio::test]
    async fn matrix_goal_original_typed_terminal_cause_precedes_outer_hint() {
        let channel = Arc::new(MatrixGoalErrorRecordingChannel::default());
        let mut lease = matrix_original_error_lease(channel.clone()).await;
        let error =
            anyhow::Error::new(zeroclaw_api::model_provider::SemanticEmptyTerminalCompletion)
                .context("earlier attempt returned HTTP 503 unavailable");
        let expected =
            zeroclaw_runtime::agent::terminal_completion_error_message(&error, None).unwrap();
        lease.publish_goal_core_error(&error).await.unwrap();
        assert_eq!(
            channel.messages.lock().await.as_slice(),
            [format!("⚠️ Error: {expected}")]
        );
    }

    #[tokio::test]
    async fn matrix_goal_original_error_reports_delivery_failure() {
        let channel = Arc::new(MatrixGoalErrorRecordingChannel {
            fail_sends: true,
            ..Default::default()
        });
        let mut lease = matrix_original_error_lease(channel.clone()).await;
        let error = anyhow::Error::msg("Goal canonical transcript is unavailable");
        let failure = lease.publish_goal_core_error(&error).await.unwrap_err();
        assert!(format!("{failure:#}").contains("synthetic Matrix delivery failure"));
        assert_eq!(
            channel.messages.lock().await.as_slice(),
            ["⚠️ Error: Goal canonical transcript is unavailable",]
        );
    }

    #[tokio::test]
    async fn matrix_goal_original_missing_candidate_delivery_is_rendered() {
        let channel = Arc::new(MatrixGoalErrorRecordingChannel::default());
        let mut lease = matrix_original_error_lease(channel.clone()).await;
        let error = lease
            .append_parent_candidate("candidate".into())
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Matrix Goal parent output has no matching delivery session"
        );
        let expected = matrix_goal_core_error_text(&error).unwrap();
        assert_eq!(channel.messages.lock().await.as_slice(), [expected]);
    }

    #[tokio::test]
    async fn matrix_goal_original_parent_assembly_error_is_rendered() {
        let channel = Arc::new(MatrixGoalErrorRecordingChannel::default());
        let mut lease = matrix_original_error_lease(channel.clone()).await;
        let operation = GoalOperationScope::new(
            GoalExecutionScope::new("fixture-task", lease.session_key.durable_id(), 1).unwrap(),
        );
        let error = lease
            .run_parent_turn(
                &operation,
                GoalParentTurn {
                    kind: GoalParentTurnKind::Start,
                    objective: "finish the admitted task".into(),
                    working_history: vec![ChatMessage::system("invalid canonical prefix")],
                },
            )
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Goal turn assembly: canonical prefix contains a system row"
        );
        let expected = matrix_goal_core_error_text(&error).unwrap();
        assert_eq!(channel.messages.lock().await.as_slice(), [expected]);
    }

    #[tokio::test]
    async fn matrix_goal_original_provider_construction_error_is_rendered() {
        let channel = Arc::new(MatrixGoalErrorRecordingChannel::default());
        let mut lease = matrix_original_error_lease(channel.clone()).await;
        lease.context.route_overrides.lock().unwrap().insert(
            lease.session_key.durable_id(),
            ChannelRouteSelection {
                model_provider: "unknown-fixture-provider.missing".into(),
                model: "fixture-model".into(),
                api_key: None,
            },
        );
        let operation = GoalOperationScope::new(
            GoalExecutionScope::new("fixture-task", lease.session_key.durable_id(), 1).unwrap(),
        );
        let error = lease
            .run_parent_turn(
                &operation,
                GoalParentTurn {
                    kind: GoalParentTurnKind::Start,
                    objective: "finish the admitted task".into(),
                    working_history: Vec::new(),
                },
            )
            .await
            .unwrap_err();
        assert!(format!("{error:#}").contains("unknown-fixture-provider"));
        let expected = matrix_goal_core_error_text(&error).unwrap();
        assert_eq!(channel.messages.lock().await.as_slice(), [expected]);
    }

    #[tokio::test]
    async fn matrix_goal_original_error_fix_does_not_change_ordinary_503_hint() {
        let channel = MatrixGoalErrorRecordingChannel::default();
        let message = ChannelMessage::new(
            "fixture-event",
            "fixture-user",
            "fixture-room",
            "work",
            "matrix",
            1,
        );
        let error = anyhow::Error::msg("HTTP 503 unavailable");
        let hint = zeroclaw_providers::reliable::transient_error_hint(&error).unwrap();
        publish_matrix_goal_core_error(&channel, &message, &error).await;
        assert_eq!(channel.messages.lock().await.as_slice(), [hint]);
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

    /// Captures the actual loop requests, optionally forcing one ordinary trim.
    struct GoalMemoryRecordingProvider {
        requests: Mutex<Vec<Vec<ChatMessage>>>,
        recover_first_request: bool,
    }

    impl zeroclaw_api::attribution::Attributable for GoalMemoryRecordingProvider {
        fn role(&self) -> zeroclaw_api::attribution::Role {
            zeroclaw_api::attribution::Role::Provider(
                zeroclaw_api::attribution::ProviderKind::Model(
                    zeroclaw_api::attribution::ModelProviderKind::Custom,
                ),
            )
        }

        fn alias(&self) -> &str {
            "goal-memory-fixture"
        }
    }

    #[async_trait]
    impl zeroclaw_providers::ModelProvider for GoalMemoryRecordingProvider {
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
            if self.recover_first_request && requests.len() == 1 {
                bail!("maximum context length exceeded")
            }
            Ok("candidate with recalled context".to_owned())
        }
    }

    async fn assert_matrix_goal_recall_is_turn_local(recover_first_request: bool) {
        let provider = Arc::new(GoalMemoryRecordingProvider {
            requests: Mutex::new(Vec::new()),
            recover_first_request,
        });
        let mut context = super::super::test_channel_ctx_with_backend_channel_and_provider(
            Arc::new(FailingAppendSessionBackend),
            Arc::new(MatrixGoalErrorRecordingChannel::default()),
            provider.clone(),
            0,
        );
        Arc::get_mut(&mut context).unwrap().memory = Arc::new(super::super::tests::RecallMemory);
        let session_key = GoalSessionKey::matrix("matrix_goal_memory_fixture").unwrap();
        let operation = GoalOperationScope::new(
            GoalExecutionScope::new("goal-memory-fixture", session_key.durable_id(), 1).unwrap(),
        );
        let mut lease = MatrixGoalExecutionLease {
            _foreground: Arc::new(tokio::sync::Mutex::new(())).lock_owned().await,
            context,
            session_key,
            message: ChannelMessage::new(
                "event",
                "user",
                "room",
                "/goal start bounded repair",
                "matrix",
                1,
            ),
            pending_delivery: None,
            pending_receipts: None,
            history_has_trim_breadcrumb: false,
        };
        let prefix = if recover_first_request {
            vec![
                ChatMessage::user("older context ".repeat(20_000)),
                ChatMessage::assistant("older answer"),
                ChatMessage::user("recent request"),
            ]
        } else {
            Vec::new()
        };
        let first = lease
            .run_parent_turn(
                &operation,
                GoalParentTurn {
                    kind: GoalParentTurnKind::Start,
                    objective: "bounded repair".to_owned(),
                    working_history: prefix,
                },
            )
            .await
            .unwrap();
        let first_history = first.working_history().to_vec();
        lease.pending_delivery.take();
        lease.pending_receipts.take();
        let mut carried = first_history.clone();
        carried.push(zeroclaw_goal::control::goal_verifier_feedback_message(
            "Finish the remaining criterion.",
        ));
        let second = lease
            .run_parent_turn(
                &operation,
                GoalParentTurn {
                    kind: GoalParentTurnKind::Continue,
                    objective: "bounded repair".to_owned(),
                    working_history: carried,
                },
            )
            .await
            .unwrap();
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), if recover_first_request { 3 } else { 2 });
        for request in requests.iter() {
            let blocks: usize = request
                .iter()
                .map(|row| {
                    row.content
                        .matches(zeroclaw_memory::MEMORY_CONTEXT_OPEN)
                        .count()
                })
                .sum();
            assert_eq!(
                blocks, 1,
                "each parent request must contain only its own recall"
            );
            assert!(
                request
                    .iter()
                    .rev()
                    .find(|row| row.role == "user")
                    .unwrap()
                    .content
                    .starts_with(zeroclaw_memory::MEMORY_CONTEXT_OPEN)
            );
        }
        for history in [&first_history[..], second.working_history()] {
            assert!(
                history
                    .iter()
                    .all(|row| !row.content.contains(zeroclaw_memory::MEMORY_CONTEXT_OPEN)),
                "the carried transcript must not retain provider-only recall"
            );
        }
        if recover_first_request {
            assert!(lease.history_has_trim_breadcrumb);
            assert_eq!(
                first_history[1].content,
                zeroclaw_runtime::agent::history_trim::breadcrumb().content
            );
        }
    }

    #[tokio::test]
    async fn matrix_goal_recall_does_not_accumulate_across_continuations() {
        assert_matrix_goal_recall_is_turn_local(false).await;
    }

    #[tokio::test]
    async fn matrix_goal_recall_strip_follows_recovery_trim_remapping() {
        assert_matrix_goal_recall_is_turn_local(true).await;
    }

    #[tokio::test]
    async fn matrix_goal_recall_preserves_a_user_authored_memory_marker() {
        let provider = Arc::new(GoalMemoryRecordingProvider {
            requests: Mutex::new(Vec::new()),
            recover_first_request: false,
        });
        let mut context = super::super::test_channel_ctx_with_backend_channel_and_provider(
            Arc::new(FailingAppendSessionBackend),
            Arc::new(MatrixGoalErrorRecordingChannel::default()),
            provider.clone(),
            0,
        );
        Arc::get_mut(&mut context).unwrap().memory = Arc::new(super::super::tests::RecallMemory);
        let session_key = GoalSessionKey::matrix("matrix_goal_marker_fixture").unwrap();
        let operation = GoalOperationScope::new(
            GoalExecutionScope::new("goal-marker-fixture", session_key.durable_id(), 1).unwrap(),
        );
        let objective = format!(
            "{}\nUser-authored context, not recalled memory.\n{}\nKeep this text intact.",
            zeroclaw_memory::MEMORY_CONTEXT_OPEN,
            zeroclaw_memory::MEMORY_CONTEXT_CLOSE,
        );
        let mut lease = MatrixGoalExecutionLease {
            _foreground: Arc::new(tokio::sync::Mutex::new(())).lock_owned().await,
            context,
            session_key,
            message: ChannelMessage::new(
                "event",
                "user",
                "room",
                "/goal start fixture",
                "matrix",
                1,
            ),
            pending_delivery: None,
            pending_receipts: None,
            history_has_trim_breadcrumb: false,
        };
        let result = lease
            .run_parent_turn(
                &operation,
                GoalParentTurn {
                    kind: GoalParentTurnKind::Start,
                    objective: objective.clone(),
                    working_history: Vec::new(),
                },
            )
            .await
            .unwrap();
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let requested_user = requests[0]
            .iter()
            .rev()
            .find(|row| row.role == "user")
            .unwrap();
        assert_eq!(requested_user.content, objective);
        let carried_user = result
            .working_history()
            .iter()
            .rev()
            .find(|row| row.role == "user")
            .unwrap();
        assert_eq!(carried_user.content, objective);
    }

    /// The same tool script drives both the ordinary ingress and Goal host.
    struct GoalPolicyProvider {
        requests: Mutex<Vec<Vec<ChatMessage>>>,
        tool_rounds: usize,
        calls_per_round: usize,
    }

    impl zeroclaw_api::attribution::Attributable for GoalPolicyProvider {
        fn role(&self) -> zeroclaw_api::attribution::Role {
            zeroclaw_api::attribution::Role::Provider(
                zeroclaw_api::attribution::ProviderKind::Model(
                    zeroclaw_api::attribution::ModelProviderKind::Custom,
                ),
            )
        }

        fn alias(&self) -> &str {
            "goal-policy-fixture"
        }
    }

    #[async_trait]
    impl zeroclaw_providers::ModelProvider for GoalPolicyProvider {
        fn supports_native_tools(&self) -> bool {
            true
        }

        async fn chat_with_system(
            &self,
            _system: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: Option<f64>,
        ) -> Result<String> {
            bail!("fixture requires the tool request boundary")
        }

        async fn chat(
            &self,
            request: zeroclaw_providers::ChatRequest<'_>,
            _model: &str,
            _temperature: Option<f64>,
        ) -> Result<zeroclaw_providers::ChatResponse> {
            let mut requests = self.requests.lock().unwrap();
            requests.push(request.messages.to_vec());
            let round = requests.len();
            Ok(zeroclaw_providers::ChatResponse {
                text: (round > self.tool_rounds).then(|| "policy candidate".to_owned()),
                tool_calls: if round <= self.tool_rounds {
                    (0..self.calls_per_round)
                        .map(|slot| zeroclaw_providers::ToolCall {
                            id: format!("policy-{round}-{slot}"),
                            name: "echo".to_owned(),
                            arguments: "{}".to_owned(),
                            extra_content: None,
                        })
                        .collect()
                } else {
                    Vec::new()
                },
                usage: Some(zeroclaw_api::model_provider::TokenUsage {
                    input_tokens: Some(10),
                    output_tokens: Some(2),
                    ..Default::default()
                }),
                reasoning_content: None,
            })
        }
    }

    struct GoalPolicyCountingTool(Arc<std::sync::atomic::AtomicUsize>);

    impl zeroclaw_api::attribution::Attributable for GoalPolicyCountingTool {
        fn role(&self) -> zeroclaw_api::attribution::Role {
            zeroclaw_api::attribution::Role::Tool(zeroclaw_api::attribution::ToolKind::Plugin)
        }

        fn alias(&self) -> &str {
            "goal-policy-fixture"
        }
    }

    #[async_trait]
    impl zeroclaw_runtime::tools::Tool for GoalPolicyCountingTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Return the stable fixture result."
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type":"object"})
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
        ) -> Result<zeroclaw_runtime::tools::ToolResult> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(zeroclaw_runtime::tools::ToolResult {
                success: true,
                output: "stable tool result".into(),
                error: None,
            })
        }
    }

    fn matrix_policy_context(
        backend: Arc<dyn zeroclaw_infra::session_backend::SessionBackend>,
        channel: Arc<MatrixGoalErrorRecordingChannel>,
        provider: Arc<GoalPolicyProvider>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Arc<ChannelRuntimeContext> {
        let mut context = super::super::test_channel_ctx_with_backend_channel_and_provider(
            backend, channel, provider, 0,
        );
        let ctx = Arc::get_mut(&mut context).unwrap();
        ctx.max_tool_iterations = 10;
        ctx.tools_registry = Arc::new(
            zeroclaw_runtime::tools::scoped::ScopedToolRegistry::from_raw_for_test(vec![Box::new(
                GoalPolicyCountingTool(calls),
            )]),
        );
        ctx.pacing.loop_detection_max_repeats = 2;
        // Exercise executable calls, not the ordinary default denial of an
        // unlisted tool. Both hosts resolve the same narrow fixture profile.
        let risk = zeroclaw_config::schema::RiskProfileConfig {
            auto_approve: vec!["echo".to_owned()],
            ..Default::default()
        };
        let mut config = ctx.prompt_config.as_ref().clone();
        config
            .risk_profiles
            .insert("policy-fixture".to_owned(), risk.clone());
        config.agents.get_mut("test").unwrap().risk_profile = "policy-fixture".into();
        ctx.agent_cfg = Arc::new(config.agents["test"].clone());
        ctx.prompt_config = Arc::new(config.clone());
        *ctx.live_config.write() = config;
        ctx.approval_manager =
            Arc::new(zeroclaw_runtime::approval::ApprovalManager::for_non_interactive(&risk));
        context
    }

    async fn run_matrix_policy_goal(
        context: Arc<ChannelRuntimeContext>,
    ) -> (Result<GoalParentTurnResult>, MatrixGoalExecutionLease) {
        let session_key = GoalSessionKey::matrix("matrix_goal_policy_fixture").unwrap();
        let operation = GoalOperationScope::new(
            GoalExecutionScope::new("goal-policy-fixture", session_key.durable_id(), 1).unwrap(),
        );
        let mut lease = MatrixGoalExecutionLease {
            _foreground: Arc::new(tokio::sync::Mutex::new(())).lock_owned().await,
            context,
            session_key,
            message: ChannelMessage::new(
                "goal-event",
                "user",
                "room",
                "/goal start bounded repair",
                "matrix",
                1,
            ),
            pending_delivery: None,
            pending_receipts: None,
            history_has_trim_breadcrumb: false,
        };
        let result = lease
            .run_parent_turn(
                &operation,
                GoalParentTurn {
                    kind: GoalParentTurnKind::Start,
                    objective: "bounded repair".to_owned(),
                    working_history: Vec::new(),
                },
            )
            .await;
        (result, lease)
    }

    #[tokio::test]
    async fn matrix_goal_policy_keeps_ordinary_same_batch_deduplication() {
        for goal in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            let backend = Arc::new(
                zeroclaw_infra::session_sqlite::SqliteSessionBackend::new(tmp.path()).unwrap(),
            );
            let channel = Arc::new(MatrixGoalErrorRecordingChannel::default());
            let provider = Arc::new(GoalPolicyProvider {
                requests: Mutex::new(Vec::new()),
                tool_rounds: 1,
                calls_per_round: 2,
            });
            let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let context =
                matrix_policy_context(backend, channel.clone(), provider.clone(), calls.clone());
            if goal {
                let (result, _) = run_matrix_policy_goal(context).await;
                assert_eq!(result.unwrap().candidate(), "policy candidate");
            } else {
                super::super::process_channel_message(
                    context,
                    ChannelMessage::new(
                        "ordinary-event",
                        "user",
                        "room",
                        "bounded repair",
                        "matrix",
                        1,
                    ),
                    tokio_util::sync::CancellationToken::new(),
                )
                .await;
                assert!(
                    channel
                        .messages
                        .lock()
                        .await
                        .iter()
                        .any(|text| text == "policy candidate")
                );
            }
            assert_eq!(
                calls.load(std::sync::atomic::Ordering::SeqCst),
                1,
                "same-batch duplicates must execute once, goal={goal}"
            );
            assert_eq!(provider.requests.lock().unwrap().len(), 2);
        }
    }

    #[derive(Default)]
    struct GoalToolAvailabilityProvider {
        requests: Mutex<Vec<(Vec<ChatMessage>, Vec<String>)>>,
        tool_rounds: usize,
    }

    impl zeroclaw_api::attribution::Attributable for GoalToolAvailabilityProvider {
        fn role(&self) -> zeroclaw_api::attribution::Role {
            zeroclaw_api::attribution::Role::Provider(
                zeroclaw_api::attribution::ProviderKind::Model(
                    zeroclaw_api::attribution::ModelProviderKind::Custom,
                ),
            )
        }

        fn alias(&self) -> &str {
            "goal-availability-fixture"
        }
    }

    #[async_trait]
    impl ModelProvider for GoalToolAvailabilityProvider {
        fn supports_native_tools(&self) -> bool {
            true
        }

        async fn chat_with_system(
            &self,
            _: Option<&str>,
            _: &str,
            _: &str,
            _: Option<f64>,
        ) -> Result<String> {
            bail!("fixture requires structured specs")
        }

        async fn chat(
            &self,
            request: ChatRequest<'_>,
            _: &str,
            _: Option<f64>,
        ) -> Result<zeroclaw_providers::ChatResponse> {
            let specs = request
                .tools
                .unwrap_or_default()
                .iter()
                .map(|tool| tool.name.clone())
                .collect::<Vec<_>>();
            let mut requests = self.requests.lock().unwrap();
            requests.push((request.messages.to_vec(), specs));
            let round = request
                .messages
                .iter()
                .rev()
                .take_while(|row| row.role != "user")
                .filter(|row| row.role == "tool")
                .count()
                + 1;
            // Honor effective specs, including the root's reserved final
            // completion. An unavailable tool must not become a model call.
            let call =
                round <= self.tool_rounds && request.tools.is_some_and(|tools| !tools.is_empty());
            Ok(zeroclaw_providers::ChatResponse {
                text: (!call).then(|| "availability candidate".into()),
                tool_calls: if call {
                    vec![zeroclaw_providers::ToolCall {
                        id: format!("availability-{round}"),
                        name: "echo".into(),
                        arguments: format!("{{\"round\":{round}}}"),
                        extra_content: None,
                    }]
                } else {
                    Vec::new()
                },
                usage: Some(zeroclaw_api::model_provider::TokenUsage {
                    input_tokens: Some(10),
                    output_tokens: Some(2),
                    ..Default::default()
                }),
                reasoning_content: None,
            })
        }
    }

    async fn observe_matrix_goal_tool_policy(
        goal: bool,
        autonomy: zeroclaw_config::autonomy::AutonomyLevel,
        excluded: bool,
        limit: Option<usize>,
        rounds: usize,
    ) -> (Vec<(Vec<ChatMessage>, Vec<String>)>, usize) {
        let tmp = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            zeroclaw_infra::session_sqlite::SqliteSessionBackend::new(tmp.path()).unwrap(),
        );
        let channel = Arc::new(MatrixGoalErrorRecordingChannel::default());
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let provider = Arc::new(GoalToolAvailabilityProvider {
            tool_rounds: rounds,
            ..Default::default()
        });
        let mut context = matrix_policy_context(
            backend,
            channel,
            Arc::new(GoalPolicyProvider {
                requests: Mutex::new(Vec::new()),
                tool_rounds: 0,
                calls_per_round: 0,
            }),
            calls.clone(),
        );
        let ctx = Arc::get_mut(&mut context).unwrap();
        ctx.model_provider = provider.clone();
        ctx.autonomy_level = autonomy;
        ctx.non_cli_excluded_tools = Arc::new(if excluded {
            vec!["echo".into()]
        } else {
            Vec::new()
        });
        let available = !excluded || autonomy == zeroclaw_config::autonomy::AutonomyLevel::Full;
        let tools: &[(&str, &str)] = if available {
            &[("echo", "Return the stable fixture result.")]
        } else {
            &[]
        };
        ctx.workspace_dir = Arc::new(tmp.path().to_owned());
        ctx.system_prompt = Arc::new(
            zeroclaw_runtime::agent::system_prompt::build_system_prompt_with_mode(
                tmp.path(),
                "test",
                tools,
                &[],
                None,
                None,
                available,
                Default::default(),
                autonomy,
            ),
        );
        let mut config = ctx.prompt_config.as_ref().clone();
        config
            .runtime_profiles
            .get_mut("test")
            .unwrap()
            .max_execution_tree_iterations = limit;
        ctx.prompt_config = Arc::new(config.clone());
        *ctx.live_config.write() = config;
        if goal {
            let (result, _) = run_matrix_policy_goal(context).await;
            assert!(
                result
                    .unwrap()
                    .candidate()
                    .starts_with("availability candidate")
            );
        } else {
            super::super::process_channel_message(
                context,
                ChannelMessage::new(
                    "ordinary-availability",
                    "user",
                    "room",
                    "bounded repair",
                    "matrix",
                    1,
                ),
                tokio_util::sync::CancellationToken::new(),
            )
            .await;
        }
        let requests = provider.requests.lock().unwrap().clone();
        (requests, calls.load(std::sync::atomic::Ordering::SeqCst))
    }

    async fn assert_matrix_goal_availability_parity(
        autonomy: zeroclaw_config::autonomy::AutonomyLevel,
        available: bool,
    ) {
        let ordinary = observe_matrix_goal_tool_policy(false, autonomy, true, None, 1).await;
        let goal = observe_matrix_goal_tool_policy(true, autonomy, true, None, 1).await;
        let expected_specs = if available {
            vec!["echo".to_owned()]
        } else {
            Vec::new()
        };
        assert_eq!(ordinary.0[0].1, expected_specs);
        assert_eq!(ordinary.1, usize::from(available));
        assert_eq!(goal.0[0].1, ordinary.0[0].1);
        assert_eq!(goal.1, ordinary.1);
        for (requests, _) in [&ordinary, &goal] {
            assert_eq!(
                requests[0].0[0]
                    .content
                    .contains(zeroclaw_runtime::agent::system_prompt::NATIVE_TOOLS_TASK_FRAMING),
                available,
                "prompt tool section must agree with effective specs"
            );
        }
        assert_eq!(goal.0.len(), ordinary.0.len());
        if available {
            let goal_result = goal.0[1].0.last().unwrap();
            let ordinary_result = ordinary.0[1].0.last().unwrap();
            assert_eq!(goal_result.role, ordinary_result.role);
            assert_eq!(goal_result.content, ordinary_result.content);
        }
    }

    #[tokio::test]
    async fn matrix_goal_policy_full_autonomy_matches_ordinary_tool_availability() {
        assert_matrix_goal_availability_parity(
            zeroclaw_config::autonomy::AutonomyLevel::Full,
            true,
        )
        .await;
    }

    #[tokio::test]
    async fn matrix_goal_policy_supervised_matches_ordinary_tool_availability() {
        assert_matrix_goal_availability_parity(
            zeroclaw_config::autonomy::AutonomyLevel::Supervised,
            false,
        )
        .await;
    }

    #[tokio::test]
    async fn matrix_goal_policy_full_autonomy_prompt_policy_matches_ordinary() {
        let (ordinary, _) = observe_matrix_goal_tool_policy(
            false,
            zeroclaw_config::autonomy::AutonomyLevel::Full,
            true,
            None,
            1,
        )
        .await;
        let (goal, _) = observe_matrix_goal_tool_policy(
            true,
            zeroclaw_config::autonomy::AutonomyLevel::Full,
            true,
            None,
            1,
        )
        .await;
        assert!(
            ordinary[0].0[0]
                .content
                .contains(zeroclaw_runtime::agent::system_prompt::NATIVE_TOOLS_TASK_FRAMING)
        );
        assert_eq!(
            goal[0].0[0]
                .content
                .contains(zeroclaw_runtime::agent::system_prompt::NATIVE_TOOLS_TASK_FRAMING),
            ordinary[0].0[0]
                .content
                .contains(zeroclaw_runtime::agent::system_prompt::NATIVE_TOOLS_TASK_FRAMING),
        );
    }

    #[tokio::test]
    async fn matrix_goal_policy_supervised_prompt_policy_matches_ordinary() {
        let (ordinary, _) = observe_matrix_goal_tool_policy(
            false,
            zeroclaw_config::autonomy::AutonomyLevel::Supervised,
            true,
            None,
            1,
        )
        .await;
        let (goal, _) = observe_matrix_goal_tool_policy(
            true,
            zeroclaw_config::autonomy::AutonomyLevel::Supervised,
            true,
            None,
            1,
        )
        .await;
        for requests in [&ordinary, &goal] {
            assert!(requests[0].1.is_empty());
            assert!(
                !requests[0].0[0]
                    .content
                    .contains(zeroclaw_runtime::agent::system_prompt::NATIVE_TOOLS_TASK_FRAMING)
            );
            assert!(
                requests[0].0[0]
                    .content
                    .contains(zeroclaw_runtime::agent::system_prompt::NO_TOOLS_TASK_FRAMING)
            );
        }
    }

    #[tokio::test]
    async fn matrix_goal_policy_reuses_ordinary_execution_tree_budget() {
        let ordinary = observe_matrix_goal_tool_policy(
            false,
            zeroclaw_config::autonomy::AutonomyLevel::Supervised,
            false,
            Some(2),
            3,
        )
        .await;
        let goal = observe_matrix_goal_tool_policy(
            true,
            zeroclaw_config::autonomy::AutonomyLevel::Supervised,
            false,
            Some(2),
            3,
        )
        .await;
        assert_eq!(ordinary.0.len(), 2);
        assert_eq!(ordinary.1, 1);
        assert_eq!(goal.0.len(), ordinary.0.len());
        assert_eq!(goal.1, ordinary.1);
        assert_eq!(goal.0[1].1, ordinary.0[1].1);
    }

    #[tokio::test]
    async fn matrix_goal_policy_uncapped_execution_tree_budget_matches_ordinary() {
        let ordinary = observe_matrix_goal_tool_policy(
            false,
            zeroclaw_config::autonomy::AutonomyLevel::Supervised,
            false,
            None,
            1,
        )
        .await;
        let goal = observe_matrix_goal_tool_policy(
            true,
            zeroclaw_config::autonomy::AutonomyLevel::Supervised,
            false,
            None,
            1,
        )
        .await;
        assert_eq!(ordinary.0.len(), 2);
        assert_eq!(goal.0.len(), ordinary.0.len());
        assert_eq!(goal.1, ordinary.1);
    }

    #[derive(Default)]
    struct GoalApprovalProbeChannel {
        requests: std::sync::atomic::AtomicUsize,
    }

    impl zeroclaw_api::attribution::Attributable for GoalApprovalProbeChannel {
        fn role(&self) -> zeroclaw_api::attribution::Role {
            zeroclaw_api::attribution::Role::Channel(zeroclaw_api::attribution::ChannelKind::Matrix)
        }

        fn alias(&self) -> &str {
            "goal-approval-fixture"
        }
    }

    #[async_trait]
    impl Channel for GoalApprovalProbeChannel {
        fn name(&self) -> &str {
            "matrix"
        }

        async fn send(&self, _: &SendMessage) -> Result<()> {
            Ok(())
        }

        async fn listen(&self, _: tokio::sync::mpsc::Sender<ChannelMessage>) -> Result<()> {
            Ok(())
        }

        async fn request_approval(
            &self,
            _: &str,
            request: &zeroclaw_api::channel::ChannelApprovalRequest,
        ) -> Result<Option<zeroclaw_api::channel::ChannelApprovalResponse>> {
            assert_eq!(request.tool_name, "echo");
            self.requests
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(Some(
                zeroclaw_api::channel::ChannelApprovalResponse::Approve,
            ))
        }
    }

    #[tokio::test]
    async fn matrix_goal_policy_preserves_always_ask_under_both_autonomy_modes() {
        for autonomy in [
            zeroclaw_config::autonomy::AutonomyLevel::Full,
            zeroclaw_config::autonomy::AutonomyLevel::Supervised,
        ] {
            for goal in [false, true] {
                let tmp = tempfile::tempdir().unwrap();
                let backend = Arc::new(
                    zeroclaw_infra::session_sqlite::SqliteSessionBackend::new(tmp.path()).unwrap(),
                );
                let channel = Arc::new(GoalApprovalProbeChannel::default());
                let provider = Arc::new(GoalToolAvailabilityProvider {
                    tool_rounds: 1,
                    ..Default::default()
                });
                let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                let mut context = super::super::test_channel_ctx_with_backend_channel_and_provider(
                    backend,
                    channel.clone(),
                    provider,
                    0,
                );
                let ctx = Arc::get_mut(&mut context).unwrap();
                ctx.autonomy_level = autonomy;
                ctx.tools_registry = Arc::new(
                    zeroclaw_runtime::tools::scoped::ScopedToolRegistry::from_raw_for_test(vec![
                        Box::new(GoalPolicyCountingTool(calls.clone())),
                    ]),
                );
                let risk = zeroclaw_config::schema::RiskProfileConfig {
                    always_ask: vec!["echo".into()],
                    ..Default::default()
                };
                let mut config = ctx.prompt_config.as_ref().clone();
                config
                    .risk_profiles
                    .insert("ask-fixture".into(), risk.clone());
                config.agents.get_mut("test").unwrap().risk_profile = "ask-fixture".into();
                ctx.agent_cfg = Arc::new(config.agents["test"].clone());
                ctx.prompt_config = Arc::new(config.clone());
                *ctx.live_config.write() = config;
                ctx.approval_manager = Arc::new(
                    zeroclaw_runtime::approval::ApprovalManager::for_non_interactive(&risk),
                );
                if goal {
                    let (result, _) = run_matrix_policy_goal(context).await;
                    assert_eq!(result.unwrap().candidate(), "availability candidate");
                } else {
                    super::super::process_channel_message(
                        context,
                        ChannelMessage::new(
                            "approval-event",
                            "user",
                            "room",
                            "bounded repair",
                            "matrix",
                            1,
                        ),
                        tokio_util::sync::CancellationToken::new(),
                    )
                    .await;
                }
                assert_eq!(
                    channel.requests.load(std::sync::atomic::Ordering::SeqCst),
                    1
                );
                assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
            }
        }
    }

    #[tokio::test]
    async fn matrix_goal_policy_receipts_remain_provider_and_presentation_evidence() {
        let tmp = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            zeroclaw_infra::session_sqlite::SqliteSessionBackend::new(tmp.path()).unwrap(),
        );
        let provider = Arc::new(GoalPolicyProvider {
            requests: Mutex::new(Vec::new()),
            tool_rounds: 1,
            calls_per_round: 1,
        });
        let mut context = matrix_policy_context(
            backend,
            Arc::new(MatrixGoalErrorRecordingChannel::default()),
            provider.clone(),
            Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        );
        let ctx = Arc::get_mut(&mut context).unwrap();
        ctx.receipt_generator =
            Some(zeroclaw_runtime::agent::tool_receipts::ReceiptGenerator::new());
        ctx.show_receipts_in_response = true;
        let (result, lease) = run_matrix_policy_goal(context).await;
        let result = result.unwrap();
        assert_eq!(result.candidate(), "policy candidate");
        assert!(
            result
                .working_history()
                .iter()
                .any(|row| { row.role == "tool" && row.content.contains("[receipt: zc-receipt-") })
        );
        assert!(
            provider.requests.lock().unwrap()[1]
                .iter()
                .any(|row| { row.role == "tool" && row.content.contains("[receipt: zc-receipt-") })
        );
        assert!(
            lease
                .pending_receipts
                .as_deref()
                .unwrap()
                .contains("Tool receipts:")
        );
        assert!(
            result
                .working_history()
                .iter()
                .all(|row| { !row.content.contains("---\nTool receipts:") })
        );
    }

    #[tokio::test]
    async fn matrix_goal_policy_keeps_ordinary_loop_break_and_error_delivery() {
        let mut observed = Vec::new();
        for goal in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            let backend = Arc::new(
                zeroclaw_infra::session_sqlite::SqliteSessionBackend::new(tmp.path()).unwrap(),
            );
            let channel = Arc::new(MatrixGoalErrorRecordingChannel::default());
            let provider = Arc::new(GoalPolicyProvider {
                requests: Mutex::new(Vec::new()),
                tool_rounds: 8,
                calls_per_round: 1,
            });
            let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let context =
                matrix_policy_context(backend, channel.clone(), provider.clone(), calls.clone());
            if goal {
                let (result, _) = run_matrix_policy_goal(context).await;
                assert!(format!("{:#}", result.unwrap_err()).contains("Circuit breaker"));
            } else {
                super::super::process_channel_message(
                    context,
                    ChannelMessage::new(
                        "ordinary-event",
                        "user",
                        "room",
                        "bounded repair",
                        "matrix",
                        1,
                    ),
                    tokio_util::sync::CancellationToken::new(),
                )
                .await;
            }
            assert_eq!(
                channel
                    .messages
                    .lock()
                    .await
                    .iter()
                    .filter(|text| text.contains("Circuit breaker"))
                    .count(),
                1,
                "ordinary core error must render once, goal={goal}"
            );
            observed.push((
                calls.load(std::sync::atomic::Ordering::SeqCst),
                provider.requests.lock().unwrap().len(),
            ));
        }
        assert_eq!(observed[0], observed[1]);
        assert!(
            observed[0].0 > 0 && observed[0].1 < 8,
            "native detector must stop the scripted loop"
        );
    }

    #[tokio::test]
    async fn matrix_goal_policy_loop_break_commits_pairing_failure_after_core_error() {
        use zeroclaw_commands::goal::{GoalBudgetSelection, GoalCommand};
        use zeroclaw_goal::{GoalAccountingState, GoalTaskRegistry};
        use zeroclaw_runtime::control_plane::{SqliteTaskStore, TaskStatus};

        let tmp = tempfile::tempdir().unwrap();
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let backend = Arc::new(
            zeroclaw_infra::session_sqlite::SqliteSessionBackend::new(tmp.path()).unwrap(),
        );
        let channel = Arc::new(MatrixGoalErrorRecordingChannel::default());
        let provider = Arc::new(GoalPolicyProvider {
            requests: Mutex::new(Vec::new()),
            tool_rounds: 8,
            calls_per_round: 1,
        });
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let context =
            matrix_policy_context(backend, channel.clone(), provider.clone(), calls.clone());
        let key = "matrix_goal_policy_terminal_fixture";
        let driver = Arc::new(
            MatrixGoalSessionDriver::new(
                context.clone(),
                key.into(),
                "user".into(),
                "matrix:test:room".into(),
                ChannelMessage::new(
                    "event",
                    "user",
                    "room",
                    "/goal start bounded repair",
                    "matrix",
                    1,
                ),
                goal_command_lock(&context.persist_locks, key)
                    .lock_owned()
                    .await,
            )
            .unwrap(),
        );
        let ingress = GoalIngressContext::trusted(
            driver.session_key.clone(),
            "test",
            "matrix:test:room",
            GoalIngressPrincipal::Matrix {
                raw_mxid: "user".into(),
            },
        )
        .unwrap();
        let settings = GoalHostSettings::new(
            true,
            GoalBudgetLimits {
                token_limit: Some(1000),
                cost_limit_usd: None,
            },
            1,
            "policy-test-boot",
        )
        .unwrap();
        let runtime = GoalRuntime::new(store.clone());
        let tracker = Arc::new(
            CostTracker::new(
                zeroclaw_config::schema::CostConfig {
                    enabled: false,
                    ..Default::default()
                },
                tmp.path(),
            )
            .unwrap(),
        );
        let accounting = Arc::new(
            RuntimeGoalOperationAccountingFactory::new(
                store.clone(),
                tracker,
                "test",
                Arc::new(ModelProviderGoalUsagePricer::new(Arc::default())),
            )
            .unwrap(),
        );
        let supervisor =
            GoalExecutionSupervisor::new(Arc::new(runtime.execution_engine(accounting)));
        let mut submission = supervisor
            .submit(
                settings.clone(),
                ingress,
                driver,
                GoalCommand::Start {
                    budget: GoalBudgetSelection::Defaults,
                    objective: "bounded repair".into(),
                },
            )
            .await
            .unwrap();
        let GoalResponse::Started(projection) = submission.response() else {
            panic!("fixture must admit a running Goal");
        };
        let scope =
            GoalExecutionScope::new(&projection.task_id, key, projection.execution_epoch).unwrap();
        supervisor
            .launch_after_response(settings, &mut submission)
            .await
            .unwrap();
        let error =
            tokio::time::timeout(std::time::Duration::from_secs(5), supervisor.drain(&scope))
                .await
                .unwrap()
                .unwrap_err();
        assert!(format!("{error:#}").contains("Circuit breaker"));
        assert!(
            format!("{error:#}").contains("Goal parent turn returned with an unpaired tool batch")
        );
        let task = store.current_goal_for_session(key).await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(
            store
                .terminal_reason_for_session_goal(&task.id, key)
                .await
                .unwrap()
                .as_deref(),
            Some("goal_tool_pairing_incomplete")
        );
        let goal = store.get_goal_task(&task.id).await.unwrap().unwrap();
        assert_eq!(goal.pending_tool_batch_id, None);
        assert_eq!(goal.pending_tool_epoch, None);
        assert_eq!(goal.pending_call_id, None);
        assert_eq!(goal.pending_call_epoch, None);
        assert_eq!(goal.accounting_state, GoalAccountingState::Complete);
        let messages = channel.messages.lock().await;
        assert_eq!(
            messages
                .iter()
                .filter(|text| text.contains("Circuit breaker"))
                .count(),
            1
        );
        let core = messages
            .iter()
            .position(|text| text.contains("Circuit breaker"))
            .unwrap();
        let failure_text = zeroclaw_runtime::i18n::get_required_cli_string(
            matrix_goal_failure_key(GoalFailureReason::ToolResultsUnpaired),
        );
        let failure = messages
            .iter()
            .position(|text| text == &failure_text)
            .unwrap();
        assert!(
            core < failure,
            "original core error precedes transition notice"
        );
        assert_eq!(
            messages
                .iter()
                .filter(|text| *text == &failure_text)
                .count(),
            1
        );
        let parent_failure_text = zeroclaw_runtime::i18n::get_required_cli_string(
            matrix_goal_failure_key(GoalFailureReason::AgentOperationFailed),
        );
        assert_eq!(
            messages
                .iter()
                .filter(|text| *text == &parent_failure_text)
                .count(),
            0
        );
        assert!(calls.load(std::sync::atomic::Ordering::SeqCst) > 0);
        assert!(provider.requests.lock().unwrap().len() < 8);
    }

    #[tokio::test]
    async fn matrix_goal_policy_continuation_has_a_fresh_parent_budget() {
        let tmp = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            zeroclaw_infra::session_sqlite::SqliteSessionBackend::new(tmp.path()).unwrap(),
        );
        let provider = Arc::new(GoalToolAvailabilityProvider {
            tool_rounds: 3,
            ..Default::default()
        });
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut context = matrix_policy_context(
            backend,
            Arc::new(MatrixGoalErrorRecordingChannel::default()),
            Arc::new(GoalPolicyProvider {
                requests: Mutex::new(Vec::new()),
                tool_rounds: 0,
                calls_per_round: 0,
            }),
            calls.clone(),
        );
        let ctx = Arc::get_mut(&mut context).unwrap();
        ctx.model_provider = provider.clone();
        let mut config = ctx.prompt_config.as_ref().clone();
        config
            .runtime_profiles
            .get_mut("test")
            .unwrap()
            .max_execution_tree_iterations = Some(5);
        ctx.prompt_config = Arc::new(config.clone());
        *ctx.live_config.write() = config;
        let (first, mut lease) = run_matrix_policy_goal(context).await;
        let first = first.unwrap();
        assert_eq!(first.candidate(), "availability candidate");
        let mut history = first.working_history().to_vec();
        lease.pending_delivery.take();
        lease.pending_receipts.take();
        history.push(zeroclaw_goal::control::goal_verifier_feedback_message(
            "Finish the remaining criterion.",
        ));
        let operation = GoalOperationScope::new(
            GoalExecutionScope::new("goal-policy-fixture", lease.session_key.durable_id(), 1)
                .unwrap(),
        );
        let second = lease
            .run_parent_turn(
                &operation,
                GoalParentTurn {
                    kind: GoalParentTurnKind::Continue,
                    objective: "bounded repair".into(),
                    working_history: history,
                },
            )
            .await
            .unwrap();
        assert_eq!(second.candidate(), "availability candidate");
        assert_eq!(provider.requests.lock().unwrap().len(), 8);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 6);
    }

    #[tokio::test]
    async fn matrix_goal_policy_warning_is_scoped_to_one_parent_turn() {
        let tmp = tempfile::tempdir().unwrap();
        let backend = Arc::new(
            zeroclaw_infra::session_sqlite::SqliteSessionBackend::new(tmp.path()).unwrap(),
        );
        let channel = Arc::new(MatrixGoalErrorRecordingChannel::default());
        let provider = Arc::new(GoalPolicyProvider {
            requests: Mutex::new(Vec::new()),
            tool_rounds: 2,
            calls_per_round: 1,
        });
        let context = matrix_policy_context(
            backend,
            channel,
            provider.clone(),
            Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        );
        let (first, mut lease) = run_matrix_policy_goal(context).await;
        let mut history = first.unwrap().working_history().to_vec();
        let first_system = &history[0].content;
        assert!(first_system.contains("[Loop Detection]"));
        assert!(
            first_system.find("[Loop Detection]").unwrap()
                > first_system.find("bounded repair").unwrap()
        );
        lease.pending_delivery.take();
        lease.pending_receipts.take();
        history.push(zeroclaw_goal::control::goal_verifier_feedback_message(
            "Complete the remaining criterion.",
        ));
        let operation = GoalOperationScope::new(
            GoalExecutionScope::new("goal-policy-fixture", lease.session_key.durable_id(), 1)
                .unwrap(),
        );
        lease
            .run_parent_turn(
                &operation,
                GoalParentTurn {
                    kind: GoalParentTurnKind::Continue,
                    objective: "bounded repair".to_owned(),
                    working_history: history,
                },
            )
            .await
            .unwrap();
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        assert!(
            !requests.last().unwrap()[0]
                .content
                .contains("[Loop Detection]"),
            "recompose the next parent system row without the previous turn's detector note"
        );
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
    async fn matrix_goal_cold_breadcrumb_provenance_survives_context_recovery() {
        use zeroclaw_infra::session_backend::SessionBackend as _;

        let tmp = tempfile::TempDir::new().unwrap();
        let store = Arc::new(zeroclaw_infra::session_store::SessionStore::new(tmp.path()).unwrap());
        let session_key = GoalSessionKey::matrix("matrix_goal_cold_recovery_fixture").unwrap();
        let key = session_key.durable_id();
        let breadcrumb = zeroclaw_runtime::agent::history_trim::breadcrumb();
        let rows = vec![
            breadcrumb.clone(),
            ChatMessage::user("old ".repeat(20_000)),
            ChatMessage::assistant("old response"),
            ChatMessage::user("recent request"),
            ChatMessage::assistant("recent response"),
            ChatMessage::user("continue the repair"),
        ];
        for row in &rows {
            store.append(&key, row).unwrap();
        }
        store.set_session_trim_breadcrumb(&key, true).unwrap();
        let provider = Arc::new(RecoveryRecordingProvider::default());
        let context = super::super::test_channel_ctx_with_backend_channel_and_provider(
            store.clone(),
            Arc::new(MatrixGoalErrorRecordingChannel::default()),
            provider.clone(),
            0,
        );
        let operation = GoalOperationScope::new(
            GoalExecutionScope::new("goal-cold-recovery-fixture", key.clone(), 1).unwrap(),
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
        let captured = lease.take_canonical_history().unwrap();
        assert!(lease.history_has_trim_breadcrumb);
        assert_eq!(
            serde_json::to_value(&captured).unwrap(),
            serde_json::to_value(&rows).unwrap()
        );
        assert!(
            lease
                .context
                .conversation_histories
                .lock()
                .unwrap()
                .peek(&key)
                .is_none()
        );
        assert!(
            lease
                .context
                .history_crumb_flags
                .lock()
                .unwrap()
                .peek(&key)
                .is_none()
        );
        assert_eq!(
            serde_json::to_value(store.try_load(&key).unwrap()).unwrap(),
            serde_json::to_value(&rows).unwrap()
        );
        assert_eq!(store.get_session_trim_breadcrumb(&key).unwrap(), Some(true));

        let result = lease
            .run_parent_turn(
                &operation,
                GoalParentTurn {
                    kind: GoalParentTurnKind::Start,
                    objective: "bounded repair".to_owned(),
                    working_history: captured,
                },
            )
            .await
            .expect("restored ownership must survive ordinary context recovery");
        let history = result.working_history();
        assert!(lease.history_has_trim_breadcrumb);
        assert_eq!(history[1].content, breadcrumb.content);
        assert_eq!(
            history
                .iter()
                .filter(|row| row.content == breadcrumb.content)
                .count(),
            1
        );
        assert!(!history.iter().any(|row| row.content == rows[1].content));
        assert_eq!(result.candidate(), "candidate after recovery");
        assert_eq!(history.last().unwrap().role, "user");
        assert_eq!(
            history.last().unwrap().content,
            rows.last().unwrap().content
        );
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[0].iter().any(|row| row.content == rows[1].content));
        assert!(!requests[1].iter().any(|row| row.content == rows[1].content));
        for request in requests.iter() {
            assert_eq!(request[1].content, breadcrumb.content);
            assert_eq!(
                request
                    .iter()
                    .filter(|row| row.content == breadcrumb.content)
                    .count(),
                1
            );
        }
        assert_eq!(
            serde_json::to_value(store.try_load(&key).unwrap()).unwrap(),
            serde_json::to_value(rows).unwrap()
        );
        assert_eq!(store.get_session_trim_breadcrumb(&key).unwrap(), Some(true));
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
