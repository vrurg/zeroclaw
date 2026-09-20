//! Matrix implementation of the transport-neutral Goal session driver.
//!
//! This module owns only live Matrix session mechanics. Durable lifecycle,
//! accounting, and execution fencing remain in `zeroclaw-runtime`.

use std::{
    collections::HashSet,
    sync::{Arc, Mutex, atomic::Ordering},
};

use anyhow::{Context as _, Error, Result, bail, ensure};
use async_trait::async_trait;
use zeroclaw_api::{
    channel::{Channel, ChannelMessage},
    model_provider::{ChatMessage, ChatRequest},
};
use zeroclaw_providers::{
    SafeguardFallbackNotice,
    reliable::{ProviderFallbackInfo, scope_provider_fallback, take_last_provider_fallback},
    scope_safeguard_fallback, take_last_safeguard_fallback,
};
use zeroclaw_runtime::{
    agent::cost::build_type_level_model_provider_pricing,
    agent::loop_::{
        LoopKnobs, ToolLoop, is_model_switch_requested, run_tool_call_loop, scope_session_key,
        scope_thread_id,
    },
    control_plane::control_plane,
    cost::CostTracker,
    goal_mode::{
        GoalExecutionNotice, GoalExecutionScope, GoalIngressContext, GoalOperationScope,
        GoalParentInterruption, GoalParentTurn, GoalParentTurnKind, GoalParentTurnResult,
        GoalSessionBinding, GoalSessionDriver, GoalSessionExecutionLease, GoalSessionKey,
        GoalSessionLease, GoalSurface, GoalVerifierTurn, dispose_unowned_session_goal,
        goal_parent_directive, goal_parent_execution_request,
        goal_parent_system_message_with_session_prompts, goal_verifier_messages,
        scope_goal_parent_turn,
    },
};

use super::foreground::{foreground_lock, goal_command_lock, goal_supervisor_slot};
use super::{
    ChannelRouteSelection, ChannelRuntimeContext, append_sender_turn,
    build_channel_system_prompt_for_message_with_signal,
    callable_protocol_exposed_for_channel_turn, find_channel_for_message, get_or_create_provider,
    get_route_selection, goal_ledger_config, load_required_session_prompt_attachments,
    model_provider_entry_for_ref, outbound_content_format_for_channel,
    persist_session_routing_context, resolve_provider_ref_for_runtime_switch,
    runtime_defaults_from_config, runtime_defaults_snapshot,
    sanitize_channel_response_for_format_with_leak_detection, set_route_selection,
    system_prompt_for_channel_turn, turn_execution::resolved_channel_execution,
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
) -> Result<(zeroclaw_runtime::goal_mode::GoalResponse, bool)> {
    // Help is local grammar. It must not require Matrix driver validation,
    // runtime configuration, or a live control plane.
    if matches!(command, zeroclaw_commands::goal::GoalCommand::Help) {
        return Ok((zeroclaw_runtime::goal_mode::GoalResponse::Help, false));
    }
    let defaults = runtime_defaults_snapshot(context.as_ref());
    if !defaults.config.goal.enabled {
        return Ok((zeroclaw_runtime::goal_mode::GoalResponse::Disabled, false));
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
        original.clone(),
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
        // A terminal worker may already have reported the error that made its
        // Goal terminal. Its control response remains useful (and must not be
        // replaced with a generic submission failure), so terminal cleanup is
        // best-effort just as the sibling lifecycle fences are.
        let _ = retired_supervisor.drain(&scope).await;
    }
    let supervisor = {
        let mut slot = supervisor_slot.lock().await;
        if let Some(supervisor) = slot.as_ref() {
            Arc::clone(supervisor)
        } else {
            let runtime = zeroclaw_runtime::goal_mode::GoalRuntime::new(Arc::clone(&registry));
            // Reuse the exact tracker which ordinary Matrix turns already
            // use. A context created while ordinary cost tracking was disabled
            // has no retained tracker; in that case, prefer the ledger another
            // resident Matrix context already established. This mirrors an
            // ordinary disabled-cost context and avoids making Goal Mode reject
            // a valid concurrent runtime solely because its old config names a
            // different directory. If none exists, open this context's original
            // ledger strictly rather than consulting hot-reloaded defaults.
            let tracker = context
                .cost_tracking
                .as_ref()
                .map(|tracking| Arc::clone(&tracking.tracker))
                .or_else(CostTracker::get_global)
                .map(Ok)
                .unwrap_or_else(|| {
                    let ledger_config = goal_ledger_config(&context);
                    CostTracker::get_or_init_global_required(
                        ledger_config.cost.clone(),
                        &ledger_config.data_dir,
                    )
                })?;
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
        GoalSessionKey::matrix(history_key.clone())?,
        context.agent_alias.to_string(),
        route,
        zeroclaw_runtime::goal_mode::GoalIngressPrincipal::Matrix {
            raw_mxid: driver.raw_mxid.clone(),
        },
    )?;
    if matches!(
        &command,
        zeroclaw_commands::goal::GoalCommand::PauseNow
            | zeroclaw_commands::goal::GoalCommand::Cancel
    ) {
        // These are the only immediate Goal controls. They request a
        // cooperative stop of the resident parent loop; the subsequent
        // runtime submission still owns the durable pause/cancel transition
        // and drains the settled worker before acknowledging the command.
        supervisor
            .interrupt_and_drain_session(
                &history_key,
                matches!(command, zeroclaw_commands::goal::GoalCommand::Cancel),
            )
            .await?;
    }
    let initial_notice = original;
    let initial_context = Arc::clone(&context);
    let submission = supervisor
        .submit_with_before_launch(settings, ingress, driver, command, move |response| {
            let message = initial_notice.clone();
            let context = Arc::clone(&initial_context);
            let rendered = super::render_goal_response(response);
            async move {
                let channel = find_channel_for_message(&context.channels_by_name, &message)
                    .context("Matrix Goal channel is no longer available")?;
                channel
                    .send(&zeroclaw_api::channel::SendMessage::reply_to(
                        &message, rendered,
                    ))
                    .await
                    .context("deliver Matrix Goal initial notice")
            }
        })
        .await?;
    if submission.response().retires_resident_supervisor() {
        // Pause and cancellation drain the old worker before the supervisor
        // returns. Rebuilding on an explicit resume picks up live policy and
        // pricing instead of keeping a stale configuration snapshot.
        clear_supervisor_slot_if_current(&supervisor_slot, &supervisor).await;
    }
    let response = submission.into_response();
    let initial_notice_delivered = matches!(
        response,
        zeroclaw_runtime::goal_mode::GoalResponse::Started(_)
            | zeroclaw_runtime::goal_mode::GoalResponse::Resumed(_)
    );
    Ok((response, initial_notice_delivered))
}

/// Apply Matrix's `/stop` shortcut to a live Goal, if this exact conversation
/// has one. The dispatcher still cancels ordinary sender turns itself; this
/// helper owns only the Goal-specific cooperative pause and deliberately does
/// not manufacture a controller response when no Goal is running.
pub(super) async fn pause_active_matrix_goal_now(
    context: Arc<ChannelRuntimeContext>,
    history_key: String,
    original: ChannelMessage,
) -> Result<Option<zeroclaw_runtime::goal_mode::GoalResponse>> {
    let control_plane = control_plane().context("Goal control plane is unavailable")?;
    let registry = control_plane.goal_store()?;
    let Some(current) = registry.current_goal_for_session(&history_key).await? else {
        return Ok(None);
    };
    if current.status != zeroclaw_runtime::control_plane::TaskStatus::Running
        && !current.status.is_terminal()
    {
        return Ok(None);
    }
    let (response, _already_delivered) = submit_matrix_goal(
        context,
        history_key,
        original,
        zeroclaw_commands::goal::GoalCommand::PauseNow,
    )
    .await?;
    Ok(Some(response))
}

/// Request the immediate, cooperative half of Matrix's `/stop` shortcut.
///
/// This intentionally does not await the worker or make a durable lifecycle
/// transition. The dispatcher calls it from a detached control task so one
/// long-running tool cannot stall ingress for every channel; the follow-up
/// [`pause_active_matrix_goal_now`] submission owns the fenced pause once the
/// parent operation has settled.
pub(super) async fn request_active_matrix_goal_pause_now(
    context: Arc<ChannelRuntimeContext>,
    history_key: &str,
) -> Result<bool> {
    if !runtime_defaults_snapshot(context.as_ref())
        .config
        .goal
        .enabled
    {
        return Ok(false);
    }
    let control_plane = control_plane().context("Goal control plane is unavailable")?;
    let registry = control_plane.goal_store()?;
    let Some(current) = registry.current_goal_for_session(history_key).await? else {
        return Ok(false);
    };
    if current.status != zeroclaw_runtime::control_plane::TaskStatus::Running {
        return Ok(false);
    }

    let supervisor_slot = goal_supervisor_slot(&context.persist_locks, history_key);
    if let Some(supervisor) = supervisor_slot.lock().await.as_ref().cloned() {
        let _ = supervisor.interrupt_session(history_key, false).await;
    }
    Ok(true)
}

/// Render only stable, actionable command failures. The original error stays
/// in structured logs; this surface must not turn arbitrary configuration or
/// provider diagnostics into user-visible text.
pub(super) fn render_matrix_goal_command_error(error: &Error) -> String {
    if error.chain().any(|cause| {
        cause
            .to_string()
            .contains("required cost tracker storage path differs")
    }) {
        return zeroclaw_runtime::i18n::get_required_cli_string(
            "goal-mode-command-accounting-storage",
        );
    }
    if error.chain().any(|cause| {
        cause
            .to_string()
            .contains("Goal control plane is unavailable")
    }) {
        return zeroclaw_runtime::i18n::get_required_cli_string(
            "goal-mode-command-control-plane-unavailable",
        );
    }
    zeroclaw_runtime::i18n::get_required_cli_string("goal-mode-command-failed")
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
            parent_candidate_history: None,
            cancellation: None,
        }))
    }
}

struct MatrixGoalExecutionLease {
    _foreground: tokio::sync::OwnedMutexGuard<()>,
    context: Arc<ChannelRuntimeContext>,
    session_key: GoalSessionKey,
    message: ChannelMessage,
    /// The exact post-hook, post-sanitization response ordinary Matrix session
    /// history would retain. It deliberately excludes recovery presentation
    /// footers, matching the non-Goal channel path.
    parent_candidate_history: Option<String>,
    cancellation: Option<tokio_util::sync::CancellationToken>,
}

impl MatrixGoalExecutionLease {
    fn history(&self) -> Vec<ChatMessage> {
        let history = self
            .context
            .conversation_histories
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .peek(&self.session_key.durable_id())
            .cloned()
            .unwrap_or_default();
        super::prepare_cached_channel_history(history)
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

    /// Resolve durable task context for each parent operation so a resumed
    /// Goal sees the same current session prompt as an ordinary Matrix turn.
    fn session_prompt_attachments(&self) -> Result<String> {
        load_required_session_prompt_attachments(
            self.context.as_ref(),
            &self.session_key.durable_id(),
        )
    }
}

fn goal_start_history(
    system_prompt: String,
    directive: ChatMessage,
    canonical_history: Vec<ChatMessage>,
    session_prompt_attachments: &str,
    max_system_prompt_chars: usize,
) -> Result<Vec<ChatMessage>> {
    let mut history = Vec::with_capacity(canonical_history.len() + 2);
    history.push(goal_parent_system_message_with_session_prompts(
        system_prompt,
        directive.content,
        session_prompt_attachments,
        max_system_prompt_chars,
    )?);
    history.extend(canonical_history);
    history.push(goal_parent_execution_request());
    Ok(history)
}

/// Rebuild a canonical Goal transcript for one parent turn.
///
/// A retained transcript owns the continuation case, but a Goal can also be
/// resumed after the resident worker was retired. In that case the canonical
/// session history is the only durable prefix and an optional user response
/// must still be the final user turn seen by the new worker. Keep the ordinary
/// execution request as well: unlike a retained transcript, a canonical prefix
/// has no guaranteed terminal role, and provider conversion coalesces the
/// adjacent user messages when a response follows it.
fn goal_canonical_history_for_parent_turn(
    turn: GoalParentTurn,
    system_prompt: String,
    directive: ChatMessage,
    session_prompt_attachments: &str,
    max_system_prompt_chars: usize,
) -> Result<Vec<ChatMessage>> {
    let mut history = goal_start_history(
        system_prompt,
        directive,
        turn.working_history,
        session_prompt_attachments,
        max_system_prompt_chars,
    )?;
    if let Some(response) = turn.resume_response {
        history.push(ChatMessage::user(response));
    }
    Ok(history)
}

fn goal_continue_history(
    system_prompt: String,
    directive: ChatMessage,
    mut history: Vec<ChatMessage>,
    session_prompt_attachments: &str,
    max_system_prompt_chars: usize,
) -> Result<Vec<ChatMessage>> {
    let first = history
        .first_mut()
        .context("Goal continuation lost its system prompt")?;
    ensure!(
        first.role == "system",
        "Goal continuation lost its system prompt"
    );
    *first = goal_parent_system_message_with_session_prompts(
        system_prompt,
        directive.content,
        session_prompt_attachments,
        max_system_prompt_chars,
    )?;
    history.push(goal_parent_execution_request());
    Ok(history)
}

/// Refresh a retained verifier-blocked transcript for an explicit resume.
/// Unlike an ordinary verifier `Continue`, a reply to the blocked candidate
/// must follow it directly. A bare resume still needs the normal user-role
/// execution request so every provider can accept the transcript.
fn goal_resume_history(
    system_prompt: String,
    directive: ChatMessage,
    mut history: Vec<ChatMessage>,
    session_prompt_attachments: &str,
    max_system_prompt_chars: usize,
    append_execution_request: bool,
) -> Result<Vec<ChatMessage>> {
    let first = history
        .first_mut()
        .context("Goal resume lost its system prompt")?;
    ensure!(first.role == "system", "Goal resume lost its system prompt");
    *first = goal_parent_system_message_with_session_prompts(
        system_prompt,
        directive.content,
        session_prompt_attachments,
        max_system_prompt_chars,
    )?;
    if append_execution_request {
        history.push(goal_parent_execution_request());
    }
    Ok(history)
}

/// Rebuild a resident Goal transcript for one Matrix parent turn.
///
/// The branch lives here rather than at the call site so the exact resume
/// semantics (explicit reply versus bare resume) have a direct regression
/// test. An explicit reply must remain the final user turn; a bare resume
/// receives the normal user-role execution request.
fn goal_continuation_history_for_parent_turn(
    turn: GoalParentTurn,
    system_prompt: String,
    directive: ChatMessage,
    session_prompt_attachments: &str,
    max_system_prompt_chars: usize,
) -> Result<Vec<ChatMessage>> {
    let mut history = if turn.kind == GoalParentTurnKind::Resume {
        goal_resume_history(
            system_prompt,
            directive,
            turn.working_history,
            session_prompt_attachments,
            max_system_prompt_chars,
            turn.resume_response.is_none(),
        )?
    } else {
        goal_continue_history(
            system_prompt,
            directive,
            turn.working_history,
            session_prompt_attachments,
            max_system_prompt_chars,
        )?
    };
    if let Some(response) = turn.resume_response {
        history.push(ChatMessage::user(response));
    }
    Ok(history)
}

/// Rebuild the sole provider-sensitive message after a runtime route switch.
///
/// The rest of the transcript already contains the switch tool call and its
/// result. Replacing only this message preserves the exact Goal directive and
/// every preceding agent event while letting the next model see its own tool
/// capabilities.
fn refresh_goal_system_prompt_after_model_switch(
    history: &mut [ChatMessage],
    system_prompt: String,
    directive: &ChatMessage,
    session_prompt_attachments: &str,
    max_system_prompt_chars: usize,
) -> Result<()> {
    let first = history
        .first_mut()
        .context("Goal model switch lost its system prompt")?;
    ensure!(
        first.role == "system",
        "Goal model switch lost its system prompt"
    );
    *first = goal_parent_system_message_with_session_prompts(
        system_prompt,
        directive.content.clone(),
        session_prompt_attachments,
        max_system_prompt_chars,
    )?;
    Ok(())
}

/// The normal channel presentation plumbing for one Goal parent operation.
///
/// Goal execution uses the same draft and tool-notification paths as an
/// ordinary channel turn. Keeping configuration and rendering decisions in the
/// existing helpers is deliberate: Goal Mode must not become a second
/// presentation policy.
struct GoalParentPresentation {
    context: Arc<ChannelRuntimeContext>,
    message: ChannelMessage,
    channel: Option<Arc<dyn Channel>>,
    draft_id: Option<String>,
    /// Per-turn receipts use the same collector as an ordinary channel turn.
    /// Keeping it here lets a Goal parent operation present every enabled
    /// tool-result surface before the verifier decides lifecycle state.
    receipts: Arc<Mutex<Vec<String>>>,
    on_delta: Option<tokio::sync::mpsc::Sender<zeroclaw_runtime::agent::loop_::StreamDelta>>,
    observer: Arc<super::ChannelNotifyObserver>,
    draft_updater: Option<tokio::task::JoinHandle<()>>,
    notify_task: Option<tokio::task::JoinHandle<()>>,
    /// Mirrors ordinary Matrix `single_message` typing from the accepted
    /// command through the parent turn's final draft delivery.
    matrix_single_message_typing_scope: Option<super::MatrixSingleMessageTypingScope>,
}

impl GoalParentPresentation {
    async fn start(context: Arc<ChannelRuntimeContext>, message: &ChannelMessage) -> Self {
        let receipts = Arc::new(Mutex::new(Vec::new()));
        let channel = find_channel_for_message(&context.channels_by_name, message).cloned();
        let use_draft_streaming = channel
            .as_ref()
            .is_some_and(|channel| channel.supports_draft_updates());
        let matrix_single_message_streaming =
            super::matrix_single_message_streaming_enabled(&context, message);
        let matrix_single_message_typing_scope = matrix_single_message_streaming
            .then(|| {
                channel.as_ref().map(|channel| {
                    super::start_matrix_single_message_typing_scope(
                        Arc::clone(channel),
                        message.reply_target.clone(),
                    )
                })
            })
            .flatten();

        let (on_delta, draft_updater, draft_id) = if use_draft_streaming {
            // Use the ordinary one-hop channel-to-renderer queue. A Goal
            // lifecycle must not introduce another buffer that can alter
            // stream backpressure or event ordering.
            let (on_delta, visible_rx) = tokio::sync::mpsc::channel(64);
            let draft_id = if let Some(channel) = channel.as_ref() {
                match channel
                    .send_draft(&zeroclaw_api::channel::SendMessage::reply_to(
                        message,
                        zeroclaw_runtime::agent::loop_::DRAFT_PLACEHOLDER,
                    ))
                    .await
                {
                    Ok(id) => id,
                    Err(error) => {
                        ::zeroclaw_log::record!(
                            DEBUG,
                            ::zeroclaw_log::Event::new(
                                module_path!(),
                                ::zeroclaw_log::Action::Note
                            )
                            .with_attrs(::serde_json::json!({"error": error.to_string()})),
                            "Goal draft creation failed"
                        );
                        None
                    }
                }
            } else {
                None
            };
            let draft_updater = match (channel.as_ref(), draft_id.clone()) {
                (Some(channel), Some(draft_id)) if matrix_single_message_streaming => {
                    let channel = Arc::clone(channel);
                    let reply_target = message.reply_target.clone();
                    let matrix_config = Arc::clone(&context.prompt_config);
                    let matrix_alias = message.channel_alias.clone().unwrap_or_default();
                    let interval_ms = super::matrix_draft_update_interval_ms(&context, message);
                    let stream_draft_lines = super::matrix_stream_draft_lines(&context, message);
                    Some(zeroclaw_spawn::spawn!(async move {
                        super::run_matrix_single_message_draft_updater(
                            visible_rx,
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
                (Some(channel), Some(draft_id)) => {
                    let channel = Arc::clone(channel);
                    let reply_target = message.reply_target.clone();
                    let known_tool_names: HashSet<String> = context
                        .tools_registry
                        .iter()
                        .map(|tool| tool.name().to_ascii_lowercase())
                        .collect();
                    Some(zeroclaw_spawn::spawn!(async move {
                        super::run_draft_updater(
                            channel,
                            reply_target,
                            draft_id,
                            known_tool_names,
                            visible_rx,
                        )
                        .await;
                    }))
                }
                _ => None,
            };
            (Some(on_delta), draft_updater, draft_id)
        } else {
            (None, None, None)
        };

        let is_partial_draft = channel.as_ref().is_some_and(|channel| {
            channel.supports_draft_updates() && !channel.supports_multi_message_streaming()
        }) || matrix_single_message_streaming;
        let (notify_tx, notify_task) = if matrix_single_message_streaming {
            (None, None)
        } else {
            let (notify_tx, mut notify_rx) = tokio::sync::mpsc::channel::<String>(128);
            let notify_channel = channel.clone();
            let reply_target = message.reply_target.clone();
            let thread_ts = super::followup_thread_id(message);
            let notify_task = if !context.show_tool_calls || is_partial_draft {
                Some(zeroclaw_spawn::spawn!(async move {
                    while notify_rx.recv().await.is_some() {}
                }))
            } else {
                Some(zeroclaw_spawn::spawn!(async move {
                    while let Some(text) = notify_rx.recv().await {
                        if let Some(channel) = notify_channel.as_ref() {
                            let _ = channel
                                .send(
                                    &zeroclaw_api::channel::SendMessage::new(&text, &reply_target)
                                        .in_thread(thread_ts.clone())
                                        .suppress_voice(),
                                )
                                .await;
                        }
                    }
                }))
            };
            (Some(notify_tx), notify_task)
        };

        Self {
            context: Arc::clone(&context),
            message: message.clone(),
            channel,
            draft_id,
            receipts,
            on_delta,
            observer: Arc::new(super::ChannelNotifyObserver {
                inner: Arc::clone(&context.observer),
                tx: notify_tx,
                tools_used: std::sync::atomic::AtomicBool::new(false),
            }),
            draft_updater,
            notify_task,
            matrix_single_message_typing_scope,
        }
    }

    /// Finish the same response presentation an ordinary Matrix turn uses.
    ///
    /// This only makes the parent response visible through the configured
    /// channel surface. The Goal controller then records that visible response
    /// in ordinary session history before verification governs the Goal
    /// lifecycle separately.
    async fn finish(
        self,
        candidate: Option<&str>,
        provider_fallback: Option<&ProviderFallbackInfo>,
        safeguard_fallback: Option<&SafeguardFallbackNotice>,
        turn_route: Option<zeroclaw_runtime::tools::TurnRoutingEntry>,
        cancellation: Option<tokio_util::sync::CancellationToken>,
    ) -> Option<String> {
        let Self {
            context,
            message,
            channel,
            draft_id,
            receipts,
            on_delta,
            observer,
            draft_updater,
            notify_task,
            matrix_single_message_typing_scope,
        } = self;
        drop(on_delta);
        if let Some(draft_updater) = draft_updater {
            let _ = draft_updater.await;
        }
        let tools_used = observer.tools_used.load(Ordering::Relaxed);
        drop(observer);
        if let Some(notify_task) = notify_task {
            let _ = notify_task.await;
        }
        if let Some(scope) = matrix_single_message_typing_scope {
            super::stop_matrix_single_message_typing_scope(scope).await;
        }
        Self::present_parent_result(
            context.as_ref(),
            &message,
            channel.as_ref(),
            draft_id.as_deref(),
            candidate,
            provider_fallback,
            safeguard_fallback,
            tools_used,
            receipts,
            turn_route,
            cancellation.as_ref(),
        )
        .await
    }

    async fn present_parent_result(
        context: &ChannelRuntimeContext,
        message: &ChannelMessage,
        channel: Option<&Arc<dyn Channel>>,
        draft_id: Option<&str>,
        candidate: Option<&str>,
        provider_fallback: Option<&ProviderFallbackInfo>,
        safeguard_fallback: Option<&SafeguardFallbackNotice>,
        tools_used: bool,
        receipts: Arc<Mutex<Vec<String>>>,
        turn_route: Option<zeroclaw_runtime::tools::TurnRoutingEntry>,
        cancellation: Option<&tokio_util::sync::CancellationToken>,
    ) -> Option<String> {
        let Some(candidate) = candidate else {
            if let (Some(channel), Some(draft_id)) = (channel, draft_id) {
                let _ = channel.cancel_draft(&message.reply_target, draft_id).await;
            }
            return None;
        };
        let mut outbound = candidate.to_owned();
        if let Some(hooks) = context.hooks.as_ref() {
            match hooks
                .run_on_message_sending(
                    message.channel.clone(),
                    message.reply_target.clone(),
                    outbound.clone(),
                )
                .await
            {
                zeroclaw_runtime::hooks::HookResult::Cancel(_) => {
                    if let (Some(channel), Some(draft_id)) = (channel, draft_id) {
                        let _ = channel.cancel_draft(&message.reply_target, draft_id).await;
                    }
                    return None;
                }
                zeroclaw_runtime::hooks::HookResult::Continue((_, _, mut content)) => {
                    if content.chars().count() > super::CHANNEL_HOOK_MAX_OUTBOUND_CHARS {
                        content = super::truncate_with_ellipsis(
                            &content,
                            super::CHANNEL_HOOK_MAX_OUTBOUND_CHARS,
                        );
                    }
                    outbound = content;
                }
            }
        }
        let sanitized = sanitize_channel_response_for_format_with_leak_detection(
            &outbound,
            context.tools_registry.as_ref(),
            &context.prompt_config.security.leak_detection,
            outbound_content_format_for_channel(&message.channel),
        );
        let delivered = if sanitized.is_empty() && !outbound.trim().is_empty() {
            super::channel_runtime_cli_string("channel-runtime-malformed-tool-output")
        } else {
            sanitized
        };
        let delivered = super::ensure_nonempty_channel_reply(
            delivered,
            &outbound,
            &message.channel,
            &message.reply_target,
        );
        // This is the normal non-Goal history value. Provider-recovery
        // footers are surface-only, so they are added only to the rendered
        // response below and are not retained in session history.
        let history_response = delivered.clone();
        let delivered = super::append_provider_fallback_footer(
            delivered,
            provider_fallback,
            safeguard_fallback,
        );
        // Ordinary channel turns retain their semantic assistant response
        // before attempting delivery. A transiently unavailable Matrix
        // channel must not make a Goal-managed session forget that turn.
        let Some(channel) = channel else {
            return Some(history_response);
        };
        let (delivery_channel, delivery_recipient, suppress_voice, force_voice, is_redirect) =
            if let Some(route) = turn_route {
                let delivery_channel = match route.channel.as_deref() {
                    None | Some("") => Some(Arc::clone(channel)),
                    Some(name) => context.channels_by_name.get(name).map(Arc::clone),
                };
                let recipient = route
                    .recipient
                    .unwrap_or_else(|| message.reply_target.clone());
                let suppress_voice = match route.modality {
                    zeroclaw_config::multi_agent::OutputModality::Text => Some(true),
                    zeroclaw_config::multi_agent::OutputModality::Voice => Some(false),
                    zeroclaw_config::multi_agent::OutputModality::Mirror => None,
                };
                let force_voice = matches!(
                    route.modality,
                    zeroclaw_config::multi_agent::OutputModality::Voice
                );
                (
                    delivery_channel,
                    recipient,
                    suppress_voice,
                    force_voice,
                    route.channel.is_some(),
                )
            } else {
                let (suppress_voice, force_voice) = super::voice_override_from_sender_verdict(
                    super::sender_prefers_voice(context, message),
                );
                (
                    Some(Arc::clone(channel)),
                    message.reply_target.clone(),
                    suppress_voice,
                    force_voice,
                    false,
                )
            };
        let Some(delivery_channel) = delivery_channel else {
            if let Some(draft_id) = draft_id {
                let _ = channel.cancel_draft(&message.reply_target, draft_id).await;
            }
            return Some(history_response);
        };
        let thread_ts = tools_used
            .then(|| super::followup_thread_id(message))
            .flatten();

        let delivered_to_channel = deliver_goal_parent_response(
            channel.as_ref(),
            delivery_channel.as_ref(),
            message,
            draft_id,
            &delivered,
            &delivery_recipient,
            suppress_voice,
            force_voice,
            thread_ts.clone(),
            is_redirect,
            cancellation,
        )
        .await;
        if delivered_to_channel {
            if let Some(hooks) = context.hooks.as_ref() {
                hooks
                    .fire_message_sent(&message.channel, &message.reply_target, &delivered)
                    .await;
            }
            if context.show_receipts_in_response {
                let receipts_block = {
                    let receipts = receipts.lock().unwrap_or_else(|error| error.into_inner());
                    zeroclaw_runtime::agent::tool_receipts::render_receipts_block(&receipts)
                };
                if let Some(block) = receipts_block {
                    let _ = channel
                        .send(
                            &zeroclaw_api::channel::SendMessage::new(&block, &delivery_recipient)
                                .in_thread(thread_ts)
                                .suppress_voice(),
                        )
                        .await;
                }
            }
        }
        Some(history_response)
    }
}

/// Reuse the ordinary response delivery modes after a Goal parent turn.
///
/// The caller has already applied the normal response sanitizer. This helper
/// deliberately owns no Goal state: it is only the channel presentation step.
async fn deliver_goal_parent_response(
    origin_channel: &dyn Channel,
    delivery_channel: &dyn Channel,
    message: &ChannelMessage,
    draft_id: Option<&str>,
    delivered: &str,
    delivery_recipient: &str,
    suppress_voice: Option<bool>,
    force_voice: bool,
    thread_ts: Option<String>,
    is_redirect: bool,
    cancellation: Option<&tokio_util::sync::CancellationToken>,
) -> bool {
    if is_redirect {
        if let Some(draft_id) = draft_id {
            let _ = origin_channel
                .cancel_draft(&message.reply_target, draft_id)
                .await;
        }
        let suppress_voice = suppress_voice.unwrap_or(false);
        let mut reply = zeroclaw_api::channel::SendMessage::new(delivered, delivery_recipient)
            .in_thread(thread_ts);
        if suppress_voice {
            reply = reply.suppress_voice();
        } else if force_voice {
            reply = reply.force_voice();
        }
        return delivery_channel.send_final(&reply).await.is_ok();
    }
    match draft_id {
        Some(draft_id) if force_voice => {
            let _ = origin_channel
                .cancel_draft(delivery_recipient, draft_id)
                .await;
            delivery_channel
                .send_final(
                    &zeroclaw_api::channel::SendMessage::new(delivered, delivery_recipient)
                        .force_voice()
                        .in_thread(thread_ts),
                )
                .await
                .is_ok()
        }
        Some(draft_id) => {
            let suppress_voice = suppress_voice.unwrap_or(false);
            match delivery_channel
                .finalize_draft(delivery_recipient, draft_id, delivered, suppress_voice)
                .await
            {
                Ok(()) => true,
                Err(_) => {
                    let mut fallback =
                        zeroclaw_api::channel::SendMessage::reply_to(message, delivered)
                            .in_thread(thread_ts);
                    if suppress_voice {
                        fallback = fallback.suppress_voice();
                    }
                    if let Some(cancellation) = cancellation {
                        fallback = fallback.with_cancellation(cancellation.clone());
                    }
                    delivery_channel.send_final(&fallback).await.is_ok()
                }
            }
        }
        None => {
            let mut reply = zeroclaw_api::channel::SendMessage::reply_to(message, delivered)
                .in_thread(thread_ts);
            if suppress_voice.unwrap_or(false) {
                reply = reply.suppress_voice();
            } else if force_voice {
                reply = reply.force_voice();
            }
            if let Some(cancellation) = cancellation {
                reply = reply.with_cancellation(cancellation.clone());
            }
            delivery_channel.send_final(&reply).await.is_ok()
        }
    }
}

#[async_trait]
impl GoalSessionExecutionLease for MatrixGoalExecutionLease {
    fn session_key(&self) -> &GoalSessionKey {
        &self.session_key
    }

    fn canonical_history(&self) -> Result<Vec<ChatMessage>> {
        Ok(self.history())
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
        operation: &GoalOperationScope,
        turn: GoalParentTurn,
    ) -> Result<GoalParentTurnResult> {
        let defaults = runtime_defaults_snapshot(self.context.as_ref());
        let mut route = get_route_selection(
            self.context.as_ref(),
            &self.message,
            &self.session_key.durable_id(),
            &defaults,
        );
        let mut provider = get_or_create_provider(
            self.context.as_ref(),
            &route.model_provider,
            route.api_key.as_deref(),
            &defaults,
        )
        .await?;
        let memory_query = turn.objective.clone();
        let directive = goal_parent_directive(&turn);
        let session_prompt_attachments = self.session_prompt_attachments()?;
        let mut history = match turn.history_source {
            zeroclaw_runtime::goal_mode::GoalParentHistorySource::Canonical => {
                goal_canonical_history_for_parent_turn(
                    turn,
                    self.goal_system_prompt(provider.as_ref(), &route),
                    directive.clone(),
                    &session_prompt_attachments,
                    self.context.agent_cfg.resolved.max_system_prompt_chars,
                )?
            }
            zeroclaw_runtime::goal_mode::GoalParentHistorySource::Continuation => {
                goal_continuation_history_for_parent_turn(
                    turn,
                    self.goal_system_prompt(provider.as_ref(), &route),
                    directive.clone(),
                    &session_prompt_attachments,
                    self.context.agent_cfg.resolved.max_system_prompt_chars,
                )?
            }
        };
        // A Goal turn restores the same saved session-prompt tail as an
        // ordinary Matrix turn. Preserve the matching tool/backend scope too:
        // otherwise the agent sees instructions for a capability Goal Mode
        // has silently removed.
        let turn_id = uuid::Uuid::new_v4().to_string();
        let mut loop_knobs = LoopKnobs::default();
        if super::matrix_single_message_streaming_enabled(self.context.as_ref(), &self.message) {
            loop_knobs.draft_reasoning =
                super::matrix_stream_reasoning(self.context.as_ref(), &self.message);
        }
        // `run_parent_turn` invokes the shared tool loop directly, so it must
        // bracket that loop exactly as an ordinary channel turn does. This
        // keeps API/observer consumers from losing the parent turn merely
        // because Goal Mode governs its lifecycle.
        let turn_observer = Arc::clone(&self.context.observer);
        let mut turn_guard = zeroclaw_runtime::observability::AgentTurnGuard::start(
            turn_observer.as_ref(),
            route.model_provider.clone(),
            route.model.clone(),
            Some(self.message.channel.clone()),
            Some(self.context.agent_alias.to_string()),
            Some(turn_id.clone()),
        );
        self.parent_candidate_history = None;
        let presentation =
            GoalParentPresentation::start(Arc::clone(&self.context), &self.message).await;
        if let Some(on_delta) = presentation.on_delta.as_ref() {
            // Match the normal channel turn's initial presentation events.
            // Matrix single-message renders its own configured progress chrome;
            // other draft-capable channels consume these typed lifecycle events.
            let _ = on_delta
                .send(zeroclaw_runtime::agent::loop_::StreamDelta::Lifecycle(
                    zeroclaw_runtime::agent::loop_::ProgressEvent::Received,
                ))
                .await;
            let _ = on_delta
                .send(zeroclaw_runtime::agent::loop_::StreamDelta::Lifecycle(
                    zeroclaw_runtime::agent::loop_::ProgressEvent::Planning,
                ))
                .await;
        }
        let thread_message_id = self.message.id.clone();
        let excluded_tools: &[String] =
            if self.context.autonomy_level == zeroclaw_config::autonomy::AutonomyLevel::Full {
                &[]
            } else {
                self.context.non_cli_excluded_tools.as_ref()
            };
        // Goal Mode changes the parent directive, not the Matrix session's
        // ordinary memory scope. Use the declared objective rather than the
        // raw `/goal` command as the retrieval query so command grammar does
        // not distort the agent's normal recall behavior.
        let memory_sessions = goal_memory_sessions(&self.message, self.session_key.durable_id());
        // `send_via` is ordinary parent-turn behavior. Give this operation its
        // own routing handle so its requested destination and modality cannot
        // leak to a concurrent Goal or ordinary channel turn.
        let turn_routing: zeroclaw_runtime::tools::TurnRoutingHandle =
            Arc::new(Mutex::new(Vec::new()));
        let receipt_scope = self.context.receipt_generator.as_ref().map(|generator| {
            zeroclaw_runtime::agent::tool_receipts::ReceiptScope {
                generator: generator.clone(),
                collector: Arc::clone(&presentation.receipts),
            }
        });
        // Mirror the normal Matrix turn's recovery scopes. The candidate stays
        // exact for verifier input and canonical history; only the surface
        // presentation receives the ordinary recovery footer.
        let (candidate, provider_fallback, safeguard_fallback) = scope_safeguard_fallback(async {
            let (candidate, provider_fallback) = scope_provider_fallback(async {
                let candidate = scope_goal_parent_turn(scope_session_key(
                    Some(self.session_key.durable_id()),
                    async {
                        loop {
                            let session_prompt_budget = zeroclaw_infra::session_backend::SessionPromptBudget::new(
                                history
                                    .first()
                                    .filter(|message| message.role == "system")
                                    .map_or(0, |message| message.content.chars().count()),
                                self.context.agent_cfg.resolved.max_system_prompt_chars,
                            );
                            let tool_loop = run_tool_call_loop(ToolLoop {
                                exec: resolved_channel_execution(
                                    self.context.as_ref(),
                                    provider.as_ref(),
                                    &route,
                                    presentation.observer.as_ref(),
                                    &loop_knobs,
                                    excluded_tools,
                                    defaults.defaults.temperature,
                                ),
                                history: &mut history,
                                channel_name: "matrix",
                                channel_reply_target: Some(self.message.reply_target.as_str()),
                                cancellation_token: self.cancellation.clone(),
                                on_delta: presentation.on_delta.clone(),
                                shared_budget: None,
                                // Keep ordinary channel approvals and their
                                // interactive representation available. Goal
                                // Mode governs the lifecycle; it must not
                                // turn a visible Matrix approval into an
                                // automatic denial by withholding the channel
                                // handle from the shared tool loop.
                                channel: presentation.channel.as_deref(),
                                collected_receipts: self
                                    .context
                                    .receipt_generator
                                    .as_ref()
                                    .map(|_| presentation.receipts.as_ref()),
                                event_tx: None,
                                steering: None,
                                new_messages_out: None,
                                image_cache: None,
                                ingress: zeroclaw_api::ingress::IngressContext::channel(),
                                memory: Some(zeroclaw_runtime::agent::memory_inject::TurnMemory {
                                    handle: self.context.memory.as_ref(),
                                    query: memory_query.clone(),
                                    sessions: memory_sessions.clone(),
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
                                sop_reassembly: Some(zeroclaw_runtime::agent::loop_::SopStepReassembly {
                                    config: self.context.prompt_config.as_ref(),
                                }),
                            });
                            let tool_loop = zeroclaw_runtime::agent::tool_receipts::TOOL_LOOP_RECEIPT_CONTEXT
                                .scope(receipt_scope.clone(), tool_loop);
                            let tool_loop = zeroclaw_runtime::tools::TURN_ROUTING
                                .scope(Some(Arc::clone(&turn_routing)), tool_loop);
                            let tool_loop = zeroclaw_api::TOOL_LOOP_SESSION_PROMPTS_ALLOWED.scope(
                                self.context.prompt_config.channels.session_prompts_enabled
                                    && self.context.session_store.is_some(),
                                tool_loop,
                            );
                            let tool_loop = zeroclaw_infra::session_backend::TOOL_LOOP_SESSION_BACKEND.scope(
                                self.context
                                    .session_store
                                    .clone()
                                    .map(zeroclaw_infra::session_backend::ScopedSessionBackend),
                                tool_loop,
                            );
                            let tool_loop = zeroclaw_infra::session_backend::TOOL_LOOP_SESSION_PROMPT_BUDGET
                                .scope(Some(session_prompt_budget), tool_loop);
                            let result = scope_thread_id(Some(thread_message_id.clone()), tool_loop).await;

                            let Err(error) = result else {
                                break result;
                            };
                            let Some((requested_provider, requested_model)) =
                                is_model_switch_requested(&error)
                            else {
                                break Err(error);
                            };
                            let resolved_provider = resolve_provider_ref_for_runtime_switch(
                                defaults.config.as_ref(),
                                &requested_provider,
                            )?;
                            let api_key = self
                                .context
                                .model_routes
                                .iter()
                                .find(|candidate| {
                                    candidate.model_provider.eq_ignore_ascii_case(&requested_provider)
                                        && (candidate.model.eq_ignore_ascii_case(&requested_model)
                                            || candidate.hint.eq_ignore_ascii_case(&requested_model))
                                })
                                .and_then(|candidate| candidate.api_key.clone());
                            let next_provider = get_or_create_provider(
                                self.context.as_ref(),
                                &resolved_provider,
                                api_key.as_deref(),
                                &defaults,
                            )
                            .await?;

                            // Match an ordinary Matrix turn: only commit the new route after its
                            // provider is available, then regenerate the provider-sensitive Goal
                            // system message while retaining the same directive and transcript.
                            provider = next_provider;
                            route = ChannelRouteSelection {
                                model_provider: resolved_provider,
                                model: requested_model,
                                api_key,
                            };
                            set_route_selection(
                                self.context.as_ref(),
                                &self.session_key.durable_id(),
                                route.clone(),
                                &defaults,
                            );
                            refresh_goal_system_prompt_after_model_switch(
                                &mut history,
                                self.goal_system_prompt(provider.as_ref(), &route),
                                &directive,
                                &session_prompt_attachments,
                                self.context.agent_cfg.resolved.max_system_prompt_chars,
                            )?;
                        }
                    },
                ))
                .await;
                (candidate, take_last_provider_fallback())
            })
            .await;
            (candidate, provider_fallback, take_last_safeguard_fallback())
        })
        .await;
        // Attribute the matching end event to the final route if a model
        // switch occurred. Goal accounting is owned by the Goal engine, so
        // this observer guard supplies lifecycle parity only.
        turn_guard.set_model_route(route.model_provider.clone(), route.model.clone());
        turn_guard.finish();
        let candidate = match candidate {
            Ok(candidate) => {
                let turn_route = turn_routing
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .last()
                    .cloned();
                let presentation = presentation
                    .finish(
                        Some(&candidate),
                        provider_fallback.as_ref(),
                        safeguard_fallback.as_ref(),
                        turn_route,
                        self.cancellation.clone(),
                    )
                    .await;
                self.parent_candidate_history = presentation;
                candidate
            }
            Err(error) => {
                presentation
                    .finish(None, None, None, None, self.cancellation.clone())
                    .await;
                let interruption =
                    if zeroclaw_runtime::agent::tool_loop_safety_interruption(&error).is_some() {
                        GoalParentInterruption::ToolLoopSafety {
                            message: error.to_string(),
                        }
                    } else if zeroclaw_providers::reliable::is_context_window_exceeded(&error) {
                        GoalParentInterruption::ContextWindowExceeded {
                            message: error.to_string(),
                        }
                    } else {
                        return Err(error);
                    };
                return Ok(GoalParentTurnResult {
                    candidate: String::new(),
                    working_history: history,
                    interruption: Some(interruption),
                });
            }
        };
        let _ = operation;
        Ok(GoalParentTurnResult {
            candidate,
            working_history: history,
            interruption: None,
        })
    }

    async fn present_core_error(&mut self, error: &Error) -> Result<()> {
        let channel = find_channel_for_message(&self.context.channels_by_name, &self.message)
            .context("Matrix Goal channel is no longer available")?;
        channel
            .send(&goal_core_error_reply(&self.message, error))
            .await
            .context("deliver Matrix Goal parent error")
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

    async fn record_presented_parent_candidate(&mut self, _candidate: String) -> Result<()> {
        let Some(delivered) = self.parent_candidate_history.take() else {
            return Ok(());
        };
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

/// Construct a Goal parent-error delivery using the exact ordinary channel
/// error rendering and modality. Goal lifecycle handling must not cause an
/// error to become a voice reply or otherwise diverge from a normal turn.
fn goal_core_error_reply(
    message: &ChannelMessage,
    error: &Error,
) -> zeroclaw_api::channel::SendMessage {
    let safe_error = zeroclaw_providers::sanitize_api_error(&error.to_string());
    let content = super::channel_user_error_message(error, &safe_error);
    zeroclaw_api::channel::SendMessage::reply_to(message, content).suppress_voice()
}

fn goal_notice_message(notice: GoalExecutionNotice) -> String {
    match notice {
        GoalExecutionNotice::Completed => {
            zeroclaw_runtime::i18n::get_required_cli_string("goal-mode-completed")
        }
        GoalExecutionNotice::PausedForBlocker { blocker_messages } => {
            let mut message =
                zeroclaw_runtime::i18n::get_required_cli_string("goal-mode-paused-blocked");
            if !blocker_messages.is_empty() {
                message.push('\n');
                message.push_str(&zeroclaw_runtime::i18n::get_required_cli_string(
                    "goal-mode-paused-blocker-heading",
                ));
                for blocker in blocker_messages {
                    message.push('\n');
                    message.push_str(&zeroclaw_runtime::i18n::get_required_cli_string_with_args(
                        "goal-mode-paused-notice-blocker",
                        &[("blocker", blocker.as_str())],
                    ));
                }
            }
            message.push('\n');
            message.push_str(&zeroclaw_runtime::i18n::get_required_cli_string(
                "goal-mode-paused-blocked-next",
            ));
            for action_key in [
                "goal-mode-paused-blocked-cancel",
                "goal-mode-paused-blocked-status",
            ] {
                message.push('\n');
                let action = zeroclaw_runtime::i18n::get_required_cli_string(action_key);
                message.push_str(&zeroclaw_runtime::i18n::get_required_cli_string_with_args(
                    "goal-mode-paused-notice-action",
                    &[("action", action.as_str())],
                ));
            }
            message
        }
        GoalExecutionNotice::PausedForInterruption => {
            let mut message = zeroclaw_runtime::i18n::get_required_cli_string(
                "goal-mode-paused-core-interruption",
            );
            message.push('\n');
            message.push_str(&zeroclaw_runtime::i18n::get_required_cli_string(
                "goal-mode-paused-core-interruption-next",
            ));
            message
        }
        GoalExecutionNotice::Failed {
            terminal_reason,
            terminal_provider,
            terminal_detail,
        } => {
            let mut message = zeroclaw_runtime::i18n::get_required_cli_string("goal-mode-failed");
            let reason = match terminal_reason {
                zeroclaw_runtime::goal_mode::GoalTerminalReason::VerifiedCompletion => {
                    "goal-mode-terminal-reason-verified-completion"
                }
                zeroclaw_runtime::goal_mode::GoalTerminalReason::AccountingOutcomeUnknown => {
                    "goal-mode-terminal-reason-accounting-outcome-unknown"
                }
                zeroclaw_runtime::goal_mode::GoalTerminalReason::AccountingMissingOrInvalid => {
                    "goal-mode-terminal-reason-accounting-missing-or-invalid"
                }
                zeroclaw_runtime::goal_mode::GoalTerminalReason::PricingUnavailable => {
                    "goal-mode-terminal-reason-pricing-unavailable"
                }
                zeroclaw_runtime::goal_mode::GoalTerminalReason::CandidateEmpty => {
                    "goal-mode-terminal-reason-candidate-empty"
                }
                zeroclaw_runtime::goal_mode::GoalTerminalReason::ParentOperationFailed => {
                    "goal-mode-terminal-reason-parent-operation-failed"
                }
                zeroclaw_runtime::goal_mode::GoalTerminalReason::ParentContextWindowExceeded => {
                    "goal-mode-terminal-reason-parent-context-window-exceeded"
                }
                zeroclaw_runtime::goal_mode::GoalTerminalReason::VerifierOperationFailed => {
                    "goal-mode-terminal-reason-verifier-operation-failed"
                }
                zeroclaw_runtime::goal_mode::GoalTerminalReason::VerifierProtocolInvalid => {
                    "goal-mode-terminal-reason-verifier-protocol-invalid"
                }
                zeroclaw_runtime::goal_mode::GoalTerminalReason::ExecutorFailed
                | zeroclaw_runtime::goal_mode::GoalTerminalReason::ExecutorStartFailed => {
                    "goal-mode-terminal-reason-executor-failed"
                }
                zeroclaw_runtime::goal_mode::GoalTerminalReason::InitialNoticeFailed => {
                    "goal-mode-terminal-reason-initial-notice-failed"
                }
                zeroclaw_runtime::goal_mode::GoalTerminalReason::GoalToolPairingIncomplete => {
                    "goal-mode-terminal-reason-tool-pairing-incomplete"
                }
                zeroclaw_runtime::goal_mode::GoalTerminalReason::GoalToolLoopSafetyLimit => {
                    "goal-mode-terminal-reason-tool-loop-safety-limit"
                }
                zeroclaw_runtime::goal_mode::GoalTerminalReason::PolicyRevoked => {
                    "goal-mode-terminal-reason-policy-revoked"
                }
                zeroclaw_runtime::goal_mode::GoalTerminalReason::SessionDisposed => {
                    "goal-mode-terminal-reason-session-disposed"
                }
                zeroclaw_runtime::goal_mode::GoalTerminalReason::Unspecified => {
                    "goal-mode-terminal-reason-unspecified"
                }
            };
            let reason = zeroclaw_runtime::i18n::get_required_cli_string(reason);
            message.push('\n');
            message.push_str(&zeroclaw_runtime::i18n::get_required_cli_string_with_args(
                "goal-mode-summary-reason",
                &[("reason", reason.as_str())],
            ));
            if let Some(provider) = terminal_provider.as_deref() {
                message.push('\n');
                message.push_str(&zeroclaw_runtime::i18n::get_required_cli_string_with_args(
                    "goal-mode-summary-provider",
                    &[("provider", provider)],
                ));
            }
            if let Some(detail) = terminal_detail.as_deref()
                && !detail.trim().is_empty()
            {
                message.push('\n');
                message.push_str(&zeroclaw_runtime::i18n::get_required_cli_string_with_args(
                    "goal-mode-summary-details",
                    &[("details", detail)],
                ));
            }
            message
        }
    }
}

/// Builds the same memory scope as an ordinary Matrix turn. Goal Mode owns the
/// directive and accounting, but not the conversation's recall boundaries.
fn goal_memory_sessions(
    message: &zeroclaw_api::channel::ChannelMessage,
    session_id: String,
) -> Vec<Option<String>> {
    let mut sessions: Vec<Option<String>> = super::sender_memory_session_ids(message, &session_id)
        .into_iter()
        .map(Some)
        .collect();
    if super::is_group_reply_target(&message.reply_target) {
        sessions.push(Some(session_id));
    }
    sessions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goal_memory_scope_matches_an_ordinary_matrix_group_turn() {
        let message = ChannelMessage::new(
            "event",
            "@user:example.test",
            "group:engineering",
            "goal",
            "matrix",
            0,
        );

        assert_eq!(
            goal_memory_sessions(&message, "matrix:group:engineering:user".to_owned()),
            vec![
                Some("_user_example_test".to_owned()),
                Some("matrix:group:engineering:user".to_owned()),
            ]
        );
    }

    #[test]
    fn goal_parent_history_uses_ordinary_matrix_cache_cleanup() {
        let history = super::super::prepare_cached_channel_history(vec![
            ChatMessage::user("task\n<tool_result>stale result</tool_result>"),
            ChatMessage::assistant("[Used tools: file_read]\ncompleted report"),
        ]);

        assert_eq!(history.len(), 2);
        assert_eq!(history[0].content, "task");
        assert_eq!(history[1].content, "completed report");
    }

    struct GoalPresentationChannel {
        events: tokio::sync::Mutex<Vec<String>>,
        final_cancellations: tokio::sync::Mutex<Vec<bool>>,
        fail_final: bool,
    }

    impl GoalPresentationChannel {
        fn new(fail_final: bool) -> Self {
            Self {
                events: tokio::sync::Mutex::new(Vec::new()),
                final_cancellations: tokio::sync::Mutex::new(Vec::new()),
                fail_final,
            }
        }
    }

    impl zeroclaw_api::attribution::Attributable for GoalPresentationChannel {
        fn role(&self) -> zeroclaw_api::attribution::Role {
            zeroclaw_api::attribution::Role::Channel(zeroclaw_api::attribution::ChannelKind::Matrix)
        }

        fn alias(&self) -> &str {
            "goal-presentation"
        }
    }

    struct CancelOutboundGoalParentHook;

    #[async_trait]
    impl zeroclaw_api::hook::HookHandler for CancelOutboundGoalParentHook {
        fn name(&self) -> &str {
            "cancel-goal-parent-output"
        }

        async fn on_message_sending(
            &self,
            _channel: String,
            _recipient: String,
            _content: String,
        ) -> zeroclaw_api::hook::HookResult<(String, String, String)> {
            zeroclaw_api::hook::HookResult::Cancel("suppressed by test hook".to_owned())
        }
    }

    struct RewriteOutboundGoalParentHook;

    #[async_trait]
    impl zeroclaw_api::hook::HookHandler for RewriteOutboundGoalParentHook {
        fn name(&self) -> &str {
            "rewrite-goal-parent-output"
        }

        async fn on_message_sending(
            &self,
            channel: String,
            recipient: String,
            _content: String,
        ) -> zeroclaw_api::hook::HookResult<(String, String, String)> {
            zeroclaw_api::hook::HookResult::Continue((
                channel,
                recipient,
                "hook-rewritten parent report".to_owned(),
            ))
        }
    }

    #[async_trait]
    impl Channel for GoalPresentationChannel {
        fn name(&self) -> &str {
            "goal-presentation"
        }

        async fn send(&self, message: &zeroclaw_api::channel::SendMessage) -> Result<()> {
            self.events
                .lock()
                .await
                .push(format!("send:{}", message.content));
            Ok(())
        }

        async fn send_final(&self, message: &zeroclaw_api::channel::SendMessage) -> Result<()> {
            self.events
                .lock()
                .await
                .push(format!("final:{}", message.content));
            self.final_cancellations
                .lock()
                .await
                .push(message.cancellation_token.is_some());
            if self.fail_final {
                bail!("test delivery failure");
            }
            Ok(())
        }

        async fn start_typing(&self, recipient: &str) -> Result<()> {
            self.events
                .lock()
                .await
                .push(format!("typing:start:{recipient}"));
            Ok(())
        }

        async fn stop_typing(&self, recipient: &str) -> Result<()> {
            self.events
                .lock()
                .await
                .push(format!("typing:stop:{recipient}"));
            Ok(())
        }

        fn supports_draft_updates(&self) -> bool {
            true
        }

        async fn send_draft(
            &self,
            _message: &zeroclaw_api::channel::SendMessage,
        ) -> Result<Option<String>> {
            Ok(Some("goal-draft".to_owned()))
        }

        async fn listen(
            &self,
            _tx: tokio::sync::mpsc::Sender<zeroclaw_api::channel::ChannelMessage>,
        ) -> Result<()> {
            Ok(())
        }

        async fn finalize_draft(
            &self,
            recipient: &str,
            draft_id: &str,
            text: &str,
            _suppress_voice: bool,
        ) -> Result<()> {
            self.events
                .lock()
                .await
                .push(format!("draft:{recipient}:{draft_id}:{text}"));
            Ok(())
        }

        async fn cancel_draft(&self, recipient: &str, draft_id: &str) -> Result<()> {
            self.events
                .lock()
                .await
                .push(format!("cancel:{recipient}:{draft_id}"));
            Ok(())
        }
    }

    #[test]
    fn blocked_notice_includes_the_verifier_blocker() {
        let rendered = goal_notice_message(GoalExecutionNotice::PausedForBlocker {
            blocker_messages: vec!["Provide the task packet reference.".to_owned()],
        });

        assert!(rendered.starts_with("⏸️ Goal paused."));
        assert!(rendered.contains("\n**Blocker:**\n• Provide the task packet reference."));
        assert!(!rendered.contains("verifier requires resolution"));
        assert!(rendered.contains(
            "\n**Next:** Resolve the blocker, then run `/goal resume [RESPONSE]` to continue."
        ));
    }

    #[tokio::test]
    async fn goal_parent_single_message_streaming_uses_the_ordinary_typing_scope() {
        let channel = Arc::new(GoalPresentationChannel::new(false));
        let channel_handle: Arc<dyn Channel> = channel.clone();
        let mut context = super::super::tests::router_test_ctx_with_hooks(None);
        let context_mut = Arc::get_mut(&mut context).expect("test context is unshared");
        context_mut.channels_by_name = Arc::new(std::collections::HashMap::from([(
            "matrix.single".to_owned(),
            channel_handle,
        )]));
        let mut config = zeroclaw_config::schema::Config::default();
        config.channels.matrix.insert(
            "single".to_owned(),
            zeroclaw_config::schema::MatrixConfig {
                stream_mode: zeroclaw_config::schema::MatrixStreamMode::SingleMessage,
                ..Default::default()
            },
        );
        context_mut.prompt_config = Arc::new(config);
        let message = ChannelMessage {
            id: "event".to_owned(),
            sender: "@user:example.test".to_owned(),
            reply_target: "!room:test".to_owned(),
            channel: "matrix".to_owned(),
            channel_alias: Some("single".to_owned()),
            ..Default::default()
        };

        let presentation = GoalParentPresentation::start(context, &message).await;
        assert!(presentation.matrix_single_message_typing_scope.is_some());
        let _ = presentation.finish(None, None, None, None, None).await;

        let events = channel.events.lock().await;
        assert!(
            events
                .iter()
                .any(|event| event == "typing:start:!room:test"),
            "Goal single-message streaming must start the ordinary typing scope: {events:?}"
        );
        assert!(
            events.iter().any(|event| event == "typing:stop:!room:test"),
            "Goal single-message streaming must stop the ordinary typing scope: {events:?}"
        );
    }

    #[test]
    fn goal_core_errors_use_the_ordinary_error_surface_and_suppress_voice() {
        let message = ChannelMessage::new(
            "event",
            "@user:example.test",
            "!room:test",
            "goal",
            "matrix",
            0,
        );
        let error = anyhow::anyhow!("provider request failed (429)");

        let reply = goal_core_error_reply(&message, &error);

        assert!(reply.suppress_voice);
        assert_eq!(reply.recipient, "!room:test");
        assert_eq!(
            reply.content,
            super::super::channel_user_error_message(&error, "provider request failed (429)")
        );
    }

    #[test]
    fn blocked_notice_without_blocker_details_stays_compact() {
        let rendered = goal_notice_message(GoalExecutionNotice::PausedForBlocker {
            blocker_messages: Vec::new(),
        });

        assert!(rendered.starts_with("⏸️ Goal paused."));
        assert!(!rendered.contains("**Blocker:**"));
        assert!(rendered.contains("\n**Next:**"));
    }

    #[test]
    fn paired_safety_interruption_notice_is_not_mislabelled_as_a_blocker() {
        let rendered = goal_notice_message(GoalExecutionNotice::PausedForInterruption);

        assert!(rendered.starts_with("⏸️ Goal paused after a recoverable agent interruption."));
        assert!(rendered.contains("**Next:** Run `/goal resume`"));
        assert!(!rendered.contains("**Blocker:**"));
    }

    #[test]
    fn blocked_notice_renders_each_blocker_as_a_uniform_localized_item() {
        let rendered = goal_notice_message(GoalExecutionNotice::PausedForBlocker {
            blocker_messages: vec![
                "Provide the task packet reference.".to_owned(),
                "State its scope.".to_owned(),
            ],
        });

        assert!(
            rendered.contains(
                "\n**Blocker:**\n• Provide the task packet reference.\n• State its scope."
            )
        );
    }

    #[test]
    fn failed_notice_includes_the_safe_terminal_reason() {
        let rendered = goal_notice_message(GoalExecutionNotice::Failed {
            terminal_reason:
                zeroclaw_runtime::goal_mode::GoalTerminalReason::GoalToolPairingIncomplete,
            terminal_provider: None,
            terminal_detail: None,
        });

        assert_eq!(
            rendered,
            "❌ Goal failed.\n**Reason:** A tool operation did not settle cleanly, so the Goal stopped to avoid an unsafe retry."
        );
    }

    #[test]
    fn failed_notice_includes_the_safe_provider_profile() {
        let rendered = goal_notice_message(GoalExecutionNotice::Failed {
            terminal_reason: zeroclaw_runtime::goal_mode::GoalTerminalReason::ParentOperationFailed,
            terminal_provider: Some("openai.default".to_owned()),
            terminal_detail: None,
        });

        assert_eq!(
            rendered,
            "❌ Goal failed.\n**Reason:** The agent's model operation failed before it produced a verified result.\n**Provider:** openai.default"
        );
    }

    #[test]
    fn failed_notice_explains_a_context_window_rejection() {
        let rendered = goal_notice_message(GoalExecutionNotice::Failed {
            terminal_reason:
                zeroclaw_runtime::goal_mode::GoalTerminalReason::ParentContextWindowExceeded,
            terminal_provider: Some("openai.default".to_owned()),
            terminal_detail: None,
        });

        assert_eq!(
            rendered,
            "❌ Goal failed.\n**Reason:** The selected model could not accept the current conversation because it exceeds that model's context window.\n**Provider:** openai.default"
        );
    }

    #[test]
    fn failed_notice_keeps_the_sanitized_causal_diagnostic() {
        let rendered = goal_notice_message(GoalExecutionNotice::Failed {
            terminal_reason: zeroclaw_runtime::goal_mode::GoalTerminalReason::ExecutorFailed,
            terminal_provider: None,
            terminal_detail: Some("agent loop: model request rejected (429)".to_owned()),
        });

        assert_eq!(
            rendered,
            "❌ Goal failed.\n**Reason:** The Goal worker could not continue.\n**Details:** agent loop: model request rejected (429)"
        );
    }

    #[test]
    fn command_error_classifies_a_conflicting_resident_ledger() {
        let rendered = render_matrix_goal_command_error(&anyhow::anyhow!(
            "required cost tracker storage path differs from the resident tracker"
        ));

        assert_eq!(
            rendered,
            "⚠️ Goal accounting cannot start because this process is using a different ledger. Ask an operator to align the configured data directory, then try again."
        );
    }

    #[test]
    fn start_history_combines_goal_directive_into_the_system_prompt() {
        let history = goal_start_history(
            "system prompt".to_owned(),
            ChatMessage::system("Goal directive"),
            vec![ChatMessage::assistant("earlier assistant response")],
            "## Session Prompts\n- id: \"task\"; content: \"persistent task\"\n",
            0,
        )
        .expect("Goal parent prompt should include its session task context");

        assert_eq!(history.len(), 3);
        assert_eq!(history[0].role, "system");
        assert!(history[0].content.contains("system prompt"));
        assert!(history[0].content.contains("Goal directive"));
        assert!(history[0].content.contains("persistent task"));
        assert!(
            history[0].content.rfind("## Session Prompts").unwrap()
                > history[0].content.find("Goal directive").unwrap(),
            "session attachments must remain the final host-owned section"
        );
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
            "## Session Prompts\n- id: \"task\"; content: \"current task\"\n",
            0,
        )
        .expect("continuation with a system prompt should build");

        assert_eq!(history[0].role, "system");
        assert!(history[0].content.contains("rebuilt system prompt"));
        assert!(history[0].content.contains("continue Goal directive"));
        assert!(history[0].content.contains("current task"));
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
    fn blocked_resume_replaces_the_system_head_without_inserting_a_new_request() {
        let mut history = goal_resume_history(
            "rebuilt system prompt".to_owned(),
            ChatMessage::system("resume Goal directive"),
            vec![
                ChatMessage::system("old system prompt\n\nstart Goal directive"),
                ChatMessage::user("original task"),
                ChatMessage::assistant("What exact target should I change?"),
            ],
            "## Session Prompts\n- id: \"task\"; content: \"current task\"\n",
            0,
            false,
        )
        .expect("blocked resume with a live transcript should build");
        history.push(ChatMessage::user("Change the authentication module."));

        assert_eq!(history[0].role, "system");
        assert!(history[0].content.contains("resume Goal directive"));
        assert!(!history[0].content.contains("start Goal directive"));
        assert_eq!(
            history
                .iter()
                .filter(|message| message.role == "system")
                .count(),
            1
        );
        assert_eq!(
            history[history.len() - 2].content,
            "What exact target should I change?"
        );
        assert_eq!(
            history.last().map(|message| message.content.as_str()),
            Some("Change the authentication module.")
        );
        assert!(history.iter().all(|message| {
            !message
                .content
                .contains("Proceed with the Goal work under the trusted runtime directive.")
        }));
    }

    #[test]
    fn bare_blocked_resume_ends_in_the_execution_request() {
        let history = goal_resume_history(
            "rebuilt system prompt".to_owned(),
            ChatMessage::system("resume Goal directive"),
            vec![
                ChatMessage::system("old system prompt"),
                ChatMessage::user("original task"),
                ChatMessage::assistant("What exact target should I change?"),
            ],
            "",
            0,
            true,
        )
        .expect("bare blocked resume should build");

        assert_eq!(
            history.last().map(|message| message.role.as_str()),
            Some("user")
        );
        assert_eq!(
            history.last().map(|message| message.content.as_str()),
            Some("Proceed with the Goal work under the trusted runtime directive.")
        );
    }

    #[test]
    fn matrix_continuation_dispatch_preserves_each_resume_tail() {
        let retained = || {
            vec![
                ChatMessage::system("old Goal system prompt"),
                ChatMessage::user("original task"),
                ChatMessage::assistant("Which module should I change?"),
            ]
        };
        let rebuild = |turn| {
            goal_continuation_history_for_parent_turn(
                turn,
                "rebuilt system prompt".to_owned(),
                ChatMessage::system("resume Goal directive"),
                "",
                0,
            )
            .expect("Matrix continuation dispatch should build")
        };

        let with_reply = rebuild(GoalParentTurn {
            kind: GoalParentTurnKind::Resume,
            objective: "finish the task".to_owned(),
            resume_response: Some("Use the authentication module.".to_owned()),
            paused_request: None,
            history_source: zeroclaw_runtime::goal_mode::GoalParentHistorySource::Continuation,
            working_history: retained(),
        });
        assert_eq!(
            with_reply.last().map(|message| message.content.as_str()),
            Some("Use the authentication module.")
        );

        let bare = rebuild(GoalParentTurn {
            kind: GoalParentTurnKind::Resume,
            objective: "finish the task".to_owned(),
            resume_response: None,
            paused_request: None,
            history_source: zeroclaw_runtime::goal_mode::GoalParentHistorySource::Continuation,
            working_history: retained(),
        });
        assert_eq!(
            bare.last().map(|message| message.role.as_str()),
            Some("user")
        );
        assert_eq!(
            bare.last().map(|message| message.content.as_str()),
            Some("Proceed with the Goal work under the trusted runtime directive.")
        );

        let continued = rebuild(GoalParentTurn {
            kind: GoalParentTurnKind::Continue,
            objective: "finish the task".to_owned(),
            resume_response: None,
            paused_request: None,
            history_source: zeroclaw_runtime::goal_mode::GoalParentHistorySource::Continuation,
            working_history: retained(),
        });
        assert_eq!(
            continued.last().map(|message| message.content.as_str()),
            Some("Proceed with the Goal work under the trusted runtime directive.")
        );
    }

    #[test]
    fn matrix_canonical_resume_preserves_the_user_response() {
        let history = goal_canonical_history_for_parent_turn(
            GoalParentTurn {
                kind: GoalParentTurnKind::Resume,
                objective: "finish the task".to_owned(),
                resume_response: Some("Use the migration documented in the task.".to_owned()),
                paused_request: None,
                history_source: zeroclaw_runtime::goal_mode::GoalParentHistorySource::Canonical,
                working_history: vec![
                    ChatMessage::user("original task"),
                    ChatMessage::assistant("Which migration should I use?"),
                ],
            },
            "rebuilt system prompt".to_owned(),
            ChatMessage::system("resume Goal directive"),
            "",
            0,
        )
        .expect("canonical resume should build");

        assert_eq!(
            history.last().map(|message| message.content.as_str()),
            Some("Use the migration documented in the task.")
        );
        assert!(history.iter().any(|message| {
            message.role == "user"
                && message.content
                    == "Proceed with the Goal work under the trusted runtime directive."
        }));
    }

    #[test]
    fn continuation_rejects_a_transcript_without_a_system_prompt() {
        let error = goal_continue_history(
            "rebuilt system prompt".to_owned(),
            ChatMessage::system("continue Goal directive"),
            vec![ChatMessage::assistant("earlier assistant response")],
            "",
            0,
        )
        .expect_err("continuation must retain the system prompt at history index zero");

        assert!(error.to_string().contains("lost its system prompt"));
    }

    #[test]
    fn model_switch_rebuilds_only_the_goal_system_prompt() {
        let directive = ChatMessage::system("Continue until the objective is met.");
        let mut history = vec![
            goal_parent_system_message_with_session_prompts(
                "old provider tools".to_owned(),
                directive.content.clone(),
                "saved session instructions",
                0,
            )
            .unwrap(),
            ChatMessage::user("Finish the migration."),
            ChatMessage::assistant("I will switch models for the next step."),
            ChatMessage::user("[Tool results]\nmodel switch accepted"),
        ];
        let retained_tail: Vec<_> = history[1..]
            .iter()
            .map(|message| (message.role.clone(), message.content.clone()))
            .collect();

        refresh_goal_system_prompt_after_model_switch(
            &mut history,
            "new provider tools".to_owned(),
            &directive,
            "saved session instructions",
            0,
        )
        .unwrap();

        assert!(history[0].content.contains("new provider tools"));
        assert!(history[0].content.contains(&directive.content));
        assert!(history[0].content.contains("saved session instructions"));
        assert_eq!(
            history[1..]
                .iter()
                .map(|message| (message.role.clone(), message.content.clone()))
                .collect::<Vec<_>>(),
            retained_tail
        );
    }

    #[tokio::test]
    async fn goal_parent_candidate_uses_the_normal_draft_finalization_surface() {
        let channel = GoalPresentationChannel::new(false);
        let message = ChannelMessage::new(
            "event",
            "@user:example.test",
            "!room:test",
            "goal",
            "matrix",
            0,
        );

        assert!(
            deliver_goal_parent_response(
                &channel,
                &channel,
                &message,
                Some("draft-id"),
                "parent report",
                "!room:test",
                Some(true),
                false,
                None,
                false,
                None,
            )
            .await
        );
        assert_eq!(
            channel.events.lock().await.as_slice(),
            ["draft:!room:test:draft-id:parent report"]
        );
    }

    #[tokio::test]
    async fn goal_parent_candidate_keeps_history_when_its_channel_has_gone_away() {
        let message = ChannelMessage::new(
            "event",
            "@user:example.test",
            "!room:test",
            "goal",
            "matrix",
            0,
        );

        let history = GoalParentPresentation::present_parent_result(
            super::super::tests::router_test_ctx_with_hooks(None).as_ref(),
            &message,
            None,
            None,
            Some("parent report"),
            None,
            None,
            false,
            Arc::new(Mutex::new(Vec::new())),
            None,
            None,
        )
        .await;

        assert_eq!(history.as_deref(), Some("parent report"));
    }

    #[tokio::test]
    async fn goal_parent_candidate_honors_outbound_hook_cancellation() {
        let channel = Arc::new(GoalPresentationChannel::new(false));
        let message = ChannelMessage::new(
            "event",
            "@user:example.test",
            "!room:test",
            "goal",
            "matrix",
            0,
        );
        let mut hooks = zeroclaw_runtime::hooks::HookRunner::new();
        hooks.register(Box::new(CancelOutboundGoalParentHook));

        let history = GoalParentPresentation::present_parent_result(
            super::super::tests::router_test_ctx_with_hooks(Some(Arc::new(hooks))).as_ref(),
            &message,
            Some(&(channel.clone() as Arc<dyn Channel>)),
            None,
            Some("suppressed parent report"),
            None,
            None,
            false,
            Arc::new(Mutex::new(Vec::new())),
            None,
            None,
        )
        .await;

        assert!(history.is_none());
        assert!(channel.events.lock().await.is_empty());
    }

    #[tokio::test]
    async fn goal_parent_candidate_retains_hook_rewritten_history() {
        let channel = Arc::new(GoalPresentationChannel::new(false));
        let message = ChannelMessage::new(
            "event",
            "@user:example.test",
            "!room:test",
            "goal",
            "matrix",
            0,
        );
        let mut hooks = zeroclaw_runtime::hooks::HookRunner::new();
        hooks.register(Box::new(RewriteOutboundGoalParentHook));

        let history = GoalParentPresentation::present_parent_result(
            super::super::tests::router_test_ctx_with_hooks(Some(Arc::new(hooks))).as_ref(),
            &message,
            Some(&(channel.clone() as Arc<dyn Channel>)),
            None,
            Some("raw parent report"),
            None,
            None,
            false,
            Arc::new(Mutex::new(Vec::new())),
            None,
            None,
        )
        .await;

        assert_eq!(history.as_deref(), Some("hook-rewritten parent report"));
        assert_eq!(
            channel.events.lock().await.as_slice(),
            ["final:hook-rewritten parent report"]
        );
    }

    #[tokio::test]
    async fn goal_parent_candidate_keeps_recovery_footer_out_of_history() {
        let channel = Arc::new(GoalPresentationChannel::new(false));
        let message = ChannelMessage::new(
            "event",
            "@user:example.test",
            "!room:test",
            "goal",
            "matrix",
            0,
        );
        let fallback = ProviderFallbackInfo {
            requested_provider: "openai.primary".to_owned(),
            requested_model: "model-a".to_owned(),
            actual_provider: "anthropic.backup".to_owned(),
            actual_model: "model-b".to_owned(),
        };

        let history = GoalParentPresentation::present_parent_result(
            super::super::tests::router_test_ctx_with_hooks(None).as_ref(),
            &message,
            Some(&(channel.clone() as Arc<dyn Channel>)),
            None,
            Some("accepted parent report"),
            Some(&fallback),
            None,
            false,
            Arc::new(Mutex::new(Vec::new())),
            None,
            None,
        )
        .await;

        assert_eq!(history.as_deref(), Some("accepted parent report"));
        let events = channel.events.lock().await;
        assert_eq!(events.len(), 1);
        assert!(events[0].starts_with("final:accepted parent report\n\n---\n"));
        assert!(events[0].contains("openai.primary"));
        assert!(events[0].contains("anthropic.backup"));
    }

    #[tokio::test]
    async fn goal_parent_candidate_keeps_history_when_delivery_fails() {
        let channel = Arc::new(GoalPresentationChannel::new(true));
        let message = ChannelMessage::new(
            "event",
            "@user:example.test",
            "!room:test",
            "goal",
            "matrix",
            0,
        );

        let history = GoalParentPresentation::present_parent_result(
            super::super::tests::router_test_ctx_with_hooks(None).as_ref(),
            &message,
            Some(&(channel.clone() as Arc<dyn Channel>)),
            None,
            Some("durable parent report"),
            None,
            None,
            false,
            Arc::new(Mutex::new(Vec::new())),
            None,
            None,
        )
        .await;

        assert_eq!(history.as_deref(), Some("durable parent report"));
        assert_eq!(
            channel.events.lock().await.as_slice(),
            ["final:durable parent report"]
        );
    }

    #[tokio::test]
    async fn goal_parent_candidate_without_a_draft_uses_the_normal_final_surface() {
        let channel = GoalPresentationChannel::new(false);
        let message = ChannelMessage::new(
            "event",
            "@user:example.test",
            "!room:test",
            "goal",
            "matrix",
            0,
        );

        assert!(
            deliver_goal_parent_response(
                &channel,
                &channel,
                &message,
                None,
                "parent report",
                "!room:test",
                None,
                false,
                None,
                false,
                None,
            )
            .await
        );
        assert_eq!(
            channel.events.lock().await.as_slice(),
            ["final:parent report"]
        );
    }

    #[tokio::test]
    async fn goal_parent_candidate_uses_the_active_cancellation_token_for_final_delivery() {
        let channel = GoalPresentationChannel::new(false);
        let message = ChannelMessage::new(
            "event",
            "@user:example.test",
            "!room:test",
            "goal",
            "matrix",
            0,
        );
        let cancellation = tokio_util::sync::CancellationToken::new();

        assert!(
            deliver_goal_parent_response(
                &channel,
                &channel,
                &message,
                None,
                "parent report",
                "!room:test",
                None,
                false,
                None,
                false,
                Some(&cancellation),
            )
            .await
        );
        assert_eq!(channel.final_cancellations.lock().await.as_slice(), [true]);
    }

    #[tokio::test]
    async fn goal_parent_candidate_honors_send_via_delivery() {
        let origin = GoalPresentationChannel::new(false);
        let destination = GoalPresentationChannel::new(false);
        let message = ChannelMessage::new(
            "event",
            "@user:example.test",
            "!room:test",
            "goal",
            "matrix",
            0,
        );

        assert!(
            deliver_goal_parent_response(
                &origin,
                &destination,
                &message,
                Some("draft-id"),
                "parent report",
                "destination",
                Some(true),
                false,
                None,
                true,
                None,
            )
            .await
        );
        assert_eq!(
            origin.events.lock().await.as_slice(),
            ["cancel:!room:test:draft-id"]
        );
        assert_eq!(
            destination.events.lock().await.as_slice(),
            ["final:parent report"]
        );
    }
}
