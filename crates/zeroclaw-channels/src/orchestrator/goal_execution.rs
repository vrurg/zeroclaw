//! Matrix-specific trusted binding helpers for the Goal session adapter.
//!
//! Goal lifecycle and execution stay in `zeroclaw-runtime`; this module only
//! reconstructs the Matrix facts that must agree with durable task state.

use std::sync::{Arc, Weak};

use anyhow::{Context as _, Result, bail};
use async_trait::async_trait;
use tokio::sync::OwnedMutexGuard;
use tokio_util::sync::CancellationToken;
use zeroclaw_api::session_keys::sanitize_session_key;
use zeroclaw_runtime::{
    agent::loop_::{
        LoopKnobs, ResolvedAgentExecution, ResolvedIo, ResolvedModelAccess, ResolvedRuntimeKnobs,
        ToolLoop, run_tool_call_loop, scope_session_key,
    },
    goal_mode::{
        GoalCommandErrorKey, GoalExecutionPreflight, GoalExecutionScope, GoalMessageKey,
        GoalOperationReservation, GoalResponse, GoalRuntime,
        execution::{GoalExecutionDriver, preflight_reservation, run_isolated_verifier},
        host::{
            GoalExecutionHost, GoalHostModelBinding, GoalHostPreflightError, GoalParentTurn,
            GoalSessionAdapter, GoalSessionBinding, GoalSessionLease, GoalVerifiedCandidate,
            GoalVerifierTurn,
        },
    },
};

use super::{
    CHANNEL_MESSAGE_TIMEOUT_SCALE_CAP, ChannelRouteSelection, ChannelRuntimeContext,
    ChannelRuntimeDefaultsSnapshot, channel_message_timeout_budget_secs_with_cap,
    composite_channel_key, conversation_history_key, foreground_lock, get_or_create_provider,
    get_route_selection, normalize_cached_channel_turns, resolve_channel_thinking,
    runtime_defaults_snapshot, sanitize_verified_goal_delivery, system_prompt_for_channel_turn,
    wait_for_foreground_lease,
};

const MATRIX_ROUTE_PREFIX: &str = "matrix:";

fn route_is_owned_by_agent(
    config: &zeroclaw_config::schema::Config,
    channel_key: &str,
    agent: &str,
) -> bool {
    config.agent_for_channel(channel_key) == Some(agent)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MatrixGoalRoute {
    alias: String,
    room_id: String,
}

impl MatrixGoalRoute {
    fn parse(route: &str) -> Result<Self> {
        let Some(rest) = route.strip_prefix(MATRIX_ROUTE_PREFIX) else {
            bail!("Goal route is not Matrix");
        };
        let Some((alias, room_id)) = rest.split_once(':') else {
            bail!("Matrix Goal route is malformed");
        };
        if alias.trim().is_empty()
            || alias.trim() != alias
            || alias.contains('\n')
            || room_id.trim().is_empty()
            || room_id.trim() != room_id
            || room_id.contains('\n')
        {
            bail!("Matrix Goal route has blank or unsafe fields");
        }
        Ok(Self {
            alias: alias.into(),
            room_id: room_id.into(),
        })
    }

    fn canonical_session_id(&self, raw_mxid: &str) -> Result<String> {
        if raw_mxid.trim().is_empty() || raw_mxid.contains('\n') {
            bail!("Matrix Goal principal is blank or unsafe");
        }
        Ok(sanitize_session_key(&format!(
            "matrix.{}_{}_{}",
            self.alias, self.room_id, raw_mxid
        )))
    }

    fn matches_binding(&self, binding: &GoalSessionBinding, expected_agent: &str) -> bool {
        binding.agent() == expected_agent
            && binding.route() == format!("{MATRIX_ROUTE_PREFIX}{}:{}", self.alias, self.room_id)
            && binding
                .principal_id()
                .and_then(|raw_mxid| self.canonical_session_id(raw_mxid).ok())
                .as_deref()
                == Some(binding.session_id())
    }
}

/// Matrix implementation of the transport-neutral Goal live-session contract.
///
/// It owns no Goal lifecycle or accounting state. The existing channel runtime
/// remains the source of the configured provider, tool policy, history, and
/// foreground lock; the generic Goal runtime owns durable control-plane work.
struct MatrixGoalSessionAdapter {
    context: Weak<ChannelRuntimeContext>,
}

impl MatrixGoalSessionAdapter {
    fn new(context: Weak<ChannelRuntimeContext>) -> Self {
        Self { context }
    }

    fn route_for_binding(
        &self,
        binding: &GoalSessionBinding,
    ) -> Result<(Arc<ChannelRuntimeContext>, MatrixGoalRoute)> {
        let context = self
            .context
            .upgrade()
            .context("Matrix Goal runtime context is no longer available")?;
        let route = MatrixGoalRoute::parse(binding.route())?;
        if !route.matches_binding(binding, context.agent_alias.as_str()) {
            bail!("Matrix Goal binding does not match the live agent or session");
        }
        let channel_key = composite_channel_key("matrix", Some(route.alias.as_str()));
        if !context.channels_by_name.contains_key(&channel_key) {
            bail!("Matrix Goal channel alias is no longer available");
        }
        let defaults = runtime_defaults_snapshot(context.as_ref());
        if !route_is_owned_by_agent(
            defaults.config.as_ref(),
            &channel_key,
            context.agent_alias.as_str(),
        ) {
            bail!("Matrix Goal channel alias is not owned by the live agent");
        }
        Ok((context, route))
    }

    fn session_message(
        binding: &GoalSessionBinding,
        route: &MatrixGoalRoute,
    ) -> Result<zeroclaw_api::channel::ChannelMessage> {
        Ok(zeroclaw_api::channel::ChannelMessage {
            sender: binding
                .principal_id()
                .context("Matrix Goal binding has no raw MXID")?
                .to_owned(),
            reply_target: route.room_id.clone(),
            channel: "matrix".into(),
            channel_alias: Some(route.alias.clone()),
            ..Default::default()
        })
    }

    fn effective_route(
        context: &ChannelRuntimeContext,
        binding: &GoalSessionBinding,
        route: &MatrixGoalRoute,
        defaults: &ChannelRuntimeDefaultsSnapshot,
    ) -> Result<ChannelRouteSelection> {
        let message = Self::session_message(binding, route)?;
        Ok(get_route_selection(
            context,
            &message,
            binding.session_id(),
            defaults,
        ))
    }
}

struct MatrixGoalSessionLease {
    _foreground_guard: OwnedMutexGuard<()>,
    context: Arc<ChannelRuntimeContext>,
    binding: GoalSessionBinding,
    route: MatrixGoalRoute,
}

impl MatrixGoalSessionLease {
    fn current_route(&self) -> Result<(ChannelRuntimeDefaultsSnapshot, ChannelRouteSelection)> {
        let defaults = runtime_defaults_snapshot(self.context.as_ref());
        let route = MatrixGoalSessionAdapter::effective_route(
            self.context.as_ref(),
            &self.binding,
            &self.route,
            &defaults,
        )?;
        Ok((defaults, route))
    }
}

#[async_trait]
impl GoalSessionAdapter for MatrixGoalSessionAdapter {
    fn owns(&self, binding: &GoalSessionBinding) -> bool {
        self.route_for_binding(binding).is_ok()
    }

    async fn preflight(
        &self,
        binding: &GoalSessionBinding,
        request: &GoalExecutionPreflight,
    ) -> std::result::Result<GoalOperationReservation, GoalHostPreflightError> {
        let (context, route) = self
            .route_for_binding(binding)
            .map_err(|_| GoalHostPreflightError::BindingUnavailable)?;
        let defaults = runtime_defaults_snapshot(context.as_ref());
        let selection = Self::effective_route(context.as_ref(), binding, &route, &defaults)
            .map_err(|_| GoalHostPreflightError::BindingUnavailable)?;
        zeroclaw_runtime::agent::agent::build_session_model_provider(
            defaults.config.as_ref(),
            &defaults.config.goal.verifier.provider,
            defaults.config.goal.verifier.model.as_deref(),
        )
        .map_err(|_| GoalHostPreflightError::VerifierUnavailable)?;
        preflight_reservation(
            defaults.config.as_ref(),
            request,
            &GoalHostModelBinding {
                provider_ref: selection.model_provider,
                model: selection.model,
            },
        )
        .map_err(|_| {
            if request.cost_limit_usd.is_some() {
                GoalHostPreflightError::PricingUnavailable
            } else {
                GoalHostPreflightError::ReservationUnavailable
            }
        })
    }

    async fn model_binding(&self, binding: &GoalSessionBinding) -> Result<GoalHostModelBinding> {
        let (context, route) = self.route_for_binding(binding)?;
        let defaults = runtime_defaults_snapshot(context.as_ref());
        let selection = Self::effective_route(context.as_ref(), binding, &route, &defaults)?;
        Ok(GoalHostModelBinding {
            provider_ref: selection.model_provider,
            model: selection.model,
        })
    }

    async fn deadline(&self, binding: &GoalSessionBinding) -> Result<Option<tokio::time::Instant>> {
        let (context, _route) = self.route_for_binding(binding)?;
        let timeout = channel_message_timeout_budget_secs_with_cap(
            context.message_timeout_secs,
            context.max_tool_iterations,
            CHANNEL_MESSAGE_TIMEOUT_SCALE_CAP,
        );
        Ok(Some(
            tokio::time::Instant::now() + std::time::Duration::from_secs(timeout),
        ))
    }

    async fn acquire(
        &self,
        scope: &GoalExecutionScope,
        binding: &GoalSessionBinding,
    ) -> Result<Box<dyn GoalSessionLease>> {
        let (context, _route) = self.route_for_binding(binding)?;
        let lock = foreground_lock(&context.persist_locks, binding.session_id());
        let foreground_guard = wait_for_foreground_lease(lock, scope.cancellation.clone())
            .await
            .context("Matrix Goal foreground lease was cancelled")?;
        let (context, route) = self.route_for_binding(binding)?;
        Ok(Box::new(MatrixGoalSessionLease {
            _foreground_guard: foreground_guard,
            context,
            binding: binding.clone(),
            route,
        }))
    }
}

/// Build a current-config controller for one canonical Matrix session.
///
/// The process-local slot retains only executor supervision; the controller,
/// control plane, and ledger are reconstructed from their canonical owners.
pub(super) async fn runtime_for_matrix_session(
    context: &Arc<ChannelRuntimeContext>,
    session_id: &str,
) -> Result<Arc<GoalRuntime>> {
    let slot = super::goal_runtime_slot(&context.persist_locks, session_id);
    let defaults = runtime_defaults_snapshot(context.as_ref());
    let executor = slot
        .get_or_try_init(|| async {
            let control_plane = zeroclaw_runtime::control_plane::control_plane()
                .context("Goal control plane is unavailable for Matrix")?;
            let tracker = zeroclaw_runtime::cost::CostTracker::get_or_init_global_for_goal(
                defaults.config.cost.clone(),
                &defaults.config.data_dir,
            )
            .context("Goal cost ledger is unavailable for Matrix")?;
            let adapter: Arc<dyn GoalSessionAdapter> =
                Arc::new(MatrixGoalSessionAdapter::new(Arc::downgrade(context)));
            let host = Arc::new(GoalExecutionHost::new(vec![adapter]));
            Ok(Arc::new(GoalExecutionDriver::new(
                host,
                Arc::clone(&control_plane.goal_store),
                Arc::clone(&defaults.config),
                Arc::clone(&tracker),
            )))
        })
        .await?;
    // Existing workers have been fenced and joined before a reload replaces
    // this source. A new controller therefore observes the live config while
    // the persistent executor retains only worker supervision.
    executor
        .replace_config_after_quiescence(Arc::clone(&defaults.config))
        .await;
    let control_plane = zeroclaw_runtime::control_plane::control_plane()
        .context("Goal control plane is unavailable for Matrix")?;
    let tracker = zeroclaw_runtime::cost::CostTracker::get_or_init_global_for_goal(
        defaults.config.cost.clone(),
        &defaults.config.data_dir,
    )
    .context("Goal cost ledger is unavailable for Matrix")?;
    Ok(Arc::new(GoalRuntime::new(
        defaults.config,
        Arc::clone(&control_plane.goal_store),
        tracker,
        executor,
        &control_plane.boot_id,
    )?))
}

/// Fence every running Goal owned by this Matrix runtime before a validated
/// configuration replacement becomes visible. The task store, rather than the
/// process-local slot cache, supplies the complete affected-session set.
pub(super) async fn cut_over_matrix_goals_for_runtime_reload(
    context: &Arc<ChannelRuntimeContext>,
    next_config: &zeroclaw_config::schema::Config,
) -> Result<()> {
    let control_plane = zeroclaw_runtime::control_plane::control_plane()
        .context("Goal control plane is unavailable for Matrix reload")?;
    cut_over_matrix_goals_with_control_plane(context, next_config, control_plane).await
}

/// Apply the production Matrix cutover using its already-booted control plane.
/// Keeping the handle explicit here lets the file-backed integration test drive
/// the same transition and quiescence ordering without replacing global daemon
/// state inside a parallel unit-test process.
pub(super) async fn cut_over_matrix_goals_with_control_plane(
    context: &Arc<ChannelRuntimeContext>,
    next_config: &zeroclaw_config::schema::Config,
    control_plane: &zeroclaw_runtime::control_plane::ControlPlaneHandle,
) -> Result<()> {
    let goals = control_plane
        .goal_store
        .nonterminal_session_goals_for_agent(context.agent_alias.as_str())
        .await
        .context("list nonterminal Matrix Goals for reload")?;
    let mut affected = Vec::new();
    let mut transitions = Vec::new();
    for goal in goals {
        // The control plane is shared with other Goal transports. Agent
        // ownership alone is deliberately insufficient: this Matrix reload
        // may act only on rows whose durable route is an exact Matrix route.
        if !is_matrix_goal_route(&goal) {
            continue;
        }
        let session_id = goal
            .session_id
            .as_deref()
            .filter(|session_id| !session_id.trim().is_empty())
            .context("running Matrix Goal has no session binding")?;
        let action =
            if matrix_goal_policy_is_revoked(next_config, context.agent_alias.as_str(), &goal) {
                zeroclaw_runtime::control_plane::GoalPolicyCutoverAction::CancelForRevocation
            } else if goal.status == zeroclaw_runtime::control_plane::TaskStatus::Running {
                zeroclaw_runtime::control_plane::GoalPolicyCutoverAction::PauseForReload
            } else {
                continue;
            };
        transitions.push(
            zeroclaw_runtime::control_plane::GoalPolicyCutoverTransition {
                task_id: goal.id.clone(),
                session_id: session_id.to_owned(),
                execution_epoch: goal.execution_epoch,
                action,
            },
        );
        affected.push(goal);
    }
    let result = control_plane
        .goal_store
        .apply_goal_policy_cutover(&transitions)
        .await?;
    if result != zeroclaw_runtime::control_plane::GoalTransitionResult::Applied {
        anyhow::bail!("atomic Matrix Goal policy cutover became stale");
    }
    for goal in affected {
        let session_id = goal
            .session_id
            .as_deref()
            .context("cut-over Matrix Goal has no session binding")?;
        runtime_for_matrix_session(context, session_id)
            .await?
            .quiesce_exact_after_policy_cutover(&goal)
            .await
            .map_err(anyhow::Error::from)?;
    }
    Ok(())
}

fn matrix_goal_route(
    task: &zeroclaw_runtime::control_plane::TaskRecord,
) -> Option<MatrixGoalRoute> {
    task.originator_route
        .as_deref()
        .and_then(|route| MatrixGoalRoute::parse(route).ok())
}

fn is_matrix_goal_route(task: &zeroclaw_runtime::control_plane::TaskRecord) -> bool {
    task.originator_route
        .as_deref()
        .is_some_and(|route| route.starts_with(MATRIX_ROUTE_PREFIX))
}

/// Determine whether a prospective configuration revokes the durable Matrix
/// binding. This deliberately uses only task-owned binding facts and the
/// prospective configuration: the process-local session slot is never policy
/// authority.
fn matrix_goal_policy_is_revoked(
    next_config: &zeroclaw_config::schema::Config,
    expected_agent: &str,
    task: &zeroclaw_runtime::control_plane::TaskRecord,
) -> bool {
    if !next_config.goal.enabled || task.agent != expected_agent {
        return true;
    }
    let Some(route) = matrix_goal_route(task) else {
        return true;
    };
    let channel_key = composite_channel_key("matrix", Some(route.alias.as_str()));
    if !route_is_owned_by_agent(next_config, &channel_key, expected_agent) {
        return true;
    }
    if super::resolved_runtime_model_provider_ref(next_config, expected_agent).is_err() {
        return true;
    }
    zeroclaw_runtime::agent::agent::build_session_model_provider(
        next_config,
        &next_config.goal.verifier.provider,
        next_config.goal.verifier.model.as_deref(),
    )
    .is_err()
}

pub(super) fn is_matrix_goal_command(message: &zeroclaw_api::channel::ChannelMessage) -> bool {
    if message.channel != "matrix" {
        return false;
    }
    let command = message.content.trim_start();
    command
        .strip_prefix("/goal")
        .is_some_and(|rest| rest.is_empty() || rest.chars().next().is_some_and(char::is_whitespace))
}

/// Parse and submit an immutable Matrix command snapshot after the parent
/// orchestrator has refreshed policy under its admission/cutover gate. The
/// caller captures `message` before hooks run; hooks may cancel the outer
/// message but cannot manufacture a Goal command or retarget this binding.
pub(super) async fn submit_matrix_goal_command(
    context: &Arc<ChannelRuntimeContext>,
    message: &zeroclaw_api::channel::ChannelMessage,
    parsed: std::result::Result<
        zeroclaw_runtime::goal_mode::GoalCommand,
        zeroclaw_runtime::goal_mode::GoalCommandParseError,
    >,
) -> String {
    let response = async {
        let command = match parsed {
            Ok(command) => command,
            Err(error) => {
                return Ok(GoalResponse {
                    key: parse_error_key(error.key),
                    status: None,
                });
            }
        };
        let defaults = runtime_defaults_snapshot(context.as_ref());
        if let Some(response) = local_goal_response(&command, defaults.config.goal.enabled) {
            return Ok(response);
        }
        let alias = message
            .channel_alias
            .as_deref()
            .filter(|alias| !alias.trim().is_empty())
            .context("Matrix Goal command has no configured channel alias")?;
        let session_id = conversation_history_key(message);
        let ingress = zeroclaw_runtime::goal_mode::GoalIngressContext::trusted(
            session_id.clone(),
            context.agent_alias.as_str().to_owned(),
            format!("matrix:{alias}:{}", message.reply_target),
            Some(message.sender.clone()),
        )?;
        let runtime = runtime_for_matrix_session(context, &session_id).await?;
        runtime
            .submit(ingress, command)
            .await
            .map_err(anyhow::Error::from)
    }
    .await;
    match response {
        Ok(response) => render_goal_response(&response),
        Err(_) => super::channel_runtime_cli_string("goal-response-controller-failed"),
    }
}

/// Responses that do not require session binding or runtime-owned state.
///
/// A disabled Goal surface is deliberately inert: reporting that it is off
/// must not initialize the daemon control plane or open the canonical ledger.
fn local_goal_response(
    command: &zeroclaw_runtime::goal_mode::GoalCommand,
    enabled: bool,
) -> Option<GoalResponse> {
    let key = if *command == zeroclaw_runtime::goal_mode::GoalCommand::Help {
        GoalMessageKey::Help
    } else if !enabled {
        GoalMessageKey::Disabled
    } else {
        return None;
    };
    Some(GoalResponse { key, status: None })
}

fn parse_error_key(error: GoalCommandErrorKey) -> GoalMessageKey {
    match error {
        GoalCommandErrorKey::InvalidBudget => GoalMessageKey::InvalidBudget,
        GoalCommandErrorKey::SuccessCriteriaRequired => GoalMessageKey::SuccessCriteriaRequired,
        GoalCommandErrorKey::InputTooLong => GoalMessageKey::InputTooLong,
        GoalCommandErrorKey::InvalidGrammar
        | GoalCommandErrorKey::ObjectiveRequired
        | GoalCommandErrorKey::InvalidArguments => GoalMessageKey::InvalidCommand,
    }
}

fn render_goal_response(response: &GoalResponse) -> String {
    let key = match response.key {
        GoalMessageKey::Help => "goal-response-help",
        GoalMessageKey::Disabled => "goal-response-disabled",
        GoalMessageKey::InvalidConfiguration => "goal-response-invalid-configuration",
        GoalMessageKey::NoCurrentGoal => "goal-response-no-current-goal",
        GoalMessageKey::AlreadyActive => "goal-response-already-active",
        GoalMessageKey::BudgetRequired => "goal-response-budget-required",
        GoalMessageKey::PricingUnavailable => "goal-response-pricing-unavailable",
        GoalMessageKey::VerifierUnavailable => "goal-response-verifier-unavailable",
        GoalMessageKey::BindingUnavailable => "goal-response-binding-unavailable",
        GoalMessageKey::Started => "goal-response-started",
        GoalMessageKey::Status => "goal-response-status",
        GoalMessageKey::Budget => "goal-response-budget",
        GoalMessageKey::Paused => "goal-response-paused",
        GoalMessageKey::AlreadyPaused => "goal-response-already-paused",
        GoalMessageKey::Resumed => "goal-response-resumed",
        GoalMessageKey::Cancelled => "goal-response-cancelled",
        GoalMessageKey::AlreadyCancelled => "goal-response-already-cancelled",
        GoalMessageKey::BudgetUpdated => "goal-response-budget-updated",
        GoalMessageKey::InvalidCommand => "goal-response-invalid-command",
        GoalMessageKey::InvalidBudget => "goal-response-invalid-budget",
        GoalMessageKey::SuccessCriteriaRequired => "goal-response-success-criteria-required",
        GoalMessageKey::InputTooLong => "goal-response-input-too-long",
        GoalMessageKey::BudgetExhausted => "goal-response-budget-exhausted",
        GoalMessageKey::AccountingIncomplete => "goal-response-accounting-incomplete",
        GoalMessageKey::ReservationUnavailable => "goal-response-reservation-unavailable",
        GoalMessageKey::ReservationExceedsBudget => "goal-response-reservation-exceeds-budget",
        GoalMessageKey::ToolPhaseNonResumable => "goal-response-tool-phase-nonresumable",
        GoalMessageKey::Terminal => "goal-response-terminal",
        GoalMessageKey::ExecutorStartFailed => "goal-response-executor-start-failed",
        GoalMessageKey::ExecutorQuiescing => "goal-response-executor-quiescing",
        GoalMessageKey::Stale => "goal-response-stale",
    };
    let mut rendered = super::channel_runtime_cli_string(key);
    if let Some(status) = &response.status {
        let status_label = match status.status {
            zeroclaw_runtime::control_plane::TaskStatus::Running => "running",
            zeroclaw_runtime::control_plane::TaskStatus::Paused => "paused",
            zeroclaw_runtime::control_plane::TaskStatus::Completed => "completed",
            zeroclaw_runtime::control_plane::TaskStatus::Failed => "failed",
            zeroclaw_runtime::control_plane::TaskStatus::Cancelled => "cancelled",
            zeroclaw_runtime::control_plane::TaskStatus::Lost => "lost",
            zeroclaw_runtime::control_plane::TaskStatus::TimedOut => "timed_out",
        };
        let resumable = status.resumable.to_string();
        let tokens = status.known_token_usage.to_string();
        let cost = format!("{:.6}", status.known_cost_usd);
        rendered.push_str(&format!(
            "\n\n{}",
            zeroclaw_runtime::i18n::get_required_cli_string_with_args(
                "goal-response-status-summary",
                &[
                    ("status", status_label),
                    ("resumable", &resumable),
                    ("tokens", &tokens),
                    ("cost", &cost),
                ],
            )
        ));
    }
    rendered
}

#[async_trait]
impl GoalSessionLease for MatrixGoalSessionLease {
    fn working_history(&self) -> Result<Vec<zeroclaw_api::model_provider::ChatMessage>> {
        let history = self
            .context
            .conversation_histories
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(self.binding.session_id())
            .cloned()
            .unwrap_or_default();
        Ok(normalize_cached_channel_turns(history))
    }

    async fn run_parent_turn(
        &mut self,
        mut turn: GoalParentTurn,
        cancellation: CancellationToken,
    ) -> Result<(String, Vec<zeroclaw_api::model_provider::ChatMessage>)> {
        let (defaults, selection) = self.current_route()?;
        let provider = get_or_create_provider(
            self.context.as_ref(),
            &selection.model_provider,
            selection.api_key.as_deref(),
            &defaults,
        )
        .await?;
        let thinking_override = self
            .context
            .thinking_overrides
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(self.binding.session_id())
            .copied();
        // A Goal declaration is controller-owned, not a user prompt. Preserve
        // the session's selected thinking level without interpreting objective
        // text as a channel thinking directive.
        let thinking = resolve_channel_thinking(
            "",
            thinking_override,
            &self.context.agent_cfg.resolved.thinking,
            defaults.defaults.temperature,
        );
        let excluded_tools: &[String] =
            if self.context.autonomy_level == zeroclaw_runtime::security::AutonomyLevel::Full {
                &[]
            } else {
                self.context.non_cli_excluded_tools.as_ref()
            };
        let read_skill_available = super::read_skill_available_for_channel_turn(
            provider.as_ref(),
            self.context.agent_cfg.resolved.strict_tool_parsing,
            self.context.tools_registry.as_ref(),
            excluded_tools,
            self.context.system_prompt.as_str(),
        );
        let system_prompt = super::build_channel_system_prompt(
            &system_prompt_for_channel_turn(
                self.context.as_ref(),
                self.context.system_prompt.as_str(),
                false,
                read_skill_available,
            ),
            "matrix",
            None,
        );
        if turn
            .working_history
            .first()
            .is_some_and(|message| message.role == "system")
        {
            turn.working_history[0] =
                zeroclaw_api::model_provider::ChatMessage::system(system_prompt);
        } else {
            turn.working_history.insert(
                0,
                zeroclaw_api::model_provider::ChatMessage::system(system_prompt),
            );
        }
        turn.working_history
            .push(zeroclaw_api::model_provider::ChatMessage::user(
                turn.directive(),
            ));
        let turn_id = uuid::Uuid::new_v4().to_string();
        let loop_knobs = LoopKnobs::default();
        let tool_loop = run_tool_call_loop(ToolLoop {
            exec: ResolvedAgentExecution::resolve(
                ResolvedModelAccess {
                    model_provider: provider.as_ref(),
                    provider_name: selection.model_provider.as_str(),
                    model: selection.model.as_str(),
                    temperature: thinking.effective_temperature,
                },
                ResolvedIo {
                    tools_registry: self.context.tools_registry.as_ref(),
                    observer: self.context.observer.as_ref(),
                    silent: true,
                    approval: Some(self.context.approval_manager.as_ref()),
                    multimodal_config: &self.context.multimodal,
                    config: Some(defaults.config.as_ref()),
                    hooks: self.context.hooks.as_deref(),
                    activated_tools: self.context.activated_tools.as_ref(),
                    model_switch_callback: None,
                    receipt_generator: self.context.receipt_generator.as_ref(),
                },
                ResolvedRuntimeKnobs {
                    max_tool_iterations: self.context.max_tool_iterations,
                    excluded_tools,
                    dedup_exempt_tools: self.context.tool_call_dedup_exempt.as_ref(),
                    pacing: &self.context.pacing,
                    strict_tool_parsing: self.context.agent_cfg.resolved.strict_tool_parsing,
                    parallel_tools: self.context.agent_cfg.resolved.parallel_tools,
                    max_tool_result_chars: self.context.max_tool_result_chars,
                    context_token_budget: self.context.context_token_budget,
                    knobs: &loop_knobs,
                },
            ),
            history: &mut turn.working_history,
            channel_name: "matrix",
            channel_reply_target: Some(self.route.room_id.as_str()),
            cancellation_token: Some(cancellation.clone()),
            on_delta: None,
            shared_budget: None,
            channel: None,
            collected_receipts: None,
            event_tx: None,
            steering: None,
            new_messages_out: None,
            image_cache: None,
            memory: None,
            ingress: zeroclaw_api::ingress::IngressContext::channel(),
            agent_alias: Some(self.context.agent_alias.as_str()),
            parent_agent_alias: None,
            turn_id: &turn_id,
            sop_reassembly: Some(zeroclaw_runtime::agent::loop_::SopStepReassembly {
                config: defaults.config.as_ref(),
            }),
        });
        let tool_loop = zeroclaw_api::NATIVE_THINKING_OVERRIDE
            .scope(thinking.params.native_thinking, tool_loop);
        let response = tokio::select! {
            result = scope_session_key(Some(self.binding.session_id().to_owned()), tool_loop) => result?,
            () = cancellation.cancelled() => bail!("Matrix Goal parent turn was cancelled"),
        };
        Ok((response, turn.working_history))
    }

    async fn run_verifier(
        &mut self,
        turn: GoalVerifierTurn,
        cancellation: CancellationToken,
    ) -> Result<String> {
        let defaults = runtime_defaults_snapshot(self.context.as_ref());
        run_isolated_verifier(defaults.config.as_ref(), &turn, cancellation).await
    }

    async fn append_verified_candidate(&mut self, candidate: GoalVerifiedCandidate) -> Result<()> {
        if candidate.session_id != self.binding.session_id() {
            bail!("Matrix Goal delivery session does not match its lease");
        }
        let candidate = sanitize_verified_goal_delivery(
            &candidate.candidate,
            self.context.tools_registry.as_ref(),
            self.context.prompt_config.as_ref(),
        );
        let channel_key = composite_channel_key("matrix", Some(self.route.alias.as_str()));
        let channel = self
            .context
            .channels_by_name
            .get(&channel_key)
            .context("Matrix Goal channel alias disappeared before delivery")?;
        channel
            .send(&zeroclaw_api::channel::SendMessage::new(
                &candidate,
                &self.route.room_id,
            ))
            .await?;
        super::append_sender_turn(
            self.context.as_ref(),
            self.binding.session_id(),
            zeroclaw_api::model_provider::ChatMessage::assistant(candidate),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use zeroclaw_config::{providers::ChannelRef, schema::AliasedAgentConfig};
    use zeroclaw_runtime::control_plane::{TaskKind, TaskRecord, TaskStatus};
    use zeroclaw_runtime::goal_mode::{GoalCommand, GoalMessageKey, host::GoalSessionBinding};

    use super::{
        MatrixGoalRoute, is_matrix_goal_command, is_matrix_goal_route, local_goal_response,
        matrix_goal_policy_is_revoked, matrix_goal_route, route_is_owned_by_agent,
    };

    fn goal_task(route: Option<&str>) -> TaskRecord {
        TaskRecord {
            id: "goal".into(),
            kind: TaskKind::Goal,
            agent: "main".into(),
            status: TaskStatus::Running,
            owner_pid: 1,
            owner_boot_id: "boot".into(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: route.map(str::to_owned),
            delivered: false,
            idem_key: None,
            principal_id: None,
            session_id: Some("session".into()),
            execution_epoch: 1,
            started_at: "2026-01-01T00:00:00Z".into(),
            finished_at: None,
        }
    }

    fn message(channel: &str, content: &str) -> zeroclaw_api::channel::ChannelMessage {
        zeroclaw_api::channel::ChannelMessage {
            channel: channel.into(),
            content: content.into(),
            ..Default::default()
        }
    }

    #[test]
    fn matrix_goal_recognition_is_exact_and_transport_scoped() {
        for content in ["/goal", "/goal status", "  /goal help"] {
            assert!(
                is_matrix_goal_command(&message("matrix", content)),
                "{content}"
            );
        }
        for (channel, content) in [
            ("matrix", "/goalpost"),
            ("matrix", "please /goal status"),
            ("telegram", "/goal status"),
        ] {
            assert!(
                !is_matrix_goal_command(&message(channel, content)),
                "{channel}: {content}"
            );
        }
    }

    #[test]
    fn disabled_goal_commands_need_no_runtime_or_control_plane() {
        let disabled = local_goal_response(&GoalCommand::Status, false)
            .expect("a disabled Goal command must return before runtime setup");
        assert_eq!(disabled.key, GoalMessageKey::Disabled);
        assert!(disabled.status.is_none());

        let help = local_goal_response(&GoalCommand::Help, false)
            .expect("help remains locally available while Goal Mode is disabled");
        assert_eq!(help.key, GoalMessageKey::Help);
        assert!(local_goal_response(&GoalCommand::Status, true).is_none());
    }

    #[test]
    fn matrix_route_recomputes_the_thread_independent_session_key() {
        let route = MatrixGoalRoute::parse("matrix:primary:!room:example.org").unwrap();
        assert_eq!(route.alias, "primary");
        assert_eq!(route.room_id, "!room:example.org");
        assert_eq!(
            route.canonical_session_id("@user:example.org").unwrap(),
            "matrix_primary__room_example_org__user_example_org"
        );
    }

    #[test]
    fn verified_goal_delivery_uses_the_standard_leak_guard() {
        let candidate = "Temporary key: AKIAABCDEFGHIJKLMNOP"; // gitleaks:allow
        let delivered = super::sanitize_verified_goal_delivery(
            candidate,
            &[],
            &zeroclaw_config::schema::Config::default(),
        );
        assert!(!delivered.contains("AKIAABCDEFGHIJKLMNOP")); // gitleaks:allow
    }

    #[test]
    fn matrix_route_rejects_noncanonical_or_unsafe_bindings() {
        assert!(MatrixGoalRoute::parse("rpc").is_err());
        assert!(MatrixGoalRoute::parse("matrix::!room:example.org").is_err());
        assert!(MatrixGoalRoute::parse("matrix: primary:!room:example.org").is_err());
        assert!(MatrixGoalRoute::parse("matrix:primary:\n").is_err());
        let route = MatrixGoalRoute::parse("matrix:primary:!room:example.org").unwrap();
        assert!(route.canonical_session_id(" ").is_err());
    }

    #[test]
    fn matrix_binding_requires_the_canonical_sender_room_session() {
        let route = MatrixGoalRoute::parse("matrix:primary:!room:example.org").unwrap();
        let binding = GoalSessionBinding::new(
            "matrix_primary__room_example_org__user_example_org".into(),
            "main".into(),
            "matrix:primary:!room:example.org".into(),
            Some("@user:example.org".into()),
        )
        .unwrap();

        assert!(route.matches_binding(&binding, "main"));

        let mismatched_session = GoalSessionBinding::new(
            "matrix_primary__other_example_org__user_example_org".into(),
            "main".into(),
            "matrix:primary:!room:example.org".into(),
            Some("@user:example.org".into()),
        )
        .unwrap();
        assert!(!route.matches_binding(&mismatched_session, "main"));
    }

    #[test]
    fn matrix_route_requires_the_configured_agent_owner() {
        let mut config = zeroclaw_config::schema::Config::default();
        config.agents.insert(
            "main".into(),
            AliasedAgentConfig {
                channels: vec![ChannelRef::new("matrix.primary")],
                ..AliasedAgentConfig::default()
            },
        );

        assert!(route_is_owned_by_agent(&config, "matrix.primary", "main"));
        assert!(!route_is_owned_by_agent(&config, "matrix.primary", "other"));
    }

    #[test]
    fn reload_filters_other_goal_transports_by_the_durable_route() {
        assert!(matrix_goal_route(&goal_task(Some("matrix:primary:!room:example.org"))).is_some());
        assert!(matrix_goal_route(&goal_task(Some("rpc:session"))).is_none());
        assert!(matrix_goal_route(&goal_task(None)).is_none());
        assert!(
            is_matrix_goal_route(&goal_task(Some("matrix:malformed"))),
            "malformed Matrix-owned rows must be classified and fail closed during reload"
        );
        assert!(!is_matrix_goal_route(&goal_task(Some("rpc:session"))));
    }

    #[test]
    fn disabled_goal_mode_revokes_a_durably_bound_matrix_goal() {
        let config = zeroclaw_config::schema::Config::default();
        assert!(matrix_goal_policy_is_revoked(
            &config,
            "main",
            &goal_task(Some("matrix:primary:!room:example.org")),
        ));
    }
}
