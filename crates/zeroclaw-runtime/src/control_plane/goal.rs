//! Goal-mode admission and controller helpers.
//!
//! This module is the single Rust admission path for slash-command and
//! agent-callable goal start/resume requests. Callers pass trusted runtime
//! context explicitly; model/user text supplies only the untrusted
//! objective/action payload.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use zeroclaw_commands::{BuiltinCommandId, CommandSurface, command_by_name};
use zeroclaw_config::cost::CostTracker;
use zeroclaw_config::schema::{AliasedAgentConfig, Config};

use crate::agent::cost::{is_goal_accounting_failure, is_goal_accounting_pricing_failure};

use super::global::control_plane;
use super::goal_task::{
    ActiveGoalControlBinding, GoalBlocker, GoalBlockerKind, GoalPauseReason, GoalPauseState,
    GoalTaskRecord, GoalTaskRegistry, TaskContinuationContext, TaskGoal,
};
use super::task_registry::{TaskKind, TaskRecord, TaskRegistry, TaskStatus};
use super::verifier::{
    GoalVerificationRequest, GoalVerifier, GoalVerifierDecision, LlmGoalVerifier,
    verifier_outage_pause,
};

tokio::task_local! {
    static GOAL_RUNTIME_SCOPE: GoalRuntimeScope;
    static GOAL_START_TOOL_BATCH: bool;
}

type GoalTaskBindingSink = Arc<dyn Fn(&str) + Send + Sync>;

/// Ephemeral task-local context for one goal-aware model/tool turn.
///
/// This is deliberately not durable goal state. Durable lifecycle facts live in
/// `TaskRecord` plus the goal extension row; this scope only carries the live
/// channel/controller handles needed while polling one turn.
#[derive(Clone, Default)]
pub struct GoalRuntimeScope {
    /// Trusted admission facts attached by channel ingress.
    ///
    /// The inner context is shared by the tools in this one live turn so a
    /// successful exact goal admission can bind its returned task id before a
    /// later approval request. This remains transient trust plumbing; the
    /// canonical task and continuation rows remain authoritative.
    admission_binding: Arc<parking_lot::Mutex<GoalAdmissionBindingState>>,
    /// Optional live channel sink for controller/verifier progress messages.
    state_update_sink: Option<GoalStateUpdateSink>,
    /// Shared marker promoted when the current turn becomes goal work.
    turn_evaluation_requested: Option<Arc<AtomicBool>>,
    /// Live configuration resolver used by model-callable goal tools.
    ///
    /// Channel slash commands already receive the current snapshot directly.
    /// Tools are assembled at startup, so they need this task-local resolver to
    /// avoid authorizing a later call with a stale captured config.
    config_resolver: Option<Arc<dyn Fn() -> Arc<Config> + Send + Sync>>,
    /// Process-local sink that binds a newly admitted exact goal to the
    /// already registered live worker executing this turn.
    task_binding_sink: Option<GoalTaskBindingSink>,
}

#[derive(Default)]
struct GoalAdmissionBindingState {
    context: Option<GoalAdmissionContext>,
    reserved_for_admission: bool,
}

impl GoalRuntimeScope {
    pub fn new(
        admission_context: Option<GoalAdmissionContext>,
        state_update_sink: Option<GoalStateUpdateSink>,
        turn_evaluation_requested: Option<Arc<AtomicBool>>,
    ) -> Self {
        Self {
            admission_binding: Arc::new(parking_lot::Mutex::new(GoalAdmissionBindingState {
                context: admission_context,
                reserved_for_admission: false,
            })),
            state_update_sink,
            turn_evaluation_requested,
            config_resolver: None,
            task_binding_sink: None,
        }
    }

    pub fn with_config_resolver(
        mut self,
        resolver: Arc<dyn Fn() -> Arc<Config> + Send + Sync>,
    ) -> Self {
        self.config_resolver = Some(resolver);
        self
    }

    pub fn with_task_binding_sink(mut self, sink: Arc<dyn Fn(&str) + Send + Sync>) -> Self {
        self.task_binding_sink = Some(sink);
        self
    }

    /// Return the current trusted admission facts for this live turn.
    ///
    /// Cloned runtime scopes share the same inner context, so a caller that
    /// retains a clone across a scoped model loop observes any exact task id
    /// bound by a successful goal tool call inside that loop.
    #[must_use]
    pub fn admission_context(&self) -> Option<GoalAdmissionContext> {
        self.admission_binding.lock().context.clone()
    }

    fn with_admission_context(mut self, admission_context: Option<GoalAdmissionContext>) -> Self {
        self.admission_binding = Arc::new(parking_lot::Mutex::new(GoalAdmissionBindingState {
            context: admission_context,
            reserved_for_admission: false,
        }));
        self
    }

    fn with_state_update_sink(mut self, state_update_sink: Option<GoalStateUpdateSink>) -> Self {
        self.state_update_sink = state_update_sink;
        self
    }

    fn with_turn_evaluation_marker(
        mut self,
        turn_evaluation_requested: Option<Arc<AtomicBool>>,
    ) -> Self {
        self.turn_evaluation_requested = turn_evaluation_requested;
        self
    }
}

/// Ephemeral channel backchannel for goal controller status messages.
///
/// The controller uses this while processing a live channel turn to publish
/// status transitions and verifier progress before the final model response is
/// ready. It is not persisted and is not replayed; restart-visible state remains
/// in the task/goal registries.
#[derive(Clone)]
pub struct GoalStateUpdateSink {
    /// Channel-local sender for controller-generated progress events.
    ///
    /// This is an ephemeral notification path. Durable lifecycle state stays in
    /// the task registry and goal extension table.
    tx: tokio::sync::mpsc::UnboundedSender<GoalStateUpdateEvent>,
}

/// User-visible progress event emitted by the goal controller while a channel
/// turn is still running.
///
/// The event carries render-ready text because localization happens at the
/// control-plane boundary where the status/verifier context is available. It is
/// not durable state and is intentionally not replayed after restart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoalStateUpdateEvent {
    /// Replace or append a visible lifecycle/status update.
    Status(String),
    /// Show a temporary "verification in progress" message while the verifier
    /// model call is pending.
    VerifierStarted(String),
}

impl GoalStateUpdateSink {
    pub fn new(tx: tokio::sync::mpsc::UnboundedSender<GoalStateUpdateEvent>) -> Self {
        Self { tx }
    }

    pub fn send(&self, event: GoalStateUpdateEvent) {
        let _ = self.tx.send(event);
    }
}

fn msg(key: &str, args: &[(&str, &str)]) -> String {
    crate::i18n::get_required_cli_string_with_args(key, args)
}

fn task_status_label(status: TaskStatus) -> String {
    let key = match status {
        TaskStatus::Running => "goal-status-running",
        TaskStatus::Paused => "goal-status-paused",
        TaskStatus::Completed => "goal-status-completed",
        TaskStatus::Failed => "goal-status-failed",
        TaskStatus::Cancelled => "goal-status-cancelled",
        TaskStatus::Lost => "goal-status-lost",
        TaskStatus::TimedOut => "goal-status-timed-out",
    };
    msg(key, &[])
}

fn pause_reason_label(reason: GoalPauseReason) -> String {
    let key = match reason {
        GoalPauseReason::OperatorPaused => "goal-pause-reason-operator-paused",
        GoalPauseReason::NeedsUserInput => "goal-pause-reason-needs-user-input",
        GoalPauseReason::HumanEscalation => "goal-pause-reason-human-escalation",
        GoalPauseReason::ExternalDependency => "goal-pause-reason-external-dependency",
        GoalPauseReason::ProviderUnavailable => "goal-pause-reason-provider-unavailable",
        GoalPauseReason::VerifierBlocked => "goal-pause-reason-verifier-blocked",
        GoalPauseReason::BudgetExhausted => "goal-pause-reason-budget-exhausted",
        GoalPauseReason::BudgetUnavailable => "goal-pause-reason-budget-unavailable",
        GoalPauseReason::DaemonRestart => "goal-pause-reason-daemon-restarted",
    };
    msg(key, &[])
}

fn formatted_cost(value: f64) -> String {
    let rounded = format!("{value:.4}");
    let amount = if value > 0.0 && rounded == "0.0000" {
        format!("{value:.4e}")
    } else {
        rounded
    };
    msg("goal-budget-cost-value", &[("amount", &amount)])
}

fn token_limit_label(limit: Option<u64>) -> String {
    limit
        .map(|value| value.to_string())
        .unwrap_or_else(|| msg("goal-budget-limit-unlimited", &[]))
}

fn cost_limit_label(limit: Option<f64>) -> String {
    limit
        .map(formatted_cost)
        .unwrap_or_else(|| msg("goal-budget-limit-unlimited", &[]))
}

/// Ledger-derived usage snapshot for one goal task.
///
/// This is a per-call materialized view over persisted `CostRecord` rows.
/// It must never be stored back into `goal_tasks`; consumed and remaining
/// budget are always derived from the ledger so budget changes cannot drift
/// from usage history.
#[derive(Debug, Clone, Copy)]
struct GoalUsageTotals {
    /// Tokens attributed to this task by the canonical cost ledger.
    total_tokens: u64,
    /// USD cost attributed to this task by the canonical cost ledger.
    cost_usd: f64,
    /// Whether every cost-bearing row had reliable pricing.
    ///
    /// Token totals remain usable when this is false, but an active cost limit
    /// must pause because unknown cost cannot be treated as free.
    cost_pricing_available: bool,
    /// Whether the ledger is configured to calculate USD amounts. Token-only
    /// goal accounting deliberately leaves this false rather than displaying
    /// a fabricated zero-dollar total.
    cost_tracking_available: bool,
    /// Whether every attributed provider call supplied usable token counts.
    usage_available: bool,
}

/// Why the controller cannot produce trustworthy budget accounting.
///
/// This is not persisted as a separate state enum. It only shapes the blocker
/// payload at the moment a goal is paused so operators can tell "ledger missing"
/// from "ledger present but USD pricing unknown".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GoalBudgetUnavailableCause {
    /// No canonical usage snapshot could be read for a budgeted goal.
    UsageUnavailable,
    /// Usage rows exist, but at least one cost-attributed call lacks pricing.
    CostPricingUnavailable,
}

impl Default for GoalUsageTotals {
    fn default() -> Self {
        Self {
            total_tokens: 0,
            cost_usd: 0.0,
            cost_pricing_available: true,
            cost_tracking_available: true,
            usage_available: true,
        }
    }
}

fn goal_usage_totals(config: Option<&Config>, task_id: &str) -> Option<GoalUsageTotals> {
    let tracker = goal_usage_ledger(config)?;
    goal_usage_totals_from_tracker(Some(tracker.as_ref()), task_id, true)
}

fn goal_usage_totals_if_tracker_ready(
    config: Option<&Config>,
    task_id: &str,
) -> Option<GoalUsageTotals> {
    let tracker = existing_cost_tracker(config)?;
    goal_usage_totals_from_tracker(Some(tracker.as_ref()), task_id, false)
}

fn goal_usage_totals_from_tracker(
    tracker: Option<&CostTracker>,
    task_id: &str,
    require_writable_ledger: bool,
) -> Option<GoalUsageTotals> {
    let tracker = tracker?;
    if require_writable_ledger && tracker.ensure_storage_ready().is_err() {
        return None;
    }
    match tracker.get_usage_totals_for_task_with_pricing(task_id) {
        Ok((total_tokens, cost_usd, cost_pricing_available, usage_available)) => {
            Some(GoalUsageTotals {
                total_tokens,
                cost_usd,
                cost_pricing_available,
                cost_tracking_available: tracker.is_enabled(),
                usage_available,
            })
        }
        Err(error) => {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                    .with_outcome(::zeroclaw_log::EventOutcome::Unknown)
                    .with_attrs(::serde_json::json!({
                        "task_id": task_id,
                        "error": format!("{error}"),
                    })),
                "Failed to derive goal usage summary"
            );
            None
        }
    }
}

fn initial_goal_usage_totals(config: Option<&Config>) -> Option<GoalUsageTotals> {
    if config.is_none() {
        return Some(GoalUsageTotals::default());
    }
    let tracker = goal_usage_ledger(config)?;
    goal_usage_ledger_is_healthy(&tracker).then_some(GoalUsageTotals {
        cost_tracking_available: tracker.is_enabled(),
        ..GoalUsageTotals::default()
    })
}

fn goal_usage_ledger(config: Option<&Config>) -> Option<std::sync::Arc<CostTracker>> {
    let config = config?;
    CostTracker::get_or_init_global_goal_usage_ledger(config.cost.clone(), &config.data_dir)
}

fn existing_cost_tracker(config: Option<&Config>) -> Option<std::sync::Arc<CostTracker>> {
    let config = config?;
    CostTracker::existing_global(config.cost.clone(), &config.data_dir)
}

fn goal_budget_summary(goal: &GoalTaskRecord, usage: Option<&GoalUsageTotals>) -> String {
    let token_limit = token_limit_label(goal.effective_token_limit);
    let cost_limit = cost_limit_label(goal.effective_cost_limit_usd);
    if let Some(usage) = usage {
        let tokens_used = usage.total_tokens.to_string();
        if !usage.usage_available {
            if usage.cost_usd == 0.0 {
                return msg(
                    "goal-budget-summary-incomplete-cost-unavailable",
                    &[("tokens_used", &tokens_used), ("token_limit", &token_limit)],
                );
            }
            let cost_used = formatted_cost(usage.cost_usd);
            return msg(
                "goal-budget-summary-incomplete",
                &[
                    ("tokens_used", &tokens_used),
                    ("token_limit", &token_limit),
                    ("cost_used", &cost_used),
                    ("cost_limit", &cost_limit),
                ],
            );
        }
        if !usage.cost_tracking_available || !usage.cost_pricing_available {
            return msg(
                "goal-budget-summary-cost-unavailable",
                &[("tokens_used", &tokens_used), ("token_limit", &token_limit)],
            );
        }
        let cost_used = formatted_cost(usage.cost_usd);
        msg(
            "goal-budget-summary",
            &[
                ("tokens_used", &tokens_used),
                ("token_limit", &token_limit),
                ("cost_used", &cost_used),
                ("cost_limit", &cost_limit),
            ],
        )
    } else {
        msg(
            "goal-budget-summary-unavailable",
            &[("token_limit", &token_limit), ("cost_limit", &cost_limit)],
        )
    }
}

fn task_goal_budget_summary(task_goal: &TaskGoal, config: Option<&Config>) -> String {
    let usage = goal_usage_totals(config, task_goal.task_id());
    goal_budget_summary(task_goal.goal(), usage.as_ref())
}

/// Render the visible restart-recovery notice for a durable goal.
///
/// Recovery is queued by task id, but the user-facing message is derived from
/// the canonical goal extension record at delivery time so objective changes
/// and consumed budget stay consistent with the rest of the control plane.
pub fn goal_recovery_status_message(goal: &GoalTaskRecord, config: Option<&Config>) -> String {
    let usage = goal_usage_totals_if_tracker_ready(config, &goal.task_id);
    let budget = goal_budget_summary(goal, usage.as_ref());
    msg(
        "goal-command-recovered",
        &[
            ("task_id", &goal.task_id),
            ("objective", &goal.objective),
            ("budget", &budget),
        ],
    )
}

/// Budget-gate decision for a single ledger snapshot.
///
/// The booleans distinguish which effective limit fired so the pause payload
/// can be explicit without duplicating the full budget state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GoalBudgetExhaustion {
    /// The task's attributed token usage has reached the effective token limit.
    tokens: bool,
    /// The task's attributed cost has reached the effective cost limit.
    cost: bool,
}

fn goal_budget_exhaustion(
    goal: &GoalTaskRecord,
    usage: Option<&GoalUsageTotals>,
) -> Option<GoalBudgetExhaustion> {
    let usage = usage?;
    let tokens = goal
        .effective_token_limit
        .is_some_and(|limit| usage.total_tokens >= limit);
    let cost = goal
        .effective_cost_limit_usd
        .is_some_and(|limit| usage.cost_usd >= limit);
    (tokens || cost).then_some(GoalBudgetExhaustion { tokens, cost })
}

fn goal_has_effective_budget(goal: &GoalTaskRecord) -> bool {
    goal.effective_token_limit.is_some() || goal.effective_cost_limit_usd.is_some()
}

/// A finite cost budget is meaningful only when USD pricing is enabled and
/// its canonical ledger is writable. Token-only goals deliberately bypass
/// this check and continue to use the same ledger for exact task attribution.
fn ensure_cost_budget_tracking_available(
    config: Option<&Config>,
    cost_limit_usd: Option<f64>,
    ledger_healthy: Option<bool>,
) -> Result<()> {
    if cost_limit_usd.is_none() {
        return Ok(());
    }
    let Some(config) = config else {
        return Ok(());
    };
    if !config.cost.enabled
        || CostTracker::get_or_init_global(config.cost.clone(), &config.data_dir).is_none_or(
            |tracker| {
                !tracker.is_enabled()
                    || !ledger_healthy.unwrap_or_else(|| goal_usage_ledger_is_healthy(&tracker))
            },
        )
    {
        bail!("{}", msg("goal-command-error-cost-tracking-required", &[]));
    }
    Ok(())
}

/// Verify that the canonical JSONL ledger can both read its existing rows and
/// accept a future exact-task observation. Tracker construction alone creates
/// only the parent directory, so it is not sufficient evidence of availability.
fn goal_usage_ledger_is_healthy(tracker: &CostTracker) -> bool {
    tracker.ensure_storage_ready().is_ok()
        && tracker
            .get_usage_totals_for_task_with_pricing("__goal_usage_ledger_health_check__")
            .is_ok()
}

fn goal_budget_pause(
    goal: &GoalTaskRecord,
    usage: Option<&GoalUsageTotals>,
) -> Option<GoalPauseState> {
    let usage = usage?;
    if !usage.usage_available {
        return Some(goal_accounting_unavailable_pause(
            goal,
            GoalBudgetUnavailableCause::UsageUnavailable,
        ));
    }
    if goal.effective_cost_limit_usd.is_some()
        && (!usage.cost_tracking_available || !usage.cost_pricing_available)
    {
        return goal_budget_unavailable_pause(
            goal,
            GoalBudgetUnavailableCause::CostPricingUnavailable,
        );
    }
    let exhaustion = goal_budget_exhaustion(goal, Some(usage))?;
    let budget = goal_budget_summary(goal, Some(usage));
    Some(GoalPauseState {
        reason: GoalPauseReason::BudgetExhausted,
        description: Some(msg(
            "goal-command-budget-exhausted-description",
            &[("budget", &budget)],
        )),
        blockers: vec![GoalBlocker {
            kind: GoalBlockerKind::Budget,
            message: msg(
                "goal-command-budget-exhausted-blocker",
                &[("budget", &budget)],
            ),
            payload: Some(serde_json::json!({
                "tokens": {
                    "exhausted": exhaustion.tokens,
                    "used": usage.total_tokens,
                    "limit": goal.effective_token_limit,
                },
                "cost": {
                    "exhausted": exhaustion.cost,
                    "used_usd": usage.cost_usd,
                    "limit_usd": goal.effective_cost_limit_usd,
                },
            })),
        }],
    })
}

fn goal_budget_unavailable_pause(
    goal: &GoalTaskRecord,
    cause: GoalBudgetUnavailableCause,
) -> Option<GoalPauseState> {
    if !goal_has_effective_budget(goal) {
        return None;
    }
    Some(goal_accounting_unavailable_pause(goal, cause))
}

fn goal_accounting_unavailable_pause(
    goal: &GoalTaskRecord,
    cause: GoalBudgetUnavailableCause,
) -> GoalPauseState {
    let budget = goal_budget_summary(goal, None);
    let usage_unavailable = matches!(cause, GoalBudgetUnavailableCause::UsageUnavailable);
    let cost_pricing_unavailable =
        matches!(cause, GoalBudgetUnavailableCause::CostPricingUnavailable);
    GoalPauseState {
        reason: GoalPauseReason::BudgetUnavailable,
        description: Some(msg(
            "goal-command-budget-unavailable-description",
            &[("budget", &budget)],
        )),
        blockers: vec![GoalBlocker {
            kind: GoalBlockerKind::Budget,
            message: msg(
                "goal-command-budget-unavailable-blocker",
                &[("budget", &budget)],
            ),
            payload: Some(serde_json::json!({
                "usage_unavailable": usage_unavailable,
                "cost_pricing_unavailable": cost_pricing_unavailable,
                "token_limit": goal.effective_token_limit,
                "cost_limit_usd": goal.effective_cost_limit_usd,
            })),
        }],
    }
}

fn goal_budget_gate_pause(
    goal: &GoalTaskRecord,
    usage: Option<&GoalUsageTotals>,
) -> Option<GoalPauseState> {
    match usage {
        Some(usage) => goal_budget_pause(goal, Some(usage)),
        None => goal_budget_unavailable_pause(goal, GoalBudgetUnavailableCause::UsageUnavailable),
    }
}

/// A missing ledger snapshot and a durable provider-usage-unavailable
/// observation are both fail-closed accounting states. The latter preserves
/// exact-task evidence and known totals, but it still cannot prove complete
/// autonomous consumption.
fn goal_usage_ledger_gate_pause(
    goal: &GoalTaskRecord,
    usage: Option<&GoalUsageTotals>,
) -> Option<GoalPauseState> {
    usage.is_none().then(|| {
        goal_accounting_unavailable_pause(goal, GoalBudgetUnavailableCause::UsageUnavailable)
    })
}

/// Evaluate the ordered accounting contract shared by configured execution
/// paths: a missing canonical ledger is an infrastructure failure before an
/// otherwise-available budget can be evaluated.
fn goal_accounting_gate_pause(
    goal: &GoalTaskRecord,
    usage: Option<&GoalUsageTotals>,
) -> Option<GoalPauseState> {
    goal_usage_ledger_gate_pause(goal, usage).or_else(|| goal_budget_gate_pause(goal, usage))
}

fn reason_for_blocker_kind(kind: GoalBlockerKind) -> GoalPauseReason {
    match kind {
        GoalBlockerKind::OperatorPause => GoalPauseReason::OperatorPaused,
        GoalBlockerKind::NeedsUserInput => GoalPauseReason::NeedsUserInput,
        GoalBlockerKind::HumanEscalation => GoalPauseReason::HumanEscalation,
        GoalBlockerKind::ExternalDependency => GoalPauseReason::ExternalDependency,
        GoalBlockerKind::Provider => GoalPauseReason::ProviderUnavailable,
        GoalBlockerKind::Verifier => GoalPauseReason::VerifierBlocked,
        GoalBlockerKind::Budget => GoalPauseReason::BudgetExhausted,
        GoalBlockerKind::RestartRecovery => GoalPauseReason::DaemonRestart,
    }
}

fn is_budget_pause_reason(reason: Option<GoalPauseReason>) -> bool {
    matches!(
        reason,
        Some(GoalPauseReason::BudgetExhausted | GoalPauseReason::BudgetUnavailable)
    )
}

fn merge_budget_pause(goal: &GoalTaskRecord, budget_pause: GoalPauseState) -> GoalPauseState {
    let mut blockers: Vec<_> = goal
        .blockers
        .iter()
        .filter(|blocker| blocker.kind != GoalBlockerKind::Budget)
        .cloned()
        .collect();
    blockers.extend(budget_pause.blockers);
    GoalPauseState {
        reason: goal.pause_reason.unwrap_or(budget_pause.reason),
        description: goal.pause_description.clone().or(budget_pause.description),
        blockers,
    }
}

fn remove_budget_pause(goal: &GoalTaskRecord) -> Option<GoalPauseState> {
    let blockers: Vec<_> = goal
        .blockers
        .iter()
        .filter(|blocker| blocker.kind != GoalBlockerKind::Budget)
        .cloned()
        .collect();
    if blockers.is_empty() {
        return None;
    }
    let reason = if is_budget_pause_reason(goal.pause_reason) {
        reason_for_blocker_kind(blockers[0].kind)
    } else {
        goal.pause_reason
            .unwrap_or_else(|| reason_for_blocker_kind(blockers[0].kind))
    };
    Some(GoalPauseState {
        reason,
        description: goal.pause_description.clone(),
        blockers,
    })
}

fn has_budget_blocker(goal: &GoalTaskRecord) -> bool {
    goal.blockers
        .iter()
        .any(|blocker| blocker.kind == GoalBlockerKind::Budget)
}

fn has_budget_pause(goal: &GoalTaskRecord) -> bool {
    is_budget_pause_reason(goal.pause_reason) || has_budget_blocker(goal)
}

fn blocker_kind_label(kind: GoalBlockerKind) -> String {
    let key = match kind {
        GoalBlockerKind::OperatorPause => "goal-blocker-kind-operator-pause",
        GoalBlockerKind::NeedsUserInput => "goal-blocker-kind-needs-user-input",
        GoalBlockerKind::HumanEscalation => "goal-blocker-kind-human-escalation",
        GoalBlockerKind::ExternalDependency => "goal-blocker-kind-external-dependency",
        GoalBlockerKind::Provider => "goal-blocker-kind-provider",
        GoalBlockerKind::Verifier => "goal-blocker-kind-verifier",
        GoalBlockerKind::Budget => "goal-blocker-kind-budget",
        GoalBlockerKind::RestartRecovery => "goal-blocker-kind-restart-recovery",
    };
    msg(key, &[])
}

fn blockers_summary(blockers: &[GoalBlocker]) -> Option<String> {
    let items = blockers
        .iter()
        .map(|blocker| {
            let kind = blocker_kind_label(blocker.kind);
            let message = blocker.message.trim();
            if message.is_empty() {
                kind
            } else {
                msg(
                    "goal-command-blocker-summary-item",
                    &[("kind", &kind), ("message", message)],
                )
            }
        })
        .collect::<Vec<_>>();
    (!items.is_empty()).then(|| items.join("; "))
}

/// Transport-neutral goal command verb after parsing user/model input.
///
/// The action chooses controller behavior, but it is still only command input.
/// Authorization, route, principal, and owner identity come from
/// [`GoalAdmissionContext`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalCommandAction {
    /// Render localized help without mutating goal state.
    Help,
    /// Create a new durable goal task or continue admission through the model
    /// tool path.
    Start,
    /// Replace the current goal's durable objective text.
    Objective,
    /// Report the latest visible state for a goal.
    Status,
    /// Replace effective limits and potentially resume a budget-paused goal.
    Budget,
    /// Explicitly pause the goal without making it terminal.
    Pause,
    /// Claim and continue a paused goal.
    Resume,
    /// Transition the goal task to a terminal cancellation state.
    Cancel,
}

/// Parsed budget option for a `/goal` command.
///
/// `Default` means the command did not mention the limit and the controller
/// should keep or derive the configured effective value. `Unlimited` is an
/// explicit operator request to clear that effective limit. `Limited` is an
/// explicit finite limit.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum GoalBudgetValue<T> {
    /// Keep the existing or configured effective limit.
    #[default]
    Default,
    /// Explicitly remove this effective limit.
    Unlimited,
    /// Replace this effective limit with the supplied finite value.
    Limited(T),
}

/// Operator-supplied budget mutations carried by a goal command.
///
/// Effective limits are stored on `GoalTaskRecord`. Consumed and remaining
/// budget are never stored here; they are derived from ledger usage records.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GoalBudgetOverrides {
    /// Token limit mutation for this command.
    pub token_limit: GoalBudgetValue<u64>,
    /// USD cost limit mutation for this command.
    pub cost_limit_usd: GoalBudgetValue<f64>,
}

/// Transport-neutral representation of a parsed goal command.
///
/// This is command input, not trusted lifecycle state. Channel handlers and the
/// model-callable tool both normalize into this type, then pass trusted runtime
/// facts separately through [`GoalAdmissionContext`]. Do not add sender, route,
/// principal, or owner fields here: those belong to ingress/runtime state and
/// eventually to the canonical [`TaskRecord`].
#[derive(Debug, Clone, PartialEq)]
pub struct GoalCommand {
    /// Requested controller action.
    pub action: GoalCommandAction,
    /// Untrusted operator/model objective text for `start` or `objective`.
    pub objective: Option<String>,
    /// Optional task id selector for inspection/control commands.
    ///
    /// `resume` deliberately does not use this selector: there is only one
    /// current paused goal in a trusted route/principal session, and completed
    /// goals are irreversible.
    pub task_id: Option<String>,
    /// Untrusted operator reason included with `/goal resume`.
    ///
    /// This is per-resume prompt input, not durable lifecycle state. The
    /// controller uses it to build the next continuation prompt and then drops
    /// it; trusted pause/blocker state remains in the task and goal registries.
    pub resume_reason: Option<String>,
    /// Requested effective budget changes.
    pub budgets: GoalBudgetOverrides,
}

/// Trusted runtime facts attached to goal admission.
///
/// The model and operator may provide the objective or subcommand text, but not
/// these fields. They are supplied by the ingress/runtime surface and are used
/// to bind goal lifecycle, routing, principal visibility, and continuation
/// delivery to the canonical task record. This struct is transient admission
/// input; only `continuation_context`, when present, is copied into durable
/// goal-continuation storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalAdmissionContext {
    /// Agent alias that owns the admitted goal.
    pub agent_alias: String,
    /// Runtime surface that parsed or invoked the command.
    pub command_surface: CommandSurface,
    /// Channel family, when the command originated from a channel turn.
    pub channel_type: Option<String>,
    /// Canonical route/reply target used for visibility and continuation.
    pub originator_route: Option<String>,
    /// Authenticated principal that originated the command, when available.
    pub principal_id: Option<String>,
    /// Legacy sanitizer-derived route for this same trusted inbound message.
    ///
    /// This is transient compatibility evidence, not a second durable route.
    /// It is accepted only with the exact task id, matching durable continuation
    /// context, and matching legacy principal, then immediately rebound.
    pub legacy_originator_route: Option<String>,
    /// Legacy sanitizer-derived principal for this same trusted inbound message.
    /// See [`Self::legacy_originator_route`] for the compatibility boundary.
    pub legacy_principal_id: Option<String>,
    /// Exact durable goal task bound to this controller turn, when one exists.
    ///
    /// Channel ingress leaves this unset for operator messages. The goal
    /// controller sets it only on trusted synthetic continuations so later
    /// admission, verifier, gate, and delegation lookups cannot drift to a
    /// replacement goal in the same route/principal context.
    pub goal_task_id: Option<String>,
    /// Minimal durable channel context needed to resume after restart.
    ///
    /// The context itself is persisted by the goal store. The rest of this
    /// admission struct is per-turn trust context and must not be stored here.
    pub continuation_context: Option<TaskContinuationContext>,
}

/// Result of applying a goal command to the durable control plane.
///
/// Callers use `continue_goal` to decide whether to enqueue another agent turn;
/// the task lifecycle itself is represented by `status` plus the canonical
/// task/goal rows, not by this transient result object. `message` is a localized
/// rendering of that state, not a policy input.
#[derive(Debug, Clone, PartialEq)]
pub struct GoalAdmission {
    /// Goal task affected by the command, when the action resolves one.
    pub task_id: Option<String>,
    /// Current canonical task status after command admission.
    pub status: TaskStatus,
    /// Localized user-visible status/error text.
    pub message: String,
    /// Untrusted operator text to include in the next continuation prompt.
    ///
    /// This is transient controller output for `/goal resume [reason]`, not
    /// durable lifecycle state. It must not be persisted into task or goal
    /// rows.
    pub continuation_reason: Option<String>,
    /// Whether the channel runtime should synthesize a continuation prompt.
    pub continue_goal: bool,
}

/// Verifier/controller decision for a completed model turn under goal mode.
///
/// This is the handoff object between the verifier and the goal controller. It
/// is intentionally transient: the controller is responsible for translating it
/// into task status, pause state, and user-visible messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoalTurnEvaluation {
    /// The verifier accepted the work as complete. The controller still owns
    /// the canonical terminal write.
    Completed {
        /// Canonical task id whose goal was evaluated.
        task_id: String,
        /// Localized channel/user status message rendered from the evaluation.
        message: String,
    },
    /// The verifier found more work and supplied untrusted notes for the next
    /// continuation prompt. These notes are prompt input only, not durable
    /// controller state.
    Continue {
        /// Canonical task id whose goal should continue.
        task_id: String,
        /// Original untrusted objective text from the goal extension.
        objective: String,
        /// Verifier-supplied untrusted notes for the next prompt.
        notes: String,
        /// Localized channel/user status message rendered from the evaluation.
        message: String,
    },
    /// The turn could not proceed without operator/provider/external action.
    Paused {
        /// Canonical task id whose goal was paused.
        task_id: String,
        /// Localized channel/user status message rendered from the pause.
        message: String,
    },
}

impl GoalAdmissionContext {
    pub fn new(agent_alias: impl Into<String>) -> Self {
        Self {
            agent_alias: agent_alias.into(),
            command_surface: CommandSurface::Channel,
            channel_type: None,
            originator_route: None,
            principal_id: None,
            legacy_originator_route: None,
            legacy_principal_id: None,
            goal_task_id: None,
            continuation_context: None,
        }
    }

    #[must_use]
    pub fn with_command_surface(mut self, command_surface: CommandSurface) -> Self {
        self.command_surface = command_surface;
        self
    }

    #[must_use]
    pub fn with_channel_type(mut self, channel_type: Option<String>) -> Self {
        self.channel_type = channel_type;
        self
    }

    #[must_use]
    pub fn with_originator_route(mut self, route: Option<String>) -> Self {
        self.originator_route = route;
        self
    }

    #[must_use]
    pub fn with_principal_id(mut self, principal_id: Option<String>) -> Self {
        self.principal_id = principal_id;
        self
    }

    #[must_use]
    pub fn with_legacy_identity(
        mut self,
        originator_route: Option<String>,
        principal_id: Option<String>,
    ) -> Self {
        self.legacy_originator_route = originator_route;
        self.legacy_principal_id = principal_id;
        self
    }

    #[must_use]
    pub fn with_goal_task_id(mut self, goal_task_id: Option<String>) -> Self {
        self.goal_task_id = goal_task_id;
        self
    }

    #[must_use]
    pub fn with_continuation_context(mut self, context: Option<TaskContinuationContext>) -> Self {
        self.continuation_context = context;
        self
    }
}

pub fn current_goal_admission_context() -> Option<GoalAdmissionContext> {
    GOAL_RUNTIME_SCOPE
        .try_with(|scope| scope.admission_binding.lock().context.clone())
        .ok()
        .flatten()
}

/// Resolve the configuration generation governing the current goal-aware turn.
pub fn current_goal_config() -> Option<Arc<Config>> {
    GOAL_RUNTIME_SCOPE
        .try_with(|scope| scope.config_resolver.as_ref().map(|resolver| resolver()))
        .ok()
        .flatten()
}

fn goal_policy_update_lock() -> &'static tokio::sync::RwLock<()> {
    static LOCK: OnceLock<tokio::sync::RwLock<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::RwLock::new(()))
}

/// Serialize a live goal-policy cutover against all goal command admissions.
///
/// The caller must durably revoke affected goals and publish the new config
/// while this write guard is held. `admit_goal_command` takes the matching read
/// guard, preventing a stale admission from landing between the cancellation
/// sweep and config publication.
pub async fn with_goal_policy_update_lock<F>(future: F) -> F::Output
where
    F: std::future::Future,
{
    let _guard = goal_policy_update_lock().write().await;
    future.await
}

/// Bind a just-admitted exact goal task to the current live turn.
///
/// This does not persist lifecycle state or resolve a goal by route: callers
/// may supply an id only after a successful controller transition returned it.
/// It lets later tools in the same turn use the durable continuation route
/// without falling back to mutable inbound delivery facts.
pub fn bind_current_goal_task(task_id: &str) -> bool {
    if task_id.trim().is_empty() {
        return false;
    }
    GOAL_RUNTIME_SCOPE
        .try_with(|scope| {
            let mut binding = scope.admission_binding.lock();
            if binding.reserved_for_admission {
                return false;
            }
            let Some(admission) = binding.context.as_mut() else {
                return false;
            };
            let bound = match admission.goal_task_id.as_deref() {
                Some(existing) => existing == task_id,
                None => {
                    admission.goal_task_id = Some(task_id.to_string());
                    true
                }
            };
            if bound && let Some(sink) = scope.task_binding_sink.as_ref() {
                sink(task_id);
            }
            bound
        })
        .unwrap_or(false)
}

/// Durably pause an exact running goal before its process-local worker stops.
///
/// Generic channel interruption is not a goal lifecycle authority. It may stop
/// a bound worker only after this guarded transition makes the durable goal
/// resumable, or after a concurrent pause/terminal transition has already
/// removed the worker's execution obligation.
pub async fn pause_running_goal_for_interruption(task_id: &str) -> Result<()> {
    if task_id.trim().is_empty() {
        bail!("cannot pause an interrupted goal without an exact task id");
    }
    let cp = control_plane().context("goal control plane unavailable during interruption")?;
    let pause = GoalPauseState {
        reason: GoalPauseReason::OperatorPaused,
        description: Some(msg("goal-interruption-pause-description", &[])),
        blockers: vec![GoalBlocker {
            kind: GoalBlockerKind::OperatorPause,
            message: msg("goal-interruption-blocker", &[]),
            payload: None,
        }],
    };
    if cp
        .goal_store
        .pause_goal_task_if_status(task_id, TaskStatus::Running, pause)
        .await
        .with_context(|| format!("pause interrupted goal {task_id}"))?
    {
        return Ok(());
    }

    let current = cp
        .store
        .get(task_id)
        .await
        .with_context(|| format!("reload interrupted goal {task_id}"))?
        .ok_or_else(|| {
            anyhow::Error::msg(format!("interrupted goal {task_id} no longer exists"))
        })?;
    if current.status == TaskStatus::Running {
        bail!("interrupted goal {task_id} remained running after guarded pause");
    }
    Ok(())
}

/// Verify that this model-tool turn can bind one exact admitted goal task.
///
/// The durable task record remains the source of truth. This only guards the
/// transient task-local binding used by later work in the same turn, and must
/// run before a model tool performs a durable start or resume transition.
pub fn ensure_current_goal_task_binding_available() -> Result<()> {
    GOAL_RUNTIME_SCOPE
        .try_with(|scope| {
            let binding = scope.admission_binding.lock();
            let admission = binding
                .context
                .as_ref()
                .ok_or_else(|| anyhow::Error::msg("goal admission context unavailable"))?;
            if binding.reserved_for_admission || admission.goal_task_id.is_some() {
                anyhow::bail!("goal admission already has an exact live task binding");
            }
            Ok(())
        })
        .map_err(|_| anyhow::Error::msg("goal admission context unavailable"))?
}

pub fn reserve_current_goal_task_binding() -> Result<GoalTaskBindingReservation> {
    GOAL_RUNTIME_SCOPE
        .try_with(|scope| {
            let mut binding = scope.admission_binding.lock();
            let admission = binding
                .context
                .as_ref()
                .ok_or_else(|| anyhow::Error::msg("goal admission context unavailable"))?;
            if admission.goal_task_id.is_some() {
                anyhow::bail!("goal admission already has an exact live task binding");
            }
            if binding.reserved_for_admission {
                anyhow::bail!("goal admission task binding is reserved");
            }
            binding.reserved_for_admission = true;
            Ok(GoalTaskBindingReservation {
                binding: Arc::clone(&scope.admission_binding),
                task_binding_sink: scope.task_binding_sink.clone(),
                active: true,
            })
        })
        .map_err(|_| anyhow::Error::msg("goal admission context unavailable"))?
}

/// Whether a model-tool admission permit exclusively owns the current turn's
/// exact-goal binding. Controller admission must leave that binding to the
/// permit after its durable transition succeeds.
pub fn current_goal_task_binding_is_reserved() -> bool {
    GOAL_RUNTIME_SCOPE
        .try_with(|scope| scope.admission_binding.lock().reserved_for_admission)
        .unwrap_or(false)
}

pub struct GoalTaskBindingReservation {
    binding: Arc<parking_lot::Mutex<GoalAdmissionBindingState>>,
    task_binding_sink: Option<GoalTaskBindingSink>,
    active: bool,
}

impl GoalTaskBindingReservation {
    pub fn bind(mut self, task_id: String) {
        if task_id.is_empty() {
            return;
        }
        let mut binding = self.binding.lock();
        let mut bound = false;
        if binding.reserved_for_admission
            && let Some(admission) = binding.context.as_mut()
            && admission.goal_task_id.is_none()
        {
            admission.goal_task_id = Some(task_id.clone());
            bound = true;
        }
        self.active = false;
        binding.reserved_for_admission = false;
        drop(binding);
        if bound && let Some(sink) = self.task_binding_sink.as_ref() {
            sink(&task_id);
        }
    }
}

impl Drop for GoalTaskBindingReservation {
    fn drop(&mut self) {
        if self.active {
            self.binding.lock().reserved_for_admission = false;
        }
    }
}

/// Durably stop the exact active goal after a provider boundary reports that
/// its usage cannot be attributed. The task-local admission context supplies
/// the task id; this never falls back to route or principal lookup.
pub async fn pause_goal_for_accounting_failure(task_id: &str, error: &anyhow::Error) -> Result<()> {
    if !is_goal_accounting_failure(error) {
        return Ok(());
    }
    let Some(cp) = control_plane() else {
        return Ok(());
    };
    let task = cp
        .store
        .get(task_id)
        .await?
        .ok_or_else(|| anyhow::Error::msg("goal task missing"))?;
    let goal = cp
        .goal_store
        .get_goal_task(task_id)
        .await?
        .ok_or_else(|| anyhow::Error::msg("goal extension missing"))?;
    let resolved = TaskGoal::new(task, goal);
    if !resolved.is_running() {
        return Ok(());
    }
    let cause = if is_goal_accounting_pricing_failure(error) {
        GoalBudgetUnavailableCause::CostPricingUnavailable
    } else {
        GoalBudgetUnavailableCause::UsageUnavailable
    };
    let pause = goal_accounting_unavailable_pause(resolved.goal(), cause);
    let tracker = crate::agent::cost::current_goal_usage_tracker();
    let usage = goal_usage_totals_from_tracker(tracker.as_deref(), task_id, true);
    let budget = goal_budget_summary(resolved.goal(), usage.as_ref());
    let admission =
        pause_goal_for_resolved_task_with_budget(cp.goal_store.as_ref(), resolved, pause, budget)
            .await?;
    publish_goal_state_update(&admission);
    Ok(())
}

fn current_goal_runtime_scope() -> GoalRuntimeScope {
    GOAL_RUNTIME_SCOPE
        .try_with(Clone::clone)
        .unwrap_or_default()
}

pub async fn scope_goal_runtime<F>(scope: GoalRuntimeScope, future: F) -> F::Output
where
    F: std::future::Future,
{
    GOAL_RUNTIME_SCOPE.scope(scope, future).await
}

pub async fn scope_goal_admission_context<F>(
    ctx: Option<GoalAdmissionContext>,
    future: F,
) -> F::Output
where
    F: std::future::Future,
{
    GOAL_RUNTIME_SCOPE
        .scope(
            current_goal_runtime_scope().with_admission_context(ctx),
            future,
        )
        .await
}

pub async fn scope_goal_state_updates<F>(sink: Option<GoalStateUpdateSink>, future: F) -> F::Output
where
    F: std::future::Future,
{
    GOAL_RUNTIME_SCOPE
        .scope(
            current_goal_runtime_scope().with_state_update_sink(sink),
            future,
        )
        .await
}

/// Scope the per-turn marker that says the channel orchestrator should run
/// goal verifier/evaluation after the model turn completes.
///
/// Admission facts alone are not enough: ordinary same-route traffic must be
/// able to start a goal through the model tool without being treated as work
/// for a previously active goal.
pub async fn scope_goal_turn_evaluation_marker<F>(
    marker: Option<Arc<AtomicBool>>,
    future: F,
) -> F::Output
where
    F: std::future::Future,
{
    GOAL_RUNTIME_SCOPE
        .scope(
            current_goal_runtime_scope().with_turn_evaluation_marker(marker),
            future,
        )
        .await
}

/// Promote the current turn into goal work after trusted goal admission
/// succeeds inside the model tool loop.
pub fn mark_current_goal_turn_for_evaluation() {
    let _ = GOAL_RUNTIME_SCOPE.try_with(|scope| {
        if let Some(marker) = &scope.turn_evaluation_requested {
            marker.store(true, Ordering::Release);
        }
    });
}

/// Report whether the current task-local turn should be subject to goal
/// verifier/evaluation and goal-only delegation policy.
pub fn current_goal_turn_evaluation_requested() -> bool {
    GOAL_RUNTIME_SCOPE
        .try_with(|scope| {
            scope
                .turn_evaluation_requested
                .as_ref()
                .is_some_and(|marker| marker.load(Ordering::Acquire))
        })
        .unwrap_or(false)
}

/// Clone the current turn-evaluation marker so spawned foreground work can
/// re-enter the same transient goal-work decision boundary.
///
/// This intentionally shares the marker instead of copying its boolean value:
/// if a child `goal_start` promotes the turn, the parent orchestrator must see
/// the same promotion before deciding whether to run post-turn goal evaluation.
pub fn current_goal_turn_evaluation_marker() -> Option<Arc<AtomicBool>> {
    GOAL_RUNTIME_SCOPE
        .try_with(|scope| scope.turn_evaluation_requested.clone())
        .ok()
        .flatten()
}

/// Scope whether the current model-requested tool batch contains a goal
/// admission/control tool.
///
/// This is a conservative policy marker, not goal state. It lets sibling tools
/// refuse actions that cannot be made safe until admission completes, even when
/// the model listed those siblings before the admission tool or the executor
/// would otherwise consider the batch parallelizable.
pub async fn scope_goal_start_tool_batch<F>(
    contains_goal_admission_tool: bool,
    future: F,
) -> F::Output
where
    F: std::future::Future,
{
    let inherited = current_goal_start_tool_batch_requested();
    GOAL_START_TOOL_BATCH
        .scope(inherited || contains_goal_admission_tool, future)
        .await
}

/// Report whether the active tool batch is attempting goal admission/control.
pub fn current_goal_start_tool_batch_requested() -> bool {
    GOAL_START_TOOL_BATCH
        .try_with(|value| *value)
        .unwrap_or(false)
}

fn publish_goal_state_update(admission: &GoalAdmission) {
    let _ = GOAL_RUNTIME_SCOPE.try_with(|scope| {
        if let Some(sink) = &scope.state_update_sink {
            let message = msg(
                "channel-goal-state-update",
                &[("message", &admission.message)],
            );
            sink.send(GoalStateUpdateEvent::Status(message));
        }
    });
}

fn publish_goal_verifier_started(task_id: &str, budget: &str) {
    let message = msg(
        "goal-command-verifying",
        &[("task_id", task_id), ("budget", budget)],
    );
    let _ = GOAL_RUNTIME_SCOPE.try_with(|scope| {
        if let Some(sink) = &scope.state_update_sink {
            sink.send(GoalStateUpdateEvent::VerifierStarted(message));
        }
    });
}

pub fn parse_goal_command(input: &str) -> Result<GoalCommand> {
    let without_prefix = strip_goal_command_prefix(input)?;
    let mut parts = without_prefix.splitn(2, char::is_whitespace);
    let Some(action) = parts.next().filter(|s| !s.is_empty()) else {
        bail!("{}", msg("goal-command-error-missing-action", &[]));
    };
    let action = action.to_ascii_lowercase();
    let rest = parts.next().unwrap_or("").trim();
    match action.as_str() {
        "help" | "--help" | "-h" => Ok(GoalCommand {
            action: GoalCommandAction::Help,
            objective: None,
            task_id: None,
            resume_reason: None,
            budgets: GoalBudgetOverrides::default(),
        }),
        "start" => {
            let (budgets, objective) = parse_start_payload(rest)?;
            if objective.is_empty() {
                bail!("{}", msg("goal-command-error-missing-objective", &[]));
            }
            Ok(GoalCommand {
                action: GoalCommandAction::Start,
                objective: Some(objective),
                task_id: None,
                resume_reason: None,
                budgets,
            })
        }
        "objective" => {
            let objective = parse_objective_payload(rest)?;
            Ok(GoalCommand {
                action: GoalCommandAction::Objective,
                objective: Some(objective),
                task_id: None,
                resume_reason: None,
                budgets: GoalBudgetOverrides::default(),
            })
        }
        "status" => Ok(GoalCommand {
            action: GoalCommandAction::Status,
            objective: None,
            task_id: parse_optional_task_id(rest)?,
            resume_reason: None,
            budgets: GoalBudgetOverrides::default(),
        }),
        "budget" => {
            let budgets = parse_budget_payload(rest)?;
            Ok(GoalCommand {
                action: GoalCommandAction::Budget,
                objective: None,
                task_id: None,
                resume_reason: None,
                budgets,
            })
        }
        "pause" => Ok(GoalCommand {
            action: GoalCommandAction::Pause,
            objective: nonempty(rest),
            task_id: None,
            resume_reason: None,
            budgets: GoalBudgetOverrides::default(),
        }),
        "resume" => {
            let resume_reason = parse_resume_payload(rest)?;
            Ok(GoalCommand {
                action: GoalCommandAction::Resume,
                objective: None,
                task_id: None,
                resume_reason,
                budgets: GoalBudgetOverrides::default(),
            })
        }
        "cancel" => Ok(GoalCommand {
            action: GoalCommandAction::Cancel,
            objective: None,
            task_id: parse_optional_task_id(rest)?,
            resume_reason: None,
            budgets: GoalBudgetOverrides::default(),
        }),
        other => {
            bail!(
                "{}",
                msg("goal-command-error-unknown-action", &[("action", other)])
            )
        }
    }
}

fn strip_goal_command_prefix(input: &str) -> Result<&str> {
    let trimmed = input.trim();
    let (command_token, rest) = trimmed
        .split_once(char::is_whitespace)
        .map_or((trimmed, ""), |(token, rest)| (token, rest.trim()));
    if !command_token.starts_with('/') {
        bail!(
            "{}",
            msg(
                "goal-command-error-invalid-command",
                &[("command", command_token)]
            )
        );
    }
    let Some(command) = command_by_name(command_token) else {
        bail!(
            "{}",
            msg(
                "goal-command-error-invalid-command",
                &[("command", command_token)]
            )
        );
    };
    if command.id != BuiltinCommandId::Goal {
        bail!(
            "{}",
            msg(
                "goal-command-error-invalid-command",
                &[("command", command_token)]
            )
        );
    }
    Ok(rest)
}

fn parse_start_payload(input: &str) -> Result<(GoalBudgetOverrides, String)> {
    let mut budgets = GoalBudgetOverrides::default();
    let mut rest = input.trim();
    while let Some(next) = rest.strip_prefix("--") {
        let (flag, tail) = next
            .split_once(char::is_whitespace)
            .map_or((next, ""), |(flag, tail)| (flag, tail.trim_start()));
        parse_budget_flag(flag, &mut budgets)?;
        rest = tail;
    }
    Ok((budgets, rest.trim().to_string()))
}

fn parse_budget_payload(input: &str) -> Result<GoalBudgetOverrides> {
    let mut budgets = GoalBudgetOverrides::default();
    let mut saw_value = false;
    for token in input.split_whitespace() {
        let flag = token.strip_prefix("--").ok_or_else(|| {
            anyhow::Error::msg(msg(
                "goal-command-error-invalid-budget-flag",
                &[("flag", token)],
            ))
        })?;
        parse_budget_flag(flag, &mut budgets)?;
        saw_value = true;
    }
    if !saw_value {
        bail!("{}", msg("goal-command-error-missing-budget", &[]));
    }
    Ok(budgets)
}

fn parse_resume_payload(input: &str) -> Result<Option<String>> {
    let rest = input.trim();
    if rest.is_empty() {
        return Ok(None);
    }
    Ok(Some(rest.to_string()))
}

fn parse_objective_payload(input: &str) -> Result<String> {
    let objective = input.trim();
    if objective.is_empty() {
        bail!("{}", msg("goal-command-error-missing-objective", &[]));
    }
    Ok(objective.to_string())
}

fn parse_optional_task_id(input: &str) -> Result<Option<String>> {
    let rest = input.trim();
    if rest.is_empty() {
        return Ok(None);
    }
    if let Some((_task_id, tail)) = rest.split_once(char::is_whitespace) {
        let args = tail.trim();
        bail!(
            "{}",
            msg("goal-command-error-unexpected-arguments", &[("args", args)])
        );
    }
    Ok(Some(rest.to_string()))
}

fn parse_budget_flag(flag: &str, budgets: &mut GoalBudgetOverrides) -> Result<()> {
    let (name, value) = flag.split_once('=').ok_or_else(|| {
        anyhow::Error::msg(msg(
            "goal-command-error-invalid-budget-flag",
            &[("flag", flag)],
        ))
    })?;
    match name {
        "tokens" => {
            budgets.token_limit = parse_token_budget_value(value)?;
            Ok(())
        }
        "cost" => {
            budgets.cost_limit_usd = parse_cost_budget_value(value)?;
            Ok(())
        }
        _ => bail!(
            "{}",
            msg("goal-command-error-invalid-budget-flag", &[("flag", flag)])
        ),
    }
}

fn parse_token_budget_value(value: &str) -> Result<GoalBudgetValue<u64>> {
    let trimmed = value.trim();
    if trimmed.eq_ignore_ascii_case("unlimited") {
        return Ok(GoalBudgetValue::Unlimited);
    }
    let parsed = trimmed.parse::<u64>().map_err(|_| {
        anyhow::Error::msg(msg(
            "goal-command-error-invalid-token-budget",
            &[("value", trimmed)],
        ))
    })?;
    if parsed == 0 {
        bail!(
            "{}",
            msg(
                "goal-command-error-invalid-token-budget",
                &[("value", trimmed)]
            )
        );
    }
    Ok(GoalBudgetValue::Limited(parsed))
}

fn parse_cost_budget_value(value: &str) -> Result<GoalBudgetValue<f64>> {
    let trimmed = value.trim();
    if trimmed.eq_ignore_ascii_case("unlimited") {
        return Ok(GoalBudgetValue::Unlimited);
    }
    let parsed = trimmed.parse::<f64>().map_err(|_| {
        anyhow::Error::msg(msg(
            "goal-command-error-invalid-cost-budget",
            &[("value", trimmed)],
        ))
    })?;
    if !parsed.is_finite() || parsed <= 0.0 {
        bail!(
            "{}",
            msg(
                "goal-command-error-invalid-cost-budget",
                &[("value", trimmed)]
            )
        );
    }
    Ok(GoalBudgetValue::Limited(parsed))
}

fn resolve_goal_limits(
    config: &Config,
    budgets: GoalBudgetOverrides,
) -> (Option<u64>, Option<f64>) {
    let token_limit = match budgets.token_limit {
        GoalBudgetValue::Default => config.goal.token_budget,
        GoalBudgetValue::Unlimited => None,
        GoalBudgetValue::Limited(value) => Some(value),
    };
    let cost_limit_usd = match budgets.cost_limit_usd {
        GoalBudgetValue::Default => config.goal.cost_budget_usd,
        GoalBudgetValue::Unlimited => None,
        GoalBudgetValue::Limited(value) => Some(value),
    };
    (token_limit, cost_limit_usd)
}

fn ensure_goal_admitted_by_config(
    ctx: &GoalAdmissionContext,
    config: &Config,
    agent_config: Option<&AliasedAgentConfig>,
) -> Result<()> {
    if !config.goal.enabled {
        bail!("{}", msg("goal-command-error-disabled", &[]));
    }
    let Some(agent_config) = agent_config else {
        bail!("{}", msg("goal-command-error-agent-disabled", &[]));
    };
    if !agent_config.enabled || !agent_config.goal.enabled {
        bail!("{}", msg("goal-command-error-agent-disabled", &[]));
    }
    let surface = ctx.command_surface.as_str();
    if !config
        .goal
        .allowed_command_surfaces
        .iter()
        .any(|candidate| candidate.trim() == surface)
    {
        bail!(
            "{}",
            msg(
                "goal-command-error-surface-disabled",
                &[("surface", surface)]
            )
        );
    }
    if ctx.command_surface == CommandSurface::Channel {
        let Some(channel_type) = ctx
            .channel_type
            .as_deref()
            .map(str::trim)
            .filter(|channel_type| !channel_type.is_empty())
        else {
            bail!("{}", msg("goal-command-error-channel-type-missing", &[]));
        };
        if !config
            .goal
            .allowed_channel_types
            .iter()
            .any(|candidate| candidate.trim() == channel_type)
        {
            bail!(
                "{}",
                msg(
                    "goal-command-error-channel-disabled",
                    &[("channel_type", channel_type)]
                )
            );
        }
        let exact_binding_allowed = ctx
            .continuation_context
            .as_ref()
            .and_then(|context| context.channel_alias.as_deref())
            .is_some_and(|channel_alias| {
                config.enabled_channel_owned_by_agent(&ctx.agent_alias, channel_type, channel_alias)
            });
        if !exact_binding_allowed {
            bail!(
                "{}",
                msg(
                    "goal-command-error-channel-binding-disabled",
                    &[("channel_type", channel_type)]
                )
            );
        }
    }
    Ok(())
}

fn goal_channel_binding_is_allowed(
    agent_alias: &str,
    channel_type: &str,
    context: &TaskContinuationContext,
    enabled_channels: &HashMap<String, String>,
) -> bool {
    context
        .channel_alias
        .as_deref()
        .is_some_and(|channel_alias| {
            let channel_ref = format!("{channel_type}.{channel_alias}");
            enabled_channels.get(&channel_ref).map(String::as_str) == Some(agent_alias)
        })
}

fn goal_channel_type(channel: &str) -> &str {
    channel
        .split_once(':')
        .map_or(channel, |(channel_type, _)| channel_type)
}

fn goal_control_binding_is_allowed_with_channels(
    binding: &ActiveGoalControlBinding,
    config: &Config,
    enabled_channels: &HashMap<String, String>,
) -> bool {
    if !config.goal.enabled {
        return false;
    }
    let Some(agent) = config.agent(&binding.agent) else {
        return false;
    };
    if !agent.enabled || !agent.goal.enabled {
        return false;
    }
    if !config
        .goal
        .allowed_command_surfaces
        .iter()
        .any(|surface| surface.trim() == CommandSurface::Channel.as_str())
    {
        return false;
    }

    // Goal admission is currently channel-only in the shared command
    // catalogue, and every admitted channel goal persists this continuation
    // binding. If another surface is added, it must first add its own durable
    // binding instead of making revocation guess from route text.
    let Some(context) = binding.continuation_context.as_ref() else {
        return false;
    };
    let channel_type = goal_channel_type(context.channel.trim());
    !channel_type.is_empty()
        && config
            .goal
            .allowed_channel_types
            .iter()
            .any(|allowed| allowed.trim() == channel_type)
        && goal_channel_binding_is_allowed(&binding.agent, channel_type, context, enabled_channels)
}

fn enabled_goal_channels(config: &Config) -> HashMap<String, String> {
    config
        .channels_by_alias()
        .into_iter()
        .filter_map(|channel| {
            let owner = channel.owning_agent?;
            channel
                .enabled
                .then(|| (format!("{}.{}", channel.channel_type, channel.alias), owner))
        })
        .collect()
}

#[cfg(test)]
fn goal_control_binding_is_allowed(binding: &ActiveGoalControlBinding, config: &Config) -> bool {
    let enabled_channels = enabled_goal_channels(config);
    goal_control_binding_is_allowed_with_channels(binding, config, &enabled_channels)
}

/// Resolve every active exact goal no longer authorized by `config`.
///
/// This is the read-only phase of live policy reconciliation. Callers that
/// coordinate configuration generations may validate that their generation is
/// still authoritative after this awaited lookup and before committing the
/// returned exact ids.
pub async fn goal_ids_revoked_by_config(
    goal_store: &dyn GoalTaskRegistry,
    config: &Config,
) -> Result<Vec<String>> {
    let bindings = goal_store
        .list_active_goal_control_bindings()
        .await
        .context("list active goals for live policy reconciliation")?;
    let enabled_channels = enabled_goal_channels(config);
    Ok(bindings
        .into_iter()
        .filter(|binding| {
            !goal_control_binding_is_allowed_with_channels(binding, config, &enabled_channels)
        })
        .map(|binding| binding.task_id)
        .collect())
}

/// Commit a previously resolved exact live-policy revocation plan.
pub async fn cancel_goals_for_policy_revocation(
    goal_store: &dyn GoalTaskRegistry,
    revoked: &[String],
) -> Result<Vec<String>> {
    goal_store
        .cancel_active_goals_for_policy_revocation(
            revoked,
            &msg("goal-terminal-reason-cancelled-by-controller", &[]),
        )
        .await
        .context("cancel goals revoked by live policy")
}

pub async fn admit_goal_command(
    ctx: GoalAdmissionContext,
    command: GoalCommand,
    fallback_config: &Config,
    fallback_agent_config: Option<&AliasedAgentConfig>,
) -> Result<GoalAdmission> {
    let _policy_guard = goal_policy_update_lock().read().await;
    let live_config = current_goal_config();
    let (config, agent_config) = match live_config.as_deref() {
        Some(config) => (config, config.agent(&ctx.agent_alias)),
        None => (
            fallback_config,
            fallback_config
                .agent(&ctx.agent_alias)
                .or(fallback_agent_config),
        ),
    };
    ensure_goal_admitted_by_config(&ctx, config, agent_config)?;
    if command.action == GoalCommandAction::Help {
        return Ok(GoalAdmission {
            task_id: None,
            status: TaskStatus::Running,
            message: msg("goal-command-help", &[]),
            continuation_reason: None,
            continue_goal: false,
        });
    }
    let cp = control_plane()
        .with_context(|| msg("goal-command-error-control-plane-unavailable", &[]))?;
    let admission = match command.action {
        GoalCommandAction::Help => unreachable!("handled before control-plane access"),
        GoalCommandAction::Start => {
            let objective = command
                .objective
                .with_context(|| msg("goal-command-error-missing-objective", &[]))?;
            let (token_limit, cost_limit_usd) = resolve_goal_limits(config, command.budgets);
            start_goal(
                cp.goal_store.as_ref(),
                &cp.boot_id,
                ctx,
                objective,
                token_limit,
                cost_limit_usd,
                Some(config),
            )
            .await
        }
        GoalCommandAction::Objective => {
            let objective = command
                .objective
                .with_context(|| msg("goal-command-error-missing-objective", &[]))?;
            update_goal_objective(
                cp.store.as_ref(),
                cp.goal_store.as_ref(),
                &ctx,
                objective,
                Some(config),
            )
            .await
        }
        GoalCommandAction::Status => {
            status_goal(
                cp.store.as_ref(),
                cp.goal_store.as_ref(),
                &ctx,
                command.task_id,
                Some(config),
            )
            .await
        }
        GoalCommandAction::Budget => {
            update_goal_budget(
                cp.store.as_ref(),
                cp.goal_store.as_ref(),
                &cp.boot_id,
                &ctx,
                command.budgets,
                Some(config),
            )
            .await
        }
        GoalCommandAction::Pause => {
            let description = command.objective;
            pause_goal_for_blocker(
                cp.store.as_ref(),
                cp.goal_store.as_ref(),
                &ctx,
                command.task_id,
                Some(config),
                GoalPauseState {
                    reason: GoalPauseReason::OperatorPaused,
                    description: description.clone(),
                    blockers: description
                        .map(|message| {
                            vec![GoalBlocker {
                                kind: GoalBlockerKind::OperatorPause,
                                message,
                                payload: None,
                            }]
                        })
                        .unwrap_or_default(),
                },
            )
            .await
        }
        GoalCommandAction::Resume => {
            let resume_reason = command.resume_reason;
            resume_goal(
                cp.store.as_ref(),
                cp.goal_store.as_ref(),
                &cp.boot_id,
                &ctx,
                resume_reason,
                Some(config),
            )
            .await
        }
        GoalCommandAction::Cancel => {
            cancel_goal(
                cp.store.as_ref(),
                cp.goal_store.as_ref(),
                &ctx,
                command.task_id,
                Some(config),
            )
            .await
        }
    }?;
    if admission.continue_goal && GOAL_RUNTIME_SCOPE.try_with(|_| ()).is_ok() {
        let task_id = admission.task_id.as_deref().ok_or_else(|| {
            anyhow::Error::msg("continuing goal admission returned no exact task id")
        })?;
        if !bind_current_goal_task(task_id) {
            anyhow::bail!("goal admission could not bind its exact live task");
        }
    }
    publish_goal_state_update(&admission);
    Ok(admission)
}

pub async fn evaluate_goal_turn(
    ctx: &GoalAdmissionContext,
    config: &Config,
    candidate_summary: &str,
) -> Result<Option<GoalTurnEvaluation>> {
    evaluate_goal_turn_with_verifier(ctx, config, candidate_summary, &LlmGoalVerifier).await
}

async fn complete_goal_after_verification(
    store: &dyn TaskRegistry,
    goal_store: &dyn GoalTaskRegistry,
    ctx: &GoalAdmissionContext,
    cost_tracker: Option<&CostTracker>,
    current: TaskGoal,
    candidate_summary: &str,
    final_usage: Option<GoalUsageTotals>,
) -> Result<Option<GoalTurnEvaluation>> {
    let task_id = current.task_id().to_string();
    if !goal_store
        .complete_running_goal_task_if_limits(
            &task_id,
            current.goal().effective_token_limit,
            current.goal().effective_cost_limit_usd,
            candidate_summary.to_string(),
        )
        .await
        .with_context(|| msg("goal-command-error-update-failed", &[("task_id", &task_id)]))?
    {
        // A completion CAS can lose to a concurrent budget edit while the
        // task remains running. Re-resolve the exact durable task and return
        // it to the live continuation path rather than abandoning a running
        // goal without an executor.
        let refreshed = resolve_goal(store, goal_store, ctx, Some(task_id.clone())).await?;
        if !refreshed.is_running() {
            return Ok(None);
        }
        let refreshed_usage =
            goal_usage_totals_from_tracker(cost_tracker, refreshed.task_id(), true);
        if let Some(pause) = goal_accounting_gate_pause(refreshed.goal(), refreshed_usage.as_ref())
        {
            let budget = goal_budget_summary(refreshed.goal(), refreshed_usage.as_ref());
            let admission =
                pause_goal_for_resolved_task_with_budget(goal_store, refreshed, pause, budget)
                    .await?;
            publish_goal_state_update(&admission);
            return Ok(Some(GoalTurnEvaluation::Paused {
                task_id,
                message: admission.message,
            }));
        }
        let budget = goal_budget_summary(refreshed.goal(), refreshed_usage.as_ref());
        let admission = GoalAdmission {
            task_id: Some(task_id.clone()),
            status: TaskStatus::Running,
            message: msg(
                "goal-command-continuing",
                &[("task_id", &task_id), ("budget", &budget)],
            ),
            continuation_reason: None,
            continue_goal: true,
        };
        publish_goal_state_update(&admission);
        return Ok(Some(GoalTurnEvaluation::Continue {
            task_id,
            objective: refreshed.objective().to_string(),
            notes: "The goal budget changed while completion was being recorded; \
                    re-evaluate the objective under the current limits."
                .to_string(),
            message: admission.message,
        }));
    }
    let budget = goal_budget_summary(current.goal(), final_usage.as_ref());
    let admission = GoalAdmission {
        task_id: Some(task_id.clone()),
        status: TaskStatus::Completed,
        message: msg(
            "goal-command-completed",
            &[("task_id", &task_id), ("budget", &budget)],
        ),
        continuation_reason: None,
        continue_goal: false,
    };
    publish_goal_state_update(&admission);
    Ok(Some(GoalTurnEvaluation::Completed {
        task_id,
        message: admission.message,
    }))
}

pub async fn evaluate_goal_turn_with_verifier(
    ctx: &GoalAdmissionContext,
    config: &Config,
    candidate_summary: &str,
    verifier: &dyn GoalVerifier,
) -> Result<Option<GoalTurnEvaluation>> {
    let cp = match control_plane() {
        Some(cp) => cp,
        None => return Ok(None),
    };
    let Some(resolved) =
        latest_active_resolved_goal(cp.store.as_ref(), cp.goal_store.as_ref(), ctx).await?
    else {
        return Ok(None);
    };
    if !resolved.is_running() {
        return Ok(None);
    }

    let cost_tracker = goal_usage_ledger(Some(config));
    let usage = goal_usage_totals_from_tracker(cost_tracker.as_deref(), resolved.task_id(), true);
    if let Some(pause) = goal_accounting_gate_pause(resolved.goal(), usage.as_ref()) {
        let task_id = resolved.task_id().to_string();
        let budget = goal_budget_summary(resolved.goal(), usage.as_ref());
        let admission = pause_goal_for_resolved_task_with_budget(
            cp.goal_store.as_ref(),
            resolved,
            pause,
            budget,
        )
        .await?;
        publish_goal_state_update(&admission);
        return Ok(Some(GoalTurnEvaluation::Paused {
            task_id,
            message: admission.message,
        }));
    }

    if !config.goal.verifier.enabled {
        return complete_goal_after_verification(
            cp.store.as_ref(),
            cp.goal_store.as_ref(),
            ctx,
            cost_tracker.as_deref(),
            resolved,
            candidate_summary,
            usage,
        )
        .await;
    }

    let verifier_goal_context = ctx
        .clone()
        .with_goal_task_id(Some(resolved.task_id().to_string()));
    let budget = goal_budget_summary(resolved.goal(), usage.as_ref());
    publish_goal_verifier_started(resolved.task_id(), &budget);

    let verifier_decision = verifier
        .verify(GoalVerificationRequest {
            config,
            agent_alias: resolved.agent(),
            goal_context: &verifier_goal_context,
            goal: resolved.goal(),
            candidate_summary,
            cost_tracker: cost_tracker.clone(),
        })
        .await;

    match verifier_decision {
        Ok(GoalVerifierDecision::Complete { notes: _ }) => {
            let current = resolve_goal(
                cp.store.as_ref(),
                cp.goal_store.as_ref(),
                ctx,
                Some(resolved.task_id().to_string()),
            )
            .await?;
            if !current.is_running() {
                return Ok(None);
            }
            let task_id = current.task_id().to_string();
            let final_usage =
                goal_usage_totals_from_tracker(cost_tracker.as_deref(), &task_id, true);
            if let Some(pause) = goal_accounting_gate_pause(current.goal(), final_usage.as_ref()) {
                let budget = goal_budget_summary(current.goal(), final_usage.as_ref());
                let admission = pause_goal_for_resolved_task_with_budget(
                    cp.goal_store.as_ref(),
                    current,
                    pause,
                    budget,
                )
                .await?;
                publish_goal_state_update(&admission);
                return Ok(Some(GoalTurnEvaluation::Paused {
                    task_id,
                    message: admission.message,
                }));
            }
            complete_goal_after_verification(
                cp.store.as_ref(),
                cp.goal_store.as_ref(),
                ctx,
                cost_tracker.as_deref(),
                current,
                candidate_summary,
                final_usage,
            )
            .await
        }
        Ok(GoalVerifierDecision::Continue { notes }) => {
            let current = resolve_goal(
                cp.store.as_ref(),
                cp.goal_store.as_ref(),
                ctx,
                Some(resolved.task_id().to_string()),
            )
            .await?;
            if !current.is_running() {
                return Ok(None);
            }
            let task_id = current.task_id().to_string();
            let usage = goal_usage_totals_from_tracker(cost_tracker.as_deref(), &task_id, true);
            if let Some(pause) = goal_accounting_gate_pause(current.goal(), usage.as_ref()) {
                let budget = goal_budget_summary(current.goal(), usage.as_ref());
                let admission = pause_goal_for_resolved_task_with_budget(
                    cp.goal_store.as_ref(),
                    current,
                    pause,
                    budget,
                )
                .await?;
                publish_goal_state_update(&admission);
                return Ok(Some(GoalTurnEvaluation::Paused {
                    task_id,
                    message: admission.message,
                }));
            }
            let budget = goal_budget_summary(current.goal(), usage.as_ref());
            let admission = GoalAdmission {
                task_id: Some(task_id.clone()),
                status: TaskStatus::Running,
                message: msg(
                    "goal-command-continuing",
                    &[("task_id", &task_id), ("budget", &budget)],
                ),
                continuation_reason: None,
                continue_goal: true,
            };
            publish_goal_state_update(&admission);
            Ok(Some(GoalTurnEvaluation::Continue {
                task_id,
                objective: current.objective().to_string(),
                notes,
                message: admission.message,
            }))
        }
        Ok(GoalVerifierDecision::Blocked { pause }) => {
            let task_id = resolved.task_id().to_string();
            let admission =
                pause_goal_for_known_blocker(cp.goal_store.as_ref(), resolved, Some(config), pause)
                    .await?;
            publish_goal_state_update(&admission);
            Ok(Some(GoalTurnEvaluation::Paused {
                task_id,
                message: admission.message,
            }))
        }
        Err(error) => {
            let task_id = resolved.task_id().to_string();
            if is_goal_accounting_failure(&error) {
                // Provider boundaries durably pause the exact task before
                // returning an accounting error. Re-resolve here so the
                // controller neither repeats that CAS from a stale Running
                // snapshot nor overwrites a newer operator/terminal state.
                let current = resolve_goal(
                    cp.store.as_ref(),
                    cp.goal_store.as_ref(),
                    ctx,
                    Some(task_id.clone()),
                )
                .await?;
                if !current.is_running() {
                    if current.status() != TaskStatus::Paused
                        || current.goal().pause_reason != Some(GoalPauseReason::BudgetUnavailable)
                    {
                        return Ok(None);
                    }
                    let usage = goal_usage_totals_from_tracker(
                        cost_tracker.as_deref(),
                        current.task_id(),
                        true,
                    );
                    let budget = goal_budget_summary(current.goal(), usage.as_ref());
                    let message = msg(
                        goal_pause_message_key(GoalPauseReason::BudgetUnavailable),
                        &[("task_id", &task_id), ("budget", &budget)],
                    );
                    return Ok(Some(GoalTurnEvaluation::Paused { task_id, message }));
                }
                let cause = if is_goal_accounting_pricing_failure(&error) {
                    GoalBudgetUnavailableCause::CostPricingUnavailable
                } else {
                    GoalBudgetUnavailableCause::UsageUnavailable
                };
                let pause = goal_accounting_unavailable_pause(current.goal(), cause);
                let usage = goal_usage_totals_from_tracker(
                    cost_tracker.as_deref(),
                    current.task_id(),
                    true,
                );
                let budget = goal_budget_summary(current.goal(), usage.as_ref());
                let admission = pause_goal_for_resolved_task_with_budget(
                    cp.goal_store.as_ref(),
                    current,
                    pause,
                    budget,
                )
                .await?;
                publish_goal_state_update(&admission);
                return Ok(Some(GoalTurnEvaluation::Paused {
                    task_id,
                    message: admission.message,
                }));
            }
            let admission = pause_goal_for_known_blocker(
                cp.goal_store.as_ref(),
                resolved,
                Some(config),
                verifier_outage_pause(&error),
            )
            .await?;
            publish_goal_state_update(&admission);
            Ok(Some(GoalTurnEvaluation::Paused {
                task_id,
                message: admission.message,
            }))
        }
    }
}

pub async fn pause_current_goal_for_human_gate(
    ctx: &GoalAdmissionContext,
    config: Option<&Config>,
    kind: GoalBlockerKind,
    message: String,
    payload: Option<serde_json::Value>,
) -> Result<GoalAdmission> {
    match kind {
        GoalBlockerKind::NeedsUserInput | GoalBlockerKind::HumanEscalation => {}
        _ => bail!("human gate pause requires a human-gate blocker kind"),
    }
    let cp = control_plane().context("goal control plane unavailable during human gate")?;
    let task_id = ctx
        .goal_task_id
        .as_deref()
        .filter(|task_id| !task_id.trim().is_empty())
        .context("goal human gate has no exact durable task binding")?;
    let resolved = resolve_goal(
        cp.store.as_ref(),
        cp.goal_store.as_ref(),
        ctx,
        Some(task_id.to_string()),
    )
    .await?;
    let admission = pause_goal_for_known_blocker(
        cp.goal_store.as_ref(),
        resolved,
        config,
        GoalPauseState {
            reason: reason_for_blocker_kind(kind),
            description: Some(message.clone()),
            blockers: vec![GoalBlocker {
                kind,
                message,
                payload,
            }],
        },
    )
    .await?;
    publish_goal_state_update(&admission);
    Ok(admission)
}

async fn start_goal(
    goal_store: &dyn GoalTaskRegistry,
    boot_id: &str,
    ctx: GoalAdmissionContext,
    objective: String,
    token_limit: Option<u64>,
    cost_limit_usd: Option<f64>,
    config: Option<&Config>,
) -> Result<GoalAdmission> {
    let initial_usage = initial_goal_usage_totals(config);
    ensure_cost_budget_tracking_available(config, cost_limit_usd, Some(initial_usage.is_some()))?;
    let continuation_context = ctx.continuation_context.clone();
    if let Some(active) = goal_store
        .latest_active_goal_for_context(
            &ctx.agent_alias,
            ctx.originator_route.as_deref(),
            ctx.principal_id.as_deref(),
        )
        .await
        .with_context(|| msg("goal-command-error-active-goal-lookup-failed", &[]))?
    {
        bail!(
            "{}",
            msg(
                "goal-command-error-active-goal-exists",
                &[("task_id", &active.id)]
            )
        );
    }
    let task_id = uuid::Uuid::new_v4().to_string();
    let started_at = chrono::Utc::now().to_rfc3339();
    let mut goal = GoalTaskRecord {
        task_id: task_id.clone(),
        objective,
        effective_token_limit: token_limit,
        effective_cost_limit_usd: cost_limit_usd,
        pause_reason: None,
        pause_description: None,
        blockers: Vec::new(),
    };
    let initial_pause =
        config.and_then(|_| goal_accounting_gate_pause(&goal, initial_usage.as_ref()));
    let (status, continue_goal, message_key) = if let Some(pause) = initial_pause {
        goal.pause_reason = Some(pause.reason);
        goal.pause_description = pause.description;
        goal.blockers = pause.blockers;
        (
            TaskStatus::Paused,
            false,
            "goal-command-started-budget-unavailable",
        )
    } else {
        (TaskStatus::Running, true, "goal-command-started")
    };
    let budget = goal_budget_summary(&goal, initial_usage.as_ref());
    let message = msg(
        message_key,
        &[
            ("task_id", &task_id),
            ("objective", &goal.objective),
            ("budget", &budget),
        ],
    );
    goal_store
        .create_goal(
            TaskRecord {
                id: task_id.clone(),
                kind: TaskKind::Goal,
                agent: ctx.agent_alias,
                status,
                owner_pid: std::process::id(),
                owner_boot_id: boot_id.to_string(),
                heartbeat_at: None,
                depth: 0,
                parent_id: None,
                originator_route: ctx.originator_route,
                delivered: false,
                idem_key: None,
                principal_id: ctx.principal_id,
                started_at,
                finished_at: None,
            },
            goal,
            continuation_context,
        )
        .await
        .map_err(|error| {
            if is_active_goal_context_conflict(&error) {
                anyhow::Error::msg(msg("goal-command-error-active-goal-conflict", &[]))
            } else {
                error.context(msg("goal-command-error-start-failed", &[]))
            }
        })?;
    Ok(GoalAdmission {
        task_id: Some(task_id.clone()),
        status,
        message,
        continuation_reason: None,
        continue_goal,
    })
}

async fn status_goal(
    store: &dyn TaskRegistry,
    goal_store: &dyn GoalTaskRegistry,
    ctx: &GoalAdmissionContext,
    task_id: Option<String>,
    config: Option<&Config>,
) -> Result<GoalAdmission> {
    let task_goal = resolve_goal(store, goal_store, ctx, task_id).await?;
    let usage = goal_usage_totals_if_tracker_ready(config, task_goal.task_id());
    let budget = goal_budget_summary(task_goal.goal(), usage.as_ref());
    Ok(GoalAdmission {
        task_id: Some(task_goal.task_id().to_string()),
        status: task_goal.status(),
        message: task_goal_status_message(&task_goal, &budget),
        continuation_reason: None,
        continue_goal: false,
    })
}

async fn update_goal_budget(
    store: &dyn TaskRegistry,
    goal_store: &dyn GoalTaskRegistry,
    boot_id: &str,
    ctx: &GoalAdmissionContext,
    budgets: GoalBudgetOverrides,
    config: Option<&Config>,
) -> Result<GoalAdmission> {
    if matches!(budgets.token_limit, GoalBudgetValue::Default)
        && matches!(budgets.cost_limit_usd, GoalBudgetValue::Default)
    {
        bail!("{}", msg("goal-command-error-missing-budget", &[]));
    }
    let current = resolve_goal(store, goal_store, ctx, None).await?;
    if current.is_terminal() {
        let status = task_status_label(current.status());
        bail!(
            "{}",
            msg(
                "goal-command-error-already-terminal",
                &[("task_id", current.task_id()), ("status", &status)]
            )
        );
    }
    let token_limit = match budgets.token_limit {
        GoalBudgetValue::Default => current.goal().effective_token_limit,
        GoalBudgetValue::Unlimited => None,
        GoalBudgetValue::Limited(value) => Some(value),
    };
    let cost_limit_usd = match budgets.cost_limit_usd {
        GoalBudgetValue::Default => current.goal().effective_cost_limit_usd,
        GoalBudgetValue::Unlimited => None,
        GoalBudgetValue::Limited(value) => Some(value),
    };
    ensure_cost_budget_tracking_available(config, cost_limit_usd, None)?;
    let task_id = current.task_id().to_string();
    goal_store
        .update_goal_limits(&task_id, token_limit, cost_limit_usd)
        .await
        .with_context(|| msg("goal-command-error-budget-failed", &[("task_id", &task_id)]))?;
    let updated = current.with_effective_limits(token_limit, cost_limit_usd);
    let usage = goal_usage_totals(config, &task_id);
    let budget = goal_budget_summary(updated.goal(), usage.as_ref());
    let pause = config
        .and_then(|_| goal_usage_ledger_gate_pause(updated.goal(), usage.as_ref()))
        .or_else(|| goal_budget_gate_pause(updated.goal(), usage.as_ref()));
    if let Some(pause) = pause {
        if !goal_store
            .pause_goal_task_if_status(
                &task_id,
                updated.status(),
                merge_budget_pause(updated.goal(), pause),
            )
            .await
            .with_context(|| msg("goal-command-error-budget-failed", &[("task_id", &task_id)]))?
        {
            bail!(
                "{}",
                msg("goal-command-error-budget-failed", &[("task_id", &task_id)])
            );
        }
        return Ok(GoalAdmission {
            task_id: Some(task_id.clone()),
            status: TaskStatus::Paused,
            message: msg(
                "goal-command-budget-updated-paused",
                &[("task_id", &task_id), ("budget", &budget)],
            ),
            continuation_reason: None,
            continue_goal: false,
        });
    }

    if updated.status() == TaskStatus::Paused && has_budget_pause(updated.goal()) {
        if let Some(pause) = remove_budget_pause(updated.goal()) {
            let blockers = blockers_summary(&pause.blockers);
            if !goal_store
                .pause_goal_task_if_status(&task_id, updated.status(), pause)
                .await
                .with_context(|| {
                    msg("goal-command-error-budget-failed", &[("task_id", &task_id)])
                })?
            {
                bail!(
                    "{}",
                    msg("goal-command-error-budget-failed", &[("task_id", &task_id)])
                );
            }
            let message = if let Some(blockers) = blockers {
                msg(
                    "goal-command-budget-updated-paused-blocked",
                    &[
                        ("task_id", &task_id),
                        ("blockers", &blockers),
                        ("budget", &budget),
                    ],
                )
            } else {
                msg(
                    "goal-command-budget-updated-paused",
                    &[("task_id", &task_id), ("budget", &budget)],
                )
            };
            return Ok(GoalAdmission {
                task_id: Some(task_id.clone()),
                status: TaskStatus::Paused,
                message,
                continuation_reason: None,
                continue_goal: false,
            });
        }

        if !goal_store
            .resume_paused_goal_task(
                &task_id,
                std::process::id(),
                boot_id,
                ctx.continuation_context.clone(),
            )
            .await
            .with_context(|| msg("goal-command-error-update-failed", &[("task_id", &task_id)]))?
        {
            bail!(
                "{}",
                msg("goal-command-error-update-failed", &[("task_id", &task_id)])
            );
        }
        return Ok(GoalAdmission {
            task_id: Some(task_id.clone()),
            status: TaskStatus::Running,
            message: msg(
                "goal-command-budget-updated-resumed",
                &[("task_id", &task_id), ("budget", &budget)],
            ),
            continuation_reason: None,
            continue_goal: true,
        });
    }

    Ok(GoalAdmission {
        task_id: Some(task_id.clone()),
        status: updated.status(),
        message: msg(
            "goal-command-budget-updated",
            &[("task_id", &task_id), ("budget", &budget)],
        ),
        continuation_reason: None,
        continue_goal: false,
    })
}

async fn update_goal_objective(
    store: &dyn TaskRegistry,
    goal_store: &dyn GoalTaskRegistry,
    ctx: &GoalAdmissionContext,
    objective: String,
    config: Option<&Config>,
) -> Result<GoalAdmission> {
    let current = resolve_goal(store, goal_store, ctx, None).await?;
    if current.is_terminal() {
        let status = task_status_label(current.status());
        bail!(
            "{}",
            msg(
                "goal-command-error-already-terminal",
                &[("task_id", current.task_id()), ("status", &status)]
            )
        );
    }
    let task_id = current.task_id().to_string();
    goal_store
        .update_goal_objective(&task_id, &objective)
        .await
        .with_context(|| msg("goal-command-error-update-failed", &[("task_id", &task_id)]))?;
    let usage = goal_usage_totals(config, &task_id);
    let budget = goal_budget_summary(current.goal(), usage.as_ref());
    Ok(GoalAdmission {
        task_id: Some(task_id.clone()),
        status: current.status(),
        message: msg(
            "goal-command-objective-updated",
            &[
                ("task_id", &task_id),
                ("objective", &objective),
                ("budget", &budget),
            ],
        ),
        continuation_reason: None,
        continue_goal: false,
    })
}

async fn pause_goal_for_blocker(
    store: &dyn TaskRegistry,
    goal_store: &dyn GoalTaskRegistry,
    ctx: &GoalAdmissionContext,
    task_id: Option<String>,
    config: Option<&Config>,
    pause: GoalPauseState,
) -> Result<GoalAdmission> {
    let resolved = resolve_goal(store, goal_store, ctx, task_id).await?;
    let budget = task_goal_budget_summary(&resolved, config);
    pause_goal_for_resolved_task_with_budget(goal_store, resolved, pause, budget).await
}

async fn pause_goal_for_known_blocker(
    goal_store: &dyn GoalTaskRegistry,
    task_goal: TaskGoal,
    config: Option<&Config>,
    pause: GoalPauseState,
) -> Result<GoalAdmission> {
    let usage = goal_usage_totals(config, task_goal.task_id());
    let budget = goal_budget_summary(task_goal.goal(), usage.as_ref());
    pause_goal_for_resolved_task_with_budget(goal_store, task_goal, pause, budget).await
}

async fn pause_goal_for_resolved_task_with_budget(
    goal_store: &dyn GoalTaskRegistry,
    task_goal: TaskGoal,
    pause: GoalPauseState,
    budget: String,
) -> Result<GoalAdmission> {
    ensure_goal_not_terminal(task_goal.task())?;
    let task_id = task_goal.task_id().to_string();
    let message_key = goal_pause_message_key(pause.reason);
    if !goal_store
        .pause_goal_task_if_status(&task_id, task_goal.status(), pause)
        .await
        .with_context(|| msg("goal-command-error-pause-failed", &[("task_id", &task_id)]))?
    {
        bail!(
            "{}",
            msg("goal-command-error-pause-failed", &[("task_id", &task_id)])
        );
    }
    Ok(GoalAdmission {
        task_id: Some(task_id.clone()),
        status: TaskStatus::Paused,
        message: msg(message_key, &[("task_id", &task_id), ("budget", &budget)]),
        continuation_reason: None,
        continue_goal: false,
    })
}

fn goal_pause_message_key(reason: GoalPauseReason) -> &'static str {
    match reason {
        GoalPauseReason::BudgetExhausted => "goal-command-budget-exhausted",
        GoalPauseReason::BudgetUnavailable => "goal-command-budget-unavailable",
        _ => "goal-command-paused",
    }
}

fn ensure_goal_not_terminal(task: &TaskRecord) -> Result<()> {
    if task.status.is_terminal() {
        let status = task_status_label(task.status);
        bail!(
            "{}",
            msg(
                "goal-command-error-already-terminal",
                &[("task_id", &task.id), ("status", &status)]
            )
        );
    }
    Ok(())
}

async fn resume_goal(
    store: &dyn TaskRegistry,
    goal_store: &dyn GoalTaskRegistry,
    boot_id: &str,
    ctx: &GoalAdmissionContext,
    resume_reason: Option<String>,
    config: Option<&Config>,
) -> Result<GoalAdmission> {
    let current = resolve_goal(store, goal_store, ctx, None).await?;
    resume_resolved_goal(goal_store, boot_id, ctx, resume_reason, config, current).await
}

async fn resume_resolved_goal(
    goal_store: &dyn GoalTaskRegistry,
    boot_id: &str,
    ctx: &GoalAdmissionContext,
    resume_reason: Option<String>,
    config: Option<&Config>,
    current: TaskGoal,
) -> Result<GoalAdmission> {
    if current.is_terminal() {
        let status = task_status_label(current.status());
        bail!(
            "{}",
            msg(
                "goal-command-error-already-terminal",
                &[("task_id", current.task_id()), ("status", &status)]
            )
        );
    }
    let task_id = current.task_id().to_string();
    let current_usage = goal_usage_totals(config, &task_id);
    let pause = config
        .and_then(|_| goal_usage_ledger_gate_pause(current.goal(), current_usage.as_ref()))
        .or_else(|| goal_budget_gate_pause(current.goal(), current_usage.as_ref()));
    if let Some(pause) = pause {
        let message_key = goal_pause_message_key(pause.reason);
        let budget = goal_budget_summary(current.goal(), current_usage.as_ref());
        if !goal_store
            .pause_goal_task_if_status(
                &task_id,
                current.status(),
                merge_budget_pause(current.goal(), pause),
            )
            .await
            .with_context(|| msg("goal-command-error-resume-failed", &[("task_id", &task_id)]))?
        {
            bail!(
                "{}",
                msg("goal-command-error-resume-failed", &[("task_id", &task_id)])
            );
        }
        return Ok(GoalAdmission {
            task_id: Some(task_id.clone()),
            status: TaskStatus::Paused,
            message: msg(message_key, &[("task_id", &task_id), ("budget", &budget)]),
            continuation_reason: None,
            continue_goal: false,
        });
    }
    if !goal_store
        .resume_paused_goal_task(
            &task_id,
            std::process::id(),
            boot_id,
            ctx.continuation_context.clone(),
        )
        .await
        .with_context(|| msg("goal-command-error-update-failed", &[("task_id", &task_id)]))?
    {
        bail!(
            "{}",
            msg("goal-command-error-update-failed", &[("task_id", &task_id)])
        );
    }
    let budget = goal_budget_summary(current.goal(), current_usage.as_ref());
    Ok(GoalAdmission {
        task_id: Some(task_id.clone()),
        status: TaskStatus::Running,
        message: msg(
            "goal-command-resumed",
            &[("task_id", &task_id), ("budget", &budget)],
        ),
        continuation_reason: resume_reason,
        continue_goal: true,
    })
}

async fn cancel_goal(
    store: &dyn TaskRegistry,
    goal_store: &dyn GoalTaskRegistry,
    ctx: &GoalAdmissionContext,
    task_id: Option<String>,
    config: Option<&Config>,
) -> Result<GoalAdmission> {
    let current = resolve_goal(store, goal_store, ctx, task_id).await?;
    cancel_resolved_goal(goal_store, current, config).await
}

async fn cancel_resolved_goal(
    goal_store: &dyn GoalTaskRegistry,
    current: TaskGoal,
    config: Option<&Config>,
) -> Result<GoalAdmission> {
    if current.is_terminal() {
        let status = task_status_label(current.status());
        bail!(
            "{}",
            msg(
                "goal-command-error-already-terminal",
                &[("task_id", current.task_id()), ("status", &status)]
            )
        );
    }
    let task_id = current.task_id().to_string();
    if !goal_store
        .cancel_goal_task_if_status(
            &task_id,
            current.status(),
            msg("goal-terminal-reason-cancelled-by-controller", &[]),
        )
        .await
        .with_context(|| msg("goal-command-error-update-failed", &[("task_id", &task_id)]))?
    {
        bail!(
            "{}",
            msg("goal-command-error-update-failed", &[("task_id", &task_id)])
        );
    }
    let usage = goal_usage_totals(config, &task_id);
    let budget = goal_budget_summary(current.goal(), usage.as_ref());
    Ok(GoalAdmission {
        task_id: Some(task_id.clone()),
        status: TaskStatus::Cancelled,
        message: msg(
            "goal-command-cancelled",
            &[("task_id", &task_id), ("budget", &budget)],
        ),
        continuation_reason: None,
        continue_goal: false,
    })
}

fn task_goal_status_message(task_goal: &TaskGoal, budget: &str) -> String {
    if let Some(reason) = task_goal.goal().pause_reason {
        let status = task_status_label(task_goal.status());
        let reason = pause_reason_label(reason);
        if let Some(blockers) = blockers_summary(&task_goal.goal().blockers) {
            msg(
                "goal-command-status-paused-blocked",
                &[
                    ("task_id", task_goal.task_id()),
                    ("status", &status),
                    ("objective", task_goal.objective()),
                    ("reason", &reason),
                    ("blockers", &blockers),
                    ("budget", budget),
                ],
            )
        } else {
            msg(
                "goal-command-status-paused",
                &[
                    ("task_id", task_goal.task_id()),
                    ("status", &status),
                    ("objective", task_goal.objective()),
                    ("reason", &reason),
                    ("budget", budget),
                ],
            )
        }
    } else {
        let status = task_status_label(task_goal.status());
        msg(
            "goal-command-status",
            &[
                ("task_id", task_goal.task_id()),
                ("status", &status),
                ("objective", task_goal.objective()),
                ("budget", budget),
            ],
        )
    }
}

/// Admit a controller-synthesized autonomous goal continuation.
///
/// This is a pre-model-call gate for synthetic goal turns. It does not create
/// usage state and does not cache budget counters: effective limits come from
/// the goal extension record, while consumed usage is derived from cost ledger
/// rows for the canonical task id. `Ok(None)` means the turn may proceed.
pub async fn admit_goal_autonomous_turn(
    ctx: &GoalAdmissionContext,
    config: &Config,
) -> Result<Option<GoalAdmission>> {
    let Some(cp) = control_plane() else {
        return Ok(None);
    };
    let Some(resolved) =
        latest_active_resolved_goal(cp.store.as_ref(), cp.goal_store.as_ref(), ctx).await?
    else {
        return Ok(None);
    };
    if !resolved.is_running() {
        let usage = goal_usage_totals(Some(config), resolved.task_id());
        let budget = goal_budget_summary(resolved.goal(), usage.as_ref());
        return Ok(Some(GoalAdmission {
            task_id: Some(resolved.task_id().to_string()),
            status: resolved.status(),
            message: task_goal_status_message(&resolved, &budget),
            continuation_reason: None,
            continue_goal: false,
        }));
    }
    let usage = goal_usage_totals(Some(config), resolved.task_id());
    if let Some(pause) = goal_accounting_gate_pause(resolved.goal(), usage.as_ref()) {
        let budget = goal_budget_summary(resolved.goal(), usage.as_ref());
        return pause_goal_for_resolved_task_with_budget(
            cp.goal_store.as_ref(),
            resolved,
            pause,
            budget,
        )
        .await
        .map(Some);
    }
    Ok(None)
}

async fn resolve_goal_task(
    store: &dyn TaskRegistry,
    goal_store: &dyn GoalTaskRegistry,
    ctx: &GoalAdmissionContext,
    task_id: Option<String>,
) -> Result<TaskRecord> {
    if let Some(task_id) = task_id {
        let task = store
            .get(&task_id)
            .await
            .with_context(|| msg("goal-command-error-lookup-failed", &[]))?
            .with_context(|| msg("goal-command-error-not-found", &[("task_id", &task_id)]))?;
        return ensure_goal_visible_or_rebind(goal_store, task, ctx).await;
    }

    let task = goal_store
        .latest_active_goal_for_context(
            &ctx.agent_alias,
            ctx.originator_route.as_deref(),
            ctx.principal_id.as_deref(),
        )
        .await
        .with_context(|| msg("goal-command-error-lookup-failed", &[]))?
        .with_context(|| msg("goal-command-error-no-active-goal", &[]))?;
    ensure_goal_visible(&task, ctx)?;
    Ok(task)
}

async fn load_goal_extension(
    goal_store: &dyn GoalTaskRegistry,
    task: TaskRecord,
) -> Result<TaskGoal> {
    let goal = goal_store
        .get_goal_task(&task.id)
        .await
        .with_context(|| msg("goal-command-error-status-failed", &[]))?
        .with_context(|| {
            msg(
                "goal-command-error-extension-missing",
                &[("task_id", &task.id)],
            )
        })?;
    Ok(TaskGoal::new(task, goal))
}

async fn resolve_goal(
    store: &dyn TaskRegistry,
    goal_store: &dyn GoalTaskRegistry,
    ctx: &GoalAdmissionContext,
    task_id: Option<String>,
) -> Result<TaskGoal> {
    let task = resolve_goal_task(store, goal_store, ctx, task_id).await?;
    load_goal_extension(goal_store, task).await
}

async fn latest_active_resolved_goal(
    store: &dyn TaskRegistry,
    goal_store: &dyn GoalTaskRegistry,
    ctx: &GoalAdmissionContext,
) -> Result<Option<TaskGoal>> {
    if let Some(task_id) = ctx.goal_task_id.clone() {
        return resolve_goal(store, goal_store, ctx, Some(task_id))
            .await
            .map(Some);
    }
    let Some(task) = goal_store
        .latest_active_goal_for_context(
            &ctx.agent_alias,
            ctx.originator_route.as_deref(),
            ctx.principal_id.as_deref(),
        )
        .await
        .with_context(|| msg("goal-command-error-active-goal-lookup-failed", &[]))?
    else {
        return Ok(None);
    };
    ensure_goal_visible(&task, ctx)?;
    load_goal_extension(goal_store, task).await.map(Some)
}

async fn ensure_goal_visible_or_rebind(
    goal_store: &dyn GoalTaskRegistry,
    task: TaskRecord,
    ctx: &GoalAdmissionContext,
) -> Result<TaskRecord> {
    let visibility_error = match ensure_goal_visible(&task, ctx) {
        Ok(()) => return Ok(task),
        Err(error) => error,
    };

    let (
        Some(current_route),
        Some(current_principal),
        Some(legacy_route),
        Some(legacy_principal),
        Some(expected_context),
        Some(stored_route),
        Some(stored_principal),
    ) = (
        ctx.originator_route.as_deref(),
        ctx.principal_id.as_deref(),
        ctx.legacy_originator_route.as_deref(),
        ctx.legacy_principal_id.as_deref(),
        ctx.continuation_context.as_ref(),
        task.originator_route.as_deref(),
        task.principal_id.as_deref(),
    )
    else {
        return Err(visibility_error);
    };

    if task.kind != TaskKind::Goal
        || task.agent != ctx.agent_alias
        || stored_route != legacy_route
        || stored_principal != legacy_principal
        || goal_store
            .get_continuation_context(&task.id)
            .await?
            .as_ref()
            != Some(expected_context)
    {
        return Err(visibility_error);
    }

    let rebound = goal_store
        .rebind_goal_task_identity(
            &task.id,
            stored_route,
            stored_principal,
            current_route,
            current_principal,
        )
        .await
        .with_context(|| msg("goal-command-error-lookup-failed", &[]))?;
    if !rebound {
        bail!(
            "{}",
            msg("goal-command-error-wrong-route", &[("task_id", &task.id)])
        );
    }

    let mut rebound_task = task;
    rebound_task.originator_route = Some(current_route.to_string());
    rebound_task.principal_id = Some(current_principal.to_string());
    Ok(rebound_task)
}

fn is_active_goal_context_conflict(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let text = cause.to_string();
        text.contains("idx_tasks_active_goal_context")
            || text.contains("UNIQUE constraint failed: index 'idx_tasks_active_goal_context'")
    })
}

fn ensure_goal_visible(task: &TaskRecord, ctx: &GoalAdmissionContext) -> Result<()> {
    if task.kind != TaskKind::Goal {
        bail!(
            "{}",
            msg("goal-command-error-not-goal", &[("task_id", &task.id)])
        );
    }
    if task.agent != ctx.agent_alias {
        bail!(
            "{}",
            msg("goal-command-error-wrong-agent", &[("task_id", &task.id)])
        );
    }
    if let Some(route) = task.originator_route.as_deref()
        && ctx.originator_route.as_deref() != Some(route)
    {
        bail!(
            "{}",
            msg("goal-command-error-wrong-route", &[("task_id", &task.id)])
        );
    }
    if let Some(principal_id) = task.principal_id.as_deref()
        && ctx.principal_id.as_deref() != Some(principal_id)
    {
        bail!(
            "{}",
            msg(
                "goal-command-error-wrong-principal",
                &[("task_id", &task.id)]
            )
        );
    }
    Ok(())
}

fn nonempty(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_plane::TaskContinuationConversationScope;
    use crate::control_plane::task_store_sqlite::SqliteTaskStore;
    use std::sync::Arc;

    #[tokio::test]
    async fn live_goal_binding_reservation_rejects_competing_binder() {
        scope_goal_admission_context(Some(GoalAdmissionContext::new("agent-a")), async {
            let reservation = reserve_current_goal_task_binding()
                .expect("unbound live task binding can be reserved");

            assert!(
                !bind_current_goal_task("goal-competing"),
                "a generic binder cannot steal a model admission reservation"
            );
            reservation.bind("goal-admitted".to_string());

            assert_eq!(
                current_goal_admission_context().and_then(|context| context.goal_task_id),
                Some("goal-admitted".to_string())
            );
        })
        .await;
    }

    fn configure_test_goal_channel(
        config: &mut Config,
        agent_alias: &str,
        channel_type: &str,
        channel_alias: &str,
    ) {
        let channel_ref = format!("{channel_type}.{channel_alias}");
        let agent = AliasedAgentConfig {
            channels: vec![zeroclaw_config::providers::ChannelRef::new(channel_ref)],
            ..AliasedAgentConfig::default()
        };
        config.agents.insert(agent_alias.to_string(), agent);
        match channel_type {
            "matrix" => {
                config.channels.matrix.insert(
                    channel_alias.to_string(),
                    zeroclaw_config::schema::MatrixConfig {
                        enabled: true,
                        ..zeroclaw_config::schema::MatrixConfig::default()
                    },
                );
            }
            "telegram" => {
                config.channels.telegram.insert(
                    channel_alias.to_string(),
                    zeroclaw_config::schema::TelegramConfig {
                        enabled: true,
                        ..zeroclaw_config::schema::TelegramConfig::default()
                    },
                );
            }
            other => panic!("unsupported goal test channel type: {other}"),
        }
    }

    fn test_config_for_agent(agent_alias: &str) -> Config {
        let mut config = Config::default();
        config.cost.enabled = false;
        config.goal.enabled = true;
        configure_test_goal_channel(&mut config, agent_alias, "matrix", "default");
        config
    }

    fn test_config() -> Config {
        test_config_for_agent("agent-a")
    }

    fn test_goal_context(agent_alias: impl Into<String>) -> GoalAdmissionContext {
        GoalAdmissionContext::new(agent_alias)
            .with_channel_type(Some("matrix".into()))
            .with_continuation_context(Some(TaskContinuationContext {
                channel: "matrix".into(),
                channel_alias: Some("default".into()),
                reply_target: "test-room".into(),
                sender: "test-operator".into(),
                thread_ts: None,
                interruption_scope_id: None,
                conversation_scope: TaskContinuationConversationScope::ReplyTarget,
            }))
    }

    fn global_test_stores() -> (Arc<dyn TaskRegistry>, Arc<dyn GoalTaskRegistry>) {
        match crate::control_plane::control_plane() {
            Some(control_plane) => (
                Arc::clone(&control_plane.store),
                Arc::clone(&control_plane.goal_store),
            ),
            None => {
                let sqlite_store =
                    Arc::new(crate::control_plane::SqliteTaskStore::new_in_memory().unwrap());
                let store: Arc<dyn TaskRegistry> = sqlite_store.clone();
                let goal_store: Arc<dyn GoalTaskRegistry> = sqlite_store;
                let _ = crate::control_plane::init_control_plane(
                    crate::control_plane::ControlPlaneHandle {
                        store: Arc::clone(&store),
                        goal_store: Arc::clone(&goal_store),
                        boot_id: "test-boot".into(),
                        recovered_goal_ids: Arc::new(std::sync::Mutex::new(Vec::new())),
                        data_dir_lock: None,
                    },
                );
                (
                    Arc::clone(&crate::control_plane::control_plane().unwrap().store),
                    Arc::clone(&crate::control_plane::control_plane().unwrap().goal_store),
                )
            }
        }
    }

    static GOAL_COST_TRACKER_TEST_LOCK: std::sync::LazyLock<tokio::sync::Mutex<()>> =
        std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

    async fn goal_cost_tracker_test_lock() -> tokio::sync::MutexGuard<'static, ()> {
        // Goal budget tests intentionally exercise the process-global
        // `CostTracker`, whose config and data-dir are hot-swapped on access.
        // Serializing only those tests prevents unrelated parallel tests from
        // disabling or retargeting the tracker between verifier usage recording
        // and the controller's budget read.
        GOAL_COST_TRACKER_TEST_LOCK.lock().await
    }

    fn cost_enabled_test_config(data_dir: &std::path::Path) -> Config {
        let mut config = test_config();
        config.data_dir = data_dir.to_path_buf();
        config.cost.enabled = true;
        config.cost.track_per_agent = true;
        config
    }

    /// Test-only fixture for a single running goal scoped to one route/principal.
    ///
    /// The fixture keeps only handles and identifiers needed by assertions; the
    /// canonical lifecycle and goal-specific state live in the in-memory task
    /// registry rows created by `create_running_goal_fixture`.
    struct RunningGoalFixture {
        /// Canonical task registry handle used to assert lifecycle transitions.
        store: Arc<dyn TaskRegistry>,
        /// Canonical goal extension registry handle used to assert pause data.
        goal_store: Arc<dyn GoalTaskRegistry>,
        /// Goal task id created for this test case.
        task_id: String,
        /// Trusted route/principal context that can see the fixture goal.
        ctx: GoalAdmissionContext,
    }

    async fn create_running_goal_fixture(objective: &str) -> RunningGoalFixture {
        let (store, goal_store) = global_test_stores();
        let task_id = format!("goal-{}", uuid::Uuid::new_v4());
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let ctx = GoalAdmissionContext::new(agent.clone())
            .with_originator_route(Some(route.clone()))
            .with_principal_id(Some(principal.clone()));
        goal_store
            .create_goal(
                TaskRecord {
                    id: task_id.clone(),
                    kind: TaskKind::Goal,
                    agent,
                    status: TaskStatus::Running,
                    owner_pid: std::process::id(),
                    owner_boot_id: "test-boot".into(),
                    heartbeat_at: None,
                    depth: 0,
                    parent_id: None,
                    originator_route: Some(route),
                    delivered: false,
                    idem_key: None,
                    principal_id: Some(principal),
                    started_at: chrono::Utc::now().to_rfc3339(),
                    finished_at: None,
                },
                GoalTaskRecord {
                    task_id: task_id.clone(),
                    objective: objective.into(),
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                },
                None,
            )
            .await
            .unwrap();
        RunningGoalFixture {
            store,
            goal_store,
            task_id,
            ctx,
        }
    }

    fn record_goal_token_usage(config: &Config, agent: &str, task_id: &str, tokens: u64) {
        let tracker = CostTracker::get_or_init_global(config.cost.clone(), &config.data_dir)
            .expect("enabled test cost tracker");
        tracker
            .record_usage_with_task_attribution(
                zeroclaw_config::cost::types::TokenUsage::new(
                    "test/model",
                    tokens,
                    0,
                    0,
                    1.0,
                    2.0,
                    0.0,
                ),
                Some(agent),
                Some(task_id),
            )
            .expect("record goal usage");
    }

    async fn create_budget_paused_goal(
        store: &SqliteTaskStore,
        ctx: &GoalAdmissionContext,
        task_id: &str,
        token_limit: u64,
        continuation_context: Option<TaskContinuationContext>,
    ) {
        store
            .create_goal(
                TaskRecord {
                    id: task_id.to_string(),
                    kind: TaskKind::Goal,
                    agent: ctx.agent_alias.clone(),
                    status: TaskStatus::Paused,
                    owner_pid: std::process::id(),
                    owner_boot_id: "boot-exhausted".into(),
                    heartbeat_at: None,
                    depth: 0,
                    parent_id: None,
                    originator_route: ctx.originator_route.clone(),
                    delivered: false,
                    idem_key: None,
                    principal_id: ctx.principal_id.clone(),
                    started_at: chrono::Utc::now().to_rfc3339(),
                    finished_at: None,
                },
                GoalTaskRecord {
                    task_id: task_id.to_string(),
                    objective: "finish budgeted work".into(),
                    effective_token_limit: Some(token_limit),
                    effective_cost_limit_usd: None,
                    pause_reason: Some(GoalPauseReason::BudgetExhausted),
                    pause_description: Some("token budget exhausted".into()),
                    blockers: vec![GoalBlocker {
                        kind: GoalBlockerKind::Budget,
                        message: "Token budget exhausted".into(),
                        payload: None,
                    }],
                },
                continuation_context,
            )
            .await
            .expect("create budget-paused goal fixture");
    }

    async fn create_policy_revocation_goal(
        store: &SqliteTaskStore,
        task_id: &str,
        agent: &str,
        channel: &str,
    ) {
        store
            .create_goal(
                TaskRecord {
                    id: task_id.to_string(),
                    kind: TaskKind::Goal,
                    agent: agent.to_string(),
                    status: TaskStatus::Running,
                    owner_pid: std::process::id(),
                    owner_boot_id: "policy-test-boot".into(),
                    heartbeat_at: None,
                    depth: 0,
                    parent_id: None,
                    originator_route: Some(format!("{channel}:route:{task_id}")),
                    delivered: false,
                    idem_key: None,
                    principal_id: Some(format!("principal:{task_id}")),
                    started_at: chrono::Utc::now().to_rfc3339(),
                    finished_at: None,
                },
                GoalTaskRecord {
                    task_id: task_id.to_string(),
                    objective: format!("exercise {task_id} policy"),
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                },
                Some(TaskContinuationContext {
                    channel: channel.to_string(),
                    channel_alias: Some("default".into()),
                    reply_target: format!("room:{task_id}"),
                    sender: "operator".into(),
                    thread_ts: None,
                    interruption_scope_id: Some(format!("scope:{task_id}")),
                    conversation_scope: TaskContinuationConversationScope::ReplyTarget,
                }),
            )
            .await
            .expect("create policy-revocation goal");
    }

    /// Deterministic verifier fixture for controller transition tests.
    ///
    /// The production verifier is a pluggable `GoalVerifier`; this fixture keeps
    /// tests focused on how the controller consumes typed verdicts without
    /// introducing model calls or mutating durable state itself.
    #[derive(Clone)]
    struct StubGoalVerifier {
        /// Verdict returned to the controller for this test case.
        decision: GoalVerifierDecision,
    }

    #[async_trait::async_trait]
    impl GoalVerifier for StubGoalVerifier {
        async fn verify(
            &self,
            request: GoalVerificationRequest<'_>,
        ) -> Result<GoalVerifierDecision> {
            assert!(!request.goal.objective.trim().is_empty());
            Ok(self.decision.clone())
        }
    }

    /// Verifier fixture that exercises controller handling of verifier outages.
    ///
    /// It deliberately returns no typed verdict so the controller must translate
    /// the failure into a durable verifier pause rather than completing the goal.
    struct FailingGoalVerifier;

    #[async_trait::async_trait]
    impl GoalVerifier for FailingGoalVerifier {
        async fn verify(
            &self,
            request: GoalVerificationRequest<'_>,
        ) -> Result<GoalVerifierDecision> {
            assert_eq!(request.candidate_summary, "looks done");
            Err(anyhow::Error::msg("provider offline"))
        }
    }

    /// Verifier fixture that replaces the active goal before recording usage.
    ///
    /// The replacement shares the same route and principal, so only the exact
    /// task id bound by the controller can prevent verifier spend from moving
    /// to the new "latest active" goal.
    struct ReplacingGoalVerifier {
        goal_store: Arc<dyn GoalTaskRegistry>,
        replacement_task_id: String,
    }

    #[async_trait::async_trait]
    impl GoalVerifier for ReplacingGoalVerifier {
        async fn verify(
            &self,
            request: GoalVerificationRequest<'_>,
        ) -> Result<GoalVerifierDecision> {
            assert_eq!(
                request.goal_context.goal_task_id.as_deref(),
                Some(request.goal.task_id.as_str()),
                "controller must bind verifier accounting to the resolved task"
            );
            assert!(
                self.goal_store
                    .cancel_goal_task_if_status(
                        &request.goal.task_id,
                        TaskStatus::Running,
                        "replaced during verifier test".into(),
                    )
                    .await?
            );
            self.goal_store
                .create_goal(
                    TaskRecord {
                        id: self.replacement_task_id.clone(),
                        kind: TaskKind::Goal,
                        agent: request.agent_alias.into(),
                        status: TaskStatus::Running,
                        owner_pid: std::process::id(),
                        owner_boot_id: "test-boot".into(),
                        heartbeat_at: None,
                        depth: 0,
                        parent_id: None,
                        originator_route: request.goal_context.originator_route.clone(),
                        delivered: false,
                        idem_key: None,
                        principal_id: request.goal_context.principal_id.clone(),
                        started_at: chrono::Utc::now().to_rfc3339(),
                        finished_at: None,
                    },
                    GoalTaskRecord {
                        task_id: self.replacement_task_id.clone(),
                        objective: "replacement goal".into(),
                        effective_token_limit: None,
                        effective_cost_limit_usd: None,
                        pause_reason: None,
                        pause_description: None,
                        blockers: Vec::new(),
                    },
                    None,
                )
                .await?;

            let tracker = request
                .cost_tracker
                .clone()
                .ok_or_else(|| anyhow::Error::msg("verifier tracker missing"))?;
            let accounting = crate::agent::cost::ToolLoopCostTrackingContext::new(
                tracker,
                Arc::new(std::collections::HashMap::new()),
            )
            .with_agent_alias(request.agent_alias)
            .with_goal_admission_context(request.goal_context);
            let usage = zeroclaw_providers::traits::TokenUsage {
                input_tokens: Some(80),
                output_tokens: Some(20),
                cached_input_tokens: Some(0),
            };
            crate::agent::cost::TOOL_LOOP_COST_TRACKING_CONTEXT
                .scope(Some(accounting), async {
                    crate::agent::cost::record_tool_loop_cost_usage(
                        "test-provider",
                        "test-model",
                        &usage,
                    )
                    .await
                })
                .await?;
            Ok(GoalVerifierDecision::Complete {
                notes: "COMPLETE".into(),
            })
        }
    }

    #[test]
    fn parse_goal_start_keeps_objective_untrusted_payload_only() {
        let parsed = parse_goal_command("/goal start ship the thing").unwrap();
        assert_eq!(parsed.action, GoalCommandAction::Start);
        assert_eq!(parsed.objective.as_deref(), Some("ship the thing"));
        assert!(parsed.task_id.is_none());
        assert!(parsed.resume_reason.is_none());

        let parsed = parse_goal_command("/goal@zeroclaw_bot START ship the thing").unwrap();
        assert_eq!(parsed.action, GoalCommandAction::Start);
        assert_eq!(parsed.objective.as_deref(), Some("ship the thing"));

        let err = parse_goal_command("/unknown start ship the thing").unwrap_err();
        assert!(err.to_string().contains("must start with `/goal`"));

        let err = parse_goal_command("start ship the thing").unwrap_err();
        assert!(err.to_string().contains("must start with `/goal`"));

        let err = parse_goal_command("goal start ship the thing").unwrap_err();
        assert!(err.to_string().contains("must start with `/goal`"));

        let parsed = parse_goal_command("/goal resume fixed").unwrap();
        assert_eq!(parsed.action, GoalCommandAction::Resume);
        assert!(parsed.task_id.is_none());
        assert_eq!(parsed.resume_reason.as_deref(), Some("fixed"));
    }

    #[test]
    fn parse_goal_objective_requires_freeform_objective_payload() {
        let parsed = parse_goal_command("/goal objective revise scope after evidence").unwrap();
        assert_eq!(parsed.action, GoalCommandAction::Objective);
        assert_eq!(
            parsed.objective.as_deref(),
            Some("revise scope after evidence")
        );
        assert!(parsed.task_id.is_none());
        assert!(parsed.resume_reason.is_none());

        let err = parse_goal_command("/goal objective").unwrap_err();
        assert!(err.to_string().contains("non-empty objective"));
    }

    #[test]
    fn parse_goal_resume_accepts_freeform_reason_payloads() {
        let parsed = parse_goal_command("/goal resume blocker fixed, retry now").unwrap();
        assert!(parsed.task_id.is_none());
        assert_eq!(
            parsed.resume_reason.as_deref(),
            Some("blocker fixed, retry now")
        );

        let parsed = parse_goal_command("/goal resume goal-123").unwrap();
        assert!(parsed.task_id.is_none());
        assert_eq!(parsed.resume_reason.as_deref(), Some("goal-123"));

        let task_id = uuid::Uuid::new_v4().to_string();
        let parsed =
            parse_goal_command(&format!("/goal resume {task_id} retry after fix")).unwrap();
        assert!(parsed.task_id.is_none());
        let expected = format!("{task_id} retry after fix");
        assert_eq!(parsed.resume_reason.as_deref(), Some(expected.as_str()));

        let parsed = parse_goal_command("/goal resume --some-flag-looking reason").unwrap();
        assert!(parsed.task_id.is_none());
        assert_eq!(
            parsed.resume_reason.as_deref(),
            Some("--some-flag-looking reason")
        );
    }

    #[test]
    fn parse_goal_task_selectors_reject_extra_arguments() {
        let status = parse_goal_command("/goal status goal-123").unwrap();
        assert_eq!(status.task_id.as_deref(), Some("goal-123"));

        let cancel = parse_goal_command("/goal cancel goal-123").unwrap();
        assert_eq!(cancel.task_id.as_deref(), Some("goal-123"));

        let status_err = parse_goal_command("/goal status goal-123 extra").unwrap_err();
        assert!(status_err.to_string().contains("Unexpected goal arguments"));

        let cancel_err = parse_goal_command("/goal cancel goal-123 extra").unwrap_err();
        assert!(cancel_err.to_string().contains("Unexpected goal arguments"));
    }

    #[test]
    fn parse_goal_help_and_budget_flags() {
        let help = parse_goal_command("/goal help").unwrap();
        assert_eq!(help.action, GoalCommandAction::Help);

        let help = parse_goal_command("/goal --help").unwrap();
        assert_eq!(help.action, GoalCommandAction::Help);

        let help = parse_goal_command("/goal -h").unwrap();
        assert_eq!(help.action, GoalCommandAction::Help);

        let err = parse_goal_command("/goal").unwrap_err();
        assert!(err.to_string().contains("requires an action"));

        let help_text = msg("goal-command-help", &[]);
        for expected in [
            "/goal start [--tokens=N|unlimited] [--cost=N|unlimited] <objective>",
            "/goal objective <objective>",
            "/goal status [task_id]",
            "/goal budget [--tokens=N|unlimited] [--cost=N|unlimited]",
            "/goal pause [reason]",
            "/goal resume [reason]",
            "/goal cancel [task_id]",
            "/goal help | /goal --help | /goal -h",
        ] {
            assert!(
                help_text.contains(expected),
                "goal help must list supported syntax {expected:?}; help was: {help_text}"
            );
        }

        let start = parse_goal_command("/goal start --tokens=50000 --cost=2.50 ship it").unwrap();
        assert_eq!(start.objective.as_deref(), Some("ship it"));
        assert_eq!(start.budgets.token_limit, GoalBudgetValue::Limited(50_000));
        assert_eq!(start.budgets.cost_limit_usd, GoalBudgetValue::Limited(2.50));

        let budget = parse_goal_command("/goal budget --tokens=unlimited --cost=1.25").unwrap();
        assert_eq!(budget.action, GoalCommandAction::Budget);
        assert_eq!(budget.budgets.token_limit, GoalBudgetValue::Unlimited);
        assert_eq!(
            budget.budgets.cost_limit_usd,
            GoalBudgetValue::Limited(1.25)
        );
    }

    #[test]
    fn goal_budget_pause_marks_exhausted_dimensions_from_ledger_summary() {
        let goal = GoalTaskRecord {
            task_id: "goal-1".into(),
            objective: "ship it".into(),
            effective_token_limit: Some(1_000),
            effective_cost_limit_usd: Some(0.50),
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
        };
        let usage = GoalUsageTotals {
            cost_usd: 0.75,
            total_tokens: 1_000,
            cost_pricing_available: true,
            cost_tracking_available: true,
            usage_available: true,
        };

        let pause = goal_budget_pause(&goal, Some(&usage)).unwrap();

        assert_eq!(pause.reason, GoalPauseReason::BudgetExhausted);
        assert_eq!(pause.blockers.len(), 1);
        assert_eq!(pause.blockers[0].kind, GoalBlockerKind::Budget);
        let payload = pause.blockers[0].payload.as_ref().unwrap();
        assert_eq!(payload["tokens"]["exhausted"], true);
        assert_eq!(payload["cost"]["exhausted"], true);
        assert!(pause.blockers[0].message.contains("Budget:"));
    }

    #[test]
    fn goal_budget_gate_pauses_when_limits_exist_without_usage_summary() {
        let goal = GoalTaskRecord {
            task_id: "goal-1".into(),
            objective: "ship it".into(),
            effective_token_limit: Some(1_000),
            effective_cost_limit_usd: None,
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
        };

        let pause = goal_budget_gate_pause(&goal, None).unwrap();

        assert_eq!(pause.reason, GoalPauseReason::BudgetUnavailable);
        assert_eq!(pause.blockers.len(), 1);
        assert_eq!(pause.blockers[0].kind, GoalBlockerKind::Budget);
        assert_eq!(
            pause.blockers[0].payload.as_ref().unwrap()["usage_unavailable"],
            true
        );
        assert_eq!(
            pause.blockers[0].payload.as_ref().unwrap()["cost_pricing_unavailable"],
            false
        );
    }

    #[test]
    fn unlimited_goal_pauses_when_usage_is_unavailable() {
        let goal = GoalTaskRecord {
            task_id: "goal-1".into(),
            objective: "ship it".into(),
            effective_token_limit: None,
            effective_cost_limit_usd: None,
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
        };
        let usage = GoalUsageTotals {
            usage_available: false,
            ..GoalUsageTotals::default()
        };

        let pause = goal_budget_gate_pause(&goal, Some(&usage))
            .expect("unlimited autonomous work still requires complete usage");
        assert_eq!(pause.reason, GoalPauseReason::BudgetUnavailable);
        assert_eq!(pause.blockers[0].kind, GoalBlockerKind::Budget);
    }

    #[test]
    fn incomplete_usage_summary_keeps_known_totals_visible_as_lower_bounds() {
        let goal = GoalTaskRecord {
            task_id: "goal-1".into(),
            objective: "ship it".into(),
            effective_token_limit: None,
            effective_cost_limit_usd: None,
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
        };
        let token_only = GoalUsageTotals {
            total_tokens: 1_250,
            cost_tracking_available: false,
            usage_available: false,
            ..GoalUsageTotals::default()
        };
        let priced = GoalUsageTotals {
            total_tokens: 1_250,
            cost_usd: 0.25,
            usage_available: false,
            cost_pricing_available: false,
            ..GoalUsageTotals::default()
        };

        let token_only_summary = goal_budget_summary(&goal, Some(&token_only));
        assert!(token_only_summary.contains("at least 1250/unlimited tokens recorded"));
        assert!(token_only_summary.contains("usage incomplete"));
        assert!(token_only_summary.contains("cost unavailable"));

        let priced_summary = goal_budget_summary(&goal, Some(&priced));
        assert!(priced_summary.contains("at least 1250/unlimited tokens"));
        assert!(priced_summary.contains("$0.2500/unlimited cost"));
        assert!(priced_summary.contains("usage incomplete"));
    }

    #[test]
    fn incomplete_ledger_usage_keeps_known_cost_visible_as_a_lower_bound() {
        let workspace = tempfile::TempDir::new().unwrap();
        let tracker = CostTracker::new(
            zeroclaw_config::schema::CostConfig {
                enabled: true,
                ..Default::default()
            },
            workspace.path(),
        )
        .unwrap();
        tracker
            .record_scoped_usage_with_owned_task_attribution(
                zeroclaw_config::cost::types::TokenUsage::new(
                    "priced-model",
                    1,
                    0,
                    0,
                    0.25,
                    0.0,
                    0.0,
                ),
                Some("agent-a"),
                Some("goal-1".into()),
            )
            .unwrap();
        tracker
            .record_scoped_usage_with_owned_task_attribution(
                zeroclaw_config::cost::types::TokenUsage::unavailable("priced-model"),
                Some("agent-a"),
                Some("goal-1".into()),
            )
            .unwrap();
        tracker.update_config(zeroclaw_config::schema::CostConfig {
            enabled: false,
            ..Default::default()
        });
        let usage = goal_usage_totals_from_tracker(Some(&tracker), "goal-1", true)
            .expect("ledger-derived usage");
        let goal = GoalTaskRecord {
            task_id: "goal-1".into(),
            objective: "ship it".into(),
            effective_token_limit: None,
            effective_cost_limit_usd: None,
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
        };

        assert!(!usage.usage_available);
        assert!(!usage.cost_pricing_available);
        assert!(!usage.cost_tracking_available);
        assert_eq!(usage.cost_usd, 0.000_000_25);
        let summary = goal_budget_summary(&goal, Some(&usage));
        assert!(summary.contains("$2.5000e-7/unlimited cost"));
        assert!(summary.contains("usage incomplete"));
    }

    #[test]
    fn unlimited_goal_pauses_when_the_canonical_ledger_is_unavailable() {
        let goal = GoalTaskRecord {
            task_id: "goal-1".into(),
            objective: "ship it".into(),
            effective_token_limit: None,
            effective_cost_limit_usd: None,
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
        };

        let pause = goal_usage_ledger_gate_pause(&goal, None)
            .expect("a missing ledger cannot retain goal-attributed usage");

        assert_eq!(pause.reason, GoalPauseReason::BudgetUnavailable);
        assert_eq!(pause.blockers[0].kind, GoalBlockerKind::Budget);
    }

    #[test]
    fn goal_budget_gate_treats_unpriced_cost_usage_as_unavailable() {
        let mut goal = GoalTaskRecord {
            task_id: "goal-1".into(),
            objective: "ship it".into(),
            effective_token_limit: Some(1_000),
            effective_cost_limit_usd: Some(0.50),
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
        };
        let usage = GoalUsageTotals {
            total_tokens: 250,
            cost_usd: 0.0,
            cost_pricing_available: false,
            cost_tracking_available: true,
            usage_available: true,
        };

        let pause = goal_budget_gate_pause(&goal, Some(&usage)).unwrap();

        assert_eq!(pause.reason, GoalPauseReason::BudgetUnavailable);
        assert_eq!(pause.blockers[0].kind, GoalBlockerKind::Budget);
        assert_eq!(
            pause.blockers[0].payload.as_ref().unwrap()["usage_unavailable"],
            false
        );
        assert_eq!(
            pause.blockers[0].payload.as_ref().unwrap()["cost_pricing_unavailable"],
            true
        );

        goal.effective_cost_limit_usd = None;
        assert!(
            goal_budget_gate_pause(&goal, Some(&usage)).is_none(),
            "unpriced cost must not block a token-only budget"
        );
    }

    #[test]
    fn removing_budget_pause_preserves_unrelated_blockers() {
        let goal = GoalTaskRecord {
            task_id: "goal-1".into(),
            objective: "ship it".into(),
            effective_token_limit: Some(1_000),
            effective_cost_limit_usd: None,
            pause_reason: Some(GoalPauseReason::BudgetExhausted),
            pause_description: Some("multiple blockers".into()),
            blockers: vec![
                GoalBlocker {
                    kind: GoalBlockerKind::NeedsUserInput,
                    message: "Need operator answer".into(),
                    payload: None,
                },
                GoalBlocker {
                    kind: GoalBlockerKind::Budget,
                    message: "Budget exhausted".into(),
                    payload: None,
                },
            ],
        };

        let pause = remove_budget_pause(&goal).unwrap();

        assert_eq!(pause.reason, GoalPauseReason::NeedsUserInput);
        assert_eq!(pause.blockers.len(), 1);
        assert_eq!(pause.blockers[0].kind, GoalBlockerKind::NeedsUserInput);

        let only_budget = GoalTaskRecord {
            blockers: vec![GoalBlocker {
                kind: GoalBlockerKind::Budget,
                message: "Budget exhausted".into(),
                payload: None,
            }],
            ..goal
        };
        assert!(remove_budget_pause(&only_budget).is_none());
    }

    #[test]
    fn goal_policy_rejects_disabled_global_and_agent_config() {
        let ctx = test_goal_context("agent-a");
        let mut config = test_config();
        config.goal.enabled = false;
        let err = ensure_goal_admitted_by_config(&ctx, &config, None).unwrap_err();
        assert!(err.to_string().contains("disabled"));

        config.goal.enabled = true;
        config.agents.get_mut("agent-a").unwrap().goal.enabled = false;
        let err =
            ensure_goal_admitted_by_config(&ctx, &config, config.agent("agent-a")).unwrap_err();
        assert!(err.to_string().contains("disabled for this agent"));

        config.agents.get_mut("agent-a").unwrap().goal.enabled = true;
        config.agents.get_mut("agent-a").unwrap().enabled = false;
        let err =
            ensure_goal_admitted_by_config(&ctx, &config, config.agent("agent-a")).unwrap_err();
        assert!(err.to_string().contains("disabled for this agent"));
    }

    #[tokio::test]
    async fn goal_help_is_rejected_when_goal_mode_is_disabled() {
        let ctx = test_goal_context("agent-a");
        let command = GoalCommand {
            action: GoalCommandAction::Help,
            objective: None,
            task_id: None,
            resume_reason: None,
            budgets: GoalBudgetOverrides::default(),
        };
        let mut config = test_config();

        let help = admit_goal_command(
            ctx.clone(),
            command.clone(),
            &config,
            config.agent("agent-a"),
        )
        .await
        .unwrap();
        assert_eq!(help.message, msg("goal-command-help", &[]));
        assert!(help.message.contains("/goal help"));
        assert!(help.message.contains("/goal --help"));
        assert!(help.message.contains("/goal -h"));
        assert!(
            help.message
                .contains("/goal budget [--tokens=N|unlimited] [--cost=N|unlimited]")
        );
        assert!(!help.continue_goal);

        config.goal.enabled = false;
        let error = admit_goal_command(ctx, command, &config, config.agent("agent-a"))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("disabled"));
    }

    #[tokio::test]
    async fn goal_admission_uses_live_policy_instead_of_enabled_fallback() {
        let fallback = test_config();
        let mut disabled = fallback.clone();
        disabled.goal.enabled = false;
        let scope = GoalRuntimeScope::new(None, None, None)
            .with_config_resolver(Arc::new(move || Arc::new(disabled.clone())));
        let ctx = test_goal_context("agent-a");

        let error = scope_goal_runtime(scope, async {
            admit_goal_command(
                ctx,
                parse_goal_command("/goal help").unwrap(),
                &fallback,
                None,
            )
            .await
            .unwrap_err()
        })
        .await;

        assert!(error.to_string().contains("disabled"));
    }

    #[tokio::test]
    async fn goal_pause_command_records_operator_pause_reason() {
        let (_store, goal_store) = global_test_stores();
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let config = test_config_for_agent(&agent);
        let ctx = test_goal_context(agent)
            .with_originator_route(Some(format!("matrix:{}", uuid::Uuid::new_v4())))
            .with_principal_id(Some(format!("principal-{}", uuid::Uuid::new_v4())));

        let started = admit_goal_command(
            ctx.clone(),
            parse_goal_command("/goal start finish the operator pause test").unwrap(),
            &config,
            None,
        )
        .await
        .unwrap();
        let task_id = started.task_id.expect("start returns task id");

        let paused = admit_goal_command(
            ctx.clone(),
            parse_goal_command("/goal pause maintenance window").unwrap(),
            &config,
            None,
        )
        .await
        .unwrap();

        assert_eq!(paused.status, TaskStatus::Paused);
        let goal = goal_store
            .get_goal_task(&task_id)
            .await
            .unwrap()
            .expect("goal extension is persisted");
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::OperatorPaused));
        assert_eq!(goal.blockers[0].kind, GoalBlockerKind::OperatorPause);
        assert_eq!(goal.blockers[0].message, "maintenance window");

        let status = admit_goal_command(
            ctx,
            parse_goal_command("/goal status").unwrap(),
            &config,
            None,
        )
        .await
        .unwrap();
        assert!(status.message.contains("operator paused"));
        assert!(
            status
                .message
                .contains("operator pause: maintenance window")
        );
        assert!(!status.message.contains("human escalation"));
    }

    #[tokio::test]
    async fn disabled_goal_mode_rejects_every_goal_command() {
        let _ = global_test_stores();
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let config = test_config_for_agent(&agent);
        let ctx = test_goal_context(agent)
            .with_originator_route(Some(format!("route-{}", uuid::Uuid::new_v4())))
            .with_principal_id(Some(format!("principal-{}", uuid::Uuid::new_v4())));
        let started = admit_goal_command(
            ctx.clone(),
            parse_goal_command("/goal start retain controls").unwrap(),
            &config,
            None,
        )
        .await
        .unwrap();
        let task_id = started.task_id.unwrap();
        let mut disabled = config.clone();
        disabled.goal.enabled = false;
        assert!(
            admit_goal_command(
                ctx.clone(),
                parse_goal_command("/goal start rejected").unwrap(),
                &disabled,
                None,
            )
            .await
            .is_err()
        );
        for raw in [
            format!("/goal status {task_id}"),
            format!("/goal pause {task_id}"),
            format!("/goal resume {task_id}"),
            format!("/goal cancel {task_id}"),
            "/goal help".to_string(),
        ] {
            let error = admit_goal_command(
                ctx.clone(),
                parse_goal_command(&raw).unwrap(),
                &disabled,
                None,
            )
            .await
            .unwrap_err();
            assert!(error.to_string().contains("disabled"), "{raw}: {error}");
        }
    }

    #[test]
    fn goal_policy_rejects_disallowed_surface_and_channel_type() {
        let mut config = test_config();
        config.goal.allowed_command_surfaces = vec!["web".into()];
        let ctx = test_goal_context("agent-a");
        let err =
            ensure_goal_admitted_by_config(&ctx, &config, config.agent("agent-a")).unwrap_err();
        assert!(err.to_string().contains("command surface `channel`"));

        config.goal.allowed_command_surfaces = vec!["channel".into()];
        config.goal.allowed_channel_types = vec!["telegram".into()];
        let err =
            ensure_goal_admitted_by_config(&ctx, &config, config.agent("agent-a")).unwrap_err();
        assert!(err.to_string().contains("channel type `matrix`"));

        let missing_channel_type = test_goal_context("agent-a").with_channel_type(None);
        let err =
            ensure_goal_admitted_by_config(&missing_channel_type, &config, config.agent("agent-a"))
                .unwrap_err();
        assert!(err.to_string().contains("channel type is unavailable"));
    }

    #[test]
    fn live_policy_binding_revokes_global_agent_surface_and_channel_scopes() {
        let agent = "agent-a";
        let binding = ActiveGoalControlBinding {
            task_id: "goal-a".into(),
            agent: agent.into(),
            continuation_context: Some(TaskContinuationContext {
                channel: "matrix".into(),
                channel_alias: Some("default".into()),
                reply_target: "room".into(),
                sender: "operator".into(),
                thread_ts: None,
                interruption_scope_id: None,
                conversation_scope: TaskContinuationConversationScope::ReplyTarget,
            }),
        };
        let config = test_config();
        assert!(goal_control_binding_is_allowed(&binding, &config));
        let allowed = config.clone();

        let mut revoked = config.clone();
        revoked.goal.enabled = false;
        assert!(!goal_control_binding_is_allowed(&binding, &revoked));

        let mut revoked = config.clone();
        revoked.agents.get_mut(agent).unwrap().goal.enabled = false;
        assert!(!goal_control_binding_is_allowed(&binding, &revoked));

        let mut revoked = config.clone();
        revoked.agents.get_mut(agent).unwrap().enabled = false;
        assert!(!goal_control_binding_is_allowed(&binding, &revoked));

        let mut revoked = config.clone();
        revoked.goal.allowed_command_surfaces = vec!["web".into()];
        assert!(!goal_control_binding_is_allowed(&binding, &revoked));

        let mut revoked = config.clone();
        revoked.goal.allowed_channel_types = vec!["telegram".into()];
        assert!(!goal_control_binding_is_allowed(&binding, &revoked));

        let mut revoked = config.clone();
        revoked.channels.matrix.remove("default");
        assert!(!goal_control_binding_is_allowed(&binding, &revoked));

        let mut revoked = config.clone();
        revoked.channels.matrix.get_mut("default").unwrap().enabled = false;
        assert!(!goal_control_binding_is_allowed(&binding, &revoked));

        let mut revoked = config;
        revoked.agents.get_mut(agent).unwrap().channels.clear();
        revoked.agents.insert(
            "agent-b".into(),
            AliasedAgentConfig {
                channels: vec![zeroclaw_config::providers::ChannelRef::new(
                    "matrix.default",
                )],
                ..AliasedAgentConfig::default()
            },
        );
        assert!(!goal_control_binding_is_allowed(&binding, &revoked));

        let missing_binding = ActiveGoalControlBinding {
            continuation_context: None,
            ..binding
        };
        assert!(!goal_control_binding_is_allowed(&missing_binding, &allowed));
    }

    #[tokio::test]
    async fn policy_revocation_cancels_each_revoked_scope_and_never_revives_it() {
        #[derive(Clone, Copy)]
        enum Revocation {
            Global,
            AgentGoal,
            AgentDisabled,
            Surface,
            ChannelType,
            ChannelRemoved,
            ChannelDisabled,
            ChannelOwnership,
            DuplicateChannelOwnership,
        }

        for revocation in [
            Revocation::Global,
            Revocation::AgentGoal,
            Revocation::AgentDisabled,
            Revocation::Surface,
            Revocation::ChannelType,
            Revocation::ChannelRemoved,
            Revocation::ChannelDisabled,
            Revocation::ChannelOwnership,
            Revocation::DuplicateChannelOwnership,
        ] {
            let store = SqliteTaskStore::new_in_memory().unwrap();
            create_policy_revocation_goal(&store, "target", "agent-a", "matrix").await;
            create_policy_revocation_goal(&store, "unaffected", "agent-b", "telegram").await;

            let mut allowed = test_config();
            allowed.goal.allowed_command_surfaces = vec!["channel".into()];
            allowed.goal.allowed_channel_types = vec!["matrix".into(), "telegram".into()];
            configure_test_goal_channel(&mut allowed, "agent-b", "telegram", "default");
            let mut revoked = allowed.clone();
            match revocation {
                Revocation::Global => revoked.goal.enabled = false,
                Revocation::AgentGoal => {
                    revoked.agents.get_mut("agent-a").unwrap().goal.enabled = false;
                }
                Revocation::AgentDisabled => {
                    revoked.agents.get_mut("agent-a").unwrap().enabled = false;
                }
                Revocation::Surface => {
                    revoked.goal.allowed_command_surfaces = vec!["web".into()];
                }
                Revocation::ChannelType => {
                    revoked.goal.allowed_channel_types = vec!["telegram".into()];
                }
                Revocation::ChannelRemoved => {
                    revoked.channels.matrix.remove("default");
                }
                Revocation::ChannelDisabled => {
                    revoked.channels.matrix.get_mut("default").unwrap().enabled = false;
                }
                Revocation::ChannelOwnership => {
                    revoked.agents.get_mut("agent-a").unwrap().channels.clear();
                }
                Revocation::DuplicateChannelOwnership => {
                    revoked.agents.insert(
                        "agent-z".into(),
                        AliasedAgentConfig {
                            channels: vec![zeroclaw_config::providers::ChannelRef::new(
                                "matrix.default",
                            )],
                            ..AliasedAgentConfig::default()
                        },
                    );
                }
            }

            let revoked_ids = goal_ids_revoked_by_config(&store, &revoked).await.unwrap();
            let cancelled = cancel_goals_for_policy_revocation(&store, &revoked_ids)
                .await
                .unwrap();
            assert!(cancelled.iter().any(|task_id| task_id == "target"));
            assert_eq!(
                store.get("target").await.unwrap().unwrap().status,
                TaskStatus::Cancelled
            );

            let unaffected_status = store.get("unaffected").await.unwrap().unwrap().status;
            match revocation {
                Revocation::Global | Revocation::Surface => {
                    assert_eq!(unaffected_status, TaskStatus::Cancelled);
                }
                Revocation::AgentGoal
                | Revocation::AgentDisabled
                | Revocation::ChannelType
                | Revocation::ChannelRemoved
                | Revocation::ChannelDisabled
                | Revocation::ChannelOwnership
                | Revocation::DuplicateChannelOwnership => {
                    assert_eq!(unaffected_status, TaskStatus::Running);
                }
            }

            assert!(
                cancel_goals_for_policy_revocation(
                    &store,
                    &goal_ids_revoked_by_config(&store, &allowed).await.unwrap(),
                )
                .await
                .unwrap()
                .iter()
                .all(|task_id| task_id != "target"),
                "re-enabling policy must not revive or re-transition a cancelled goal"
            );
            assert_eq!(
                store.get("target").await.unwrap().unwrap().status,
                TaskStatus::Cancelled
            );
        }
    }

    #[tokio::test]
    async fn scoped_goal_state_update_publishes_channel_message() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let admission = GoalAdmission {
            task_id: Some("goal-1".into()),
            status: TaskStatus::Running,
            message: "Goal `goal-1` started.".into(),
            continuation_reason: None,
            continue_goal: true,
        };

        scope_goal_state_updates(Some(GoalStateUpdateSink::new(tx)), async {
            publish_goal_state_update(&admission);
        })
        .await;

        assert_eq!(
            rx.recv().await,
            Some(GoalStateUpdateEvent::Status(
                "Goal `goal-1` started.".into()
            ))
        );
    }

    #[tokio::test]
    async fn scoped_goal_verifier_start_publishes_progress_event() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut config = test_config();
        config.goal.verifier.enabled = true;
        let goal = GoalTaskRecord {
            task_id: "goal-1".into(),
            objective: "ship it".into(),
            effective_token_limit: Some(10_000),
            effective_cost_limit_usd: Some(1.25),
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
        };

        scope_goal_state_updates(Some(GoalStateUpdateSink::new(tx)), async {
            let usage = goal_usage_totals(Some(&config), "goal-1");
            let budget = goal_budget_summary(&goal, usage.as_ref());
            publish_goal_verifier_started("goal-1", &budget);
        })
        .await;

        let Some(GoalStateUpdateEvent::VerifierStarted(message)) = rx.recv().await else {
            panic!("verifier progress should use a typed progress event");
        };
        assert!(message.starts_with("🔎 Verifying goal `goal-1` status."));
        assert!(message.contains("Budget:"));
    }

    #[tokio::test]
    async fn duplicate_resume_publishes_only_the_committed_controller_outcome() {
        let fixture = create_running_goal_fixture("resume once").await;
        let running = resolve_goal(
            fixture.store.as_ref(),
            fixture.goal_store.as_ref(),
            &fixture.ctx,
            Some(fixture.task_id.clone()),
        )
        .await
        .unwrap();
        pause_goal_for_resolved_task_with_budget(
            fixture.goal_store.as_ref(),
            running,
            GoalPauseState {
                reason: GoalPauseReason::OperatorPaused,
                description: Some("pause before duplicate resume".into()),
                blockers: Vec::new(),
            },
            "Budget: test".into(),
        )
        .await
        .unwrap();
        let first = resolve_goal(
            fixture.store.as_ref(),
            fixture.goal_store.as_ref(),
            &fixture.ctx,
            Some(fixture.task_id.clone()),
        )
        .await
        .unwrap();
        let duplicate = first.clone();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        scope_goal_state_updates(Some(GoalStateUpdateSink::new(tx)), async {
            let committed = resume_resolved_goal(
                fixture.goal_store.as_ref(),
                "test-boot",
                &fixture.ctx,
                None,
                None,
                first,
            )
            .await
            .unwrap();
            assert!(committed.continue_goal);
            publish_goal_state_update(&committed);

            let error = resume_resolved_goal(
                fixture.goal_store.as_ref(),
                "test-boot",
                &fixture.ctx,
                None,
                None,
                duplicate,
            )
            .await
            .expect_err("duplicate stale resume must lose its CAS");
            assert!(error.to_string().contains("Failed to update goal"));
        })
        .await;

        assert!(matches!(rx.try_recv(), Ok(GoalStateUpdateEvent::Status(_))));
        assert!(rx.try_recv().is_err(), "only one resume may publish");
        assert_eq!(
            fixture
                .store
                .get(&fixture.task_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Running
        );
    }

    #[tokio::test]
    async fn cancel_losing_to_terminal_transition_emits_no_controller_outcome() {
        let fixture = create_running_goal_fixture("complete before cancel").await;
        let stale = resolve_goal(
            fixture.store.as_ref(),
            fixture.goal_store.as_ref(),
            &fixture.ctx,
            Some(fixture.task_id.clone()),
        )
        .await
        .unwrap();
        assert!(
            fixture
                .goal_store
                .complete_running_goal_task(&fixture.task_id, "already complete".into())
                .await
                .unwrap()
        );

        let error = cancel_resolved_goal(fixture.goal_store.as_ref(), stale, None)
            .await
            .expect_err("stale cancel must lose to terminal completion");
        assert!(error.to_string().contains("Failed to update goal"));
        assert_eq!(
            fixture
                .store
                .get(&fixture.task_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Completed
        );
    }

    #[tokio::test]
    async fn stale_pause_cannot_overwrite_newer_operator_pause() {
        let fixture = create_running_goal_fixture("preserve newest pause").await;
        let newer = resolve_goal(
            fixture.store.as_ref(),
            fixture.goal_store.as_ref(),
            &fixture.ctx,
            Some(fixture.task_id.clone()),
        )
        .await
        .unwrap();
        let stale = newer.clone();
        pause_goal_for_resolved_task_with_budget(
            fixture.goal_store.as_ref(),
            newer,
            GoalPauseState {
                reason: GoalPauseReason::OperatorPaused,
                description: Some("newer operator pause".into()),
                blockers: Vec::new(),
            },
            "Budget: test".into(),
        )
        .await
        .unwrap();

        let error = pause_goal_for_resolved_task_with_budget(
            fixture.goal_store.as_ref(),
            stale,
            GoalPauseState {
                reason: GoalPauseReason::BudgetUnavailable,
                description: Some("stale accounting pause".into()),
                blockers: Vec::new(),
            },
            "Budget: stale".into(),
        )
        .await
        .expect_err("stale pause must lose to the newer operator pause");
        assert!(error.to_string().contains("Failed to pause goal"));

        let goal = fixture
            .goal_store
            .get_goal_task(&fixture.task_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::OperatorPaused));
        assert_eq!(
            goal.pause_description.as_deref(),
            Some("newer operator pause")
        );
    }

    #[tokio::test]
    async fn autonomous_turn_budget_gate_allows_running_goal_under_limit() {
        let _cost_guard = goal_cost_tracker_test_lock().await;
        let (_store, goal_store) = global_test_stores();
        let task_id = format!("goal-{}", uuid::Uuid::new_v4());
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let ctx = GoalAdmissionContext::new(agent.clone())
            .with_originator_route(Some(route.clone()))
            .with_principal_id(Some(principal.clone()));
        goal_store
            .create_goal(
                TaskRecord {
                    id: task_id.clone(),
                    kind: TaskKind::Goal,
                    agent,
                    status: TaskStatus::Running,
                    owner_pid: std::process::id(),
                    owner_boot_id: "test-boot".into(),
                    heartbeat_at: None,
                    depth: 0,
                    parent_id: None,
                    originator_route: Some(route),
                    delivered: false,
                    idem_key: None,
                    principal_id: Some(principal),
                    started_at: chrono::Utc::now().to_rfc3339(),
                    finished_at: None,
                },
                GoalTaskRecord {
                    task_id,
                    objective: "ship it".into(),
                    effective_token_limit: Some(10_000),
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                },
                None,
            )
            .await
            .unwrap();
        let tmp = tempfile::TempDir::new().unwrap();
        let mut config = test_config();
        config.data_dir = tmp.path().to_path_buf();
        config.cost.enabled = true;
        let _tracker =
            CostTracker::get_or_init_global(config.cost.clone(), &config.data_dir).unwrap();

        let admission = admit_goal_autonomous_turn(&ctx, &config).await.unwrap();

        assert!(admission.is_none());
    }

    #[tokio::test]
    async fn autonomous_turn_uses_bound_exact_task_over_newer_context_match() {
        let fixture = create_running_goal_fixture("finish the original").await;
        let original = resolve_goal(
            fixture.store.as_ref(),
            fixture.goal_store.as_ref(),
            &fixture.ctx,
            Some(fixture.task_id.clone()),
        )
        .await
        .unwrap();
        cancel_resolved_goal(fixture.goal_store.as_ref(), original, None)
            .await
            .unwrap();

        let original_task = fixture.store.get(&fixture.task_id).await.unwrap().unwrap();
        let replacement_task_id = format!("goal-{}", uuid::Uuid::new_v4());
        fixture
            .goal_store
            .create_goal(
                TaskRecord {
                    id: replacement_task_id.clone(),
                    status: TaskStatus::Running,
                    started_at: chrono::Utc::now().to_rfc3339(),
                    finished_at: None,
                    ..original_task
                },
                GoalTaskRecord {
                    task_id: replacement_task_id,
                    objective: "newer replacement".into(),
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                },
                None,
            )
            .await
            .unwrap();

        let exact_ctx = fixture
            .ctx
            .clone()
            .with_goal_task_id(Some(fixture.task_id.clone()));
        let admission = admit_goal_autonomous_turn(&exact_ctx, &test_config())
            .await
            .unwrap()
            .expect("the bound cancelled goal must stop its stale continuation");

        assert_eq!(admission.task_id.as_deref(), Some(fixture.task_id.as_str()));
        assert_eq!(admission.status, TaskStatus::Cancelled);
    }

    #[tokio::test]
    async fn autonomous_turn_budget_gate_pauses_before_next_model_turn() {
        let _cost_guard = goal_cost_tracker_test_lock().await;
        let (store, goal_store) = global_test_stores();
        let task_id = format!("goal-{}", uuid::Uuid::new_v4());
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let ctx = GoalAdmissionContext::new(agent.clone())
            .with_originator_route(Some(route.clone()))
            .with_principal_id(Some(principal.clone()));
        goal_store
            .create_goal(
                TaskRecord {
                    id: task_id.clone(),
                    kind: TaskKind::Goal,
                    agent: agent.clone(),
                    status: TaskStatus::Running,
                    owner_pid: std::process::id(),
                    owner_boot_id: "test-boot".into(),
                    heartbeat_at: None,
                    depth: 0,
                    parent_id: None,
                    originator_route: Some(route),
                    delivered: false,
                    idem_key: None,
                    principal_id: Some(principal),
                    started_at: chrono::Utc::now().to_rfc3339(),
                    finished_at: None,
                },
                GoalTaskRecord {
                    task_id: task_id.clone(),
                    objective: "ship it".into(),
                    effective_token_limit: Some(1_000),
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                },
                None,
            )
            .await
            .unwrap();
        let tmp = tempfile::TempDir::new().unwrap();
        let mut config = test_config();
        config.data_dir = tmp.path().to_path_buf();
        config.cost.enabled = true;
        config.cost.track_per_agent = true;
        let tracker =
            CostTracker::get_or_init_global(config.cost.clone(), &config.data_dir).unwrap();
        tracker
            .record_usage_with_task_attribution(
                zeroclaw_config::cost::types::TokenUsage::new(
                    "test/model",
                    1_000,
                    500,
                    0,
                    1.0,
                    2.0,
                    0.0,
                ),
                Some(&agent),
                Some(&task_id),
            )
            .unwrap();

        let admission = admit_goal_autonomous_turn(&ctx, &config)
            .await
            .unwrap()
            .expect("exhausted budget should block the autonomous turn");

        assert_eq!(admission.status, TaskStatus::Paused);
        assert!(!admission.continue_goal);
        assert!(admission.message.contains("Budget:"));
        let task = store.get(&task_id).await.unwrap().unwrap();
        let goal = goal_store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Paused);
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::BudgetExhausted));
        assert_eq!(goal.blockers[0].kind, GoalBlockerKind::Budget);
    }

    #[tokio::test]
    async fn autonomous_turn_gate_reports_already_paused_goal() {
        let (store, goal_store) = global_test_stores();
        let task_id = format!("goal-{}", uuid::Uuid::new_v4());
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let ctx = GoalAdmissionContext::new(agent.clone())
            .with_originator_route(Some(route.clone()))
            .with_principal_id(Some(principal.clone()));
        goal_store
            .create_goal(
                TaskRecord {
                    id: task_id.clone(),
                    kind: TaskKind::Goal,
                    agent,
                    status: TaskStatus::Paused,
                    owner_pid: std::process::id(),
                    owner_boot_id: "test-boot".into(),
                    heartbeat_at: None,
                    depth: 0,
                    parent_id: None,
                    originator_route: Some(route),
                    delivered: false,
                    idem_key: None,
                    principal_id: Some(principal),
                    started_at: chrono::Utc::now().to_rfc3339(),
                    finished_at: None,
                },
                GoalTaskRecord {
                    task_id: task_id.clone(),
                    objective: "wait for operator".into(),
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: Some(GoalPauseReason::NeedsUserInput),
                    pause_description: Some("Need operator answer".into()),
                    blockers: vec![GoalBlocker {
                        kind: GoalBlockerKind::NeedsUserInput,
                        message: "Need operator answer".into(),
                        payload: None,
                    }],
                },
                None,
            )
            .await
            .unwrap();

        let admission = admit_goal_autonomous_turn(&ctx, &test_config())
            .await
            .unwrap()
            .expect("paused goal should stop stale autonomous continuation");

        assert_eq!(admission.status, TaskStatus::Paused);
        assert!(!admission.continue_goal);
        assert!(admission.message.contains("needs user input"));
        assert!(!admission.message.contains("needs_user_input"));
        assert_eq!(
            store.get(&task_id).await.unwrap().unwrap().status,
            TaskStatus::Paused
        );
        assert_eq!(
            goal_store
                .get_goal_task(&task_id)
                .await
                .unwrap()
                .unwrap()
                .pause_reason,
            Some(GoalPauseReason::NeedsUserInput)
        );
    }

    #[tokio::test]
    async fn evaluate_goal_turn_completes_running_goal_when_verifier_disabled() {
        let (store, goal_store) = global_test_stores();
        let task_id = format!("goal-{}", uuid::Uuid::new_v4());
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let ctx = GoalAdmissionContext::new(agent.clone())
            .with_originator_route(Some(route.clone()))
            .with_principal_id(Some(principal.clone()));
        goal_store
            .create_goal(
                TaskRecord {
                    id: task_id.clone(),
                    kind: TaskKind::Goal,
                    agent: agent.clone(),
                    status: TaskStatus::Running,
                    owner_pid: std::process::id(),
                    owner_boot_id: "test-boot".into(),
                    heartbeat_at: None,
                    depth: 0,
                    parent_id: None,
                    originator_route: Some(route),
                    delivered: false,
                    idem_key: None,
                    principal_id: Some(principal),
                    started_at: chrono::Utc::now().to_rfc3339(),
                    finished_at: None,
                },
                GoalTaskRecord {
                    task_id: task_id.clone(),
                    objective: "ship it".into(),
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                },
                None,
            )
            .await
            .unwrap();
        let mut config = test_config();
        config.goal.verifier.enabled = false;

        let outcome = evaluate_goal_turn(&ctx, &config, "done").await.unwrap();

        let Some(GoalTurnEvaluation::Completed {
            task_id: completed_id,
            message,
        }) = outcome
        else {
            panic!("running goal should complete when verifier is disabled");
        };
        assert_eq!(completed_id, task_id);
        assert!(message.starts_with("✅ Goal"));
        assert!(message.contains("Budget:"));
        assert_eq!(
            store.get(&task_id).await.unwrap().unwrap().status,
            TaskStatus::Completed
        );
    }

    #[tokio::test]
    async fn evaluate_goal_turn_uses_injected_verifier() {
        let (store, goal_store) = global_test_stores();
        let task_id = format!("goal-{}", uuid::Uuid::new_v4());
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let ctx = GoalAdmissionContext::new(agent.clone())
            .with_originator_route(Some(route.clone()))
            .with_principal_id(Some(principal.clone()));
        goal_store
            .create_goal(
                TaskRecord {
                    id: task_id.clone(),
                    kind: TaskKind::Goal,
                    agent: agent.clone(),
                    status: TaskStatus::Running,
                    owner_pid: std::process::id(),
                    owner_boot_id: "test-boot".into(),
                    heartbeat_at: None,
                    depth: 0,
                    parent_id: None,
                    originator_route: Some(route),
                    delivered: false,
                    idem_key: None,
                    principal_id: Some(principal),
                    started_at: chrono::Utc::now().to_rfc3339(),
                    finished_at: None,
                },
                GoalTaskRecord {
                    task_id: task_id.clone(),
                    objective: "ship it".into(),
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                },
                None,
            )
            .await
            .unwrap();
        let mut config = test_config();
        config.goal.verifier.enabled = true;
        let verifier = StubGoalVerifier {
            decision: GoalVerifierDecision::Continue {
                notes: "CONTINUE\nstub says keep going".into(),
            },
        };

        let outcome = evaluate_goal_turn_with_verifier(&ctx, &config, "not done", &verifier)
            .await
            .unwrap();

        let Some(GoalTurnEvaluation::Continue {
            task_id: continued_id,
            objective,
            notes,
            message,
        }) = outcome
        else {
            panic!("stub verifier should request another autonomous turn");
        };
        assert_eq!(continued_id, task_id);
        assert_eq!(objective, "ship it");
        assert!(notes.contains("stub says keep going"));
        assert!(message.starts_with("🔁 Goal"));
        assert!(message.contains("Budget:"));
        assert_eq!(
            store.get(&task_id).await.unwrap().unwrap().status,
            TaskStatus::Running
        );
    }

    #[tokio::test]
    async fn verifier_usage_stays_with_resolved_goal_during_replacement_race() {
        let _cost_guard = goal_cost_tracker_test_lock().await;
        let fixture = create_running_goal_fixture("ship original goal").await;
        let replacement_task_id = format!("goal-replacement-{}", uuid::Uuid::new_v4());
        let workspace = tempfile::TempDir::new().unwrap();
        let mut config = test_config();
        config.data_dir = workspace.path().to_path_buf();
        config.goal.verifier.enabled = true;
        let tracker = goal_usage_ledger(Some(&config)).expect("goal usage ledger");
        let verifier = ReplacingGoalVerifier {
            goal_store: Arc::clone(&fixture.goal_store),
            replacement_task_id: replacement_task_id.clone(),
        };

        let outcome =
            evaluate_goal_turn_with_verifier(&fixture.ctx, &config, "looks done", &verifier)
                .await
                .unwrap();

        assert!(
            outcome.is_none(),
            "controller must stop after the original goal loses lifecycle ownership"
        );
        assert_eq!(
            fixture
                .store
                .get(&fixture.task_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Cancelled
        );
        assert_eq!(
            fixture
                .store
                .get(&replacement_task_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Running
        );
        let (original_tokens, _, _, original_usage_available) = tracker
            .get_usage_totals_for_task_with_pricing(&fixture.task_id)
            .unwrap();
        let (replacement_tokens, _, _, replacement_usage_available) = tracker
            .get_usage_totals_for_task_with_pricing(&replacement_task_id)
            .unwrap();
        assert_eq!(original_tokens, 100);
        assert!(original_usage_available);
        assert_eq!(replacement_tokens, 0);
        assert!(replacement_usage_available);
    }

    #[tokio::test]
    async fn completion_limit_cas_loss_returns_running_goal_to_continuation() {
        let _cost_guard = goal_cost_tracker_test_lock().await;
        let fixture = create_running_goal_fixture("ship it").await;
        let workspace = tempfile::TempDir::new().unwrap();
        let tracker = CostTracker::new(
            zeroclaw_config::schema::CostConfig::default(),
            workspace.path(),
        )
        .unwrap();
        let stale = resolve_goal(
            fixture.store.as_ref(),
            fixture.goal_store.as_ref(),
            &fixture.ctx,
            Some(fixture.task_id.clone()),
        )
        .await
        .unwrap();
        fixture
            .goal_store
            .update_goal_limits(&fixture.task_id, Some(1_000), None)
            .await
            .unwrap();

        let outcome = complete_goal_after_verification(
            fixture.store.as_ref(),
            fixture.goal_store.as_ref(),
            &fixture.ctx,
            Some(&tracker),
            stale,
            "candidate completion",
            Some(GoalUsageTotals::default()),
        )
        .await
        .unwrap();

        let Some(GoalTurnEvaluation::Continue {
            task_id,
            objective,
            notes,
            ..
        }) = outcome
        else {
            panic!("a limit CAS loss while Running must retain an executor");
        };
        assert_eq!(task_id, fixture.task_id);
        assert_eq!(objective, "ship it");
        assert!(notes.contains("budget changed"));
        assert_eq!(
            fixture
                .store
                .get(&fixture.task_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Running
        );
        assert_eq!(
            fixture
                .goal_store
                .get_goal_task(&fixture.task_id)
                .await
                .unwrap()
                .unwrap()
                .effective_token_limit,
            Some(1_000)
        );
    }

    #[tokio::test]
    async fn completion_cas_loss_to_terminal_state_stops_without_continuation() {
        let _cost_guard = goal_cost_tracker_test_lock().await;
        let fixture = create_running_goal_fixture("ship it").await;
        let stale = resolve_goal(
            fixture.store.as_ref(),
            fixture.goal_store.as_ref(),
            &fixture.ctx,
            Some(fixture.task_id.clone()),
        )
        .await
        .unwrap();
        assert!(
            fixture
                .goal_store
                .cancel_goal_task_if_status(
                    &fixture.task_id,
                    TaskStatus::Running,
                    "operator cancelled".into(),
                )
                .await
                .unwrap()
        );

        let outcome = complete_goal_after_verification(
            fixture.store.as_ref(),
            fixture.goal_store.as_ref(),
            &fixture.ctx,
            None,
            stale,
            "candidate completion",
            Some(GoalUsageTotals::default()),
        )
        .await
        .unwrap();

        assert!(
            outcome.is_none(),
            "a terminal lifecycle winner must not publish a continuation"
        );
        assert_eq!(
            fixture
                .store
                .get(&fixture.task_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Cancelled
        );
    }

    #[tokio::test]
    async fn evaluate_goal_turn_pauses_when_verifier_blocks_completion() {
        let fixture = create_running_goal_fixture("ship it").await;
        let mut config = test_config();
        config.goal.verifier.enabled = true;
        let verifier = StubGoalVerifier {
            decision: GoalVerifierDecision::Blocked {
                pause: GoalPauseState {
                    reason: GoalPauseReason::VerifierBlocked,
                    description: Some("verifier requested operator review".into()),
                    blockers: vec![GoalBlocker {
                        kind: GoalBlockerKind::Verifier,
                        message: "Verifier requested operator review".into(),
                        payload: Some(serde_json::json!({"verdict": "blocked"})),
                    }],
                },
            },
        };

        let outcome =
            evaluate_goal_turn_with_verifier(&fixture.ctx, &config, "looks done", &verifier)
                .await
                .unwrap();

        let Some(GoalTurnEvaluation::Paused {
            task_id: paused_id,
            message,
        }) = outcome
        else {
            panic!("blocked verifier verdict must pause the goal");
        };
        assert_eq!(paused_id, fixture.task_id);
        assert!(message.starts_with("⏸️ Goal"));
        assert_eq!(
            fixture
                .store
                .get(&fixture.task_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Paused
        );
        let goal = fixture
            .goal_store
            .get_goal_task(&fixture.task_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::VerifierBlocked));
        assert_eq!(
            goal.pause_description.as_deref(),
            Some("verifier requested operator review")
        );
        assert_eq!(goal.blockers.len(), 1);
        assert_eq!(goal.blockers[0].kind, GoalBlockerKind::Verifier);
        assert_eq!(
            goal.blockers[0].payload.as_ref().unwrap()["verdict"],
            "blocked"
        );
    }

    #[tokio::test]
    async fn evaluate_goal_turn_pauses_when_verifier_errors() {
        let fixture = create_running_goal_fixture("ship it").await;
        let mut config = test_config();
        config.goal.verifier.enabled = true;

        let outcome = evaluate_goal_turn_with_verifier(
            &fixture.ctx,
            &config,
            "looks done",
            &FailingGoalVerifier,
        )
        .await
        .unwrap();

        let Some(GoalTurnEvaluation::Paused {
            task_id: paused_id,
            message,
        }) = outcome
        else {
            panic!("verifier outage must pause the goal");
        };
        assert_eq!(paused_id, fixture.task_id);
        assert!(message.starts_with("⏸️ Goal"));
        assert_eq!(
            fixture
                .store
                .get(&fixture.task_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Paused
        );
        let goal = fixture
            .goal_store
            .get_goal_task(&fixture.task_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::VerifierBlocked));
        assert!(
            goal.pause_description
                .as_deref()
                .unwrap()
                .contains("provider offline")
        );
        assert_eq!(goal.blockers.len(), 1);
        assert_eq!(goal.blockers[0].kind, GoalBlockerKind::Verifier);
        assert!(goal.blockers[0].message.contains("provider offline"));
    }

    #[tokio::test]
    async fn evaluate_goal_turn_pauses_when_verifier_continue_exhausts_budget() {
        let _cost_guard = goal_cost_tracker_test_lock().await;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use zeroclaw_config::schema::{CustomModelProviderConfig, ModelProviderConfig};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "CONTINUE\nMore work remains."
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 75,
                    "completion_tokens": 50
                }
            })))
            .mount(&server)
            .await;

        let (store, goal_store) = global_test_stores();
        let task_id = format!("goal-{}", uuid::Uuid::new_v4());
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let ctx = GoalAdmissionContext::new(agent.clone())
            .with_originator_route(Some(route.clone()))
            .with_principal_id(Some(principal.clone()));
        goal_store
            .create_goal(
                TaskRecord {
                    id: task_id.clone(),
                    kind: TaskKind::Goal,
                    agent: agent.clone(),
                    status: TaskStatus::Running,
                    owner_pid: std::process::id(),
                    owner_boot_id: "test-boot".into(),
                    heartbeat_at: None,
                    depth: 0,
                    parent_id: None,
                    originator_route: Some(route),
                    delivered: false,
                    idem_key: None,
                    principal_id: Some(principal),
                    started_at: chrono::Utc::now().to_rfc3339(),
                    finished_at: None,
                },
                GoalTaskRecord {
                    task_id: task_id.clone(),
                    objective: "ship it".into(),
                    effective_token_limit: Some(100),
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                },
                None,
            )
            .await
            .unwrap();

        let temp = tempfile::tempdir().unwrap();
        let mut config = Config {
            data_dir: temp.path().to_path_buf(),
            ..test_config()
        };
        config.cost.enabled = true;
        config.goal.verifier.enabled = true;
        config.goal.verifier.model_provider = "custom.verifier".into();
        config.goal.verifier.model = Some("model".into());
        config.providers.models.custom.insert(
            "verifier".into(),
            CustomModelProviderConfig {
                base: ModelProviderConfig {
                    api_key: Some("test-key".into()),
                    uri: Some(server.uri()),
                    model: Some("model".into()),
                    pricing: [("model.input".into(), 1.0), ("model.output".into(), 2.0)]
                        .into_iter()
                        .collect(),
                    ..ModelProviderConfig::default()
                },
            },
        );

        let outcome = evaluate_goal_turn(&ctx, &config, "looks done")
            .await
            .unwrap();

        let Some(GoalTurnEvaluation::Paused {
            task_id: paused_id,
            message,
        }) = outcome
        else {
            panic!("verifier usage should exhaust the goal budget before continuation");
        };
        assert_eq!(paused_id, task_id);
        assert!(message.starts_with("⏸️ Goal"));
        assert!(message.contains("Budget:"));
        let task = store.get(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Paused);
        let goal = goal_store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::BudgetExhausted));
    }

    async fn assert_unlimited_verifier_usage_failure_pauses(response: serde_json::Value) {
        let _cost_guard = goal_cost_tracker_test_lock().await;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use zeroclaw_config::schema::{CustomModelProviderConfig, ModelProviderConfig};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(response))
            .mount(&server)
            .await;

        let fixture = create_running_goal_fixture("ship it").await;

        let temp = tempfile::tempdir().unwrap();
        let mut config = Config {
            data_dir: temp.path().to_path_buf(),
            ..test_config()
        };
        config.goal.verifier.enabled = true;
        config.goal.verifier.model_provider = "custom.verifier".into();
        config.goal.verifier.model = Some("model".into());
        config.providers.models.custom.insert(
            "verifier".into(),
            CustomModelProviderConfig {
                base: ModelProviderConfig {
                    api_key: Some("test-key".into()),
                    uri: Some(server.uri()),
                    model: Some("model".into()),
                    ..ModelProviderConfig::default()
                },
            },
        );

        let outcome = evaluate_goal_turn(&fixture.ctx, &config, "looks done")
            .await
            .unwrap();

        let Some(GoalTurnEvaluation::Paused {
            task_id: paused_id,
            message,
        }) = outcome
        else {
            panic!("unmetered verifier call must pause an unlimited goal: {outcome:?}");
        };
        assert_eq!(paused_id, fixture.task_id);
        assert!(message.contains("budget accounting is unavailable"));
        assert_eq!(
            fixture
                .store
                .get(&fixture.task_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Paused
        );
        let goal = fixture
            .goal_store
            .get_goal_task(&fixture.task_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::BudgetUnavailable));
        let usage = goal_usage_totals(Some(&config), &fixture.task_id).unwrap();
        assert!(!usage.usage_available);
        assert!(!usage.cost_tracking_available);
    }

    #[tokio::test]
    async fn unlimited_goal_pauses_when_verifier_omits_usage() {
        assert_unlimited_verifier_usage_failure_pauses(serde_json::json!({
            "choices": [{
                "message": {"role": "assistant", "content": "COMPLETE\nverified"}
            }]
        }))
        .await;
    }

    #[tokio::test]
    async fn unlimited_goal_pauses_when_verifier_reports_empty_usage() {
        assert_unlimited_verifier_usage_failure_pauses(serde_json::json!({
            "choices": [{
                "message": {"role": "assistant", "content": "COMPLETE\nverified"}
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0
            }
        }))
        .await;
    }

    #[tokio::test]
    async fn evaluate_goal_turn_pauses_when_verifier_usage_is_unpriced_under_cost_budget() {
        let _cost_guard = goal_cost_tracker_test_lock().await;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use zeroclaw_config::schema::{CustomModelProviderConfig, ModelProviderConfig};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "CONTINUE\nMore work remains."
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 75,
                    "completion_tokens": 50
                }
            })))
            .mount(&server)
            .await;

        let (store, goal_store) = global_test_stores();
        let task_id = format!("goal-{}", uuid::Uuid::new_v4());
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let model = format!("unpriced-goal-model-{}", uuid::Uuid::new_v4());
        let ctx = GoalAdmissionContext::new(agent.clone())
            .with_originator_route(Some(route.clone()))
            .with_principal_id(Some(principal.clone()));
        goal_store
            .create_goal(
                TaskRecord {
                    id: task_id.clone(),
                    kind: TaskKind::Goal,
                    agent: agent.clone(),
                    status: TaskStatus::Running,
                    owner_pid: std::process::id(),
                    owner_boot_id: "test-boot".into(),
                    heartbeat_at: None,
                    depth: 0,
                    parent_id: None,
                    originator_route: Some(route),
                    delivered: false,
                    idem_key: None,
                    principal_id: Some(principal),
                    started_at: chrono::Utc::now().to_rfc3339(),
                    finished_at: None,
                },
                GoalTaskRecord {
                    task_id: task_id.clone(),
                    objective: "ship it".into(),
                    effective_token_limit: None,
                    effective_cost_limit_usd: Some(0.01),
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                },
                None,
            )
            .await
            .unwrap();

        let temp = tempfile::tempdir().unwrap();
        let mut config = Config {
            data_dir: temp.path().to_path_buf(),
            ..test_config()
        };
        config.cost.enabled = true;
        config.goal.verifier.enabled = true;
        config.goal.verifier.model_provider = "custom.verifier".into();
        config.goal.verifier.model = Some(model.clone());
        config.providers.models.custom.insert(
            "verifier".into(),
            CustomModelProviderConfig {
                base: ModelProviderConfig {
                    api_key: Some("test-key".into()),
                    uri: Some(server.uri()),
                    model: Some(model),
                    ..ModelProviderConfig::default()
                },
            },
        );

        let outcome = evaluate_goal_turn(&ctx, &config, "looks done")
            .await
            .unwrap();

        let Some(GoalTurnEvaluation::Paused {
            task_id: paused_id,
            message,
        }) = outcome
        else {
            panic!("unpriced verifier usage under a cost budget must pause");
        };
        assert_eq!(paused_id, task_id);
        assert!(
            message.contains("budget accounting is unavailable"),
            "{message}"
        );
        let task = store.get(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Paused);
        let goal = goal_store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::BudgetUnavailable));
    }

    #[tokio::test]
    async fn goal_start_resolves_config_default_and_explicit_budget_limits() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let ctx = GoalAdmissionContext::new("agent-a").with_channel_type(Some("matrix".into()));
        let mut config = test_config();
        config.goal.token_budget = Some(12_000);
        config.goal.cost_budget_usd = Some(3.25);
        let (token_limit, cost_limit_usd) = resolve_goal_limits(
            &config,
            GoalBudgetOverrides {
                token_limit: GoalBudgetValue::Unlimited,
                cost_limit_usd: GoalBudgetValue::Default,
            },
        );

        let started = start_goal(
            &store,
            "boot-a",
            ctx,
            "ship it".into(),
            token_limit,
            cost_limit_usd,
            None,
        )
        .await
        .unwrap();
        let task_id = started.task_id.unwrap();
        assert!(started.message.contains("Goal `"));
        assert!(started.message.contains("started"));
        assert!(started.message.contains("Objective:"));
        assert!(started.message.contains("ship it"));
        assert!(started.message.contains("Budget: tokens 0/unlimited"));
        assert!(started.message.contains("$0.0000/$3.2500"));
        let goal = store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert_eq!(goal.effective_token_limit, None);
        assert_eq!(goal.effective_cost_limit_usd, Some(3.25));
    }

    #[tokio::test]
    async fn token_limited_goal_starts_when_cost_tracking_is_disabled() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let ctx = GoalAdmissionContext::new("agent-a");
        let config = test_config();

        let started = start_goal(
            &store,
            "boot-a",
            ctx,
            "ship it".into(),
            Some(10),
            None,
            Some(&config),
        )
        .await
        .unwrap();

        assert_eq!(started.status, TaskStatus::Running);
        assert!(started.continue_goal);
        assert!(started.message.contains("cost unavailable"));
        assert!(started.message.contains("Objective:"));
        assert!(started.message.contains("ship it"));
        let task_id = started.task_id.unwrap();
        let task = store.get(&task_id).await.unwrap().unwrap();
        let goal = store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Running);
        assert_eq!(goal.pause_reason, None);
        assert!(goal.blockers.is_empty());
    }

    #[tokio::test]
    async fn unlimited_goal_starts_paused_when_the_canonical_ledger_is_unusable() {
        let _cost_guard = goal_cost_tracker_test_lock().await;
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(temp.path().join("state/costs.jsonl")).unwrap();
        let mut config = test_config();
        config.data_dir = temp.path().to_path_buf();

        let started = start_goal(
            &store,
            "boot-a",
            GoalAdmissionContext::new("agent-a"),
            "ship it".into(),
            None,
            None,
            Some(&config),
        )
        .await
        .unwrap();

        assert_eq!(started.status, TaskStatus::Paused);
        assert!(!started.continue_goal);
        let task_id = started.task_id.unwrap();
        assert_eq!(
            store.get(&task_id).await.unwrap().unwrap().status,
            TaskStatus::Paused
        );
        let goal = store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::BudgetUnavailable));
    }

    #[tokio::test]
    async fn cost_limited_goal_is_rejected_before_task_creation_when_cost_tracking_is_disabled() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let config = test_config();

        let error = start_goal(
            &store,
            "boot-a",
            GoalAdmissionContext::new("agent-a"),
            "ship it".into(),
            None,
            Some(1.0),
            Some(&config),
        )
        .await
        .expect_err("disabled cost tracking must reject a finite cost limit");

        assert!(
            error
                .to_string()
                .contains("finite goal cost budget requires enabled cost tracking")
        );
        assert!(
            store
                .latest_active_goal_for_agent("agent-a")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn goal_recovery_status_message_includes_objective_and_budget() {
        let mut config = test_config();
        config.goal.token_budget = Some(25_000);
        let goal = GoalTaskRecord {
            task_id: "goal-recovered".into(),
            objective: "finish the restart smoke".into(),
            effective_token_limit: Some(12_000),
            effective_cost_limit_usd: None,
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
        };

        let message = goal_recovery_status_message(&goal, Some(&config));

        assert!(message.contains("recovered after service restart"));
        assert!(message.contains("Objective:"));
        assert!(message.contains("finish the restart smoke"));
        assert!(message.contains("Budget:"));
        assert!(message.contains("tokens"));
    }

    #[tokio::test]
    async fn goal_start_persists_restart_continuation_context() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let continuation_context = TaskContinuationContext {
            channel: "matrix".into(),
            channel_alias: Some("work".into()),
            reply_target: "!room:example.org".into(),
            sender: "@operator:example.org".into(),
            thread_ts: Some("$root".into()),
            interruption_scope_id: Some("$root".into()),
            conversation_scope: TaskContinuationConversationScope::ReplyTarget,
        };
        let ctx = GoalAdmissionContext::new("agent-a")
            .with_channel_type(Some("matrix".into()))
            .with_originator_route(Some("matrix_work__room_example_org".into()))
            .with_principal_id(Some("principal-a".into()))
            .with_continuation_context(Some(continuation_context.clone()));

        let started = start_goal(&store, "boot-a", ctx, "ship it".into(), None, None, None)
            .await
            .unwrap();
        let task_id = started.task_id.unwrap();

        assert_eq!(
            store.get_continuation_context(&task_id).await.unwrap(),
            Some(continuation_context)
        );
    }

    #[tokio::test]
    async fn goal_lifecycle_uses_task_record_for_status() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let ctx = GoalAdmissionContext::new("agent-a")
            .with_originator_route(Some("telegram:chat-1".into()))
            .with_principal_id(Some("principal-1".into()));
        let started = start_goal(
            &store,
            "boot-a",
            ctx.clone(),
            "ship it".into(),
            None,
            None,
            None,
        )
        .await
        .unwrap();
        let task_id = started.task_id.clone().unwrap();
        let task = store.get(&task_id).await.unwrap().unwrap();
        let goal = store.get_goal_task(&task_id).await.unwrap().unwrap();

        assert_eq!(task.kind, TaskKind::Goal);
        assert_eq!(task.status, TaskStatus::Running);
        assert_eq!(task.originator_route.as_deref(), Some("telegram:chat-1"));
        assert_eq!(task.principal_id.as_deref(), Some("principal-1"));
        assert_eq!(goal.objective, "ship it");
        assert!(goal.effective_token_limit.is_none());

        let cancelled = cancel_goal(&store, &store, &ctx, Some(task_id.clone()), None)
            .await
            .unwrap();
        assert_eq!(cancelled.status, TaskStatus::Cancelled);
        assert_eq!(
            store.get(&task_id).await.unwrap().unwrap().status,
            TaskStatus::Cancelled
        );

        let err = cancel_goal(&store, &store, &ctx, Some(task_id), None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("already terminal"));
    }

    #[tokio::test]
    async fn goal_objective_updates_canonical_goal_extension_only() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let ctx = GoalAdmissionContext::new("agent-a")
            .with_originator_route(Some("matrix:room-1".into()))
            .with_principal_id(Some("principal-1".into()));
        let started = start_goal(
            &store,
            "boot-a",
            ctx.clone(),
            "ship initial scope".into(),
            None,
            None,
            None,
        )
        .await
        .unwrap();
        let task_id = started.task_id.clone().unwrap();

        let amended = update_goal_objective(
            &store,
            &store,
            &ctx,
            "ship amended scope after evidence".into(),
            Some(&test_config()),
        )
        .await
        .unwrap();

        assert_eq!(amended.task_id.as_deref(), Some(task_id.as_str()));
        assert_eq!(amended.status, TaskStatus::Running);
        assert!(
            !amended.continue_goal,
            "objective edits must not synthesize a second model turn"
        );
        assert!(amended.message.contains("objective updated"));
        let task = store.get(&task_id).await.unwrap().unwrap();
        let goal = store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Running);
        assert_eq!(goal.objective, "ship amended scope after evidence");
        assert!(goal.effective_token_limit.is_none());
        assert!(goal.effective_cost_limit_usd.is_none());
    }

    #[tokio::test]
    async fn goal_visibility_enforces_route_and_principal() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let owner = GoalAdmissionContext::new("agent-a")
            .with_originator_route(Some("telegram:chat-1".into()))
            .with_principal_id(Some("principal-1".into()));
        let started = start_goal(
            &store,
            "boot-a",
            owner.clone(),
            "ship it".into(),
            None,
            None,
            None,
        )
        .await
        .unwrap();
        let task_id = started.task_id.clone().unwrap();

        status_goal(&store, &store, &owner, Some(task_id.clone()), None)
            .await
            .unwrap();

        let wrong_route = GoalAdmissionContext::new("agent-a")
            .with_originator_route(Some("telegram:chat-2".into()))
            .with_principal_id(Some("principal-1".into()));
        let err = status_goal(&store, &store, &wrong_route, Some(task_id.clone()), None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not visible from this route"));

        let wrong_principal = GoalAdmissionContext::new("agent-a")
            .with_originator_route(Some("telegram:chat-1".into()))
            .with_principal_id(Some("principal-2".into()));
        let err = status_goal(&store, &store, &wrong_principal, Some(task_id), None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not visible to this principal"));
    }

    #[tokio::test]
    async fn exact_legacy_goal_context_rebinds_before_pause_and_cancel() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let task_id = "legacy-route-goal";
        let continuation = TaskContinuationContext {
            channel: "matrix".into(),
            channel_alias: Some("main".into()),
            reply_target: "!room:example".into(),
            sender: "@alice:example".into(),
            thread_ts: None,
            interruption_scope_id: None,
            conversation_scope: TaskContinuationConversationScope::ReplyTarget,
        };
        let legacy_route = "matrix_main__room_example";
        let legacy_principal = "matrix_main__alice_example";
        let canonical_route =
            "6:matrix|4:main|14:@alice:example|13:!room:example|0:|12:reply_target";
        let canonical_principal = "6:matrix|4:main|14:@alice:example";
        store
            .create_goal(
                TaskRecord {
                    id: task_id.into(),
                    kind: TaskKind::Goal,
                    agent: "agent-a".into(),
                    status: TaskStatus::Running,
                    owner_pid: 1,
                    owner_boot_id: "boot-a".into(),
                    heartbeat_at: None,
                    depth: 0,
                    parent_id: None,
                    originator_route: Some(legacy_route.into()),
                    delivered: false,
                    idem_key: None,
                    principal_id: Some(legacy_principal.into()),
                    started_at: chrono::Utc::now().to_rfc3339(),
                    finished_at: None,
                },
                GoalTaskRecord {
                    task_id: task_id.into(),
                    objective: "retain existing goal controls after upgrade".into(),
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                },
                Some(continuation.clone()),
            )
            .await
            .unwrap();
        let ctx = GoalAdmissionContext::new("agent-a")
            .with_originator_route(Some(canonical_route.into()))
            .with_principal_id(Some(canonical_principal.into()))
            .with_legacy_identity(Some(legacy_route.into()), Some(legacy_principal.into()))
            .with_goal_task_id(Some(task_id.into()))
            .with_continuation_context(Some(continuation));

        let paused = pause_goal_for_blocker(
            &store,
            &store,
            &ctx,
            Some(task_id.into()),
            Some(&test_config()),
            GoalPauseState {
                reason: GoalPauseReason::OperatorPaused,
                description: Some("operator requested pause".into()),
                blockers: Vec::new(),
            },
        )
        .await
        .unwrap();
        assert_eq!(paused.status, TaskStatus::Paused);

        let rebound = store.get(task_id).await.unwrap().unwrap();
        assert_eq!(rebound.originator_route.as_deref(), Some(canonical_route));
        assert_eq!(rebound.principal_id.as_deref(), Some(canonical_principal));

        let cancelled = cancel_goal(
            &store,
            &store,
            &ctx,
            Some(task_id.into()),
            Some(&test_config()),
        )
        .await
        .unwrap();
        assert_eq!(cancelled.status, TaskStatus::Cancelled);
    }

    #[tokio::test]
    async fn legacy_identity_never_rebinds_without_exact_durable_continuation() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let task_id = "legacy-route-mismatch";
        let stored_context = TaskContinuationContext {
            channel: "matrix".into(),
            channel_alias: None,
            reply_target: "!room:example".into(),
            sender: "@alice:example".into(),
            thread_ts: None,
            interruption_scope_id: None,
            conversation_scope: TaskContinuationConversationScope::ReplyTarget,
        };
        store
            .create_goal(
                TaskRecord {
                    id: task_id.into(),
                    kind: TaskKind::Goal,
                    agent: "agent-a".into(),
                    status: TaskStatus::Running,
                    owner_pid: 1,
                    owner_boot_id: "boot-a".into(),
                    heartbeat_at: None,
                    depth: 0,
                    parent_id: None,
                    originator_route: Some("legacy-route".into()),
                    delivered: false,
                    idem_key: None,
                    principal_id: Some("legacy-principal".into()),
                    started_at: chrono::Utc::now().to_rfc3339(),
                    finished_at: None,
                },
                GoalTaskRecord {
                    task_id: task_id.into(),
                    objective: "must not be claimed by a collision".into(),
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                },
                Some(stored_context),
            )
            .await
            .unwrap();
        let mismatched_context = TaskContinuationContext {
            channel: "matrix".into(),
            channel_alias: None,
            reply_target: "!room:example".into(),
            sender: "@mallory:example".into(),
            thread_ts: None,
            interruption_scope_id: None,
            conversation_scope: TaskContinuationConversationScope::ReplyTarget,
        };
        let ctx = GoalAdmissionContext::new("agent-a")
            .with_originator_route(Some("canonical-route".into()))
            .with_principal_id(Some("canonical-principal".into()))
            .with_legacy_identity(Some("legacy-route".into()), Some("legacy-principal".into()))
            .with_goal_task_id(Some(task_id.into()))
            .with_continuation_context(Some(mismatched_context));

        let err = status_goal(&store, &store, &ctx, Some(task_id.into()), None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not visible from this route"));
        let unchanged = store.get(task_id).await.unwrap().unwrap();
        assert_eq!(unchanged.originator_route.as_deref(), Some("legacy-route"));
        assert_eq!(unchanged.principal_id.as_deref(), Some("legacy-principal"));
    }

    #[tokio::test]
    async fn goal_status_reports_recovered_daemon_restart_pause() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let route = "matrix:room-1".to_string();
        let principal = "principal-1".to_string();
        store
            .create_goal(
                TaskRecord {
                    id: "goal-recovered-paused".into(),
                    kind: TaskKind::Goal,
                    agent: "agent-a".into(),
                    status: TaskStatus::Running,
                    owner_pid: 999_999,
                    owner_boot_id: "boot-old".into(),
                    heartbeat_at: None,
                    depth: 0,
                    parent_id: None,
                    originator_route: Some(route.clone()),
                    delivered: false,
                    idem_key: None,
                    principal_id: Some(principal.clone()),
                    started_at: chrono::Utc::now().to_rfc3339(),
                    finished_at: None,
                },
                GoalTaskRecord {
                    task_id: "goal-recovered-paused".into(),
                    objective: "finish restart validation".into(),
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                },
                None,
            )
            .await
            .unwrap();

        let report = crate::control_plane::reaper::recovery_pass(
            &store,
            &store,
            "boot-new",
            zeroclaw_config::schema::GoalRestartRecovery::Paused,
        )
        .await
        .unwrap();
        assert_eq!(report.recovered, 1);

        let ctx = GoalAdmissionContext::new("agent-a")
            .with_originator_route(Some(route))
            .with_principal_id(Some(principal));
        let status = status_goal(
            &store,
            &store,
            &ctx,
            Some("goal-recovered-paused".into()),
            Some(&test_config()),
        )
        .await
        .unwrap();

        assert_eq!(status.status, TaskStatus::Paused);
        assert!(!status.continue_goal);
        assert!(status.message.contains("daemon restarted"));
        assert!(!status.message.contains("daemon_restarted"));
        assert!(status.message.contains("restart recovery"));
        assert!(status.message.contains("finish restart validation"));
    }

    #[tokio::test]
    async fn goal_status_does_not_initialize_cost_tracker_storage() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let ctx = GoalAdmissionContext::new("agent-a");
        let started = start_goal(
            &store,
            "boot-a",
            ctx.clone(),
            "inspect status only".into(),
            Some(10_000),
            None,
            None,
        )
        .await
        .unwrap();
        let task_id = started.task_id.clone().unwrap();
        let tmp = tempfile::TempDir::new().unwrap();
        let mut config = test_config();
        config.data_dir = tmp.path().to_path_buf();
        config.cost.enabled = true;

        let status = status_goal(&store, &store, &ctx, Some(task_id), Some(&config))
            .await
            .unwrap();

        assert_eq!(status.status, TaskStatus::Running);
        assert!(
            status.message.contains("usage unavailable"),
            "status should report unavailable usage instead of initializing the ledger"
        );
        assert!(
            !tmp.path().join("state").join("costs.jsonl").exists(),
            "read-only status must not create cost tracker storage"
        );
    }

    #[test]
    fn read_only_goal_usage_keeps_disabled_cost_token_totals_visible() {
        let tmp = tempfile::TempDir::new().unwrap();
        let tracker = CostTracker::new(
            zeroclaw_config::schema::CostConfig {
                enabled: false,
                ..Default::default()
            },
            tmp.path(),
        )
        .unwrap();
        tracker
            .record_scoped_usage_with_owned_task_attribution(
                zeroclaw_config::cost::TokenUsage::new("test/model", 1_000, 500, 0, 0.0, 0.0, 0.0),
                Some("agent-a"),
                Some("goal-a".into()),
            )
            .unwrap();

        let usage = goal_usage_totals_from_tracker(Some(&tracker), "goal-a", false)
            .expect("read-only status should see the disabled-cost goal ledger");

        assert_eq!(usage.total_tokens, 1_500);
        assert!(usage.usage_available);
        assert!(!usage.cost_tracking_available);
    }

    #[tokio::test]
    async fn goal_start_rejects_duplicate_active_context() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let ctx = GoalAdmissionContext::new("agent-a")
            .with_originator_route(Some("telegram:chat-1".into()))
            .with_principal_id(Some("principal-1".into()));

        start_goal(
            &store,
            "boot-a",
            ctx.clone(),
            "ship it".into(),
            None,
            None,
            None,
        )
        .await
        .unwrap();
        let err = start_goal(
            &store,
            "boot-a",
            ctx,
            "ship another".into(),
            None,
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("already active"));
    }

    #[tokio::test]
    async fn concurrent_goal_start_allows_one_active_context() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let ctx = GoalAdmissionContext::new("agent-a")
            .with_originator_route(Some("telegram:chat-1".into()))
            .with_principal_id(Some("principal-1".into()));

        let (a, b) = tokio::join!(
            start_goal(
                &store,
                "boot-a",
                ctx.clone(),
                "ship one".into(),
                None,
                None,
                None,
            ),
            start_goal(&store, "boot-a", ctx, "ship two".into(), None, None, None,)
        );
        let successes = usize::from(a.is_ok()) + usize::from(b.is_ok());
        assert_eq!(successes, 1);
        let errors = [a.err(), b.err()]
            .into_iter()
            .flatten()
            .map(|error| error.to_string())
            .collect::<Vec<_>>();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("already active"));
    }

    #[tokio::test]
    async fn pause_and_resume_store_goal_specific_blockers_only() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let ctx = GoalAdmissionContext::new("agent-a");
        let started = start_goal(
            &store,
            "boot-a",
            ctx.clone(),
            "ship it".into(),
            None,
            None,
            None,
        )
        .await
        .unwrap();
        let task_id = started.task_id.clone().unwrap();

        let paused = pause_goal_for_blocker(
            &store,
            &store,
            &ctx,
            Some(task_id.clone()),
            None,
            GoalPauseState {
                reason: GoalPauseReason::NeedsUserInput,
                description: Some("need answer".into()),
                blockers: vec![GoalBlocker {
                    kind: GoalBlockerKind::NeedsUserInput,
                    message: "Need operator answer".into(),
                    payload: Some(serde_json::json!({"prompt": "continue?"})),
                }],
            },
        )
        .await
        .unwrap();
        assert_eq!(paused.status, TaskStatus::Paused);
        let task = store.get(&task_id).await.unwrap().unwrap();
        let goal = store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Paused);
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::NeedsUserInput));
        assert_eq!(goal.blockers.len(), 1);

        let resumed = resume_goal(&store, &store, "boot-resumed", &ctx, None, None)
            .await
            .unwrap();
        assert_eq!(resumed.status, TaskStatus::Running);
        let task = store.get(&task_id).await.unwrap().unwrap();
        let goal = store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Running);
        assert_eq!(task.owner_boot_id, "boot-resumed");
        assert!(goal.pause_reason.is_none());
        assert!(goal.blockers.is_empty());
    }

    #[tokio::test]
    async fn resume_reason_survives_as_transient_continuation_input_only() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let ctx = GoalAdmissionContext::new("agent-a");
        let started = start_goal(
            &store,
            "boot-a",
            ctx.clone(),
            "ship it".into(),
            None,
            None,
            None,
        )
        .await
        .unwrap();
        let task_id = started.task_id.clone().unwrap();

        pause_goal_for_blocker(
            &store,
            &store,
            &ctx,
            Some(task_id.clone()),
            None,
            GoalPauseState {
                reason: GoalPauseReason::NeedsUserInput,
                description: Some("need answer".into()),
                blockers: vec![GoalBlocker {
                    kind: GoalBlockerKind::NeedsUserInput,
                    message: "Need operator answer".into(),
                    payload: Some(serde_json::json!({"prompt": "continue?"})),
                }],
            },
        )
        .await
        .unwrap();

        let resumed = resume_goal(
            &store,
            &store,
            "boot-resumed",
            &ctx,
            Some("operator confirmed the external deploy is healthy".into()),
            None,
        )
        .await
        .unwrap();

        assert!(resumed.continue_goal);
        assert_eq!(
            resumed.continuation_reason.as_deref(),
            Some("operator confirmed the external deploy is healthy")
        );
        let goal = store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert!(goal.pause_reason.is_none());
        assert!(goal.pause_description.is_none());
        assert!(goal.blockers.is_empty());
    }

    #[tokio::test]
    async fn human_gate_pause_uses_scoped_active_goal() {
        let (store, goal_store) = global_test_stores();
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let ctx = GoalAdmissionContext::new(agent)
            .with_originator_route(Some(route))
            .with_principal_id(Some(principal));
        let started = start_goal(
            goal_store.as_ref(),
            "boot-a",
            ctx.clone(),
            "wait for operator input".into(),
            None,
            None,
            None,
        )
        .await
        .unwrap();
        let task_id = started.task_id.unwrap();
        let ctx = ctx.with_goal_task_id(Some(task_id.clone()));

        let admission = pause_current_goal_for_human_gate(
            &ctx,
            None,
            GoalBlockerKind::NeedsUserInput,
            "Need operator answer".into(),
            Some(serde_json::json!({"tool": "ask_user", "question": "continue?"})),
        )
        .await
        .unwrap();

        assert_eq!(admission.task_id.as_deref(), Some(task_id.as_str()));
        assert_eq!(admission.status, TaskStatus::Paused);
        let task = store.get(&task_id).await.unwrap().unwrap();
        let goal = goal_store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Paused);
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::NeedsUserInput));
        assert_eq!(goal.blockers[0].kind, GoalBlockerKind::NeedsUserInput);
        assert_eq!(
            goal.blockers[0].payload.as_ref().unwrap()["tool"],
            "ask_user"
        );
    }

    #[tokio::test]
    async fn budget_update_resumes_goal_when_budget_blocker_clears() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let continuation_context = TaskContinuationContext {
            channel: "matrix".into(),
            channel_alias: Some("work".into()),
            reply_target: "!room:example.org".into(),
            sender: "@operator:example.org".into(),
            thread_ts: Some("$root".into()),
            interruption_scope_id: Some("$root".into()),
            conversation_scope: TaskContinuationConversationScope::ReplyTarget,
        };
        let ctx = GoalAdmissionContext::new("agent-a")
            .with_originator_route(Some("matrix:room".into()))
            .with_principal_id(Some("principal-a".into()))
            .with_continuation_context(Some(continuation_context.clone()));
        let started = start_goal(
            &store,
            "boot-a",
            ctx.clone(),
            "ship it".into(),
            Some(10),
            None,
            None,
        )
        .await
        .unwrap();
        let task_id = started.task_id.clone().unwrap();
        pause_goal_for_blocker(
            &store,
            &store,
            &ctx,
            Some(task_id.clone()),
            None,
            GoalPauseState {
                reason: GoalPauseReason::BudgetExhausted,
                description: Some("budget exhausted".into()),
                blockers: vec![GoalBlocker {
                    kind: GoalBlockerKind::Budget,
                    message: "Budget exhausted".into(),
                    payload: None,
                }],
            },
        )
        .await
        .unwrap();

        let admitted = update_goal_budget(
            &store,
            &store,
            "boot-resumed",
            &ctx,
            GoalBudgetOverrides {
                token_limit: GoalBudgetValue::Unlimited,
                cost_limit_usd: GoalBudgetValue::Default,
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(admitted.status, TaskStatus::Running);
        assert!(admitted.continue_goal);
        let task = store.get(&task_id).await.unwrap().unwrap();
        let goal = store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Running);
        assert_eq!(task.owner_boot_id, "boot-resumed");
        assert!(goal.effective_token_limit.is_none());
        assert!(goal.pause_reason.is_none());
        assert!(goal.blockers.is_empty());
        assert_eq!(
            store.get_continuation_context(&task_id).await.unwrap(),
            Some(continuation_context)
        );
    }

    #[tokio::test]
    async fn budget_update_does_not_resume_when_the_canonical_ledger_is_unusable() {
        let _cost_guard = goal_cost_tracker_test_lock().await;
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let mut config = test_config();
        config.data_dir = temp.path().to_path_buf();
        let ctx = GoalAdmissionContext::new("agent-a");
        let started = start_goal(
            &store,
            "boot-a",
            ctx.clone(),
            "ship it".into(),
            Some(10),
            None,
            Some(&config),
        )
        .await
        .unwrap();
        let task_id = started.task_id.unwrap();
        pause_goal_for_blocker(
            &store,
            &store,
            &ctx,
            Some(task_id.clone()),
            Some(&config),
            GoalPauseState {
                reason: GoalPauseReason::BudgetUnavailable,
                description: Some("ledger unavailable".into()),
                blockers: vec![GoalBlocker {
                    kind: GoalBlockerKind::Budget,
                    message: "Ledger unavailable".into(),
                    payload: None,
                }],
            },
        )
        .await
        .unwrap();
        std::fs::remove_file(temp.path().join("state/costs.jsonl")).unwrap();
        std::fs::create_dir_all(temp.path().join("state/costs.jsonl")).unwrap();

        let admitted = update_goal_budget(
            &store,
            &store,
            "boot-resumed",
            &ctx,
            GoalBudgetOverrides {
                token_limit: GoalBudgetValue::Unlimited,
                cost_limit_usd: GoalBudgetValue::Unlimited,
            },
            Some(&config),
        )
        .await
        .unwrap();

        assert_eq!(admitted.status, TaskStatus::Paused);
        assert!(!admitted.continue_goal);
        assert_eq!(
            store.get(&task_id).await.unwrap().unwrap().status,
            TaskStatus::Paused
        );
        let goal = store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::BudgetUnavailable));
    }

    #[tokio::test]
    async fn budget_update_keeps_goal_paused_when_new_token_limit_is_still_exhausted() {
        let _cost_guard = goal_cost_tracker_test_lock().await;
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let task_id = format!("goal-{}", uuid::Uuid::new_v4());
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let ctx = GoalAdmissionContext::new(agent.clone())
            .with_originator_route(Some(format!("route-{}", uuid::Uuid::new_v4())))
            .with_principal_id(Some(format!("principal-{}", uuid::Uuid::new_v4())));
        let tmp = tempfile::TempDir::new().unwrap();
        let config = cost_enabled_test_config(tmp.path());

        // User-visible policy: lowering an exhausted 100k-token goal to 80k is
        // only a limit update. It must not clear the budget blocker or spend
        // another autonomous turn.
        create_budget_paused_goal(&store, &ctx, &task_id, 100_000, None).await;
        record_goal_token_usage(&config, &agent, &task_id, 100_000);

        let admitted = update_goal_budget(
            &store,
            &store,
            "boot-budget-update",
            &ctx,
            GoalBudgetOverrides {
                token_limit: GoalBudgetValue::Limited(80_000),
                cost_limit_usd: GoalBudgetValue::Default,
            },
            Some(&config),
        )
        .await
        .unwrap();

        assert_eq!(admitted.status, TaskStatus::Paused);
        assert!(!admitted.continue_goal);
        assert!(admitted.message.contains("budget updated; goal is paused"));
        assert!(admitted.message.contains("tokens 100000/80000"));
        let task = store.get(&task_id).await.unwrap().unwrap();
        let goal = store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Paused);
        assert_eq!(goal.effective_token_limit, Some(80_000));
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::BudgetExhausted));
        assert_eq!(goal.blockers.len(), 1);
        assert_eq!(goal.blockers[0].kind, GoalBlockerKind::Budget);
    }

    #[tokio::test]
    async fn budget_update_resumes_goal_when_new_token_limit_clears_exhaustion() {
        let _cost_guard = goal_cost_tracker_test_lock().await;
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let task_id = format!("goal-{}", uuid::Uuid::new_v4());
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let continuation_context = TaskContinuationContext {
            channel: "matrix".into(),
            channel_alias: Some("work".into()),
            reply_target: "!room:example.org".into(),
            sender: "@operator:example.org".into(),
            thread_ts: Some("$root".into()),
            interruption_scope_id: Some("$root".into()),
            conversation_scope: TaskContinuationConversationScope::ReplyTarget,
        };
        let ctx = GoalAdmissionContext::new(agent.clone())
            .with_originator_route(Some(format!("route-{}", uuid::Uuid::new_v4())))
            .with_principal_id(Some(format!("principal-{}", uuid::Uuid::new_v4())))
            .with_continuation_context(Some(continuation_context.clone()));
        let tmp = tempfile::TempDir::new().unwrap();
        let config = cost_enabled_test_config(tmp.path());

        // User-visible policy: raising an exhausted 100k-token goal to 120k
        // clears a pure budget pause and re-enters the trusted continuation
        // path instead of requiring a separate `/goal resume`.
        create_budget_paused_goal(
            &store,
            &ctx,
            &task_id,
            100_000,
            Some(continuation_context.clone()),
        )
        .await;
        record_goal_token_usage(&config, &agent, &task_id, 100_000);

        let admitted = update_goal_budget(
            &store,
            &store,
            "boot-budget-update",
            &ctx,
            GoalBudgetOverrides {
                token_limit: GoalBudgetValue::Limited(120_000),
                cost_limit_usd: GoalBudgetValue::Default,
            },
            Some(&config),
        )
        .await
        .unwrap();

        assert_eq!(admitted.status, TaskStatus::Running);
        assert!(admitted.continue_goal);
        assert!(admitted.message.contains("budget updated and resumed"));
        assert!(admitted.message.contains("tokens 100000/120000"));
        let task = store.get(&task_id).await.unwrap().unwrap();
        let goal = store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Running);
        assert_eq!(task.owner_boot_id, "boot-budget-update");
        assert_eq!(goal.effective_token_limit, Some(120_000));
        assert!(goal.pause_reason.is_none());
        assert!(goal.blockers.is_empty());
        assert_eq!(
            store.get_continuation_context(&task_id).await.unwrap(),
            Some(continuation_context)
        );
    }

    #[tokio::test]
    async fn budget_update_reports_remaining_non_budget_blockers() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let ctx = GoalAdmissionContext::new("agent-a");
        let started = start_goal(
            &store,
            "boot-a",
            ctx.clone(),
            "ship it".into(),
            Some(10),
            None,
            None,
        )
        .await
        .unwrap();
        let task_id = started.task_id.clone().unwrap();
        pause_goal_for_blocker(
            &store,
            &store,
            &ctx,
            Some(task_id.clone()),
            None,
            GoalPauseState {
                reason: GoalPauseReason::BudgetExhausted,
                description: Some("multiple blockers".into()),
                blockers: vec![
                    GoalBlocker {
                        kind: GoalBlockerKind::NeedsUserInput,
                        message: "Need operator answer".into(),
                        payload: None,
                    },
                    GoalBlocker {
                        kind: GoalBlockerKind::Budget,
                        message: "Budget exhausted".into(),
                        payload: None,
                    },
                ],
            },
        )
        .await
        .unwrap();

        let admitted = update_goal_budget(
            &store,
            &store,
            "boot-resumed",
            &ctx,
            GoalBudgetOverrides {
                token_limit: GoalBudgetValue::Unlimited,
                cost_limit_usd: GoalBudgetValue::Default,
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(admitted.status, TaskStatus::Paused);
        assert!(!admitted.continue_goal);
        assert!(admitted.message.contains("still paused"));
        assert!(
            admitted
                .message
                .contains("user input: Need operator answer")
        );
        let task = store.get(&task_id).await.unwrap().unwrap();
        let goal = store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Paused);
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::NeedsUserInput));
        assert_eq!(goal.blockers.len(), 1);
        assert_eq!(goal.blockers[0].kind, GoalBlockerKind::NeedsUserInput);
    }
}
