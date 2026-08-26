//! Durable accounting owner for Goal-scoped logical provider operations.
//!
//! This module is deliberately transport-neutral: it receives the normal
//! runtime turn through a private task-local seam and owns only Goal admission,
//! ledger persistence, and exact pending-operation settlement.

use std::{collections::HashMap, future::Future, sync::Arc};

use anyhow::Context as _;
use async_trait::async_trait;
use serde::Deserialize;
use tokio::{
    sync::{Mutex, OwnedSemaphorePermit},
    task::JoinHandle,
};
use uuid::Uuid;
use zeroclaw_config::{
    cost::{CostTracker, StrictTaskUsageTotals},
    schema::Config,
};

use super::{
    GoalExecutionPreflight, GoalExecutionReadiness, GoalExecutionScope, GoalExecutor,
    host::{GoalExecutionHost, GoalHostPreflightError, GoalParentTurn, GoalVerifierTurn},
};
use crate::{
    agent::cost::{
        GoalOperationAccounting, GoalOperationRequest, GoalOperationSettlement, GoalUsageEvent,
    },
    control_plane::{
        GoalAccountingState, GoalBlocker, GoalBlockerKind, GoalPauseReason, GoalPauseState,
        GoalTaskRecord, GoalTaskRegistry, GoalTransitionResult, TaskRecord, TaskStatus,
    },
};

const VERIFIER_REASON_MAX_CHARS: usize = 2_048;
const VERIFIER_RESPONSE_MAX_BYTES: usize = 65_536;
const VERIFIER_BLOCKERS_MAX: usize = 16;
const VERIFIER_BLOCKER_MESSAGE_MAX_CHARS: usize = 1_024;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum VerifierDecision {
    Complete,
    Continue,
    Blocked,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VerifierResponse {
    decision: VerifierDecision,
    reason: String,
    #[serde(default)]
    blockers: Option<Vec<VerifierBlocker>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VerifierBlocker {
    kind: GoalBlockerKind,
    message: String,
}

enum VerifierOutcome {
    Complete,
    Continue(String),
    Blocked(String, Vec<GoalBlocker>),
}

#[derive(Debug)]
struct GoalBudgetExhausted;

impl std::fmt::Display for GoalBudgetExhausted {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("goal budget is exhausted")
    }
}

impl std::error::Error for GoalBudgetExhausted {}

#[derive(Debug)]
struct GoalDeadlineExceeded;

impl std::fmt::Display for GoalDeadlineExceeded {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("goal supervisory deadline elapsed")
    }
}

impl std::error::Error for GoalDeadlineExceeded {}

#[derive(Debug)]
enum GoalAccountingFailure {
    MissingOrInvalid,
    OutcomeUnknown,
    PricingUnavailable,
}

impl std::fmt::Display for GoalAccountingFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::MissingOrInvalid => "goal accounting is missing or invalid",
            Self::OutcomeUnknown => "goal accounting outcome is unknown",
            Self::PricingUnavailable => "goal pricing is unavailable",
        })
    }
}

impl std::error::Error for GoalAccountingFailure {}

async fn run_with_goal_deadline<T>(
    scope: &GoalExecutionScope,
    operation: impl Future<Output = anyhow::Result<T>>,
) -> anyhow::Result<T> {
    let Some(deadline) = scope.deadline else {
        return operation.await;
    };
    match tokio::time::timeout_at(deadline, operation).await {
        Ok(result) => result,
        Err(_) => Err(GoalDeadlineExceeded.into()),
    }
}

struct ActiveGoalOperation {
    call_id: String,
    cost_limit_required: bool,
    _permit: OwnedSemaphorePermit,
}

/// Private accountant installed by a Goal execution host around an ordinary
/// configured turn. It cannot construct or select a provider.
pub(crate) struct GoalOperationAccountant {
    goals: Arc<dyn GoalTaskRegistry>,
    config: Arc<Config>,
    cost_tracker: Arc<CostTracker>,
    scope: GoalExecutionScope,
    agent_alias: String,
    active: Mutex<Option<ActiveGoalOperation>>,
}

impl GoalOperationAccountant {
    pub(crate) fn new(
        goals: Arc<dyn GoalTaskRegistry>,
        config: Arc<Config>,
        cost_tracker: Arc<CostTracker>,
        scope: GoalExecutionScope,
        agent_alias: String,
    ) -> Self {
        Self {
            goals,
            config,
            cost_tracker,
            scope,
            agent_alias,
            active: Mutex::new(None),
        }
    }

    async fn current_goal(&self) -> anyhow::Result<(TaskRecord, GoalTaskRecord)> {
        let task = self
            .goals
            .current_goal_for_session(&self.scope.session_id)
            .await?
            .context("goal current task is missing")?;
        let goal = self
            .goals
            .get_goal_task(&task.id)
            .await?
            .context("goal extension is missing")?;
        if task.id != self.scope.task_id
            || task.execution_epoch != self.scope.execution_epoch
            || task.status != TaskStatus::Running
        {
            anyhow::bail!("goal execution scope is stale");
        }
        Ok((task, goal))
    }

    async fn strict_totals(&self) -> anyhow::Result<StrictTaskUsageTotals> {
        let tracker = Arc::clone(&self.cost_tracker);
        let task_id = self.scope.task_id.clone();
        tokio::task::spawn_blocking(move || tracker.get_strict_usage_totals_for_task(&task_id))
            .await
            .context("goal ledger reader task failed")?
    }

    fn request_input_token_estimate(request: &GoalOperationRequest) -> anyhow::Result<u64> {
        let messages = serde_json::to_vec(&request.messages)
            .context("serialize Goal messages for reservation")?;
        let tools =
            serde_json::to_vec(&request.tools).context("serialize Goal tools for reservation")?;
        u64::try_from(
            messages
                .len()
                .checked_add(tools.len())
                .context("Goal operation reservation input overflows")?,
        )
        .context("Goal operation reservation input overflows")
    }

    fn reservation(
        &self,
        task: &TaskRecord,
        goal: &GoalTaskRecord,
        request: &GoalOperationRequest,
    ) -> anyhow::Result<(Option<u64>, Option<f64>)> {
        if goal.effective_token_limit.is_none() && goal.effective_cost_limit_usd.is_none() {
            return Ok((None, None));
        }
        let options =
            zeroclaw_providers::provider_runtime_options_for_agent(&self.config, &task.agent);
        let plan = zeroclaw_providers::resolve_routed_model_operation_plan_with_options(
            &self.config,
            &request.model_provider,
            &self.config.reliability,
            &self.config.model_routes,
            &request.model,
            &options,
            &request.model,
        )?;
        let input_tokens = Self::request_input_token_estimate(request)?;
        let reserved_tokens = goal
            .effective_token_limit
            .map(|_| {
                plan.conservative_token_reservation_for_request(input_tokens, request.thinking)
            })
            .transpose()?;
        let reserved_cost_usd = goal
            .effective_cost_limit_usd
            .map(|_| conservative_cost_reservation(&self.config, &plan, input_tokens))
            .transpose()?;
        Ok((reserved_tokens, reserved_cost_usd))
    }

    async fn record_settlement(
        &self,
        settlement: GoalOperationSettlement,
    ) -> anyhow::Result<(GoalAccountingState, bool)> {
        let GoalOperationSettlement {
            accounting_state,
            pricing_complete: reported_pricing_complete,
            usage_events,
        } = settlement;
        let pricing_complete = reported_pricing_complete
            && usage_events
                .iter()
                .all(|event| event.usage.pricing_available);
        let mut state =
            if accounting_state == GoalAccountingState::Complete && usage_events.is_empty() {
                GoalAccountingState::Missing
            } else {
                accounting_state
            };
        for event in usage_events {
            if !valid_usage_event(&event) {
                state = combine_accounting_state(state, GoalAccountingState::Invalid);
                continue;
            }
            let tracker = Arc::clone(&self.cost_tracker);
            let agent_alias = self.agent_alias.clone();
            let task_id = self.scope.task_id.clone();
            let provider_ref = event.model_provider;
            tokio::task::spawn_blocking(move || {
                tracker.record_scoped_usage_with_owned_task_and_provider_attribution(
                    event.usage,
                    Some(&agent_alias),
                    Some(task_id),
                    provider_ref,
                )
            })
            .await
            .context("goal ledger writer task failed")??;
        }
        Ok((state, pricing_complete))
    }

    /// Resolve an admitted operation whose normal completion future was
    /// cancelled or dropped. A deadline must not convert an in-flight remote
    /// operation into a cleanly settled terminal Goal merely because the
    /// supervisor stopped awaiting it.
    async fn classify_unsettled_operation(&self) -> bool {
        let Some(call_id) = self
            .active
            .lock()
            .await
            .as_ref()
            .map(|operation| operation.call_id.clone())
        else {
            return false;
        };
        loop {
            match self
                .goals
                .settle_pending_call(
                    &self.scope.task_id,
                    &self.scope.session_id,
                    self.scope.execution_epoch,
                    &call_id,
                    GoalAccountingState::OutcomeUnknown,
                )
                .await
            {
                Ok(GoalTransitionResult::Applied) => {
                    self.active.lock().await.take();
                    return true;
                }
                Ok(GoalTransitionResult::Stale | GoalTransitionResult::Missing) => {
                    match self
                        .goals
                        .current_goal_for_session(&self.scope.session_id)
                        .await
                    {
                        Ok(Some(task))
                            if task.id == self.scope.task_id
                                && task.execution_epoch == self.scope.execution_epoch
                                && task.status == TaskStatus::Running =>
                        {
                            match self.goals.get_goal_task(&task.id).await {
                                Ok(Some(goal))
                                    if goal.pending_call_id.is_none()
                                        && goal.pending_call_epoch.is_none() =>
                                {
                                    // The normal settlement committed before
                                    // its future was dropped.
                                    self.active.lock().await.take();
                                    return false;
                                }
                                Ok(_) | Err(_) => {}
                            }
                        }
                        Ok(Some(_)) | Ok(None) => {
                            self.active.lock().await.take();
                            return false;
                        }
                        Err(_) => {}
                    }
                }
                Err(_) => tokio::time::sleep(std::time::Duration::from_millis(50)).await,
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }
}

#[async_trait]
impl GoalOperationAccounting for GoalOperationAccountant {
    fn config(&self) -> &Config {
        &self.config
    }

    async fn admit(&self, request: GoalOperationRequest) -> anyhow::Result<()> {
        let mut active = self.active.lock().await;
        if active.is_some() {
            anyhow::bail!("goal logical operation is already active");
        }
        let permit = Arc::clone(&self.scope.operation_permit)
            .acquire_owned()
            .await
            .context("goal logical-operation permit is closed")?;
        let (task, goal) = self.current_goal().await?;
        if goal.accounting_state != GoalAccountingState::Complete
            || goal.pending_call_id.is_some()
            || goal.pending_call_epoch.is_some()
        {
            anyhow::bail!("goal accounting is not settled");
        }
        let totals = self.strict_totals().await?;
        if goal.effective_cost_limit_usd.is_some() && !totals.pricing_available {
            return Err(GoalAccountingFailure::PricingUnavailable.into());
        }
        let (reserved_tokens, reserved_cost_usd) = self.reservation(&task, &goal, &request)?;
        if let Some(limit) = goal.effective_token_limit {
            let reserved = reserved_tokens.context("goal token reservation is missing")?;
            let projected = totals
                .total_tokens
                .checked_add(reserved)
                .context("goal token reservation overflows")?;
            if projected > limit {
                return Err(GoalBudgetExhausted.into());
            }
        }
        if let Some(limit) = goal.effective_cost_limit_usd {
            let reserved = reserved_cost_usd.context("goal cost reservation is missing")?;
            let projected = totals.cost_usd + reserved;
            if !projected.is_finite() {
                anyhow::bail!("goal cost reservation overflows");
            }
            if projected > limit {
                return Err(GoalBudgetExhausted.into());
            }
        }
        let call_id = Uuid::new_v4().to_string();
        match self
            .goals
            .admit_pending_call_with_reservation(
                &self.scope.task_id,
                &self.scope.session_id,
                self.scope.execution_epoch,
                goal.effective_token_limit,
                goal.effective_cost_limit_usd,
                &call_id,
                reserved_tokens,
                reserved_cost_usd,
            )
            .await?
        {
            GoalTransitionResult::Applied => {
                *active = Some(ActiveGoalOperation {
                    call_id,
                    cost_limit_required: goal.effective_cost_limit_usd.is_some(),
                    _permit: permit,
                });
                Ok(())
            }
            GoalTransitionResult::Stale | GoalTransitionResult::Missing => {
                anyhow::bail!("goal logical operation admission is stale")
            }
        }
    }

    async fn settle(&self, settlement: GoalOperationSettlement) -> anyhow::Result<()> {
        let (call_id, cost_limit_required) = {
            let active = self.active.lock().await;
            let operation = active
                .as_ref()
                .context("goal logical operation has no pending admission")?;
            (operation.call_id.clone(), operation.cost_limit_required)
        };
        // A response has already crossed the provider boundary. If canonical
        // persistence cannot prove every event reached JSONL, settle the exact
        // pending operation as outcome-unknown before surfacing the error; the
        // controller must never leave it resumable or replay it.
        let recorded = self.record_settlement(settlement).await;
        let (state, pricing_complete) = match &recorded {
            Ok(result) => *result,
            Err(_) => (GoalAccountingState::OutcomeUnknown, false),
        };
        match self
            .goals
            .settle_pending_call(
                &self.scope.task_id,
                &self.scope.session_id,
                self.scope.execution_epoch,
                &call_id,
                state,
            )
            .await?
        {
            GoalTransitionResult::Applied => {}
            GoalTransitionResult::Stale | GoalTransitionResult::Missing => {
                anyhow::bail!("goal logical operation settlement is stale")
            }
        }
        let cleared = self.active.lock().await.take();
        if cleared
            .as_ref()
            .is_none_or(|operation| operation.call_id != call_id)
        {
            anyhow::bail!("goal logical operation changed during settlement");
        }
        if recorded.is_err() || state == GoalAccountingState::OutcomeUnknown {
            return Err(GoalAccountingFailure::OutcomeUnknown.into());
        }
        if state != GoalAccountingState::Complete {
            return Err(GoalAccountingFailure::MissingOrInvalid.into());
        }
        if cost_limit_required && !pricing_complete {
            return Err(GoalAccountingFailure::PricingUnavailable.into());
        }
        Ok(())
    }
}

/// Transport-neutral owner of Goal parent/verifier execution.
///
/// A channel or RPC adapter supplies session-specific model turns through
/// [`GoalExecutionHost`]. This driver owns only durable Goal transitions,
/// strict verifier parsing, and accounting scope installation, so it can move
/// to a dedicated crate without carrying a transport dependency.
pub struct GoalExecutionDriver {
    host: Arc<GoalExecutionHost>,
    goals: Arc<dyn GoalTaskRegistry>,
    config: Arc<Config>,
    cost_tracker: Arc<CostTracker>,
    workers: Arc<Mutex<HashMap<String, JoinHandle<()>>>>,
}

impl GoalExecutionDriver {
    pub fn new(
        host: Arc<GoalExecutionHost>,
        goals: Arc<dyn GoalTaskRegistry>,
        config: Arc<Config>,
        cost_tracker: Arc<CostTracker>,
    ) -> Self {
        Self {
            host,
            goals,
            config,
            cost_tracker,
            workers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn terminal_reason_for_operation_error(
        error: &anyhow::Error,
        fallback: &'static str,
    ) -> &'static str {
        match error.downcast_ref::<GoalAccountingFailure>() {
            Some(GoalAccountingFailure::MissingOrInvalid) => "accounting_missing_or_invalid",
            Some(GoalAccountingFailure::OutcomeUnknown) => "accounting_outcome_unknown",
            Some(GoalAccountingFailure::PricingUnavailable) => "pricing_unavailable",
            None => fallback,
        }
    }

    async fn scope_still_running(&self, scope: &GoalExecutionScope) -> bool {
        match self.goals.current_goal_for_session(&scope.session_id).await {
            Ok(Some(task)) => {
                task.id == scope.task_id
                    && task.execution_epoch == scope.execution_epoch
                    && task.status == TaskStatus::Running
            }
            Ok(None) => false,
            Err(_) => true,
        }
    }

    /// A worker may exit only after its exact durable scope has transitioned or
    /// a newer owner has made that scope stale. Retrying storage failures keeps
    /// a failed operation from leaving an ownerless `Running` Goal behind.
    async fn finish_terminal(
        &self,
        scope: &GoalExecutionScope,
        status: TaskStatus,
        reason: Option<String>,
    ) -> bool {
        loop {
            match self
                .goals
                .finish_session_goal(
                    &scope.task_id,
                    &scope.session_id,
                    scope.execution_epoch,
                    status,
                    reason.clone(),
                )
                .await
            {
                Ok(GoalTransitionResult::Applied) => return true,
                Ok(GoalTransitionResult::Stale | GoalTransitionResult::Missing)
                    if !self.scope_still_running(scope).await =>
                {
                    return false;
                }
                Ok(_) | Err(_) => tokio::time::sleep(std::time::Duration::from_millis(50)).await,
            }
        }
    }

    async fn finish_failed(&self, scope: &GoalExecutionScope, reason: &'static str) {
        self.finish_terminal(scope, TaskStatus::Failed, Some(reason.into()))
            .await;
    }

    async fn finish_timed_out(&self, scope: &GoalExecutionScope) {
        self.finish_terminal(
            scope,
            TaskStatus::TimedOut,
            Some("goal_supervisory_deadline_elapsed".into()),
        )
        .await;
    }

    async fn pause_exact(&self, scope: &GoalExecutionScope, pause: GoalPauseState) -> bool {
        loop {
            match self
                .goals
                .pause_session_goal(
                    &scope.task_id,
                    &scope.session_id,
                    scope.execution_epoch,
                    pause.clone(),
                )
                .await
            {
                Ok(GoalTransitionResult::Applied) => return true,
                Ok(GoalTransitionResult::Stale | GoalTransitionResult::Missing)
                    if !self.scope_still_running(scope).await =>
                {
                    return false;
                }
                Ok(_) | Err(_) => tokio::time::sleep(std::time::Duration::from_millis(50)).await,
            }
        }
    }

    async fn pause_budget_exhausted(&self, scope: &GoalExecutionScope) {
        self.pause_exact(
            scope,
            GoalPauseState {
                reason: GoalPauseReason::BudgetExhausted,
                description: None,
                blockers: Vec::new(),
            },
        )
        .await;
    }

    async fn run_scope(&self, mut scope: GoalExecutionScope) {
        let (task, goal) = match GoalOperationAccountant::new(
            Arc::clone(&self.goals),
            Arc::clone(&self.config),
            Arc::clone(&self.cost_tracker),
            scope.clone(),
            scope.binding.agent().to_owned(),
        )
        .current_goal()
        .await
        {
            Ok(current) => current,
            Err(_) => {
                self.finish_failed(&scope, "goal_executor_initialization_failed")
                    .await;
                return;
            }
        };
        if task.agent != scope.binding.agent() {
            self.finish_failed(&scope, "goal_executor_binding_failed")
                .await;
            return;
        }
        scope.deadline = match self.host.deadline(&scope.binding).await {
            Ok(deadline) => deadline,
            Err(_) => {
                self.finish_failed(&scope, "goal_executor_binding_failed")
                    .await;
                return;
            }
        };
        let mut lease = match self.host.acquire(&scope, &scope.binding).await {
            Ok(lease) => lease,
            Err(_) => {
                self.finish_failed(&scope, "goal_executor_binding_failed")
                    .await;
                return;
            }
        };
        let mut working_history = match lease.working_history() {
            Ok(history) => history,
            Err(_) => {
                self.finish_failed(&scope, "goal_transcript_initialization_failed")
                    .await;
                return;
            }
        };
        let accountant = Arc::new(GoalOperationAccountant::new(
            Arc::clone(&self.goals),
            Arc::clone(&self.config),
            Arc::clone(&self.cost_tracker),
            scope.clone(),
            task.agent,
        ));
        let accounting: Arc<dyn GoalOperationAccounting> = accountant.clone();
        let tool_phase_fence = Arc::new(crate::agent::cost::GoalToolPhaseFence::new(
            Arc::clone(&self.goals),
            scope.task_id.clone(),
            scope.session_id.clone(),
            scope.execution_epoch,
        ));
        let mut turn_kind = scope.turn_kind;
        loop {
            if scope.cancellation.is_cancelled() {
                if accountant.classify_unsettled_operation().await {
                    self.finish_failed(&scope, "accounting_outcome_unknown")
                        .await;
                } else {
                    self.pause_exact(
                        &scope,
                        GoalPauseState {
                            reason: GoalPauseReason::OperatorPaused,
                            description: None,
                            blockers: Vec::new(),
                        },
                    )
                    .await;
                }
                return;
            }
            let parent = run_with_goal_deadline(
                &scope,
                crate::agent::cost::GOAL_OPERATION_ACCOUNTING.scope(
                    Some(Arc::clone(&accounting)),
                    crate::agent::cost::GOAL_TOOL_PHASE_FENCE.scope(
                        Some(Arc::clone(&tool_phase_fence)),
                        lease.run_parent_turn(
                            GoalParentTurn {
                                objective: goal.objective.clone(),
                                success_criteria: goal.success_criteria.clone(),
                                turn_kind,
                                working_history,
                            },
                            scope.cancellation.child_token(),
                        ),
                    ),
                ),
            )
            .await;
            let (candidate, next_history) = match parent {
                Ok(value) => value,
                Err(error) if error.downcast_ref::<GoalBudgetExhausted>().is_some() => {
                    self.pause_budget_exhausted(&scope).await;
                    return;
                }
                Err(error) if error.downcast_ref::<GoalDeadlineExceeded>().is_some() => {
                    accountant.classify_unsettled_operation().await;
                    self.finish_timed_out(&scope).await;
                    return;
                }
                Err(error) => {
                    let outcome_unknown = accountant.classify_unsettled_operation().await;
                    if scope.cancellation.is_cancelled() && !outcome_unknown {
                        self.pause_exact(
                            &scope,
                            GoalPauseState {
                                reason: GoalPauseReason::OperatorPaused,
                                description: None,
                                blockers: Vec::new(),
                            },
                        )
                        .await;
                        return;
                    }
                    let reason = if outcome_unknown {
                        "accounting_outcome_unknown"
                    } else {
                        Self::terminal_reason_for_operation_error(
                            &error,
                            "goal_parent_execution_failed",
                        )
                    };
                    self.finish_failed(&scope, reason).await;
                    return;
                }
            };
            if candidate.trim().is_empty() {
                self.finish_failed(&scope, "candidate_empty").await;
                return;
            }
            let verifier = run_with_goal_deadline(
                &scope,
                crate::agent::cost::GOAL_OPERATION_ACCOUNTING.scope(
                    Some(Arc::clone(&accounting)),
                    lease.run_verifier(
                        GoalVerifierTurn::isolated(
                            goal.objective.clone(),
                            goal.success_criteria.clone(),
                            candidate.clone(),
                        ),
                        scope.cancellation.child_token(),
                    ),
                ),
            )
            .await
            .and_then(|response| parse_verifier_response(&response));
            match verifier {
                Ok(VerifierOutcome::Complete) => {
                    if self
                        .finish_terminal(&scope, TaskStatus::Completed, None)
                        .await
                        && lease
                            .append_verified_candidate(super::host::GoalVerifiedCandidate {
                                task_id: scope.task_id.clone(),
                                session_id: scope.session_id.clone(),
                                execution_epoch: scope.execution_epoch,
                                candidate,
                            })
                            .await
                            .is_err()
                    {
                        ::zeroclaw_log::record!(
                            ERROR,
                            ::zeroclaw_log::Event::new(
                                module_path!(),
                                ::zeroclaw_log::Action::Fail,
                            )
                            .with_category(::zeroclaw_log::EventCategory::Agent)
                            .with_outcome(::zeroclaw_log::EventOutcome::Failure),
                            "verified Goal candidate delivery failed after completion"
                        );
                    }
                    return;
                }
                Ok(VerifierOutcome::Continue(feedback)) => {
                    working_history = next_history;
                    working_history.push(zeroclaw_api::model_provider::ChatMessage::user(format!(
                        "[UNTRUSTED VERIFIER FEEDBACK]\n{feedback}\n[END UNTRUSTED VERIFIER FEEDBACK]"
                    )));
                    turn_kind = super::GoalTurnKind::Continue;
                }
                Ok(VerifierOutcome::Blocked(reason, blockers)) => {
                    self.pause_exact(
                        &scope,
                        GoalPauseState {
                            reason: GoalPauseReason::VerifierBlocked,
                            description: Some(reason),
                            blockers,
                        },
                    )
                    .await;
                    return;
                }
                Err(error) if error.downcast_ref::<GoalBudgetExhausted>().is_some() => {
                    self.pause_budget_exhausted(&scope).await;
                    return;
                }
                Err(error) if error.downcast_ref::<GoalDeadlineExceeded>().is_some() => {
                    accountant.classify_unsettled_operation().await;
                    self.finish_timed_out(&scope).await;
                    return;
                }
                Err(error) => {
                    let outcome_unknown = accountant.classify_unsettled_operation().await;
                    if scope.cancellation.is_cancelled() && !outcome_unknown {
                        self.pause_exact(
                            &scope,
                            GoalPauseState {
                                reason: GoalPauseReason::OperatorPaused,
                                description: None,
                                blockers: Vec::new(),
                            },
                        )
                        .await;
                        return;
                    }
                    let reason = if outcome_unknown {
                        "accounting_outcome_unknown"
                    } else {
                        Self::terminal_reason_for_operation_error(&error, "goal_verifier_failed")
                    };
                    self.finish_failed(&scope, reason).await;
                    return;
                }
            }
        }
    }
}

#[async_trait]
impl GoalExecutor for GoalExecutionDriver {
    async fn preflight(&self, request: &GoalExecutionPreflight) -> GoalExecutionReadiness {
        match self.host.preflight(&request.binding, request).await {
            Ok(reservation) => GoalExecutionReadiness::Ready(reservation),
            Err(GoalHostPreflightError::PricingUnavailable) => {
                GoalExecutionReadiness::PricingUnavailable
            }
            Err(GoalHostPreflightError::VerifierUnavailable) => {
                GoalExecutionReadiness::VerifierUnavailable
            }
            Err(GoalHostPreflightError::ReservationUnavailable) => {
                GoalExecutionReadiness::ReservationUnavailable
            }
            Err(GoalHostPreflightError::BindingUnavailable) => {
                GoalExecutionReadiness::BindingUnavailable
            }
        }
    }

    async fn launch(&self, scope: GoalExecutionScope) -> Result<(), ()> {
        let mut workers = self.workers.lock().await;
        workers.retain(|_, handle| !handle.is_finished());
        if workers.contains_key(&scope.task_id) {
            return Err(());
        }
        let driver = Self {
            host: Arc::clone(&self.host),
            goals: Arc::clone(&self.goals),
            config: Arc::clone(&self.config),
            cost_tracker: Arc::clone(&self.cost_tracker),
            workers: Arc::clone(&self.workers),
        };
        let key = scope.task_id.clone();
        let handle = zeroclaw_spawn::spawn!(async move { driver.run_scope(scope).await });
        workers.insert(key, handle);
        Ok(())
    }
}

fn parse_verifier_response(text: &str) -> anyhow::Result<VerifierOutcome> {
    if text.len() > VERIFIER_RESPONSE_MAX_BYTES {
        anyhow::bail!("goal verifier response exceeds the protocol size limit");
    }
    let parsed: VerifierResponse =
        serde_json::from_str(text).context("goal verifier returned malformed JSON")?;
    if parsed.reason.trim().is_empty() || parsed.reason.chars().count() > VERIFIER_REASON_MAX_CHARS
    {
        anyhow::bail!("goal verifier reason is invalid");
    }
    match (parsed.decision, parsed.blockers) {
        (VerifierDecision::Complete, None) => Ok(VerifierOutcome::Complete),
        (VerifierDecision::Continue, None) => Ok(VerifierOutcome::Continue(parsed.reason)),
        (VerifierDecision::Blocked, Some(blockers)) => {
            if blockers.is_empty() || blockers.len() > VERIFIER_BLOCKERS_MAX {
                anyhow::bail!("goal verifier blockers are invalid");
            }
            let mut validated = Vec::with_capacity(blockers.len());
            for blocker in blockers {
                if blocker.message.trim().is_empty()
                    || blocker.message.chars().count() > VERIFIER_BLOCKER_MESSAGE_MAX_CHARS
                    || !matches!(
                        blocker.kind,
                        GoalBlockerKind::NeedsUserInput
                            | GoalBlockerKind::HumanEscalation
                            | GoalBlockerKind::ExternalDependency
                    )
                {
                    anyhow::bail!("goal verifier blocker is invalid");
                }
                validated.push(GoalBlocker {
                    kind: blocker.kind,
                    message: blocker.message,
                    payload: None,
                });
            }
            Ok(VerifierOutcome::Blocked(parsed.reason, validated))
        }
        _ => anyhow::bail!("goal verifier decision has invalid blockers"),
    }
}

fn combine_accounting_state(
    current: GoalAccountingState,
    next: GoalAccountingState,
) -> GoalAccountingState {
    use GoalAccountingState::{Complete, Invalid, Missing, OutcomeUnknown};
    match (current, next) {
        (OutcomeUnknown, _) | (_, OutcomeUnknown) => OutcomeUnknown,
        (Invalid, _) | (_, Invalid) => Invalid,
        (Missing, _) | (_, Missing) => Missing,
        (Complete, Complete) => Complete,
    }
}

fn valid_usage_event(event: &GoalUsageEvent) -> bool {
    let usage = &event.usage;
    !event.model_provider.trim().is_empty()
        && !event.model.trim().is_empty()
        && usage.model == event.model
        && usage
            .input_tokens
            .checked_add(usage.output_tokens)
            .is_some_and(|total| total == usage.total_tokens && total > 0)
        && usage.cached_input_tokens <= usage.input_tokens
        && usage.cost_usd.is_finite()
        && usage.cost_usd >= 0.0
}

fn conservative_cost_reservation(
    config: &Config,
    plan: &zeroclaw_providers::ResolvedModelOperationPlan,
    input_tokens: u64,
) -> anyhow::Result<f64> {
    plan.candidates
        .iter()
        .try_fold(0.0_f64, |total, candidate| {
            let rates = crate::agent::cost::goal_pricing_rates(
                config,
                &candidate.provider_ref,
                &candidate.model,
            );
            let input = rates
                .input_per_mtok
                .context("goal candidate input pricing is unavailable")?;
            let output = rates
                .output_per_mtok
                .context("goal candidate output pricing is unavailable")?;
            if !input.is_finite() || input < 0.0 || !output.is_finite() || output < 0.0 {
                anyhow::bail!("goal candidate pricing is invalid");
            }
            let per_attempt = (input_tokens as f64 / 1_000_000.0) * input
                + (f64::from(candidate.max_output_tokens) / 1_000_000.0) * output;
            let cost = per_attempt * f64::from(candidate.attempt_count);
            if !cost.is_finite() || cost < 0.0 {
                anyhow::bail!("goal cost reservation is invalid");
            }
            let total = total + cost;
            if !total.is_finite() {
                anyhow::bail!("goal cost reservation overflows");
            }
            Ok(total)
        })
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Mutex as StdMutex,
        atomic::{AtomicBool, Ordering},
    };

    use super::{
        GoalExecutionDriver, GoalOperationAccountant, GoalOperationAccounting, VerifierOutcome,
        parse_verifier_response,
    };
    use async_trait::async_trait;
    use zeroclaw_config::{
        cost::CostTracker,
        schema::{Config, CostConfig},
    };

    use crate::{
        control_plane::{GoalTaskRegistry, SqliteTaskStore, TaskStatus},
        goal_mode::{
            GoalBudgetRequest, GoalCommand, GoalExecutionPreflight, GoalExecutionScope,
            GoalOperationReservation, GoalRuntime,
            host::{
                GoalExecutionHost, GoalHostModelBinding, GoalHostPreflightError, GoalParentTurn,
                GoalSessionAdapter, GoalSessionBinding, GoalSessionLease, GoalVerifierTurn,
            },
        },
    };

    struct Adapter {
        leave_tool_phase_in_flight: bool,
        deadline: Option<tokio::time::Instant>,
        parent_never_returns: bool,
        parent_waits_for_cancellation: bool,
        verifier_waits_for_cancellation: bool,
        verifier_started: Option<Arc<AtomicBool>>,
        preflight_error: Option<GoalHostPreflightError>,
    }

    struct Lease {
        leave_tool_phase_in_flight: bool,
        parent_never_returns: bool,
        parent_waits_for_cancellation: bool,
        verifier_waits_for_cancellation: bool,
        verifier_started: Option<Arc<AtomicBool>>,
    }

    struct ContinueAdapter {
        parent_turn_kinds: Arc<StdMutex<Vec<crate::goal_mode::GoalTurnKind>>>,
        verifier_calls: Arc<StdMutex<usize>>,
    }

    struct ContinueLease {
        parent_turn_kinds: Arc<StdMutex<Vec<crate::goal_mode::GoalTurnKind>>>,
        verifier_calls: Arc<StdMutex<usize>>,
    }

    struct AdmittingDeadlineAdapter;

    struct AdmittingDeadlineLease;

    struct AdmittingCancellationAdapter;

    struct AdmittingCancellationLease;

    #[async_trait]
    impl GoalSessionAdapter for Adapter {
        fn owns(&self, binding: &GoalSessionBinding) -> bool {
            binding.route() == "route"
        }

        async fn preflight(
            &self,
            _binding: &GoalSessionBinding,
            _request: &GoalExecutionPreflight,
        ) -> std::result::Result<GoalOperationReservation, GoalHostPreflightError> {
            if let Some(error) = self.preflight_error {
                return Err(error);
            }
            Ok(GoalOperationReservation {
                token_count: Some(1),
                cost_usd: None,
            })
        }

        async fn model_binding(
            &self,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<GoalHostModelBinding> {
            Ok(GoalHostModelBinding {
                provider_ref: "test".into(),
                model: "model".into(),
            })
        }

        async fn deadline(
            &self,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<Option<tokio::time::Instant>> {
            Ok(self.deadline)
        }

        async fn acquire(
            &self,
            _scope: &GoalExecutionScope,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<Box<dyn GoalSessionLease>> {
            Ok(Box::new(Lease {
                leave_tool_phase_in_flight: self.leave_tool_phase_in_flight,
                parent_never_returns: self.parent_never_returns,
                parent_waits_for_cancellation: self.parent_waits_for_cancellation,
                verifier_waits_for_cancellation: self.verifier_waits_for_cancellation,
                verifier_started: self.verifier_started.clone(),
            }))
        }
    }

    #[async_trait]
    impl GoalSessionAdapter for ContinueAdapter {
        fn owns(&self, binding: &GoalSessionBinding) -> bool {
            binding.route() == "continue-route"
        }

        async fn preflight(
            &self,
            _binding: &GoalSessionBinding,
            _request: &GoalExecutionPreflight,
        ) -> std::result::Result<GoalOperationReservation, GoalHostPreflightError> {
            Ok(GoalOperationReservation::default())
        }

        async fn model_binding(
            &self,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<GoalHostModelBinding> {
            Ok(GoalHostModelBinding {
                provider_ref: "test".into(),
                model: "model".into(),
            })
        }

        async fn deadline(
            &self,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<Option<tokio::time::Instant>> {
            Ok(None)
        }

        async fn acquire(
            &self,
            _scope: &GoalExecutionScope,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<Box<dyn GoalSessionLease>> {
            Ok(Box::new(ContinueLease {
                parent_turn_kinds: Arc::clone(&self.parent_turn_kinds),
                verifier_calls: Arc::clone(&self.verifier_calls),
            }))
        }
    }

    #[async_trait]
    impl GoalSessionAdapter for AdmittingDeadlineAdapter {
        fn owns(&self, binding: &GoalSessionBinding) -> bool {
            binding.route() == "admitting-deadline-route"
        }

        async fn preflight(
            &self,
            _binding: &GoalSessionBinding,
            _request: &GoalExecutionPreflight,
        ) -> std::result::Result<GoalOperationReservation, GoalHostPreflightError> {
            Ok(GoalOperationReservation::default())
        }

        async fn model_binding(
            &self,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<GoalHostModelBinding> {
            Ok(GoalHostModelBinding {
                provider_ref: "test".into(),
                model: "model".into(),
            })
        }

        async fn deadline(
            &self,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<Option<tokio::time::Instant>> {
            Ok(Some(
                tokio::time::Instant::now() + std::time::Duration::from_millis(20),
            ))
        }

        async fn acquire(
            &self,
            _scope: &GoalExecutionScope,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<Box<dyn GoalSessionLease>> {
            Ok(Box::new(AdmittingDeadlineLease))
        }
    }

    #[async_trait]
    impl GoalSessionAdapter for AdmittingCancellationAdapter {
        fn owns(&self, binding: &GoalSessionBinding) -> bool {
            binding.route() == "admitting-cancellation-route"
        }

        async fn preflight(
            &self,
            _binding: &GoalSessionBinding,
            _request: &GoalExecutionPreflight,
        ) -> std::result::Result<GoalOperationReservation, GoalHostPreflightError> {
            Ok(GoalOperationReservation::default())
        }

        async fn model_binding(
            &self,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<GoalHostModelBinding> {
            Ok(GoalHostModelBinding {
                provider_ref: "test".into(),
                model: "model".into(),
            })
        }

        async fn deadline(
            &self,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<Option<tokio::time::Instant>> {
            Ok(None)
        }

        async fn acquire(
            &self,
            _scope: &GoalExecutionScope,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<Box<dyn GoalSessionLease>> {
            Ok(Box::new(AdmittingCancellationLease))
        }
    }

    #[async_trait]
    impl GoalSessionLease for Lease {
        fn working_history(
            &self,
        ) -> anyhow::Result<Vec<zeroclaw_api::model_provider::ChatMessage>> {
            Ok(Vec::new())
        }

        async fn run_parent_turn(
            &mut self,
            _turn: GoalParentTurn,
            _cancellation: tokio_util::sync::CancellationToken,
        ) -> anyhow::Result<(String, Vec<zeroclaw_api::model_provider::ChatMessage>)> {
            if self.parent_never_returns {
                std::future::pending::<()>().await;
            }
            if self.parent_waits_for_cancellation {
                _cancellation.cancelled().await;
                anyhow::bail!("simulated clean operation cancellation");
            }
            crate::agent::cost::begin_goal_tool_phase_if_scoped().await?;
            if self.leave_tool_phase_in_flight {
                anyhow::bail!("simulated tool-result pairing failure");
            }
            crate::agent::cost::complete_goal_tool_phase_if_scoped().await?;
            Ok(("verified candidate".into(), Vec::new()))
        }

        async fn run_verifier(
            &mut self,
            _turn: GoalVerifierTurn,
            cancellation: tokio_util::sync::CancellationToken,
        ) -> anyhow::Result<String> {
            if let Some(started) = &self.verifier_started {
                started.store(true, Ordering::Release);
            }
            if self.verifier_waits_for_cancellation {
                cancellation.cancelled().await;
                anyhow::bail!("simulated verifier cancellation");
            }
            Ok(r#"{"decision":"complete","reason":"meets criteria"}"#.into())
        }

        async fn append_verified_candidate(
            &mut self,
            _candidate: super::super::host::GoalVerifiedCandidate,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl GoalSessionLease for ContinueLease {
        fn working_history(
            &self,
        ) -> anyhow::Result<Vec<zeroclaw_api::model_provider::ChatMessage>> {
            Ok(Vec::new())
        }

        async fn run_parent_turn(
            &mut self,
            turn: GoalParentTurn,
            _cancellation: tokio_util::sync::CancellationToken,
        ) -> anyhow::Result<(String, Vec<zeroclaw_api::model_provider::ChatMessage>)> {
            self.parent_turn_kinds.lock().unwrap().push(turn.turn_kind);
            Ok(("verified candidate".into(), turn.working_history))
        }

        async fn run_verifier(
            &mut self,
            _turn: GoalVerifierTurn,
            _cancellation: tokio_util::sync::CancellationToken,
        ) -> anyhow::Result<String> {
            let mut calls = self.verifier_calls.lock().unwrap();
            *calls += 1;
            Ok(if *calls == 1 {
                r#"{"decision":"continue","reason":"add detail"}"#.into()
            } else {
                r#"{"decision":"complete","reason":"accepted"}"#.into()
            })
        }

        async fn append_verified_candidate(
            &mut self,
            _candidate: super::super::host::GoalVerifiedCandidate,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl GoalSessionLease for AdmittingDeadlineLease {
        fn working_history(
            &self,
        ) -> anyhow::Result<Vec<zeroclaw_api::model_provider::ChatMessage>> {
            Ok(Vec::new())
        }

        async fn run_parent_turn(
            &mut self,
            _turn: GoalParentTurn,
            _cancellation: tokio_util::sync::CancellationToken,
        ) -> anyhow::Result<(String, Vec<zeroclaw_api::model_provider::ChatMessage>)> {
            crate::agent::cost::admit_goal_operation_if_scoped(
                crate::agent::cost::GoalOperationRequest {
                    model_provider: "test.provider".into(),
                    model: "test-model".into(),
                    temperature: None,
                    thinking: None,
                    messages: Vec::new(),
                    tools: Vec::new(),
                },
            )
            .await?;
            std::future::pending::<()>().await;
            unreachable!()
        }

        async fn run_verifier(
            &mut self,
            _turn: GoalVerifierTurn,
            _cancellation: tokio_util::sync::CancellationToken,
        ) -> anyhow::Result<String> {
            unreachable!()
        }

        async fn append_verified_candidate(
            &mut self,
            _candidate: super::super::host::GoalVerifiedCandidate,
        ) -> anyhow::Result<()> {
            unreachable!()
        }
    }

    #[async_trait]
    impl GoalSessionLease for AdmittingCancellationLease {
        fn working_history(
            &self,
        ) -> anyhow::Result<Vec<zeroclaw_api::model_provider::ChatMessage>> {
            Ok(Vec::new())
        }

        async fn run_parent_turn(
            &mut self,
            _turn: GoalParentTurn,
            cancellation: tokio_util::sync::CancellationToken,
        ) -> anyhow::Result<(String, Vec<zeroclaw_api::model_provider::ChatMessage>)> {
            crate::agent::cost::admit_goal_operation_if_scoped(
                crate::agent::cost::GoalOperationRequest {
                    model_provider: "test.provider".into(),
                    model: "test-model".into(),
                    temperature: None,
                    thinking: None,
                    messages: Vec::new(),
                    tools: Vec::new(),
                },
            )
            .await?;
            cancellation.cancelled().await;
            anyhow::bail!("simulated unsettled operation cancellation");
        }

        async fn run_verifier(
            &mut self,
            _turn: GoalVerifierTurn,
            _cancellation: tokio_util::sync::CancellationToken,
        ) -> anyhow::Result<String> {
            unreachable!()
        }

        async fn append_verified_candidate(
            &mut self,
            _candidate: super::super::host::GoalVerifiedCandidate,
        ) -> anyhow::Result<()> {
            unreachable!()
        }
    }

    #[test]
    fn verifier_protocol_accepts_only_complete_without_blockers() {
        assert!(matches!(
            parse_verifier_response(r#"{"decision":"complete","reason":"meets criteria"}"#),
            Ok(VerifierOutcome::Complete)
        ));
        assert!(
            parse_verifier_response(
                r#"{"decision":"complete","reason":"meets criteria","blockers":[]}"#
            )
            .is_err()
        );
    }

    #[test]
    fn verifier_protocol_rejects_unknown_fields_and_untrusted_blocker_payloads() {
        assert!(
            parse_verifier_response(
                r#"{"decision":"continue","reason":"needs work","extra":"ignored?"}"#
            )
            .is_err()
        );
        assert!(parse_verifier_response(
            r#"{"decision":"blocked","reason":"needs input","blockers":[{"kind":"needs_user_input","message":"provide value","payload":{"bad":true}}]}"#
        )
        .is_err());
    }

    #[tokio::test]
    async fn supervisory_deadline_stops_a_goal_operation_before_completion() {
        let scope = GoalExecutionScope {
            task_id: "goal-task".into(),
            session_id: "session".into(),
            binding: GoalSessionBinding::new("session".into(), "main".into(), "route".into(), None)
                .unwrap(),
            execution_epoch: 1,
            turn_kind: crate::goal_mode::GoalTurnKind::Start,
            cancellation: tokio_util::sync::CancellationToken::new(),
            deadline: Some(tokio::time::Instant::now()),
            operation_permit: Arc::new(tokio::sync::Semaphore::new(1)),
        };
        let error = super::run_with_goal_deadline(&scope, async {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            Ok::<_, anyhow::Error>(())
        })
        .await
        .unwrap_err();
        assert!(
            error
                .downcast_ref::<super::GoalDeadlineExceeded>()
                .is_some()
        );
    }

    #[tokio::test]
    async fn generic_driver_completes_only_after_the_isolated_verifier_accepts() {
        let mut config = Config::default();
        config.goal.enabled = true;
        config.goal.default_token_limit = Some(10);
        config.goal.verifier.provider = "openai.verifier".into();
        let _ = config.providers.models.ensure("openai", "verifier");
        let config = Arc::new(config);
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let directory = tempfile::tempdir().unwrap();
        let tracker = Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap());
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        let host = Arc::new(GoalExecutionHost::new(vec![Arc::new(Adapter {
            leave_tool_phase_in_flight: false,
            deadline: None,
            parent_never_returns: false,
            parent_waits_for_cancellation: false,
            verifier_waits_for_cancellation: false,
            verifier_started: None,
            preflight_error: None,
        })]));
        let driver = Arc::new(GoalExecutionDriver::new(
            host,
            goals.clone(),
            Arc::clone(&config),
            Arc::clone(&tracker),
        ));
        let runtime = GoalRuntime::new(config, goals, tracker, driver, "boot").unwrap();
        let ingress =
            crate::goal_mode::GoalIngressContext::trusted("session", "main", "route", None)
                .unwrap();

        let response = runtime
            .submit(
                ingress,
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    objective: "objective".into(),
                    success_criteria: "criteria".into(),
                },
            )
            .await
            .unwrap();
        assert_eq!(response.key, crate::goal_mode::GoalMessageKey::Started);
        for _ in 0..20 {
            if let Some(task) = store.current_goal_for_session("session").await.unwrap()
                && task.status == TaskStatus::Completed
            {
                assert_eq!(
                    store
                        .get_goal_task(&task.id)
                        .await
                        .unwrap()
                        .unwrap()
                        .tool_phase,
                    crate::control_plane::GoalToolPhase::Clean,
                );
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        panic!("generic Goal driver did not complete the exact session task");
    }

    #[tokio::test]
    async fn inherited_host_deadline_transitions_the_running_goal_to_timed_out() {
        let mut config = Config::default();
        config.goal.enabled = true;
        config.goal.verifier.provider = "openai.verifier".into();
        let _ = config.providers.models.ensure("openai", "verifier");
        let config = Arc::new(config);
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        let directory = tempfile::tempdir().unwrap();
        let tracker = Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap());
        let host = Arc::new(GoalExecutionHost::new(vec![Arc::new(Adapter {
            leave_tool_phase_in_flight: false,
            deadline: Some(tokio::time::Instant::now()),
            parent_never_returns: true,
            parent_waits_for_cancellation: false,
            verifier_waits_for_cancellation: false,
            verifier_started: None,
            preflight_error: None,
        })]));
        let driver = Arc::new(GoalExecutionDriver::new(
            host,
            goals.clone(),
            Arc::clone(&config),
            Arc::clone(&tracker),
        ));
        let runtime = GoalRuntime::new(config, goals, tracker, driver, "boot").unwrap();
        let ingress =
            crate::goal_mode::GoalIngressContext::trusted("session", "main", "route", None)
                .unwrap();

        runtime
            .submit(
                ingress,
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Unlimited,
                    objective: "objective".into(),
                    success_criteria: "criteria".into(),
                },
            )
            .await
            .unwrap();
        for _ in 0..50 {
            if store
                .current_goal_for_session("session")
                .await
                .unwrap()
                .is_some_and(|task| task.status == TaskStatus::TimedOut)
            {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("Goal did not transition to timed out after its inherited deadline");
    }

    #[tokio::test]
    async fn deadline_after_admission_marks_the_pending_operation_outcome_unknown() {
        let mut config = Config::default();
        config.goal.enabled = true;
        config.goal.verifier.provider = "openai.verifier".into();
        let _ = config.providers.models.ensure("openai", "verifier");
        let config = Arc::new(config);
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        let directory = tempfile::tempdir().unwrap();
        let tracker = Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap());
        let driver = Arc::new(GoalExecutionDriver::new(
            Arc::new(GoalExecutionHost::new(vec![Arc::new(
                AdmittingDeadlineAdapter,
            )])),
            goals.clone(),
            Arc::clone(&config),
            Arc::clone(&tracker),
        ));
        let runtime = GoalRuntime::new(config, goals, tracker, driver, "boot").unwrap();
        runtime
            .submit(
                crate::goal_mode::GoalIngressContext::trusted(
                    "session",
                    "main",
                    "admitting-deadline-route",
                    None,
                )
                .unwrap(),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Unlimited,
                    objective: "objective".into(),
                    success_criteria: "criteria".into(),
                },
            )
            .await
            .unwrap();
        for _ in 0..100 {
            let Some(task) = store.current_goal_for_session("session").await.unwrap() else {
                panic!("Goal disappeared");
            };
            if task.status == TaskStatus::TimedOut {
                let goal = store.get_goal_task(&task.id).await.unwrap().unwrap();
                assert_eq!(
                    goal.accounting_state,
                    crate::control_plane::GoalAccountingState::OutcomeUnknown
                );
                assert!(goal.pending_call_id.is_none());
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        panic!("admitted Goal did not time out");
    }

    #[tokio::test]
    async fn cancellation_during_a_clean_parent_turn_pauses_the_exact_goal() {
        let mut config = Config::default();
        config.goal.enabled = true;
        config.goal.verifier.provider = "openai.verifier".into();
        let _ = config.providers.models.ensure("openai", "verifier");
        let config = Arc::new(config);
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        let directory = tempfile::tempdir().unwrap();
        let tracker = Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap());
        let driver = GoalExecutionDriver::new(
            Arc::new(GoalExecutionHost::new(vec![Arc::new(Adapter {
                leave_tool_phase_in_flight: false,
                deadline: None,
                parent_never_returns: false,
                parent_waits_for_cancellation: true,
                verifier_waits_for_cancellation: false,
                verifier_started: None,
                preflight_error: None,
            })])),
            goals.clone(),
            Arc::clone(&config),
            tracker,
        );
        let task = crate::control_plane::TaskRecord {
            id: "goal-task".into(),
            kind: crate::control_plane::TaskKind::Goal,
            agent: "main".into(),
            status: TaskStatus::Running,
            owner_pid: 1,
            owner_boot_id: "boot".into(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: Some("route".into()),
            delivered: false,
            idem_key: None,
            principal_id: None,
            session_id: Some("session".into()),
            execution_epoch: 0,
            started_at: chrono::Utc::now().to_rfc3339(),
            finished_at: None,
        };
        let goal = crate::control_plane::GoalTaskRecord {
            task_id: task.id.clone(),
            objective: "objective".into(),
            success_criteria: "criteria".into(),
            ..Default::default()
        };
        assert_eq!(
            store
                .create_or_replace_session_goal(task, goal)
                .await
                .unwrap(),
            crate::control_plane::GoalTransitionResult::Applied
        );
        let cancellation = tokio_util::sync::CancellationToken::new();
        let scope = GoalExecutionScope {
            task_id: "goal-task".into(),
            session_id: "session".into(),
            binding: GoalSessionBinding::new("session".into(), "main".into(), "route".into(), None)
                .unwrap(),
            execution_epoch: 1,
            turn_kind: crate::goal_mode::GoalTurnKind::Start,
            cancellation: cancellation.clone(),
            deadline: None,
            operation_permit: Arc::new(tokio::sync::Semaphore::new(1)),
        };
        let task = tokio::spawn(async move { driver.run_scope(scope).await });
        tokio::task::yield_now().await;
        cancellation.cancel();
        task.await.unwrap();
        let current = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(current.status, TaskStatus::Paused);
        assert_eq!(
            store
                .get_goal_task(&current.id)
                .await
                .unwrap()
                .unwrap()
                .pause_reason,
            Some(crate::control_plane::GoalPauseReason::OperatorPaused)
        );
    }

    #[tokio::test]
    async fn cancellation_during_a_clean_verifier_turn_pauses_the_exact_goal() {
        let mut config = Config::default();
        config.goal.enabled = true;
        config.goal.verifier.provider = "openai.verifier".into();
        let _ = config.providers.models.ensure("openai", "verifier");
        let config = Arc::new(config);
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        let directory = tempfile::tempdir().unwrap();
        let tracker = Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap());
        let verifier_started = Arc::new(AtomicBool::new(false));
        let driver = GoalExecutionDriver::new(
            Arc::new(GoalExecutionHost::new(vec![Arc::new(Adapter {
                leave_tool_phase_in_flight: false,
                deadline: None,
                parent_never_returns: false,
                parent_waits_for_cancellation: false,
                verifier_waits_for_cancellation: true,
                verifier_started: Some(Arc::clone(&verifier_started)),
                preflight_error: None,
            })])),
            goals.clone(),
            Arc::clone(&config),
            tracker,
        );
        let task = crate::control_plane::TaskRecord {
            id: "goal-task".into(),
            kind: crate::control_plane::TaskKind::Goal,
            agent: "main".into(),
            status: TaskStatus::Running,
            owner_pid: 1,
            owner_boot_id: "boot".into(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: Some("route".into()),
            delivered: false,
            idem_key: None,
            principal_id: None,
            session_id: Some("session".into()),
            execution_epoch: 0,
            started_at: chrono::Utc::now().to_rfc3339(),
            finished_at: None,
        };
        let goal = crate::control_plane::GoalTaskRecord {
            task_id: task.id.clone(),
            objective: "objective".into(),
            success_criteria: "criteria".into(),
            ..Default::default()
        };
        assert_eq!(
            store
                .create_or_replace_session_goal(task, goal)
                .await
                .unwrap(),
            crate::control_plane::GoalTransitionResult::Applied
        );
        let cancellation = tokio_util::sync::CancellationToken::new();
        let scope = GoalExecutionScope {
            task_id: "goal-task".into(),
            session_id: "session".into(),
            binding: GoalSessionBinding::new("session".into(), "main".into(), "route".into(), None)
                .unwrap(),
            execution_epoch: 1,
            turn_kind: crate::goal_mode::GoalTurnKind::Start,
            cancellation: cancellation.clone(),
            deadline: None,
            operation_permit: Arc::new(tokio::sync::Semaphore::new(1)),
        };
        let task = tokio::spawn(async move { driver.run_scope(scope).await });
        for _ in 0..100 {
            if verifier_started.load(Ordering::Acquire) {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(
            verifier_started.load(Ordering::Acquire),
            "verifier did not begin before cancellation"
        );
        cancellation.cancel();
        task.await.unwrap();
        let current = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(current.status, TaskStatus::Paused);
        assert_eq!(
            store
                .get_goal_task(&current.id)
                .await
                .unwrap()
                .unwrap()
                .pause_reason,
            Some(crate::control_plane::GoalPauseReason::OperatorPaused)
        );
    }

    #[tokio::test]
    async fn initial_current_goal_read_failure_fences_the_running_goal() {
        let mut config = Config::default();
        config.goal.enabled = true;
        config.goal.verifier.provider = "openai.verifier".into();
        let _ = config.providers.models.ensure("openai", "verifier");
        let config = Arc::new(config);
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        let directory = tempfile::tempdir().unwrap();
        let tracker = Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap());
        let driver = GoalExecutionDriver::new(
            Arc::new(GoalExecutionHost::new(vec![Arc::new(Adapter {
                leave_tool_phase_in_flight: false,
                deadline: None,
                parent_never_returns: false,
                parent_waits_for_cancellation: false,
                verifier_waits_for_cancellation: false,
                verifier_started: None,
                preflight_error: None,
            })])),
            goals.clone(),
            Arc::clone(&config),
            tracker,
        );
        let task = crate::control_plane::TaskRecord {
            id: "goal-task".into(),
            kind: crate::control_plane::TaskKind::Goal,
            agent: "main".into(),
            status: TaskStatus::Running,
            owner_pid: 1,
            owner_boot_id: "boot".into(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: Some("route".into()),
            delivered: false,
            idem_key: None,
            principal_id: None,
            session_id: Some("session".into()),
            execution_epoch: 0,
            started_at: chrono::Utc::now().to_rfc3339(),
            finished_at: None,
        };
        let goal = crate::control_plane::GoalTaskRecord {
            task_id: task.id.clone(),
            objective: "objective".into(),
            success_criteria: "criteria".into(),
            ..Default::default()
        };
        assert_eq!(
            store
                .create_or_replace_session_goal(task, goal)
                .await
                .unwrap(),
            crate::control_plane::GoalTransitionResult::Applied
        );
        let scope = GoalExecutionScope {
            task_id: "goal-task".into(),
            session_id: "session".into(),
            binding: GoalSessionBinding::new("session".into(), "main".into(), "route".into(), None)
                .unwrap(),
            execution_epoch: 1,
            turn_kind: crate::goal_mode::GoalTurnKind::Start,
            cancellation: tokio_util::sync::CancellationToken::new(),
            deadline: None,
            operation_permit: Arc::new(tokio::sync::Semaphore::new(1)),
        };
        store.fail_next_current_goal_read_for_test();
        driver.run_scope(scope).await;
        let current = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(current.status, TaskStatus::Failed);
        assert_eq!(
            store
                .terminal_reason_for_session_goal(&current.id, "session")
                .await
                .unwrap(),
            Some("goal_executor_initialization_failed".into())
        );
    }

    #[tokio::test]
    async fn cancellation_during_an_admitted_operation_fails_outcome_unknown() {
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        let directory = tempfile::tempdir().unwrap();
        let tracker = Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap());
        let driver = GoalExecutionDriver::new(
            Arc::new(GoalExecutionHost::new(vec![Arc::new(
                AdmittingCancellationAdapter,
            )])),
            goals.clone(),
            Arc::new(Config::default()),
            tracker,
        );
        let task = crate::control_plane::TaskRecord {
            id: "goal-task".into(),
            kind: crate::control_plane::TaskKind::Goal,
            agent: "main".into(),
            status: TaskStatus::Running,
            owner_pid: 1,
            owner_boot_id: "boot".into(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: Some("admitting-cancellation-route".into()),
            delivered: false,
            idem_key: None,
            principal_id: None,
            session_id: Some("session".into()),
            execution_epoch: 0,
            started_at: chrono::Utc::now().to_rfc3339(),
            finished_at: None,
        };
        let goal = crate::control_plane::GoalTaskRecord {
            task_id: task.id.clone(),
            objective: "objective".into(),
            success_criteria: "criteria".into(),
            ..Default::default()
        };
        assert_eq!(
            store
                .create_or_replace_session_goal(task, goal)
                .await
                .unwrap(),
            crate::control_plane::GoalTransitionResult::Applied
        );
        let cancellation = tokio_util::sync::CancellationToken::new();
        let scope = GoalExecutionScope {
            task_id: "goal-task".into(),
            session_id: "session".into(),
            binding: GoalSessionBinding::new(
                "session".into(),
                "main".into(),
                "admitting-cancellation-route".into(),
                None,
            )
            .unwrap(),
            execution_epoch: 1,
            turn_kind: crate::goal_mode::GoalTurnKind::Start,
            cancellation: cancellation.clone(),
            deadline: None,
            operation_permit: Arc::new(tokio::sync::Semaphore::new(1)),
        };
        let task = tokio::spawn(async move { driver.run_scope(scope).await });
        tokio::task::yield_now().await;
        cancellation.cancel();
        task.await.unwrap();
        let current = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(current.status, TaskStatus::Failed);
        let goal = store.get_goal_task(&current.id).await.unwrap().unwrap();
        assert_eq!(
            goal.accounting_state,
            crate::control_plane::GoalAccountingState::OutcomeUnknown
        );
        assert!(goal.pending_call_id.is_none());
    }

    #[tokio::test]
    async fn execution_driver_preserves_host_preflight_classes() {
        let cases = [
            (
                GoalHostPreflightError::BindingUnavailable,
                crate::goal_mode::GoalMessageKey::BindingUnavailable,
            ),
            (
                GoalHostPreflightError::VerifierUnavailable,
                crate::goal_mode::GoalMessageKey::VerifierUnavailable,
            ),
            (
                GoalHostPreflightError::ReservationUnavailable,
                crate::goal_mode::GoalMessageKey::ReservationUnavailable,
            ),
            (
                GoalHostPreflightError::PricingUnavailable,
                crate::goal_mode::GoalMessageKey::PricingUnavailable,
            ),
        ];
        for (preflight_error, expected) in cases {
            let mut config = Config::default();
            config.goal.enabled = true;
            config.goal.verifier.provider = "openai.verifier".into();
            let _ = config.providers.models.ensure("openai", "verifier");
            let config = Arc::new(config);
            let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
            let goals: Arc<dyn GoalTaskRegistry> = store.clone();
            let directory = tempfile::tempdir().unwrap();
            let tracker =
                Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap());
            let driver = Arc::new(GoalExecutionDriver::new(
                Arc::new(GoalExecutionHost::new(vec![Arc::new(Adapter {
                    leave_tool_phase_in_flight: false,
                    deadline: None,
                    parent_never_returns: false,
                    parent_waits_for_cancellation: false,
                    verifier_waits_for_cancellation: false,
                    verifier_started: None,
                    preflight_error: Some(preflight_error),
                })])),
                goals.clone(),
                Arc::clone(&config),
                Arc::clone(&tracker),
            ));
            let runtime = GoalRuntime::new(config, goals, tracker, driver, "boot").unwrap();
            let budget = if preflight_error == GoalHostPreflightError::PricingUnavailable {
                GoalBudgetRequest::Limits {
                    token_limit: None,
                    cost_limit_usd: Some(1.0),
                }
            } else {
                GoalBudgetRequest::Unlimited
            };
            let response = runtime
                .submit(
                    crate::goal_mode::GoalIngressContext::trusted("session", "main", "route", None)
                        .unwrap(),
                    GoalCommand::Start {
                        budget,
                        objective: "objective".into(),
                        success_criteria: "criteria".into(),
                    },
                )
                .await
                .unwrap();
            assert_eq!(response.key, expected);
            assert!(
                store
                    .current_goal_for_session("session")
                    .await
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[tokio::test]
    async fn verifier_continue_runs_the_next_parent_turn_as_continue() {
        let mut config = Config::default();
        config.goal.enabled = true;
        config.goal.verifier.provider = "openai.verifier".into();
        let _ = config.providers.models.ensure("openai", "verifier");
        let config = Arc::new(config);
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        let directory = tempfile::tempdir().unwrap();
        let tracker = Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap());
        let parent_turn_kinds = Arc::new(StdMutex::new(Vec::new()));
        let host = Arc::new(GoalExecutionHost::new(vec![Arc::new(ContinueAdapter {
            parent_turn_kinds: Arc::clone(&parent_turn_kinds),
            verifier_calls: Arc::new(StdMutex::new(0)),
        })]));
        let driver = Arc::new(GoalExecutionDriver::new(
            host,
            goals.clone(),
            Arc::clone(&config),
            Arc::clone(&tracker),
        ));
        let runtime = GoalRuntime::new(config, goals, tracker, driver, "boot").unwrap();
        let ingress = crate::goal_mode::GoalIngressContext::trusted(
            "session",
            "main",
            "continue-route",
            None,
        )
        .unwrap();

        runtime
            .submit(
                ingress,
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Unlimited,
                    objective: "objective".into(),
                    success_criteria: "criteria".into(),
                },
            )
            .await
            .unwrap();
        for _ in 0..50 {
            if store
                .current_goal_for_session("session")
                .await
                .unwrap()
                .is_some_and(|task| task.status == TaskStatus::Completed)
            {
                assert_eq!(
                    *parent_turn_kinds.lock().unwrap(),
                    vec![
                        crate::goal_mode::GoalTurnKind::Start,
                        crate::goal_mode::GoalTurnKind::Continue,
                    ],
                );
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("Goal did not complete after verifier continue");
    }

    #[tokio::test]
    async fn ledger_write_failure_settles_the_exact_pending_operation_as_unknown() {
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        let task = crate::control_plane::TaskRecord {
            id: "goal-task".into(),
            kind: crate::control_plane::TaskKind::Goal,
            agent: "main".into(),
            status: TaskStatus::Running,
            owner_pid: 1,
            owner_boot_id: "boot".into(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: Some("route".into()),
            delivered: false,
            idem_key: None,
            principal_id: None,
            session_id: Some("session".into()),
            execution_epoch: 0,
            started_at: chrono::Utc::now().to_rfc3339(),
            finished_at: None,
        };
        let goal = crate::control_plane::GoalTaskRecord {
            task_id: task.id.clone(),
            objective: "objective".into(),
            success_criteria: "criteria".into(),
            ..Default::default()
        };
        assert_eq!(
            store
                .create_or_replace_session_goal(task, goal)
                .await
                .unwrap(),
            crate::control_plane::GoalTransitionResult::Applied,
        );
        assert_eq!(
            store
                .admit_pending_call_with_reservation(
                    "goal-task",
                    "session",
                    1,
                    None,
                    None,
                    "call",
                    None,
                    None,
                )
                .await
                .unwrap(),
            crate::control_plane::GoalTransitionResult::Applied,
        );
        let directory = tempfile::tempdir().unwrap();
        let tracker = Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap());
        std::fs::create_dir_all(directory.path().join("state").join("costs.jsonl")).unwrap();
        let permit = Arc::new(tokio::sync::Semaphore::new(1))
            .acquire_owned()
            .await
            .unwrap();
        let scope = GoalExecutionScope {
            task_id: "goal-task".into(),
            session_id: "session".into(),
            binding: GoalSessionBinding::new("session".into(), "main".into(), "route".into(), None)
                .unwrap(),
            execution_epoch: 1,
            turn_kind: crate::goal_mode::GoalTurnKind::Start,
            cancellation: tokio_util::sync::CancellationToken::new(),
            deadline: None,
            operation_permit: Arc::new(tokio::sync::Semaphore::new(1)),
        };
        let accountant = GoalOperationAccountant::new(
            goals,
            Arc::new(Config::default()),
            tracker,
            scope,
            "main".into(),
        );
        *accountant.active.lock().await = Some(super::ActiveGoalOperation {
            call_id: "call".into(),
            cost_limit_required: false,
            _permit: permit,
        });

        assert!(
            accountant
                .settle(super::GoalOperationSettlement {
                    accounting_state: crate::control_plane::GoalAccountingState::Complete,
                    pricing_complete: true,
                    usage_events: vec![super::GoalUsageEvent {
                        model_provider: "actual.provider".into(),
                        model: "actual-model".into(),
                        usage: crate::cost::types::TokenUsage::new(
                            "actual-model",
                            1,
                            1,
                            0,
                            1.0,
                            1.0,
                            0.0,
                        ),
                    }],
                })
                .await
                .is_err()
        );
        let goal = store.get_goal_task("goal-task").await.unwrap().unwrap();
        assert_eq!(
            goal.accounting_state,
            crate::control_plane::GoalAccountingState::OutcomeUnknown,
        );
        assert!(goal.pending_call_id.is_none());
        assert!(goal.pending_call_epoch.is_none());
    }
}
