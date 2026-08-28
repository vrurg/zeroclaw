//! Strict accounting owner for Goal-scoped ordinary provider operations.
//!
//! This module owns admission and settlement only. It does not construct a
//! provider, select a transport, or duplicate the agent tool loop.

use std::{
    collections::HashMap,
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use anyhow::Context as _;
use async_trait::async_trait;
use serde::Deserialize;
use tokio::sync::{Mutex, OwnedSemaphorePermit};
use uuid::Uuid;
use zeroclaw_config::{cost::CostTracker, schema::Config};

use super::{
    GoalExecutionPreflight, GoalExecutionReadiness, GoalExecutionScope, GoalExecutor, GoalTurnKind,
    host::{
        GoalDeliveryScope, GoalExecutionHost, GoalHostPreflightError, GoalParentTurn,
        GoalSessionLease, GoalVerifierTurn,
    },
};
use crate::{
    agent::{
        cost::{GoalOperationAccounting, GoalOperationRequest},
        history::estimate_model_request_input_tokens,
    },
    control_plane::{
        GoalAccountingState, GoalBlocker, GoalBlockerKind, GoalTaskRecord, GoalTaskRegistry,
        GoalToolPhase, GoalTransitionResult, TaskRecord, TaskStatus,
    },
    cost::types::TokenUsage as CostTokenUsage,
};

struct ActiveGoalOperation {
    call_id: String,
    requires_complete_pricing: bool,
    _permit: OwnedSemaphorePermit,
}

#[derive(Debug, thiserror::Error)]
enum GoalOperationError {
    #[error("goal budget is exhausted")]
    BudgetExhausted,
    #[error("goal accounting is missing or invalid")]
    AccountingMissingOrInvalid,
    #[error("goal accounting outcome is unknown")]
    AccountingOutcomeUnknown,
    #[error("goal pricing is unavailable")]
    PricingUnavailable,
}

type GoalWorkerKey = (String, i64);
type GoalWorkerMap = HashMap<GoalWorkerKey, Arc<GoalWorker>>;

/// One runtime-owned worker remains discoverable until its entire final
/// delivery action has ended. Concurrent replacement commands share the same
/// join state, so none can observe the terminal row and overtake delivery.
struct GoalWorker {
    handle: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl GoalWorker {
    fn new(handle: tokio::task::JoinHandle<()>) -> Self {
        Self {
            handle: Mutex::new(Some(handle)),
        }
    }

    async fn quiesce(&self) {
        let mut handle = self.handle.lock().await;
        let Some(join) = handle.as_mut() else {
            return;
        };
        let _ = join.await;
        *handle = None;
    }
}

/// A worker which still owns a running Goal must not exit merely because the
/// durable transition that releases that ownership hit a transient error.
/// `Stale` and `Missing` prove that another durable transition already fenced
/// it; only those results may end the retry loop.
async fn retry_required_goal_transition<F, Fut>(mut transition: F) -> GoalTransitionResult
where
    F: FnMut() -> Fut,
    Fut: Future<Output = anyhow::Result<GoalTransitionResult>>,
{
    loop {
        match transition().await {
            Ok(result) => return result,
            Err(error) => {
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                        .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                        .with_attrs(::serde_json::json!({"error": error.to_string()})),
                    "Goal lifecycle transition persistence failed; retaining worker ownership"
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }
}

/// Goal-only controller for one execution epoch's logical provider operations.
///
/// It is installed task-locally around the existing session turn by the Goal
/// driver. The ordinary provider loop remains responsible for Reliable retry,
/// fallback, stream recovery, and producing the immutable attempt report.
pub(crate) struct GoalOperationAccountant {
    goals: Arc<dyn GoalTaskRegistry>,
    config: Arc<Config>,
    cost_tracker: Arc<CostTracker>,
    scope: GoalExecutionScope,
    agent_alias: String,
    active: Mutex<Option<ActiveGoalOperation>>,
    settled_operations: AtomicUsize,
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
            settled_operations: AtomicUsize::new(0),
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
            || task.agent != self.scope.binding.agent
        {
            anyhow::bail!("goal execution scope is stale");
        }
        Ok((task, goal))
    }

    async fn strict_totals(&self) -> anyhow::Result<(u64, f64, bool)> {
        let tracker = Arc::clone(&self.cost_tracker);
        let task_id = self.scope.task_id.clone();
        tokio::task::spawn_blocking(move || {
            tracker.get_strict_usage_totals_for_task_with_pricing(&task_id)
        })
        .await
        .context("goal ledger reader task failed")?
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
        let runtime_options =
            zeroclaw_providers::provider_runtime_options_for_agent(&self.config, &task.agent);
        let plan = zeroclaw_providers::resolve_routed_model_operation_plan_with_options(
            &self.config,
            &request.model_provider,
            &self.config.reliability,
            &self.config.model_routes,
            &request.model,
            &runtime_options,
            &request.model,
        )?;
        let input_tokens = estimate_model_request_input_tokens(
            &request.messages,
            (!request.tools.is_empty()).then_some(request.tools.as_slice()),
        )?;
        let reserved_tokens = goal
            .effective_token_limit
            .map(|_| {
                plan.conservative_token_reservation_for_request(input_tokens, request.thinking)
            })
            .transpose()?;
        let reserved_cost_usd = goal
            .effective_cost_limit_usd
            .map(|_| conservative_cost_reservation(&self.config, &plan, input_tokens))
            .transpose()
            .map_err(|error| {
                anyhow::Error::new(GoalOperationError::PricingUnavailable).context(error)
            })?;
        Ok((reserved_tokens, reserved_cost_usd))
    }

    async fn record_attempts(
        &self,
        attempts: &[zeroclaw_providers::dispatch::AccountedAttempt],
    ) -> anyhow::Result<(GoalAccountingState, bool)> {
        use zeroclaw_providers::dispatch::AttemptUsageOutcome;

        let mut state = if attempts.is_empty() {
            GoalAccountingState::Missing
        } else {
            GoalAccountingState::Complete
        };
        let mut pricing_complete = true;
        for attempt in attempts {
            let (usage, attempt_state) = match attempt.outcome() {
                AttemptUsageOutcome::Complete(usage) => {
                    (Some(usage), GoalAccountingState::Complete)
                }
                AttemptUsageOutcome::Missing => (None, GoalAccountingState::Missing),
                AttemptUsageOutcome::Invalid { .. } => (None, GoalAccountingState::Invalid),
                AttemptUsageOutcome::OutcomeUnknown { observed } => {
                    (observed.as_ref(), GoalAccountingState::OutcomeUnknown)
                }
            };
            state = combine_accounting_state(state, attempt_state);
            let Some(usage) = usage else { continue };
            let cost_usage = match goal_usage_from_attempt(
                &self.config,
                attempt.provider_ref(),
                attempt.model(),
                usage,
            ) {
                Ok(usage) => usage,
                Err(_) => {
                    state = combine_accounting_state(state, GoalAccountingState::Invalid);
                    continue;
                }
            };
            pricing_complete &= cost_usage.pricing_available;
            let tracker = Arc::clone(&self.cost_tracker);
            let agent_alias = self.agent_alias.clone();
            let task_id = self.scope.task_id.clone();
            tokio::task::spawn_blocking(move || {
                tracker.record_scoped_usage_with_owned_task_attribution(
                    cost_usage,
                    Some(&agent_alias),
                    Some(task_id),
                )
            })
            .await
            .context("goal ledger writer task failed")??;
        }
        Ok((state, pricing_complete))
    }

    /// A failed ledger write leaves the provider outcome uncertain. Retain the
    /// pending operation until that fact is durably visible; returning a raw
    /// storage error here would let the worker misclassify an admitted spend as
    /// an ordinary parent failure while the daemon remains alive.
    async fn retain_outcome_unknown(&self, call_id: &str) -> anyhow::Result<()> {
        let goals = Arc::clone(&self.goals);
        let task_id = self.scope.task_id.clone();
        let session_id = self.scope.session_id.clone();
        let epoch = self.scope.execution_epoch;
        let call_id = call_id.to_owned();
        match retry_required_goal_transition(move || {
            let goals = Arc::clone(&goals);
            let task_id = task_id.clone();
            let session_id = session_id.clone();
            let call_id = call_id.clone();
            async move {
                goals
                    .mark_pending_call_outcome_unknown(&task_id, &session_id, epoch, &call_id)
                    .await
            }
        })
        .await
        {
            GoalTransitionResult::Applied => Ok(()),
            GoalTransitionResult::Stale | GoalTransitionResult::Missing => {
                match self.goals.get_goal_task(&self.scope.task_id).await {
                    Ok(Some(goal)) if goal.accounting_state == GoalAccountingState::OutcomeUnknown => {
                        Ok(())
                    }
                    Ok(_) => anyhow::bail!(
                        "goal ledger failure classification was fenced before outcome reconciliation"
                    ),
                    Err(error) => Err(error).context(
                        "goal ledger failure classification was fenced before outcome reconciliation",
                    ),
                }
            }
        }
    }

    fn settled_operation_count(&self) -> usize {
        self.settled_operations.load(Ordering::Acquire)
    }
}

#[async_trait]
impl GoalOperationAccounting for GoalOperationAccountant {
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
            || goal.tool_phase != GoalToolPhase::Clean
            || goal.pending_call_id.is_some()
            || goal.pending_call_epoch.is_some()
        {
            anyhow::bail!("goal accounting is not settled");
        }
        let (known_tokens, known_cost, cost_pricing_complete) = self.strict_totals().await?;
        if goal.effective_cost_limit_usd.is_some() && !cost_pricing_complete {
            return Err(GoalOperationError::PricingUnavailable.into());
        }
        if goal
            .effective_token_limit
            .is_some_and(|limit| known_tokens >= limit)
            || goal
                .effective_cost_limit_usd
                .is_some_and(|limit| known_cost >= limit)
        {
            return Err(GoalOperationError::BudgetExhausted.into());
        }
        let (reserved_tokens, reserved_cost_usd) = self.reservation(&task, &goal, &request)?;
        if let Some(limit) = goal.effective_token_limit {
            let reserved = reserved_tokens.context("goal token reservation is missing")?;
            if known_tokens
                .checked_add(reserved)
                .context("goal token reservation overflows")?
                > limit
            {
                return Err(GoalOperationError::BudgetExhausted.into());
            }
        }
        if let Some(limit) = goal.effective_cost_limit_usd {
            let reserved = reserved_cost_usd.context("goal cost reservation is missing")?;
            let projected = known_cost + reserved;
            if !projected.is_finite() || projected > limit {
                return Err(GoalOperationError::BudgetExhausted.into());
            }
        }
        let call_id = Uuid::new_v4().to_string();
        match self
            .goals
            .admit_pending_call_with_reservation(
                &self.scope.task_id,
                &self.scope.session_id,
                self.scope.execution_epoch,
                &call_id,
                reserved_tokens,
                reserved_cost_usd,
            )
            .await?
        {
            GoalTransitionResult::Applied => {
                *active = Some(ActiveGoalOperation {
                    call_id,
                    requires_complete_pricing: goal.effective_cost_limit_usd.is_some(),
                    _permit: permit,
                });
                Ok(())
            }
            GoalTransitionResult::Stale | GoalTransitionResult::Missing => {
                anyhow::bail!("goal operation admission is stale")
            }
        }
    }

    async fn settle(
        &self,
        attempts: Vec<zeroclaw_providers::dispatch::AccountedAttempt>,
        _accepted_attempt: Option<usize>,
    ) -> anyhow::Result<()> {
        let operation = self
            .active
            .lock()
            .await
            .take()
            .context("goal operation has no pending admission")?;
        let (state, pricing_complete) = match self.record_attempts(&attempts).await {
            Ok(result) => result,
            Err(error) => {
                self.retain_outcome_unknown(&operation.call_id).await?;
                return Err(
                    anyhow::Error::new(GoalOperationError::AccountingOutcomeUnknown).context(error),
                );
            }
        };
        match self
            .goals
            .settle_pending_call(
                &self.scope.task_id,
                &self.scope.session_id,
                self.scope.execution_epoch,
                &operation.call_id,
                state,
            )
            .await?
        {
            GoalTransitionResult::Applied => {}
            GoalTransitionResult::Stale | GoalTransitionResult::Missing => {
                anyhow::bail!("goal operation settlement is stale")
            }
        }
        if state != GoalAccountingState::Complete {
            return Err(match state {
                GoalAccountingState::Missing | GoalAccountingState::Invalid => {
                    GoalOperationError::AccountingMissingOrInvalid
                }
                GoalAccountingState::OutcomeUnknown => GoalOperationError::AccountingOutcomeUnknown,
                GoalAccountingState::Complete => unreachable!("handled above"),
            }
            .into());
        }
        if operation.requires_complete_pricing && !pricing_complete {
            return Err(GoalOperationError::PricingUnavailable.into());
        }
        self.settled_operations.fetch_add(1, Ordering::Release);
        Ok(())
    }
}

/// Transport-neutral worker that owns exact Goal epochs.
///
/// Concrete adapters remain outside this module. They lease their own live
/// session and invoke the existing agent/provider path under the accountant's
/// task-local scope.
pub struct GoalExecutionDriver {
    host: Arc<GoalExecutionHost>,
    goals: Arc<dyn GoalTaskRegistry>,
    config: Arc<Config>,
    cost_tracker: Arc<CostTracker>,
    workers: Arc<Mutex<GoalWorkerMap>>,
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

    async fn finish_failed(
        &self,
        scope: &GoalExecutionScope,
        reason: &'static str,
    ) -> GoalTransitionResult {
        let goals = Arc::clone(&self.goals);
        let task_id = scope.task_id.clone();
        let session_id = scope.session_id.clone();
        let epoch = scope.execution_epoch;
        retry_required_goal_transition(move || {
            let goals = Arc::clone(&goals);
            let task_id = task_id.clone();
            let session_id = session_id.clone();
            async move {
                goals
                    .finish_session_goal(
                        &task_id,
                        &session_id,
                        epoch,
                        TaskStatus::Failed,
                        Some(reason.into()),
                    )
                    .await
            }
        })
        .await
    }

    async fn pause_budget_exhausted(&self, scope: &GoalExecutionScope) -> GoalTransitionResult {
        self.pause_goal(
            scope,
            crate::control_plane::GoalPauseState {
                reason: crate::control_plane::GoalPauseReason::BudgetExhausted,
                description: None,
                blockers: Vec::new(),
            },
        )
        .await
    }

    async fn pause_goal(
        &self,
        scope: &GoalExecutionScope,
        pause: crate::control_plane::GoalPauseState,
    ) -> GoalTransitionResult {
        let goals = Arc::clone(&self.goals);
        let task_id = scope.task_id.clone();
        let session_id = scope.session_id.clone();
        let epoch = scope.execution_epoch;
        retry_required_goal_transition(move || {
            let goals = Arc::clone(&goals);
            let task_id = task_id.clone();
            let session_id = session_id.clone();
            let pause = pause.clone();
            async move {
                goals
                    .pause_session_goal(&task_id, &session_id, epoch, pause)
                    .await
            }
        })
        .await
    }

    async fn load_initial_goal(&self, scope: &GoalExecutionScope) -> Option<GoalTaskRecord> {
        loop {
            match self.goals.get_goal_task(&scope.task_id).await {
                Ok(Some(goal)) => return Some(goal),
                Ok(None) => {
                    self.finish_failed(scope, "goal_extension_missing").await;
                    return None;
                }
                Err(error) => {
                    ::zeroclaw_log::record!(
                        WARN,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                            .with_attrs(::serde_json::json!({"error": error.to_string()})),
                        "Goal initial state read failed; retaining worker ownership"
                    );
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        }
    }

    async fn handle_operation_error(&self, scope: &GoalExecutionScope, error: &anyhow::Error) {
        match error.downcast_ref::<GoalOperationError>() {
            Some(GoalOperationError::BudgetExhausted) => {
                self.pause_budget_exhausted(scope).await;
            }
            Some(GoalOperationError::AccountingMissingOrInvalid) => {
                self.finish_failed(scope, "accounting_missing_or_invalid")
                    .await;
            }
            Some(GoalOperationError::AccountingOutcomeUnknown) => {
                self.finish_failed(scope, "accounting_outcome_unknown")
                    .await;
            }
            Some(GoalOperationError::PricingUnavailable) => {
                self.finish_failed(scope, "pricing_unavailable").await;
            }
            None => {
                self.finish_failed(scope, "goal_parent_execution_failed")
                    .await;
            }
        }
    }

    async fn run_scope(
        &self,
        scope: GoalExecutionScope,
        mut lease: Box<dyn GoalSessionLease>,
        accountant: Arc<GoalOperationAccountant>,
    ) {
        let Some(goal) = self.load_initial_goal(&scope).await else {
            return;
        };
        let mut working_transcript = Vec::new();
        let mut turn_kind = scope.turn_kind;
        let tool_phase_fence = Arc::new(crate::agent::cost::GoalToolPhaseFence::new(
            Arc::clone(&self.goals),
            scope.task_id.clone(),
            scope.session_id.clone(),
            scope.execution_epoch,
        ));
        loop {
            if scope.cancellation.is_cancelled() {
                return;
            }
            let before_parent = accountant.settled_operation_count();
            let parent = crate::agent::cost::GOAL_OPERATION_ACCOUNTING
                .scope(
                    Some(Arc::clone(&accountant) as Arc<dyn GoalOperationAccounting>),
                    crate::agent::cost::GOAL_TOOL_PHASE_FENCE.scope(
                        Some(Arc::clone(&tool_phase_fence)),
                        lease.run_parent_turn(
                            GoalParentTurn {
                                objective: goal.objective.clone(),
                                success_criteria: goal.success_criteria.clone(),
                                turn_kind,
                                working_transcript,
                            },
                            scope.cancellation.child_token(),
                        ),
                    ),
                )
                .await;
            let parent = match parent {
                Ok(parent) if accountant.settled_operation_count() > before_parent => parent,
                Ok(_) => {
                    self.finish_failed(&scope, "goal_parent_accounting_missing")
                        .await;
                    return;
                }
                Err(error) => {
                    self.handle_operation_error(&scope, &error).await;
                    return;
                }
            };
            if parent.candidate.trim().is_empty() {
                self.finish_failed(&scope, "candidate_empty").await;
                return;
            }
            let before_verifier = accountant.settled_operation_count();
            let verifier = crate::agent::cost::GOAL_OPERATION_ACCOUNTING
                .scope(
                    Some(Arc::clone(&accountant) as Arc<dyn GoalOperationAccounting>),
                    lease.run_verifier(
                        GoalVerifierTurn {
                            objective: goal.objective.clone(),
                            success_criteria: goal.success_criteria.clone(),
                            candidate: parent.candidate.clone(),
                        },
                        scope.cancellation.child_token(),
                    ),
                )
                .await;
            let verifier = match verifier {
                Ok(response) if accountant.settled_operation_count() > before_verifier => {
                    parse_verifier_response(&response)
                }
                Ok(_) => {
                    self.finish_failed(&scope, "goal_verifier_accounting_missing")
                        .await;
                    return;
                }
                Err(error) => {
                    match error.downcast_ref::<GoalOperationError>() {
                        Some(GoalOperationError::BudgetExhausted) => {
                            self.pause_budget_exhausted(&scope).await;
                        }
                        Some(GoalOperationError::AccountingMissingOrInvalid) => {
                            self.finish_failed(&scope, "accounting_missing_or_invalid")
                                .await;
                        }
                        Some(GoalOperationError::AccountingOutcomeUnknown) => {
                            self.finish_failed(&scope, "accounting_outcome_unknown")
                                .await;
                        }
                        Some(GoalOperationError::PricingUnavailable) => {
                            self.finish_failed(&scope, "pricing_unavailable").await;
                        }
                        None => {
                            self.finish_failed(&scope, "goal_verifier_failed").await;
                        }
                    }
                    return;
                }
            };
            match verifier {
                Ok(VerifierOutcome::Complete) => {
                    if accountant.current_goal().await.is_err() {
                        return;
                    }
                    let goals = Arc::clone(&self.goals);
                    let task_id = scope.task_id.clone();
                    let session_id = scope.session_id.clone();
                    let epoch = scope.execution_epoch;
                    if retry_required_goal_transition(move || {
                        let goals = Arc::clone(&goals);
                        let task_id = task_id.clone();
                        let session_id = session_id.clone();
                        async move {
                            goals
                                .finish_session_goal(
                                    &task_id,
                                    &session_id,
                                    epoch,
                                    TaskStatus::Completed,
                                    None,
                                )
                                .await
                        }
                    })
                    .await
                        != GoalTransitionResult::Applied
                    {
                        return;
                    }
                    let Ok(delivery_scope) = GoalDeliveryScope::from_completed_execution(&scope)
                    else {
                        return;
                    };
                    let delivered = match lease
                        .append_verified_candidate(&delivery_scope, &parent.candidate)
                        .await
                    {
                        Ok(()) => true,
                        Err(error) => {
                            ::zeroclaw_log::record!(
                                WARN,
                                ::zeroclaw_log::Event::new(
                                    module_path!(),
                                    ::zeroclaw_log::Action::Fail,
                                )
                                .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                                .with_attrs(::serde_json::json!({"error": error.to_string()})),
                                "verified Goal candidate delivery failed"
                            );
                            false
                        }
                    };
                    if delivered {
                        let goals = Arc::clone(&self.goals);
                        let task_id = scope.task_id.clone();
                        let session_id = scope.session_id.clone();
                        let completed_epoch = delivery_scope.completed_epoch;
                        let _ = retry_required_goal_transition(move || {
                            let goals = Arc::clone(&goals);
                            let task_id = task_id.clone();
                            let session_id = session_id.clone();
                            async move {
                                goals
                                    .acknowledge_session_goal_delivery(
                                        &task_id,
                                        &session_id,
                                        completed_epoch,
                                    )
                                    .await
                            }
                        })
                        .await;
                    }
                    return;
                }
                Ok(VerifierOutcome::Continue(feedback)) => {
                    working_transcript = parent.working_transcript;
                    working_transcript.push(zeroclaw_api::model_provider::ChatMessage::user(
                        format!(
                            "[UNTRUSTED VERIFIER FEEDBACK]\n{feedback}\n[END UNTRUSTED VERIFIER FEEDBACK]"
                        ),
                    ));
                    turn_kind = GoalTurnKind::Continue;
                }
                Ok(VerifierOutcome::Blocked(reason, blockers)) => {
                    self.pause_goal(
                        &scope,
                        crate::control_plane::GoalPauseState {
                            reason: crate::control_plane::GoalPauseReason::VerifierBlocked,
                            description: Some(reason),
                            blockers,
                        },
                    )
                    .await;
                    return;
                }
                Err(_) => {
                    self.finish_failed(&scope, "goal_verifier_protocol_failed")
                        .await;
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
            Err(GoalHostPreflightError::ReservationUnavailable) => {
                GoalExecutionReadiness::ReservationUnavailable
            }
            Err(
                GoalHostPreflightError::BindingUnavailable
                | GoalHostPreflightError::VerifierUnavailable,
            ) => GoalExecutionReadiness::VerifierUnavailable,
        }
    }

    async fn quiesce_terminal_predecessor(
        &self,
        task_id: &str,
        _session_id: &str,
    ) -> Result<(), ()> {
        let workers: Vec<_> = self
            .workers
            .lock()
            .await
            .iter()
            .filter(|((candidate_task_id, _), _)| candidate_task_id == task_id)
            .map(|(key, worker)| (key.clone(), Arc::clone(worker)))
            .collect();
        for (_, worker) in &workers {
            worker.quiesce().await;
        }
        let mut live_workers = self.workers.lock().await;
        for (key, worker) in workers {
            if live_workers
                .get(&key)
                .is_some_and(|current| Arc::ptr_eq(current, &worker))
            {
                live_workers.remove(&key);
            }
        }
        Ok(())
    }

    async fn launch(&self, scope: GoalExecutionScope) -> Result<(), ()> {
        let accountant = Arc::new(GoalOperationAccountant::new(
            Arc::clone(&self.goals),
            Arc::clone(&self.config),
            Arc::clone(&self.cost_tracker),
            scope.clone(),
            scope.binding.agent.clone(),
        ));
        if accountant.current_goal().await.is_err() {
            return Err(());
        }
        let lease = self
            .host
            .acquire(&scope, &scope.binding)
            .await
            .map_err(|_| ())?;
        if accountant.current_goal().await.is_err() {
            return Err(());
        }
        let key = (scope.task_id.clone(), scope.execution_epoch);
        let mut workers = self.workers.lock().await;
        if workers.contains_key(&key) {
            return Err(());
        }
        let runner = Self {
            host: Arc::clone(&self.host),
            goals: Arc::clone(&self.goals),
            config: Arc::clone(&self.config),
            cost_tracker: Arc::clone(&self.cost_tracker),
            workers: Arc::clone(&self.workers),
        };
        let handle = zeroclaw_spawn::spawn!(async move {
            runner.run_scope(scope, lease, accountant).await;
        });
        workers.insert(key, Arc::new(GoalWorker::new(handle)));
        Ok(())
    }
}

fn goal_usage_from_attempt(
    config: &Config,
    provider_ref: &str,
    model: &str,
    usage: &zeroclaw_providers::traits::TokenUsage,
) -> anyhow::Result<CostTokenUsage> {
    if provider_ref.trim().is_empty() || model.trim().is_empty() {
        anyhow::bail!("goal usage is missing actual provider or model attribution");
    }
    let input = usage
        .input_tokens
        .context("goal usage is missing input tokens")?;
    let output = usage
        .output_tokens
        .context("goal usage is missing output tokens")?;
    let cached = usage.cached_input_tokens.unwrap_or(0);
    if cached > input
        || input
            .checked_add(output)
            .filter(|total| *total > 0)
            .is_none()
    {
        anyhow::bail!("goal usage is invalid");
    }
    let rates = crate::agent::cost::goal_pricing_rates(config, provider_ref, model);
    if [
        rates.input_per_mtok,
        rates.output_per_mtok,
        rates.cached_input_per_mtok,
    ]
    .into_iter()
    .flatten()
    .any(|rate| !rate.is_finite() || rate < 0.0)
    {
        anyhow::bail!("goal usage resolved invalid pricing");
    }
    let pricing_available = rates.input_per_mtok.is_some() && rates.output_per_mtok.is_some();
    let mut usage = CostTokenUsage::new_with_cache(
        model,
        input,
        cached,
        output,
        rates.input_per_mtok.unwrap_or(0.0),
        rates.cached_input_per_mtok.unwrap_or(0.0),
        rates.output_per_mtok.unwrap_or(0.0),
    );
    usage.pricing_available = pricing_available;
    if !usage.cost_usd.is_finite() || usage.cost_usd < 0.0 {
        anyhow::bail!("goal usage has invalid cost");
    }
    Ok(usage)
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
            let next = total + cost;
            if !next.is_finite() || next < 0.0 {
                anyhow::bail!("goal cost reservation is invalid");
            }
            Ok(next)
        })
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

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use anyhow::Result;
    use async_trait::async_trait;
    use chrono::Utc;
    use tokio::sync::oneshot;
    use tokio_util::sync::CancellationToken;
    use zeroclaw_api::attribution::{Attributable, ModelProviderKind, ProviderKind, Role};
    use zeroclaw_api::model_provider::{ChatRequest, ChatResponse, ModelProvider};
    use zeroclaw_config::{
        cost::{CostTracker, TokenUsage as CostTokenUsage},
        schema::Config,
    };
    use zeroclaw_providers::{
        ProviderDispatch,
        dispatch::{AccountedChatScope, with_exact_dispatch_route},
    };

    use super::{
        GoalExecutionDriver, VerifierOutcome, goal_usage_from_attempt, parse_verifier_response,
    };
    use crate::{
        control_plane::{
            GoalTaskRecord, GoalTaskRegistry, GoalTransitionResult, SqliteTaskStore, TaskKind,
            TaskRecord, TaskStatus,
        },
        goal_mode::host::GoalDeliveryScope,
        goal_mode::{
            GoalExecutionScope, GoalExecutor, GoalOperationReservation, GoalTurnKind,
            host::{
                GoalExecutionHost, GoalHostPreflightError, GoalParentTurn, GoalParentTurnResult,
                GoalSessionAdapter, GoalSessionBinding, GoalSessionLease, GoalVerifierTurn,
            },
        },
    };

    struct AccountedLeaf;

    impl Attributable for AccountedLeaf {
        fn role(&self) -> Role {
            Role::Provider(ProviderKind::Model(ModelProviderKind::Custom))
        }

        fn alias(&self) -> &str {
            "accounted-leaf"
        }
    }

    #[async_trait]
    impl ModelProvider for AccountedLeaf {
        async fn chat_with_system(
            &self,
            _: Option<&str>,
            _: &str,
            _: &str,
            _: Option<f64>,
        ) -> Result<String> {
            Ok("ok".into())
        }

        async fn chat(&self, _: ChatRequest<'_>, _: &str, _: Option<f64>) -> Result<ChatResponse> {
            Ok(ChatResponse {
                text: Some("ok".into()),
                tool_calls: Vec::new(),
                usage: Some(zeroclaw_providers::traits::TokenUsage {
                    input_tokens: Some(2),
                    output_tokens: Some(1),
                    cached_input_tokens: Some(0),
                }),
                reasoning_content: None,
            })
        }
    }

    #[test]
    fn verifier_protocol_rejects_unknown_fields_and_invalid_blockers() {
        assert!(matches!(
            parse_verifier_response(r#"{"decision":"complete","reason":"accepted"}"#),
            Ok(VerifierOutcome::Complete)
        ));
        assert!(
            parse_verifier_response(
                r#"{"decision":"continue","reason":"retry","extra":"ignored"}"#
            )
            .is_err()
        );
        assert!(
            parse_verifier_response(
                r#"{"decision":"blocked","reason":"need input","blockers":[]}"#
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn lifecycle_transition_retries_a_storage_error_before_releasing_the_worker() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let result = super::retry_required_goal_transition({
            let attempts = Arc::clone(&attempts);
            move || {
                let attempts = Arc::clone(&attempts);
                async move {
                    if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                        anyhow::bail!("injected pause persistence failure");
                    }
                    Ok(GoalTransitionResult::Applied)
                }
            }
        })
        .await;
        assert_eq!(result, GoalTransitionResult::Applied);
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn uncertain_accounting_classification_retries_and_retains_the_pending_operation() {
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        assert_eq!(
            goals
                .create_or_replace_session_goal(
                    TaskRecord {
                        id: "goal".into(),
                        kind: TaskKind::Goal,
                        agent: "main".into(),
                        status: TaskStatus::Running,
                        owner_pid: std::process::id(),
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
                        started_at: Utc::now().to_rfc3339(),
                        finished_at: None,
                    },
                    GoalTaskRecord {
                        task_id: "goal".into(),
                        objective: "objective".into(),
                        success_criteria: "criteria".into(),
                        ..GoalTaskRecord::default()
                    },
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            goals
                .admit_pending_call_with_reservation("goal", "session", 1, "operation", None, None)
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        let attempts = Arc::new(AtomicUsize::new(0));
        let result = super::retry_required_goal_transition({
            let goals = Arc::clone(&goals);
            let attempts = Arc::clone(&attempts);
            move || {
                let goals = Arc::clone(&goals);
                let attempts = Arc::clone(&attempts);
                async move {
                    if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                        anyhow::bail!("injected outcome-unknown write failure");
                    }
                    goals
                        .mark_pending_call_outcome_unknown("goal", "session", 1, "operation")
                        .await
                }
            }
        })
        .await;
        assert_eq!(result, GoalTransitionResult::Applied);
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        let goal = store.get_goal_task("goal").await.unwrap().unwrap();
        assert_eq!(
            goal.accounting_state,
            crate::control_plane::GoalAccountingState::OutcomeUnknown
        );
        assert_eq!(goal.pending_call_id.as_deref(), Some("operation"));
        assert_eq!(goal.pending_call_epoch, Some(1));
    }

    #[tokio::test]
    async fn predecessor_quiescence_waits_for_the_shared_final_delivery_join() {
        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let goals: Arc<dyn GoalTaskRegistry> = store;
        let driver = Arc::new(GoalExecutionDriver::new(
            Arc::new(GoalExecutionHost::new(Vec::new())),
            goals,
            Arc::new(Config::default()),
            Arc::new(CostTracker::new(Default::default(), directory.path()).unwrap()),
        ));
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let final_delivery = zeroclaw_spawn::spawn!(async move {
            let _ = started_tx.send(());
            let _ = release_rx.await;
        });
        driver.workers.lock().await.insert(
            ("predecessor".into(), 1),
            Arc::new(super::GoalWorker::new(final_delivery)),
        );
        started_rx.await.unwrap();

        let first = {
            let driver = Arc::clone(&driver);
            zeroclaw_spawn::spawn!(async move {
                driver
                    .quiesce_terminal_predecessor("predecessor", "session")
                    .await
            })
        };
        tokio::task::yield_now().await;
        assert!(
            !first.is_finished(),
            "replacement must wait until the final delivery acknowledgement returns"
        );
        let second = {
            let driver = Arc::clone(&driver);
            zeroclaw_spawn::spawn!(async move {
                driver
                    .quiesce_terminal_predecessor("predecessor", "session")
                    .await
            })
        };
        tokio::task::yield_now().await;
        assert!(
            !second.is_finished(),
            "concurrent replacement must share the predecessor join rather than overtake it"
        );

        release_tx.send(()).unwrap();
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
        assert!(driver.workers.lock().await.is_empty());
    }

    #[test]
    fn usage_rejects_missing_zero_and_inconsistent_token_reports() {
        let config = Config::default();
        let missing = zeroclaw_providers::traits::TokenUsage {
            input_tokens: None,
            output_tokens: Some(1),
            cached_input_tokens: None,
        };
        assert!(goal_usage_from_attempt(&config, "provider", "model", &missing).is_err());
        let zero = zeroclaw_providers::traits::TokenUsage {
            input_tokens: Some(0),
            output_tokens: Some(0),
            cached_input_tokens: Some(0),
        };
        assert!(goal_usage_from_attempt(&config, "provider", "model", &zero).is_err());
        let inconsistent = zeroclaw_providers::traits::TokenUsage {
            input_tokens: Some(1),
            output_tokens: Some(1),
            cached_input_tokens: Some(2),
        };
        assert!(goal_usage_from_attempt(&config, "provider", "model", &inconsistent).is_err());
        let valid = zeroclaw_providers::traits::TokenUsage {
            input_tokens: Some(1),
            output_tokens: Some(1),
            cached_input_tokens: None,
        };
        assert!(goal_usage_from_attempt(&config, "", "model", &valid).is_err());
        assert!(goal_usage_from_attempt(&config, "provider", "", &valid).is_err());
    }

    struct UnaccountedAdapter;

    struct UnaccountedLease;

    struct BudgetExhaustedAdapter;

    struct BudgetExhaustedLease;

    #[async_trait]
    impl GoalSessionAdapter for UnaccountedAdapter {
        fn owns(&self, _: &GoalSessionBinding) -> bool {
            true
        }

        async fn preflight(
            &self,
            _: &GoalSessionBinding,
            _: &crate::goal_mode::GoalExecutionPreflight,
        ) -> std::result::Result<GoalOperationReservation, GoalHostPreflightError> {
            Ok(GoalOperationReservation::default())
        }

        async fn acquire(
            &self,
            _: &GoalExecutionScope,
            _: &GoalSessionBinding,
        ) -> Result<Box<dyn GoalSessionLease>> {
            Ok(Box::new(UnaccountedLease))
        }
    }

    #[async_trait]
    impl GoalSessionAdapter for BudgetExhaustedAdapter {
        fn owns(&self, _: &GoalSessionBinding) -> bool {
            true
        }

        async fn preflight(
            &self,
            _: &GoalSessionBinding,
            _: &crate::goal_mode::GoalExecutionPreflight,
        ) -> std::result::Result<GoalOperationReservation, GoalHostPreflightError> {
            Ok(GoalOperationReservation::default())
        }

        async fn acquire(
            &self,
            _: &GoalExecutionScope,
            _: &GoalSessionBinding,
        ) -> Result<Box<dyn GoalSessionLease>> {
            Ok(Box::new(BudgetExhaustedLease))
        }
    }

    #[async_trait]
    impl GoalSessionLease for UnaccountedLease {
        async fn run_parent_turn(
            &mut self,
            turn: GoalParentTurn,
            _: CancellationToken,
        ) -> Result<GoalParentTurnResult> {
            Ok(GoalParentTurnResult {
                candidate: "unaccounted candidate".into(),
                working_transcript: turn.working_transcript,
            })
        }

        async fn run_verifier(
            &mut self,
            _: GoalVerifierTurn,
            _: CancellationToken,
        ) -> Result<String> {
            Ok(r#"{"decision":"complete","reason":"accepted"}"#.into())
        }

        async fn append_verified_candidate(
            &mut self,
            _: &GoalDeliveryScope,
            _: &str,
        ) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl GoalSessionLease for BudgetExhaustedLease {
        async fn run_parent_turn(
            &mut self,
            _: GoalParentTurn,
            _: CancellationToken,
        ) -> Result<GoalParentTurnResult> {
            Err(super::GoalOperationError::BudgetExhausted.into())
        }

        async fn run_verifier(
            &mut self,
            _: GoalVerifierTurn,
            _: CancellationToken,
        ) -> Result<String> {
            unreachable!("the exhausted parent operation cannot reach verification")
        }

        async fn append_verified_candidate(
            &mut self,
            _: &GoalDeliveryScope,
            _: &str,
        ) -> Result<()> {
            unreachable!("the exhausted parent operation cannot complete")
        }
    }

    #[tokio::test]
    async fn unaccounted_parent_output_fails_without_completing_the_goal() {
        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        let task = TaskRecord {
            id: "goal".into(),
            kind: TaskKind::Goal,
            agent: "main".into(),
            status: TaskStatus::Running,
            owner_pid: std::process::id(),
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
            started_at: Utc::now().to_rfc3339(),
            finished_at: None,
        };
        let goal = GoalTaskRecord {
            task_id: "goal".into(),
            objective: "objective".into(),
            success_criteria: "criteria".into(),
            ..GoalTaskRecord::default()
        };
        assert_eq!(
            goals
                .create_or_replace_session_goal(task, goal)
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        let host = Arc::new(GoalExecutionHost::new(vec![Arc::new(UnaccountedAdapter)]));
        let driver = GoalExecutionDriver::new(
            host,
            Arc::clone(&goals),
            Arc::new(Config::default()),
            Arc::new(CostTracker::new(Default::default(), directory.path()).unwrap()),
        );
        let binding =
            GoalSessionBinding::new("session".into(), "main".into(), "route".into(), None).unwrap();
        driver
            .launch(GoalExecutionScope {
                task_id: "goal".into(),
                session_id: "session".into(),
                binding,
                operation_permit: Arc::new(tokio::sync::Semaphore::new(1)),
                execution_epoch: 1,
                turn_kind: GoalTurnKind::Start,
                cancellation: CancellationToken::new(),
            })
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if store
                    .current_goal_for_session("session")
                    .await
                    .unwrap()
                    .unwrap()
                    .status
                    .is_terminal()
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("unaccounted worker must finish");
        let current = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(current.status, TaskStatus::Failed);
        assert_eq!(
            store
                .terminal_reason_for_session_goal("goal", "session")
                .await
                .unwrap()
                .as_deref(),
            Some("goal_parent_accounting_missing")
        );
    }

    #[tokio::test]
    async fn budget_refusal_pauses_instead_of_failing_the_goal() {
        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        let task = TaskRecord {
            id: "goal".into(),
            kind: TaskKind::Goal,
            agent: "main".into(),
            status: TaskStatus::Running,
            owner_pid: std::process::id(),
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
            started_at: Utc::now().to_rfc3339(),
            finished_at: None,
        };
        assert_eq!(
            goals
                .create_or_replace_session_goal(
                    task,
                    GoalTaskRecord {
                        task_id: "goal".into(),
                        objective: "objective".into(),
                        success_criteria: "criteria".into(),
                        ..GoalTaskRecord::default()
                    },
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        let driver = GoalExecutionDriver::new(
            Arc::new(GoalExecutionHost::new(vec![Arc::new(
                BudgetExhaustedAdapter,
            )])),
            Arc::clone(&goals),
            Arc::new(Config::default()),
            Arc::new(CostTracker::new(Default::default(), directory.path()).unwrap()),
        );
        driver
            .launch(GoalExecutionScope {
                task_id: "goal".into(),
                session_id: "session".into(),
                binding: GoalSessionBinding::new(
                    "session".into(),
                    "main".into(),
                    "route".into(),
                    None,
                )
                .unwrap(),
                operation_permit: Arc::new(tokio::sync::Semaphore::new(1)),
                execution_epoch: 1,
                turn_kind: GoalTurnKind::Start,
                cancellation: CancellationToken::new(),
            })
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if store
                    .current_goal_for_session("session")
                    .await
                    .unwrap()
                    .unwrap()
                    .status
                    == TaskStatus::Paused
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("budget refusal must pause the worker");
        let goal = store.get_goal_task("goal").await.unwrap().unwrap();
        assert_eq!(
            goal.pause_reason,
            Some(crate::control_plane::GoalPauseReason::BudgetExhausted)
        );
    }

    #[tokio::test]
    async fn unpriced_cost_limited_admission_returns_the_typed_pricing_failure() {
        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        assert_eq!(
            goals
                .create_or_replace_session_goal(
                    TaskRecord {
                        id: "goal".into(),
                        kind: TaskKind::Goal,
                        agent: "main".into(),
                        status: TaskStatus::Running,
                        owner_pid: std::process::id(),
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
                        started_at: Utc::now().to_rfc3339(),
                        finished_at: None,
                    },
                    GoalTaskRecord {
                        task_id: "goal".into(),
                        objective: "objective".into(),
                        success_criteria: "criteria".into(),
                        effective_cost_limit_usd: Some(1.0),
                        ..GoalTaskRecord::default()
                    },
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        let tracker = Arc::new(CostTracker::new(Default::default(), directory.path()).unwrap());
        tracker
            .record_scoped_usage_with_owned_task_attribution(
                CostTokenUsage {
                    model: "unpriced-model".into(),
                    input_tokens: 1,
                    output_tokens: 0,
                    cached_input_tokens: 0,
                    total_tokens: 1,
                    cost_usd: 0.0,
                    pricing_available: false,
                    timestamp: Utc::now(),
                },
                Some("main"),
                Some("goal".into()),
            )
            .unwrap();
        let accountant = super::GoalOperationAccountant::new(
            goals,
            Arc::new(Config::default()),
            tracker,
            GoalExecutionScope {
                task_id: "goal".into(),
                session_id: "session".into(),
                binding: GoalSessionBinding::new(
                    "session".into(),
                    "main".into(),
                    "route".into(),
                    None,
                )
                .unwrap(),
                operation_permit: Arc::new(tokio::sync::Semaphore::new(1)),
                execution_epoch: 1,
                turn_kind: GoalTurnKind::Start,
                cancellation: CancellationToken::new(),
            },
            "main".into(),
        );
        let error = crate::agent::cost::GoalOperationAccounting::admit(
            &accountant,
            crate::agent::cost::GoalOperationRequest {
                model_provider: "provider".into(),
                model: "model".into(),
                temperature: None,
                thinking: None,
                messages: vec![zeroclaw_api::model_provider::ChatMessage::user("goal")],
                tools: Vec::new(),
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(
            error.downcast_ref::<super::GoalOperationError>(),
            Some(super::GoalOperationError::PricingUnavailable)
        ));
    }

    #[test]
    fn unpriced_resolved_candidate_returns_the_typed_pricing_failure() {
        let mut config = Config::default();
        let _ = config.providers.models.ensure("provider", "model");
        let task = TaskRecord {
            id: "goal".into(),
            kind: TaskKind::Goal,
            agent: "main".into(),
            status: TaskStatus::Running,
            owner_pid: 0,
            owner_boot_id: "boot".into(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: Some("route".into()),
            delivered: false,
            idem_key: None,
            principal_id: None,
            session_id: Some("session".into()),
            execution_epoch: 1,
            started_at: Utc::now().to_rfc3339(),
            finished_at: None,
        };
        let goal = GoalTaskRecord {
            task_id: "goal".into(),
            objective: "objective".into(),
            success_criteria: "criteria".into(),
            effective_cost_limit_usd: Some(1.0),
            ..GoalTaskRecord::default()
        };
        let directory = tempfile::tempdir().unwrap();
        let accountant = super::GoalOperationAccountant::new(
            Arc::new(SqliteTaskStore::new_in_memory().unwrap()),
            Arc::new(config),
            Arc::new(CostTracker::new(Default::default(), directory.path()).unwrap()),
            GoalExecutionScope {
                task_id: "goal".into(),
                session_id: "session".into(),
                binding: GoalSessionBinding::new(
                    "session".into(),
                    "main".into(),
                    "route".into(),
                    None,
                )
                .unwrap(),
                operation_permit: Arc::new(tokio::sync::Semaphore::new(1)),
                execution_epoch: 1,
                turn_kind: GoalTurnKind::Start,
                cancellation: CancellationToken::new(),
            },
            "main".into(),
        );
        let error = accountant
            .reservation(
                &task,
                &goal,
                &crate::agent::cost::GoalOperationRequest {
                    model_provider: "provider".into(),
                    model: "model".into(),
                    temperature: None,
                    thinking: None,
                    messages: vec![zeroclaw_api::model_provider::ChatMessage::user("goal")],
                    tools: Vec::new(),
                },
            )
            .unwrap_err();
        assert!(matches!(
            error.downcast_ref::<super::GoalOperationError>(),
            Some(super::GoalOperationError::PricingUnavailable)
        ));
    }

    #[tokio::test]
    async fn accounted_attempts_clear_pending_and_persist_each_task_usage_event() {
        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        let task = TaskRecord {
            id: "goal".into(),
            kind: TaskKind::Goal,
            agent: "main".into(),
            status: TaskStatus::Running,
            owner_pid: std::process::id(),
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
            started_at: Utc::now().to_rfc3339(),
            finished_at: None,
        };
        let goal = GoalTaskRecord {
            task_id: "goal".into(),
            objective: "objective".into(),
            success_criteria: "criteria".into(),
            ..GoalTaskRecord::default()
        };
        assert_eq!(
            goals
                .create_or_replace_session_goal(task, goal)
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        let tracker = Arc::new(CostTracker::new(Default::default(), directory.path()).unwrap());
        let scope = GoalExecutionScope {
            task_id: "goal".into(),
            session_id: "session".into(),
            binding: GoalSessionBinding::new("session".into(), "main".into(), "route".into(), None)
                .unwrap(),
            operation_permit: Arc::new(tokio::sync::Semaphore::new(1)),
            execution_epoch: 1,
            turn_kind: GoalTurnKind::Start,
            cancellation: CancellationToken::new(),
        };
        let accountant = Arc::new(super::GoalOperationAccountant::new(
            goals,
            Arc::new(Config::default()),
            Arc::clone(&tracker),
            scope,
            "main".into(),
        ));
        let messages = vec![zeroclaw_api::model_provider::ChatMessage::user("goal")];
        crate::agent::cost::GOAL_OPERATION_ACCOUNTING
            .scope(
                Some(
                    Arc::clone(&accountant) as Arc<dyn crate::agent::cost::GoalOperationAccounting>
                ),
                async {
                    crate::agent::cost::admit_goal_operation_if_scoped(
                        crate::agent::cost::GoalOperationRequest {
                            model_provider: "provider.test".into(),
                            model: "model-test".into(),
                            temperature: None,
                            thinking: None,
                            messages: messages.clone(),
                            tools: Vec::new(),
                        },
                    )
                    .await
                    .unwrap();
                    let first = AccountedLeaf;
                    let second = AccountedLeaf;
                    let accounting = AccountedChatScope::new();
                    let first_response = accounting
                        .scope(with_exact_dispatch_route(
                            "provider.first".into(),
                            "model-a".into(),
                            ProviderDispatch::from_ref(&first).chat(
                                ChatRequest {
                                    messages: &messages,
                                    tools: None,
                                    thinking: None,
                                },
                                "model-a",
                                None,
                            ),
                        ))
                        .await;
                    assert!(first_response.is_ok());
                    let second_response = accounting
                        .scope(with_exact_dispatch_route(
                            "provider.second".into(),
                            "model-b".into(),
                            ProviderDispatch::from_ref(&second).chat(
                                ChatRequest {
                                    messages: &messages,
                                    tools: None,
                                    thinking: None,
                                },
                                "model-b",
                                None,
                            ),
                        ))
                        .await;
                    assert!(second_response.is_ok());
                    accounting.mark_logical_success();
                    let report = accounting.take();
                    assert_eq!(report.attempts().len(), 2);
                    crate::agent::cost::settle_provider_attempts(report.attempts(), Some(0))
                        .await
                        .unwrap();
                },
            )
            .await;
        assert_eq!(
            tracker
                .get_strict_usage_totals_for_task_with_pricing("goal")
                .unwrap(),
            (6, 0.0, false)
        );
        assert_eq!(
            tracker.get_summary_for_task("goal").unwrap().request_count,
            2
        );
        let persisted = store.get_goal_task("goal").await.unwrap().unwrap();
        assert_eq!(
            persisted.accounting_state,
            crate::control_plane::GoalAccountingState::Complete
        );
        assert!(persisted.pending_call_id.is_none());
        assert!(persisted.pending_call_epoch.is_none());
    }
}
