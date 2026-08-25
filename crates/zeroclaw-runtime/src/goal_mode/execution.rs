//! Durable accounting owner for Goal-scoped logical provider operations.
//!
//! This module is deliberately transport-neutral: it receives the normal
//! runtime turn through a private task-local seam and owns only Goal admission,
//! ledger persistence, and exact pending-operation settlement.

use std::{collections::HashMap, sync::Arc};

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
    host::{GoalExecutionHost, GoalParentTurn, GoalVerifierTurn},
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
            anyhow::bail!("goal cost pricing is unavailable");
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
        let operation = self
            .active
            .lock()
            .await
            .take()
            .context("goal logical operation has no pending admission")?;
        let cost_limit_required = operation.cost_limit_required;
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
                &operation.call_id,
                state,
            )
            .await?
        {
            GoalTransitionResult::Applied => {}
            GoalTransitionResult::Stale | GoalTransitionResult::Missing => {
                anyhow::bail!("goal logical operation settlement is stale")
            }
        }
        drop(operation);
        recorded?;
        if state != GoalAccountingState::Complete {
            anyhow::bail!("goal accounting is incomplete");
        }
        if cost_limit_required && !pricing_complete {
            anyhow::bail!("goal pricing is unavailable")
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

    async fn finish_failed(&self, scope: &GoalExecutionScope, reason: &'static str) {
        let _ = self
            .goals
            .finish_session_goal(
                &scope.task_id,
                &scope.session_id,
                scope.execution_epoch,
                TaskStatus::Failed,
                Some(reason.into()),
            )
            .await;
    }

    async fn pause_budget_exhausted(&self, scope: &GoalExecutionScope) {
        let _ = self
            .goals
            .pause_session_goal(
                &scope.task_id,
                &scope.session_id,
                scope.execution_epoch,
                GoalPauseState {
                    reason: GoalPauseReason::BudgetExhausted,
                    description: None,
                    blockers: Vec::new(),
                },
            )
            .await;
    }

    async fn run_scope(&self, scope: GoalExecutionScope) {
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
            Err(_) => return,
        };
        if task.agent != scope.binding.agent() {
            self.finish_failed(&scope, "goal_executor_binding_failed")
                .await;
            return;
        }
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
        let accountant: Arc<dyn GoalOperationAccounting> = Arc::new(GoalOperationAccountant::new(
            Arc::clone(&self.goals),
            Arc::clone(&self.config),
            Arc::clone(&self.cost_tracker),
            scope.clone(),
            task.agent,
        ));
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
            let parent = crate::agent::cost::GOAL_OPERATION_ACCOUNTING
                .scope(
                    Some(Arc::clone(&accountant)),
                    crate::agent::cost::GOAL_TOOL_PHASE_FENCE.scope(
                        Some(Arc::clone(&tool_phase_fence)),
                        lease.run_parent_turn(
                            GoalParentTurn {
                                objective: goal.objective.clone(),
                                success_criteria: goal.success_criteria.clone(),
                                turn_kind: scope.turn_kind,
                                working_history,
                            },
                            scope.cancellation.child_token(),
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
                Err(_) => {
                    self.finish_failed(&scope, "goal_parent_execution_failed")
                        .await;
                    return;
                }
            };
            if candidate.trim().is_empty() {
                self.finish_failed(&scope, "candidate_empty").await;
                return;
            }
            let verifier = crate::agent::cost::GOAL_OPERATION_ACCOUNTING
                .scope(
                    Some(Arc::clone(&accountant)),
                    lease.run_verifier(
                        GoalVerifierTurn {
                            objective: goal.objective.clone(),
                            success_criteria: goal.success_criteria.clone(),
                            candidate: candidate.clone(),
                        },
                        scope.cancellation.child_token(),
                    ),
                )
                .await
                .and_then(|response| parse_verifier_response(&response));
            match verifier {
                Ok(VerifierOutcome::Complete) => {
                    if matches!(
                        self.goals
                            .finish_session_goal(
                                &scope.task_id,
                                &scope.session_id,
                                scope.execution_epoch,
                                TaskStatus::Completed,
                                None,
                            )
                            .await,
                        Ok(GoalTransitionResult::Applied)
                    ) {
                        lease.append_verified_candidate(&candidate);
                    }
                    return;
                }
                Ok(VerifierOutcome::Continue(feedback)) => {
                    working_history = next_history;
                    working_history.push(zeroclaw_api::model_provider::ChatMessage::user(format!(
                        "[UNTRUSTED VERIFIER FEEDBACK]\n{feedback}\n[END UNTRUSTED VERIFIER FEEDBACK]"
                    )));
                }
                Ok(VerifierOutcome::Blocked(reason, blockers)) => {
                    let _ = self
                        .goals
                        .pause_session_goal(
                            &scope.task_id,
                            &scope.session_id,
                            scope.execution_epoch,
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
                Err(_) => {
                    self.finish_failed(&scope, "goal_verifier_failed").await;
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
            Err(_) => GoalExecutionReadiness::VerifierUnavailable,
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
    use std::sync::Arc;

    use super::{GoalExecutionDriver, VerifierOutcome, parse_verifier_response};
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
                GoalExecutionHost, GoalHostModelBinding, GoalParentTurn, GoalSessionAdapter,
                GoalSessionBinding, GoalSessionLease, GoalVerifierTurn,
            },
        },
    };

    struct Adapter {
        leave_tool_phase_in_flight: bool,
    }

    struct Lease {
        leave_tool_phase_in_flight: bool,
    }

    #[async_trait]
    impl GoalSessionAdapter for Adapter {
        fn owns(&self, binding: &GoalSessionBinding) -> bool {
            binding.route() == "route"
        }

        async fn preflight(
            &self,
            _binding: &GoalSessionBinding,
            _request: &GoalExecutionPreflight,
        ) -> anyhow::Result<GoalOperationReservation> {
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

        async fn acquire(
            &self,
            _scope: &GoalExecutionScope,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<Box<dyn GoalSessionLease>> {
            Ok(Box::new(Lease {
                leave_tool_phase_in_flight: self.leave_tool_phase_in_flight,
            }))
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
            _cancellation: tokio_util::sync::CancellationToken,
        ) -> anyhow::Result<String> {
            Ok(r#"{"decision":"complete","reason":"meets criteria"}"#.into())
        }

        fn append_verified_candidate(&mut self, _candidate: &str) {}
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
            tokio::task::yield_now().await;
        }
        panic!("generic Goal driver did not complete the exact session task");
    }
}
