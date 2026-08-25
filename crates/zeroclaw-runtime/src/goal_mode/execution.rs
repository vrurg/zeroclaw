//! Strict accounting ownership for one Goal execution scope.
//!
//! The ordinary provider turn continues to select and execute the configured
//! high-level provider path.  This module owns only the Goal admission and
//! settlement fence around that logical operation; it deliberately knows
//! nothing about RPC, Matrix, or a live session host.

use std::sync::Arc;

use anyhow::Context as _;
use async_trait::async_trait;
use tokio::sync::{Mutex, OwnedSemaphorePermit};
use uuid::Uuid;
use zeroclaw_config::{
    cost::{CostTracker, StrictTaskUsageTotals},
    schema::Config,
};

use super::GoalExecutionScope;
use crate::{
    agent::cost::{GoalOperationAccounting, GoalOperationRequest, GoalOperationSettlement},
    control_plane::{GoalAccountingState, GoalTaskRegistry, GoalTransitionResult, TaskStatus},
};

struct ActiveGoalOperation {
    call_id: String,
    cost_limit_required: bool,
    _permit: OwnedSemaphorePermit,
}

/// Private accounting owner injected into the existing provider turn by its
/// Goal execution host.  It has no transport authority and cannot construct a
/// provider, so the ordinary Reliable/direct-provider behavior remains the
/// source of execution semantics.
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

    async fn current_goal(
        &self,
    ) -> anyhow::Result<(
        crate::control_plane::TaskRecord,
        crate::control_plane::GoalTaskRecord,
    )> {
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

    /// Estimate a prepared logical operation without reproducing provider wire
    /// serialization.  The provider plan accounts for the configured Reliable
    /// candidates and retries; the request estimate is an intentionally
    /// conservative, provider-agnostic byte upper bound.
    fn request_input_token_estimate(request: &GoalOperationRequest) -> anyhow::Result<u64> {
        let messages = serde_json::to_vec(&request.messages)
            .context("serialize Goal messages for reservation")?;
        let tools =
            serde_json::to_vec(&request.tools).context("serialize Goal tools for reservation")?;
        let bytes = messages
            .len()
            .checked_add(tools.len())
            .context("Goal operation reservation input overflows")?;
        // Four UTF-8 bytes per token is deliberately a lower-bound unsafe
        // estimate. One byte per token is conservative for admission and does
        // not require a model-specific tokenizer or provider wire format.
        u64::try_from(bytes).context("Goal operation reservation input overflows")
    }

    fn reservation(
        &self,
        task: &crate::control_plane::TaskRecord,
        goal: &crate::control_plane::GoalTaskRecord,
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
            .map(|_| plan.conservative_token_reservation(input_tokens))
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
        if let Some(limit) = goal.effective_token_limit
            && totals.total_tokens >= limit
        {
            anyhow::bail!("goal budget is exhausted");
        }
        if let Some(limit) = goal.effective_cost_limit_usd
            && totals.cost_usd >= limit
        {
            anyhow::bail!("goal budget is exhausted");
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
        let mut active = self.active.lock().await;
        let operation = active
            .take()
            .context("goal logical operation has no pending admission")?;
        let cost_limit_required = operation.cost_limit_required;
        // Do not require `Running` here: pause/cancel can fence an already
        // admitted provider operation while it returns. The pending-call CAS
        // keeps the exact task/session/epoch/call binding for settlement.
        let (state, pricing_complete) = self.record_settlement(settlement).await?;
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
        if state != GoalAccountingState::Complete {
            anyhow::bail!("goal accounting is incomplete");
        }
        if cost_limit_required && !pricing_complete {
            // Valid token rows are already durable and the pending operation
            // is settled. The execution owner maps this to the distinct
            // `pricing_unavailable` lifecycle failure.
            anyhow::bail!("goal pricing is unavailable");
        }
        Ok(())
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

fn valid_usage_event(event: &crate::agent::cost::GoalUsageEvent) -> bool {
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
