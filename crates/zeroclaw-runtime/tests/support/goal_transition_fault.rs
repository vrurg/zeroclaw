use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use async_trait::async_trait;
use tokio::sync::Notify;
use zeroclaw_goal::task::GoalPolicyTarget;
use zeroclaw_goal::{
    GoalAccountingState, GoalPauseState, GoalTaskRecord, GoalTaskRegistry, GoalTransitionResult,
    TaskContinuationContext,
};
use zeroclaw_runtime::control_plane::{SqliteTaskStore, TaskRecord, TaskStatus};

#[derive(Debug)]
pub(super) struct OriginalTransitionFailure;

impl std::fmt::Display for OriginalTransitionFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("distinctive original Goal transition failure")
    }
}

impl std::error::Error for OriginalTransitionFailure {}

#[derive(Debug)]
pub(super) struct OriginalSettlementFailure;

impl std::fmt::Display for OriginalSettlementFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("distinctive original Goal settlement failure")
    }
}

impl std::error::Error for OriginalSettlementFailure {}

/// Delegates all lifecycle state to the real store; named finish, settlement
/// or pairing-transition calls fail once when armed. Injected failures never
/// mutate or fabricate a durable row.
pub(super) struct FinishFaultRegistry {
    store: Arc<SqliteTaskStore>,
    remaining_reason: Mutex<Option<&'static str>>,
    failures: AtomicUsize,
    pause_gate: Option<Arc<FinishFaultPauseGate>>,
    settlement_fault: AtomicBool,
    pairing_transition_fault: AtomicBool,
    settlement_failures: AtomicUsize,
    pairing_transition_failures: AtomicUsize,
    fence_observation: Option<Arc<FenceAdmissionObservation>>,
    observation: Option<Arc<GoalRegistryObservation>>,
}

/// Observe real registry calls and control commits without injecting faults.
#[derive(Default)]
pub(super) struct FenceAdmissionObservation {
    pub(super) tool_admissions: AtomicUsize,
    pub(super) tool_settlements: AtomicUsize,
    pub(super) operation_admissions: AtomicUsize,
    pub(super) control_committed: Notify,
}

#[derive(Default)]
pub(super) struct GoalRegistryObservation {
    pub(super) parent_returned: AtomicBool,
    pub(super) calls: Mutex<Vec<GoalRegistryCall>>,
    pub(super) before_retry: Option<Arc<GoalModelAdmissionGate>>,
    pub(super) model_admissions: AtomicUsize,
}

#[derive(Default)]
pub(super) struct GoalModelAdmissionGate {
    pub(super) observed: Notify,
    pub(super) released: Notify,
}

#[derive(Debug, Clone)]
#[allow(
    dead_code,
    reason = "shared test support: observations are read by the switch contract target"
)]
pub(super) struct GoalRegistryCall {
    pub(super) kind: &'static str,
    pub(super) batch_id: Option<String>,
    pub(super) pending_call_id: Option<String>,
    pub(super) pending_tool_batch_id: Option<String>,
    pub(super) parent_returned: bool,
}

/// Hold the failed worker transition until an explicit pause fence is durable.
#[derive(Default)]
pub(super) struct FinishFaultPauseGate {
    pub(super) observed: Notify,
    paused: Notify,
}

impl FinishFaultRegistry {
    pub(super) fn new(
        store: Arc<SqliteTaskStore>,
        reason: &'static str,
        pause_gate: Option<Arc<FinishFaultPauseGate>>,
        pairing_fault: Option<bool>,
    ) -> Self {
        Self {
            store,
            remaining_reason: Mutex::new(Some(reason)),
            failures: AtomicUsize::new(0),
            pause_gate,
            settlement_fault: AtomicBool::new(pairing_fault.is_some()),
            pairing_transition_fault: AtomicBool::new(pairing_fault.unwrap_or(false)),
            settlement_failures: AtomicUsize::new(0),
            pairing_transition_failures: AtomicUsize::new(0),
            fence_observation: None,
            observation: None,
        }
    }

    #[allow(
        dead_code,
        reason = "shared support: fence contract installs observations"
    )]
    pub(super) fn observing_fence(
        store: Arc<SqliteTaskStore>,
        observation: Arc<FenceAdmissionObservation>,
    ) -> Self {
        let mut registry = Self::new(store, "unused-fixture-fault", None, None);
        *registry.remaining_reason.lock().unwrap() = None;
        registry.fence_observation = Some(observation);
        registry
    }

    #[allow(
        dead_code,
        reason = "shared test support: only the switch contract target installs observations"
    )]
    pub(super) fn observing(
        store: Arc<SqliteTaskStore>,
        observation: Arc<GoalRegistryObservation>,
    ) -> Self {
        let mut registry = Self::new(store, "unused-fixture-fault", None, None);
        *registry.remaining_reason.lock().unwrap() = None;
        registry.observation = Some(observation);
        registry
    }

    async fn observe(
        &self,
        kind: &'static str,
        task_id: &str,
        batch_id: Option<&str>,
    ) -> anyhow::Result<()> {
        if let Some(observation) = &self.observation {
            let task = self.store.get_goal_task(task_id).await?.unwrap();
            observation.calls.lock().unwrap().push(GoalRegistryCall {
                kind,
                batch_id: batch_id.map(str::to_owned),
                pending_call_id: task.pending_call_id,
                pending_tool_batch_id: task.pending_tool_batch_id,
                parent_returned: observation.parent_returned.load(Ordering::SeqCst),
            });
            if kind == "model_admit"
                && observation.model_admissions.fetch_add(1, Ordering::SeqCst) == 1
                && let Some(gate) = &observation.before_retry
            {
                gate.observed.notify_one();
                gate.released.notified().await;
            }
        }
        Ok(())
    }

    pub(super) fn fault_counts(&self) -> (usize, usize, usize) {
        (
            self.failures.load(Ordering::SeqCst),
            self.settlement_failures.load(Ordering::SeqCst),
            self.pairing_transition_failures.load(Ordering::SeqCst),
        )
    }
}

#[async_trait]
impl GoalTaskRegistry for FinishFaultRegistry {
    async fn list_nonterminal_session_goals(&self) -> anyhow::Result<Vec<TaskRecord>> {
        self.store.list_nonterminal_session_goals().await
    }

    async fn cancel_policy_targets(
        &self,
        targets: &[GoalPolicyTarget],
    ) -> anyhow::Result<GoalTransitionResult> {
        self.store.cancel_policy_targets(targets).await
    }

    async fn admit_pending_tool_batch(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        batch_id: &str,
    ) -> anyhow::Result<GoalTransitionResult> {
        if let Some(observation) = &self.fence_observation {
            observation.tool_admissions.fetch_add(1, Ordering::SeqCst);
        }
        let result = self
            .store
            .admit_pending_tool_batch(task_id, session_id, expected_epoch, batch_id)
            .await?;
        self.observe("tool_admit", task_id, Some(batch_id)).await?;
        Ok(result)
    }

    async fn settle_pending_tool_batch(
        &self,
        task_id: &str,
        session_id: &str,
        admitted_epoch: i64,
        batch_id: &str,
    ) -> anyhow::Result<GoalTransitionResult> {
        if let Some(observation) = &self.fence_observation {
            observation.tool_settlements.fetch_add(1, Ordering::SeqCst);
        }
        self.observe("tool_settle", task_id, Some(batch_id)).await?;
        if self.settlement_fault.swap(false, Ordering::SeqCst) {
            self.settlement_failures.fetch_add(1, Ordering::SeqCst);
            return Err(OriginalSettlementFailure.into());
        }
        self.store
            .settle_pending_tool_batch(task_id, session_id, admitted_epoch, batch_id)
            .await
    }

    async fn fail_unpaired_tool_batch(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        admitted_epoch: i64,
        batch_id: &str,
    ) -> anyhow::Result<GoalTransitionResult> {
        if self.pairing_transition_fault.swap(false, Ordering::SeqCst) {
            self.pairing_transition_failures
                .fetch_add(1, Ordering::SeqCst);
            return Err(OriginalTransitionFailure.into());
        }
        self.store
            .fail_unpaired_tool_batch(
                task_id,
                session_id,
                expected_epoch,
                admitted_epoch,
                batch_id,
            )
            .await
    }

    async fn clear_terminal_tool_batch(
        &self,
        task_id: &str,
        session_id: &str,
        admitted_epoch: i64,
        batch_id: &str,
    ) -> anyhow::Result<GoalTransitionResult> {
        self.store
            .clear_terminal_tool_batch(task_id, session_id, admitted_epoch, batch_id)
            .await
    }
    async fn latest_active_goal_for_agent(
        &self,
        agent: &str,
    ) -> anyhow::Result<Option<TaskRecord>> {
        self.store.latest_active_goal_for_agent(agent).await
    }

    async fn latest_active_goal_for_context(
        &self,
        agent: &str,
        originator_route: Option<&str>,
        principal_id: Option<&str>,
    ) -> anyhow::Result<Option<TaskRecord>> {
        self.store
            .latest_active_goal_for_context(agent, originator_route, principal_id)
            .await
    }

    async fn latest_active_goal_id_for_context(
        &self,
        agent: &str,
        originator_route: Option<&str>,
        principal_id: Option<&str>,
    ) -> anyhow::Result<Option<String>> {
        self.store
            .latest_active_goal_id_for_context(agent, originator_route, principal_id)
            .await
    }

    async fn get_goal_task(&self, task_id: &str) -> anyhow::Result<Option<GoalTaskRecord>> {
        self.store.get_goal_task(task_id).await
    }

    async fn update_goal_limits(
        &self,
        task_id: &str,
        token_limit: Option<u64>,
        cost_limit_usd: Option<f64>,
    ) -> anyhow::Result<()> {
        self.store
            .update_goal_limits(task_id, token_limit, cost_limit_usd)
            .await
    }

    async fn update_goal_pause(
        &self,
        task_id: &str,
        pause: Option<GoalPauseState>,
    ) -> anyhow::Result<()> {
        self.store.update_goal_pause(task_id, pause).await
    }

    async fn set_continuation_context(
        &self,
        task_id: &str,
        context: Option<TaskContinuationContext>,
    ) -> anyhow::Result<()> {
        self.store.set_continuation_context(task_id, context).await
    }

    async fn get_continuation_context(
        &self,
        task_id: &str,
    ) -> anyhow::Result<Option<TaskContinuationContext>> {
        self.store.get_continuation_context(task_id).await
    }

    async fn current_goal_for_session(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<TaskRecord>> {
        self.store.current_goal_for_session(session_id).await
    }

    async fn terminal_reason_for_session_goal(
        &self,
        task_id: &str,
        session_id: &str,
    ) -> anyhow::Result<Option<String>> {
        self.store
            .terminal_reason_for_session_goal(task_id, session_id)
            .await
    }

    async fn create_or_replace_session_goal(
        &self,
        task: TaskRecord,
        goal: GoalTaskRecord,
    ) -> anyhow::Result<GoalTransitionResult> {
        self.store.create_or_replace_session_goal(task, goal).await
    }

    async fn pause_session_goal(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        pause: GoalPauseState,
    ) -> anyhow::Result<GoalTransitionResult> {
        let result = self
            .store
            .pause_session_goal(task_id, session_id, expected_epoch, pause)
            .await?;
        if result == GoalTransitionResult::Applied
            && let Some(gate) = &self.pause_gate
        {
            gate.paused.notify_one();
        }
        if result == GoalTransitionResult::Applied
            && let Some(observation) = &self.fence_observation
        {
            observation.control_committed.notify_one();
        }
        Ok(result)
    }

    async fn resume_session_goal(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        owner_pid: u32,
        owner_boot_id: &str,
    ) -> anyhow::Result<GoalTransitionResult> {
        self.store
            .resume_session_goal(
                task_id,
                session_id,
                expected_epoch,
                owner_pid,
                owner_boot_id,
            )
            .await
    }

    async fn finish_session_goal(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        status: TaskStatus,
        error: Option<String>,
    ) -> anyhow::Result<GoalTransitionResult> {
        let inject = {
            let mut remaining = self.remaining_reason.lock().unwrap();
            if status == TaskStatus::Failed
                && remaining.is_some()
                && remaining.as_deref() == error.as_deref()
            {
                remaining.take();
                true
            } else {
                false
            }
        };
        if inject {
            self.failures.fetch_add(1, Ordering::SeqCst);
            if let Some(gate) = &self.pause_gate {
                gate.observed.notify_one();
                gate.paused.notified().await;
            }
            return Err(OriginalTransitionFailure.into());
        }
        self.store
            .finish_session_goal(task_id, session_id, expected_epoch, status, error)
            .await
    }

    async fn admit_pending_operation(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        operation_id: &str,
    ) -> anyhow::Result<GoalTransitionResult> {
        if let Some(observation) = &self.fence_observation {
            observation
                .operation_admissions
                .fetch_add(1, Ordering::SeqCst);
        }
        self.observe("model_admit", task_id, None).await?;
        self.store
            .admit_pending_operation(task_id, session_id, expected_epoch, operation_id)
            .await
    }

    async fn settle_pending_operation(
        &self,
        task_id: &str,
        session_id: &str,
        admitted_epoch: i64,
        operation_id: &str,
        accounting_state: GoalAccountingState,
    ) -> anyhow::Result<GoalTransitionResult> {
        self.store
            .settle_pending_operation(
                task_id,
                session_id,
                admitted_epoch,
                operation_id,
                accounting_state,
            )
            .await
    }

    async fn update_session_goal_limits(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        token_limit: Option<u64>,
        cost_limit_usd: Option<f64>,
    ) -> anyhow::Result<GoalTransitionResult> {
        self.store
            .update_session_goal_limits(
                task_id,
                session_id,
                expected_epoch,
                token_limit,
                cost_limit_usd,
            )
            .await
    }

    async fn delete_session_goal(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
    ) -> anyhow::Result<GoalTransitionResult> {
        self.store
            .delete_session_goal(task_id, session_id, expected_epoch)
            .await
    }
}
