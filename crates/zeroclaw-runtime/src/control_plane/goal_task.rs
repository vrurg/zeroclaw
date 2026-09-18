//! Goal-specific task extensions for the durable control plane.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::task_registry::{TaskRecord, TaskStatus};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoalTaskRecord {
    /// Foreign key to the canonical [`TaskRecord`].
    pub task_id: String,
    /// Immutable, user-declared success criterion. It is supplied as
    /// untrusted prompt data to the parent and verifier, never as policy or
    /// authority data.
    pub objective: String,
    #[serde(default)]
    pub effective_token_limit: Option<u64>,
    #[serde(default)]
    pub effective_cost_limit_usd: Option<f64>,
    /// Controller-readable reason the goal is paused.
    /// This explains a canonical [`TaskStatus::Paused`] state, but does not
    /// replace it. Terminal lifecycle state remains on the canonical task row.
    #[serde(default)]
    pub pause_reason: Option<GoalPauseReason>,
    /// Human-facing pause summary. Policy must branch on `pause_reason` and
    /// `blockers`, not by parsing this text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pause_description: Option<String>,
    /// Structured blockers that explain what must change before continuation.
    /// The blocker list is the durable machine-readable resume surface for
    /// goal-specific pauses.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blockers: Vec<GoalBlocker>,
    /// Durable fence for one admitted logical provider operation.
    #[serde(default)]
    pub pending_call_id: Option<String>,
    /// Epoch that admitted `pending_call_id`; always paired with the identifier.
    #[serde(default)]
    pub pending_call_epoch: Option<i64>,
    /// Durable fence for an executable tool batch whose assistant tool-use
    /// record has not yet been paired with every result in session history.
    #[serde(default)]
    pub pending_tool_batch_id: Option<String>,
    /// Epoch that admitted `pending_tool_batch_id`; always paired with it.
    #[serde(default)]
    pub pending_tool_epoch: Option<i64>,
    /// Whether all Goal-attributed usage is known enough to admit another operation.
    #[serde(default)]
    pub accounting_state: GoalAccountingState,
}

impl Default for GoalTaskRecord {
    fn default() -> Self {
        Self {
            task_id: String::new(),
            objective: String::new(),
            effective_token_limit: None,
            effective_cost_limit_usd: None,
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
            pending_call_id: None,
            pending_call_epoch: None,
            pending_tool_batch_id: None,
            pending_tool_epoch: None,
            accounting_state: GoalAccountingState::Complete,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GoalAccountingState {
    #[default]
    Complete,
    Missing,
    Invalid,
    OutcomeUnknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalTransitionResult {
    /// The guarded mutation committed for the exact task, session, and epoch.
    Applied,
    /// The task row exists, but at least one lifecycle, epoch, kind, or
    /// session-binding predicate no longer matches. Callers must reload
    /// canonical state before making another control decision.
    Stale,
    /// No canonical task row exists for the requested task id.
    Missing,
}

/// Exact durable identity captured while preparing a prospective-policy
/// cutover.  It prevents a policy decision made for one Goal epoch from
/// cancelling a later replacement or resumed executor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalPolicyTarget {
    pub task_id: String,
    pub session_id: String,
    pub execution_epoch: i64,
}

#[derive(Debug, Clone)]
pub struct TaskGoal {
    /// Canonical task row: lifecycle, ownership, route, principal, timestamps.
    task: TaskRecord,
    /// Goal extension row: objective, effective limits, pause/blocker detail.
    goal: GoalTaskRecord,
}

impl TaskGoal {
    /// Build an on-demand goal view from its canonical task row and goal
    /// extension row.
    pub fn new(task: TaskRecord, goal: GoalTaskRecord) -> Self {
        Self { task, goal }
    }

    /// Borrow the canonical task row.
    pub fn task(&self) -> &TaskRecord {
        &self.task
    }

    /// Borrow the goal extension row.
    pub fn goal(&self) -> &GoalTaskRecord {
        &self.goal
    }

    /// Canonical task id for this goal.
    pub fn task_id(&self) -> &str {
        &self.task.id
    }

    /// Agent alias that owns this goal.
    pub fn agent(&self) -> &str {
        &self.task.agent
    }

    /// Canonical lifecycle state for this goal task.
    pub fn status(&self) -> TaskStatus {
        self.task.status
    }

    /// True when the canonical task status is `Running`.
    pub fn is_running(&self) -> bool {
        self.status() == TaskStatus::Running
    }

    /// True when the canonical task status is terminal.
    pub fn is_terminal(&self) -> bool {
        self.status().is_terminal()
    }

    /// Immutable, user-declared success criterion from the goal extension.
    /// It remains untrusted prompt data, not policy or authority data.
    pub fn objective(&self) -> &str {
        &self.goal.objective
    }

    pub fn with_effective_limits(
        mut self,
        token_limit: Option<u64>,
        cost_limit_usd: Option<f64>,
    ) -> Self {
        self.goal.effective_token_limit = token_limit;
        self.goal.effective_cost_limit_usd = cost_limit_usd;
        self
    }

    pub fn into_task(self) -> TaskRecord {
        self.task
    }

    /// Consume the view and return both canonical rows.
    pub fn into_parts(self) -> (TaskRecord, GoalTaskRecord) {
        (self.task, self.goal)
    }
}

/// Typed policy input for why a goal is paused.
/// A pause reason is goal-specific explanation layered on top of
/// [`TaskStatus::Paused`]. It must not be used as a second lifecycle enum.
/// Values remain deserializable as durable Goal audit data even where the V1
/// controller no longer creates that particular pause path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalPauseReason {
    /// An operator explicitly paused the goal through the control plane.
    OperatorPaused,
    /// The agent needs an operator answer before it can continue.
    NeedsUserInput,
    /// The agent escalated work to a human.
    HumanEscalation,
    /// A non-human dependency outside ZeroClaw is blocking progress.
    ExternalDependency,
    /// The selected provider or provider configuration is unavailable.
    ProviderUnavailable,
    /// The verifier could not produce a usable decision.
    VerifierBlocked,
    /// Goal-attributed usage reached an effective limit.
    BudgetExhausted,
    /// An effective limit exists but canonical usage records are unavailable.
    BudgetUnavailable,
    /// The daemon stopped before the goal could finish and restart recovery
    /// chose not to auto-continue it.
    #[serde(rename = "daemon_restarted", alias = "daemon_restart")]
    DaemonRestart,
}

/// Structured blocker packet attached to a paused goal.
/// Free-form text is only explanatory. Policy branches on `kind` and payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoalBlocker {
    /// Machine-readable blocker class used for policy and resume routing.
    pub kind: GoalBlockerKind,
    /// Human-readable explanation of the blocker.
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<Value>,
}

/// Coarse class of a blocker attached to a paused goal.
/// This is intentionally separate from [`GoalPauseReason`]: a pause has one
/// primary reason, while the blocker list can contain several actionable items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalBlockerKind {
    /// Waiting for an operator to resume an explicitly paused goal.
    OperatorPause,
    /// Waiting for an operator answer.
    NeedsUserInput,
    /// Waiting for human escalation handling.
    HumanEscalation,
    /// Waiting on an external system or dependency.
    ExternalDependency,
    /// Provider configuration or availability problem.
    Provider,
    /// Verifier outage or refusal to decide.
    Verifier,
    /// Effective budget limit or usage-ledger availability problem.
    Budget,
    /// Restart recovery state that needs continuation or operator action.
    RestartRecovery,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoalPauseState {
    /// Primary pause reason for controller policy.
    pub reason: GoalPauseReason,
    /// Optional human-visible summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Actionable blockers associated with the pause.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blockers: Vec<GoalBlocker>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskContinuationContext {
    /// Channel family that should receive the continuation turn.
    pub channel: String,
    /// Configured channel alias when multiple bots share the same channel
    /// family.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel_alias: Option<String>,
    /// Channel-native target used for replies or room/channel sends.
    pub reply_target: String,
    /// Original sender identity for history scope and user-visible routing.
    pub sender: String,
    /// Channel-native thread/topic id when the transport supports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_ts: Option<String>,
    /// Optional debouncer/interruption scope id to keep continuation ordering
    /// consistent with live channel turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interruption_scope_id: Option<String>,
    /// Conversation history scope to hydrate before injecting the continuation
    /// prompt.
    pub conversation_scope: TaskContinuationConversationScope,
}

/// Durable representation of the channel history scope for a continuation
/// prompt. Kept local to the control plane so the store does not depend on
/// channel transport structs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskContinuationConversationScope {
    /// Continue in the same sender-scoped history used by direct chats.
    Sender,
    /// Continue in the same reply-target scoped history used by shared rooms or
    /// channels.
    ReplyTarget,
}

#[async_trait::async_trait]
pub trait GoalTaskRegistry: Send + Sync {
    /// Resolve an observational latest non-terminal goal for `agent`.
    ///
    /// This is not a V1 control or attribution authority: multiple sessions
    /// may have Goals with the same agent, route, and principal. Goal-owned
    /// work must carry its exact task id and session binding instead.
    async fn latest_active_goal_for_agent(&self, agent: &str)
    -> anyhow::Result<Option<TaskRecord>>;

    async fn latest_active_goal_for_context(
        &self,
        agent: &str,
        originator_route: Option<&str>,
        principal_id: Option<&str>,
    ) -> anyhow::Result<Option<TaskRecord>>;

    /// Resolve an observational latest non-terminal Goal id for a context.
    ///
    /// It is not canonical for V1 attribution or lifecycle: use the exact
    /// task id carried by the Goal execution scope instead.
    async fn latest_active_goal_id_for_context(
        &self,
        agent: &str,
        originator_route: Option<&str>,
        principal_id: Option<&str>,
    ) -> anyhow::Result<Option<String>>;

    async fn get_goal_task(&self, task_id: &str) -> anyhow::Result<Option<GoalTaskRecord>>;

    /// Replace the persisted effective budget limits for a goal.
    /// These are creation/update-time policy limits only. Consumed and
    /// remaining usage stay derived from canonical usage ledger rows.
    async fn update_goal_limits(
        &self,
        task_id: &str,
        token_limit: Option<u64>,
        cost_limit_usd: Option<f64>,
    ) -> anyhow::Result<()>;

    async fn update_goal_pause(
        &self,
        task_id: &str,
        pause: Option<GoalPauseState>,
    ) -> anyhow::Result<()>;

    async fn set_continuation_context(
        &self,
        task_id: &str,
        context: Option<TaskContinuationContext>,
    ) -> anyhow::Result<()>;

    async fn get_continuation_context(
        &self,
        task_id: &str,
    ) -> anyhow::Result<Option<TaskContinuationContext>>;

    /// Return the single current Goal for a canonical session. Terminal rows
    /// remain visible until a replacement or session disposal removes them.
    async fn current_goal_for_session(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<TaskRecord>>;

    /// Enumerate all nonterminal session-bound Goals for prospective-policy
    /// classification. The returned records are observations only; a caller
    /// must pass their exact identities to [`Self::cancel_policy_targets`] to
    /// commit a revocation.
    async fn list_nonterminal_session_goals(&self) -> anyhow::Result<Vec<TaskRecord>> {
        anyhow::bail!("goal registry does not support policy classification")
    }

    /// Atomically cancel an exact set of nonterminal Goals because the
    /// successor runtime policy revokes their eligibility.
    ///
    /// Any stale or missing target rolls back the whole set. The durable
    /// cancellation fence is committed before the corresponding process-local
    /// workers may be interrupted or drained.
    async fn cancel_policy_targets(
        &self,
        targets: &[GoalPolicyTarget],
    ) -> anyhow::Result<GoalTransitionResult> {
        let _ = targets;
        anyhow::bail!("goal registry does not support policy revocation")
    }

    /// Read the raw terminal reason for one exact session-bound Goal. Callers
    /// must sanitize it before presenting it outside the control plane.
    async fn terminal_reason_for_session_goal(
        &self,
        task_id: &str,
        session_id: &str,
    ) -> anyhow::Result<Option<String>>;

    /// Atomically create a session-bound Goal or replace its fully settled
    /// terminal predecessor. New Goals always begin at execution epoch one.
    async fn create_or_replace_session_goal(
        &self,
        task: TaskRecord,
        goal: GoalTaskRecord,
    ) -> anyhow::Result<GoalTransitionResult>;

    /// Fence a running Goal and persist its pause state. An operation already
    /// admitted for the fenced epoch is allowed to settle; resume remains
    /// unavailable until that settlement clears the pending-operation slot.
    async fn pause_session_goal(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        pause: GoalPauseState,
    ) -> anyhow::Result<GoalTransitionResult>;

    /// Start a fresh executor epoch after a durable pause.
    async fn resume_session_goal(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        owner_pid: u32,
        owner_boot_id: &str,
    ) -> anyhow::Result<GoalTransitionResult>;

    /// Complete a lifecycle transition only for the exact running or paused
    /// Goal epoch. Terminal state remains owned by the task record.
    async fn finish_session_goal(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        status: TaskStatus,
        error: Option<String>,
    ) -> anyhow::Result<GoalTransitionResult>;

    /// Reserve the one durable pending-operation slot for the exact running
    /// Goal epoch. This is an execution fence, not a usage reservation.
    /// Callers must settle the exact slot on every local completion path; an
    /// unsettled slot remains fenced until recovery classifies it.
    async fn admit_pending_operation(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        pending_call_id: &str,
    ) -> anyhow::Result<GoalTransitionResult>;

    /// Settle the matching pending-operation slot after accounting has reached
    /// a durable classification.
    async fn settle_pending_operation(
        &self,
        task_id: &str,
        session_id: &str,
        admitted_epoch: i64,
        pending_call_id: &str,
        accounting_state: GoalAccountingState,
    ) -> anyhow::Result<GoalTransitionResult>;

    /// Fence one executable tool batch for the exact running Goal epoch.
    /// This is independent of the provider-operation fence: a tool batch has
    /// no spend reservation, but it must be durably paired before resumption.
    async fn admit_pending_tool_batch(
        &self,
        _task_id: &str,
        _session_id: &str,
        _expected_epoch: i64,
        _batch_id: &str,
    ) -> anyhow::Result<GoalTransitionResult> {
        anyhow::bail!("goal registry does not support durable tool pairing")
    }

    /// Clear the matching tool-batch fence after complete history pairing.
    /// Settlement remains legal after a pause has fenced the Goal to a later
    /// epoch, because it cannot admit a successor operation.
    async fn settle_pending_tool_batch(
        &self,
        _task_id: &str,
        _session_id: &str,
        _admitted_epoch: i64,
        _batch_id: &str,
    ) -> anyhow::Result<GoalTransitionResult> {
        anyhow::bail!("goal registry does not support durable tool pairing")
    }

    /// Atomically terminalize a running or paused Goal whose exact admitted
    /// tool batch could not be paired. Clearing the marker separately would
    /// create a resumable gap, so storage owns both mutations in one guard.
    async fn fail_unpaired_tool_batch(
        &self,
        _task_id: &str,
        _session_id: &str,
        _expected_epoch: i64,
        _admitted_epoch: i64,
        _batch_id: &str,
    ) -> anyhow::Result<GoalTransitionResult> {
        anyhow::bail!("goal registry does not support durable tool pairing")
    }

    /// Remove a matching marker from a terminal Goal. Terminal Goals cannot
    /// resume, so this preserves replacement/disposal progress without
    /// claiming that an interrupted batch paired cleanly.
    async fn clear_terminal_tool_batch(
        &self,
        _task_id: &str,
        _session_id: &str,
        _admitted_epoch: i64,
        _batch_id: &str,
    ) -> anyhow::Result<GoalTransitionResult> {
        anyhow::bail!("goal registry does not support durable tool pairing")
    }

    /// Atomically replace both effective limits for a running or paused Goal.
    async fn update_session_goal_limits(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        token_limit: Option<u64>,
        cost_limit_usd: Option<f64>,
    ) -> anyhow::Result<GoalTransitionResult>;

    /// Hard-delete Goal control state after it has been fenced, quiesced, and
    /// settled. Usage ledger rows are deliberately outside this operation.
    async fn delete_session_goal(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
    ) -> anyhow::Result<GoalTransitionResult>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daemon_restart_pause_reason_accepts_legacy_alias() {
        // Restart recovery's current spelling is covered with every other
        // pause reason below. Old local rows used this draft spelling.
        let legacy: GoalPauseReason = serde_json::from_str("\"daemon_restart\"").unwrap();
        assert_eq!(legacy, GoalPauseReason::DaemonRestart);
    }

    #[test]
    fn every_goal_pause_reason_remains_readable_from_persisted_control_state() {
        let reasons = [
            GoalPauseReason::OperatorPaused,
            GoalPauseReason::NeedsUserInput,
            GoalPauseReason::HumanEscalation,
            GoalPauseReason::ExternalDependency,
            GoalPauseReason::ProviderUnavailable,
            GoalPauseReason::VerifierBlocked,
            GoalPauseReason::BudgetExhausted,
            GoalPauseReason::BudgetUnavailable,
            GoalPauseReason::DaemonRestart,
        ];
        for reason in reasons {
            let wire_name = match reason {
                GoalPauseReason::OperatorPaused => "operator_paused",
                GoalPauseReason::NeedsUserInput => "needs_user_input",
                GoalPauseReason::HumanEscalation => "human_escalation",
                GoalPauseReason::ExternalDependency => "external_dependency",
                GoalPauseReason::ProviderUnavailable => "provider_unavailable",
                GoalPauseReason::VerifierBlocked => "verifier_blocked",
                GoalPauseReason::BudgetExhausted => "budget_exhausted",
                GoalPauseReason::BudgetUnavailable => "budget_unavailable",
                GoalPauseReason::DaemonRestart => "daemon_restarted",
            };
            let serialized = serde_json::to_string(&reason).unwrap();
            assert_eq!(serialized, format!("\"{wire_name}\""));
            let parsed: GoalPauseReason = serde_json::from_str(&serialized).unwrap();
            assert_eq!(parsed, reason);
        }
    }

    #[test]
    fn goal_task_loads_without_effective_limits() {
        let legacy = r#"{
            "task_id": "goal-1",
            "objective": "ship goal mode"
        }"#;
        let rec: GoalTaskRecord = serde_json::from_str(legacy).unwrap();
        assert_eq!(rec.task_id, "goal-1");
        assert!(rec.effective_token_limit.is_none());
        assert!(rec.effective_cost_limit_usd.is_none());
        assert!(rec.pause_reason.is_none());
        assert!(rec.pause_description.is_none());
        assert!(rec.blockers.is_empty());
    }
}
