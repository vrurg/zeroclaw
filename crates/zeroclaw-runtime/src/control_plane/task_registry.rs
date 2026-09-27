//! The durable task/run registry contract — EPIC A's stable seam.

pub use zeroclaw_api::task::{TaskKind, TaskRecord, TaskStatus};

/// One atomic read of a task's lifecycle and terminal payload columns.
///
/// `TaskRecord` remains the shared lifecycle record used by generic task
/// producers. Consumers that need terminal output use this view so status and
/// payload cannot come from different reads or stores.
#[derive(Debug, Clone)]
pub struct TaskSnapshot {
    pub task: TaskRecord,
    pub output: Option<String>,
    pub error: Option<String>,
}

/// A durably recorded terminal transition that has not yet been applied to the
/// canonical [`TaskRecord`]. The intent is recovery metadata, not a second
/// lifecycle authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalSettlementIntent {
    pub task_id: String,
    pub owner_pid: u32,
    pub owner_boot_id: String,
    pub desired_status: TaskStatus,
    pub artifact_path: String,
    pub artifact_ref: Option<String>,
    pub artifact_sha256: String,
    pub terminal_error: Option<String>,
}

/// THE stable seam. One trait, backed once by SQLite. The ACP session store and the
/// delegate/subagent/peer producers all converge here (CROSS-CUTTING epic-A D1).
#[async_trait::async_trait]
pub trait TaskRegistry: Send + Sync {
    /// Register a new non-Goal unit of work. Idempotent on `rec.id`; Goal
    /// admission must use the session-bound [`super::GoalTaskRegistry`] API.
    async fn create(&self, rec: TaskRecord) -> anyhow::Result<()>;
    /// Stamp a liveness beat for `id` from the heart-beating owner.
    async fn heartbeat(&self, id: &str, owner_boot_id: &str) -> anyhow::Result<()>;
    /// Transition `id` to `status`, optionally recording terminal output/error.
    async fn update_status(
        &self,
        id: &str,
        status: TaskStatus,
        output: Option<String>,
        error: Option<String>,
    ) -> anyhow::Result<()>;
    /// Atomically settle a non-terminal task and record its terminal payload.
    /// Returns `true` only for the caller that won the terminal transition.
    async fn transition_terminal(
        &self,
        id: &str,
        status: TaskStatus,
        output: Option<String>,
        error: Option<String>,
    ) -> anyhow::Result<bool> {
        let _ = (id, status, output, error);
        anyhow::bail!("task registry does not support atomic terminal transitions")
    }
    /// Atomically settle `id` only when the captured owner still matches and
    /// the task remains non-terminal. Returns `false` when ownership or
    /// lifecycle changed first.
    async fn transition_terminal_if_owner(
        &self,
        id: &str,
        owner_pid: u32,
        owner_boot_id: &str,
        status: TaskStatus,
        output: Option<String>,
        error: Option<String>,
    ) -> anyhow::Result<bool> {
        let _ = (id, owner_pid, owner_boot_id, status, output, error);
        anyhow::bail!("task registry does not support owner-checked terminal transitions")
    }
    /// Record an unapplied terminal transition before its artifact is published.
    /// Returns `false` when the task is no longer non-terminal or owner-matched.
    async fn persist_terminal_settlement_intent(
        &self,
        intent: TerminalSettlementIntent,
    ) -> anyhow::Result<bool> {
        let _ = intent;
        anyhow::bail!("task registry does not support terminal settlement intents")
    }
    /// List terminal transitions that still need recovery.
    async fn list_terminal_settlement_intents(
        &self,
    ) -> anyhow::Result<Vec<TerminalSettlementIntent>> {
        anyhow::bail!("task registry does not support terminal settlement intents")
    }
    /// Apply a settlement intent and delete that unchanged intent in one SQLite
    /// transaction. `resolved_status` is normally the intent's desired status;
    /// recovery may conservatively resolve a corrupt artifact as `Failed`.
    async fn promote_terminal_settlement(
        &self,
        intent: &TerminalSettlementIntent,
        resolved_status: TaskStatus,
        output: Option<String>,
        error: Option<String>,
    ) -> anyhow::Result<bool> {
        let _ = (intent, resolved_status, output, error);
        anyhow::bail!("task registry does not support terminal settlement intents")
    }
    /// Remove an unchanged stale intent without changing the canonical task row.
    async fn discard_terminal_settlement_intent(
        &self,
        intent: &TerminalSettlementIntent,
    ) -> anyhow::Result<bool> {
        let _ = intent;
        anyhow::bail!("task registry does not support terminal settlement intents")
    }
    async fn claim_owner(
        &self,
        id: &str,
        owner_pid: u32,
        owner_boot_id: &str,
    ) -> anyhow::Result<()>;
    async fn get(&self, id: &str) -> anyhow::Result<Option<TaskRecord>>;
    async fn get_snapshot(&self, id: &str) -> anyhow::Result<Option<TaskSnapshot>> {
        Ok(self.get(id).await?.map(|task| TaskSnapshot {
            task,
            output: None,
            error: None,
        }))
    }
    async fn list_running(&self) -> anyhow::Result<Vec<TaskRecord>>;
    async fn list_by_agent(&self, agent: &str) -> anyhow::Result<Vec<TaskRecord>>;
    /// Reaper/recovery seam: mark a record terminal-loss ONLY when this process is
    /// authoritative for it. Returns `false` (no write) when another live process
    /// owns it. `now_boot_id` remains part of the recovery seam for reaper filtering.
    /// See [`crate::control_plane::authority::is_authoritative`].
    async fn reconcile_lost(&self, id: &str, now_boot_id: &str) -> anyhow::Result<bool>;
    /// Mark a stale-heartbeat task timed out only if the observed owner and
    /// heartbeat still match. Returns `false` when liveness changed first.
    async fn reconcile_timed_out(
        &self,
        id: &str,
        owner_pid: u32,
        owner_boot_id: &str,
        heartbeat_at: &str,
    ) -> anyhow::Result<bool> {
        let _ = (id, owner_pid, owner_boot_id, heartbeat_at);
        anyhow::bail!("task registry does not support atomic timeout reconciliation")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_paths_reexport_the_canonical_api_task_types() {
        let api_kind: zeroclaw_api::task::TaskKind = TaskKind::Goal;
        let module_kind: crate::control_plane::task_registry::TaskKind = api_kind;
        let control_plane_kind: crate::control_plane::TaskKind = module_kind;
        assert_eq!(control_plane_kind, TaskKind::Goal);

        let api_status = zeroclaw_api::task::TaskStatus::Paused;
        let module_status: crate::control_plane::task_registry::TaskStatus = api_status;
        let control_plane_status: crate::control_plane::TaskStatus = module_status;
        assert_eq!(control_plane_status, TaskStatus::Paused);

        let api_record: zeroclaw_api::task::TaskRecord = serde_json::from_str(
            r#"{
                "id": "11111111-1111-1111-1111-111111111111",
                "kind": "goal",
                "agent": "main",
                "status": "paused",
                "started_at": "2026-06-18T00:00:00Z"
            }"#,
        )
        .unwrap();
        let module_record: crate::control_plane::task_registry::TaskRecord = api_record;
        let control_plane_record: crate::control_plane::TaskRecord = module_record;
        assert_eq!(
            control_plane_record.id,
            "11111111-1111-1111-1111-111111111111"
        );
    }
}
