//! Durable-goal recovery delivery ownership shared by process startup and reload.
//!
//! The durable task and goal stores remain the lifecycle source of truth. The
//! queue guards in this module own only transient responsibility for delivering
//! a continuation or moving the durable goal to a non-running state.

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

use super::boot::ControlPlaneHandle;
use super::goal_task::{GoalBlocker, GoalBlockerKind, GoalPauseReason, GoalPauseState};
use super::task_registry::{TaskKind, TaskStatus};

/// Controller-owned reason a recovered goal could not be re-enqueued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveredGoalContinuationBlocker {
    /// Goal owner no longer resolves to an enabled agent runtime.
    AgentUnavailable,
    /// The effective model provider for the recovered turn cannot be constructed.
    ProviderUnavailable,
    /// Durable goal row has no continuation delivery context.
    MissingContinuationContext,
    /// Continuation context could not be read from the goal store.
    ContinuationContextReadFailed,
    /// Continuation context disagrees with the canonical task route or principal.
    InvalidContinuationContext,
    /// Durable task row has no goal-specific extension record.
    MissingGoalExtension,
    /// Goal-specific extension could not be read from the goal store.
    GoalExtensionReadFailed,
    /// The channel referenced by the continuation context is unavailable.
    ChannelUnavailable,
    /// The continuation channel is not assigned to the goal's owning agent.
    ChannelOwnerUnavailable,
    /// The channel worker queue rejected the recovered continuation.
    QueueUnavailable,
}

impl RecoveredGoalContinuationBlocker {
    /// Stable machine-readable blocker code persisted with the goal.
    pub fn code(self) -> &'static str {
        match self {
            Self::AgentUnavailable => "agent_unavailable",
            Self::ProviderUnavailable => "provider_unavailable",
            Self::MissingContinuationContext => "missing_continuation_context",
            Self::ContinuationContextReadFailed => "continuation_context_read_failed",
            Self::InvalidContinuationContext => "invalid_continuation_context",
            Self::MissingGoalExtension => "missing_goal_extension",
            Self::GoalExtensionReadFailed => "goal_extension_read_failed",
            Self::ChannelUnavailable => "channel_unavailable",
            Self::ChannelOwnerUnavailable => "channel_owner_unavailable",
            Self::QueueUnavailable => "queue_unavailable",
        }
    }

    fn reason_key(self) -> &'static str {
        match self {
            Self::AgentUnavailable => "goal-command-restart-recovery-reason-agent-unavailable",
            Self::ProviderUnavailable => {
                "goal-command-restart-recovery-reason-provider-unavailable"
            }
            Self::MissingContinuationContext => {
                "goal-command-restart-recovery-reason-missing-continuation"
            }
            Self::ContinuationContextReadFailed | Self::InvalidContinuationContext => {
                "goal-command-restart-recovery-reason-read-continuation-failed"
            }
            Self::MissingGoalExtension => {
                "goal-command-restart-recovery-reason-missing-goal-extension"
            }
            Self::GoalExtensionReadFailed => {
                "goal-command-restart-recovery-reason-read-goal-extension-failed"
            }
            Self::ChannelUnavailable | Self::ChannelOwnerUnavailable => {
                "goal-command-restart-recovery-reason-channel-unavailable"
            }
            Self::QueueUnavailable => "goal-command-restart-recovery-reason-queue-unavailable",
        }
    }
}

/// Build the persisted pause state for a recovered goal whose continuation
/// cannot be delivered.
pub fn recovered_goal_continuation_blocked_pause(
    blocker: RecoveredGoalContinuationBlocker,
) -> GoalPauseState {
    let reason = crate::i18n::get_required_cli_string(blocker.reason_key());
    GoalPauseState {
        reason: GoalPauseReason::DaemonRestart,
        description: Some(crate::i18n::get_required_cli_string(
            "goal-command-restart-recovery-paused-description",
        )),
        blockers: vec![GoalBlocker {
            kind: GoalBlockerKind::RestartRecovery,
            message: crate::i18n::get_required_cli_string_with_args(
                "goal-command-restart-recovery-blocker",
                &[("reason", &reason)],
            ),
            payload: Some(serde_json::json!({ "reason_code": blocker.code() })),
        }],
    }
}

/// Ownership guard for a destructively drained transient recovery batch.
///
/// Consumers call [`Self::commit`] only after every ID has reached a safe
/// continuation or durable non-running state. Any early return, cancellation,
/// unwinding panic, or task abort restores the entire ordered batch for retry.
/// An aborting process relies on normal prior-boot recovery at its next start.
pub struct RecoveredGoalBatch<'a> {
    control_plane: &'a ControlPlaneHandle,
    task_ids: Vec<String>,
    committed: bool,
}

/// Per-task ownership guard for a recovered continuation after queue delivery.
///
/// The guard follows the message through channel admission and execution.
/// Dropping any uncommitted lease puts the task id back into the transient
/// recovery queue for the next safe handoff.
pub struct RecoveredGoalLease {
    recovered_goal_ids: Arc<Mutex<Vec<String>>>,
    task_id: Option<String>,
}

impl RecoveredGoalLease {
    /// Start retaining one recovered task until the consumer commits it.
    pub fn new(control_plane: &ControlPlaneHandle, task_id: String) -> Self {
        Self {
            recovered_goal_ids: Arc::clone(&control_plane.recovered_goal_ids),
            task_id: Some(task_id),
        }
    }

    /// Canonical task id owned by this lease.
    pub fn task_id(&self) -> &str {
        // `commit` consumes the lease, so every still-borrowable lease owns an ID.
        self.task_id
            .as_deref()
            .expect("uncommitted recovery lease must own a task id")
    }

    /// Release recovery ownership after the durable task is no longer running.
    pub fn commit(mut self) {
        self.task_id = None;
    }
}

impl Drop for RecoveredGoalLease {
    fn drop(&mut self) {
        if let Some(task_id) = self.task_id.take() {
            restore_recovered_goal_ids(&self.recovered_goal_ids, &[task_id]);
        }
    }
}

impl<'a> RecoveredGoalBatch<'a> {
    /// Drain the current transient recovery queue under restoration ownership.
    pub fn take(control_plane: &'a ControlPlaneHandle) -> Self {
        Self {
            control_plane,
            task_ids: control_plane.take_recovered_goal_ids(),
            committed: false,
        }
    }

    /// Ordered task IDs owned by this batch.
    pub fn task_ids(&self) -> &[String] {
        &self.task_ids
    }

    /// Transfer one queued task into a guard that survives channel enqueue.
    pub fn lease(&self, task_id: &str) -> RecoveredGoalLease {
        debug_assert!(self.task_ids.iter().any(|queued| queued == task_id));
        RecoveredGoalLease::new(self.control_plane, task_id.to_string())
    }

    /// Mark batch ownership handled or transferred to per-task leases.
    ///
    /// Dropping the batch then does not restore its complete ID list; any
    /// outstanding [`RecoveredGoalLease`] remains responsible for its task.
    pub fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for RecoveredGoalBatch<'_> {
    fn drop(&mut self) {
        if !self.committed {
            restore_recovered_goal_ids(
                &self.control_plane.recovered_goal_ids,
                self.task_ids.as_slice(),
            );
        }
    }
}

impl ControlPlaneHandle {
    /// Drain goal IDs awaiting continuation after process boot or in-process
    /// reload recovery.
    ///
    /// This is a transient continuation work queue, not canonical lifecycle state.
    /// If the process crashes before the channel loop consumes it, the next boot
    /// will recover the goal again under its new `boot_id`.
    pub fn take_recovered_goal_ids(&self) -> Vec<String> {
        std::mem::take(
            &mut *self
                .recovered_goal_ids
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
        )
    }

    /// Queue current-boot goals that remain `Running` after an in-process daemon
    /// reload has stopped their executors.
    ///
    /// The durable task table remains the lifecycle source of truth. This only
    /// rebuilds the transient delivery queue consumed when channels start again.
    /// Callers must invoke it after channel workers have exited so a goal that
    /// completes during cooperative shutdown is not spuriously continued.
    pub(crate) async fn queue_running_goals_for_reload(&self) -> Result<usize> {
        let running_tasks = self.store.list_running().await?;

        let mut queue = self
            .recovered_goal_ids
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let previous_len = queue.len();
        for task in running_tasks {
            if task.kind == TaskKind::Goal
                && task.owner_boot_id == self.boot_id
                && !queue.contains(&task.id)
            {
                queue.push(task.id);
            }
        }
        Ok(queue.len() - previous_len)
    }

    /// Pause recovered goals when the next daemon iteration has no channel
    /// executor capable of consuming their transient continuation queue.
    pub async fn pause_recovered_goals_without_channel_executor(&self) -> Result<usize> {
        let recovered_goals = RecoveredGoalBatch::take(self);
        let mut paused = 0;

        for task_id in recovered_goals.task_ids() {
            let task = self
                .store
                .get(task_id)
                .await
                .with_context(|| format!("read recovered goal {task_id} before pausing"))?;
            let Some(task) = task else {
                continue;
            };
            if task.kind != TaskKind::Goal || task.status != TaskStatus::Running {
                continue;
            }

            let pause = recovered_goal_continuation_blocked_pause(
                RecoveredGoalContinuationBlocker::ChannelUnavailable,
            );
            let did_pause = self
                .goal_store
                .pause_running_goal_task(task_id, pause)
                .await
                .with_context(|| {
                    format!("pause recovered goal {task_id} without channel executor")
                })?;
            paused += usize::from(did_pause);
        }

        recovered_goals.commit();
        Ok(paused)
    }
}

fn restore_recovered_goal_ids(queue: &Mutex<Vec<String>>, task_ids: &[String]) {
    let mut queue = queue.lock().unwrap_or_else(|error| error.into_inner());
    for task_id in task_ids {
        if !queue.contains(task_id) {
            queue.push(task_id.clone());
        }
    }
}
