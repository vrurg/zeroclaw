//! Durable Goal fencing for an executable tool batch.
//!
//! The marker is intentionally narrower than tool execution itself: ordinary
//! tools retain their existing parallel behavior.  It only records that a
//! Goal-owned assistant tool-use record is between dispatch and its complete
//! paired history append, which makes a crash or forced interruption
//! non-resumable instead of guessing which effects occurred.

use std::future::Future;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::control_plane::{GoalTaskRegistry, GoalTransitionResult};
use crate::goal_mode::GoalExecutionScope;

tokio::task_local! {
    static GOAL_TOOL_PAIRING: Option<Arc<GoalToolPairing>>;
}

struct GoalToolPairing {
    registry: Arc<dyn GoalTaskRegistry>,
    scope: GoalExecutionScope,
    active: Mutex<Option<ActiveGoalToolBatch>>,
}

/// One durable marker spans a nested foreground parent/child tool tree. This
/// deliberately does not serialize unrelated ordinary tool calls, nor does it
/// act as a child-admission fence: it only preserves the history-pairing
/// boundary for the exact Goal.
struct ActiveGoalToolBatch {
    batch_id: String,
    nested_loops: usize,
}

/// Runs one Goal driver turn with its exact durable identity available to the
/// shared agentic tool loop. Nested foreground children inherit the scope; a
/// separate Goal execution never shares it.
pub(crate) async fn scope_goal_tool_pairing<F>(
    registry: Arc<dyn GoalTaskRegistry>,
    scope: GoalExecutionScope,
    future: F,
) -> F::Output
where
    F: Future,
{
    GOAL_TOOL_PAIRING
        .scope(
            Some(Arc::new(GoalToolPairing {
                registry,
                scope,
                active: Mutex::new(None),
            })),
            future,
        )
        .await
}

/// Admit a marker only when the prepared batch contains executable calls.
/// A `None` result means ordinary work, or a Goal turn with no dispatchable
/// calls (for example, a synthesized preparation refusal).
pub(crate) async fn admit_goal_tool_batch(
    executable_call_count: usize,
) -> Result<Option<GoalToolBatch>> {
    if executable_call_count == 0 {
        return Ok(None);
    }
    let Some(pairing) = GOAL_TOOL_PAIRING.try_with(Clone::clone).ok().flatten() else {
        return Ok(None);
    };
    let mut active = pairing.active.lock().await;
    if let Some(active) = active.as_mut() {
        active.nested_loops += 1;
        return Ok(Some(GoalToolBatch {
            pairing: Arc::clone(&pairing),
        }));
    }

    let batch_id = Uuid::new_v4().to_string();
    match pairing
        .registry
        .admit_pending_tool_batch(
            pairing.scope.task_id(),
            pairing.scope.session_id(),
            pairing.scope.execution_epoch(),
            &batch_id,
        )
        .await
        .context("admit Goal tool batch")?
    {
        GoalTransitionResult::Applied => {
            *active = Some(ActiveGoalToolBatch {
                batch_id,
                nested_loops: 1,
            });
            Ok(Some(GoalToolBatch {
                pairing: Arc::clone(&pairing),
            }))
        }
        GoalTransitionResult::Stale | GoalTransitionResult::Missing => {
            anyhow::bail!("Goal tool batch lost its execution fence")
        }
    }
}

/// A durable tool-pairing marker admitted before dispatch.
///
/// It has no `Drop` settlement by design: an error, cancellation, or crash
/// before a clean history append must leave evidence that the Goal cannot be
/// resumed safely.
pub(crate) struct GoalToolBatch {
    pairing: Arc<GoalToolPairing>,
}

impl GoalToolBatch {
    /// Clear this marker after the complete assistant/tool round has been
    /// appended. A pause may have fenced the Goal to a later epoch meanwhile;
    /// matching the admitted epoch prevents this settlement from clearing a
    /// successor batch.
    pub(crate) async fn settle(self) -> Result<()> {
        let mut active = self.pairing.active.lock().await;
        let active_batch = active
            .as_mut()
            .context("Goal tool batch settlement has no active durable marker")?;
        ensure!(
            active_batch.nested_loops > 0,
            "Goal tool batch settlement underflowed its nested-loop count"
        );
        active_batch.nested_loops -= 1;
        if active_batch.nested_loops > 0 {
            return Ok(());
        }
        let batch_id = active_batch.batch_id.clone();
        let result = self
            .pairing
            .registry
            .settle_pending_tool_batch(
                self.pairing.scope.task_id(),
                self.pairing.scope.session_id(),
                self.pairing.scope.execution_epoch(),
                &batch_id,
            )
            .await
            .context("settle Goal tool batch after history pairing")?;
        ensure!(
            result == GoalTransitionResult::Applied,
            "Goal tool batch settlement lost its durable pairing fence"
        );
        *active = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{admit_goal_tool_batch, scope_goal_tool_pairing};
    use crate::control_plane::{
        GoalTaskRecord, GoalTaskRegistry, GoalTransitionResult, SqliteTaskStore, TaskKind,
        TaskRecord, TaskStatus,
    };
    use crate::goal_mode::GoalExecutionScope;

    #[tokio::test]
    async fn goal_tool_batch_marker_clears_only_after_explicit_settlement() {
        let store = Arc::new(SqliteTaskStore::new_in_memory().expect("create store"));
        let task = TaskRecord {
            id: "goal-tool-pairing".to_owned(),
            kind: TaskKind::Goal,
            agent: "main".to_owned(),
            status: TaskStatus::Running,
            owner_pid: 1,
            owner_boot_id: "boot-a".to_owned(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: Some("matrix:alias:room".to_owned()),
            delivered: false,
            idem_key: None,
            principal_id: Some("matrix:@operator:example.test".to_owned()),
            session_id: Some("matrix-session".to_owned()),
            execution_epoch: 0,
            started_at: "now".to_owned(),
            finished_at: None,
        };
        assert_eq!(
            store
                .create_or_replace_session_goal(
                    task,
                    GoalTaskRecord {
                        task_id: "goal-tool-pairing".to_owned(),
                        objective: "finish safely".to_owned(),
                        ..GoalTaskRecord::default()
                    },
                )
                .await
                .expect("create Goal"),
            GoalTransitionResult::Applied
        );
        let scope = GoalExecutionScope::new("goal-tool-pairing", "matrix-session", 1)
            .expect("create exact Goal scope");
        let observed_store = Arc::clone(&store);
        let pending_store = Arc::clone(&observed_store);
        scope_goal_tool_pairing(store as Arc<dyn GoalTaskRegistry>, scope, async move {
            assert!(
                admit_goal_tool_batch(0)
                    .await
                    .expect("no-op preparation is allowed")
                    .is_none()
            );
            let parent_batch = admit_goal_tool_batch(1)
                .await
                .expect("admit executable batch")
                .expect("Goal scope persists a marker");
            let child_batch = admit_goal_tool_batch(1)
                .await
                .expect("nested child batch reuses the parent marker")
                .expect("Goal scope preserves a nested marker handle");
            let pending = pending_store
                .get_goal_task("goal-tool-pairing")
                .await
                .expect("read Goal extension")
                .expect("Goal extension exists");
            assert!(pending.pending_tool_batch_id.is_some());
            assert_eq!(pending.pending_tool_epoch, Some(1));
            child_batch
                .settle()
                .await
                .expect("settle nested child history pairing");
            let still_pending = pending_store
                .get_goal_task("goal-tool-pairing")
                .await
                .expect("read Goal extension")
                .expect("Goal extension exists");
            assert!(
                still_pending.pending_tool_batch_id.is_some(),
                "the parent history pairing still owns the durable marker"
            );
            parent_batch
                .settle()
                .await
                .expect("settle outer paired batch");
        })
        .await;
        let settled = observed_store
            .get_goal_task("goal-tool-pairing")
            .await
            .expect("read Goal extension")
            .expect("Goal extension exists");
        assert!(settled.pending_tool_batch_id.is_none());
        assert!(settled.pending_tool_epoch.is_none());
    }
}
