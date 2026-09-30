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
use parking_lot::Mutex;
use uuid::Uuid;

use crate::control::GoalExecutionScope;
use crate::task::{GoalTaskRegistry, GoalTransitionResult};

/// Identifies only a store failure while settling a paired parent transcript.
/// Its display preserves the pre-existing error context; invariant failures
/// must not acquire this marker.
#[derive(Debug)]
pub(crate) struct GoalToolSettlementStoreContext;

impl std::fmt::Display for GoalToolSettlementStoreContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("settle Goal tool batch after parent-turn pairing")
    }
}

tokio::task_local! {
    static GOAL_TOOL_PAIRING: Arc<GoalToolPairing>;
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
    open_batches: usize,
    nested_pairing_failed: bool,
    settling: bool,
}

/// Runs one Goal driver turn with its exact durable identity available to the
/// shared agentic tool loop. Nested foreground children inherit the scope; a
/// separate Goal execution never shares it.
pub async fn scope_goal_tool_pairing<F>(
    registry: Arc<dyn GoalTaskRegistry>,
    scope: GoalExecutionScope,
    future: F,
) -> F::Output
where
    F: Future,
{
    GOAL_TOOL_PAIRING
        .scope(
            Arc::new(GoalToolPairing {
                registry,
                scope,
                active: Mutex::new(None),
            }),
            future,
        )
        .await
}

/// Admit a marker only when the prepared batch contains executable calls.
/// A `None` result means ordinary work, or a Goal turn with no dispatchable
/// calls (for example, a synthesized preparation refusal).
pub async fn admit_goal_tool_batch(executable_call_count: usize) -> Result<Option<GoalToolBatch>> {
    if executable_call_count == 0 {
        return Ok(None);
    }
    let Some(pairing) = GOAL_TOOL_PAIRING.try_with(Clone::clone).ok() else {
        return Ok(None);
    };
    let nested = {
        let mut active = pairing.active.lock();
        if let Some(active) = active.as_mut() {
            ensure!(
                !active.settling,
                "Goal tool batch admitted while its prior history pairing settles"
            );
            active.open_batches += 1;
            true
        } else {
            false
        }
    };
    if nested {
        return Ok(Some(GoalToolBatch {
            pairing,
            settled: false,
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
            {
                let mut active = pairing.active.lock();
                ensure!(
                    active.is_none(),
                    "Goal tool batch acquired a nested owner during durable admission"
                );
                *active = Some(ActiveGoalToolBatch {
                    batch_id,
                    open_batches: 1,
                    nested_pairing_failed: false,
                    settling: false,
                });
            }
            Ok(Some(GoalToolBatch {
                pairing,
                settled: false,
            }))
        }
        GoalTransitionResult::Stale | GoalTransitionResult::Missing => {
            anyhow::bail!("Goal tool batch lost its execution fence")
        }
    }
}

/// A durable tool-pairing marker admitted before dispatch.
///
/// Dropping an unsettled handle never clears the durable marker. It instead
/// records that the outer phase is unpaired, so its eventual settlement fails
/// closed and the Goal engine terminalizes the exact Goal.
pub struct GoalToolBatch {
    pairing: Arc<GoalToolPairing>,
    settled: bool,
}

impl GoalToolBatch {
    /// Mark one assistant/tool round as paired in the isolated transcript.
    ///
    /// The durable marker remains until the enclosing parent turn returns its
    /// complete working transcript to the Goal executor. That one boundary
    /// covers every sequential or nested round without clearing crash evidence
    /// between an external tool effect and the parent-turn result.
    pub fn settle(mut self) -> Result<()> {
        let mut active = self.pairing.active.lock();
        let active_batch = active
            .as_mut()
            .context("Goal tool batch settlement has no active durable marker")?;
        ensure!(
            active_batch.open_batches > 0,
            "Goal tool batch settlement underflowed its open-batch count"
        );
        active_batch.open_batches -= 1;
        self.settled = true;
        ensure!(
            !active_batch.nested_pairing_failed,
            "Goal nested tool loop exited before clean history pairing"
        );
        Ok(())
    }
}

impl Drop for GoalToolBatch {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        let mut active = self.pairing.active.lock();
        let Some(active_batch) = active.as_mut() else {
            return;
        };
        if active_batch.open_batches == 0 {
            return;
        }
        active_batch.open_batches -= 1;
        active_batch.nested_pairing_failed = true;
    }
}

/// Clear a durable marker only after the complete isolated parent turn has
/// returned its paired working transcript to the Goal executor.
pub async fn finalize_goal_tool_pairing() -> Result<()> {
    let Some(pairing) = GOAL_TOOL_PAIRING.try_with(Clone::clone).ok() else {
        return Ok(());
    };
    let batch_id = {
        let mut active = pairing.active.lock();
        let Some(active_batch) = active.as_mut() else {
            return Ok(());
        };
        ensure!(
            active_batch.open_batches == 0,
            "Goal parent turn returned with an unfinished tool batch"
        );
        ensure!(
            !active_batch.nested_pairing_failed,
            "Goal parent turn returned with an unpaired tool batch"
        );
        ensure!(
            !active_batch.settling,
            "Goal parent turn attempted duplicate tool-pairing settlement"
        );
        active_batch.settling = true;
        active_batch.batch_id.clone()
    };
    let result = pairing
        .registry
        .settle_pending_tool_batch(
            pairing.scope.task_id(),
            pairing.scope.session_id(),
            pairing.scope.execution_epoch(),
            &batch_id,
        )
        .await
        .context(GoalToolSettlementStoreContext)?;
    ensure!(
        result == GoalTransitionResult::Applied,
        "Goal tool batch settlement lost its durable pairing fence"
    );
    let mut active = pairing.active.lock();
    let active_batch = active
        .as_ref()
        .context("Goal tool batch disappeared while settlement completed")?;
    ensure!(
        active_batch.settling
            && active_batch.open_batches == 0
            && active_batch.batch_id == batch_id,
        "Goal tool batch changed while settlement completed"
    );
    *active = None;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use zeroclaw_api::task::{TaskRecord, TaskStatus};

    use super::*;
    use crate::task::{
        GoalAccountingState, GoalPauseState, GoalPolicyTarget, GoalTaskRecord,
        TaskContinuationContext,
    };

    #[derive(Debug)]
    struct SettlementStoreFailure;

    impl std::fmt::Display for SettlementStoreFailure {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("fixture settlement store failure")
        }
    }

    impl std::error::Error for SettlementStoreFailure {}

    #[derive(Default)]
    struct SettlementRegistry {
        admitted_batch: Mutex<Option<String>>,
        settlement_calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl GoalTaskRegistry for SettlementRegistry {
        async fn latest_active_goal_for_agent(&self, _: &str) -> Result<Option<TaskRecord>> {
            panic!("unexpected registry lookup")
        }

        async fn latest_active_goal_for_context(
            &self,
            _: &str,
            _: Option<&str>,
            _: Option<&str>,
        ) -> Result<Option<TaskRecord>> {
            panic!("unexpected registry lookup")
        }

        async fn latest_active_goal_id_for_context(
            &self,
            _: &str,
            _: Option<&str>,
            _: Option<&str>,
        ) -> Result<Option<String>> {
            panic!("unexpected registry lookup")
        }

        async fn get_goal_task(&self, _: &str) -> Result<Option<GoalTaskRecord>> {
            panic!("unexpected registry lookup")
        }

        async fn update_goal_limits(&self, _: &str, _: Option<u64>, _: Option<f64>) -> Result<()> {
            panic!("unexpected budget mutation")
        }

        async fn update_goal_pause(&self, _: &str, _: Option<GoalPauseState>) -> Result<()> {
            panic!("unexpected pause mutation")
        }

        async fn set_continuation_context(
            &self,
            _: &str,
            _: Option<TaskContinuationContext>,
        ) -> Result<()> {
            panic!("unexpected continuation mutation")
        }

        async fn get_continuation_context(
            &self,
            _: &str,
        ) -> Result<Option<TaskContinuationContext>> {
            panic!("unexpected continuation lookup")
        }

        async fn current_goal_for_session(&self, _: &str) -> Result<Option<TaskRecord>> {
            panic!("unexpected session lookup")
        }

        async fn list_nonterminal_session_goals(&self) -> Result<Vec<TaskRecord>> {
            panic!("unexpected policy lookup")
        }

        async fn cancel_policy_targets(
            &self,
            _: &[GoalPolicyTarget],
        ) -> Result<GoalTransitionResult> {
            panic!("unexpected policy revocation")
        }

        async fn terminal_reason_for_session_goal(
            &self,
            _: &str,
            _: &str,
        ) -> Result<Option<String>> {
            panic!("unexpected terminal lookup")
        }

        async fn create_or_replace_session_goal(
            &self,
            _: TaskRecord,
            _: GoalTaskRecord,
        ) -> Result<GoalTransitionResult> {
            panic!("unexpected creation")
        }

        async fn pause_session_goal(
            &self,
            _: &str,
            _: &str,
            _: i64,
            _: GoalPauseState,
        ) -> Result<GoalTransitionResult> {
            panic!("unexpected pause")
        }

        async fn resume_session_goal(
            &self,
            _: &str,
            _: &str,
            _: i64,
            _: u32,
            _: &str,
        ) -> Result<GoalTransitionResult> {
            panic!("unexpected resume")
        }

        async fn finish_session_goal(
            &self,
            _: &str,
            _: &str,
            _: i64,
            _: TaskStatus,
            _: Option<String>,
        ) -> Result<GoalTransitionResult> {
            panic!("unexpected lifecycle transition")
        }

        async fn admit_pending_operation(
            &self,
            _: &str,
            _: &str,
            _: i64,
            _: &str,
        ) -> Result<GoalTransitionResult> {
            panic!("unexpected model admission")
        }

        async fn settle_pending_operation(
            &self,
            _: &str,
            _: &str,
            _: i64,
            _: &str,
            _: GoalAccountingState,
        ) -> Result<GoalTransitionResult> {
            panic!("unexpected model settlement")
        }

        async fn admit_pending_tool_batch(
            &self,
            task: &str,
            session: &str,
            epoch: i64,
            batch: &str,
        ) -> Result<GoalTransitionResult> {
            assert_eq!(
                (task, session, epoch),
                ("fixture-task", "fixture-session", 1)
            );
            assert!(self.admitted_batch.lock().replace(batch.into()).is_none());
            Ok(GoalTransitionResult::Applied)
        }

        async fn settle_pending_tool_batch(
            &self,
            task: &str,
            session: &str,
            epoch: i64,
            batch: &str,
        ) -> Result<GoalTransitionResult> {
            assert_eq!(
                (task, session, epoch),
                ("fixture-task", "fixture-session", 1)
            );
            assert_eq!(self.admitted_batch.lock().as_deref(), Some(batch));
            assert_eq!(self.settlement_calls.fetch_add(1, Ordering::SeqCst), 0);
            Err(SettlementStoreFailure.into())
        }

        async fn update_session_goal_limits(
            &self,
            _: &str,
            _: &str,
            _: i64,
            _: Option<u64>,
            _: Option<f64>,
        ) -> Result<GoalTransitionResult> {
            panic!("unexpected budget mutation")
        }

        async fn fail_unpaired_tool_batch(
            &self,
            _: &str,
            _: &str,
            _: i64,
            _: i64,
            _: &str,
        ) -> Result<GoalTransitionResult> {
            panic!("unexpected unpaired lifecycle transition")
        }

        async fn clear_terminal_tool_batch(
            &self,
            _: &str,
            _: &str,
            _: i64,
            _: &str,
        ) -> Result<GoalTransitionResult> {
            panic!("unexpected terminal batch clearing")
        }

        async fn delete_session_goal(
            &self,
            _: &str,
            _: &str,
            _: i64,
        ) -> Result<GoalTransitionResult> {
            panic!("unexpected deletion")
        }
    }

    fn scope() -> GoalExecutionScope {
        GoalExecutionScope::new("fixture-task", "fixture-session", 1).unwrap()
    }

    #[tokio::test]
    async fn settlement_store_error_keeps_chain_and_private_marker() {
        let registry = Arc::new(SettlementRegistry::default());
        let error = scope_goal_tool_pairing(registry.clone(), scope(), async {
            admit_goal_tool_batch(1)
                .await
                .unwrap()
                .unwrap()
                .settle()
                .unwrap();
            finalize_goal_tool_pairing().await.unwrap_err()
        })
        .await;
        assert_eq!(registry.settlement_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            error.to_string(),
            "settle Goal tool batch after parent-turn pairing"
        );
        assert_eq!(
            format!("{error:#}"),
            "settle Goal tool batch after parent-turn pairing: fixture settlement store failure"
        );
        assert!(error.downcast_ref::<SettlementStoreFailure>().is_some());
        assert!(
            error
                .downcast_ref::<GoalToolSettlementStoreContext>()
                .is_some()
        );
    }

    #[tokio::test]
    async fn unfinished_batch_is_an_unmarked_invariant_without_store_settlement() {
        let registry = Arc::new(SettlementRegistry::default());
        let error = scope_goal_tool_pairing(registry.clone(), scope(), async {
            let batch = admit_goal_tool_batch(1).await.unwrap().unwrap();
            let error = finalize_goal_tool_pairing().await.unwrap_err();
            drop(batch);
            error
        })
        .await;
        assert_eq!(
            error.to_string(),
            "Goal parent turn returned with an unfinished tool batch"
        );
        assert!(
            error
                .downcast_ref::<GoalToolSettlementStoreContext>()
                .is_none()
        );
        assert_eq!(registry.settlement_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn abandoned_batch_is_an_unmarked_invariant_without_store_settlement() {
        let registry = Arc::new(SettlementRegistry::default());
        let error = scope_goal_tool_pairing(registry.clone(), scope(), async {
            drop(admit_goal_tool_batch(1).await.unwrap().unwrap());
            finalize_goal_tool_pairing().await.unwrap_err()
        })
        .await;
        assert_eq!(
            error.to_string(),
            "Goal parent turn returned with an unpaired tool batch"
        );
        assert!(
            error
                .downcast_ref::<GoalToolSettlementStoreContext>()
                .is_none()
        );
        assert_eq!(registry.settlement_calls.load(Ordering::SeqCst), 0);
    }
}
