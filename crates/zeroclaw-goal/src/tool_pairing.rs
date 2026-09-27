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
        .context("settle Goal tool batch after parent-turn pairing")?;
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
