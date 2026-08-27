//! Process-local Goal executor cache attached to one canonical conversation.
//!
//! This is intentionally not a task, identity, or authorization store. The
//! durable control plane remains authoritative; the slot only keeps the one
//! runtime-owned executor alive while a session has Goal work to supervise.

use std::{future::Future, sync::Arc};

use anyhow::Result;
use tokio::sync::OnceCell;
use zeroclaw_runtime::goal_mode::execution::GoalExecutionDriver;

#[derive(Default)]
pub(crate) struct GoalRuntimeSlot {
    executor: OnceCell<Arc<GoalExecutionDriver>>,
}

impl GoalRuntimeSlot {
    /// Keep only worker supervision in the session slot. A fresh controller is
    /// built from the current runtime configuration for every command, while
    /// this driver preserves cancellation and quiescence ownership across
    /// those short-lived controller values.
    pub(crate) async fn get_or_try_init<F, Fut>(&self, init: F) -> Result<Arc<GoalExecutionDriver>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Arc<GoalExecutionDriver>>>,
    {
        self.executor.get_or_try_init(init).await.map(Arc::clone)
    }
}
