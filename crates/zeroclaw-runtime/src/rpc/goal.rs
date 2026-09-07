//! Process-local ZeroCode ownership for Goal execution workers.
//!
//! Durable lifecycle and accounting state remain in the control plane. This
//! registry only retains the exact join handles needed to drain a session's
//! fenced execution epoch when an RPC lifecycle operation replaces or removes
//! that session.

use std::{collections::HashMap, sync::Arc};

use tokio::sync::Mutex;

use crate::goal_mode::GoalExecutionSupervisor;

/// ZeroCode's process-local Goal worker and command-lock registry.
#[derive(Default)]
pub struct RpcGoalRuntime {
    supervisors: Mutex<HashMap<String, Arc<GoalExecutionSupervisor>>>,
    command_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl RpcGoalRuntime {
    /// Return the command lease for one raw RPC session identifier.
    pub async fn command_lock(&self, session_id: &str) -> Arc<Mutex<()>> {
        let mut locks = self.command_locks.lock().await;
        Arc::clone(
            locks
                .entry(session_id.to_owned())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        )
    }

    /// Return a resident supervisor for this session, if it has one.
    pub async fn supervisor(&self, session_id: &str) -> Option<Arc<GoalExecutionSupervisor>> {
        self.supervisors.lock().await.get(session_id).cloned()
    }

    /// Install the sole supervisor for one session.
    pub async fn install_supervisor(
        &self,
        session_id: String,
        supervisor: Arc<GoalExecutionSupervisor>,
    ) {
        self.supervisors.lock().await.insert(session_id, supervisor);
    }

    /// Remove the resident supervisor after its worker has drained.
    pub async fn remove_supervisor(&self, session_id: &str) {
        self.supervisors.lock().await.remove(session_id);
    }
}
