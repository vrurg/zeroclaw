//! Process-local, per-conversation foreground coordination.
//!
//! The channel orchestrator owns the canonical conversation key and chooses
//! which channel paths acquire the foreground lease. This module holds no
//! session, authority, or durable Goal state.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use tokio_util::sync::CancellationToken;
use zeroclaw_runtime::goal_mode::GoalExecutionSupervisor;

pub(crate) struct ConversationLocks {
    persist: Arc<Mutex<()>>,
    matrix_goal: OnceLock<MatrixGoalLocks>,
}

/// Locks required only after a Matrix conversation enters the Goal control
/// path. Ordinary channel traffic retains the historical persistence lock
/// without paying for three unused async mutex allocations per conversation.
struct MatrixGoalLocks {
    goal_command: Arc<tokio::sync::Mutex<()>>,
    foreground: Arc<tokio::sync::Mutex<()>>,
    goal_supervisor: Arc<tokio::sync::Mutex<Option<Arc<GoalExecutionSupervisor>>>>,
}

impl ConversationLocks {
    pub(crate) fn new() -> Self {
        Self {
            persist: Arc::new(Mutex::new(())),
            matrix_goal: OnceLock::new(),
        }
    }

    fn matrix_goal(&self) -> &MatrixGoalLocks {
        self.matrix_goal.get_or_init(|| MatrixGoalLocks {
            goal_command: Arc::new(tokio::sync::Mutex::new(())),
            foreground: Arc::new(tokio::sync::Mutex::new(())),
            goal_supervisor: Arc::new(tokio::sync::Mutex::new(None)),
        })
    }

    #[cfg(test)]
    fn has_matrix_goal_locks(&self) -> bool {
        self.matrix_goal.get().is_some()
    }
}

fn locks_for_key(
    locks: &Arc<Mutex<HashMap<String, Arc<ConversationLocks>>>>,
    key: &str,
) -> Arc<ConversationLocks> {
    let mut locks = locks.lock().unwrap_or_else(|error| error.into_inner());
    locks
        .entry(key.to_owned())
        .or_insert_with(|| Arc::new(ConversationLocks::new()))
        .clone()
}

pub(crate) fn persist_lock(
    locks: &Arc<Mutex<HashMap<String, Arc<ConversationLocks>>>>,
    key: &str,
) -> Arc<Mutex<()>> {
    Arc::clone(&locks_for_key(locks, key).persist)
}

pub(crate) fn foreground_lock(
    locks: &Arc<Mutex<HashMap<String, Arc<ConversationLocks>>>>,
    key: &str,
) -> Arc<tokio::sync::Mutex<()>> {
    Arc::clone(&locks_for_key(locks, key).matrix_goal().foreground)
}

pub(crate) fn goal_command_lock(
    locks: &Arc<Mutex<HashMap<String, Arc<ConversationLocks>>>>,
    key: &str,
) -> Arc<tokio::sync::Mutex<()>> {
    Arc::clone(&locks_for_key(locks, key).matrix_goal().goal_command)
}

pub(crate) fn goal_supervisor_slot(
    locks: &Arc<Mutex<HashMap<String, Arc<ConversationLocks>>>>,
    key: &str,
) -> Arc<tokio::sync::Mutex<Option<Arc<GoalExecutionSupervisor>>>> {
    Arc::clone(&locks_for_key(locks, key).matrix_goal().goal_supervisor)
}

pub(crate) async fn wait_for_foreground_lease(
    lock: Arc<tokio::sync::Mutex<()>>,
    cancellation: CancellationToken,
) -> Option<tokio::sync::OwnedMutexGuard<()>> {
    tokio::select! {
        guard = lock.lock_owned() => Some(guard),
        () = cancellation.cancelled() => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn foreground_lease_serializes_one_conversation_only() {
        let locks = Arc::new(Mutex::new(HashMap::new()));
        let first = foreground_lock(&locks, "matrix:room:thread");
        let same = foreground_lock(&locks, "matrix:room:thread");
        let other = foreground_lock(&locks, "matrix:other-room:thread");
        assert!(Arc::ptr_eq(&first, &same));
        assert!(!Arc::ptr_eq(&first, &other));

        let held = first.lock_owned().await;
        let mut same_waiter = zeroclaw_spawn::spawn!(async move { same.lock_owned().await });
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), &mut same_waiter)
                .await
                .is_err()
        );
        let other_guard =
            tokio::time::timeout(std::time::Duration::from_millis(50), other.lock_owned())
                .await
                .expect("independent conversations must not wait");
        drop(other_guard);
        drop(held);
        tokio::time::timeout(std::time::Duration::from_millis(500), same_waiter)
            .await
            .expect("same conversation must proceed after release")
            .expect("foreground waiter must not panic");
    }

    #[test]
    fn persistence_only_conversation_does_not_allocate_matrix_goal_locks() {
        let locks = Arc::new(Mutex::new(HashMap::new()));
        let _persist = persist_lock(&locks, "channel:ordinary");
        let entry = locks
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get("channel:ordinary")
            .expect("persistence lookup creates the conversation entry")
            .clone();
        assert!(!entry.has_matrix_goal_locks());

        let _foreground = foreground_lock(&locks, "channel:ordinary");
        assert!(entry.has_matrix_goal_locks());
    }

    #[tokio::test]
    async fn cancelled_foreground_wait_never_acquires_a_lease() {
        let locks = Arc::new(Mutex::new(HashMap::new()));
        let lock = foreground_lock(&locks, "matrix:room:thread");
        let held = lock.clone().lock_owned().await;
        let cancellation = CancellationToken::new();
        let waiter_cancellation = cancellation.clone();
        let waiter = zeroclaw_spawn::spawn!(wait_for_foreground_lease(lock, waiter_cancellation));
        cancellation.cancel();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(500), waiter)
                .await
                .expect("cancelled waiter must complete")
                .expect("foreground waiter must not panic")
                .is_none()
        );
        drop(held);
    }
}
