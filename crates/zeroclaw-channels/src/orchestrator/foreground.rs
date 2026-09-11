//! Process-local, per-conversation foreground coordination.
//!
//! The channel orchestrator owns the canonical conversation key and chooses
//! which channel paths acquire the foreground lease. This module holds no
//! session, authority, or durable Goal state.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use tokio_util::sync::CancellationToken;
use zeroclaw_runtime::goal_mode::{GoalExecutionScope, GoalExecutionSupervisor};

#[derive(Clone)]
pub(crate) struct ConversationLocks {
    persist: Arc<Mutex<()>>,
    goal_command: Arc<tokio::sync::Mutex<()>>,
    foreground: Arc<tokio::sync::Mutex<()>>,
    goal_supervisor: Arc<tokio::sync::Mutex<Option<Arc<GoalExecutionSupervisor>>>>,
}

impl ConversationLocks {
    pub(crate) fn new() -> Self {
        Self {
            persist: Arc::new(Mutex::new(())),
            goal_command: Arc::new(tokio::sync::Mutex::new(())),
            foreground: Arc::new(tokio::sync::Mutex::new(())),
            goal_supervisor: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }
}

fn locks_for_key(
    locks: &Arc<Mutex<HashMap<String, ConversationLocks>>>,
    key: &str,
) -> ConversationLocks {
    let mut locks = locks.lock().unwrap_or_else(|error| error.into_inner());
    locks
        .entry(key.to_owned())
        .or_insert_with(ConversationLocks::new)
        .clone()
}

pub(crate) fn persist_lock(
    locks: &Arc<Mutex<HashMap<String, ConversationLocks>>>,
    key: &str,
) -> Arc<Mutex<()>> {
    locks_for_key(locks, key).persist
}

pub(crate) fn foreground_lock(
    locks: &Arc<Mutex<HashMap<String, ConversationLocks>>>,
    key: &str,
) -> Arc<tokio::sync::Mutex<()>> {
    locks_for_key(locks, key).foreground
}

pub(crate) fn goal_command_lock(
    locks: &Arc<Mutex<HashMap<String, ConversationLocks>>>,
    key: &str,
) -> Arc<tokio::sync::Mutex<()>> {
    locks_for_key(locks, key).goal_command
}

pub(crate) fn goal_supervisor_slot(
    locks: &Arc<Mutex<HashMap<String, ConversationLocks>>>,
    key: &str,
) -> Arc<tokio::sync::Mutex<Option<Arc<GoalExecutionSupervisor>>>> {
    locks_for_key(locks, key).goal_supervisor
}

/// Pause and drain every Goal worker registered by one channel runtime before
/// that runtime is replaced. The slots are process-local ownership handles,
/// not a second source of Goal lifecycle truth.
pub(crate) async fn pause_goal_supervisors_for_runtime_interruption(
    locks: &Arc<Mutex<HashMap<String, ConversationLocks>>>,
) -> Result<()> {
    let slots = {
        let locks = locks.lock().unwrap_or_else(|error| error.into_inner());
        locks
            .values()
            .map(|conversation| Arc::clone(&conversation.goal_supervisor))
            .collect::<Vec<_>>()
    };
    let mut supervisors = Vec::new();
    for slot in slots {
        if let Some(supervisor) = slot.lock().await.as_ref().cloned()
            && !supervisors
                .iter()
                .any(|existing: &Arc<GoalExecutionSupervisor>| Arc::ptr_eq(existing, &supervisor))
        {
            supervisors.push(supervisor);
        }
    }
    for supervisor in supervisors {
        supervisor.pause_all_for_runtime_interruption().await?;
    }
    Ok(())
}

pub(crate) async fn drain_policy_revoked_goal(
    locks: &Arc<Mutex<HashMap<String, ConversationLocks>>>,
    scope: &GoalExecutionScope,
) -> Result<()> {
    let mut slot = goal_supervisor_slot(locks, scope.session_id())
        .lock_owned()
        .await;
    let Some(supervisor) = slot.as_ref().cloned() else {
        return Ok(());
    };
    if supervisor.owns_scope(scope).await {
        supervisor.drain_policy_revoked(scope).await?;
        if !supervisor.owns_scope(scope).await
            && slot
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, &supervisor))
        {
            *slot = None;
        }
    }
    Ok(())
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
