//! Process-local, per-conversation foreground coordination.
//!
//! The channel orchestrator owns the canonical conversation key and decides
//! which transports participate. The optional Goal runtime slot is a
//! transport-agnostic executor cache, not another session or authority store;
//! a future transport adapter can reuse the same lease and slot contract.

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
};

use tokio_util::sync::CancellationToken;

use super::goal_runtime_slot::GoalRuntimeSlot;

// This lock carries no Goal lifecycle, identity, policy, or accounting fact.
// It only serializes one daemon process's Matrix Goal admission and config
// cutover across every agent runtime context. Durable state remains in the
// control plane, and separate daemon processes rely on guarded SQLite writes.
static GOAL_POLICY_CUTOVER_LOCK: LazyLock<tokio::sync::Mutex<()>> =
    LazyLock::new(|| tokio::sync::Mutex::new(()));

#[derive(Clone)]
pub(crate) struct ConversationLocks {
    persist: Arc<Mutex<()>>,
    foreground: Arc<tokio::sync::Mutex<()>>,
    goal_runtime: Arc<GoalRuntimeSlot>,
}

impl ConversationLocks {
    pub(crate) fn new() -> Self {
        Self {
            persist: Arc::new(Mutex::new(())),
            foreground: Arc::new(tokio::sync::Mutex::new(())),
            goal_runtime: Arc::new(GoalRuntimeSlot::default()),
        }
    }
}

fn locks_for_key(
    locks: &Arc<Mutex<HashMap<String, ConversationLocks>>>,
    key: &str,
) -> ConversationLocks {
    let mut locks = locks.lock().unwrap_or_else(|error| error.into_inner());
    locks
        .entry(key.to_string())
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

pub(crate) fn goal_runtime_slot(
    locks: &Arc<Mutex<HashMap<String, ConversationLocks>>>,
    key: &str,
) -> Arc<GoalRuntimeSlot> {
    locks_for_key(locks, key).goal_runtime
}

/// One daemon-wide Matrix Goal admission/cutover gate. This is process-local
/// coordination only; durable Goal lifecycle remains in the control plane.
pub(crate) fn goal_policy_cutover_lock() -> &'static tokio::sync::Mutex<()> {
    &GOAL_POLICY_CUTOVER_LOCK
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
    async fn foreground_lease_serializes_one_key_only() {
        let locks = Arc::new(Mutex::new(HashMap::new()));
        let first = foreground_lock(&locks, "matrix:room:thread");
        let same_conversation = foreground_lock(&locks, "matrix:room:thread");
        let other_conversation = foreground_lock(&locks, "matrix:other-room:thread");

        assert!(Arc::ptr_eq(&first, &same_conversation));
        assert!(!Arc::ptr_eq(&first, &other_conversation));

        let held = first.lock_owned().await;
        let mut same_waiter =
            zeroclaw_spawn::spawn!(async move { same_conversation.lock_owned().await });
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), &mut same_waiter)
                .await
                .is_err(),
            "the same conversation must wait for its foreground lease"
        );

        let other_guard = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            other_conversation.lock_owned(),
        )
        .await
        .expect("an independent conversation must not wait");
        drop(other_guard);
        drop(held);

        tokio::time::timeout(std::time::Duration::from_millis(500), same_waiter)
            .await
            .expect("the queued conversation must proceed after release")
            .expect("lease waiter task must not panic");
    }

    #[test]
    fn goal_runtime_slot_is_scoped_to_the_canonical_conversation() {
        let locks = Arc::new(Mutex::new(HashMap::new()));
        let first = goal_runtime_slot(&locks, "matrix:room:thread");
        let same_conversation = goal_runtime_slot(&locks, "matrix:room:thread");
        let other_conversation = goal_runtime_slot(&locks, "matrix:other-room:thread");

        assert!(Arc::ptr_eq(&first, &same_conversation));
        assert!(!Arc::ptr_eq(&first, &other_conversation));
    }

    #[tokio::test]
    async fn goal_policy_cutover_gate_is_shared_by_independent_context_maps() {
        let first_map = Arc::new(Mutex::new(HashMap::new()));
        let second_map = Arc::new(Mutex::new(HashMap::new()));
        let first = goal_policy_cutover_lock();
        let second = goal_policy_cutover_lock();

        let held = first.lock().await;
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), second.lock())
                .await
                .is_err(),
            "independent agent context maps must share the daemon Goal cutover gate"
        );
        drop(held);

        // The maps remain independent for conversation-scoped coordination.
        assert!(!Arc::ptr_eq(
            &foreground_lock(&first_map, "matrix:room"),
            &foreground_lock(&second_map, "matrix:room"),
        ));
    }

    #[tokio::test]
    async fn cancelled_foreground_wait_never_acquires_the_lease() {
        let locks = Arc::new(Mutex::new(HashMap::new()));
        let lock = foreground_lock(&locks, "matrix:room:thread");
        let held = lock.clone().lock_owned().await;
        let cancellation = CancellationToken::new();
        let waiter_cancellation = cancellation.clone();
        let waiter = zeroclaw_spawn::spawn!(wait_for_foreground_lease(lock, waiter_cancellation,));

        cancellation.cancel();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(500), waiter)
                .await
                .expect("cancelled foreground wait must complete")
                .expect("lease waiter task must not panic")
                .is_none(),
            "a cancelled queued request must not acquire the foreground lease"
        );
        drop(held);
    }
}
