//! Process-local, per-conversation foreground coordination.
//!
//! This module intentionally depends only on Tokio and the standard library.
//! The channel orchestrator owns the canonical conversation key and decides
//! which transports participate; a future transport adapter can reuse this
//! lease without creating another session or authority store.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use tokio_util::sync::CancellationToken;

#[derive(Clone)]
pub(crate) struct ConversationLocks {
    persist: Arc<Mutex<()>>,
    foreground: Arc<tokio::sync::Mutex<()>>,
}

impl ConversationLocks {
    pub(crate) fn new() -> Self {
        Self {
            persist: Arc::new(Mutex::new(())),
            foreground: Arc::new(tokio::sync::Mutex::new(())),
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
