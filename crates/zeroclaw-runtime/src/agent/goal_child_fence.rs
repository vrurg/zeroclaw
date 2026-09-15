//! Process-local admission for Goal-owned foreground children.
//!
//! V1 permits one foreground child at a time. This state deliberately remains
//! local to an isolated Goal parent turn: a later concurrent-child supervisor
//! can replace its capacity without changing durable task or ledger state.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

tokio::task_local! {
    static GOAL_CHILD_FENCE: Option<Arc<GoalChildFence>>;
}

#[derive(Debug)]
pub(crate) struct GoalChildFence {
    permits: Arc<Semaphore>,
    accepts_children: bool,
}

impl GoalChildFence {
    pub(crate) fn open() -> Arc<Self> {
        Arc::new(Self {
            permits: Arc::new(Semaphore::new(1)),
            accepts_children: true,
        })
    }

    fn closed() -> Arc<Self> {
        Arc::new(Self {
            permits: Arc::new(Semaphore::new(0)),
            accepts_children: false,
        })
    }

    async fn acquire(self: &Arc<Self>) -> Result<GoalChildGuard> {
        anyhow::ensure!(
            self.accepts_children,
            "Goal mode V1 does not admit a nested foreground child"
        );
        Ok(GoalChildGuard {
            _permit: Arc::clone(&self.permits).acquire_owned().await?,
        })
    }
}

/// Held from Goal-child admission through inline child execution.
pub(crate) struct GoalChildGuard {
    _permit: OwnedSemaphorePermit,
}

/// Returns a guard only inside a Goal parent or child turn. Ordinary turns
/// retain their existing child-concurrency behavior.
pub(crate) async fn admit_goal_child() -> Result<Option<GoalChildGuard>> {
    let Some(fence) = GOAL_CHILD_FENCE.try_with(Clone::clone).ok().flatten() else {
        return Ok(None);
    };
    fence.acquire().await.map(Some)
}

/// Whether the current task is inside any isolated Goal turn.
pub(crate) fn is_goal_scoped() -> bool {
    GOAL_CHILD_FENCE
        .try_with(|fence| fence.is_some())
        .unwrap_or(false)
}

/// Installs an open child-admission scope for one isolated Goal parent turn.
pub(crate) async fn scope_goal_parent<F: Future>(future: F) -> F::Output {
    GOAL_CHILD_FENCE
        .scope(Some(GoalChildFence::open()), future)
        .await
}

/// Re-scopes an inline child so it cannot recursively create Goal-owned work.
/// The child still has no global tool fence: its ordinary tool batch remains
/// independently parallel according to normal agent configuration.
pub(crate) async fn scope_goal_child<'a, T>(
    future: Pin<Box<dyn Future<Output = T> + Send + 'a>>,
) -> T {
    if is_goal_scoped() {
        GOAL_CHILD_FENCE
            .scope(Some(GoalChildFence::closed()), future)
            .await
    } else {
        future.await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::{admit_goal_child, scope_goal_child, scope_goal_parent};

    #[tokio::test]
    async fn parent_scope_serializes_foreground_children() {
        scope_goal_parent(async {
            let active = Arc::new(AtomicUsize::new(0));
            let maximum = Arc::new(AtomicUsize::new(0));
            let first = {
                let active = Arc::clone(&active);
                let maximum = Arc::clone(&maximum);
                async move {
                    let _guard = admit_goal_child().await.unwrap().unwrap();
                    let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                    maximum.fetch_max(current, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    active.fetch_sub(1, Ordering::SeqCst);
                }
            };
            let second = {
                let active = Arc::clone(&active);
                let maximum = Arc::clone(&maximum);
                async move {
                    let _guard = admit_goal_child().await.unwrap().unwrap();
                    let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                    maximum.fetch_max(current, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    active.fetch_sub(1, Ordering::SeqCst);
                }
            };
            tokio::join!(first, second);
            assert_eq!(maximum.load(Ordering::SeqCst), 1);
        })
        .await;
    }

    #[tokio::test]
    async fn child_scope_rejects_recursive_goal_children() {
        scope_goal_parent(async {
            let _guard = admit_goal_child().await.unwrap().unwrap();
            scope_goal_child(Box::pin(async {
                let error = match admit_goal_child().await {
                    Ok(_) => panic!("nested Goal child admission must be rejected"),
                    Err(error) => error,
                };
                assert!(error.to_string().contains("nested foreground child"));
            }))
            .await;
        })
        .await;
    }

    #[tokio::test]
    async fn independent_parent_scopes_do_not_share_a_permit() {
        let first = scope_goal_parent(async {
            let _guard = admit_goal_child().await.unwrap().unwrap();
            tokio::time::sleep(Duration::from_millis(40)).await;
        });
        let second = scope_goal_parent(async {
            tokio::time::timeout(Duration::from_millis(20), async {
                let _guard = admit_goal_child().await.unwrap().unwrap();
            })
            .await
            .expect("a distinct parent fence must not wait for another parent");
        });
        tokio::join!(first, second);
    }
}
