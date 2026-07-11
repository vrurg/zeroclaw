//! Boot wiring for the control-plane — minted once per daemon run.
//!
//! [`ControlPlaneHandle`] bundles the durable [`TaskRegistry`], the goal task
//! extension store, and the run's `boot_id` (the authority key that distinguishes
//! this daemon's live tasks from prior-boot orphans). `DaemonRegistry` owns the
//! spawned reaper task's lifetime via its cancel.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::{fs::OpenOptions, io::Write};

use anyhow::{Context, Result};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use zeroclaw_config::schema::GoalRestartRecovery;

use super::goal_task::GoalTaskRegistry;
use super::reaper;
use super::task_registry::TaskRegistry;
use super::task_store_sqlite::SqliteTaskStore;

/// The live control-plane, shared (cheaply, via `Arc`/clone) across producers and
/// the reaper.
#[derive(Clone)]
pub struct ControlPlaneHandle {
    /// Generic task registry. Owns canonical lifecycle, route, principal, and
    /// ownership state for every durable task kind.
    pub store: Arc<dyn TaskRegistry>,
    /// Goal extension registry. Owns only goal-specific rows and continuation
    /// context keyed by the canonical task id.
    pub goal_store: Arc<dyn GoalTaskRegistry>,
    /// Current daemon owner id used by recovery/reaper authority checks.
    pub boot_id: String,
    /// Goal ids recovered during process boot or in-process reload that need
    /// channel continuation after channel handles are available.
    pub(crate) recovered_goal_ids: Arc<Mutex<Vec<String>>>,
    /// Process-wide data-dir ownership guard.
    ///
    /// This is not control-plane state. It keeps the OS advisory lock on
    /// `<data_dir>/control_plane.lock` alive so restart recovery cannot run in
    /// two daemon processes over the same durable store.
    pub(crate) data_dir_lock: Option<Arc<ControlPlaneDataDirLock>>,
}

impl ControlPlaneHandle {
    /// Open the durable store at `<data_dir>/control_plane.db`, mint a fresh
    /// `boot_id`, and run the one-shot crash-recovery sweep. Prior-boot non-goal
    /// `Running` tasks become `Lost`; prior-boot goals follow the configured
    /// restart recovery policy. Additive and fail-safe: a fresh install gets an
    /// empty DB.
    ///
    /// Single-writer invariant: recovery treats a different `boot_id` as proof
    /// the prior owner is gone, so startup first acquires the data-dir lock
    /// kept in [`ControlPlaneHandle::data_dir_lock`]. Without that lock two
    /// daemon processes could both recover/reap the same durable task table.
    pub async fn start(
        data_dir: &Path,
        goal_restart_recovery: GoalRestartRecovery,
    ) -> Result<Self> {
        let run_id = uuid::Uuid::new_v4().to_string();
        Self::start_with_boot_id(data_dir, run_id, goal_restart_recovery).await
    }

    /// As [`Self::start`] but with a caller-supplied `boot_id` — lets `DaemonRegistry`
    /// reuse a process-stable run-id across reloads instead of a fresh UUID.
    pub async fn start_with_boot_id(
        data_dir: &Path,
        boot_id: String,
        goal_restart_recovery: GoalRestartRecovery,
    ) -> Result<Self> {
        let data_dir_lock = Arc::new(ControlPlaneDataDirLock::acquire(data_dir)?);
        let sqlite_store = Arc::new(SqliteTaskStore::new(data_dir)?);
        let store: Arc<dyn TaskRegistry> = sqlite_store.clone();
        let goal_store: Arc<dyn GoalTaskRegistry> = sqlite_store;
        let recovery = reaper::recovery_pass(
            store.as_ref(),
            goal_store.as_ref(),
            &boot_id,
            goal_restart_recovery,
        )
        .await?;
        if recovery.recovered > 0 {
            ::zeroclaw_log::record!(
                INFO,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                    .with_attrs(::serde_json::json!({
                        "recovered": recovery.recovered,
                        "restart_goal_count": recovery.restart_goal_ids.len(),
                        "boot_id": boot_id,
                    })),
                "control-plane: recovered prior-boot tasks at startup"
            );
        }
        Ok(Self {
            store,
            goal_store,
            boot_id,
            recovered_goal_ids: Arc::new(Mutex::new(recovery.restart_goal_ids)),
            data_dir_lock: Some(data_dir_lock),
        })
    }

    /// Spawn the periodic reaper as a detached task whose lifetime `DaemonRegistry`
    /// owns via `cancel`. Errors inside the loop are logged, never propagated.
    ///
    /// Uses `zeroclaw_spawn::spawn!` (NOT raw `tokio::spawn`, which `clippy.toml`
    /// bans workspace-wide) so the reaper task inherits the caller's tracing span.
    pub fn spawn_reaper(
        &self,
        max_runtime_secs: i64,
        goal_restart_recovery: GoalRestartRecovery,
        cancel: CancellationToken,
    ) -> JoinHandle<()> {
        debug_assert!(
            self.data_dir_lock.is_some() || cfg!(test),
            "production control-plane handles must hold the data-dir lock"
        );
        // Hoist owned clones to locals so the spawn! future captures them by value
        // (not `&self`, which the macro would otherwise hold across the 'static boundary).
        let store = Arc::clone(&self.store);
        let goal_store = Arc::clone(&self.goal_store);
        let boot_id = self.boot_id.clone();
        zeroclaw_spawn::spawn!(reaper::reaper_loop(
            store,
            goal_store,
            boot_id,
            max_runtime_secs,
            goal_restart_recovery,
            cancel
        ))
    }
}

/// Held OS advisory lock for one control-plane data directory.
///
/// The file lock, not this Rust object, is the source of truth for single
/// writer ownership. The object only owns the open file descriptor/handle so
/// the lock remains held for as long as the installed control-plane handle
/// lives.
#[derive(Debug)]
pub(crate) struct ControlPlaneDataDirLock {
    file: std::fs::File,
}

impl ControlPlaneDataDirLock {
    fn acquire(data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir)
            .with_context(|| format!("create control-plane data dir {}", data_dir.display()))?;
        let lock_path = data_dir.join("control_plane.lock");
        let mut file = open_locked_file(&lock_path)?;
        file.set_len(0)
            .with_context(|| format!("truncate control-plane lock {}", lock_path.display()))?;
        writeln!(
            file,
            "pid={}\nstarted_at={}",
            std::process::id(),
            chrono::Utc::now().to_rfc3339()
        )
        .with_context(|| format!("write control-plane lock {}", lock_path.display()))?;
        Ok(Self { file })
    }
}

#[cfg(unix)]
fn open_locked_file(path: &Path) -> Result<std::fs::File> {
    use std::os::fd::AsRawFd;

    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        // Do not truncate until after the advisory lock is held; otherwise a
        // rejected second owner could still rewrite the incumbent's lock file.
        .truncate(false)
        .open(path)
        .with_context(|| format!("open control-plane lock {}", path.display()))?;
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc == 0 {
        return Ok(file);
    }
    let error = std::io::Error::last_os_error();
    anyhow::bail!(
        "control-plane data dir is already locked at {}: {error}",
        path.display()
    );
}

#[cfg(unix)]
impl Drop for ControlPlaneDataDirLock {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;

        let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
    }
}

#[cfg(windows)]
fn open_locked_file(path: &Path) -> Result<std::fs::File> {
    use std::os::windows::fs::OpenOptionsExt;

    OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .share_mode(0)
        .open(path)
        .with_context(|| {
            format!(
                "open exclusive control-plane lock {}; another daemon may already own this data dir",
                path.display()
            )
        })
}

#[cfg(not(any(unix, windows)))]
fn open_locked_file(path: &Path) -> Result<std::fs::File> {
    OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| {
            format!(
                "create control-plane lock {}; this platform has no advisory lock implementation",
                path.display()
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_plane::goal_task::GoalTaskRecord;
    use crate::control_plane::recovery::{
        RecoveredGoalBatch, RecoveredGoalContinuationBlocker,
        recovered_goal_continuation_blocked_pause,
    };
    use crate::control_plane::task_registry::{TaskKind, TaskRecord, TaskStatus};

    fn running_task(id: &str, kind: TaskKind, boot_id: &str) -> TaskRecord {
        TaskRecord {
            id: id.into(),
            kind,
            agent: "main".into(),
            status: TaskStatus::Running,
            owner_pid: std::process::id(),
            owner_boot_id: boot_id.into(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: Some(format!("test-route:{id}")),
            delivered: false,
            idem_key: None,
            principal_id: None,
            started_at: "2020-01-01T00:00:00Z".into(),
            finished_at: None,
        }
    }

    fn make_task_unreadable(data_dir: &Path, task_id: &str) {
        rusqlite::Connection::open(data_dir.join("control_plane.db"))
            .unwrap()
            .execute(
                "UPDATE tasks SET kind = 'future-task-kind' WHERE id = ?1",
                [task_id],
            )
            .unwrap();
    }

    #[test]
    fn recovered_goal_blocker_payloads_contain_only_stable_reason_codes() {
        for (blocker, code) in [
            (
                RecoveredGoalContinuationBlocker::AgentUnavailable,
                "agent_unavailable",
            ),
            (
                RecoveredGoalContinuationBlocker::ProviderUnavailable,
                "provider_unavailable",
            ),
            (
                RecoveredGoalContinuationBlocker::MissingContinuationContext,
                "missing_continuation_context",
            ),
            (
                RecoveredGoalContinuationBlocker::ContinuationContextReadFailed,
                "continuation_context_read_failed",
            ),
            (
                RecoveredGoalContinuationBlocker::InvalidContinuationContext,
                "invalid_continuation_context",
            ),
            (
                RecoveredGoalContinuationBlocker::MissingGoalExtension,
                "missing_goal_extension",
            ),
            (
                RecoveredGoalContinuationBlocker::GoalExtensionReadFailed,
                "goal_extension_read_failed",
            ),
            (
                RecoveredGoalContinuationBlocker::ChannelUnavailable,
                "channel_unavailable",
            ),
            (
                RecoveredGoalContinuationBlocker::ChannelOwnerUnavailable,
                "channel_owner_unavailable",
            ),
            (
                RecoveredGoalContinuationBlocker::QueueUnavailable,
                "queue_unavailable",
            ),
        ] {
            let pause = recovered_goal_continuation_blocked_pause(blocker);
            let payload = pause.blockers[0]
                .payload
                .as_ref()
                .unwrap()
                .as_object()
                .unwrap();
            assert_eq!(
                payload.len(),
                1,
                "{code} payload must not duplicate runtime state"
            );
            assert_eq!(payload["reason_code"], code);
        }
    }

    struct BlockingGetTaskRegistry {
        started: Arc<tokio::sync::Notify>,
    }

    #[async_trait::async_trait]
    impl TaskRegistry for BlockingGetTaskRegistry {
        async fn create(&self, _rec: TaskRecord) -> Result<()> {
            anyhow::bail!("unused")
        }

        async fn heartbeat(&self, _id: &str, _owner_boot_id: &str) -> Result<()> {
            anyhow::bail!("unused")
        }

        async fn update_status(
            &self,
            _id: &str,
            _status: TaskStatus,
            _output: Option<String>,
            _error: Option<String>,
        ) -> Result<()> {
            anyhow::bail!("unused")
        }

        async fn claim_owner(
            &self,
            _id: &str,
            _owner_pid: u32,
            _owner_boot_id: &str,
        ) -> Result<()> {
            anyhow::bail!("unused")
        }

        async fn get(&self, _id: &str) -> Result<Option<TaskRecord>> {
            self.started.notify_one();
            std::future::pending().await
        }

        async fn list_running(&self) -> Result<Vec<TaskRecord>> {
            anyhow::bail!("unused")
        }

        async fn list_by_agent(&self, _agent: &str) -> Result<Vec<TaskRecord>> {
            anyhow::bail!("unused")
        }

        async fn reconcile_lost(&self, _id: &str, _now_boot_id: &str) -> Result<bool> {
            anyhow::bail!("unused")
        }
    }

    #[tokio::test]
    async fn start_in_tempdir_and_reap_handle() {
        let dir = tempfile::tempdir().unwrap();
        let h = ControlPlaneHandle::start(dir.path(), GoalRestartRecovery::default())
            .await
            .unwrap();
        assert!(!h.boot_id.is_empty());
        // a reaper spawns and stops cleanly on cancel
        let cancel = CancellationToken::new();
        let jh = h.spawn_reaper(600, GoalRestartRecovery::default(), cancel.clone());
        cancel.cancel();
        jh.await.unwrap();
    }

    #[tokio::test]
    async fn start_rejects_second_live_owner_for_same_data_dir() {
        let dir = tempfile::tempdir().unwrap();
        let _h = ControlPlaneHandle::start(dir.path(), GoalRestartRecovery::default())
            .await
            .unwrap();

        let second = ControlPlaneHandle::start(dir.path(), GoalRestartRecovery::default()).await;
        let err = second.err().expect("second live owner should fail");

        assert!(
            err.to_string()
                .contains("control-plane data dir is already locked")
                || err.to_string().contains("control-plane lock"),
            "unexpected lock error: {err:#}"
        );
    }

    #[tokio::test]
    async fn boot_id_distinguishes_runs_over_the_same_db() {
        let dir = tempfile::tempdir().unwrap();
        // First "boot" registers a running task, then the daemon "dies".
        let h1 = ControlPlaneHandle::start_with_boot_id(
            dir.path(),
            "boot-1".into(),
            GoalRestartRecovery::default(),
        )
        .await
        .unwrap();
        h1.store
            .create(TaskRecord {
                id: "t".into(),
                kind: TaskKind::Delegate,
                agent: "main".into(),
                status: TaskStatus::Running,
                owner_pid: 999_999,
                owner_boot_id: "boot-1".into(),
                heartbeat_at: None,
                depth: 0,
                parent_id: None,
                originator_route: None,
                delivered: false,
                idem_key: None,
                principal_id: None,
                started_at: "2026-06-18T00:00:00Z".into(),
                finished_at: None,
            })
            .await
            .unwrap();
        drop(h1);
        // Second boot recovers the non-goal orphan at startup.
        let h2 = ControlPlaneHandle::start_with_boot_id(
            dir.path(),
            "boot-2".into(),
            GoalRestartRecovery::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            h2.store.get("t").await.unwrap().unwrap().status,
            TaskStatus::Lost
        );
    }

    #[tokio::test]
    async fn startup_recovery_fails_on_unreadable_running_task() {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteTaskStore::new(dir.path()).unwrap();
        store
            .create(running_task("unreadable", TaskKind::Goal, "prior-boot"))
            .await
            .unwrap();
        drop(store);
        make_task_unreadable(dir.path(), "unreadable");

        let error = ControlPlaneHandle::start_with_boot_id(
            dir.path(),
            "current-boot".into(),
            GoalRestartRecovery::LastState,
        )
        .await
        .err()
        .expect("startup recovery must not skip an unreadable running row");

        assert!(error.to_string().contains("decode list_running rows"));
    }

    #[tokio::test]
    async fn reload_queues_only_current_boot_running_goals_once() {
        let dir = tempfile::tempdir().unwrap();
        let handle = ControlPlaneHandle::start_with_boot_id(
            dir.path(),
            "current-boot".into(),
            GoalRestartRecovery::LastState,
        )
        .await
        .unwrap();

        for task in [
            running_task("current-goal", TaskKind::Goal, "current-boot"),
            running_task("prior-goal", TaskKind::Goal, "prior-boot"),
            running_task("delegate", TaskKind::Delegate, "current-boot"),
        ] {
            handle.store.create(task).await.unwrap();
        }

        assert_eq!(handle.queue_running_goals_for_reload().await.unwrap(), 1);
        assert_eq!(handle.queue_running_goals_for_reload().await.unwrap(), 0);
        assert_eq!(handle.take_recovered_goal_ids(), vec!["current-goal"]);
    }

    #[tokio::test]
    async fn reload_recovery_fails_on_unreadable_running_task() {
        let dir = tempfile::tempdir().unwrap();
        let handle = ControlPlaneHandle::start_with_boot_id(
            dir.path(),
            "current-boot".into(),
            GoalRestartRecovery::LastState,
        )
        .await
        .unwrap();
        handle
            .store
            .create(running_task("unreadable", TaskKind::Goal, "current-boot"))
            .await
            .unwrap();
        make_task_unreadable(dir.path(), "unreadable");

        let error = handle
            .queue_running_goals_for_reload()
            .await
            .expect_err("reload recovery must not skip an unreadable running row");

        assert!(error.to_string().contains("decode list_running rows"));
        assert!(handle.take_recovered_goal_ids().is_empty());
    }

    #[test]
    fn recovered_goal_lease_restores_after_batch_commit() {
        let sqlite_store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let handle = ControlPlaneHandle {
            store: sqlite_store.clone(),
            goal_store: sqlite_store,
            boot_id: "current-boot".into(),
            recovered_goal_ids: Arc::new(Mutex::new(vec!["goal-1".into()])),
            data_dir_lock: None,
        };
        let batch = RecoveredGoalBatch::take(&handle);
        let lease = batch.lease("goal-1");
        batch.commit();
        assert!(handle.take_recovered_goal_ids().is_empty());

        drop(lease);

        assert_eq!(handle.take_recovered_goal_ids(), vec!["goal-1"]);
    }

    #[test]
    fn committed_recovered_goal_lease_does_not_restore() {
        let sqlite_store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let handle = ControlPlaneHandle {
            store: sqlite_store.clone(),
            goal_store: sqlite_store,
            boot_id: "current-boot".into(),
            recovered_goal_ids: Arc::new(Mutex::new(vec!["goal-1".into()])),
            data_dir_lock: None,
        };
        let batch = RecoveredGoalBatch::take(&handle);
        let lease = batch.lease("goal-1");
        batch.commit();

        lease.commit();

        assert!(handle.take_recovered_goal_ids().is_empty());
    }

    #[tokio::test]
    async fn no_channel_pause_restores_entire_batch_after_failure() {
        let dir = tempfile::tempdir().unwrap();
        let handle = ControlPlaneHandle::start_with_boot_id(
            dir.path(),
            "current-boot".into(),
            GoalRestartRecovery::LastState,
        )
        .await
        .unwrap();

        for task_id in ["paused-first", "still-pending"] {
            handle
                .goal_store
                .create_goal(
                    running_task(task_id, TaskKind::Goal, "current-boot"),
                    GoalTaskRecord {
                        task_id: task_id.into(),
                        objective: "keep working".into(),
                        effective_token_limit: None,
                        effective_cost_limit_usd: None,
                        pause_reason: None,
                        pause_description: None,
                        blockers: Vec::new(),
                    },
                    None,
                )
                .await
                .unwrap();
        }
        handle
            .store
            .create(running_task(
                "missing-goal-extension",
                TaskKind::Goal,
                "current-boot",
            ))
            .await
            .unwrap();
        *handle
            .recovered_goal_ids
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = vec![
            "paused-first".into(),
            "missing-goal-extension".into(),
            "still-pending".into(),
        ];

        let error = handle
            .pause_recovered_goals_without_channel_executor()
            .await
            .expect_err("missing goal extension must fail the atomic pause");

        assert!(error.to_string().contains("missing-goal-extension"));
        assert_eq!(
            handle.take_recovered_goal_ids(),
            vec!["paused-first", "missing-goal-extension", "still-pending"]
        );
        assert_eq!(
            handle
                .store
                .get("paused-first")
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Paused
        );
        assert_eq!(
            handle
                .store
                .get("still-pending")
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Running
        );
    }

    #[tokio::test]
    async fn no_channel_pause_abort_restores_recovery_batch() {
        let started = Arc::new(tokio::sync::Notify::new());
        let goal_store: Arc<dyn GoalTaskRegistry> =
            Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let handle = ControlPlaneHandle {
            store: Arc::new(BlockingGetTaskRegistry {
                started: Arc::clone(&started),
            }),
            goal_store,
            boot_id: "current-boot".into(),
            recovered_goal_ids: Arc::new(Mutex::new(vec!["aborted-goal".into()])),
            data_dir_lock: None,
        };
        let worker_handle = handle.clone();
        let worker = zeroclaw_spawn::spawn!(async move {
            worker_handle
                .pause_recovered_goals_without_channel_executor()
                .await
        });
        started.notified().await;

        worker.abort();
        let _ = worker.await;

        assert_eq!(handle.take_recovered_goal_ids(), vec!["aborted-goal"]);
    }
}
