//! Boot wiring for the control-plane — minted once per daemon run.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::reaper;
use super::task_registry::TaskRegistry;
use super::task_store_sqlite::SqliteTaskStore;

/// The live control-plane, shared (cheaply, via `Arc`/clone) across producers and
/// the reaper.
#[derive(Clone)]
pub struct ControlPlaneHandle {
    pub store: Arc<dyn TaskRegistry>,
    pub boot_id: String,
    recovery_enabled: bool,
}

impl ControlPlaneHandle {
    pub async fn start(data_dir: &Path) -> Result<Self> {
        let run_id = uuid::Uuid::new_v4().to_string();
        Self::start_with_boot_id(data_dir, run_id).await
    }

    /// As [`Self::start`] but with a caller-supplied `boot_id` — lets `DaemonRegistry`
    /// reuse a process-stable run-id across reloads instead of a fresh UUID.
    pub async fn start_with_boot_id(data_dir: &Path, boot_id: String) -> Result<Self> {
        let sqlite_store = Arc::new(SqliteTaskStore::new(data_dir)?);
        let recovery_enabled = !sqlite_store.is_future_schema_read_only();
        let goal_recovered = if recovery_enabled {
            crate::task_store_sqlite::goal::recover_on_boot(&sqlite_store, &boot_id)?
        } else {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                    .with_attrs(::serde_json::json!({ "boot_id": boot_id })),
                "control-plane: skipped boot recovery for a future-schema database"
            );
            0
        };
        let store: Arc<dyn TaskRegistry> = sqlite_store;
        let reclaimed = if recovery_enabled {
            reaper::recovery_pass(store.as_ref(), &boot_id).await?
        } else {
            0
        };
        if reclaimed + goal_recovered > 0 {
            ::zeroclaw_log::record!(
                INFO,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                    .with_attrs(
                        ::serde_json::json!({ "reclaimed": reclaimed, "goal_recovered": goal_recovered, "boot_id": boot_id })
                    ),
                "control-plane: reclaimed prior-boot orphan tasks at startup"
            );
        }
        Ok(Self {
            store,
            boot_id,
            recovery_enabled,
        })
    }

    pub fn spawn_reaper(&self, max_runtime_secs: i64, cancel: CancellationToken) -> JoinHandle<()> {
        if !self.recovery_enabled {
            return zeroclaw_spawn::spawn!(async move { cancel.cancelled().await });
        }
        // Hoist owned clones to locals so the spawn! future captures them by value
        // (not `&self`, which the macro would otherwise hold across the 'static boundary).
        let store = Arc::clone(&self.store);
        let boot_id = self.boot_id.clone();
        zeroclaw_spawn::spawn!(reaper::reaper_loop(
            store,
            boot_id,
            max_runtime_secs,
            cancel
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GoalAccountingState, GoalPauseReason, GoalTaskRecord, GoalTaskRegistry,
        GoalTransitionResult, TaskKind, TaskRecord, TaskStatus,
    };

    #[tokio::test]
    async fn start_in_tempdir_and_reap_handle() {
        let dir = tempfile::tempdir().unwrap();
        let h = ControlPlaneHandle::start(dir.path()).await.unwrap();
        assert!(!h.boot_id.is_empty());
        // a reaper spawns and stops cleanly on cancel
        let cancel = CancellationToken::new();
        let jh = h.spawn_reaper(600, cancel.clone());
        cancel.cancel();
        jh.await.unwrap();
    }

    #[tokio::test]
    async fn boot_id_distinguishes_runs_over_the_same_db() {
        use crate::task_registry::{TaskKind, TaskRecord, TaskStatus};
        let dir = tempfile::tempdir().unwrap();
        // First "boot" registers a running task, then the daemon "dies".
        let h1 = ControlPlaneHandle::start_with_boot_id(dir.path(), "boot-1".into())
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
                session_id: None,
                execution_epoch: 0,
                started_at: "2026-06-18T00:00:00Z".into(),
                finished_at: None,
            })
            .await
            .unwrap();
        // Second boot recovers the orphan at startup.
        let h2 = ControlPlaneHandle::start_with_boot_id(dir.path(), "boot-2".into())
            .await
            .unwrap();
        assert_eq!(
            h2.store.get("t").await.unwrap().unwrap().status,
            TaskStatus::Lost
        );
    }

    #[tokio::test]
    async fn future_schema_boot_is_read_only_and_skips_recovery() {
        use rusqlite::Connection;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("control_plane.db");
        let db = Connection::open(&path).unwrap();
        db.execute_batch(
            "PRAGMA journal_mode = DELETE;
             CREATE TABLE tasks (
                 id TEXT PRIMARY KEY, kind TEXT NOT NULL, agent TEXT NOT NULL, status TEXT NOT NULL,
                 owner_pid INTEGER NOT NULL DEFAULT 0, owner_boot_id TEXT NOT NULL DEFAULT '', heartbeat_at TEXT,
                 depth INTEGER NOT NULL DEFAULT 0, parent_id TEXT, originator_route TEXT, delivered INTEGER NOT NULL DEFAULT 0,
                 idem_key TEXT, principal_id TEXT, session_id TEXT, execution_epoch INTEGER NOT NULL DEFAULT 0,
                 started_at TEXT NOT NULL, finished_at TEXT, output TEXT, error TEXT
             );
             CREATE TABLE goal_tasks (
                 task_id TEXT PRIMARY KEY, objective TEXT NOT NULL, effective_token_limit INTEGER,
                 effective_cost_limit_usd REAL, pause_reason TEXT, pause_description TEXT,
                 blockers_json TEXT NOT NULL DEFAULT '[]', pending_call_id TEXT,
                 pending_call_epoch INTEGER, accounting_state TEXT NOT NULL DEFAULT 'complete'
             );
             INSERT INTO tasks (id,kind,agent,status,owner_boot_id,session_id,execution_epoch,started_at)
                 VALUES ('future-goal','goal','main','running','old-boot','session',7,'now');
             INSERT INTO goal_tasks (task_id,objective) VALUES ('future-goal','unchanged');
             CREATE TRIGGER future_schema_no_task_mutation
                BEFORE UPDATE ON tasks FOR EACH ROW
                BEGIN SELECT RAISE(ABORT, 'future schema must not be mutated'); END;
             CREATE TRIGGER future_schema_no_goal_mutation
                BEFORE UPDATE ON goal_tasks FOR EACH ROW
                BEGIN SELECT RAISE(ABORT, 'future schema must not be mutated'); END;
             PRAGMA user_version = 16;",
        )
        .unwrap();
        let journal_mode: String = db
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        assert_eq!(journal_mode, "delete");
        drop(db);

        let handle = ControlPlaneHandle::start_with_boot_id(dir.path(), "new-boot".into())
            .await
            .expect("future schema opens warning-only without boot recovery");
        drop(handle);

        let db = Connection::open(&path).unwrap();
        let state: (String, String, i64, Option<String>) = db
            .query_row(
                "SELECT status, owner_boot_id, execution_epoch, error FROM tasks WHERE id = 'future-goal'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(state, ("running".into(), "old-boot".into(), 7, None));
        assert_eq!(
            db.query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            16
        );
        assert_eq!(
            db.query_row("PRAGMA journal_mode", [], |row| row.get::<_, String>(0))
                .unwrap(),
            "delete",
        );
    }

    #[tokio::test]
    async fn boot_pauses_prior_goal_with_daemon_restart_instead_of_losing_it() {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteTaskStore::new(dir.path()).unwrap();
        let admitted = store
            .create_or_replace_session_goal(
                TaskRecord {
                    id: "goal".into(),
                    kind: TaskKind::Goal,
                    agent: "main".into(),
                    status: TaskStatus::Running,
                    owner_pid: 1,
                    owner_boot_id: "old".into(),
                    heartbeat_at: None,
                    depth: 0,
                    parent_id: None,
                    originator_route: None,
                    delivered: false,
                    idem_key: None,
                    principal_id: None,
                    session_id: Some("session".into()),
                    execution_epoch: 1,
                    started_at: "now".into(),
                    finished_at: None,
                },
                GoalTaskRecord {
                    task_id: "goal".into(),
                    objective: "x".into(),
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                    pending_call_id: None,
                    pending_call_epoch: None,
                    accounting_state: GoalAccountingState::Complete,
                },
            )
            .await
            .unwrap();
        assert_eq!(admitted, GoalTransitionResult::Applied);
        let booted = ControlPlaneHandle::start_with_boot_id(dir.path(), "new".into())
            .await
            .unwrap();
        let task = booted.store.get("goal").await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Paused);
        assert_eq!(task.execution_epoch, 2);
        let goal = store.get_goal_task("goal").await.unwrap().unwrap();
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::DaemonRestart));
    }

    #[tokio::test]
    async fn boot_fails_unsafe_running_goals_without_overwriting_known_accounting() {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteTaskStore::new(dir.path()).unwrap();
        for (id, session) in [
            ("missing", "missing-session"),
            ("invalid", "invalid-session"),
        ] {
            let task = TaskRecord {
                id: id.into(),
                kind: TaskKind::Goal,
                agent: "main".into(),
                status: TaskStatus::Running,
                owner_pid: 1,
                owner_boot_id: "old".into(),
                heartbeat_at: None,
                depth: 0,
                parent_id: None,
                originator_route: None,
                delivered: false,
                idem_key: None,
                principal_id: None,
                session_id: Some(session.into()),
                execution_epoch: 1,
                started_at: "now".into(),
                finished_at: None,
            };
            let goal = GoalTaskRecord {
                task_id: id.into(),
                objective: "x".into(),
                effective_token_limit: None,
                effective_cost_limit_usd: None,
                pause_reason: None,
                pause_description: None,
                blockers: Vec::new(),
                pending_call_id: None,
                pending_call_epoch: None,
                accounting_state: GoalAccountingState::Complete,
            };
            assert_eq!(
                store
                    .create_or_replace_session_goal(task, goal)
                    .await
                    .unwrap(),
                GoalTransitionResult::Applied
            );
        }
        assert_eq!(
            store
                .admit_pending_call("missing", "missing-session", 1, "missing-call")
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            store
                .settle_pending_call(
                    "missing",
                    "missing-session",
                    1,
                    "missing-call",
                    GoalAccountingState::Missing
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            store
                .admit_pending_call("invalid", "invalid-session", 1, "invalid-call")
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            store
                .settle_pending_call(
                    "invalid",
                    "invalid-session",
                    1,
                    "invalid-call",
                    GoalAccountingState::Invalid
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        let task = TaskRecord {
            id: "pending".into(),
            kind: TaskKind::Goal,
            agent: "main".into(),
            status: TaskStatus::Running,
            owner_pid: 1,
            owner_boot_id: "old".into(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: None,
            delivered: false,
            idem_key: None,
            principal_id: None,
            session_id: Some("pending-session".into()),
            execution_epoch: 1,
            started_at: "now".into(),
            finished_at: None,
        };
        let goal = GoalTaskRecord {
            task_id: "pending".into(),
            objective: "x".into(),
            effective_token_limit: None,
            effective_cost_limit_usd: None,
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
            pending_call_id: None,
            pending_call_epoch: None,
            accounting_state: GoalAccountingState::Complete,
        };
        assert_eq!(
            store
                .create_or_replace_session_goal(task, goal)
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            store
                .admit_pending_call("pending", "pending-session", 1, "call")
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        let booted = ControlPlaneHandle::start_with_boot_id(dir.path(), "new".into())
            .await
            .unwrap();
        for id in ["missing", "invalid", "pending"] {
            assert_eq!(
                booted.store.get(id).await.unwrap().unwrap().status,
                TaskStatus::Failed
            );
        }
        assert_eq!(
            store
                .get_goal_task("missing")
                .await
                .unwrap()
                .unwrap()
                .accounting_state,
            GoalAccountingState::Missing
        );
        assert_eq!(
            store
                .get_goal_task("invalid")
                .await
                .unwrap()
                .unwrap()
                .accounting_state,
            GoalAccountingState::Invalid
        );
        let pending = store.get_goal_task("pending").await.unwrap().unwrap();
        assert_eq!(
            pending.accounting_state,
            GoalAccountingState::OutcomeUnknown
        );
        assert_eq!(pending.pending_call_id.as_deref(), Some("call"));
        assert_eq!(pending.pending_call_epoch, Some(1));

        let malformed_dir = tempfile::tempdir().unwrap();
        let db = rusqlite::Connection::open(malformed_dir.path().join("control_plane.db")).unwrap();
        db.execute_batch("CREATE TABLE tasks (id TEXT PRIMARY KEY, kind TEXT NOT NULL, agent TEXT NOT NULL, status TEXT NOT NULL, owner_pid INTEGER NOT NULL DEFAULT 0, owner_boot_id TEXT NOT NULL DEFAULT '', heartbeat_at TEXT, depth INTEGER NOT NULL DEFAULT 0, parent_id TEXT, originator_route TEXT, delivered INTEGER NOT NULL DEFAULT 0, idem_key TEXT, principal_id TEXT, session_id TEXT, execution_epoch INTEGER NOT NULL DEFAULT 0, started_at TEXT NOT NULL, finished_at TEXT, output TEXT, error TEXT); CREATE TABLE goal_tasks (task_id TEXT PRIMARY KEY, objective TEXT NOT NULL, effective_token_limit INTEGER, effective_cost_limit_usd REAL, pause_reason TEXT, pause_description TEXT, blockers_json TEXT NOT NULL DEFAULT '[]', pending_call_id TEXT, pending_call_epoch INTEGER, accounting_state TEXT NOT NULL DEFAULT 'complete'); INSERT INTO tasks (id,kind,agent,status,session_id,execution_epoch,owner_boot_id,started_at) VALUES ('malformed','goal','main','running','malformed-session',1,'old','now'); PRAGMA user_version=8;").unwrap();
        drop(db);
        let malformed = ControlPlaneHandle::start_with_boot_id(malformed_dir.path(), "new".into())
            .await
            .unwrap();
        assert_eq!(
            malformed
                .store
                .get("malformed")
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Failed
        );
    }
}
