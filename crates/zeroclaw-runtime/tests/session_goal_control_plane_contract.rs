//! File-backed contract tests for session-bound durable Goal control state.

use std::sync::{Arc, Barrier};

use zeroclaw_runtime::control_plane::{
    ControlPlaneHandle, GoalAccountingState, GoalTaskRecord, GoalTaskRegistry,
    GoalTransitionResult, SqliteTaskStore, TaskKind, TaskRecord, TaskRegistry, TaskStatus,
};

fn session_goal(id: &str, session_id: &str) -> (TaskRecord, GoalTaskRecord) {
    (
        TaskRecord {
            id: id.into(),
            kind: TaskKind::Goal,
            agent: "main".into(),
            status: TaskStatus::Running,
            owner_pid: 1,
            owner_boot_id: "boot-a".into(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: Some("room-a".into()),
            delivered: false,
            idem_key: None,
            principal_id: Some("principal-a".into()),
            session_id: Some(session_id.into()),
            execution_epoch: 0,
            started_at: "2026-08-20T00:00:00Z".into(),
            finished_at: None,
        },
        GoalTaskRecord {
            task_id: id.into(),
            objective: "finish the task".into(),
            effective_token_limit: Some(10),
            effective_cost_limit_usd: None,
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
            pending_call_id: None,
            pending_call_epoch: None,
            accounting_state: GoalAccountingState::Complete,
        },
    )
}

#[tokio::test]
async fn session_goal_admission_is_unique_and_epoch_fenced() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (first, first_goal) = session_goal("first", "session-a");

    assert_eq!(
        store
            .create_or_replace_session_goal(first, first_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );

    let current = store
        .current_goal_for_session("session-a")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(current.id, "first");
    assert_eq!(current.execution_epoch, 1);

    let (second, second_goal) = session_goal("second", "session-a");
    assert_eq!(
        store
            .create_or_replace_session_goal(second, second_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );

    assert_eq!(
        store
            .admit_pending_call("first", "session-a", 1, "call-a")
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .pause_session_goal(
                "first",
                "session-a",
                1,
                zeroclaw_runtime::control_plane::GoalPauseState {
                    reason: zeroclaw_runtime::control_plane::GoalPauseReason::OperatorPaused,
                    description: None,
                    blockers: Vec::new(),
                },
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .settle_pending_call(
                "first",
                "session-a",
                1,
                "call-a",
                GoalAccountingState::Complete,
            )
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
}

#[test]
fn sqlite_rejects_invalid_session_and_pending_operation_rows() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    drop(store);
    let database = rusqlite::Connection::open(directory.path().join("control_plane.db")).unwrap();

    assert!(
        database
            .execute(
                "INSERT INTO tasks (id, kind, agent, status, session_id, started_at)
             VALUES ('blank', 'goal', 'main', 'running', ' ', 'now')",
                [],
            )
            .is_err()
    );
    database
        .execute(
            "INSERT INTO tasks
                (id, kind, agent, status, session_id, execution_epoch, started_at)
             VALUES ('goal-a', 'goal', 'main', 'running', 'session-a', 1, 'now')",
            [],
        )
        .unwrap();
    assert!(
        database
            .execute(
                "INSERT INTO goal_tasks (task_id, objective, pending_call_id)
             VALUES ('goal-a', 'objective', 'call-a')",
                [],
            )
            .is_err()
    );
    database
        .execute(
            "INSERT INTO goal_tasks
                (task_id, objective, pending_call_id, pending_call_epoch)
             VALUES ('goal-a', 'objective', 'call-a', 1)",
            [],
        )
        .unwrap();
    assert!(
        database
            .execute(
                "UPDATE goal_tasks SET accounting_state = 'made_up' WHERE task_id = 'goal-a'",
                [],
            )
            .is_err()
    );
    assert!(
        database
            .execute(
                "UPDATE tasks SET session_id = 'session-b' WHERE id = 'goal-a'",
                [],
            )
            .is_err()
    );
    database
        .execute(
            "INSERT INTO tasks (id, kind, agent, status, started_at)
             VALUES ('delegate-a', 'delegate', 'main', 'running', 'now')",
            [],
        )
        .unwrap();
    assert!(
        database
            .execute(
                "UPDATE tasks
                SET kind = 'goal', session_id = 'manufactured-session'
              WHERE id = 'delegate-a'",
                [],
            )
            .is_err()
    );
}

#[tokio::test]
async fn sqlite_cannot_replace_or_rekey_an_existing_goal_row() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (goal, extension) = session_goal("active-goal", "session-a");
    assert_eq!(
        store
            .create_or_replace_session_goal(goal, extension)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    drop(store);

    let database = rusqlite::Connection::open(directory.path().join("control_plane.db")).unwrap();
    database
        .execute_batch("PRAGMA foreign_keys = OFF; PRAGMA recursive_triggers = OFF;")
        .unwrap();
    let goal_rowid: i64 = database
        .query_row(
            "SELECT rowid FROM tasks WHERE id = 'active-goal'",
            [],
            |row| row.get(0),
        )
        .unwrap();

    assert!(
        database
            .execute(
                "INSERT OR REPLACE INTO tasks
                    (rowid, id, kind, agent, status, session_id, execution_epoch, started_at)
                 VALUES (?1, 'replacement', 'delegate', 'main', 'running', NULL, 0, 'now')",
                [goal_rowid],
            )
            .is_err(),
        "a rowid conflict must not replace a canonical Goal row"
    );
    assert!(
        database
            .execute(
                "INSERT OR REPLACE INTO tasks
                    (id, kind, agent, status, session_id, execution_epoch, started_at)
                 VALUES ('replacement-goal', 'goal', 'main', 'running', 'session-a', 1, 'now')",
                [],
            )
            .is_err(),
        "a unique session conflict must not replace a canonical Goal row"
    );
    assert!(
        database
            .execute(
                "INSERT OR REPLACE INTO tasks
                    (id, kind, agent, status, session_id, execution_epoch, started_at)
                 VALUES ('active-goal', 'delegate', 'main', 'running', NULL, 0, 'now')",
                [],
            )
            .is_err(),
        "a primary-key conflict must not replace a canonical Goal row"
    );
    database
        .execute(
            "INSERT INTO tasks (id, kind, agent, status, started_at)
             VALUES ('delegate-a', 'delegate', 'main', 'running', 'now')",
            [],
        )
        .unwrap();
    assert!(
        database
            .execute(
                "UPDATE OR REPLACE tasks SET id = 'active-goal' WHERE id = 'delegate-a'",
                [],
            )
            .is_err(),
        "a key update must not replace a canonical Goal row"
    );

    let goal_kind: String = database
        .query_row(
            "SELECT kind FROM tasks WHERE id = 'active-goal'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(goal_kind, "goal");
    let extension_count: i64 = database
        .query_row(
            "SELECT COUNT(*) FROM goal_tasks WHERE task_id = 'active-goal'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(extension_count, 1);
}

#[tokio::test]
async fn generic_task_reregistration_keeps_its_documented_goal_noop_semantics() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("idempotent-goal", "session-a");
    store
        .create_or_replace_session_goal(task.clone(), goal)
        .await
        .unwrap();

    store
        .create(task)
        .await
        .expect("re-registering a canonical Goal by id is a no-op");
    assert_eq!(
        store
            .current_goal_for_session("session-a")
            .await
            .unwrap()
            .unwrap()
            .id,
        "idempotent-goal"
    );
}

#[tokio::test]
async fn independent_connections_can_reregister_the_same_task_idempotently() {
    let directory = tempfile::tempdir().unwrap();
    SqliteTaskStore::new(directory.path()).unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let mut workers = Vec::new();
    for _ in 0..2 {
        let data_dir = directory.path().to_owned();
        let barrier = Arc::clone(&barrier);
        workers.push(std::thread::spawn(move || {
            let store = SqliteTaskStore::new(&data_dir).unwrap();
            let (task, _) = session_goal("concurrent-idempotent", "session-idempotent");
            barrier.wait();
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(store.create(task))
        }));
    }
    for worker in workers {
        worker
            .join()
            .unwrap()
            .expect("concurrent re-registration must be an idempotent no-op");
    }

    let store = SqliteTaskStore::new(directory.path()).unwrap();
    assert!(store.get("concurrent-idempotent").await.unwrap().is_some());
}

#[tokio::test]
async fn migration_installs_identity_guards_before_reopening_a_goal_database() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (goal, extension) = session_goal("migration-goal", "session-a");
    store
        .create_or_replace_session_goal(goal, extension)
        .await
        .unwrap();
    drop(store);

    let path = directory.path().join("control_plane.db");
    let database = rusqlite::Connection::open(&path).unwrap();
    database
        .execute_batch(
            "DROP TRIGGER trg_tasks_goal_identity_insert;
             DROP TRIGGER trg_tasks_goal_identity_update;
             PRAGMA user_version = 9;",
        )
        .unwrap();
    drop(database);

    let store = SqliteTaskStore::new(directory.path()).unwrap();
    drop(store);
    let database = rusqlite::Connection::open(path).unwrap();
    let version: i64 = database
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(version, 11);
    database
        .execute_batch("PRAGMA foreign_keys = OFF; PRAGMA recursive_triggers = OFF;")
        .unwrap();
    assert!(
        database
            .execute(
                "INSERT OR REPLACE INTO tasks
                    (id, kind, agent, status, session_id, execution_epoch, started_at)
                 VALUES ('migration-goal', 'delegate', 'main', 'running', NULL, 0, 'now')",
                [],
            )
            .is_err()
    );
    assert_eq!(
        database
            .query_row(
                "SELECT COUNT(*) FROM goal_tasks WHERE task_id = 'migration-goal'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
}

#[test]
fn v8_migration_failure_rolls_back_and_reopens_after_the_source_is_repaired() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("control_plane.db");
    let database = rusqlite::Connection::open(&path).unwrap();
    database
        .execute_batch(
            "CREATE TABLE tasks (
                 id TEXT PRIMARY KEY, kind TEXT NOT NULL, agent TEXT NOT NULL,
                 status TEXT NOT NULL, owner_pid INTEGER NOT NULL DEFAULT 0,
                 owner_boot_id TEXT NOT NULL DEFAULT '', heartbeat_at TEXT,
                 depth INTEGER NOT NULL DEFAULT 0, parent_id TEXT,
                 originator_route TEXT, delivered INTEGER NOT NULL DEFAULT 0,
                 idem_key TEXT, principal_id TEXT, session_id TEXT,
                 started_at TEXT NOT NULL, finished_at TEXT, output TEXT, error TEXT
             );
             CREATE TABLE goal_tasks (
                 task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,
                 objective TEXT NOT NULL, effective_token_limit INTEGER,
                 effective_cost_limit_usd REAL, pause_reason TEXT,
                 pause_description TEXT, blockers_json TEXT NOT NULL DEFAULT '[]'
             );
             INSERT INTO tasks (id, kind, agent, status, session_id, started_at)
             VALUES
                 ('first', 'goal', 'main', 'running', 'duplicate-session', 'now'),
                 ('second', 'goal', 'main', 'running', 'duplicate-session', 'now');
             INSERT INTO goal_tasks (task_id, objective)
             VALUES ('first', 'first'), ('second', 'second');
             PRAGMA user_version = 7;",
        )
        .unwrap();
    drop(database);

    assert!(SqliteTaskStore::new(directory.path()).is_err());
    let database = rusqlite::Connection::open(&path).unwrap();
    let version: i64 = database
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(
        version, 7,
        "a failed v8 migration must not advance its version"
    );
    let execution_epoch_columns: i64 = database
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('tasks') WHERE name = 'execution_epoch'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        execution_epoch_columns, 0,
        "v8 column additions must roll back with the failed unique index"
    );
    database
        .execute("UPDATE tasks SET session_id = NULL WHERE id = 'second'", [])
        .unwrap();
    drop(database);

    let store = SqliteTaskStore::new(directory.path()).unwrap();
    drop(store);
    let database = rusqlite::Connection::open(path).unwrap();
    let version: i64 = database
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(version, 11);
}

#[test]
fn independent_connections_admit_exactly_one_goal_for_a_session() {
    let directory = tempfile::tempdir().unwrap();
    SqliteTaskStore::new(directory.path()).unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let mut workers = Vec::new();
    for id in ["left", "right"] {
        let data_dir = directory.path().to_owned();
        let barrier = Arc::clone(&barrier);
        workers.push(std::thread::spawn(move || {
            let store = SqliteTaskStore::new(&data_dir).unwrap();
            let (task, goal) = session_goal(id, "session-race");
            barrier.wait();
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(store.create_or_replace_session_goal(task, goal))
                .unwrap()
        }));
    }
    let outcomes: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect();
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| **outcome == GoalTransitionResult::Applied)
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| **outcome == GoalTransitionResult::Stale)
            .count(),
        1
    );
}

#[tokio::test]
async fn restart_pauses_settled_goals_and_fails_unsettled_operations() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (settled, settled_goal) = session_goal("settled", "session-settled");
    let (unsettled, unsettled_goal) = session_goal("unsettled", "session-unsettled");
    store
        .create_or_replace_session_goal(settled, settled_goal)
        .await
        .unwrap();
    store
        .create_or_replace_session_goal(unsettled, unsettled_goal)
        .await
        .unwrap();
    store
        .admit_pending_call("unsettled", "session-unsettled", 1, "call-unsettled")
        .await
        .unwrap();
    drop(store);

    let handle = ControlPlaneHandle::start_with_boot_id(directory.path(), "boot-b".into())
        .await
        .unwrap();
    let settled = handle.store.get("settled").await.unwrap().unwrap();
    let unsettled = handle.store.get("unsettled").await.unwrap().unwrap();
    assert_eq!(settled.status, TaskStatus::Paused);
    assert_eq!(settled.execution_epoch, 2);
    assert_eq!(unsettled.status, TaskStatus::Failed);
    assert_eq!(unsettled.execution_epoch, 2);
    drop(handle);

    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let unsettled_goal = store.get_goal_task("unsettled").await.unwrap().unwrap();
    assert_eq!(
        unsettled_goal.accounting_state,
        GoalAccountingState::OutcomeUnknown
    );
    assert_eq!(
        unsettled_goal.pending_call_id.as_deref(),
        Some("call-unsettled")
    );
}

#[tokio::test]
async fn restart_marks_pending_accounting_unknown_even_with_a_prior_error() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("preerrored", "session-preerrored");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    store
        .admit_pending_call("preerrored", "session-preerrored", 1, "pending-1")
        .await
        .unwrap();
    drop(store);

    let database = rusqlite::Connection::open(directory.path().join("control_plane.db")).unwrap();
    database
        .execute(
            "UPDATE tasks SET error = 'preexisting diagnostic' WHERE id = 'preerrored'",
            [],
        )
        .unwrap();
    drop(database);

    let handle = ControlPlaneHandle::start_with_boot_id(directory.path(), "boot-b".into())
        .await
        .unwrap();
    assert_eq!(
        handle
            .store
            .get("preerrored")
            .await
            .unwrap()
            .unwrap()
            .status,
        TaskStatus::Failed
    );
    drop(handle);

    let store = SqliteTaskStore::new(directory.path()).unwrap();
    assert_eq!(
        store
            .get_goal_task("preerrored")
            .await
            .unwrap()
            .unwrap()
            .accounting_state,
        GoalAccountingState::OutcomeUnknown
    );
}

#[tokio::test]
async fn terminal_goal_remains_current_until_replacement_and_sessions_are_independent() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (first, first_goal) = session_goal("first", "session-a");
    let (other, other_goal) = session_goal("other", "session-b");
    assert_eq!(
        store
            .create_or_replace_session_goal(first, first_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .create_or_replace_session_goal(other, other_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied,
        "identical route and principal do not define Goal identity"
    );
    assert_eq!(
        store
            .finish_session_goal(
                "first",
                "session-a",
                1,
                TaskStatus::Failed,
                Some("test terminal transition".into()),
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    let terminal = store
        .current_goal_for_session("session-a")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(terminal.id, "first");
    assert_eq!(terminal.status, TaskStatus::Failed);
    assert_eq!(terminal.execution_epoch, 2);

    let (replacement, replacement_goal) = session_goal("replacement", "session-a");
    assert_eq!(
        store
            .create_or_replace_session_goal(replacement, replacement_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    let current = store
        .current_goal_for_session("session-a")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(current.id, "replacement");
    assert_eq!(current.execution_epoch, 1);
    assert!(store.get("first").await.unwrap().is_none());
    assert_eq!(
        store
            .current_goal_for_session("session-b")
            .await
            .unwrap()
            .unwrap()
            .id,
        "other"
    );
}

#[tokio::test]
async fn migration_preserves_legacy_goal_rows_but_boot_fails_legacy_active_work() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("control_plane.db");
    let database = rusqlite::Connection::open(&path).unwrap();
    database
        .execute_batch(
            "CREATE TABLE tasks (
                 id TEXT PRIMARY KEY, kind TEXT NOT NULL, agent TEXT NOT NULL,
                 status TEXT NOT NULL, owner_pid INTEGER NOT NULL DEFAULT 0,
                 owner_boot_id TEXT NOT NULL DEFAULT '', heartbeat_at TEXT,
                 depth INTEGER NOT NULL DEFAULT 0, parent_id TEXT,
                 originator_route TEXT, delivered INTEGER NOT NULL DEFAULT 0,
                 idem_key TEXT, principal_id TEXT, started_at TEXT NOT NULL,
                 finished_at TEXT, output TEXT, error TEXT
             );
             CREATE TABLE goal_tasks (
                 task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,
                 objective TEXT NOT NULL, effective_token_limit INTEGER,
                 effective_cost_limit_usd REAL, pause_reason TEXT,
                 pause_description TEXT, blockers_json TEXT NOT NULL DEFAULT '[]'
             );
             INSERT INTO tasks
                 (id, kind, agent, status, owner_boot_id, started_at)
             VALUES ('legacy-running', 'goal', 'main', 'running', 'old-boot', 'now');
             INSERT INTO goal_tasks (task_id, objective)
             VALUES ('legacy-running', 'old objective');
             PRAGMA user_version = 7;",
        )
        .unwrap();
    drop(database);

    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let legacy = store.get("legacy-running").await.unwrap().unwrap();
    assert!(legacy.session_id.is_none());
    assert_eq!(legacy.execution_epoch, 0);
    assert!(
        store
            .current_goal_for_session("any-session")
            .await
            .unwrap()
            .is_none()
    );
    drop(store);

    let handle = ControlPlaneHandle::start_with_boot_id(directory.path(), "new-boot".into())
        .await
        .unwrap();
    let legacy = handle.store.get("legacy-running").await.unwrap().unwrap();
    assert_eq!(legacy.status, TaskStatus::Failed);
    assert_eq!(legacy.execution_epoch, 1);
    drop(handle);

    let database = rusqlite::Connection::open(path).unwrap();
    let error: String = database
        .query_row(
            "SELECT error FROM tasks WHERE id = 'legacy-running'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(error, "legacy_goal_missing_session");
}

#[tokio::test]
async fn terminal_replacement_requires_the_existing_goal_bindings() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (first, first_goal) = session_goal("first", "session-a");
    store
        .create_or_replace_session_goal(first, first_goal)
        .await
        .unwrap();
    store
        .finish_session_goal(
            "first",
            "session-a",
            1,
            TaskStatus::Failed,
            Some("terminal".into()),
        )
        .await
        .unwrap();

    let (mut replacement, replacement_goal) = session_goal("replacement", "session-a");
    replacement.principal_id = Some("principal-b".into());
    assert_eq!(
        store
            .create_or_replace_session_goal(replacement, replacement_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
    assert_eq!(
        store
            .current_goal_for_session("session-a")
            .await
            .unwrap()
            .unwrap()
            .id,
        "first"
    );
}

#[tokio::test]
async fn session_goal_creation_and_completion_reject_impossible_lifecycle_states() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (mut dirty_task, dirty_goal) = session_goal("dirty", "session-dirty");
    dirty_task.status = TaskStatus::Paused;
    assert!(
        store
            .create_or_replace_session_goal(dirty_task, dirty_goal)
            .await
            .is_err(),
        "new session Goals must start Running"
    );

    let (task, goal) = session_goal("paused", "session-paused");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    assert_eq!(
        store
            .pause_session_goal(
                "paused",
                "session-paused",
                1,
                zeroclaw_runtime::control_plane::GoalPauseState {
                    reason: zeroclaw_runtime::control_plane::GoalPauseReason::OperatorPaused,
                    description: None,
                    blockers: Vec::new(),
                },
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .finish_session_goal("paused", "session-paused", 2, TaskStatus::Completed, None,)
            .await
            .unwrap(),
        GoalTransitionResult::Stale,
        "only a Running Goal may become Completed"
    );
}
