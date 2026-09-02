use std::sync::{Arc, Barrier};

use rusqlite::params;
use zeroclaw_runtime::control_plane::{
    GoalTaskRecord, GoalTaskRegistry, GoalTransitionResult, SqliteTaskStore, TaskKind, TaskRecord,
    TaskRegistry, TaskStatus,
};

fn task(id: &str, session_id: &str) -> TaskRecord {
    TaskRecord {
        id: id.into(),
        kind: TaskKind::Goal,
        agent: "main".into(),
        status: TaskStatus::Running,
        owner_pid: 1,
        owner_boot_id: "test-boot".into(),
        heartbeat_at: None,
        depth: 0,
        parent_id: None,
        originator_route: Some("matrix:room".into()),
        delivered: false,
        idem_key: None,
        principal_id: Some("@raw-mxid:example.test".into()),
        session_id: Some(session_id.into()),
        execution_epoch: 0,
        started_at: "2026-09-02T00:00:00Z".into(),
        finished_at: None,
    }
}

fn goal(task_id: &str) -> GoalTaskRecord {
    GoalTaskRecord {
        task_id: task_id.into(),
        objective: "Stop when the requested task is complete.".into(),
        ..GoalTaskRecord::default()
    }
}

#[test]
fn independent_file_backed_connections_admit_one_current_goal_per_session() {
    let dir = tempfile::tempdir().unwrap();
    // Complete schema initialization before the workers synchronize. The race
    // below deliberately covers admission only.
    SqliteTaskStore::new(dir.path()).unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let mut joins = Vec::new();
    for id in ["goal-a", "goal-b"] {
        let path = dir.path().to_owned();
        let barrier = Arc::clone(&barrier);
        joins.push(std::thread::spawn(move || {
            let store = SqliteTaskStore::new(&path).unwrap();
            barrier.wait();
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(
                    store.create_or_replace_session_goal(task(id, "matrix:room:thread"), goal(id)),
                )
                .unwrap()
        }));
    }

    let results = joins
        .into_iter()
        .map(|join| join.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        results
            .iter()
            .filter(|result| **result == GoalTransitionResult::Applied)
            .count(),
        1
    );
    assert_eq!(
        results
            .iter()
            .filter(|result| **result == GoalTransitionResult::Stale)
            .count(),
        1
    );

    let store = SqliteTaskStore::new(dir.path()).unwrap();
    let current = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(store.current_goal_for_session("matrix:room:thread"))
        .unwrap()
        .unwrap();
    assert_eq!(current.execution_epoch, 1);
    assert!(matches!(current.id.as_str(), "goal-a" | "goal-b"));
}

#[test]
fn concurrent_first_open_serializes_schema_migration() {
    let dir = tempfile::tempdir().unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let mut joins = Vec::new();
    for _ in 0..2 {
        let path = dir.path().to_owned();
        let barrier = Arc::clone(&barrier);
        joins.push(std::thread::spawn(move || {
            barrier.wait();
            SqliteTaskStore::new(&path)
        }));
    }

    for join in joins {
        join.join()
            .expect("first-open worker must not panic")
            .expect("concurrent first-open migration must succeed");
    }

    let conn = rusqlite::Connection::open(dir.path().join("control_plane.db")).unwrap();
    let version: i64 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(version, 9);
    let session_id_column = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('tasks') WHERE name = 'session_id'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap();
    assert_eq!(session_id_column, 1);
}

#[tokio::test]
async fn generic_task_mutations_cannot_bypass_session_goal_guards() {
    let store = SqliteTaskStore::new_in_memory().unwrap();
    assert_eq!(
        store
            .create_or_replace_session_goal(
                task("goal-guarded", "matrix:room:thread"),
                goal("goal-guarded")
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );

    for error in [
        store
            .heartbeat("goal-guarded", "test-boot")
            .await
            .expect_err("generic heartbeat must reject a session Goal"),
        store
            .update_status(
                "goal-guarded",
                TaskStatus::Completed,
                Some("must not be written".into()),
                None,
            )
            .await
            .expect_err("generic status update must reject a session Goal"),
        store
            .claim_owner("goal-guarded", 2, "other-boot")
            .await
            .expect_err("generic owner claim must reject a session Goal"),
        store
            .reconcile_lost("goal-guarded", "other-boot")
            .await
            .expect_err("generic loss reconciliation must reject a session Goal"),
        store
            .delete_by_agent("main")
            .expect_err("generic agent deletion must reject a session Goal"),
    ] {
        assert!(format!("{error:#}").contains("session-bound Goal"));
    }

    let retained = store.get("goal-guarded").await.unwrap().unwrap();
    assert_eq!(retained.status, TaskStatus::Running);
    assert_eq!(retained.execution_epoch, 1);
}

#[test]
fn failed_migration_rolls_back_schema_and_version() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("control_plane.db");
    let conn = rusqlite::Connection::open(&path).unwrap();
    conn.execute_batch(
        "CREATE TABLE tasks (
             id TEXT PRIMARY KEY, kind TEXT NOT NULL, agent TEXT NOT NULL, status TEXT NOT NULL,
             started_at TEXT NOT NULL
         );
         CREATE VIEW goal_tasks AS SELECT id AS task_id FROM tasks;
         PRAGMA user_version = 7;",
    )
    .unwrap();
    drop(conn);

    assert!(
        SqliteTaskStore::new(dir.path()).is_err(),
        "a malformed legacy Goal extension must fail"
    );

    let conn = rusqlite::Connection::open(path).unwrap();
    let version: i64 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(version, 7, "failed migrations must not advance the version");
    let session_id_column = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('tasks') WHERE name = 'session_id'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap();
    assert_eq!(
        session_id_column, 0,
        "failed migrations must roll back prior steps"
    );
}

#[tokio::test]
async fn migration_fails_nonterminal_legacy_goals_without_erasing_terminal_audit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("control_plane.db");
    let conn = rusqlite::Connection::open(&path).unwrap();
    conn.execute_batch(
        "CREATE TABLE tasks (
             id TEXT PRIMARY KEY, kind TEXT NOT NULL, agent TEXT NOT NULL, status TEXT NOT NULL,
             owner_pid INTEGER NOT NULL DEFAULT 0, owner_boot_id TEXT NOT NULL DEFAULT '',
             heartbeat_at TEXT, depth INTEGER NOT NULL DEFAULT 0, parent_id TEXT,
             originator_route TEXT, delivered INTEGER NOT NULL DEFAULT 0, idem_key TEXT,
             principal_id TEXT, started_at TEXT NOT NULL, finished_at TEXT, output TEXT, error TEXT
         );
         CREATE TABLE goal_tasks (
             task_id TEXT PRIMARY KEY, objective TEXT NOT NULL,
             effective_token_limit INTEGER, effective_cost_limit_usd REAL,
             pause_reason TEXT, pause_description TEXT, blockers_json TEXT NOT NULL DEFAULT '[]'
         );
         PRAGMA user_version = 7;",
    )
    .unwrap();
    for (id, status) in [
        ("legacy-running", "running"),
        ("legacy-terminal", "completed"),
    ] {
        conn.execute(
            "INSERT INTO tasks (id, kind, agent, status, started_at)
             VALUES (?1, 'goal', 'main', ?2, '2026-01-01T00:00:00Z')",
            params![id, status],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO goal_tasks (task_id, objective) VALUES (?1, 'legacy objective')",
            params![id],
        )
        .unwrap();
    }
    drop(conn);

    let store = SqliteTaskStore::new(dir.path()).unwrap();
    assert_eq!(
        store.get("legacy-running").await.unwrap().unwrap().status,
        TaskStatus::Failed
    );
    let legacy_running = store.get("legacy-running").await.unwrap().unwrap();
    assert!(
        chrono::DateTime::parse_from_rfc3339(legacy_running.finished_at.as_deref().unwrap())
            .is_ok()
    );
    assert_eq!(
        store.get("legacy-terminal").await.unwrap().unwrap().status,
        TaskStatus::Completed
    );
    assert!(
        store
            .current_goal_for_session("matrix:any-session")
            .await
            .unwrap()
            .is_none()
    );
}
