//! Compile-time contract for the durable control-plane's stable runtime seam.

use zeroclaw_control_plane::{
    ControlPlaneHandle, GoalAccountingState, GoalTaskRecord, GoalTaskRegistry,
    GoalTransitionResult, SqliteTaskStore, TaskKind, TaskRecord, TaskRegistry, TaskStatus,
    control_plane, init_control_plane,
};

#[test]
fn exposes_the_canonical_control_plane_contract() {
    let _ = ControlPlaneHandle::start;
    let _ = SqliteTaskStore::new_in_memory;
    let _ = control_plane;
    let _ = init_control_plane;
    let _: Option<&dyn TaskRegistry> = None;
    let _: Option<&dyn GoalTaskRegistry> = None;
}

fn session_goal(id: &str, session_id: &str) -> (TaskRecord, GoalTaskRecord) {
    (
        TaskRecord {
            id: id.into(),
            kind: TaskKind::Goal,
            agent: "main".into(),
            status: TaskStatus::Running,
            owner_pid: 1,
            owner_boot_id: "boot-1".into(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: Some("route".into()),
            delivered: false,
            idem_key: None,
            principal_id: Some("principal".into()),
            session_id: Some(session_id.into()),
            execution_epoch: 0,
            started_at: "2026-08-09T00:00:00Z".into(),
            finished_at: None,
        },
        GoalTaskRecord {
            task_id: id.into(),
            objective: "durable test".into(),
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
}

fn v7_legacy_goal_store() -> (tempfile::TempDir, SqliteTaskStore) {
    use rusqlite::Connection;
    let dir = tempfile::tempdir().unwrap();
    let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
    db.execute_batch("CREATE TABLE tasks (id TEXT PRIMARY KEY, kind TEXT NOT NULL, agent TEXT NOT NULL, status TEXT NOT NULL, owner_pid INTEGER NOT NULL DEFAULT 0, owner_boot_id TEXT NOT NULL DEFAULT '', heartbeat_at TEXT, depth INTEGER NOT NULL DEFAULT 0, parent_id TEXT, originator_route TEXT, delivered INTEGER NOT NULL DEFAULT 0, idem_key TEXT, principal_id TEXT, started_at TEXT NOT NULL, finished_at TEXT, output TEXT, error TEXT); CREATE TABLE goal_tasks (task_id TEXT PRIMARY KEY, objective TEXT NOT NULL, effective_token_limit INTEGER, effective_cost_limit_usd REAL, pause_reason TEXT, pause_description TEXT, blockers_json TEXT NOT NULL DEFAULT '[]'); INSERT INTO tasks (id,kind,agent,status,started_at) VALUES ('legacy-goal','goal','main','completed','now'); INSERT INTO goal_tasks (task_id,objective) VALUES ('legacy-goal','legacy'); PRAGMA user_version=7;").unwrap();
    drop(db);
    let store = SqliteTaskStore::new(dir.path()).unwrap();
    (dir, store)
}

#[test]
fn session_goal_admission_is_fenced_across_file_backed_connections() {
    use std::sync::{Arc, Barrier};
    let dir = tempfile::tempdir().unwrap();
    let left = Arc::new(SqliteTaskStore::new(dir.path()).unwrap());
    let right = Arc::new(SqliteTaskStore::new(dir.path()).unwrap());
    let (left_task, left_goal) = session_goal("left", "session-a");
    let (right_task, right_goal) = session_goal("right", "session-a");
    let barrier = Arc::new(Barrier::new(3));
    let spawn = |store: Arc<SqliteTaskStore>, task, goal, barrier: Arc<Barrier>| {
        std::thread::spawn(move || {
            barrier.wait();
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(store.create_or_replace_session_goal(task, goal))
        })
    };
    let first = spawn(
        Arc::clone(&left),
        left_task,
        left_goal,
        Arc::clone(&barrier),
    );
    let second = spawn(
        Arc::clone(&right),
        right_task,
        right_goal,
        Arc::clone(&barrier),
    );
    barrier.wait();
    let results = [
        first.join().unwrap().unwrap(),
        second.join().unwrap().unwrap(),
    ];
    assert_eq!(
        results
            .iter()
            .filter(|value| **value == GoalTransitionResult::Applied)
            .count(),
        1
    );
    assert_eq!(
        results
            .iter()
            .filter(|value| **value == GoalTransitionResult::Stale)
            .count(),
        1
    );
    let current = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(left.current_goal_for_session("session-a"))
        .unwrap()
        .unwrap();
    assert_eq!(current.execution_epoch, 1);
}

#[tokio::test]
async fn raw_replace_cannot_orphan_a_terminal_session_goal_extension() {
    use rusqlite::Connection;

    let dir = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(dir.path()).unwrap();
    let (predecessor, predecessor_goal) = session_goal("predecessor", "session");
    assert_eq!(
        store
            .create_or_replace_session_goal(predecessor, predecessor_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .finish_session_goal("predecessor", "session", 1, TaskStatus::Completed, None)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );

    let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
    db.execute_batch("PRAGMA foreign_keys = OFF; PRAGMA recursive_triggers = OFF;")
        .unwrap();
    let foreign_keys: i64 = db
        .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
        .unwrap();
    let recursive_triggers: i64 = db
        .query_row("PRAGMA recursive_triggers", [], |row| row.get(0))
        .unwrap();
    assert_eq!(foreign_keys, 0);
    assert_eq!(recursive_triggers, 0);

    let error = db
        .execute(
            "INSERT OR REPLACE INTO tasks \
         (id, kind, agent, status, session_id, execution_epoch, started_at) \
         VALUES ('replacement', 'goal', 'main', 'running', 'session', 1, 'now')",
            [],
        )
        .expect_err("raw replacement must not orphan the terminal goal extension");
    assert!(format!("{error}").contains("session goal replacement requires extension disposal"));

    let predecessor_extensions: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM goal_tasks WHERE task_id = 'predecessor'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(predecessor_extensions, 1);
    let predecessor_exists: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM tasks WHERE id = 'predecessor'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(predecessor_exists, 1);
}

#[test]
fn raw_goal_extension_replace_cannot_implicitly_delete_an_existing_extension() {
    use rusqlite::Connection;

    for replacement in [
        "INSERT OR REPLACE INTO goal_tasks \
         (task_id, objective, pending_call_id, pending_call_epoch) \
         VALUES ('task-id-owner', 'replaced', NULL, NULL)",
        "REPLACE INTO goal_tasks \
         (task_id, objective, pending_call_id, pending_call_epoch) \
         VALUES ('task-id-owner', 'replaced', NULL, NULL)",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteTaskStore::new(dir.path()).unwrap();
        drop(store);
        let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
        db.execute_batch(
            "DROP TRIGGER trg_goal_tasks_prevent_conflict_replace;
             PRAGMA user_version = 14;",
        )
        .unwrap();
        drop(db);
        let store = SqliteTaskStore::new(dir.path()).unwrap();
        drop(store);
        let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
        db.execute_batch(
            "PRAGMA foreign_keys = OFF;
             PRAGMA recursive_triggers = OFF;
             INSERT INTO tasks (id, kind, agent, status, session_id, execution_epoch, started_at)
                VALUES
                  ('task-id-owner', 'goal', 'main', 'running', 'task-id-session', 1, 'now'),
                  ('pending-call-owner', 'goal', 'main', 'running', 'pending-call-session', 1, 'now'),
                  ('pending-call-candidate', 'goal', 'main', 'running', 'candidate-session', 1, 'now');
             INSERT INTO goal_tasks (task_id, objective, pending_call_id, pending_call_epoch)
                VALUES
                  ('task-id-owner', 'task-id incumbent', NULL, NULL),
                  ('pending-call-owner', 'pending-call incumbent', 'call-1', 1);",
        )
        .unwrap();

        let error = db
            .execute(replacement, [])
            .expect_err("raw goal extension replacement must be rejected before conflict deletion");
        assert!(
            format!("{error}")
                .contains("goal extension replacement requires control-plane disposal")
        );

        let task_id_incumbent: (String, Option<String>) = db
            .query_row(
                "SELECT objective, pending_call_id FROM goal_tasks WHERE task_id = 'task-id-owner'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(task_id_incumbent.0, "task-id incumbent");
        assert_eq!(task_id_incumbent.1, None);

        let pending_call_replacement = replacement.replace(
            "'task-id-owner', 'replaced', NULL, NULL",
            "'pending-call-candidate', 'replaced', 'call-1', 1",
        );
        let error = db
            .execute(&pending_call_replacement, [])
            .expect_err("raw pending-call replacement must be rejected before conflict deletion");
        assert!(
            format!("{error}")
                .contains("goal extension replacement requires control-plane disposal")
        );

        let pending_call_incumbent: (String, String) = db
            .query_row(
                "SELECT objective, pending_call_id FROM goal_tasks WHERE task_id = 'pending-call-owner'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            pending_call_incumbent,
            ("pending-call incumbent".into(), "call-1".into())
        );
        assert_eq!(
            db.query_row(
                "SELECT COUNT(*) FROM goal_tasks WHERE task_id = 'pending-call-candidate'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
            0,
            "the rejected incoming extension must not persist"
        );
    }
}

#[tokio::test]
async fn raw_task_id_rewrite_cannot_swap_or_orphan_goal_extensions() {
    use rusqlite::{Connection, params};

    let dir = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(dir.path()).unwrap();
    let (predecessor, predecessor_goal) = session_goal("predecessor", "predecessor-session");
    assert_eq!(
        store
            .create_or_replace_session_goal(predecessor, predecessor_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    let (replacement, replacement_goal) = session_goal("replacement", "replacement-session");
    assert_eq!(
        store
            .create_or_replace_session_goal(replacement, replacement_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    drop(store);

    // Model a durable V10 database. The hostile connection below must be unable
    // to reinterpret either goal extension by replacing one task ID with another.
    let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
    db.execute_batch(
        "DROP TRIGGER IF EXISTS trg_tasks_id_immutable;
         PRAGMA user_version = 10;",
    )
    .unwrap();
    drop(db);

    let store = SqliteTaskStore::new(dir.path()).unwrap();
    let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
    db.execute_batch("PRAGMA foreign_keys = OFF; PRAGMA recursive_triggers = OFF;")
        .unwrap();
    assert_eq!(
        db.query_row("PRAGMA foreign_keys", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        db.query_row("PRAGMA recursive_triggers", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        0
    );

    let error = db
        .execute(
            "UPDATE OR REPLACE tasks SET id = ?1 WHERE id = ?2",
            params!["predecessor", "replacement"],
        )
        .expect_err("task IDs must remain immutable across raw replace updates");
    assert!(format!("{error}").contains("task id is immutable"));

    let error = db
        .execute(
            "UPDATE OR REPLACE tasks SET id = NULL WHERE id = ?1",
            params!["replacement"],
        )
        .expect_err("task IDs must remain immutable when a raw replace update assigns NULL");
    assert!(format!("{error}").contains("task id is immutable"));

    let error = db
        .execute(
            "UPDATE goal_tasks SET task_id = NULL WHERE task_id = ?1",
            params!["predecessor"],
        )
        .expect_err("goal extension IDs must remain immutable when a raw update assigns NULL");
    assert!(format!("{error}").contains("goal task_id is immutable"));
    drop(db);

    for (id, session_id) in [
        ("predecessor", "predecessor-session"),
        ("replacement", "replacement-session"),
    ] {
        let task = store
            .get(id)
            .await
            .unwrap()
            .expect("task must remain present");
        assert_eq!(task.session_id.as_deref(), Some(session_id));
        assert!(
            store.get_goal_task(id).await.unwrap().is_some(),
            "{id} goal extension must remain attached to its task"
        );
    }
}

#[tokio::test]
async fn epochs_fence_goal_mutations_and_pending_calls() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(dir.path()).unwrap();
    let (task, goal) = session_goal("goal", "session-a");
    assert_eq!(
        store
            .create_or_replace_session_goal(task, goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .admit_pending_call("goal", "session-a", 1, "call-a")
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .pause_session_goal(
                "goal",
                "session-a",
                1,
                zeroclaw_control_plane::GoalPauseState {
                    reason: zeroclaw_control_plane::GoalPauseReason::OperatorPaused,
                    description: None,
                    blockers: Vec::new(),
                }
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .resume_session_goal("goal", "session-a", 2, 9, "boot-2")
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
    assert_eq!(
        store
            .settle_pending_call(
                "goal",
                "session-a",
                1,
                "call-a",
                GoalAccountingState::Complete,
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .resume_session_goal("goal", "session-a", 2, 9, "boot-2")
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .admit_pending_call("goal", "session-a", 3, "call-b")
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .finish_session_goal("goal", "session-a", 1, TaskStatus::Cancelled, None)
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
    assert_eq!(
        store
            .finish_session_goal("goal", "session-a", 3, TaskStatus::Cancelled, None)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    let current = store
        .current_goal_for_session("session-a")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(current.execution_epoch, 4);
}

#[tokio::test]
async fn restart_recovery_fails_paused_pending_accounting_and_rejects_stale_settlement() {
    let dir = tempfile::tempdir().unwrap();
    let (task, goal) = session_goal("goal", "session-a");
    let store = SqliteTaskStore::new(dir.path()).unwrap();
    assert_eq!(
        store
            .create_or_replace_session_goal(task, goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .admit_pending_call("goal", "session-a", 1, "call-a")
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .pause_session_goal(
                "goal",
                "session-a",
                1,
                zeroclaw_control_plane::GoalPauseState {
                    reason: zeroclaw_control_plane::GoalPauseReason::OperatorPaused,
                    description: None,
                    blockers: Vec::new(),
                },
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    drop(store);

    let recovered = ControlPlaneHandle::start_with_boot_id(dir.path(), "boot-restarted".into())
        .await
        .unwrap();
    let task = recovered.store.get("goal").await.unwrap().unwrap();
    assert_eq!(task.status, TaskStatus::Failed);
    assert_eq!(task.execution_epoch, 3);
    let db = rusqlite::Connection::open(dir.path().join("control_plane.db")).unwrap();
    let error: Option<String> = db
        .query_row("SELECT error FROM tasks WHERE id = 'goal'", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(error.as_deref(), Some("accounting_outcome_unknown"));

    let stale = SqliteTaskStore::new(dir.path()).unwrap();
    assert_eq!(
        stale
            .settle_pending_call(
                "goal",
                "session-a",
                1,
                "call-a",
                GoalAccountingState::Complete,
            )
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
    let goal = stale.get_goal_task("goal").await.unwrap().unwrap();
    assert_eq!(goal.accounting_state, GoalAccountingState::OutcomeUnknown);
    assert_eq!(goal.pending_call_id.as_deref(), Some("call-a"));
    assert_eq!(goal.pending_call_epoch, Some(1));
    assert_eq!(
        stale
            .resume_session_goal("goal", "session-a", 3, 2, "boot-restarted")
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
}

#[tokio::test]
async fn pending_settlement_persists_reconciliation_state_and_gates_readmission() {
    for (suffix, state, can_readmit) in [
        ("complete", GoalAccountingState::Complete, true),
        ("missing", GoalAccountingState::Missing, false),
        ("invalid", GoalAccountingState::Invalid, false),
    ] {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let id = format!("goal-{suffix}");
        let session = format!("session-{suffix}");
        let (task, goal) = session_goal(&id, &session);
        assert_eq!(
            store
                .create_or_replace_session_goal(task, goal)
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            store
                .admit_pending_call(&id, &session, 1, "call")
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            store
                .settle_pending_call(&id, &session, 1, "call", state)
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            store
                .get_goal_task(&id)
                .await
                .unwrap()
                .unwrap()
                .accounting_state,
            state
        );
        assert_eq!(
            store
                .admit_pending_call(&id, &session, 1, "next")
                .await
                .unwrap(),
            if can_readmit {
                GoalTransitionResult::Applied
            } else {
                GoalTransitionResult::Stale
            }
        );
    }
}

#[tokio::test]
async fn completion_requires_resolved_complete_accounting() {
    for (suffix, settlement) in [
        ("missing", Some(GoalAccountingState::Missing)),
        ("invalid", Some(GoalAccountingState::Invalid)),
        ("unknown", Some(GoalAccountingState::OutcomeUnknown)),
        ("pending", None),
    ] {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let id = format!("completion-{suffix}");
        let session = format!("session-{suffix}");
        let (task, goal) = session_goal(&id, &session);
        assert_eq!(
            store
                .create_or_replace_session_goal(task, goal)
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            store
                .admit_pending_call(&id, &session, 1, "call")
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        if let Some(state) = settlement {
            assert_eq!(
                store
                    .settle_pending_call(&id, &session, 1, "call", state)
                    .await
                    .unwrap(),
                GoalTransitionResult::Applied
            );
        }
        assert_eq!(
            store
                .finish_session_goal(&id, &session, 1, TaskStatus::Completed, None)
                .await
                .unwrap(),
            GoalTransitionResult::Stale
        );
        assert_eq!(
            store.get(&id).await.unwrap().unwrap().status,
            TaskStatus::Running
        );
    }
    for (id, status) in [
        ("failed", TaskStatus::Failed),
        ("cancelled", TaskStatus::Cancelled),
    ] {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let (task, goal) = session_goal(id, id);
        assert_eq!(
            store
                .create_or_replace_session_goal(task, goal)
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            store.admit_pending_call(id, id, 1, "call").await.unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            store
                .finish_session_goal(id, id, 1, status, None)
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
    }
}

#[test]
fn sqlite_rejects_blank_goal_sessions_and_unpaired_or_duplicate_pending_calls() {
    use rusqlite::{Connection, params};
    let dir = tempfile::tempdir().unwrap();
    let _store = SqliteTaskStore::new(dir.path()).unwrap();
    let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
    let blank = db.execute(
        "INSERT INTO tasks (id,kind,agent,status,started_at,session_id) VALUES (?1,'goal','main','running','now',' ')",
        params!["blank"],
    );
    assert!(blank.is_err());
    db.execute("INSERT INTO tasks (id,kind,agent,status,started_at,session_id,execution_epoch) VALUES ('goal','goal','main','running','now','session',1)", []).unwrap();
    let unpaired = db.execute(
        "INSERT INTO goal_tasks (task_id,objective,pending_call_id) VALUES ('goal','x','call')",
        [],
    );
    assert!(unpaired.is_err());
    db.execute("INSERT INTO goal_tasks (task_id,objective,pending_call_id,pending_call_epoch) VALUES ('goal','x','call',1)", []).unwrap();
    db.execute("INSERT INTO tasks (id,kind,agent,status,started_at,session_id,execution_epoch) VALUES ('other','goal','main','running','now','other',1)", []).unwrap();
    let duplicate = db.execute("INSERT INTO goal_tasks (task_id,objective,pending_call_id,pending_call_epoch) VALUES ('other','x','call',1)", []);
    assert!(duplicate.is_err());

    db.execute(
        "INSERT INTO tasks (id,kind,agent,status,started_at) \
         VALUES ('ordinary','delegate','main','running','now')",
        [],
    )
    .unwrap();
    assert!(
        db.execute(
            "UPDATE tasks SET kind='goal', session_id='reclassified' WHERE id='ordinary'",
            [],
        )
        .is_err()
    );
    assert!(
        db.execute("UPDATE tasks SET kind='delegate' WHERE id='goal'", [])
            .is_err()
    );
    assert!(
        db.execute(
            "UPDATE tasks SET session_id='different-session' WHERE id='goal'",
            [],
        )
        .is_err()
    );
    db.execute("INSERT INTO tasks (id,kind,agent,status,started_at,session_id,execution_epoch) VALUES ('active','goal','main','running','now','active-session',1)", []).unwrap();
    db.execute(
        "INSERT INTO goal_tasks (task_id,objective) VALUES ('active','x')",
        [],
    )
    .unwrap();
    assert!(db.execute("INSERT OR REPLACE INTO tasks (id,kind,agent,status,started_at,session_id,execution_epoch) VALUES ('replacement','goal','main','running','now','active-session',1)", []).is_err());
    assert!(db.execute("REPLACE INTO tasks (id,kind,agent,status,started_at,session_id,execution_epoch) VALUES ('replacement-two','goal','main','running','now','active-session',1)", []).is_err());
    assert!(db.execute("INSERT OR REPLACE INTO tasks (id,kind,agent,status,started_at,session_id,execution_epoch) VALUES ('active','goal','main','running','now','session-b',1)", []).is_err());
    assert!(db.execute("REPLACE INTO tasks (id,kind,agent,status,started_at) VALUES ('active','delegate','main','running','now')", []).is_err());
    db.execute("INSERT INTO tasks (id,kind,agent,status,started_at) VALUES ('other-row','delegate','main','running','now')", []).unwrap();
    assert!(
        db.execute(
            "UPDATE OR REPLACE tasks SET id='active' WHERE id='other-row'",
            []
        )
        .is_err()
    );
    let active: (String, Option<String>) = db
        .query_row(
            "SELECT kind, session_id FROM tasks WHERE id='active'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(active.0, "goal");
    assert_eq!(active.1.as_deref(), Some("active-session"));
    let active_extension: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM goal_tasks WHERE task_id='active'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(active_extension, 1);
    db.execute("INSERT INTO tasks (id,kind,agent,status,started_at,session_id,execution_epoch) VALUES ('rebind-target','goal','main','running','now','rebind-session',1)", []).unwrap();
    assert!(
        db.execute(
            "UPDATE goal_tasks SET task_id='rebind-target' WHERE task_id='active'",
            []
        )
        .is_err()
    );

    let null_goal = db.execute(
        "INSERT INTO tasks (id,kind,agent,status,started_at,session_id,execution_epoch) \
         VALUES (NULL,'goal','main','running','now','null-session',1)",
        [],
    );
    assert!(
        null_goal.is_err(),
        "a raw NULL task ID must not create a session goal"
    );
    let null_goal_extension = db.execute(
        "INSERT INTO goal_tasks (task_id,objective) VALUES (NULL,'must not persist')",
        [],
    );
    assert!(
        null_goal_extension.is_err(),
        "a raw NULL goal extension key must not create an unowned extension"
    );
    db.execute(
        "INSERT INTO tasks (id,kind,agent,status,started_at) \
         VALUES ('ordinary-after-null-guard','delegate','main','running','now')",
        [],
    )
    .unwrap();
}

#[test]
fn v8_goal_extension_migrates_to_durable_parent_delete_guard() {
    use rusqlite::Connection;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("control_plane.db");
    let db = Connection::open(&path).unwrap();
    db.execute_batch("CREATE TABLE tasks (id TEXT PRIMARY KEY, kind TEXT NOT NULL, agent TEXT NOT NULL, status TEXT NOT NULL, owner_pid INTEGER NOT NULL DEFAULT 0, owner_boot_id TEXT NOT NULL DEFAULT '', heartbeat_at TEXT, depth INTEGER NOT NULL DEFAULT 0, parent_id TEXT, originator_route TEXT, delivered INTEGER NOT NULL DEFAULT 0, idem_key TEXT, principal_id TEXT, session_id TEXT, execution_epoch INTEGER NOT NULL DEFAULT 0, started_at TEXT NOT NULL, finished_at TEXT, output TEXT, error TEXT); CREATE TABLE goal_tasks (task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE, objective TEXT NOT NULL, effective_token_limit INTEGER, effective_cost_limit_usd REAL, pause_reason TEXT, pause_description TEXT, blockers_json TEXT NOT NULL DEFAULT '[]', pending_call_id TEXT, pending_call_epoch INTEGER, accounting_state TEXT NOT NULL DEFAULT 'complete'); INSERT INTO tasks (id,kind,agent,status,session_id,execution_epoch,started_at) VALUES ('goal','goal','main','paused','session',3,'now'); INSERT INTO goal_tasks VALUES ('goal','objective',7,1.5,'operator_paused','description','[\"blocker\"]','call',3,'missing'); PRAGMA user_version=8;").unwrap();
    drop(db);
    let _store = SqliteTaskStore::new(dir.path()).unwrap();
    let db = Connection::open(&path).unwrap();
    let row: (String, i64, f64, String, String, i64, String) = db.query_row("SELECT objective,effective_token_limit,effective_cost_limit_usd,pause_reason,pending_call_id,pending_call_epoch,accounting_state FROM goal_tasks WHERE task_id='goal'", [], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?))).unwrap();
    assert_eq!(
        row,
        (
            "objective".into(),
            7,
            1.5,
            "operator_paused".into(),
            "call".into(),
            3,
            "missing".into()
        )
    );
    assert!(
        db.execute("DELETE FROM goal_tasks WHERE task_id='goal'", [])
            .is_err()
    );
    db.execute("UPDATE tasks SET status='completed' WHERE id='goal'", [])
        .unwrap();
    db.execute_batch("PRAGMA foreign_keys = OFF;").unwrap();
    assert!(db.execute("DELETE FROM tasks WHERE id='goal'", []).is_err());
    let task_count: i64 = db
        .query_row("SELECT COUNT(*) FROM tasks WHERE id='goal'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(task_count, 1);
    let extension_count: i64 = db
        .query_row(
            "SELECT COUNT(*) FROM goal_tasks WHERE task_id='goal'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(extension_count, 1);
    db.execute_batch("PRAGMA recursive_triggers = OFF;")
        .unwrap();
    assert!(
        db.execute(
            "INSERT OR REPLACE INTO tasks \
             (id,kind,agent,status,started_at,session_id,execution_epoch) \
             VALUES ('replacement','goal','main','running','now','session',1)",
            [],
        )
        .is_err()
    );
}

#[tokio::test]
async fn v7_legacy_sessionless_goals_migrate_and_recover_without_a_bypass() {
    use rusqlite::{Connection, params};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("control_plane.db");
    let db = Connection::open(&path).unwrap();
    db.execute_batch("CREATE TABLE tasks (id TEXT PRIMARY KEY, kind TEXT NOT NULL, agent TEXT NOT NULL, status TEXT NOT NULL, owner_pid INTEGER NOT NULL DEFAULT 0, owner_boot_id TEXT NOT NULL DEFAULT '', heartbeat_at TEXT, depth INTEGER NOT NULL DEFAULT 0, parent_id TEXT, originator_route TEXT, delivered INTEGER NOT NULL DEFAULT 0, idem_key TEXT, principal_id TEXT, started_at TEXT NOT NULL, finished_at TEXT, output TEXT, error TEXT); CREATE TABLE goal_tasks (task_id TEXT PRIMARY KEY, objective TEXT NOT NULL, effective_token_limit INTEGER, effective_cost_limit_usd REAL, pause_reason TEXT, pause_description TEXT, blockers_json TEXT NOT NULL DEFAULT '[]'); PRAGMA user_version = 7;").unwrap();
    for (id, status) in [("legacy-running", "running"), ("legacy-done", "completed")] {
        db.execute("INSERT INTO tasks (id,kind,agent,status,started_at) VALUES (?1,'goal','main',?2,'now')", params![id, status]).unwrap();
        db.execute(
            "INSERT INTO goal_tasks (task_id,objective) VALUES (?1,'x')",
            params![id],
        )
        .unwrap();
    }
    drop(db);
    let _store = ControlPlaneHandle::start_with_boot_id(dir.path(), "boot".into())
        .await
        .unwrap();
    let db = Connection::open(&path).unwrap();
    let running: (String, Option<String>) = db
        .query_row(
            "SELECT status, error FROM tasks WHERE id='legacy-running'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(running.0, "failed");
    assert_eq!(running.1.as_deref(), Some("legacy_goal_missing_session"));
    let done: String = db
        .query_row("SELECT status FROM tasks WHERE id='legacy-done'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(done, "completed");
    db.execute(
        "UPDATE tasks SET status='failed', error='legacy_goal_missing_session' \
         WHERE id='legacy-done'",
        [],
    )
    .unwrap();
    assert!(
        db.execute(
            "UPDATE tasks SET session_id='rebound' WHERE id='legacy-done'",
            [],
        )
        .is_err()
    );
    db.execute("INSERT INTO tasks (id,kind,agent,status,started_at,session_id) VALUES ('bound','goal','main','completed','now','session')", []).unwrap();
    assert!(
        db.execute("UPDATE tasks SET session_id = NULL WHERE id='bound'", [])
            .is_err()
    );
    assert!(
        db.execute("UPDATE tasks SET session_id = ' ' WHERE id='bound'", [])
            .is_err()
    );
    for state in ["complete", "missing", "invalid", "outcome_unknown"] {
        db.execute(
            "UPDATE goal_tasks SET accounting_state=?1 WHERE task_id='legacy-done'",
            params![state],
        )
        .unwrap();
    }
    assert!(
        db.execute(
            "UPDATE goal_tasks SET accounting_state='wrong' WHERE task_id='legacy-done'",
            []
        )
        .is_err()
    );
}

#[tokio::test]
async fn terminal_goal_stays_current_until_atomic_replacement() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(dir.path()).unwrap();
    let (first, first_goal) = session_goal("first", "session");
    assert_eq!(
        store
            .create_or_replace_session_goal(first, first_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .finish_session_goal("first", "session", 1, TaskStatus::Completed, None)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap()
            .id,
        "first"
    );
    let (second, second_goal) = session_goal("second", "session");
    assert_eq!(
        store
            .create_or_replace_session_goal(second, second_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap()
            .id,
        "second"
    );
    assert!(store.get("first").await.unwrap().is_none());
    assert!(store.get_goal_task("first").await.unwrap().is_none());
    assert_eq!(
        store
            .finish_session_goal("second", "session", 1, TaskStatus::Completed, None)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .delete_session_goal("second", "session", 2)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert!(store.get("second").await.unwrap().is_none());
    assert!(store.get_goal_task("second").await.unwrap().is_none());
}

#[tokio::test]
async fn terminal_goal_replacement_requires_the_existing_trusted_binding() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(dir.path()).unwrap();
    let (predecessor, predecessor_goal) = session_goal("predecessor", "session");
    assert_eq!(
        store
            .create_or_replace_session_goal(predecessor, predecessor_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .finish_session_goal("predecessor", "session", 1, TaskStatus::Completed, None)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );

    for field in ["agent", "route", "principal"] {
        let replacement_id = format!("replacement-{field}");
        let (mut replacement, replacement_goal) = session_goal(&replacement_id, "session");
        match field {
            "agent" => replacement.agent = "other-agent".into(),
            "route" => replacement.originator_route = Some("other-route".into()),
            "principal" => replacement.principal_id = Some("other-principal".into()),
            _ => unreachable!("binding field is covered by the table"),
        }
        assert_eq!(
            store
                .create_or_replace_session_goal(replacement, replacement_goal)
                .await
                .unwrap(),
            GoalTransitionResult::Stale,
            "{field} mismatch must preserve the predecessor"
        );
        let current = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .expect("mismatched admission must not remove the predecessor");
        assert_eq!(current.id, "predecessor");
        assert_eq!(current.agent, "main");
        assert_eq!(current.originator_route.as_deref(), Some("route"));
        assert_eq!(current.principal_id.as_deref(), Some("principal"));
        assert!(store.get_goal_task("predecessor").await.unwrap().is_some());
        assert!(store.get(&replacement_id).await.unwrap().is_none());
    }
}

#[test]
fn terminal_goal_binding_guard_is_atomic_across_file_backed_replacements() {
    use std::sync::{Arc, Barrier};

    let dir = tempfile::tempdir().unwrap();
    let left = Arc::new(SqliteTaskStore::new(dir.path()).unwrap());
    let right = Arc::new(SqliteTaskStore::new(dir.path()).unwrap());
    let (predecessor, predecessor_goal) = session_goal("predecessor", "session");
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(left.create_or_replace_session_goal(predecessor, predecessor_goal))
        .unwrap();
    assert_eq!(
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(left.finish_session_goal(
                "predecessor",
                "session",
                1,
                TaskStatus::Completed,
                None,
            ))
            .unwrap(),
        GoalTransitionResult::Applied
    );

    let (matching_task, matching_goal) = session_goal("matching", "session");
    let (mut mismatched_task, mismatched_goal) = session_goal("mismatched", "session");
    mismatched_task.agent = "other-agent".into();
    let barrier = Arc::new(Barrier::new(3));
    let spawn = |store: Arc<SqliteTaskStore>, task, goal, barrier: Arc<Barrier>| {
        std::thread::spawn(move || {
            barrier.wait();
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(store.create_or_replace_session_goal(task, goal))
        })
    };
    let matching = spawn(
        Arc::clone(&left),
        matching_task,
        matching_goal,
        Arc::clone(&barrier),
    );
    let mismatched = spawn(
        right,
        mismatched_task,
        mismatched_goal,
        Arc::clone(&barrier),
    );
    barrier.wait();

    assert_eq!(
        matching.join().unwrap().unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        mismatched.join().unwrap().unwrap(),
        GoalTransitionResult::Stale
    );
    let current = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(left.current_goal_for_session("session"))
        .unwrap()
        .expect("matching replacement remains current");
    assert_eq!(current.id, "matching");
    assert_eq!(current.agent, "main");
}

#[tokio::test]
async fn session_goal_admission_id_collisions_are_stale_and_atomic() {
    let store = SqliteTaskStore::new_in_memory().unwrap();
    let (first, first_goal) = session_goal("shared-id", "session-a");
    assert_eq!(
        store
            .create_or_replace_session_goal(first, first_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    let (collision, collision_goal) = session_goal("shared-id", "session-b");
    assert_eq!(
        store
            .create_or_replace_session_goal(collision, collision_goal)
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
        "shared-id"
    );
    assert!(
        store
            .current_goal_for_session("session-b")
            .await
            .unwrap()
            .is_none()
    );

    let (predecessor, predecessor_goal) = session_goal("predecessor", "replaceable");
    assert_eq!(
        store
            .create_or_replace_session_goal(predecessor, predecessor_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .finish_session_goal("predecessor", "replaceable", 1, TaskStatus::Completed, None)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    let (replacement, replacement_goal) = session_goal("shared-id", "replaceable");
    assert_eq!(
        store
            .create_or_replace_session_goal(replacement, replacement_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
    assert_eq!(
        store
            .current_goal_for_session("replaceable")
            .await
            .unwrap()
            .unwrap()
            .id,
        "predecessor"
    );
}

#[tokio::test]
async fn generic_and_legacy_creation_cannot_bypass_goal_control_plane() {
    let store = SqliteTaskStore::new_in_memory().unwrap();
    let (task, goal) = session_goal("goal", "session");
    assert!(store.create(task.clone()).await.is_err());
    assert!(store.get("goal").await.unwrap().is_none());
    assert!(store.create_goal(task, goal, None).await.is_err());
    assert!(store.get("goal").await.unwrap().is_none());
}

#[tokio::test]
async fn delete_by_agent_preserves_goal_records() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(dir.path()).unwrap();
    let (task, goal) = session_goal("session-goal", "session");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    store
        .create(TaskRecord {
            id: "ordinary".into(),
            kind: TaskKind::Delegate,
            agent: "main".into(),
            status: TaskStatus::Running,
            owner_pid: 0,
            owner_boot_id: String::new(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: None,
            delivered: false,
            idem_key: None,
            principal_id: None,
            session_id: None,
            execution_epoch: 0,
            started_at: "now".into(),
            finished_at: None,
        })
        .await
        .unwrap();
    assert_eq!(store.delete_by_agent("main").unwrap(), 1);
    assert!(store.get("ordinary").await.unwrap().is_none());
    assert!(store.get("session-goal").await.unwrap().is_some());
    let (_dir, legacy_store) = v7_legacy_goal_store();
    assert_eq!(legacy_store.delete_by_agent("main").unwrap(), 0);
    assert!(legacy_store.get("legacy-goal").await.unwrap().is_some());
    assert!(
        legacy_store
            .get_goal_task("legacy-goal")
            .await
            .unwrap()
            .is_some()
    );
}
