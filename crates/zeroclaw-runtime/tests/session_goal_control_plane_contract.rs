//! File-backed contract coverage for session-bound Goal durable state.

use std::sync::{Arc, Barrier};

use rusqlite::Connection;
use zeroclaw_runtime::control_plane::{
    GoalAccountingState, GoalTaskRecord, GoalTaskRegistry, GoalToolPhase, GoalTransitionResult,
    SqliteTaskStore, TaskKind, TaskRecord, TaskRegistry, TaskStatus,
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
            started_at: "2026-08-24T00:00:00Z".into(),
            finished_at: None,
        },
        GoalTaskRecord {
            task_id: id.into(),
            objective: "finish the task".into(),
            success_criteria: "the task is complete".into(),
            effective_token_limit: Some(10),
            ..GoalTaskRecord::default()
        },
    )
}

fn create_v7_goal_database(path: &std::path::Path, with_session_id: bool) {
    let conn = Connection::open(path.join("control_plane.db")).unwrap();
    let session_column = if with_session_id {
        ", session_id TEXT"
    } else {
        ""
    };
    conn.execute_batch(&format!(
        "CREATE TABLE tasks (
             id TEXT PRIMARY KEY, kind TEXT NOT NULL, agent TEXT NOT NULL,
             status TEXT NOT NULL, owner_pid INTEGER NOT NULL DEFAULT 0,
             owner_boot_id TEXT NOT NULL DEFAULT '', heartbeat_at TEXT,
             depth INTEGER NOT NULL DEFAULT 0, parent_id TEXT,
             originator_route TEXT, delivered INTEGER NOT NULL DEFAULT 0,
             idem_key TEXT, principal_id TEXT{session_column}, started_at TEXT NOT NULL,
             finished_at TEXT, output TEXT, error TEXT
         );
         CREATE TABLE goal_tasks (
             task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,
             objective TEXT NOT NULL, effective_token_limit INTEGER,
             effective_cost_limit_usd REAL, pause_reason TEXT,
             pause_description TEXT, blockers_json TEXT NOT NULL DEFAULT '[]'
         );
         PRAGMA user_version = 7;"
    ))
    .unwrap();
}

#[test]
fn independent_connections_admit_one_goal_and_one_pending_operation() {
    let directory = tempfile::tempdir().unwrap();
    SqliteTaskStore::new(directory.path()).unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let starts: Vec<_> = ["left", "right"]
        .into_iter()
        .map(|id| {
            let path = directory.path().to_owned();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let store = SqliteTaskStore::new(&path).unwrap();
                barrier.wait();
                tokio::runtime::Runtime::new().unwrap().block_on(
                    store.create_or_replace_session_goal(
                        session_goal(id, "race").0,
                        session_goal(id, "race").1,
                    ),
                )
            })
        })
        .collect();
    let outcomes: Vec<_> = starts
        .into_iter()
        .map(|thread| thread.join().unwrap().unwrap())
        .collect();
    assert_eq!(
        outcomes
            .iter()
            .filter(|result| **result == GoalTransitionResult::Applied)
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|result| **result == GoalTransitionResult::Stale)
            .count(),
        1
    );

    let winner = SqliteTaskStore::new(directory.path()).unwrap();
    let winner_id = tokio::runtime::Runtime::new().unwrap().block_on(async {
        winner
            .current_goal_for_session("race")
            .await
            .unwrap()
            .unwrap()
            .id
    });
    drop(winner);
    let barrier = Arc::new(Barrier::new(2));
    let calls: Vec<_> = ["call-left", "call-right"]
        .into_iter()
        .map(|call_id| {
            let path = directory.path().to_owned();
            let winner_id = winner_id.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let store = SqliteTaskStore::new(&path).unwrap();
                barrier.wait();
                tokio::runtime::Runtime::new().unwrap().block_on(
                    store.admit_pending_call_with_reservation(
                        &winner_id,
                        "race",
                        1,
                        Some(10),
                        None,
                        call_id,
                        Some(1),
                        None,
                    ),
                )
            })
        })
        .collect();
    let outcomes: Vec<_> = calls
        .into_iter()
        .map(|thread| thread.join().unwrap().unwrap())
        .collect();
    assert_eq!(
        outcomes
            .iter()
            .filter(|result| **result == GoalTransitionResult::Applied)
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|result| **result == GoalTransitionResult::Stale)
            .count(),
        1
    );
}

#[test]
fn migration_failure_rolls_back_and_prior_schema_is_reconciled() {
    let directory = tempfile::tempdir().unwrap();
    create_v7_goal_database(directory.path(), true);
    let path = directory.path().join("control_plane.db");
    let conn = Connection::open(&path).unwrap();
    for id in ["first", "second"] {
        conn.execute("INSERT INTO tasks (id, kind, agent, status, session_id, started_at) VALUES (?1, 'goal', 'main', 'completed', 'same', 'now')", [id]).unwrap();
        conn.execute(
            "INSERT INTO goal_tasks (task_id, objective) VALUES (?1, 'done')",
            [id],
        )
        .unwrap();
    }
    drop(conn);
    assert!(SqliteTaskStore::new(directory.path()).is_err());
    let conn = Connection::open(path).unwrap();
    assert_eq!(
        conn.query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        7
    );
    assert!(
        !conn
            .prepare("PRAGMA table_info(tasks)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .any(|name| matches!(name, Ok(name) if name == "execution_epoch"))
    );
}

#[tokio::test]
async fn terminal_inflight_goal_can_be_replaced_after_its_epoch_is_fenced() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("terminal", "terminal-session");
    assert_eq!(
        store
            .create_or_replace_session_goal(task, goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .begin_goal_tool_phase("terminal", "terminal-session", 1)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .finish_session_goal("terminal", "terminal-session", 1, TaskStatus::Failed, None)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    let (replacement, replacement_goal) = session_goal("replacement", "terminal-session");
    assert_eq!(
        store
            .create_or_replace_session_goal(replacement, replacement_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .current_goal_for_session("terminal-session")
            .await
            .unwrap()
            .unwrap()
            .id,
        "replacement"
    );
}

#[tokio::test]
async fn boot_classifies_pending_operations_after_pause_or_terminal_fencing() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();

    let (paused, paused_goal) = session_goal("paused", "paused-session");
    store
        .create_or_replace_session_goal(paused, paused_goal)
        .await
        .unwrap();
    store
        .admit_pending_call_with_reservation(
            "paused",
            "paused-session",
            1,
            Some(10),
            None,
            "paused-call",
            Some(1),
            None,
        )
        .await
        .unwrap();
    store
        .pause_session_goal(
            "paused",
            "paused-session",
            1,
            zeroclaw_runtime::control_plane::GoalPauseState {
                reason: zeroclaw_runtime::control_plane::GoalPauseReason::OperatorPaused,
                description: None,
                blockers: vec![],
            },
        )
        .await
        .unwrap();

    let (terminal, terminal_goal) = session_goal("terminal", "terminal-session");
    store
        .create_or_replace_session_goal(terminal, terminal_goal)
        .await
        .unwrap();
    store
        .admit_pending_call_with_reservation(
            "terminal",
            "terminal-session",
            1,
            Some(10),
            None,
            "terminal-call",
            Some(1),
            None,
        )
        .await
        .unwrap();
    store
        .finish_session_goal(
            "terminal",
            "terminal-session",
            1,
            TaskStatus::Cancelled,
            None,
        )
        .await
        .unwrap();
    drop(store);

    let reopened = SqliteTaskStore::new(directory.path()).unwrap();
    assert_eq!(reopened.reconcile_goal_boot_state("boot-b").unwrap(), 2);
    for (id, session_id) in [
        ("paused", "paused-session"),
        ("terminal", "terminal-session"),
    ] {
        let task = reopened.get(id).await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(
            reopened
                .terminal_reason_for_session_goal(id, session_id)
                .await
                .unwrap()
                .as_deref(),
            Some("accounting_outcome_unknown")
        );
        let goal = reopened.get_goal_task(id).await.unwrap().unwrap();
        assert_eq!(goal.accounting_state, GoalAccountingState::OutcomeUnknown);
        assert_eq!(goal.pending_call_id, None);
        assert_eq!(goal.pending_call_epoch, None);
    }
    let (replacement, replacement_goal) = session_goal("replacement", "terminal-session");
    assert_eq!(
        reopened
            .create_or_replace_session_goal(replacement, replacement_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
}

#[tokio::test]
async fn guarded_phase_is_independently_observable_and_legacy_mutators_cannot_bypass_it() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("phase", "phase-session");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    assert_eq!(
        store
            .begin_goal_tool_phase("phase", "phase-session", 1)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    let observer = SqliteTaskStore::new(directory.path()).unwrap();
    assert_eq!(
        observer
            .get_goal_task("phase")
            .await
            .unwrap()
            .unwrap()
            .tool_phase,
        GoalToolPhase::InFlight
    );
    assert!(
        store
            .update_status("phase", TaskStatus::Paused, None, None)
            .await
            .is_err()
    );
    assert!(
        store
            .update_goal_limits("phase", Some(2), None)
            .await
            .is_err()
    );
    assert_eq!(
        store
            .complete_goal_tool_phase("phase", "phase-session", 1)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        observer
            .get_goal_task("phase")
            .await
            .unwrap()
            .unwrap()
            .tool_phase,
        GoalToolPhase::Clean
    );
}

#[test]
fn sqlite_rejects_hostile_pending_and_session_mutations() {
    let directory = tempfile::tempdir().unwrap();
    SqliteTaskStore::new(directory.path()).unwrap();
    let conn = Connection::open(directory.path().join("control_plane.db")).unwrap();
    assert!(conn.execute("INSERT INTO tasks (id, kind, agent, status, session_id, started_at) VALUES ('blank', 'goal', 'main', 'running', ' ', 'now')", []).is_err());
    conn.execute("INSERT INTO tasks (id, kind, agent, status, session_id, execution_epoch, started_at) VALUES ('goal', 'goal', 'main', 'running', 'session', 1, 'now')", []).unwrap();
    assert!(conn.execute("INSERT INTO goal_tasks (task_id, objective, success_criteria, pending_call_id) VALUES ('goal', 'o', 'c', 'call')", []).is_err());
    conn.execute("INSERT INTO goal_tasks (task_id, objective, success_criteria, pending_call_id, pending_call_epoch, pending_reserved_tokens) VALUES ('goal', 'o', 'c', 'call', 1, 1)", []).unwrap();
    assert!(
        conn.execute(
            "UPDATE goal_tasks SET pending_call_epoch = 'not-an-epoch' WHERE task_id = 'goal'",
            []
        )
        .is_err()
    );
    assert!(
        conn.execute(
            "UPDATE goal_tasks SET pending_call_epoch = 1.5 WHERE task_id = 'goal'",
            []
        )
        .is_err()
    );
    assert!(
        conn.execute(
            "UPDATE goal_tasks SET pending_reserved_tokens = -1 WHERE task_id = 'goal'",
            []
        )
        .is_err()
    );
    assert!(
        conn.execute(
            "UPDATE goal_tasks SET pending_reserved_tokens = 'not-tokens' WHERE task_id = 'goal'",
            []
        )
        .is_err()
    );
    assert!(
        conn.execute(
            "UPDATE goal_tasks SET pending_reserved_tokens = 1.5 WHERE task_id = 'goal'",
            []
        )
        .is_err()
    );
    assert!(
        conn.execute(
            "UPDATE goal_tasks SET pending_reserved_cost_usd = -0.1 WHERE task_id = 'goal'",
            []
        )
        .is_err()
    );
    assert!(
        conn.execute(
            "UPDATE goal_tasks SET pending_reserved_cost_usd = 1e999 WHERE task_id = 'goal'",
            []
        )
        .is_err()
    );
    assert!(
        conn.execute(
            "UPDATE tasks SET session_id = 'other' WHERE id = 'goal'",
            []
        )
        .is_err()
    );
    assert!(conn.execute("UPDATE goal_tasks SET pending_call_id = NULL, pending_call_epoch = NULL WHERE task_id = 'goal'", []).is_err());
    assert_eq!(
        conn.query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        10
    );
    drop(conn);
    let reopened = SqliteTaskStore::new(directory.path()).unwrap();
    drop(reopened);
    let conn = Connection::open(directory.path().join("control_plane.db")).unwrap();
    assert_eq!(
        conn.query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        10
    );
}

#[tokio::test]
async fn restart_fails_saturated_epoch_and_normalizes_blank_terminal_legacy_binding() {
    let directory = tempfile::tempdir().unwrap();
    create_v7_goal_database(directory.path(), true);
    let path = directory.path().join("control_plane.db");
    let conn = Connection::open(&path).unwrap();
    conn.execute("INSERT INTO tasks (id, kind, agent, status, session_id, started_at, finished_at) VALUES ('blank', 'goal', 'main', 'completed', '  ', 'now', 'then')", []).unwrap();
    conn.execute(
        "INSERT INTO goal_tasks (task_id, objective) VALUES ('blank', 'old')",
        [],
    )
    .unwrap();
    drop(conn);
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    assert_eq!(store.get("blank").await.unwrap().unwrap().session_id, None);
    let (task, goal) = session_goal("saturated", "saturated-session");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    drop(store);
    let conn = Connection::open(path).unwrap();
    conn.execute(
        "UPDATE tasks SET execution_epoch = ?1 WHERE id = 'saturated'",
        [i64::MAX],
    )
    .unwrap();
    drop(conn);
    let reopened = SqliteTaskStore::new(directory.path()).unwrap();
    reopened.reconcile_goal_boot_state("boot-b").unwrap();
    assert_eq!(
        reopened.get("saturated").await.unwrap().unwrap().status,
        TaskStatus::Failed
    );
}

#[tokio::test]
async fn duplicate_task_id_does_not_create_a_phantom_session_goal() {
    let store = SqliteTaskStore::new_in_memory().unwrap();
    let (first, goal) = session_goal("shared", "first");
    store
        .create_or_replace_session_goal(first, goal)
        .await
        .unwrap();
    store
        .finish_session_goal("shared", "first", 1, TaskStatus::Completed, None)
        .await
        .unwrap();
    let (duplicate, goal) = session_goal("shared", "second");
    assert_eq!(
        store
            .create_or_replace_session_goal(duplicate, goal)
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
    assert!(
        store
            .current_goal_for_session("second")
            .await
            .unwrap()
            .is_none()
    );
}
