//! File-backed contract coverage for session-bound durable Goal control state.

use std::sync::{Arc, Barrier};

use rusqlite::Connection;
use zeroclaw_runtime::control_plane::{
    ControlPlaneHandle, GoalAccountingState, GoalPauseReason, GoalPauseState, GoalTaskRecord,
    GoalTaskRegistry, GoalTransitionResult, SqliteTaskStore, TaskKind, TaskRecord, TaskRegistry,
    TaskStatus,
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
            effective_cost_limit_usd: None,
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
            pending_call_id: None,
            pending_call_epoch: None,
            pending_reserved_tokens: None,
            pending_reserved_cost_usd: None,
            accounting_state: Default::default(),
            tool_phase: Default::default(),
        },
    )
}

fn create_v7_goal_database(path: &std::path::Path, include_session_id: bool) {
    let conn = Connection::open(path.join("control_plane.db")).unwrap();
    let session_column = if include_session_id {
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
}

#[tokio::test]
async fn pause_and_resume_fence_the_previous_executor_epoch() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("fenced", "session-fenced");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();

    assert_eq!(
        store
            .pause_session_goal(
                "fenced",
                "session-fenced",
                1,
                GoalPauseState {
                    reason: GoalPauseReason::OperatorPaused,
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
            .resume_session_goal("fenced", "session-fenced", 2, 2, "boot-b")
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .finish_session_goal("fenced", "session-fenced", 1, TaskStatus::Completed, None,)
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
    assert_eq!(
        store.get("fenced").await.unwrap().unwrap().execution_epoch,
        3
    );
}

#[tokio::test]
async fn unfinished_tool_phase_prevents_resume_after_a_pause() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("tool-phase", "session-tool-phase");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    assert_eq!(
        store
            .begin_goal_tool_phase("tool-phase", "session-tool-phase", 1)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .pause_session_goal(
                "tool-phase",
                "session-tool-phase",
                1,
                GoalPauseState {
                    reason: GoalPauseReason::OperatorPaused,
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
            .resume_session_goal("tool-phase", "session-tool-phase", 2, 2, "boot-b")
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
}

#[tokio::test]
async fn restart_pauses_a_clean_goal_without_transferring_execution() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("restart-clean", "session-restart-clean");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    drop(store);

    let boot = ControlPlaneHandle::start_with_boot_id(directory.path(), "boot-b".into())
        .await
        .unwrap();
    let task = boot.store.get("restart-clean").await.unwrap().unwrap();
    assert_eq!(task.status, TaskStatus::Paused);
    assert_eq!(task.execution_epoch, 2);
}

#[test]
fn sqlite_rejects_hostile_session_and_pending_operation_mutations() {
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
            "INSERT INTO tasks (id, kind, agent, status, session_id, execution_epoch, started_at)
             VALUES ('goal-a', 'goal', 'main', 'running', 'session-a', 1, 'now')",
            [],
        )
        .unwrap();
    assert!(
        database
            .execute(
                "INSERT INTO goal_tasks (task_id, objective, success_criteria, pending_call_id)
             VALUES ('goal-a', 'objective', 'criteria', 'call-a')",
                [],
            )
            .is_err()
    );
    database
        .execute(
            "INSERT INTO goal_tasks (task_id, objective, success_criteria,
                                     pending_call_id, pending_call_epoch, pending_reserved_tokens)
             VALUES ('goal-a', 'objective', 'criteria', 'call-a', 1, 10)",
            [],
        )
        .unwrap();
    assert!(
        database
            .execute(
                "UPDATE tasks SET session_id = 'session-b' WHERE id = 'goal-a'",
                [],
            )
            .is_err()
    );
    assert!(
        database
            .execute(
                "UPDATE goal_tasks SET objective = 'replacement' WHERE task_id = 'goal-a'",
                [],
            )
            .is_err()
    );
    assert!(
        database
            .execute(
                "UPDATE goal_tasks SET pending_call_id = NULL, pending_call_epoch = NULL
              WHERE task_id = 'goal-a'",
                [],
            )
            .is_err()
    );
}

#[test]
fn independent_connections_admit_exactly_one_goal_for_a_session() {
    let directory = tempfile::tempdir().unwrap();
    SqliteTaskStore::new(directory.path()).unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let workers: Vec<_> = ["left", "right"]
        .into_iter()
        .map(|id| {
            let directory = directory.path().to_owned();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let store = SqliteTaskStore::new(&directory).unwrap();
                let (task, goal) = session_goal(id, "session-race");
                barrier.wait();
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(store.create_or_replace_session_goal(task, goal))
            })
        })
        .collect();
    let outcomes: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().unwrap().unwrap())
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

#[test]
fn independent_connections_admit_exactly_one_pending_operation() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("pending-race", "session-pending-race");
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(store.create_or_replace_session_goal(task, goal))
        .unwrap();
    drop(store);
    let barrier = Arc::new(Barrier::new(2));
    let workers: Vec<_> = ["left", "right"]
        .into_iter()
        .map(|call_id| {
            let directory = directory.path().to_owned();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let store = SqliteTaskStore::new(&directory).unwrap();
                barrier.wait();
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(store.admit_pending_call_with_reservation(
                        "pending-race",
                        "session-pending-race",
                        1,
                        call_id,
                        Some(10),
                        None,
                    ))
            })
        })
        .collect();
    let outcomes: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().unwrap().unwrap())
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
async fn terminal_replacement_waits_for_matching_pending_settlement() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("terminal", "session-terminal");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    store
        .admit_pending_call_with_reservation(
            "terminal",
            "session-terminal",
            1,
            "call-terminal",
            Some(10),
            None,
        )
        .await
        .unwrap();
    store
        .finish_session_goal(
            "terminal",
            "session-terminal",
            1,
            TaskStatus::Cancelled,
            Some("operator_cancelled".into()),
        )
        .await
        .unwrap();
    let (replacement, replacement_goal) = session_goal("replacement", "session-terminal");
    assert_eq!(
        store
            .create_or_replace_session_goal(replacement.clone(), replacement_goal.clone())
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
    assert_eq!(
        store
            .settle_pending_call(
                "terminal",
                "session-terminal",
                1,
                "call-terminal",
                GoalAccountingState::OutcomeUnknown,
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .create_or_replace_session_goal(replacement, replacement_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert!(store.get("terminal").await.unwrap().is_none());
}

#[tokio::test]
async fn restart_fails_a_goal_with_an_unsettled_operation() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("unsettled", "session-unsettled");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    store
        .admit_pending_call_with_reservation(
            "unsettled",
            "session-unsettled",
            1,
            "call-unsettled",
            Some(10),
            None,
        )
        .await
        .unwrap();
    drop(store);
    let boot = ControlPlaneHandle::start_with_boot_id(directory.path(), "boot-b".into())
        .await
        .unwrap();
    assert_eq!(
        boot.store.get("unsettled").await.unwrap().unwrap().status,
        TaskStatus::Failed
    );
    drop(boot);
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let goal = store.get_goal_task("unsettled").await.unwrap().unwrap();
    assert_eq!(goal.accounting_state, GoalAccountingState::OutcomeUnknown);
    assert_eq!(goal.pending_call_id.as_deref(), Some("call-unsettled"));
}

#[test]
fn failed_v8_migration_rolls_back_the_entire_schema_change() {
    let directory = tempfile::tempdir().unwrap();
    create_v7_goal_database(directory.path(), true);
    let db_path = directory.path().join("control_plane.db");
    let conn = Connection::open(&db_path).unwrap();
    for id in ["first", "second"] {
        conn.execute(
            "INSERT INTO tasks (id, kind, agent, status, session_id, started_at)
             VALUES (?1, 'goal', 'main', 'completed', 'same-session', '2026-08-24T00:00:00Z')",
            [id],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO goal_tasks (task_id, objective) VALUES (?1, 'done')",
            [id],
        )
        .unwrap();
    }
    drop(conn);

    assert!(SqliteTaskStore::new(directory.path()).is_err());

    let conn = Connection::open(db_path).unwrap();
    let version: i64 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(version, 7);
    let has_epoch: bool = conn
        .prepare("PRAGMA table_info(tasks)")
        .unwrap()
        .query_map([], |row| row.get::<_, String>(1))
        .unwrap()
        .any(|name| matches!(name, Ok(name) if name == "execution_epoch"));
    assert!(!has_epoch);
}

#[tokio::test]
async fn migration_fences_legacy_running_goals_without_a_session() {
    let directory = tempfile::tempdir().unwrap();
    create_v7_goal_database(directory.path(), false);
    let db_path = directory.path().join("control_plane.db");
    let conn = Connection::open(&db_path).unwrap();
    conn.execute(
        "INSERT INTO tasks (id, kind, agent, status, started_at)
         VALUES ('legacy', 'goal', 'main', 'running', '2026-08-24T00:00:00Z')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO goal_tasks (task_id, objective) VALUES ('legacy', 'old goal')",
        [],
    )
    .unwrap();
    drop(conn);

    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let task = store.get("legacy").await.unwrap().unwrap();
    assert_eq!(task.status, TaskStatus::Failed);
    assert_eq!(task.execution_epoch, 1);
    assert!(
        store
            .current_goal_for_session("legacy-session")
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn terminal_goal_stays_current_until_explicit_disposal() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("terminal-visible", "session-terminal-visible");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    assert_eq!(
        store
            .finish_session_goal(
                "terminal-visible",
                "session-terminal-visible",
                1,
                TaskStatus::Completed,
                None,
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    let current = store
        .current_goal_for_session("session-terminal-visible")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(current.status, TaskStatus::Completed);
    assert_eq!(current.execution_epoch, 2);

    assert_eq!(
        store
            .delete_session_goal("terminal-visible", "session-terminal-visible", 2)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert!(
        store
            .current_goal_for_session("session-terminal-visible")
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn legacy_mutators_cannot_bypass_session_goal_epoch_fencing() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("guarded-only", "session-guarded-only");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();

    assert!(
        store
            .update_status("guarded-only", TaskStatus::Paused, None, None)
            .await
            .is_err()
    );
    assert!(
        store
            .claim_owner("guarded-only", 2, "boot-b")
            .await
            .is_err()
    );
    assert!(store.heartbeat("guarded-only", "boot-a").await.is_err());
    assert!(
        store
            .update_goal_limits("guarded-only", Some(5), None)
            .await
            .is_err()
    );
    assert!(store.update_goal_pause("guarded-only", None).await.is_err());
    assert!(
        store
            .pause_goal_task(
                "guarded-only",
                GoalPauseState {
                    reason: GoalPauseReason::OperatorPaused,
                    description: None,
                    blockers: Vec::new(),
                },
            )
            .await
            .is_err()
    );
    assert!(
        store
            .resume_goal_task("guarded-only", 2, "boot-b", None)
            .await
            .is_err()
    );
    assert!(
        store
            .set_continuation_context("guarded-only", None)
            .await
            .is_err()
    );
    assert!(
        !store
            .reconcile_lost("guarded-only", "boot-b")
            .await
            .unwrap()
    );
    assert_eq!(
        store.get("guarded-only").await.unwrap().unwrap().status,
        TaskStatus::Running
    );
}

#[tokio::test]
async fn terminal_goal_with_unpaired_tool_phase_cannot_be_replaced() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("unpaired-terminal", "session-unpaired-terminal");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    store
        .begin_goal_tool_phase("unpaired-terminal", "session-unpaired-terminal", 1)
        .await
        .unwrap();
    store
        .finish_session_goal(
            "unpaired-terminal",
            "session-unpaired-terminal",
            1,
            TaskStatus::Failed,
            Some("tool pairing incomplete".into()),
        )
        .await
        .unwrap();
    let (replacement, replacement_goal) =
        session_goal("replacement-tool", "session-unpaired-terminal");
    assert_eq!(
        store
            .create_or_replace_session_goal(replacement, replacement_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
}

#[tokio::test]
async fn duplicate_task_id_never_reports_a_phantom_session_goal() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (first, first_goal) = session_goal("shared-id", "session-first");
    store
        .create_or_replace_session_goal(first, first_goal)
        .await
        .unwrap();
    store
        .finish_session_goal("shared-id", "session-first", 1, TaskStatus::Completed, None)
        .await
        .unwrap();
    let (second, second_goal) = session_goal("shared-id", "session-second");
    assert_eq!(
        store
            .create_or_replace_session_goal(second, second_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
    assert!(
        store
            .current_goal_for_session("session-second")
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn restart_pauses_an_unpaired_tool_phase_without_erasing_its_pause_contract() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("restart-tool", "session-restart-tool");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    store
        .begin_goal_tool_phase("restart-tool", "session-restart-tool", 1)
        .await
        .unwrap();
    drop(store);

    let boot = ControlPlaneHandle::start_with_boot_id(directory.path(), "boot-b".into())
        .await
        .unwrap();
    let task = boot.store.get("restart-tool").await.unwrap().unwrap();
    assert_eq!(task.status, TaskStatus::Paused);
    assert_eq!(task.execution_epoch, 2);
    drop(boot);
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    assert_eq!(
        store
            .get_goal_task("restart-tool")
            .await
            .unwrap()
            .unwrap()
            .tool_phase,
        zeroclaw_runtime::control_plane::GoalToolPhase::InFlight
    );
    assert_eq!(
        store
            .resume_session_goal("restart-tool", "session-restart-tool", 2, 2, "boot-c")
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
}

#[tokio::test]
async fn restart_does_not_rewrite_a_preexisting_paused_goal() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("already-paused", "session-already-paused");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    store
        .pause_session_goal(
            "already-paused",
            "session-already-paused",
            1,
            GoalPauseState {
                reason: GoalPauseReason::OperatorPaused,
                description: Some("keep this reason".into()),
                blockers: Vec::new(),
            },
        )
        .await
        .unwrap();
    drop(store);

    let boot = ControlPlaneHandle::start_with_boot_id(directory.path(), "boot-b".into())
        .await
        .unwrap();
    drop(boot);
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let goal = store
        .get_goal_task("already-paused")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(goal.pause_reason, Some(GoalPauseReason::OperatorPaused));
    assert_eq!(goal.pause_description.as_deref(), Some("keep this reason"));
}

#[tokio::test]
async fn restart_fails_a_saturated_epoch_instead_of_losing_the_goal() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("saturated", "session-saturated");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    drop(store);
    let conn = Connection::open(directory.path().join("control_plane.db")).unwrap();
    conn.execute(
        "UPDATE tasks SET execution_epoch = ?1 WHERE id = 'saturated'",
        [i64::MAX],
    )
    .unwrap();
    drop(conn);

    let boot = ControlPlaneHandle::start_with_boot_id(directory.path(), "boot-b".into())
        .await
        .unwrap();
    let task = boot.store.get("saturated").await.unwrap().unwrap();
    assert_eq!(task.status, TaskStatus::Failed);
    assert_eq!(task.execution_epoch, i64::MAX);
    drop(boot);
}

#[tokio::test]
async fn migration_normalizes_blank_terminal_legacy_session_bindings() {
    let directory = tempfile::tempdir().unwrap();
    create_v7_goal_database(directory.path(), true);
    let db_path = directory.path().join("control_plane.db");
    let conn = Connection::open(&db_path).unwrap();
    conn.execute(
        "INSERT INTO tasks (id, kind, agent, status, session_id, started_at, finished_at)
         VALUES ('blank-terminal', 'goal', 'main', 'completed', '   ',
                 '2026-08-24T00:00:00Z', '2026-08-24T00:00:01Z')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO goal_tasks (task_id, objective) VALUES ('blank-terminal', 'old goal')",
        [],
    )
    .unwrap();
    drop(conn);

    let store = SqliteTaskStore::new(directory.path()).unwrap();
    assert!(store.current_goal_for_session("").await.unwrap().is_none());
    assert_eq!(
        store
            .get("blank-terminal")
            .await
            .unwrap()
            .unwrap()
            .session_id,
        None
    );
}
