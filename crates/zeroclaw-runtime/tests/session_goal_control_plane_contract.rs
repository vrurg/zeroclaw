//! File-backed contract tests for session-bound durable Goal control state.

use std::sync::{Arc, Barrier};

use zeroclaw_runtime::control_plane::{
    ControlPlaneHandle, GoalAccountingState, GoalResumeSafety, GoalSessionFence, GoalTaskRecord,
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
            started_at: "2026-08-20T00:00:00Z".into(),
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
            accounting_state: GoalAccountingState::Complete,
            resume_safety: GoalResumeSafety::Clean,
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
            .admit_pending_call_with_reservation("first", "session-a", 1, "call-a", Some(10), None,)
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
        GoalTransitionResult::Applied,
        "a pause fence prevents response effects but must not strand the admitted accounting settlement"
    );
}

#[tokio::test]
async fn running_goal_enumeration_excludes_paused_terminal_and_unrelated_tasks() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (running, running_goal) = session_goal("running-goal", "session-running");
    store
        .create_or_replace_session_goal(running, running_goal)
        .await
        .unwrap();
    let (paused, paused_goal) = session_goal("paused-goal", "session-paused");
    store
        .create_or_replace_session_goal(paused, paused_goal)
        .await
        .unwrap();
    store
        .pause_session_goal(
            "paused-goal",
            "session-paused",
            1,
            zeroclaw_runtime::control_plane::GoalPauseState {
                reason: zeroclaw_runtime::control_plane::GoalPauseReason::OperatorPaused,
                description: None,
                blockers: Vec::new(),
            },
        )
        .await
        .unwrap();
    let unrelated = TaskRecord {
        id: "ordinary-running-task".into(),
        kind: TaskKind::Delegate,
        agent: "main".into(),
        status: TaskStatus::Running,
        owner_pid: 1,
        owner_boot_id: "boot-a".into(),
        heartbeat_at: None,
        depth: 0,
        parent_id: None,
        originator_route: None,
        delivered: false,
        idem_key: None,
        principal_id: None,
        session_id: None,
        execution_epoch: 0,
        started_at: "2026-08-20T00:00:00Z".into(),
        finished_at: None,
    };
    store.create(unrelated).await.unwrap();

    let running = store.list_running_goal_statuses().await.unwrap();
    assert_eq!(running.len(), 1);
    assert_eq!(running[0].task.id, "running-goal");
    assert_eq!(running[0].task.execution_epoch, 1);
}

#[tokio::test]
async fn policy_revocation_cancels_all_exact_goals_or_rolls_back() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (first, first_goal) = session_goal("policy-first", "session-policy-first");
    let (second, second_goal) = session_goal("policy-second", "session-policy-second");
    store
        .create_or_replace_session_goal(first, first_goal)
        .await
        .unwrap();
    store
        .create_or_replace_session_goal(second, second_goal)
        .await
        .unwrap();

    let first_fence = GoalSessionFence::new("policy-first", "session-policy-first", 1);
    let missing_fence = GoalSessionFence::new("missing-goal", "session-policy-missing", 1);
    assert_eq!(
        store
            .cancel_session_goals(&[first_fence.clone(), missing_fence], "goal_policy_revoked",)
            .await
            .unwrap(),
        GoalTransitionResult::Missing,
        "a failed batch must not leave an earlier exact Goal cancelled"
    );
    let first_after_rollback = store
        .current_goal_for_session("session-policy-first")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(first_after_rollback.status, TaskStatus::Running);
    assert_eq!(first_after_rollback.execution_epoch, 1);

    let second_fence = GoalSessionFence::new("policy-second", "session-policy-second", 1);
    assert_eq!(
        store
            .cancel_session_goals(&[first_fence, second_fence], "goal_policy_revoked",)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    for session_id in ["session-policy-first", "session-policy-second"] {
        let goal = store
            .current_goal_for_session(session_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(goal.status, TaskStatus::Cancelled);
        assert_eq!(goal.execution_epoch, 2);
    }
}

#[tokio::test]
async fn resume_refuses_execution_epoch_overflow() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("epoch-max", "session-epoch-max");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    drop(store);

    let database = rusqlite::Connection::open(directory.path().join("control_plane.db")).unwrap();
    database
        .execute(
            "UPDATE tasks
                SET status = 'paused', execution_epoch = 9223372036854775807
              WHERE id = 'epoch-max'",
            [],
        )
        .unwrap();
    drop(database);

    let store = SqliteTaskStore::new(directory.path()).unwrap();
    assert_eq!(
        store
            .resume_session_goal("epoch-max", "session-epoch-max", i64::MAX, 2, "boot-b",)
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
    assert_eq!(
        store
            .get("epoch-max")
            .await
            .unwrap()
            .unwrap()
            .execution_epoch,
        i64::MAX
    );
}

#[tokio::test]
async fn session_goal_persists_criteria_reservation_and_monotonic_tool_safety() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, mut goal) = session_goal("criteria", "session-criteria");
    goal.success_criteria = "The release notes are complete".into();
    store
        .create_or_replace_session_goal(task.clone(), goal)
        .await
        .unwrap();

    assert_eq!(
        store
            .admit_pending_call_with_reservation(
                "criteria",
                "session-criteria",
                1,
                "reservation-a",
                Some(100),
                Some(1.25),
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    let admitted = store.get_goal_task("criteria").await.unwrap().unwrap();
    assert_eq!(admitted.success_criteria, "The release notes are complete");
    assert_eq!(admitted.pending_reserved_tokens, Some(100));
    assert_eq!(admitted.pending_reserved_cost_usd, Some(1.25));
    assert_eq!(admitted.resume_safety, GoalResumeSafety::Clean);

    assert_eq!(
        store
            .mark_goal_tool_executed("criteria", "session-criteria", 1)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .settle_pending_call(
                "criteria",
                "session-criteria",
                1,
                "reservation-a",
                GoalAccountingState::Complete,
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .pause_session_goal(
                "criteria",
                "session-criteria",
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
            .resume_session_goal("criteria", "session-criteria", 2, 2, "boot-b")
            .await
            .unwrap(),
        GoalTransitionResult::Stale,
        "a tool-executed Goal may not be resumed after its executor ends"
    );

    let persisted = store.get_goal_task("criteria").await.unwrap().unwrap();
    assert_eq!(persisted.resume_safety, GoalResumeSafety::ToolExecuted);
    assert!(persisted.pending_call_id.is_none());
    assert!(persisted.pending_reserved_tokens.is_none());
    assert!(persisted.pending_reserved_cost_usd.is_none());
}

#[tokio::test]
async fn terminal_replacement_never_deletes_an_unsettled_operation_owner() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("pending-terminal", "session-pending-terminal");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    assert_eq!(
        store
            .admit_pending_call_with_reservation(
                "pending-terminal",
                "session-pending-terminal",
                1,
                "call-pending-terminal",
                Some(10),
                None,
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    drop(store);

    let database = rusqlite::Connection::open(directory.path().join("control_plane.db")).unwrap();
    database
        .execute(
            "UPDATE tasks SET status = 'failed', execution_epoch = 2 WHERE id = 'pending-terminal'",
            [],
        )
        .unwrap();
    drop(database);

    let store = SqliteTaskStore::new(directory.path()).unwrap();
    assert_eq!(
        store
            .delete_session_goal("pending-terminal", "session-pending-terminal", 2)
            .await
            .unwrap(),
        GoalTransitionResult::Stale,
        "hard deletion must retain unsettled accounting until it is classified"
    );
    let (replacement, replacement_goal) =
        session_goal("pending-terminal-replacement", "session-pending-terminal");
    assert_eq!(
        store
            .create_or_replace_session_goal(replacement, replacement_goal)
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
    assert!(store.get("pending-terminal").await.unwrap().is_some());
    assert_eq!(
        store
            .get_goal_task("pending-terminal")
            .await
            .unwrap()
            .unwrap()
            .pending_call_id
            .as_deref(),
        Some("call-pending-terminal")
    );
}

#[tokio::test]
async fn cancellation_fence_allows_only_matching_pending_settlement() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("cancelled-settlement", "session-cancelled-settlement");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    store
        .admit_pending_call_with_reservation(
            "cancelled-settlement",
            "session-cancelled-settlement",
            1,
            "call-cancelled",
            Some(10),
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        store
            .finish_session_goal(
                "cancelled-settlement",
                "session-cancelled-settlement",
                1,
                TaskStatus::Cancelled,
                Some("operator_cancelled".into()),
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .settle_pending_call(
                "cancelled-settlement",
                "session-cancelled-settlement",
                1,
                "different-call",
                GoalAccountingState::Complete,
            )
            .await
            .unwrap(),
        GoalTransitionResult::Stale
    );
    assert_eq!(
        store
            .settle_pending_call(
                "cancelled-settlement",
                "session-cancelled-settlement",
                1,
                "call-cancelled",
                GoalAccountingState::OutcomeUnknown,
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    let unsettled = store
        .get_goal_task("cancelled-settlement")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        unsettled.accounting_state,
        GoalAccountingState::OutcomeUnknown
    );
    assert_eq!(unsettled.pending_call_id.as_deref(), Some("call-cancelled"));
    assert_eq!(unsettled.pending_call_epoch, Some(1));
    assert_eq!(unsettled.pending_reserved_tokens, Some(10));
}

#[tokio::test]
async fn hard_disposal_requires_pending_accounting_to_be_classified_first() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("dispose-outcome-unknown", "session-dispose-unknown");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    store
        .admit_pending_call_with_reservation(
            "dispose-outcome-unknown",
            "session-dispose-unknown",
            1,
            "dispose-pending-call",
            Some(10),
            None,
        )
        .await
        .unwrap();
    store
        .finish_session_goal(
            "dispose-outcome-unknown",
            "session-dispose-unknown",
            1,
            TaskStatus::Cancelled,
            Some("session_disposed".into()),
        )
        .await
        .unwrap();

    assert_eq!(
        store
            .dispose_session_goal("dispose-outcome-unknown", "session-dispose-unknown", 2)
            .await
            .unwrap(),
        GoalTransitionResult::Stale,
        "a pending operation cannot disappear before it is classified"
    );

    assert_eq!(
        store
            .settle_pending_call(
                "dispose-outcome-unknown",
                "session-dispose-unknown",
                1,
                "dispose-pending-call",
                GoalAccountingState::OutcomeUnknown,
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .dispose_session_goal("dispose-outcome-unknown", "session-dispose-unknown", 2)
            .await
            .unwrap(),
        GoalTransitionResult::Applied,
        "the classified terminal control record may be removed with its session"
    );
    assert!(
        store
            .current_goal_for_session("session-dispose-unknown")
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn completed_goal_delivery_mark_is_exact_and_idempotent() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("completed-delivery", "session-completed-delivery");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    assert_eq!(
        store
            .finish_session_goal(
                "completed-delivery",
                "session-completed-delivery",
                1,
                TaskStatus::Completed,
                None,
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    for _ in 0..2 {
        assert_eq!(
            store
                .mark_completed_session_goal_delivered(
                    "completed-delivery",
                    "session-completed-delivery",
                    2,
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
    }
    assert!(
        store
            .get("completed-delivery")
            .await
            .unwrap()
            .unwrap()
            .delivered
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
                "INSERT INTO goal_tasks (task_id, objective, success_criteria, pending_call_id)
             VALUES ('goal-a', 'objective', 'criteria', 'call-a')",
                [],
            )
            .is_err()
    );
    database
        .execute(
            "INSERT INTO goal_tasks
                (task_id, objective, success_criteria, pending_call_id, pending_call_epoch)
             VALUES ('goal-a', 'objective', 'criteria', 'call-a', 1)",
            [],
        )
        .unwrap();
    database
        .execute(
            "INSERT INTO tasks
                (id, kind, agent, status, session_id, execution_epoch, started_at)
             VALUES ('goal-finite', 'goal', 'main', 'running', 'session-finite', 1, 'now')",
            [],
        )
        .unwrap();
    assert!(
        database
            .execute(
                "INSERT INTO goal_tasks
                    (task_id, objective, success_criteria, effective_token_limit,
                     pending_call_id, pending_call_epoch)
                 VALUES ('goal-finite', 'objective', 'criteria', 10, 'call-finite', 1)",
                [],
            )
            .is_err(),
        "a finite Goal pending operation must carry its reservation"
    );
    database
        .execute(
            "INSERT INTO tasks
                (id, kind, agent, status, session_id, execution_epoch, started_at)
             VALUES ('goal-whitespace', 'goal', 'main', 'running', 'session-whitespace', 1, 'now')",
            [],
        )
        .unwrap();
    assert!(
        database
            .execute(
                "INSERT INTO goal_tasks (task_id, objective, success_criteria)
                 VALUES ('goal-whitespace', 'objective', char(9) || char(10) || ' ')",
                [],
            )
            .is_err(),
        "success criteria must contain a non-whitespace character"
    );
    assert!(
        database
            .execute(
                "INSERT INTO goal_tasks (task_id, objective, success_criteria)
                 VALUES ('goal-whitespace', 'objective', char(160))",
                [],
            )
            .is_err(),
        "success criteria reject Unicode nonbreaking whitespace"
    );
    assert!(
        database
            .execute(
                "INSERT INTO goal_tasks (task_id, objective, success_criteria)
                 VALUES ('goal-whitespace', char(8195), char(8195))",
                [],
            )
            .is_err(),
        "objective and criteria reject Unicode em-space whitespace"
    );
    assert!(
        database
            .execute(
                "INSERT INTO goal_tasks (task_id, objective, success_criteria)
                 VALUES ('goal-whitespace', char(133), char(133))",
                [],
            )
            .is_err(),
        "objective and criteria reject Unicode next-line whitespace"
    );
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
                "UPDATE goal_tasks SET success_criteria = 'replacement' WHERE task_id = 'goal-a'",
                [],
            )
            .is_err(),
        "success criteria are immutable after Goal creation"
    );
    assert!(
        database
            .execute(
                "UPDATE goal_tasks SET objective = 'replacement' WHERE task_id = 'goal-a'",
                [],
            )
            .is_err(),
        "session Goal objectives are immutable after creation"
    );
    database
        .execute(
            "UPDATE goal_tasks SET pending_call_id = NULL, pending_call_epoch = NULL WHERE task_id = 'goal-a'",
            [],
        )
        .unwrap();
    assert!(
        database
            .execute(
                "UPDATE goal_tasks SET pending_reserved_tokens = 1 WHERE task_id = 'goal-a'",
                [],
            )
            .is_err(),
        "a reservation cannot exist without its pending operation"
    );
    database
        .execute(
            "UPDATE goal_tasks SET resume_safety = 'tool_executed' WHERE task_id = 'goal-a'",
            [],
        )
        .unwrap();
    assert!(
        database
            .execute(
                "UPDATE goal_tasks SET resume_safety = 'clean' WHERE task_id = 'goal-a'",
                [],
            )
            .is_err(),
        "the tool-executed marker cannot be reset through raw SQL"
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
    assert_eq!(version, 12);
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
    assert_eq!(version, 12);
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
async fn independent_connections_reserve_exactly_one_pending_operation() {
    let directory = tempfile::tempdir().unwrap();
    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let (task, goal) = session_goal("reserve-race", "session-reserve-race");
    store
        .create_or_replace_session_goal(task, goal)
        .await
        .unwrap();
    drop(store);

    let barrier = Arc::new(Barrier::new(2));
    let mut workers = Vec::new();
    for call_id in ["reserve-left", "reserve-right"] {
        let data_dir = directory.path().to_owned();
        let barrier = Arc::clone(&barrier);
        workers.push(std::thread::spawn(move || {
            let store = SqliteTaskStore::new(&data_dir).unwrap();
            barrier.wait();
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(store.admit_pending_call_with_reservation(
                    "reserve-race",
                    "session-reserve-race",
                    1,
                    call_id,
                    Some(10),
                    None,
                ))
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

    let store = SqliteTaskStore::new(directory.path()).unwrap();
    let goal = store.get_goal_task("reserve-race").await.unwrap().unwrap();
    assert!(matches!(
        goal.pending_call_id.as_deref(),
        Some("reserve-left" | "reserve-right")
    ));
    assert_eq!(goal.pending_reserved_tokens, Some(10));
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
        .admit_pending_call_with_reservation(
            "preerrored",
            "session-preerrored",
            1,
            "pending-1",
            Some(10),
            None,
        )
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
    let status = store
        .current_goal_status_for_session("session-a")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(status.task.id, "first");
    assert_eq!(status.goal.task_id, "first");
    assert_eq!(
        status.terminal_error.as_deref(),
        Some("test terminal transition"),
        "status joins the canonical task error rather than copying it into Goal state"
    );
    assert_eq!(
        store
            .mark_completed_session_goal_delivered("first", "session-a", 2)
            .await
            .unwrap(),
        GoalTransitionResult::Stale,
        "only a completed Goal can receive a completion delivery mark"
    );

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
async fn boot_fails_session_bound_legacy_goal_without_success_criteria() {
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
                 execution_epoch INTEGER NOT NULL DEFAULT 0, started_at TEXT NOT NULL,
                 finished_at TEXT, output TEXT, error TEXT
             );
             CREATE TABLE goal_tasks (
                 task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,
                 objective TEXT NOT NULL, effective_token_limit INTEGER,
                 effective_cost_limit_usd REAL, pause_reason TEXT,
                 pause_description TEXT, blockers_json TEXT NOT NULL DEFAULT '[]',
                 pending_call_id TEXT, pending_call_epoch INTEGER,
                 accounting_state TEXT NOT NULL DEFAULT 'complete'
             );
             INSERT INTO tasks
                 (id, kind, agent, status, session_id, execution_epoch, owner_boot_id, started_at)
             VALUES ('legacy-criteria', 'goal', 'main', 'running', 'session-legacy', 1, 'old-boot', 'now');
             INSERT INTO goal_tasks (task_id, objective)
             VALUES ('legacy-criteria', 'old objective');
             PRAGMA user_version = 11;",
        )
        .unwrap();
    drop(database);

    let handle = ControlPlaneHandle::start_with_boot_id(directory.path(), "new-boot".into())
        .await
        .unwrap();
    let legacy = handle.store.get("legacy-criteria").await.unwrap().unwrap();
    assert_eq!(legacy.status, TaskStatus::Failed);
    drop(handle);

    let database = rusqlite::Connection::open(path).unwrap();
    let error: String = database
        .query_row(
            "SELECT error FROM tasks WHERE id = 'legacy-criteria'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(error, "legacy_goal_missing_success_criteria");
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
