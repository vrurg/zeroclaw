//! File-backed contract coverage for session-bound Goal control-plane state.

use std::sync::{Arc, Barrier};

use rusqlite::{Connection, ErrorCode, params};
use zeroclaw_runtime::control_plane::{
    GoalAccountingState, GoalPauseReason, GoalPauseState, GoalTaskRecord, GoalTaskRegistry,
    GoalToolPhase, GoalTransitionResult, SqliteTaskStore, TaskKind, TaskRecord, TaskRegistry,
    TaskStatus,
};

fn insert_current_goal(
    connection: &Connection,
    task_id: &str,
    session_id: &str,
) -> rusqlite::Result<usize> {
    connection.execute(
        "INSERT INTO tasks (
                 id, kind, agent, status, owner_pid, owner_boot_id, session_id,
                 execution_epoch, started_at
             ) VALUES (?1, 'goal', 'main', 'running', 1, 'boot-a', ?2, 1, 'now')",
        params![task_id, session_id],
    )
}

fn session_goal_task(task_id: &str, session_id: &str) -> TaskRecord {
    TaskRecord {
        id: task_id.into(),
        kind: TaskKind::Goal,
        agent: "main".into(),
        status: TaskStatus::Running,
        owner_pid: 1,
        owner_boot_id: "boot-a".into(),
        heartbeat_at: None,
        depth: 0,
        parent_id: None,
        originator_route: Some("matrix:!room:example.test".into()),
        delivered: false,
        idem_key: None,
        principal_id: Some("matrix:@operator:example.test".into()),
        session_id: Some(session_id.into()),
        execution_epoch: 0,
        started_at: "2026-08-30T00:00:00Z".into(),
        finished_at: None,
    }
}

fn session_goal_extension(task_id: &str) -> GoalTaskRecord {
    GoalTaskRecord {
        task_id: task_id.into(),
        objective: "produce a verified result".into(),
        success_criteria: "the verifier accepts the exact candidate".into(),
        ..GoalTaskRecord::default()
    }
}

#[test]
fn independent_connections_admit_exactly_one_current_goal_for_a_session() {
    let directory = tempfile::tempdir().expect("create temporary control-plane directory");
    SqliteTaskStore::new(directory.path()).expect("initialize control-plane schema");

    let barrier = Arc::new(Barrier::new(2));
    let outcomes: Vec<_> = ["left", "right"]
        .into_iter()
        .map(|task_id| {
            let database = directory.path().join("control_plane.db");
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let connection = Connection::open(database).expect("open independent connection");
                connection
                    .busy_timeout(std::time::Duration::from_secs(5))
                    .expect("configure independent connection busy timeout");
                barrier.wait();
                insert_current_goal(&connection, task_id, "shared-session")
            })
        })
        .collect();

    let successes = outcomes
        .into_iter()
        .map(|thread| thread.join().expect("join concurrent writer"))
        .filter(|outcome| outcome.as_ref().is_ok_and(|rows| *rows == 1))
        .count();

    assert_eq!(successes, 1, "a session has one current Goal row");

    let connection = Connection::open(directory.path().join("control_plane.db"))
        .expect("open verification connection");
    let current_goals: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM tasks
             WHERE kind = 'goal' AND session_id = 'shared-session'",
            [],
            |row| row.get(0),
        )
        .expect("count current session Goal rows");
    assert_eq!(current_goals, 1);

    let duplicate_error = connection
        .execute(
            "INSERT INTO tasks (
                 id, kind, agent, status, owner_pid, owner_boot_id, session_id,
                 execution_epoch, started_at
             ) VALUES ('third', 'goal', 'main', 'running', 1, 'boot-a', 'shared-session', 1, 'now')",
            [],
        )
        .expect_err("a second current Goal for one session must be rejected");
    assert_eq!(
        duplicate_error.sqlite_error_code(),
        Some(ErrorCode::ConstraintViolation)
    );
}

#[test]
fn independent_goal_registries_admit_exactly_one_current_goal_for_a_session() {
    let directory = tempfile::tempdir().expect("create temporary control-plane directory");
    let left = SqliteTaskStore::new(directory.path()).expect("open left registry");
    let right = SqliteTaskStore::new(directory.path()).expect("open right registry");
    let barrier = Arc::new(Barrier::new(2));

    let outcomes: Vec<_> = [("left", left), ("right", right)]
        .into_iter()
        .map(|(task_id, store)| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("build thread-local runtime")
                    .block_on(async move {
                        store
                            .create_or_replace_session_goal(
                                session_goal_task(task_id, "shared-session"),
                                session_goal_extension(task_id),
                            )
                            .await
                    })
            })
        })
        .collect();

    let outcomes: Vec<_> = outcomes
        .into_iter()
        .map(|thread| thread.join().expect("join concurrent registry"))
        .collect();
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, Ok(GoalTransitionResult::Applied)))
            .count(),
        1,
        "exactly one guarded admission succeeds"
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, Ok(GoalTransitionResult::Stale)))
            .count(),
        1,
        "the concurrent loser observes the current nonterminal Goal"
    );
}

#[tokio::test]
async fn guarded_transitions_fence_stale_epochs_and_terminal_replacement() {
    let store = SqliteTaskStore::new_in_memory().expect("initialize store");
    let task = session_goal_task("goal-one", "session-one");
    let extension = session_goal_extension("goal-one");

    assert_eq!(
        store
            .create_or_replace_session_goal(task, extension)
            .await
            .expect("create session Goal"),
        GoalTransitionResult::Applied
    );
    let current = store
        .current_goal_for_session("session-one")
        .await
        .expect("read current Goal")
        .expect("Goal exists");
    assert_eq!(current.id, "goal-one");
    assert_eq!(current.execution_epoch, 1);

    assert_eq!(
        store
            .pause_session_goal(
                "goal-one",
                "session-one",
                1,
                GoalPauseState {
                    reason: GoalPauseReason::OperatorPaused,
                    description: None,
                    blockers: Vec::new(),
                },
            )
            .await
            .expect("pause exact epoch"),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .finish_session_goal("goal-one", "session-one", 1, TaskStatus::Completed, None)
            .await
            .expect("classify stale finish"),
        GoalTransitionResult::Stale
    );
    assert_eq!(
        store
            .resume_session_goal("goal-one", "session-one", 2, 2, "boot-b")
            .await
            .expect("resume exact epoch"),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .begin_goal_tool_phase("goal-one", "session-one", 2)
            .await
            .expect("fence stale tool start"),
        GoalTransitionResult::Stale
    );
    assert_eq!(
        store
            .begin_goal_tool_phase("goal-one", "session-one", 3)
            .await
            .expect("begin exact tool phase"),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .complete_goal_tool_phase("goal-one", "session-one", 2)
            .await
            .expect("fence stale tool completion"),
        GoalTransitionResult::Stale
    );
    assert_eq!(
        store
            .complete_goal_tool_phase("goal-one", "session-one", 3)
            .await
            .expect("complete exact tool phase"),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .update_session_goal_limits("goal-one", "session-one", 2, Some(11), None)
            .await
            .expect("fence stale limit update"),
        GoalTransitionResult::Stale
    );
    assert_eq!(
        store
            .update_session_goal_limits("goal-one", "session-one", 3, Some(11), Some(1.25))
            .await
            .expect("update limits at exact epoch"),
        GoalTransitionResult::Applied
    );
    let limits = store
        .get_goal_task("goal-one")
        .await
        .expect("read updated Goal limits")
        .expect("Goal extension exists");
    assert_eq!(limits.effective_token_limit, Some(11));
    assert_eq!(limits.effective_cost_limit_usd, Some(1.25));
    assert_eq!(
        store
            .admit_pending_operation("goal-one", "session-one", 3, "operation-one")
            .await
            .expect("admit operation"),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .pause_session_goal(
                "goal-one",
                "session-one",
                3,
                GoalPauseState {
                    reason: GoalPauseReason::OperatorPaused,
                    description: None,
                    blockers: Vec::new(),
                },
            )
            .await
            .expect("fence admitted operation"),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .settle_pending_operation(
                "goal-one",
                "session-one",
                3,
                "operation-one",
                GoalAccountingState::Complete,
            )
            .await
            .expect("settle matching admitted operation"),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .finish_session_goal(
                "goal-one",
                "session-one",
                4,
                TaskStatus::Completed,
                Some("verified completion".into()),
            )
            .await
            .expect("finish fenced Goal"),
        GoalTransitionResult::Applied
    );
    let terminal = store
        .current_goal_for_session("session-one")
        .await
        .expect("read terminal Goal")
        .expect("terminal Goal stays current until replacement");
    assert_eq!(terminal.status, TaskStatus::Completed);
    assert_eq!(terminal.execution_epoch, 5);
    assert_eq!(
        store
            .terminal_reason_for_session_goal("goal-one", "session-one")
            .await
            .expect("read terminal reason")
            .as_deref(),
        Some("verified completion")
    );

    assert_eq!(
        store
            .create_or_replace_session_goal(
                session_goal_task("goal-two", "session-one"),
                session_goal_extension("goal-two"),
            )
            .await
            .expect("replace settled terminal Goal"),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .finish_session_goal("goal-one", "session-one", 5, TaskStatus::Completed, None)
            .await
            .expect("classify removed predecessor"),
        GoalTransitionResult::Missing
    );
}

#[tokio::test]
async fn pausing_an_in_flight_tool_phase_fences_it_and_leaves_the_goal_resumable() {
    let store = SqliteTaskStore::new_in_memory().expect("initialize store");
    assert_eq!(
        store
            .create_or_replace_session_goal(
                session_goal_task("tool-pause", "tool-pause-session"),
                session_goal_extension("tool-pause"),
            )
            .await
            .expect("create session Goal"),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .begin_goal_tool_phase("tool-pause", "tool-pause-session", 1)
            .await
            .expect("begin tool phase"),
        GoalTransitionResult::Applied
    );

    assert_eq!(
        store
            .pause_session_goal(
                "tool-pause",
                "tool-pause-session",
                1,
                GoalPauseState {
                    reason: GoalPauseReason::OperatorPaused,
                    description: None,
                    blockers: Vec::new(),
                },
            )
            .await
            .expect("pause Goal while a tool phase is in flight"),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .complete_goal_tool_phase("tool-pause", "tool-pause-session", 1)
            .await
            .expect("stale tool completion is fenced"),
        GoalTransitionResult::Stale
    );
    assert_eq!(
        store
            .resume_session_goal("tool-pause", "tool-pause-session", 2, 2, "boot-new")
            .await
            .expect("resume paused Goal"),
        GoalTransitionResult::Applied
    );
}

#[tokio::test]
async fn different_sessions_do_not_share_goal_admission_or_disposal_state() {
    let store = SqliteTaskStore::new_in_memory().expect("initialize store");
    for (task_id, session_id) in [
        ("goal-left", "session-left"),
        ("goal-right", "session-right"),
    ] {
        assert_eq!(
            store
                .create_or_replace_session_goal(
                    session_goal_task(task_id, session_id),
                    session_goal_extension(task_id),
                )
                .await
                .expect("create independent Goal"),
            GoalTransitionResult::Applied
        );
    }
    assert_eq!(
        store
            .finish_session_goal("goal-left", "session-left", 1, TaskStatus::Cancelled, None)
            .await
            .expect("cancel left Goal"),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .delete_session_goal("goal-left", "session-left", 2)
            .await
            .expect("dispose left control state"),
        GoalTransitionResult::Applied
    );
    assert!(
        store
            .current_goal_for_session("session-left")
            .await
            .expect("read deleted session")
            .is_none()
    );
    assert_eq!(
        store
            .current_goal_for_session("session-right")
            .await
            .expect("read unaffected session")
            .expect("right Goal survives left disposal")
            .id,
        "goal-right"
    );
}

#[tokio::test]
async fn generic_task_registration_cannot_create_a_goal() {
    let store = SqliteTaskStore::new_in_memory().expect("initialize store");
    let error = store
        .create(session_goal_task("goal-direct", "session-direct"))
        .await
        .expect_err("Goal creation must use the guarded session API");
    assert!(format!("{error:#}").contains("Goal tasks require"));
}

#[tokio::test]
async fn generic_lifecycle_mutators_cannot_bypass_session_bound_goal_fencing() {
    let store = SqliteTaskStore::new_in_memory().expect("initialize store");
    assert_eq!(
        store
            .create_or_replace_session_goal(
                session_goal_task("goal-generic-fence", "session-generic-fence"),
                session_goal_extension("goal-generic-fence"),
            )
            .await
            .expect("create session Goal"),
        GoalTransitionResult::Applied
    );

    for result in [
        TaskRegistry::heartbeat(&store, "goal-generic-fence", "boot-a").await,
        TaskRegistry::claim_owner(&store, "goal-generic-fence", 2, "boot-b").await,
        TaskRegistry::update_status(
            &store,
            "goal-generic-fence",
            TaskStatus::Completed,
            None,
            None,
        )
        .await,
    ] {
        let error = result.expect_err("generic lifecycle mutation must reject a session Goal");
        assert!(format!("{error:#}").contains("session-bound Goal"));
    }
    let error = store
        .delete_by_agent("main")
        .expect_err("generic agent deletion must reject a session Goal");
    assert!(format!("{error:#}").contains("session-bound Goal"));
}

#[tokio::test]
async fn sqlite_guards_session_binding_pending_pairing_and_goal_state_domains() {
    let directory = tempfile::tempdir().expect("create temporary control-plane directory");
    let store = SqliteTaskStore::new(directory.path()).expect("initialize control-plane schema");
    for (task_id, session_id) in [
        ("goal-raw-one", "session-raw-one"),
        ("goal-raw-two", "session-raw-two"),
    ] {
        assert_eq!(
            store
                .create_or_replace_session_goal(
                    session_goal_task(task_id, session_id),
                    session_goal_extension(task_id),
                )
                .await
                .expect("create session Goal"),
            GoalTransitionResult::Applied
        );
    }
    let connection = Connection::open(directory.path().join("control_plane.db"))
        .expect("open independent verification connection");
    let assert_constraint = |result: rusqlite::Result<usize>, scenario: &str| {
        let error = result.expect_err(scenario);
        assert_eq!(
            error.sqlite_error_code(),
            Some(ErrorCode::ConstraintViolation)
        );
    };

    assert_constraint(
        connection.execute(
            "INSERT INTO tasks (
                 id, kind, agent, status, owner_pid, owner_boot_id, session_id,
                 execution_epoch, started_at
             ) VALUES ('goal-null-session', 'goal', 'main', 'running', 1, 'boot-a', NULL, 1, 'now')",
            [],
        ),
        "raw SQL must reject a null Goal session",
    );
    assert_constraint(
        connection.execute(
            "INSERT INTO tasks (
                 id, kind, agent, status, owner_pid, owner_boot_id, session_id,
                 execution_epoch, started_at
             ) VALUES ('goal-blank-session', 'goal', 'main', 'running', 1, 'boot-a', '  ', 1, 'now')",
            [],
        ),
        "raw SQL must reject a blank Goal session",
    );
    assert_constraint(
        connection.execute(
            "UPDATE tasks SET session_id = 'other-session' WHERE id = 'goal-raw-one'",
            [],
        ),
        "raw SQL must not rebind a Goal session",
    );
    assert_constraint(
        connection.execute(
            "UPDATE tasks SET kind = 'delegate' WHERE id = 'goal-raw-one'",
            [],
        ),
        "raw SQL must not demote a Goal before unbinding it",
    );
    connection
        .execute(
            "INSERT INTO tasks (
                 id, kind, agent, status, owner_pid, owner_boot_id, session_id,
                 execution_epoch, started_at
             ) VALUES ('delegate-promote', 'delegate', 'main', 'running', 1, 'boot-a',
                       'promotion-session', 0, 'now')",
            [],
        )
        .expect("insert non-Goal row for promotion guard");
    assert_constraint(
        connection.execute(
            "UPDATE tasks SET kind = 'goal' WHERE id = 'delegate-promote'",
            [],
        ),
        "raw SQL must not promote a non-Goal into a session-bound Goal",
    );
    assert_constraint(
        connection.execute(
            "UPDATE goal_tasks SET pending_call_id = 'call-one' WHERE task_id = 'goal-raw-one'",
            [],
        ),
        "raw SQL must require a pending operation epoch",
    );
    connection
        .execute(
            "UPDATE goal_tasks SET pending_call_id = 'call-one', pending_call_epoch = 1
             WHERE task_id = 'goal-raw-one'",
            [],
        )
        .expect("write one valid pending operation");
    assert_constraint(
        connection.execute(
            "UPDATE goal_tasks SET pending_call_id = 'call-one', pending_call_epoch = 1
             WHERE task_id = 'goal-raw-two'",
            [],
        ),
        "raw SQL must reject a reused pending operation id",
    );
    assert_constraint(
        connection.execute(
            "UPDATE goal_tasks SET accounting_state = 'not-a-state'
             WHERE task_id = 'goal-raw-two'",
            [],
        ),
        "raw SQL must reject an unknown accounting state",
    );
    assert_constraint(
        connection.execute(
            "UPDATE goal_tasks SET tool_phase = 'not-a-phase' WHERE task_id = 'goal-raw-two'",
            [],
        ),
        "raw SQL must reject an unknown tool phase",
    );
}

#[tokio::test]
async fn migration_fails_nonterminal_legacy_goals_but_keeps_terminal_audit_rows() {
    let directory = tempfile::tempdir().expect("create temporary control-plane directory");
    let store = SqliteTaskStore::new(directory.path()).expect("initialize current schema");
    let database = directory.path().join("control_plane.db");
    drop(store);

    let connection = Connection::open(&database).expect("open fixture database");
    connection
        .execute_batch(
            "DROP TRIGGER trg_goal_tasks_require_session_insert;
             DROP TRIGGER trg_goal_tasks_require_session_update;
             INSERT INTO tasks (
                 id, kind, agent, status, owner_pid, owner_boot_id, execution_epoch, started_at
             ) VALUES
                 ('legacy-running', 'goal', 'main', 'running', 1, 'boot-old', 0, 'now'),
                 ('legacy-completed', 'goal', 'main', 'completed', 1, 'boot-old', 0, 'now');
             INSERT INTO goal_tasks (task_id, objective) VALUES
                 ('legacy-running', 'old goal'),
                 ('legacy-completed', 'old terminal goal');
             PRAGMA user_version = 7;",
        )
        .expect("write pre-v8 Goal fixtures");
    drop(connection);

    let migrated = SqliteTaskStore::new(directory.path()).expect("migrate fixture database");
    let verify = Connection::open(&database).expect("open migrated database");
    let schema_version: i64 = verify
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .expect("read migrated schema version");
    assert_eq!(
        schema_version, 8,
        "migration records the v8 immutable Goal guard"
    );
    let running: (String, Option<String>, Option<String>) = verify
        .query_row(
            "SELECT status, error, session_id FROM tasks WHERE id = 'legacy-running'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .expect("read migrated nonterminal Goal");
    assert_eq!(running.0, "failed");
    assert_eq!(running.1.as_deref(), Some("legacy_goal_missing_session"));
    assert!(running.2.is_none());
    let terminal_status: String = verify
        .query_row(
            "SELECT status FROM tasks WHERE id = 'legacy-completed'",
            [],
            |row| row.get(0),
        )
        .expect("read terminal legacy Goal");
    assert_eq!(terminal_status, "completed");
    assert_eq!(
        migrated
            .finish_session_goal(
                "legacy-completed",
                "session-never-bound",
                0,
                TaskStatus::Completed,
                None,
            )
            .await
            .expect("classify legacy Goal transition"),
        GoalTransitionResult::Stale,
        "legacy null-session rows must return a typed nonmatching-binding result"
    );
    assert!(
        migrated
            .current_goal_for_session("legacy-session")
            .await
            .expect("current Goal query")
            .is_none()
    );
}

#[tokio::test]
async fn boot_recovery_pauses_clean_goals_and_fails_unsettled_operations() {
    let directory = tempfile::tempdir().expect("create temporary control-plane directory");
    let store = SqliteTaskStore::new(directory.path()).expect("initialize store");
    assert_eq!(
        store
            .create_or_replace_session_goal(
                session_goal_task("clean", "clean-session"),
                session_goal_extension("clean"),
            )
            .await
            .expect("create clean Goal"),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .create_or_replace_session_goal(
                session_goal_task("in-flight", "in-flight-session"),
                session_goal_extension("in-flight"),
            )
            .await
            .expect("create in-flight Goal"),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .begin_goal_tool_phase("in-flight", "in-flight-session", 1)
            .await
            .expect("record unpaired process-local tool phase"),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .create_or_replace_session_goal(
                session_goal_task("pending", "pending-session"),
                session_goal_extension("pending"),
            )
            .await
            .expect("create pending Goal"),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .admit_pending_operation("pending", "pending-session", 1, "operation-pending")
            .await
            .expect("admit pending operation"),
        GoalTransitionResult::Applied
    );
    drop(store);

    let reopened = SqliteTaskStore::new(directory.path()).expect("reopen store");
    assert_eq!(
        reopened
            .reconcile_goal_boot_state("boot-new")
            .expect("reconcile previous boot"),
        3
    );
    let clean = reopened
        .current_goal_for_session("clean-session")
        .await
        .expect("read clean Goal")
        .expect("clean Goal exists");
    assert_eq!(clean.status, TaskStatus::Paused);
    assert_eq!(clean.execution_epoch, 2);
    assert_eq!(
        reopened
            .get_goal_task("clean")
            .await
            .expect("read clean extension")
            .expect("extension exists")
            .tool_phase,
        GoalToolPhase::Clean
    );
    assert_eq!(
        reopened
            .resume_session_goal("clean", "clean-session", 2, 2, "boot-new")
            .await
            .expect("resume post-restart Goal"),
        GoalTransitionResult::Applied
    );
    let in_flight = reopened
        .current_goal_for_session("in-flight-session")
        .await
        .expect("read in-flight Goal")
        .expect("in-flight Goal exists");
    assert_eq!(in_flight.status, TaskStatus::Failed);
    assert_eq!(in_flight.execution_epoch, 2);
    assert_eq!(
        reopened
            .resume_session_goal("in-flight", "in-flight-session", 2, 2, "boot-new")
            .await
            .expect("reject resume of interrupted tool phase"),
        GoalTransitionResult::Stale
    );
    assert_eq!(
        reopened
            .get_goal_task("in-flight")
            .await
            .expect("read in-flight extension")
            .expect("extension exists")
            .tool_phase,
        GoalToolPhase::InFlight
    );
    let pending = reopened
        .current_goal_for_session("pending-session")
        .await
        .expect("read pending Goal")
        .expect("pending Goal exists");
    assert_eq!(pending.status, TaskStatus::Failed);
    let pending_extension = reopened
        .get_goal_task("pending")
        .await
        .expect("read pending extension")
        .expect("extension exists");
    assert_eq!(
        pending_extension.accounting_state,
        GoalAccountingState::OutcomeUnknown
    );
    assert!(pending_extension.pending_call_id.is_none());
}

#[tokio::test]
async fn boot_recovery_fails_corrupt_session_goal_without_extension() {
    let directory = tempfile::tempdir().expect("create temporary control-plane directory");
    let store = SqliteTaskStore::new(directory.path()).expect("initialize store");
    let database = directory.path().join("control_plane.db");
    drop(store);
    Connection::open(&database)
        .expect("open fixture database")
        .execute(
            "INSERT INTO tasks (
                 id, kind, agent, status, owner_pid, owner_boot_id, session_id,
                 execution_epoch, started_at
             ) VALUES ('corrupt', 'goal', 'main', 'running', 1, 'boot-old', 'session-corrupt', 1, 'now')",
            [],
        )
        .expect("insert corrupt Goal fixture");

    let reopened = SqliteTaskStore::new(directory.path()).expect("reopen store");
    assert_eq!(
        reopened
            .reconcile_goal_boot_state("boot-new")
            .expect("reconcile corrupt Goal"),
        1
    );
    assert_eq!(
        reopened
            .get("corrupt")
            .await
            .expect("read corrupt Goal")
            .expect("row remains auditable")
            .status,
        TaskStatus::Failed
    );
    assert_eq!(
        reopened
            .delete_session_goal("corrupt", "session-corrupt", 2)
            .await
            .expect("dispose terminal corrupt Goal"),
        GoalTransitionResult::Applied,
        "a missing extension cannot hold a pending operation or block disposal"
    );
    assert!(
        reopened
            .current_goal_for_session("session-corrupt")
            .await
            .expect("read disposed corrupt Goal")
            .is_none()
    );

    Connection::open(&database)
        .expect("open replacement fixture database")
        .execute(
            "INSERT INTO tasks (
                 id, kind, agent, status, owner_pid, owner_boot_id, session_id,
                 execution_epoch, started_at
             ) VALUES ('corrupt-replace', 'goal', 'main', 'running', 1, 'boot-old',
                       'session-replace', 1, 'now')",
            [],
        )
        .expect("insert corrupt replacement fixture");
    let reopened = SqliteTaskStore::new(directory.path()).expect("reopen replacement fixture");
    assert_eq!(
        reopened
            .reconcile_goal_boot_state("boot-newer")
            .expect("fail corrupt predecessor"),
        1
    );
    assert_eq!(
        reopened
            .create_or_replace_session_goal(
                session_goal_task("replacement", "session-replace"),
                session_goal_extension("replacement"),
            )
            .await
            .expect("replace terminal corrupt Goal"),
        GoalTransitionResult::Applied,
        "a missing extension cannot block terminal replacement"
    );
}
