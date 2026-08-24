//! Contract coverage for the dormant, transport-neutral Goal controller.

use std::{fs::OpenOptions, io::Write, sync::Arc};

use async_trait::async_trait;
use parking_lot::Mutex;
use zeroclaw_config::{
    cost::{CostTracker, TokenUsage},
    schema::{Config, CostConfig},
};
use zeroclaw_runtime::{
    control_plane::{
        GoalPauseReason, GoalPauseState, GoalResumeSafety, GoalTaskRegistry, GoalTransitionResult,
        SqliteTaskStore, TaskRegistry, TaskStatus,
    },
    goal_mode::{
        GoalBudgetRequest, GoalCommand, GoalCommandErrorKey, GoalExecutionPreflight,
        GoalExecutionReadiness, GoalExecutionScope, GoalExecutor, GoalIngressContext,
        GoalMessageKey, GoalRuntime, GoalRuntimeError, parse_goal_command,
    },
};

#[derive(Clone)]
enum LaunchAction {
    Succeed,
    Fail,
    FenceThenFail(Arc<SqliteTaskStore>),
}

struct FakeExecutor {
    readiness: GoalExecutionReadiness,
    action: LaunchAction,
    launches: Mutex<Vec<GoalExecutionScope>>,
    preflights: Mutex<Vec<GoalExecutionPreflight>>,
    cancellations: Mutex<Vec<(String, String, i64)>>,
    cancel_failures_remaining: Mutex<usize>,
}

impl FakeExecutor {
    fn new(readiness: GoalExecutionReadiness, action: LaunchAction) -> Self {
        Self {
            readiness,
            action,
            launches: Mutex::new(Vec::new()),
            preflights: Mutex::new(Vec::new()),
            cancellations: Mutex::new(Vec::new()),
            cancel_failures_remaining: Mutex::new(0),
        }
    }

    fn with_cancel_failures(mut self, failures: usize) -> Self {
        *self.cancel_failures_remaining.get_mut() = failures;
        self
    }
}

#[async_trait]
impl GoalExecutor for FakeExecutor {
    async fn preflight(&self, request: &GoalExecutionPreflight) -> GoalExecutionReadiness {
        self.preflights.lock().push(request.clone());
        self.readiness
    }

    async fn launch(&self, scope: GoalExecutionScope) -> Result<(), ()> {
        self.launches.lock().push(scope.clone());
        match &self.action {
            LaunchAction::Succeed => Ok(()),
            LaunchAction::Fail => Err(()),
            LaunchAction::FenceThenFail(store) => {
                assert_eq!(
                    store
                        .pause_session_goal(
                            &scope.task_id,
                            &scope.session_id,
                            scope.execution_epoch,
                            GoalPauseState {
                                reason: GoalPauseReason::OperatorPaused,
                                description: None,
                                blockers: Vec::new(),
                            },
                        )
                        .await
                        .expect("fence persists"),
                    GoalTransitionResult::Applied
                );
                Err(())
            }
        }
    }

    async fn cancel(&self, task_id: &str, session_id: &str, epoch: i64) -> Result<(), ()> {
        self.cancellations
            .lock()
            .push((task_id.to_owned(), session_id.to_owned(), epoch));
        let mut failures = self.cancel_failures_remaining.lock();
        if *failures > 0 {
            *failures -= 1;
            return Err(());
        }
        Ok(())
    }
}

fn enabled_config() -> Arc<Config> {
    let mut config = Config::default();
    config.goal.enabled = true;
    config.goal.default_token_limit = Some(64);
    config.goal.verifier.provider = "openai.verifier".into();
    let _ = config.providers.models.ensure("openai", "verifier");
    Arc::new(config)
}

fn matrix(session_id: &str) -> GoalIngressContext {
    GoalIngressContext::matrix(session_id, "main", "matrix.room", "@alice:example.test")
        .expect("trusted test ingress")
}

fn controller_with_config(
    config: Arc<Config>,
    executor: Arc<FakeExecutor>,
) -> (
    GoalRuntime,
    Arc<SqliteTaskStore>,
    Arc<CostTracker>,
    tempfile::TempDir,
) {
    let store = Arc::new(SqliteTaskStore::new_in_memory().expect("test storage"));
    let goals: Arc<dyn GoalTaskRegistry> = store.clone();
    let cost_dir = tempfile::tempdir().expect("temporary canonical cost ledger");
    let tracker = Arc::new(
        CostTracker::new(CostConfig::default(), cost_dir.path()).expect("canonical cost tracker"),
    );
    (
        GoalRuntime::new(config, goals, tracker.clone(), executor, "test-boot")
            .expect("controller"),
        store,
        tracker,
        cost_dir,
    )
}

fn controller(
    executor: Arc<FakeExecutor>,
) -> (
    GoalRuntime,
    Arc<SqliteTaskStore>,
    Arc<CostTracker>,
    tempfile::TempDir,
) {
    controller_with_config(enabled_config(), executor)
}

fn test_tracker() -> (Arc<CostTracker>, tempfile::TempDir) {
    let cost_dir = tempfile::tempdir().expect("temporary canonical cost ledger");
    let tracker = Arc::new(
        CostTracker::new(CostConfig::default(), cost_dir.path()).expect("canonical cost tracker"),
    );
    (tracker, cost_dir)
}

#[tokio::test]
async fn launched_goal_scope_owns_one_logical_operation_permit() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, _, _, _) = controller(Arc::clone(&executor));

    runtime
        .submit(
            matrix("session-permit"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Limits {
                    token_limit: Some(64),
                    cost_limit_usd: None,
                },
                success_criteria: "return a verified result".into(),
                objective: "exercise the operation permit".into(),
            },
        )
        .await
        .expect("start should launch the scope");

    let scope = executor.launches.lock()[0].clone();
    let permit = scope
        .operation_permit
        .try_acquire()
        .expect("first operation owns the only permit");
    assert!(scope.operation_permit.try_acquire().is_err());
    drop(permit);
}

#[tokio::test]
async fn start_preflight_receives_the_exact_immutable_ingress_binding() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, _, _, _) = controller(Arc::clone(&executor));

    runtime
        .submit(
            matrix("session-binding"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Limits {
                    token_limit: Some(64),
                    cost_limit_usd: None,
                },
                success_criteria: "return a verified result".into(),
                objective: "exercise trusted ingress binding".into(),
            },
        )
        .await
        .expect("start should preflight trusted ingress");

    let binding = executor.preflights.lock()[0].binding.clone();
    assert_eq!(binding.session_id(), "session-binding");
    assert_eq!(binding.agent(), "main");
    assert_eq!(binding.route(), "matrix.room");
    assert_eq!(binding.principal_id(), Some("@alice:example.test"));
}

#[test]
fn start_requires_delimiters_and_preserves_criteria_and_objective_verbatim() {
    let command = parse_goal_command(
        "  /goal start --tokens 250 --cost-usd 1.25 --success-criteria  Ship the release safely   --  Ship the release -- safely  ",
    )
    .expect("the V1 start grammar should parse");

    assert_eq!(
        command,
        GoalCommand::Start {
            budget: GoalBudgetRequest::Limits {
                token_limit: Some(250),
                cost_limit_usd: Some(1.25),
            },
            success_criteria: "  Ship the release safely   ".into(),
            objective: "  Ship the release -- safely  ".into(),
        }
    );
}

#[test]
fn no_legacy_goal_syntax_or_extra_authority_input_is_accepted() {
    for input in [
        "/goal objective mutate this",
        "/goal start --tokens=42 --success-criteria complete -- objective",
        "/goal status task-id",
        "/goal pause because",
        "/goal start objective without delimiter",
    ] {
        assert!(parse_goal_command(input).is_err(), "{input}");
    }
}

#[test]
fn start_without_criteria_has_a_stable_typed_error() {
    assert_eq!(
        parse_goal_command("/goal start -- objective")
            .unwrap_err()
            .key,
        GoalCommandErrorKey::SuccessCriteriaRequired
    );
    assert_eq!(
        parse_goal_command("/goal start --success-criteria complete")
            .unwrap_err()
            .key,
        GoalCommandErrorKey::ObjectiveRequired
    );
}

#[test]
fn grammar_rejects_every_disallowed_budget_criteria_and_objective_form() {
    for input in [
        "/goal start --",
        "/goal start --tokens 0 --success-criteria complete -- objective",
        "/goal start --tokens 42 --tokens 43 --success-criteria complete -- objective",
        "/goal start --cost-usd NaN --success-criteria complete -- objective",
        "/goal start --cost-usd inf --success-criteria complete -- objective",
        "/goal start --unlimited --tokens 42 --success-criteria complete -- objective",
        "/goal start --unknown 42 --success-criteria complete -- objective",
        "/goal start -- objective without criteria",
        "/goal start --success-criteria -- objective",
        "/goal start --success-criteria complete objective without delimiter",
        "/goal start --success-criteria complete --success-criteria duplicate -- objective",
        "/goal start --success-criteria complete --tokens 42 -- objective",
        "/goal start --success-criteria=complete -- objective",
        "/goal budget set",
        "/goal budget set --tokens 42 trailing",
        "/goal budget set --cost-usd=1.25",
        "/goal resume later",
        "/goal cancel task-id",
    ] {
        assert!(parse_goal_command(input).is_err(), "{input}");
    }
}

#[test]
fn every_permitted_non_start_command_has_one_exact_shape() {
    let cases = [
        ("/goal status", GoalCommand::Status),
        ("/goal budget", GoalCommand::Budget),
        ("/goal pause", GoalCommand::Pause),
        ("/goal resume", GoalCommand::Resume),
        ("/goal cancel", GoalCommand::Cancel),
        ("/goal help", GoalCommand::Help),
        (
            "/goal budget set --tokens 42",
            GoalCommand::SetBudget(GoalBudgetRequest::Limits {
                token_limit: Some(42),
                cost_limit_usd: None,
            }),
        ),
        (
            "/goal budget set --unlimited",
            GoalCommand::SetBudget(GoalBudgetRequest::Unlimited),
        ),
        (
            "/goal start --unlimited --success-criteria All checks pass -- retained objective",
            GoalCommand::Start {
                budget: GoalBudgetRequest::Unlimited,
                success_criteria: " All checks pass ".into(),
                objective: " retained objective".into(),
            },
        ),
        (
            "/goal start --success-criteria completion is verified -- defaulted objective",
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: " completion is verified ".into(),
                objective: " defaulted objective".into(),
            },
        ),
    ];

    for (input, expected) in cases {
        assert_eq!(parse_goal_command(input).unwrap(), expected, "{input}");
    }
}

#[tokio::test]
async fn commands_follow_the_session_bound_state_table() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, store, _, _) = controller(executor.clone());

    let started = runtime
        .submit(
            matrix("session-a"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "complete".into(),
                objective: "finish the migration".into(),
            },
        )
        .await
        .unwrap();
    assert_eq!(started.key, GoalMessageKey::Started);
    assert_eq!(executor.launches.lock().len(), 1);
    let goal = store
        .get_goal_task(
            &store
                .current_goal_for_session("session-a")
                .await
                .unwrap()
                .expect("current goal")
                .id,
        )
        .await
        .unwrap()
        .expect("goal extension");
    assert_eq!(goal.success_criteria, "complete");

    assert_eq!(
        runtime
            .submit(matrix("session-a"), GoalCommand::Pause)
            .await
            .unwrap()
            .key,
        GoalMessageKey::Paused
    );
    assert_eq!(
        runtime
            .submit(
                matrix("session-a"),
                GoalCommand::SetBudget(GoalBudgetRequest::Unlimited)
            )
            .await
            .unwrap()
            .key,
        GoalMessageKey::BudgetUpdated
    );
    assert_eq!(
        runtime
            .submit(matrix("session-a"), GoalCommand::Status)
            .await
            .unwrap()
            .status
            .expect("current status")
            .status,
        TaskStatus::Paused,
        "budget replacement must never resume a paused goal"
    );
    assert_eq!(
        runtime
            .submit(matrix("session-a"), GoalCommand::Resume)
            .await
            .unwrap()
            .key,
        GoalMessageKey::Resumed
    );
    assert_eq!(executor.launches.lock().len(), 2);

    let current = store
        .current_goal_for_session("session-a")
        .await
        .unwrap()
        .expect("current goal");
    assert_eq!(current.status, TaskStatus::Running);
    assert_eq!(current.execution_epoch, 3);
}

#[tokio::test]
async fn invalid_typed_budget_and_executor_failure_fail_closed() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Fail,
    ));
    let (runtime, store, _, _) = controller(executor);

    assert_eq!(
        runtime
            .submit(
                matrix("session-a"),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Limits {
                        token_limit: Some(0),
                        cost_limit_usd: None,
                    },
                    success_criteria: "complete".into(),
                    objective: "invalid".into(),
                },
            )
            .await
            .unwrap()
            .key,
        GoalMessageKey::InvalidBudget
    );
    assert_eq!(
        runtime
            .submit(
                matrix("session-empty-start"),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Limits {
                        token_limit: None,
                        cost_limit_usd: None,
                    },
                    success_criteria: "complete".into(),
                    objective: "must not become unlimited".into(),
                },
            )
            .await
            .unwrap()
            .key,
        GoalMessageKey::InvalidBudget,
        "only the explicit Unlimited variant may disable both limits"
    );
    assert_eq!(
        runtime
            .submit(
                matrix("session-a"),
                GoalCommand::SetBudget(GoalBudgetRequest::Defaults),
            )
            .await
            .unwrap()
            .key,
        GoalMessageKey::InvalidCommand,
        "only start may resolve configured defaults"
    );
    assert_eq!(
        runtime
            .submit(
                matrix("session-a"),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    success_criteria: "complete".into(),
                    objective: " \t ".into(),
                },
            )
            .await
            .unwrap()
            .key,
        GoalMessageKey::InvalidCommand,
        "typed commands must not bypass the nonblank-objective grammar invariant"
    );

    let response = runtime
        .submit(
            matrix("session-a"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "complete".into(),
                objective: "will fail to launch".into(),
            },
        )
        .await
        .unwrap();
    assert_eq!(response.key, GoalMessageKey::ExecutorStartFailed);
    let task = store
        .current_goal_for_session("session-a")
        .await
        .unwrap()
        .expect("failed goal remains visible");
    assert_eq!(task.status, TaskStatus::Failed);

    assert_eq!(
        runtime
            .submit(
                matrix("session-empty-criteria"),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    success_criteria: " \t ".into(),
                    objective: "must retain explicit completion criteria".into(),
                },
            )
            .await
            .unwrap()
            .key,
        GoalMessageKey::SuccessCriteriaRequired,
        "typed commands must not bypass the nonblank-criteria grammar invariant"
    );
}

#[tokio::test]
async fn status_projects_pending_reservations_and_nonresumable_tool_phase() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, store, tracker, _cost_dir) = controller(executor.clone());

    runtime
        .submit(
            matrix("session-a"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "the work is complete".into(),
                objective: "perform the work".into(),
            },
        )
        .await
        .unwrap();
    let current = store
        .current_goal_for_session("session-a")
        .await
        .unwrap()
        .expect("current goal");
    tracker
        .record_scoped_usage_with_owned_task_attribution(
            TokenUsage::new("test/model", 8, 2, 0, 0.0, 0.0, 0.0),
            Some("main"),
            Some(current.id.clone()),
        )
        .expect("known lower-bound usage");
    assert_eq!(
        store
            .admit_pending_call_with_reservation(
                &current.id,
                "session-a",
                current.execution_epoch,
                "pending-operation",
                Some(32),
                None,
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    let status = runtime
        .submit(matrix("session-a"), GoalCommand::Status)
        .await
        .unwrap()
        .status
        .expect("status projection");
    assert_eq!(status.pending_reserved_tokens, Some(32));
    assert_eq!(status.known_token_usage, 10);
    assert!(!status.token_accounting_complete);
    assert!(!status.cost_accounting_complete);
    assert_eq!(status.remaining_tokens, None);
    assert_eq!(status.remaining_cost_usd, None);
    assert!(!status.resumable);

    assert_eq!(
        store
            .settle_pending_call(
                &current.id,
                "session-a",
                current.execution_epoch,
                "pending-operation",
                zeroclaw_runtime::control_plane::GoalAccountingState::Complete,
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .mark_goal_tool_executed(&current.id, "session-a", current.execution_epoch)
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );
    assert_eq!(
        store
            .pause_session_goal(
                &current.id,
                "session-a",
                current.execution_epoch,
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
    let status = runtime
        .submit(matrix("session-a"), GoalCommand::Status)
        .await
        .unwrap()
        .status
        .expect("status projection");
    assert_eq!(status.resume_safety, GoalResumeSafety::ToolExecuted);
    assert!(!status.resumable);
    assert_eq!(
        runtime
            .submit(matrix("session-a"), GoalCommand::Resume)
            .await
            .unwrap()
            .key,
        GoalMessageKey::ToolPhaseNonResumable
    );
    assert_eq!(executor.launches.lock().len(), 1);
}

#[tokio::test]
async fn status_never_advertises_resume_at_the_execution_epoch_limit() {
    let directory = tempfile::tempdir().expect("temporary control-plane directory");
    let store = Arc::new(SqliteTaskStore::new(directory.path()).expect("test storage"));
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let goals: Arc<dyn GoalTaskRegistry> = store.clone();
    let (tracker, _cost_dir) = test_tracker();
    let runtime = GoalRuntime::new(enabled_config(), goals, tracker, executor, "test-boot")
        .expect("controller");

    runtime
        .submit(
            matrix("session-max-epoch"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "the work is complete".into(),
                objective: "perform the work".into(),
            },
        )
        .await
        .unwrap();
    runtime
        .submit(matrix("session-max-epoch"), GoalCommand::Pause)
        .await
        .unwrap();
    let current = store
        .current_goal_for_session("session-max-epoch")
        .await
        .unwrap()
        .expect("current Goal");
    rusqlite::Connection::open(directory.path().join("control_plane.db"))
        .expect("raw connection")
        .execute(
            "UPDATE tasks SET execution_epoch = ?1 WHERE id = ?2",
            rusqlite::params![i64::MAX, current.id],
        )
        .expect("test epoch mutation");

    let status = runtime
        .submit(matrix("session-max-epoch"), GoalCommand::Status)
        .await
        .unwrap()
        .status
        .expect("status projection");
    assert!(!status.resumable);
}

#[tokio::test]
async fn typed_empty_budget_replacement_does_not_become_unlimited() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, store, _, _) = controller(executor);

    assert_eq!(
        runtime
            .submit(
                matrix("session-a"),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    success_criteria: "complete".into(),
                    objective: "retain the finite default".into(),
                },
            )
            .await
            .unwrap()
            .key,
        GoalMessageKey::Started
    );
    assert_eq!(
        runtime
            .submit(
                matrix("session-a"),
                GoalCommand::SetBudget(GoalBudgetRequest::Limits {
                    token_limit: None,
                    cost_limit_usd: None,
                }),
            )
            .await
            .unwrap()
            .key,
        GoalMessageKey::InvalidBudget,
        "only the explicit Unlimited variant may disable both limits"
    );

    let current = store
        .current_goal_for_session("session-a")
        .await
        .unwrap()
        .expect("the active goal remains current");
    let goal = store
        .get_goal_task(&current.id)
        .await
        .unwrap()
        .expect("the current task retains its Goal extension");
    assert_eq!(goal.effective_token_limit, Some(64));
    assert_eq!(goal.effective_cost_limit_usd, None);
}

#[tokio::test]
async fn stale_executor_failure_cannot_overwrite_a_newer_fence() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().expect("test storage"));
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::FenceThenFail(store.clone()),
    ));
    let goals: Arc<dyn GoalTaskRegistry> = store.clone();
    let (tracker, _cost_dir) = test_tracker();
    let runtime =
        GoalRuntime::new(enabled_config(), goals, tracker, executor, "test-boot").unwrap();

    let response = runtime
        .submit(
            matrix("session-a"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "complete".into(),
                objective: "be fenced during launch".into(),
            },
        )
        .await
        .unwrap();
    assert_eq!(response.key, GoalMessageKey::Stale);
    assert_eq!(
        store
            .current_goal_for_session("session-a")
            .await
            .unwrap()
            .expect("current goal")
            .status,
        TaskStatus::Paused
    );
}

#[tokio::test]
async fn session_disposal_fences_quiesces_then_deletes_goal_control_state() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, store, _, _) = controller(Arc::clone(&executor));

    runtime
        .submit(
            matrix("session-dispose"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "complete the work".into(),
                objective: "exercise session disposal".into(),
            },
        )
        .await
        .expect("start should create a Goal");
    let task_id = store
        .current_goal_for_session("session-dispose")
        .await
        .expect("read current Goal")
        .expect("Goal exists")
        .id;

    runtime
        .dispose_session(&matrix("session-dispose"))
        .await
        .expect("disposal should quiesce and delete a settled Goal");

    assert_eq!(
        executor.cancellations.lock().as_slice(),
        [(task_id.clone(), "session-dispose".to_owned(), 1)],
        "the old execution epoch must be cancelled after its durable fence"
    );
    assert!(
        store
            .current_goal_for_session("session-dispose")
            .await
            .expect("read current Goal after disposal")
            .is_none(),
        "session disposal must hard-delete Goal control state"
    );
    assert!(
        store
            .get(&task_id)
            .await
            .expect("read disposed task")
            .is_none(),
        "the terminal task row must not remain as a historical current Goal"
    );
}

#[tokio::test]
async fn session_disposal_retries_a_durably_fenced_executor_before_deleting_control_state() {
    let executor = Arc::new(
        FakeExecutor::new(GoalExecutionReadiness::Ready, LaunchAction::Succeed)
            .with_cancel_failures(1),
    );
    let (runtime, store, _, _) = controller(Arc::clone(&executor));

    runtime
        .submit(
            matrix("session-dispose-retry"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "complete the work".into(),
                objective: "exercise retry-safe disposal".into(),
            },
        )
        .await
        .expect("start should create a Goal");
    let task_id = store
        .current_goal_for_session("session-dispose-retry")
        .await
        .expect("read current Goal")
        .expect("Goal exists")
        .id;

    assert!(
        runtime
            .dispose_session(&matrix("session-dispose-retry"))
            .await
            .is_err(),
        "a failed executor join must retain the fenced Goal for retry"
    );
    assert_eq!(
        store
            .current_goal_for_session("session-dispose-retry")
            .await
            .expect("read retained Goal")
            .expect("Goal remains while quiescence is unproved")
            .status,
        TaskStatus::Cancelled
    );

    runtime
        .dispose_session(&matrix("session-dispose-retry"))
        .await
        .expect("retry must quiesce the original fenced executor before deletion");
    assert!(
        store
            .current_goal_for_session("session-dispose-retry")
            .await
            .expect("read disposed Goal")
            .is_none()
    );
    assert_eq!(
        executor.cancellations.lock().as_slice(),
        [
            (task_id.clone(), "session-dispose-retry".to_owned(), 1),
            (task_id, "session-dispose-retry".to_owned(), 1)
        ],
        "retry must target only the original fenced executor epoch"
    );
}

#[tokio::test]
async fn idempotent_pause_retries_a_prior_failed_executor_quiescence() {
    let executor = Arc::new(
        FakeExecutor::new(GoalExecutionReadiness::Ready, LaunchAction::Succeed)
            .with_cancel_failures(1),
    );
    let (runtime, store, _, _) = controller(Arc::clone(&executor));
    let ingress = matrix("session-pause-retry");
    runtime
        .submit(
            ingress.clone(),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "complete the work".into(),
                objective: "exercise idempotent pause retry".into(),
            },
        )
        .await
        .expect("start Goal");
    let task_id = store
        .current_goal_for_session("session-pause-retry")
        .await
        .expect("read Goal")
        .expect("Goal exists")
        .id;

    assert!(
        runtime
            .submit(ingress.clone(), GoalCommand::Pause)
            .await
            .is_err(),
        "the first pause retains a durable fence when executor quiescence fails"
    );
    assert_eq!(
        runtime
            .submit(ingress, GoalCommand::Pause)
            .await
            .expect("idempotent pause retries the exact fence")
            .key,
        GoalMessageKey::AlreadyPaused
    );
    assert_eq!(
        executor.cancellations.lock().as_slice(),
        [
            (task_id.clone(), "session-pause-retry".to_owned(), 1),
            (task_id, "session-pause-retry".to_owned(), 1),
        ]
    );
}

#[tokio::test]
async fn idempotent_cancel_retries_a_prior_failed_executor_quiescence() {
    let executor = Arc::new(
        FakeExecutor::new(GoalExecutionReadiness::Ready, LaunchAction::Succeed)
            .with_cancel_failures(1),
    );
    let (runtime, store, _, _) = controller(Arc::clone(&executor));
    let ingress = matrix("session-cancel-retry");
    runtime
        .submit(
            ingress.clone(),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "complete the work".into(),
                objective: "exercise idempotent cancel retry".into(),
            },
        )
        .await
        .expect("start Goal");
    let task_id = store
        .current_goal_for_session("session-cancel-retry")
        .await
        .expect("read Goal")
        .expect("Goal exists")
        .id;

    assert!(
        runtime
            .submit(ingress.clone(), GoalCommand::Cancel)
            .await
            .is_err(),
        "the first cancel retains a durable fence when executor quiescence fails"
    );
    assert_eq!(
        runtime
            .submit(ingress, GoalCommand::Cancel)
            .await
            .expect("idempotent cancel retries the exact fence")
            .key,
        GoalMessageKey::AlreadyCancelled
    );
    assert_eq!(
        executor.cancellations.lock().as_slice(),
        [
            (task_id.clone(), "session-cancel-retry".to_owned(), 1),
            (task_id, "session-cancel-retry".to_owned(), 1),
        ]
    );
}

#[tokio::test]
async fn resume_refuses_a_paused_goal_until_its_fenced_executor_quiesces() {
    let executor = Arc::new(
        FakeExecutor::new(GoalExecutionReadiness::Ready, LaunchAction::Succeed)
            .with_cancel_failures(2),
    );
    let (runtime, store, _, _) = controller(Arc::clone(&executor));
    let ingress = matrix("session-resume-quiescence");
    runtime
        .submit(
            ingress.clone(),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "complete the work".into(),
                objective: "exercise resume quiescence".into(),
            },
        )
        .await
        .expect("start Goal");

    assert!(
        runtime.interrupt_session(&ingress).await.is_err(),
        "the first external interruption leaves the exact executor fenced but unquiesced"
    );
    assert!(
        runtime
            .submit(ingress.clone(), GoalCommand::Resume)
            .await
            .is_err(),
        "resume must not launch a successor while the fenced executor still fails to quiesce"
    );
    let current = store
        .current_goal_for_session("session-resume-quiescence")
        .await
        .expect("read Goal")
        .expect("Goal remains");
    assert_eq!(current.status, TaskStatus::Paused);
    assert_eq!(current.execution_epoch, 2);
    assert_eq!(executor.launches.lock().len(), 1);
}

#[tokio::test]
async fn runtime_cutover_fences_and_quiesces_every_running_goal() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, store, _, _) = controller(Arc::clone(&executor));

    for session_id in ["session-cutover-a", "session-cutover-b"] {
        runtime
            .submit(
                matrix(session_id),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    success_criteria: "complete the work".into(),
                    objective: format!("exercise daemon cutover for {session_id}"),
                },
            )
            .await
            .expect("start should create a running Goal");
    }

    assert_eq!(
        runtime
            .pause_running_goals_for_runtime_cutover()
            .await
            .expect("cutover should fence every running Goal"),
        2
    );
    assert_eq!(
        runtime
            .submit(
                matrix("session-cutover-successor"),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    success_criteria: "complete the work".into(),
                    objective: "must not be admitted after cutover".into(),
                },
            )
            .await
            .expect("cutover rejection is a typed response")
            .key,
        GoalMessageKey::Stale,
        "the outgoing runtime generation must not admit a successor Goal"
    );

    for session_id in ["session-cutover-a", "session-cutover-b"] {
        let status = store
            .current_goal_status_for_session(session_id)
            .await
            .expect("read cutover Goal")
            .expect("Goal remains retained for explicit resume");
        assert_eq!(status.task.status, TaskStatus::Paused);
        assert_eq!(status.task.execution_epoch, 2);
        assert_eq!(
            status.goal.pause_reason,
            Some(GoalPauseReason::DaemonRestart)
        );
    }
    let cancellations = executor.cancellations.lock();
    assert_eq!(cancellations.len(), 2);
    assert!(cancellations.iter().all(|(_, _, epoch)| *epoch == 1));
}

#[tokio::test]
async fn runtime_cutover_retries_a_previous_fence_before_reporting_success() {
    let executor = Arc::new(
        FakeExecutor::new(GoalExecutionReadiness::Ready, LaunchAction::Succeed)
            .with_cancel_failures(1),
    );
    let (runtime, store, _, _) = controller(Arc::clone(&executor));
    runtime
        .submit(
            matrix("session-cutover-retry"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "complete the work".into(),
                objective: "exercise retry-safe runtime cutover".into(),
            },
        )
        .await
        .expect("start should create a running Goal");
    let task_id = store
        .current_goal_for_session("session-cutover-retry")
        .await
        .expect("read current Goal")
        .expect("Goal exists")
        .id;

    assert!(
        runtime
            .pause_running_goals_for_runtime_cutover()
            .await
            .is_err(),
        "cutover must not report success before its fenced executor quiesces"
    );
    assert_eq!(
        store
            .current_goal_for_session("session-cutover-retry")
            .await
            .expect("read fenced Goal")
            .expect("Goal remains")
            .status,
        TaskStatus::Paused
    );
    assert_eq!(
        runtime
            .pause_running_goals_for_runtime_cutover()
            .await
            .expect("retry quiesces the retained fence"),
        0,
        "the retry must not pretend it freshly paused an already fenced Goal"
    );
    assert_eq!(
        executor.cancellations.lock().as_slice(),
        [
            (task_id.clone(), "session-cutover-retry".to_owned(), 1),
            (task_id, "session-cutover-retry".to_owned(), 1)
        ]
    );
}

#[tokio::test]
async fn generation_cutover_fences_only_the_matching_live_session_owner() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, store, _, _) = controller(Arc::clone(&executor));
    runtime
        .submit(
            matrix("session-matrix-generation"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "complete the work".into(),
                objective: "exercise Matrix generation cutover".into(),
            },
        )
        .await
        .expect("start Matrix Goal");
    let rpc = GoalIngressContext::zero_code("session-rpc-generation", "main", "rpc", "tui-a")
        .expect("trusted RPC ingress");
    runtime
        .submit(
            rpc,
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "complete the work".into(),
                objective: "must remain live across Matrix generation cutover".into(),
            },
        )
        .await
        .expect("start RPC Goal");

    assert_eq!(
        runtime
            .pause_running_goals_for_lifecycle_cutover(|task| {
                task.originator_route.as_deref() == Some("matrix.room")
            })
            .await
            .expect("generation cutover should fence the matching Goal"),
        1
    );
    assert_eq!(
        store
            .current_goal_for_session("session-matrix-generation")
            .await
            .unwrap()
            .expect("Matrix Goal")
            .status,
        TaskStatus::Paused
    );
    assert_eq!(
        store
            .current_goal_for_session("session-rpc-generation")
            .await
            .unwrap()
            .expect("RPC Goal")
            .status,
        TaskStatus::Running,
        "a Matrix generation replacement must not pause an unrelated RPC Goal"
    );
    assert_eq!(executor.cancellations.lock().len(), 1);
}

#[tokio::test]
async fn policy_revocation_cancels_matching_running_and_paused_goals() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, store, _, _) = controller(Arc::clone(&executor));
    for session_id in ["session-revoke-running", "session-revoke-paused"] {
        runtime
            .submit(
                matrix(session_id),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    success_criteria: "complete the work".into(),
                    objective: format!("exercise policy revocation for {session_id}"),
                },
            )
            .await
            .expect("start Goal");
    }
    runtime
        .submit(matrix("session-revoke-paused"), GoalCommand::Pause)
        .await
        .expect("pause retained Goal");

    assert_eq!(
        runtime
            .cancel_nonterminal_goals_for_policy_revocation(|task| {
                task.originator_route.as_deref() == Some("matrix.room")
            })
            .await
            .expect("policy revocation should cancel matching Goals"),
        2
    );
    for session_id in ["session-revoke-running", "session-revoke-paused"] {
        assert_eq!(
            store
                .current_goal_for_session(session_id)
                .await
                .unwrap()
                .expect("retained cancellation audit")
                .status,
            TaskStatus::Cancelled
        );
    }
    assert_eq!(
        store
            .current_goal_for_session("session-revoke-running")
            .await
            .unwrap()
            .expect("running Goal")
            .execution_epoch,
        2
    );
}

#[tokio::test]
async fn policy_revocation_retries_every_durably_cancelled_executor_before_success() {
    let executor = Arc::new(
        FakeExecutor::new(GoalExecutionReadiness::Ready, LaunchAction::Succeed)
            .with_cancel_failures(1),
    );
    let (runtime, store, _, _) = controller(Arc::clone(&executor));
    runtime
        .submit(
            matrix("session-revoke-retry"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "complete the work".into(),
                objective: "exercise retry-safe policy revocation".into(),
            },
        )
        .await
        .expect("start Goal");
    let task_id = store
        .current_goal_for_session("session-revoke-retry")
        .await
        .expect("read Goal")
        .expect("Goal exists")
        .id;

    assert!(
        runtime
            .cancel_nonterminal_goals_for_policy_revocation(|task| {
                task.originator_route.as_deref() == Some("matrix.room")
            })
            .await
            .is_err(),
        "policy installation must stop while a durably cancelled executor remains unquiesced"
    );
    assert_eq!(
        store
            .current_goal_for_session("session-revoke-retry")
            .await
            .expect("read retained audit")
            .expect("Goal remains")
            .status,
        TaskStatus::Cancelled
    );

    assert_eq!(
        runtime
            .cancel_nonterminal_goals_for_policy_revocation(|task| {
                task.originator_route.as_deref() == Some("matrix.room")
            })
            .await
            .expect("retry must quiesce before policy installation"),
        0,
        "the retry must not count a second lifecycle transition"
    );
    assert_eq!(
        executor.cancellations.lock().as_slice(),
        [
            (task_id.clone(), "session-revoke-retry".to_owned(), 1),
            (task_id, "session-revoke-retry".to_owned(), 1)
        ]
    );
}

#[tokio::test]
async fn session_disposal_retains_a_goal_when_quiescence_cannot_classify_pending_accounting() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, store, _, _) = controller(Arc::clone(&executor));

    runtime
        .submit(
            matrix("session-pending-dispose"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "complete the work".into(),
                objective: "exercise unsettled session disposal".into(),
            },
        )
        .await
        .expect("start should create a Goal");
    let current = store
        .current_goal_for_session("session-pending-dispose")
        .await
        .expect("read current Goal")
        .expect("Goal exists");
    assert_eq!(
        store
            .admit_pending_call_with_reservation(
                &current.id,
                "session-pending-dispose",
                current.execution_epoch,
                "pending-disposal-operation",
                Some(64),
                None,
            )
            .await
            .expect("admit pending operation"),
        GoalTransitionResult::Applied
    );

    assert_eq!(
        runtime
            .dispose_session(&matrix("session-pending-dispose"))
            .await
            .expect("unclassified accounting must not turn into an error"),
        GoalTransitionResult::Stale,
        "deletion is forbidden until the executor classifies the admitted operation"
    );
    let retained = store
        .current_goal_for_session("session-pending-dispose")
        .await
        .expect("read retained Goal")
        .expect("unclassified Goal remains durable");
    assert_eq!(retained.id, current.id);
    assert_eq!(retained.status, TaskStatus::Cancelled);
    assert_eq!(
        executor.cancellations.lock().as_slice(),
        [(current.id, "session-pending-dispose".to_owned(), 1)]
    );
}

#[tokio::test]
async fn session_disposal_cannot_target_a_goal_owned_by_another_transport_binding() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, store, _, _) = controller(executor);
    runtime
        .submit(
            matrix("shared-session-id"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "complete the work".into(),
                objective: "retain transport ownership".into(),
            },
        )
        .await
        .expect("start should create a Matrix Goal");
    let rpc_binding =
        GoalIngressContext::zero_code("shared-session-id", "main", "rpc", "tui-connection")
            .expect("trusted RPC binding");

    assert_eq!(
        runtime
            .dispose_session(&rpc_binding)
            .await
            .expect("foreign binding is not an error"),
        GoalTransitionResult::Missing
    );
    assert_eq!(
        store
            .current_goal_for_session("shared-session-id")
            .await
            .unwrap()
            .expect("the Matrix Goal remains")
            .status,
        TaskStatus::Running
    );
}

#[tokio::test]
async fn disabled_mode_and_untrusted_session_rebinding_fail_closed() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (disabled_runtime, _, _, _) = controller_with_config(Arc::new(Config::default()), executor);
    assert_eq!(
        disabled_runtime
            .submit(
                matrix("session-a"),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Unlimited,
                    success_criteria: "complete".into(),
                    objective: "must not start".into(),
                },
            )
            .await
            .unwrap()
            .key,
        GoalMessageKey::Disabled
    );

    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, _, _, _) = controller(executor);
    runtime
        .submit(
            matrix("session-a"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "complete".into(),
                objective: "bound to Alice".into(),
            },
        )
        .await
        .unwrap();
    let bob = GoalIngressContext::matrix("session-a", "main", "matrix.room", "@bob:example.test")
        .unwrap();
    assert_eq!(
        runtime.submit(bob, GoalCommand::Status).await.unwrap().key,
        GoalMessageKey::NoCurrentGoal
    );
}

#[tokio::test]
async fn pricing_readiness_only_blocks_cost_limited_admission() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::PricingUnavailable,
        LaunchAction::Succeed,
    ));
    let (runtime, store, _, _) = controller(executor);
    assert_eq!(
        runtime
            .submit(
                matrix("session-a"),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Limits {
                        token_limit: Some(10),
                        cost_limit_usd: None,
                    },
                    success_criteria: "complete".into(),
                    objective: "token only".into(),
                },
            )
            .await
            .unwrap()
            .key,
        GoalMessageKey::Started
    );
    assert!(
        store
            .current_goal_for_session("session-a")
            .await
            .unwrap()
            .is_some()
    );

    let (runtime, store, _, _) = controller(Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::PricingUnavailable,
        LaunchAction::Succeed,
    )));
    assert_eq!(
        runtime
            .submit(
                matrix("session-b"),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Limits {
                        token_limit: None,
                        cost_limit_usd: Some(1.0),
                    },
                    success_criteria: "complete".into(),
                    objective: "cost limited".into(),
                },
            )
            .await
            .unwrap()
            .key,
        GoalMessageKey::PricingUnavailable
    );
    assert!(
        store
            .current_goal_for_session("session-b")
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn status_and_budget_derive_complete_remaining_limits_from_the_canonical_ledger() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, _store, tracker, _cost_dir) = controller(executor);
    runtime
        .submit(
            matrix("session-ledger"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Limits {
                    token_limit: Some(100),
                    cost_limit_usd: Some(1.0),
                },
                success_criteria: "all required work is complete".into(),
                objective: "derive status from durable usage".into(),
            },
        )
        .await
        .expect("Goal starts");

    let task_id = runtime
        .submit(matrix("session-ledger"), GoalCommand::Status)
        .await
        .expect("initial status")
        .status
        .expect("status projection")
        .task_id;
    tracker
        .record_scoped_usage_with_owned_task_attribution(
            TokenUsage::new("test/model", 30, 20, 0, 1_000.0, 1_000.0, 0.0),
            Some("main"),
            Some(task_id),
        )
        .expect("durable usage record");

    for command in [GoalCommand::Status, GoalCommand::Budget] {
        let projection = runtime
            .submit(matrix("session-ledger"), command)
            .await
            .expect("read-only projection")
            .status
            .expect("status projection");
        assert_eq!(projection.known_token_usage, 50);
        assert!((projection.known_cost_usd - 0.05).abs() < f64::EPSILON);
        assert!(projection.token_accounting_complete);
        assert!(projection.cost_accounting_complete);
        assert_eq!(projection.remaining_tokens, Some(50));
        assert!(
            (projection.remaining_cost_usd.expect("cost remaining") - 0.95).abs() < f64::EPSILON
        );
    }
}

#[tokio::test]
async fn status_with_unpriced_usage_keeps_known_tokens_but_hides_cost_remaining() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, _store, tracker, _cost_dir) = controller(executor);
    runtime
        .submit(
            matrix("session-unpriced"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Limits {
                    token_limit: Some(100),
                    cost_limit_usd: Some(1.0),
                },
                success_criteria: "all required work is complete".into(),
                objective: "surface only complete budget dimensions".into(),
            },
        )
        .await
        .expect("Goal starts");
    let task_id = runtime
        .submit(matrix("session-unpriced"), GoalCommand::Status)
        .await
        .expect("initial status")
        .status
        .expect("status projection")
        .task_id;
    let mut usage = TokenUsage::new("test/unpriced", 40, 10, 0, 0.0, 0.0, 0.0);
    usage.pricing_available = false;
    tracker
        .record_scoped_usage_with_owned_task_attribution(usage, Some("main"), Some(task_id))
        .expect("durable unpriced usage");

    let projection = runtime
        .submit(matrix("session-unpriced"), GoalCommand::Status)
        .await
        .expect("status")
        .status
        .expect("status projection");
    assert_eq!(projection.known_token_usage, 50);
    assert!(projection.token_accounting_complete);
    assert!(!projection.cost_accounting_complete);
    assert_eq!(projection.remaining_tokens, Some(50));
    assert_eq!(projection.remaining_cost_usd, None);
}

#[tokio::test]
async fn malformed_canonical_usage_fails_status_without_claiming_zero_consumption() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, store, tracker, cost_dir) = controller(executor);
    runtime
        .submit(
            matrix("session-malformed-ledger"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "all required work is complete".into(),
                objective: "reject malformed durable usage".into(),
            },
        )
        .await
        .expect("Goal starts");
    let task_id = runtime
        .submit(matrix("session-malformed-ledger"), GoalCommand::Status)
        .await
        .expect("initial status")
        .status
        .expect("status projection")
        .task_id;
    tracker
        .record_scoped_usage_with_owned_task_attribution(
            TokenUsage::new("test/model", 1, 1, 0, 0.0, 0.0, 0.0),
            Some("main"),
            Some(task_id),
        )
        .expect("canonical ledger exists");
    let before = store
        .current_goal_for_session("session-malformed-ledger")
        .await
        .expect("current Goal lookup")
        .expect("current Goal");
    let mut ledger = OpenOptions::new()
        .append(true)
        .open(cost_dir.path().join("state/costs.jsonl"))
        .expect("canonical ledger");
    writeln!(ledger, "not-json").expect("corrupt fixture");
    ledger.sync_all().expect("fixture sync");

    assert!(matches!(
        runtime
            .submit(matrix("session-malformed-ledger"), GoalCommand::Status)
            .await,
        Err(GoalRuntimeError::Accounting(_))
    ));
    let after = store
        .current_goal_for_session("session-malformed-ledger")
        .await
        .expect("current Goal lookup")
        .expect("current Goal");
    assert_eq!(after.status, before.status);
    assert_eq!(after.execution_epoch, before.execution_epoch);
}

#[tokio::test]
async fn status_covers_no_usage_unlimited_exact_limit_and_overshoot() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, _store, tracker, _cost_dir) = controller(executor);

    runtime
        .submit(
            matrix("session-no-usage"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "all required work is complete".into(),
                objective: "show a clean no-usage projection".into(),
            },
        )
        .await
        .expect("Goal starts");
    let no_usage = runtime
        .submit(matrix("session-no-usage"), GoalCommand::Status)
        .await
        .expect("no-usage status")
        .status
        .expect("projection");
    assert_eq!(no_usage.known_token_usage, 0);
    assert!(no_usage.token_accounting_complete);
    assert_eq!(no_usage.remaining_tokens, Some(64));

    runtime
        .submit(
            matrix("session-unlimited-status"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Unlimited,
                success_criteria: "all required work is complete".into(),
                objective: "show unlimited dimensions as unbounded".into(),
            },
        )
        .await
        .expect("unlimited Goal starts");
    let unlimited = runtime
        .submit(matrix("session-unlimited-status"), GoalCommand::Status)
        .await
        .expect("unlimited status")
        .status
        .expect("projection");
    assert!(unlimited.token_accounting_complete);
    assert!(unlimited.cost_accounting_complete);
    assert_eq!(unlimited.remaining_tokens, None);
    assert_eq!(unlimited.remaining_cost_usd, None);

    runtime
        .submit(
            matrix("session-limit-edges"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Limits {
                    token_limit: Some(10),
                    cost_limit_usd: Some(0.01),
                },
                success_criteria: "all required work is complete".into(),
                objective: "show exact-limit and overshoot behavior".into(),
            },
        )
        .await
        .expect("finite Goal starts");
    let task_id = runtime
        .submit(matrix("session-limit-edges"), GoalCommand::Status)
        .await
        .expect("initial finite status")
        .status
        .expect("projection")
        .task_id;
    tracker
        .record_scoped_usage_with_owned_task_attribution(
            TokenUsage::new("test/model", 5, 5, 0, 1_000.0, 1_000.0, 0.0),
            Some("main"),
            Some(task_id.clone()),
        )
        .expect("exact-limit usage");
    let exact = runtime
        .submit(matrix("session-limit-edges"), GoalCommand::Status)
        .await
        .expect("exact-limit status")
        .status
        .expect("projection");
    assert_eq!(exact.remaining_tokens, Some(0));
    assert_eq!(exact.remaining_cost_usd, Some(0.0));
    tracker
        .record_scoped_usage_with_owned_task_attribution(
            TokenUsage::new("test/model", 1, 0, 0, 1_000.0, 1_000.0, 0.0),
            Some("main"),
            Some(task_id),
        )
        .expect("overshoot usage");
    let overshot = runtime
        .submit(matrix("session-limit-edges"), GoalCommand::Status)
        .await
        .expect("overshoot status")
        .status
        .expect("projection");
    assert_eq!(overshot.known_token_usage, 11);
    assert_eq!(overshot.remaining_tokens, Some(0));
    assert_eq!(overshot.remaining_cost_usd, Some(0.0));
}

#[tokio::test]
async fn incomplete_accounting_states_preserve_known_totals_but_hide_remaining_budget() {
    for (suffix, accounting_state) in [
        (
            "missing",
            zeroclaw_runtime::control_plane::GoalAccountingState::Missing,
        ),
        (
            "invalid",
            zeroclaw_runtime::control_plane::GoalAccountingState::Invalid,
        ),
        (
            "outcome-unknown",
            zeroclaw_runtime::control_plane::GoalAccountingState::OutcomeUnknown,
        ),
    ] {
        let executor = Arc::new(FakeExecutor::new(
            GoalExecutionReadiness::Ready,
            LaunchAction::Succeed,
        ));
        let (runtime, store, tracker, _cost_dir) = controller(executor);
        let session = format!("session-{suffix}");
        runtime
            .submit(
                matrix(&session),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Limits {
                        token_limit: Some(20),
                        cost_limit_usd: Some(1.0),
                    },
                    success_criteria: "all required work is complete".into(),
                    objective: "retain lower-bound usage after accounting failure".into(),
                },
            )
            .await
            .expect("Goal starts");
        let current = store
            .current_goal_for_session(&session)
            .await
            .expect("current Goal lookup")
            .expect("current Goal");
        tracker
            .record_scoped_usage_with_owned_task_attribution(
                TokenUsage::new("test/model", 6, 4, 0, 0.0, 0.0, 0.0),
                Some("main"),
                Some(current.id.clone()),
            )
            .expect("known lower-bound usage");
        assert_eq!(
            store
                .admit_pending_call_with_reservation(
                    &current.id,
                    &session,
                    current.execution_epoch,
                    "pending-operation",
                    Some(20),
                    Some(1.0),
                )
                .await
                .expect("admit pending"),
            GoalTransitionResult::Applied
        );
        store
            .settle_pending_call(
                &current.id,
                &session,
                current.execution_epoch,
                "pending-operation",
                accounting_state,
            )
            .await
            .expect("settle accounting state");

        let projection = runtime
            .submit(matrix(&session), GoalCommand::Status)
            .await
            .expect("status")
            .status
            .expect("projection");
        assert_eq!(projection.known_token_usage, 10, "{suffix}");
        assert!(!projection.token_accounting_complete, "{suffix}");
        assert!(!projection.cost_accounting_complete, "{suffix}");
        assert_eq!(projection.remaining_tokens, None, "{suffix}");
        assert_eq!(projection.remaining_cost_usd, None, "{suffix}");
    }
}
