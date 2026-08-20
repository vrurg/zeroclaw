//! Contract coverage for the dormant, transport-neutral Goal controller.

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use zeroclaw_config::schema::Config;
use zeroclaw_runtime::{
    control_plane::{
        GoalPauseReason, GoalPauseState, GoalTaskRegistry, GoalTransitionResult, SqliteTaskStore,
        TaskStatus,
    },
    goal_mode::{
        GoalBudgetRequest, GoalCommand, GoalExecutionPreflight, GoalExecutionReadiness,
        GoalExecutionScope, GoalExecutor, GoalIngressContext, GoalMessageKey, GoalRuntime,
        parse_goal_command,
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
}

impl FakeExecutor {
    fn new(readiness: GoalExecutionReadiness, action: LaunchAction) -> Self {
        Self {
            readiness,
            action,
            launches: Mutex::new(Vec::new()),
            preflights: Mutex::new(Vec::new()),
        }
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
) -> (GoalRuntime, Arc<SqliteTaskStore>) {
    let store = Arc::new(SqliteTaskStore::new_in_memory().expect("test storage"));
    let goals: Arc<dyn GoalTaskRegistry> = store.clone();
    (
        GoalRuntime::new(config, goals, executor, "test-boot").expect("controller"),
        store,
    )
}

fn controller(executor: Arc<FakeExecutor>) -> (GoalRuntime, Arc<SqliteTaskStore>) {
    controller_with_config(enabled_config(), executor)
}

#[test]
fn start_requires_the_delimiter_and_preserves_the_objective_verbatim() {
    let command = parse_goal_command(
        "/goal start --tokens 250 --cost-usd 1.25 -- Ship the release -- safely",
    )
    .expect("the V1 start grammar should parse");

    assert_eq!(
        command,
        GoalCommand::Start {
            budget: GoalBudgetRequest::Limits {
                token_limit: Some(250),
                cost_limit_usd: Some(1.25),
            },
            objective: "Ship the release -- safely".into(),
        }
    );
}

#[test]
fn no_legacy_goal_syntax_or_extra_authority_input_is_accepted() {
    for input in [
        "/goal objective mutate this",
        "/goal start --tokens=42 -- objective",
        "/goal status task-id",
        "/goal pause because",
        "/goal start objective without delimiter",
    ] {
        assert!(parse_goal_command(input).is_err(), "{input}");
    }
}

#[test]
fn grammar_rejects_every_disallowed_budget_and_objective_form() {
    for input in [
        "/goal start --",
        "/goal start --tokens 0 -- objective",
        "/goal start --tokens 42 --tokens 43 -- objective",
        "/goal start --cost-usd NaN -- objective",
        "/goal start --cost-usd inf -- objective",
        "/goal start --unlimited --tokens 42 -- objective",
        "/goal start --unknown 42 -- objective",
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
            "/goal start --unlimited -- retained objective",
            GoalCommand::Start {
                budget: GoalBudgetRequest::Unlimited,
                objective: "retained objective".into(),
            },
        ),
        (
            "/goal start -- defaulted objective",
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                objective: "defaulted objective".into(),
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
    let (runtime, store) = controller(executor.clone());

    let started = runtime
        .submit(
            matrix("session-a"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                objective: "finish the migration".into(),
            },
        )
        .await
        .unwrap();
    assert_eq!(started.key, GoalMessageKey::Started);
    assert_eq!(executor.launches.lock().len(), 1);

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
    let (runtime, store) = controller(executor);

    assert_eq!(
        runtime
            .submit(
                matrix("session-a"),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Limits {
                        token_limit: Some(0),
                        cost_limit_usd: None,
                    },
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
}

#[tokio::test]
async fn typed_empty_budget_replacement_does_not_become_unlimited() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (runtime, store) = controller(executor);

    assert_eq!(
        runtime
            .submit(
                matrix("session-a"),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
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
    let runtime = GoalRuntime::new(enabled_config(), goals, executor, "test-boot").unwrap();

    let response = runtime
        .submit(
            matrix("session-a"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
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
async fn disabled_mode_and_untrusted_session_rebinding_fail_closed() {
    let executor = Arc::new(FakeExecutor::new(
        GoalExecutionReadiness::Ready,
        LaunchAction::Succeed,
    ));
    let (disabled_runtime, _) = controller_with_config(Arc::new(Config::default()), executor);
    assert_eq!(
        disabled_runtime
            .submit(
                matrix("session-a"),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Unlimited,
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
    let (runtime, _) = controller(executor);
    runtime
        .submit(
            matrix("session-a"),
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
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
    let (runtime, store) = controller(executor);
    assert_eq!(
        runtime
            .submit(
                matrix("session-a"),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Limits {
                        token_limit: Some(10),
                        cost_limit_usd: None,
                    },
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

    let (runtime, store) = controller(Arc::new(FakeExecutor::new(
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
