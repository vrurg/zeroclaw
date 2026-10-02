use std::sync::{
    Arc, Mutex, Once,
    atomic::{AtomicUsize, Ordering},
};

use async_trait::async_trait;
use tokio::{
    sync::Notify,
    time::{Duration, timeout},
};
use zeroclaw_api::{
    model_provider::{ChatMessage, TokenUsage},
    task::TaskStatus,
};
use zeroclaw_commands::goal::{GoalBudgetLimits, GoalBudgetSelection, GoalCommand};
use zeroclaw_config::{cost::CostTracker, schema::CostConfig};
use zeroclaw_goal::{
    GoalAccountingState, GoalExecutionNotice, GoalExecutionScope, GoalExecutionSupervisor,
    GoalHostSettings, GoalIngressContext, GoalIngressPrincipal, GoalOperationScope, GoalParentTurn,
    GoalParentTurnCheckpoint, GoalParentTurnKind, GoalParentTurnResult, GoalPauseReason,
    GoalResponse, GoalRuntime, GoalSessionBinding, GoalSessionDriver, GoalSessionExecutionLease,
    GoalSessionKey, GoalSessionLease, GoalTaskRegistry, GoalTransitionResult, GoalVerifierResult,
    GoalVerifierTurn,
    accounting::{
        GOAL_OPERATION_ACCOUNTING, GoalOperationRequest, GoalOperationSettlement, GoalUsageEvent,
    },
};
use zeroclaw_runtime::{
    control_plane::SqliteTaskStore,
    goal_accounting::{ModelProviderGoalUsagePricer, RuntimeGoalOperationAccountingFactory},
};

#[path = "support/goal_transition_fault.rs"]
mod goal_transition_fault;
use goal_transition_fault::{FenceAdmissionObservation, FinishFaultRegistry};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Window {
    Parent,
    Verifier,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Control {
    Pause,
    PauseAndResume,
    Restart,
    ExternalCancellation,
    Cancel,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Uncertainty {
    None,
    Pending,
    Missing,
}

struct FenceGate {
    ready: Notify,
    released: Notify,
    reached: AtomicUsize,
}

struct FenceDriver {
    key: GoalSessionKey,
    gate: Arc<FenceGate>,
    window: Window,
    plain_error: bool,
    uncertainty: Uncertainty,
    store: Arc<SqliteTaskStore>,
    events: Arc<Mutex<Vec<String>>>,
}

struct FenceLease {
    key: GoalSessionKey,
    gate: Arc<FenceGate>,
    window: Window,
    plain_error: bool,
    uncertainty: Uncertainty,
    store: Arc<SqliteTaskStore>,
    events: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl GoalSessionDriver for FenceDriver {
    fn session_key(&self) -> &GoalSessionKey {
        &self.key
    }

    async fn bind(&self, _: &GoalIngressContext) -> anyhow::Result<GoalSessionLease> {
        Ok(GoalSessionLease::new(
            GoalSessionBinding::new(self.key.clone()),
            (),
        ))
    }

    async fn acquire_execution(
        &self,
        _: &GoalIngressContext,
        _: &GoalExecutionScope,
        _: &GoalParentTurnCheckpoint,
    ) -> anyhow::Result<Box<dyn GoalSessionExecutionLease>> {
        Ok(Box::new(FenceLease {
            key: self.key.clone(),
            gate: self.gate.clone(),
            window: self.window,
            plain_error: self.plain_error,
            uncertainty: self.uncertainty,
            store: self.store.clone(),
            events: self.events.clone(),
        }))
    }
}

impl FenceLease {
    async fn settled_operation(&self) -> anyhow::Result<()> {
        let accountant = GOAL_OPERATION_ACCOUNTING.with(Clone::clone).unwrap();
        accountant
            .admit(GoalOperationRequest::new("fixture.route", "fixture-model"))
            .await?;
        accountant
            .settle(GoalOperationSettlement {
                accounting_state: GoalAccountingState::Complete,
                events: vec![GoalUsageEvent {
                    provider_ref: "fixture.route".into(),
                    model: "fixture-model".into(),
                    usage: TokenUsage {
                        input_tokens: Some(10),
                        output_tokens: Some(2),
                        cached_input_tokens: Some(0),
                        cache_creation_input_tokens: Some(0),
                    },
                    priceable: true,
                }],
            })
            .await
    }

    async fn fenced_admission(&self, scope: &GoalOperationScope) -> anyhow::Result<()> {
        // Inject uncertainty only in the named negative controls. Already-known
        // usage and real tool pairing remain unchanged in every scenario.
        if self.uncertainty != Uncertainty::None {
            let execution = scope.execution();
            assert_eq!(
                self.store
                    .admit_pending_operation(
                        execution.task_id(),
                        execution.session_id(),
                        execution.execution_epoch(),
                        "fixture-unsettled-operation",
                    )
                    .await?,
                GoalTransitionResult::Applied
            );
            if self.uncertainty == Uncertainty::Missing {
                assert_eq!(
                    self.store
                        .settle_pending_operation(
                            execution.task_id(),
                            execution.session_id(),
                            execution.execution_epoch(),
                            "fixture-unsettled-operation",
                            GoalAccountingState::Missing,
                        )
                        .await?,
                    GoalTransitionResult::Applied
                );
            }
        }
        self.gate.reached.fetch_add(1, Ordering::SeqCst);
        self.gate.ready.notify_one();
        self.gate.released.notified().await;
        let error = if self.plain_error {
            anyhow::anyhow!("fixture plain parent failure after pause")
        } else {
            GOAL_OPERATION_ACCOUNTING
                .with(Clone::clone)
                .unwrap()
                .admit(GoalOperationRequest::new("fixture.route", "fixture-model"))
                .await
                .expect_err("the exact epoch was fenced before admission")
        };
        self.events
            .lock()
            .unwrap()
            .push(format!("host_error:{error}"));
        Err(error)
    }
}

#[async_trait]
impl GoalSessionExecutionLease for FenceLease {
    fn session_key(&self) -> &GoalSessionKey {
        &self.key
    }
    fn canonical_history(&self) -> anyhow::Result<Vec<ChatMessage>> {
        Ok(vec![])
    }

    async fn run_parent_turn(
        &mut self,
        scope: &GoalOperationScope,
        turn: GoalParentTurn,
    ) -> anyhow::Result<GoalParentTurnResult> {
        self.settled_operation().await?;
        let batch = zeroclaw_goal::tool_pairing::admit_goal_tool_batch(1)
            .await?
            .unwrap();
        batch.settle()?;
        if self.window == Window::Parent && turn.kind != GoalParentTurnKind::Resume {
            self.fenced_admission(scope).await?;
        }
        GoalParentTurnResult::from_completed_agent_turn(
            "bounded candidate".into(),
            vec![ChatMessage::assistant("bounded candidate")],
        )
    }

    async fn run_verifier(
        &mut self,
        scope: &GoalOperationScope,
        _: GoalVerifierTurn,
    ) -> anyhow::Result<GoalVerifierResult> {
        self.settled_operation().await?;
        if self.window == Window::Parent {
            assert_eq!(self.gate.reached.load(Ordering::SeqCst), 1);
            return Ok(GoalVerifierResult {
                response: r#"{"decision":"complete","reason":"done","blockers":[]}"#.into(),
                attribution: None,
            });
        }
        self.fenced_admission(scope).await?;
        unreachable!("the fenced verifier cannot produce a decision")
    }

    async fn append_parent_candidate(&mut self, _: String) -> anyhow::Result<()> {
        Ok(())
    }

    async fn publish_goal_core_error(&mut self, error: &anyhow::Error) -> anyhow::Result<()> {
        self.events
            .lock()
            .unwrap()
            .push(format!("core_error:{error:#}"));
        Ok(())
    }

    async fn publish_goal_notice(&mut self, notice: GoalExecutionNotice) -> anyhow::Result<()> {
        self.events
            .lock()
            .unwrap()
            .push(format!("notice:{notice:?}"));
        Ok(())
    }
}

async fn run_case(
    key: GoalSessionKey,
    window: Window,
    control: Control,
    plain_error: bool,
    uncertainty: Uncertainty,
) {
    static LOG_INIT: Once = Once::new();
    LOG_INIT.call_once(|| {
        zeroclaw_log::try_install_capture_subscriber();
        drop(zeroclaw_log::subscribe_or_install());
    });
    let mut log_events = zeroclaw_log::subscribe().unwrap();
    let ingress = GoalIngressContext::trusted(
        key.clone(),
        "fixture-agent",
        "fixture-origin",
        match &key {
            GoalSessionKey::Matrix { .. } => GoalIngressPrincipal::Matrix {
                raw_mxid: "@fixture:example.invalid".into(),
            },
            GoalSessionKey::ZeroCode { .. } => GoalIngressPrincipal::ZeroCode {
                tui_id: "fixture-owner".into(),
            },
        },
    )
    .unwrap();
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let observation = Arc::new(FenceAdmissionObservation::default());
    let registry = Arc::new(FinishFaultRegistry::observing_fence(
        store.clone(),
        observation.clone(),
    ));
    let runtime = GoalRuntime::new(registry.clone());
    let directory = tempfile::tempdir().unwrap();
    let tracker = Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap());
    let accounting = Arc::new(
        RuntimeGoalOperationAccountingFactory::new(
            registry.clone(),
            tracker.clone(),
            "fixture-agent",
            Arc::new(ModelProviderGoalUsagePricer::new(Arc::default())),
        )
        .unwrap(),
    );
    let supervisor = Arc::new(GoalExecutionSupervisor::new(Arc::new(
        runtime.execution_engine(accounting),
    )));
    let gate = Arc::new(FenceGate {
        ready: Notify::new(),
        released: Notify::new(),
        reached: AtomicUsize::new(0),
    });
    let events = Arc::new(Mutex::new(vec![]));
    let driver = Arc::new(FenceDriver {
        key: key.clone(),
        gate: gate.clone(),
        window,
        plain_error,
        uncertainty,
        store: store.clone(),
        events: events.clone(),
    });
    let settings = GoalHostSettings::new(
        true,
        GoalBudgetLimits {
            token_limit: Some(100),
            cost_limit_usd: None,
        },
        42,
        "fixture-boot",
    )
    .unwrap();
    let mut started = supervisor
        .submit(
            settings.clone(),
            ingress.clone(),
            driver.clone(),
            GoalCommand::Start {
                budget: GoalBudgetSelection::Defaults,
                objective: "finish bounded work".into(),
            },
        )
        .await
        .unwrap();
    let original = store
        .current_goal_for_session(&key.durable_id())
        .await
        .unwrap()
        .unwrap();
    zeroclaw_log::record!(
        INFO,
        zeroclaw_log::Event::new(module_path!(), zeroclaw_log::Action::Note)
            .with_attrs(serde_json::json!({"task_id":original.id,"fixture_diagnostic_probe":true})),
        "Fence contract diagnostic capture probe"
    );
    supervisor
        .launch_after_response(settings.clone(), &mut started)
        .await
        .unwrap();
    timeout(Duration::from_secs(5), gate.ready.notified())
        .await
        .unwrap();
    let control_result = match control {
        Control::Pause | Control::PauseAndResume | Control::Cancel => {
            let mut submission = supervisor
                .submit(
                    settings.clone(),
                    ingress.clone(),
                    driver.clone(),
                    if control == Control::Cancel {
                        GoalCommand::Cancel
                    } else {
                        GoalCommand::Pause { immediate: true }
                    },
                )
                .await
                .unwrap();
            let fenced = store
                .current_goal_for_session(&key.durable_id())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(fenced.execution_epoch, original.execution_epoch + 1);
            gate.released.notify_one();
            timeout(
                Duration::from_secs(5),
                supervisor.launch_after_response(settings.clone(), &mut submission),
            )
            .await
            .unwrap()
            .unwrap()
        }
        Control::Restart | Control::ExternalCancellation => {
            let controller = supervisor.clone();
            let session = key.durable_id();
            let task = tokio::spawn(async move {
                if control == Control::Restart {
                    controller.pause_for_restart().await
                } else {
                    controller
                        .pause_for_external_cancellation(&session)
                        .await
                        .map(|_| ())
                }
            });
            timeout(
                Duration::from_secs(5),
                observation.control_committed.notified(),
            )
            .await
            .unwrap();
            gate.released.notify_one();
            timeout(Duration::from_secs(5), task)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            None
        }
    };
    let task = store
        .current_goal_for_session(&key.durable_id())
        .await
        .unwrap()
        .unwrap();
    let goal = store.get_goal_task(&task.id).await.unwrap().unwrap();
    assert_eq!(gate.reached.load(Ordering::SeqCst), 1);
    assert_eq!(registry.fault_counts(), (0, 0, 0));
    assert_eq!(
        observation.operation_admissions.load(Ordering::SeqCst),
        if window == Window::Verifier { 2 } else { 1 }
    );
    assert_eq!(observation.tool_admissions.load(Ordering::SeqCst), 1);
    assert_eq!(observation.tool_settlements.load(Ordering::SeqCst), 1);
    assert!(goal.pending_tool_batch_id.is_none() && goal.pending_tool_epoch.is_none());
    assert!(goal.pending_call_id.is_none() && goal.pending_call_epoch.is_none());
    assert_eq!(
        tracker.get_usage_totals_for_task(&task.id).unwrap().0,
        if window == Window::Verifier { 24 } else { 12 }
    );
    let expected_status = if control == Control::Cancel {
        TaskStatus::Cancelled
    } else if plain_error || uncertainty != Uncertainty::None {
        TaskStatus::Failed
    } else {
        TaskStatus::Paused
    };
    let events = events.lock().unwrap().clone();
    let mut terminal_verifier_diagnostics = 0;
    let mut diagnostic_probes = 0;
    let mut verifier_operation_diagnostics = 0;
    loop {
        match log_events.try_recv() {
            Ok(event) => {
                if event["attributes"]["task_id"] == task.id {
                    if event["attributes"]["fixture_diagnostic_probe"] == true {
                        diagnostic_probes += 1;
                    }
                    if event["attributes"]["validation_outcome"] == "operation_failed" {
                        verifier_operation_diagnostics += 1;
                    }
                }
                if event["attributes"]["task_id"] == task.id
                    && event["attributes"]["controller_action"] == "terminal_failure"
                {
                    terminal_verifier_diagnostics += 1;
                }
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => break,
            Err(error) => panic!("canonical diagnostic capture lost evidence: {error}"),
        }
    }
    assert_eq!(
        diagnostic_probes, 1,
        "the canonical diagnostic observer must be live"
    );
    if window == Window::Verifier {
        eprintln!(
            "verifier operation diagnostics: {verifier_operation_diagnostics}; terminal: {terminal_verifier_diagnostics}"
        );
    }
    assert_eq!(
        task.status, expected_status,
        "events={events:?}; late={control_result:?}"
    );
    assert_eq!(
        goal.accounting_state,
        match uncertainty {
            Uncertainty::None => GoalAccountingState::Complete,
            Uncertainty::Pending => GoalAccountingState::OutcomeUnknown,
            Uncertainty::Missing => GoalAccountingState::Missing,
        }
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event.starts_with("host_error:"))
            .count(),
        1
    );
    if expected_status != TaskStatus::Failed {
        assert_eq!(events.len(), 1);
        assert_eq!(terminal_verifier_diagnostics, 0);
        assert_eq!(verifier_operation_diagnostics, 0);
        assert_eq!(task.execution_epoch, original.execution_epoch + 1);
        if expected_status == TaskStatus::Paused {
            assert_eq!(
                goal.pause_reason,
                Some(if control == Control::Restart {
                    GoalPauseReason::DaemonRestart
                } else {
                    GoalPauseReason::OperatorPaused
                })
            );
            if matches!(control, Control::Pause | Control::PauseAndResume) {
                assert!(matches!(control_result, Some(GoalResponse::Paused(_))));
            }
        }
    } else {
        assert_eq!(events.len(), 3);
        assert_eq!(
            events[1],
            "core_error:Goal failure lost its execution fence"
        );
        assert_eq!(
            store
                .terminal_reason_for_session_goal(&task.id, &key.durable_id())
                .await
                .unwrap()
                .as_deref(),
            Some(if uncertainty == Uncertainty::Pending {
                "accounting_outcome_unknown"
            } else {
                "executor_failed"
            })
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event.starts_with("notice:"))
                .count(),
            1
        );
    }
    if control == Control::PauseAndResume {
        let mut resumed = supervisor
            .submit(
                settings.clone(),
                ingress,
                driver,
                GoalCommand::Resume { response: None },
            )
            .await
            .unwrap();
        assert!(matches!(resumed.response(), GoalResponse::Resumed(_)));
        let running = store
            .current_goal_for_session(&key.durable_id())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(running.id, original.id);
        assert_eq!(running.execution_epoch, task.execution_epoch + 1);
        let scope = GoalExecutionScope::new(&running.id, key.durable_id(), running.execution_epoch)
            .unwrap();
        supervisor
            .launch_after_response(settings, &mut resumed)
            .await
            .unwrap();
        assert_eq!(
            timeout(Duration::from_secs(5), supervisor.drain(&scope))
                .await
                .unwrap()
                .unwrap(),
            zeroclaw_goal::GoalExecutionOutcome::Completed
        );
        assert_eq!(
            store
                .current_goal_for_session(&key.durable_id())
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Completed
        );
        assert_eq!(
            tracker.get_usage_totals_for_task(&original.id).unwrap().0,
            36
        );
        assert_eq!(observation.tool_admissions.load(Ordering::SeqCst), 2);
        assert_eq!(observation.tool_settlements.load(Ordering::SeqCst), 2);
        assert_eq!(gate.reached.load(Ordering::SeqCst), 1);
        assert_eq!(observation.operation_admissions.load(Ordering::SeqCst), 3);
    }
}

fn matrix_key() -> GoalSessionKey {
    GoalSessionKey::matrix("matrix_fixture_fenced_admission").unwrap()
}
fn zerocode_key() -> GoalSessionKey {
    GoalSessionKey::zero_code("fixture-session").unwrap()
}

macro_rules! fence_case {
    ($name:ident, $key:ident, $window:ident, $control:ident, $plain:expr, $uncertainty:ident) => {
        #[tokio::test]
        async fn $name() {
            run_case(
                $key(),
                Window::$window,
                Control::$control,
                $plain,
                Uncertainty::$uncertainty,
            )
            .await;
        }
    };
}
fence_case!(
    fenced_admission_matrix_answerless_resume,
    matrix_key,
    Parent,
    PauseAndResume,
    false,
    None
);
fence_case!(
    fenced_admission_zerocode_answerless_resume,
    zerocode_key,
    Parent,
    PauseAndResume,
    false,
    None
);
fence_case!(
    fenced_admission_matrix_pause,
    matrix_key,
    Parent,
    Pause,
    false,
    None
);
fence_case!(
    fenced_admission_zerocode_pause,
    zerocode_key,
    Parent,
    Pause,
    false,
    None
);
fence_case!(
    fenced_admission_matrix_restart,
    matrix_key,
    Parent,
    Restart,
    false,
    None
);
fence_case!(
    fenced_admission_zerocode_restart,
    zerocode_key,
    Parent,
    Restart,
    false,
    None
);
fence_case!(
    fenced_admission_matrix_external_cancel,
    matrix_key,
    Parent,
    ExternalCancellation,
    false,
    None
);
fence_case!(
    fenced_admission_zerocode_external_cancel,
    zerocode_key,
    Parent,
    ExternalCancellation,
    false,
    None
);
fence_case!(
    fenced_admission_matrix_verifier,
    matrix_key,
    Verifier,
    Pause,
    false,
    None
);
fence_case!(
    fenced_admission_zerocode_verifier,
    zerocode_key,
    Verifier,
    Pause,
    false,
    None
);
fence_case!(
    fenced_admission_matrix_cancel,
    matrix_key,
    Parent,
    Cancel,
    false,
    None
);
fence_case!(
    fenced_admission_zerocode_cancel,
    zerocode_key,
    Parent,
    Cancel,
    false,
    None
);
fence_case!(
    fenced_admission_matrix_plain_error,
    matrix_key,
    Parent,
    Pause,
    true,
    None
);
fence_case!(
    fenced_admission_zerocode_plain_error,
    zerocode_key,
    Parent,
    Pause,
    true,
    None
);
fence_case!(
    fenced_admission_matrix_pending,
    matrix_key,
    Parent,
    Pause,
    false,
    Pending
);
fence_case!(
    fenced_admission_zerocode_pending,
    zerocode_key,
    Parent,
    Pause,
    false,
    Pending
);
fence_case!(
    fenced_admission_matrix_missing,
    matrix_key,
    Parent,
    Pause,
    false,
    Missing
);
fence_case!(
    fenced_admission_zerocode_missing,
    zerocode_key,
    Parent,
    Pause,
    false,
    Missing
);
