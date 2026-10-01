use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[path = "support/goal_transition_fault.rs"]
mod goal_transition_fault;
use goal_transition_fault::{
    FinishFaultRegistry, OriginalSettlementFailure, OriginalTransitionFailure,
};

use async_trait::async_trait;
use tokio::{
    sync::Notify,
    time::{Duration, timeout},
};
use zeroclaw_api::{model_provider::ChatMessage, task::TaskStatus};
use zeroclaw_commands::goal::{GoalBudgetLimits, GoalBudgetSelection, GoalCommand};
use zeroclaw_config::{cost::CostTracker, schema::CostConfig};
use zeroclaw_goal::{
    GoalExecutionNotice, GoalExecutionScope, GoalExecutionSupervisor, GoalHostSettings,
    GoalIngressContext, GoalIngressPrincipal, GoalOperationScope, GoalParentTurn,
    GoalParentTurnCheckpoint, GoalParentTurnResult, GoalRuntime, GoalSessionBinding,
    GoalSessionDriver, GoalSessionExecutionLease, GoalSessionKey, GoalSessionLease,
    GoalTaskRegistry, GoalVerifierResult, GoalVerifierTurn,
};
use zeroclaw_runtime::{
    control_plane::SqliteTaskStore,
    goal_accounting::{ModelProviderGoalUsagePricer, RuntimeGoalOperationAccountingFactory},
};

#[derive(Clone, Copy)]
enum BatchExit {
    Abandoned,
    HeldOpen,
    Paired,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExecutionPath {
    Supervised,
    PublicRun,
}

#[derive(Debug)]
struct OriginalParentFailure;

impl std::fmt::Display for OriginalParentFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("distinctive original parent failure")
    }
}

impl std::error::Error for OriginalParentFailure {}

struct PairingDriver {
    key: GoalSessionKey,
    parent_fails: bool,
    batch_exit: BatchExit,
    acquisitions: Arc<AtomicUsize>,
    expected_error_lease: usize,
    events: Arc<Mutex<Vec<&'static str>>>,
    notice: Arc<Notify>,
}

struct PairingLease {
    key: GoalSessionKey,
    parent_fails: bool,
    batch_exit: BatchExit,
    acquisition: usize,
    expected_error_lease: usize,
    open_batch: Option<zeroclaw_goal::tool_pairing::GoalToolBatch>,
    events: Arc<Mutex<Vec<&'static str>>>,
    notice: Arc<Notify>,
}

#[async_trait]
impl GoalSessionDriver for PairingDriver {
    fn session_key(&self) -> &GoalSessionKey {
        &self.key
    }

    async fn bind(&self, _ingress: &GoalIngressContext) -> anyhow::Result<GoalSessionLease> {
        Ok(GoalSessionLease::new(
            GoalSessionBinding::new(self.key.clone()),
            (),
        ))
    }

    async fn acquire_execution(
        &self,
        _ingress: &GoalIngressContext,
        _scope: &GoalExecutionScope,
        _checkpoint: &GoalParentTurnCheckpoint,
    ) -> anyhow::Result<Box<dyn GoalSessionExecutionLease>> {
        Ok(Box::new(PairingLease {
            key: self.key.clone(),
            parent_fails: self.parent_fails,
            batch_exit: self.batch_exit,
            acquisition: self.acquisitions.fetch_add(1, Ordering::SeqCst) + 1,
            expected_error_lease: self.expected_error_lease,
            open_batch: None,
            events: self.events.clone(),
            notice: self.notice.clone(),
        }))
    }
}

#[async_trait]
impl GoalSessionExecutionLease for PairingLease {
    fn session_key(&self) -> &GoalSessionKey {
        &self.key
    }

    fn canonical_history(&self) -> anyhow::Result<Vec<ChatMessage>> {
        Ok(vec![])
    }

    async fn run_parent_turn(
        &mut self,
        _scope: &GoalOperationScope,
        _turn: GoalParentTurn,
    ) -> anyhow::Result<GoalParentTurnResult> {
        let batch = zeroclaw_goal::tool_pairing::admit_goal_tool_batch(1)
            .await?
            .expect("parent runs under the real tool-pairing scope");
        match self.batch_exit {
            BatchExit::HeldOpen => self.open_batch = Some(batch),
            BatchExit::Abandoned => drop(batch),
            BatchExit::Paired => batch.settle()?,
        }
        if self.parent_fails {
            self.events.lock().unwrap().push("host_error");
            return Err(OriginalParentFailure.into());
        }
        GoalParentTurnResult::from_completed_agent_turn(
            "bounded candidate".into(),
            vec![ChatMessage::assistant("bounded candidate")],
        )
    }

    async fn run_verifier(
        &mut self,
        _scope: &GoalOperationScope,
        _turn: GoalVerifierTurn,
    ) -> anyhow::Result<GoalVerifierResult> {
        panic!("pairing failure must prevent verifier admission")
    }

    async fn append_parent_candidate(&mut self, _candidate: String) -> anyhow::Result<()> {
        panic!("pairing failure must prevent candidate publication")
    }

    async fn publish_goal_core_error(&mut self, error: &anyhow::Error) -> anyhow::Result<()> {
        assert!(error.downcast_ref::<OriginalSettlementFailure>().is_some());
        assert!(error.downcast_ref::<OriginalTransitionFailure>().is_none());
        assert_eq!(self.acquisition, self.expected_error_lease);
        assert_eq!(
            error.to_string(),
            "settle Goal tool batch after parent-turn pairing"
        );
        self.events.lock().unwrap().push("original_error");
        Ok(())
    }

    async fn publish_goal_notice(&mut self, notice: GoalExecutionNotice) -> anyhow::Result<()> {
        assert!(matches!(notice, GoalExecutionNotice::Failed(_)));
        self.events.lock().unwrap().push("failure_notice");
        self.notice.notify_one();
        Ok(())
    }
}

async fn assert_invariant_pairing_failure(
    key: GoalSessionKey,
    parent_fails: bool,
    retain_open_batch: bool,
) {
    assert_pairing_failure(
        key,
        parent_fails,
        if retain_open_batch {
            BatchExit::HeldOpen
        } else {
            BatchExit::Abandoned
        },
        None,
    )
    .await;
}

async fn assert_pairing_failure(
    key: GoalSessionKey,
    parent_fails: bool,
    batch_exit: BatchExit,
    fail_transition: Option<bool>,
) {
    assert_pairing_failure_path(
        key,
        parent_fails,
        batch_exit,
        fail_transition,
        ExecutionPath::Supervised,
    )
    .await;
}

async fn assert_pairing_failure_path(
    key: GoalSessionKey,
    parent_fails: bool,
    batch_exit: BatchExit,
    fail_transition: Option<bool>,
    path: ExecutionPath,
) {
    let ingress = match &key {
        GoalSessionKey::Matrix { .. } => GoalIngressContext::trusted(
            key.clone(),
            "fixture-agent",
            "matrix:fixture",
            GoalIngressPrincipal::Matrix {
                raw_mxid: "@fixture:example.invalid".into(),
            },
        ),
        GoalSessionKey::ZeroCode { .. } => GoalIngressContext::trusted(
            key.clone(),
            "fixture-agent",
            "zerocode:fixture",
            GoalIngressPrincipal::ZeroCode {
                tui_id: "fixture-owner".into(),
            },
        ),
    }
    .unwrap();
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let fault = fail_transition.map(|fail_transition| {
        Arc::new(FinishFaultRegistry::new(
            store.clone(),
            "fixture_never_finish_fault",
            None,
            Some(fail_transition),
        ))
    });
    let registry: Arc<dyn GoalTaskRegistry> = match &fault {
        Some(fault) => fault.clone(),
        None => store.clone(),
    };
    let runtime = GoalRuntime::new(registry.clone());
    let directory = tempfile::tempdir().unwrap();
    let tracker = Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap());
    let accounting = Arc::new(
        RuntimeGoalOperationAccountingFactory::new(
            registry,
            tracker,
            "fixture-agent",
            Arc::new(ModelProviderGoalUsagePricer::new(Arc::default())),
        )
        .unwrap(),
    );
    let engine = Arc::new(runtime.execution_engine(accounting));
    let events = Arc::new(Mutex::new(vec![]));
    let notice = Arc::new(Notify::new());
    let acquisitions = Arc::new(AtomicUsize::new(0));
    let expected_error_lease = if path == ExecutionPath::Supervised && fail_transition == Some(true)
    {
        2
    } else {
        1
    };
    let driver = Arc::new(PairingDriver {
        key: key.clone(),
        parent_fails,
        batch_exit,
        acquisitions: acquisitions.clone(),
        expected_error_lease,
        events: events.clone(),
        notice: notice.clone(),
    });
    let settings = GoalHostSettings::new(
        true,
        GoalBudgetLimits {
            token_limit: Some(100),
            cost_limit_usd: Some(1.0),
        },
        42,
        "fixture-boot",
    )
    .unwrap();
    let command = GoalCommand::Start {
        budget: GoalBudgetSelection::Defaults,
        objective: "finish bounded work".into(),
    };
    let completion = match path {
        ExecutionPath::PublicRun => {
            let request = runtime
                .submit(&settings, ingress, driver, command)
                .await
                .unwrap()
                .into_parts()
                .1
                .expect("start yields an admitted request");
            let error = engine
                .run(&settings, request, GoalParentTurnCheckpoint::new())
                .await
                .unwrap_err();
            // The public path must retain its existing causal chain, not just
            // produce the same flattened completion string as a supervisor.
            assert_eq!(
                error.downcast_ref::<OriginalParentFailure>().is_some(),
                parent_fails
            );
            assert_eq!(
                error.downcast_ref::<OriginalSettlementFailure>().is_some(),
                !parent_fails && fault.is_some()
            );
            assert!(error.downcast_ref::<OriginalTransitionFailure>().is_none());
            error
        }
        ExecutionPath::Supervised => {
            let supervisor = GoalExecutionSupervisor::new(engine);
            let mut started = supervisor
                .submit(settings.clone(), ingress, driver, command)
                .await
                .unwrap();
            let task = store
                .current_goal_for_session(&key.durable_id())
                .await
                .unwrap()
                .unwrap();
            let scope =
                GoalExecutionScope::new(&task.id, key.durable_id(), task.execution_epoch).unwrap();
            supervisor
                .launch_after_response(settings, &mut started)
                .await
                .unwrap();
            timeout(Duration::from_secs(5), notice.notified())
                .await
                .unwrap();
            timeout(Duration::from_secs(5), supervisor.drain(&scope))
                .await
                .unwrap()
                .unwrap_err()
        }
    };
    let pairing_error = match batch_exit {
        BatchExit::HeldOpen => "Goal parent turn returned with an unfinished tool batch",
        BatchExit::Abandoned => "Goal parent turn returned with an unpaired tool batch",
        BatchExit::Paired => {
            "settle Goal tool batch after parent-turn pairing: distinctive original Goal settlement failure"
        }
    };
    let expected = if fail_transition == Some(true) && parent_fails {
        format!(
            "Goal parent operation failed; tool pairing also failed ({pairing_error}); terminal transition also failed (distinctive original Goal transition failure): distinctive original parent failure"
        )
    } else if fail_transition == Some(true) {
        format!(
            "Goal tool pairing failed and terminal transition also failed: distinctive original Goal transition failure: {pairing_error}"
        )
    } else if parent_fails {
        format!(
            "Goal parent operation failed; tool pairing also failed ({pairing_error}): distinctive original parent failure"
        )
    } else {
        format!("Goal tool batch could not be paired: {pairing_error}")
    };
    assert_eq!(format!("{completion:#}"), expected);
    let task = store
        .current_goal_for_session(&key.durable_id())
        .await
        .unwrap()
        .unwrap();
    let transitioned = path == ExecutionPath::Supervised || fail_transition != Some(true);
    assert_eq!(
        task.status,
        if transitioned {
            TaskStatus::Failed
        } else {
            TaskStatus::Running
        }
    );
    assert_eq!(
        store
            .terminal_reason_for_session_goal(&task.id, &key.durable_id())
            .await
            .unwrap()
            .as_deref(),
        if transitioned {
            Some("goal_tool_pairing_incomplete")
        } else {
            None
        }
    );
    let goal = store.get_goal_task(&task.id).await.unwrap().unwrap();
    assert_eq!(goal.pending_tool_batch_id.is_none(), transitioned);
    assert_eq!(acquisitions.load(Ordering::SeqCst), expected_error_lease);
    if let Some(fault) = &fault {
        assert_eq!(fault.fault_counts().0, 0);
        assert_eq!(
            (fault.fault_counts().1, fault.fault_counts().2),
            (1, usize::from(fail_transition == Some(true)))
        );
    }
    let expected_events = if path == ExecutionPath::PublicRun {
        if parent_fails {
            vec!["host_error"]
        } else {
            vec![]
        }
    } else if fault.is_some() && parent_fails {
        vec!["host_error", "original_error", "failure_notice"]
    } else if fault.is_some() {
        vec!["original_error", "failure_notice"]
    } else if parent_fails {
        vec!["host_error", "failure_notice"]
    } else {
        vec!["failure_notice"]
    };
    assert_eq!(*events.lock().unwrap(), expected_events);
}

#[tokio::test]
async fn matrix_public_run_preserves_pairing_error_chain() {
    for parent_fails in [false, true] {
        for fail_transition in [false, true] {
            assert_pairing_failure_path(
                GoalSessionKey::matrix("matrix_pairing_fixture").unwrap(),
                parent_fails,
                BatchExit::Paired,
                Some(fail_transition),
                ExecutionPath::PublicRun,
            )
            .await;
        }
    }
}

#[tokio::test]
async fn zerocode_public_run_preserves_pairing_error_chain() {
    for parent_fails in [false, true] {
        for fail_transition in [false, true] {
            assert_pairing_failure_path(
                GoalSessionKey::zero_code("pairing-fixture").unwrap(),
                parent_fails,
                BatchExit::Paired,
                Some(fail_transition),
                ExecutionPath::PublicRun,
            )
            .await;
        }
    }
}

// Separate test functions ensure an expected visibility failure in one arm
// does not prevent the remaining double-fault arms from being observed.
macro_rules! settlement_case {
    ($name:ident, $key:expr, $parent_fails:expr, $transition_fails:expr) => {
        #[tokio::test]
        async fn $name() {
            assert_pairing_failure(
                $key,
                $parent_fails,
                BatchExit::Paired,
                Some($transition_fails),
            )
            .await;
        }
    };
}

settlement_case!(
    matrix_settlement_parent_ok,
    GoalSessionKey::matrix("matrix_pairing_fixture").unwrap(),
    false,
    false
);
settlement_case!(
    matrix_settlement_parent_error,
    GoalSessionKey::matrix("matrix_pairing_fixture").unwrap(),
    true,
    false
);
settlement_case!(
    matrix_settlement_parent_ok_transition_error,
    GoalSessionKey::matrix("matrix_pairing_fixture").unwrap(),
    false,
    true
);
settlement_case!(
    matrix_settlement_parent_error_transition_error,
    GoalSessionKey::matrix("matrix_pairing_fixture").unwrap(),
    true,
    true
);
settlement_case!(
    zerocode_settlement_parent_ok,
    GoalSessionKey::zero_code("pairing-fixture").unwrap(),
    false,
    false
);
settlement_case!(
    zerocode_settlement_parent_error,
    GoalSessionKey::zero_code("pairing-fixture").unwrap(),
    true,
    false
);
settlement_case!(
    zerocode_settlement_parent_ok_transition_error,
    GoalSessionKey::zero_code("pairing-fixture").unwrap(),
    false,
    true
);
settlement_case!(
    zerocode_settlement_parent_error_transition_error,
    GoalSessionKey::zero_code("pairing-fixture").unwrap(),
    true,
    true
);

#[tokio::test]
async fn matrix_pairing_invariant_is_a_verdict_not_an_original_error() {
    for parent_fails in [false, true] {
        for retain_open_batch in [false, true] {
            assert_invariant_pairing_failure(
                GoalSessionKey::matrix("matrix_pairing_fixture").unwrap(),
                parent_fails,
                retain_open_batch,
            )
            .await;
        }
    }
}

#[tokio::test]
async fn zerocode_pairing_invariant_is_a_verdict_not_an_original_error() {
    for parent_fails in [false, true] {
        for retain_open_batch in [false, true] {
            assert_invariant_pairing_failure(
                GoalSessionKey::zero_code("pairing-fixture").unwrap(),
                parent_fails,
                retain_open_batch,
            )
            .await;
        }
    }
}
