use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use async_trait::async_trait;
use zeroclaw_commands::goal::GoalCommand;
use zeroclaw_runtime::control_plane::{
    GoalAccountingState, GoalPauseState, GoalTaskRecord, GoalTaskRegistry, GoalTransitionResult,
    SqliteTaskStore, TaskContinuationContext, TaskRecord, TaskStatus,
};
use zeroclaw_runtime::goal_mode::{
    GoalController, GoalExecutionHost, GoalHostSettings, GoalIngressContext, GoalIngressPrincipal,
    GoalResponse, GoalSessionBinding, GoalSessionDriver, GoalSessionKey, GoalSessionLease,
    GoalSurface,
};

struct RecordingDriver {
    binding: GoalSessionBinding,
    binds: AtomicUsize,
}

#[async_trait]
impl GoalSessionDriver for RecordingDriver {
    fn surface(&self) -> GoalSurface {
        self.binding.surface()
    }

    fn session_key(&self) -> &GoalSessionKey {
        self.binding.session_key()
    }

    async fn bind(&self, _ingress: &GoalIngressContext) -> anyhow::Result<GoalSessionLease> {
        self.binds.fetch_add(1, Ordering::SeqCst);
        Ok(GoalSessionLease::new(self.binding.clone(), ()))
    }
}

struct LeaseDriver {
    binding: GoalSessionBinding,
    available_for_reconnect: Arc<AtomicUsize>,
}

impl LeaseDriver {
    fn reconnect_is_allowed(&self) -> bool {
        self.available_for_reconnect.load(Ordering::SeqCst) == 1
    }
}

struct ReconnectBlocker {
    available_for_reconnect: Arc<AtomicUsize>,
}

impl Drop for ReconnectBlocker {
    fn drop(&mut self) {
        self.available_for_reconnect.store(1, Ordering::SeqCst);
    }
}

#[async_trait]
impl GoalSessionDriver for LeaseDriver {
    fn surface(&self) -> GoalSurface {
        self.binding.surface()
    }

    fn session_key(&self) -> &GoalSessionKey {
        self.binding.session_key()
    }

    async fn bind(&self, _ingress: &GoalIngressContext) -> anyhow::Result<GoalSessionLease> {
        if self
            .available_for_reconnect
            .compare_exchange(1, 0, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            anyhow::bail!("session replacement is already in progress");
        }
        Ok(GoalSessionLease::new(
            self.binding.clone(),
            ReconnectBlocker {
                available_for_reconnect: self.available_for_reconnect.clone(),
            },
        ))
    }
}

struct LeaseObservingRegistry {
    task: TaskRecord,
    goal: GoalTaskRecord,
    reconnect_available: Arc<AtomicUsize>,
    pause_observed: AtomicUsize,
}

impl LeaseObservingRegistry {
    fn running_for(ingress: &GoalIngressContext, reconnect_available: Arc<AtomicUsize>) -> Self {
        let task_id = "goal-under-lease".to_owned();
        Self {
            task: TaskRecord {
                id: task_id.clone(),
                kind: zeroclaw_runtime::control_plane::TaskKind::Goal,
                agent: ingress.agent().to_owned(),
                status: TaskStatus::Running,
                owner_pid: 42,
                owner_boot_id: "test-boot".to_owned(),
                heartbeat_at: None,
                depth: 0,
                parent_id: None,
                originator_route: Some(ingress.route().to_owned()),
                delivered: false,
                idem_key: None,
                principal_id: Some("@alice:example.org".to_owned()),
                session_id: Some(ingress.session_key().durable_id()),
                execution_epoch: 1,
                started_at: "2026-09-02T00:00:00Z".to_owned(),
                finished_at: None,
            },
            goal: GoalTaskRecord {
                task_id,
                ..GoalTaskRecord::default()
            },
            reconnect_available,
            pause_observed: AtomicUsize::new(0),
        }
    }

    fn assert_lease_is_held(&self) {
        assert_eq!(
            self.reconnect_available.load(Ordering::SeqCst),
            0,
            "the session lease must remain held during the durable transition"
        );
    }
}

#[async_trait]
impl GoalTaskRegistry for LeaseObservingRegistry {
    async fn create_goal(
        &self,
        _task: TaskRecord,
        _goal: GoalTaskRecord,
        _continuation_context: Option<TaskContinuationContext>,
    ) -> anyhow::Result<()> {
        panic!("unexpected legacy create_goal call")
    }

    async fn latest_active_goal_for_agent(
        &self,
        _agent: &str,
    ) -> anyhow::Result<Option<TaskRecord>> {
        panic!("unexpected legacy agent lookup")
    }

    async fn latest_active_goal_for_context(
        &self,
        _agent: &str,
        _originator_route: Option<&str>,
        _principal_id: Option<&str>,
    ) -> anyhow::Result<Option<TaskRecord>> {
        panic!("unexpected legacy context lookup")
    }

    async fn latest_active_goal_id_for_context(
        &self,
        _agent: &str,
        _originator_route: Option<&str>,
        _principal_id: Option<&str>,
    ) -> anyhow::Result<Option<String>> {
        panic!("unexpected legacy context-id lookup")
    }

    async fn get_goal_task(&self, task_id: &str) -> anyhow::Result<Option<GoalTaskRecord>> {
        assert_eq!(task_id, self.task.id);
        Ok(Some(self.goal.clone()))
    }

    async fn update_goal_objective(&self, _task_id: &str, _objective: &str) -> anyhow::Result<()> {
        panic!("unexpected legacy objective update")
    }

    async fn update_goal_limits(
        &self,
        _task_id: &str,
        _token_limit: Option<u64>,
        _cost_limit_usd: Option<f64>,
    ) -> anyhow::Result<()> {
        panic!("unexpected legacy limits update")
    }

    async fn update_goal_pause(
        &self,
        _task_id: &str,
        _pause: Option<GoalPauseState>,
    ) -> anyhow::Result<()> {
        panic!("unexpected legacy pause update")
    }

    async fn pause_goal_task(&self, _task_id: &str, _pause: GoalPauseState) -> anyhow::Result<()> {
        panic!("unexpected legacy pause")
    }

    async fn resume_goal_task(
        &self,
        _task_id: &str,
        _owner_pid: u32,
        _owner_boot_id: &str,
        _continuation_context: Option<TaskContinuationContext>,
    ) -> anyhow::Result<()> {
        panic!("unexpected legacy resume")
    }

    async fn set_continuation_context(
        &self,
        _task_id: &str,
        _context: Option<TaskContinuationContext>,
    ) -> anyhow::Result<()> {
        panic!("unexpected continuation update")
    }

    async fn get_continuation_context(
        &self,
        _task_id: &str,
    ) -> anyhow::Result<Option<TaskContinuationContext>> {
        panic!("unexpected continuation lookup")
    }

    async fn current_goal_for_session(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<TaskRecord>> {
        assert_eq!(self.task.session_id.as_deref(), Some(session_id));
        Ok(Some(self.task.clone()))
    }

    async fn create_or_replace_session_goal(
        &self,
        _task: TaskRecord,
        _goal: GoalTaskRecord,
    ) -> anyhow::Result<GoalTransitionResult> {
        panic!("unexpected create-or-replace")
    }

    async fn pause_session_goal(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        _pause: GoalPauseState,
    ) -> anyhow::Result<GoalTransitionResult> {
        assert_eq!(task_id, self.task.id);
        assert_eq!(self.task.session_id.as_deref(), Some(session_id));
        assert_eq!(expected_epoch, self.task.execution_epoch);
        self.assert_lease_is_held();
        self.pause_observed.fetch_add(1, Ordering::SeqCst);
        Ok(GoalTransitionResult::Applied)
    }

    async fn resume_session_goal(
        &self,
        _task_id: &str,
        _session_id: &str,
        _expected_epoch: i64,
        _owner_pid: u32,
        _owner_boot_id: &str,
    ) -> anyhow::Result<GoalTransitionResult> {
        panic!("unexpected resume")
    }

    async fn finish_session_goal(
        &self,
        _task_id: &str,
        _session_id: &str,
        _expected_epoch: i64,
        _status: TaskStatus,
        _error: Option<String>,
    ) -> anyhow::Result<GoalTransitionResult> {
        panic!("unexpected finish")
    }

    async fn admit_pending_operation(
        &self,
        _task_id: &str,
        _session_id: &str,
        _expected_epoch: i64,
        _operation_id: &str,
    ) -> anyhow::Result<GoalTransitionResult> {
        panic!("unexpected operation admission")
    }

    async fn settle_pending_operation(
        &self,
        _task_id: &str,
        _session_id: &str,
        _admitted_epoch: i64,
        _operation_id: &str,
        _accounting_state: GoalAccountingState,
    ) -> anyhow::Result<GoalTransitionResult> {
        panic!("unexpected operation settlement")
    }

    async fn begin_goal_tool_phase(
        &self,
        _task_id: &str,
        _session_id: &str,
        _expected_epoch: i64,
    ) -> anyhow::Result<GoalTransitionResult> {
        panic!("unexpected tool-phase admission")
    }

    async fn complete_goal_tool_phase(
        &self,
        _task_id: &str,
        _session_id: &str,
        _admitted_epoch: i64,
    ) -> anyhow::Result<GoalTransitionResult> {
        panic!("unexpected tool-phase completion")
    }

    async fn update_session_goal_limits(
        &self,
        _task_id: &str,
        _session_id: &str,
        _expected_epoch: i64,
        _token_limit: Option<u64>,
        _cost_limit_usd: Option<f64>,
    ) -> anyhow::Result<GoalTransitionResult> {
        panic!("unexpected session limits update")
    }

    async fn delete_session_goal(
        &self,
        _task_id: &str,
        _session_id: &str,
        _expected_epoch: i64,
    ) -> anyhow::Result<GoalTransitionResult> {
        panic!("unexpected goal deletion")
    }
}

fn matrix_ingress() -> GoalIngressContext {
    GoalIngressContext::trusted(
        GoalSessionKey::matrix("matrix_room:!room:example.org:@alice:example.org").unwrap(),
        "main",
        "matrix:primary",
        GoalIngressPrincipal::Matrix {
            raw_mxid: "@alice:example.org".into(),
        },
    )
    .unwrap()
}

fn zerocode_ingress(agent: &str) -> GoalIngressContext {
    GoalIngressContext::trusted(
        GoalSessionKey::zero_code("same-live-session").unwrap(),
        agent,
        "zerocode:local",
        GoalIngressPrincipal::ZeroCode {
            tui_id: "current-tui-connection".into(),
        },
    )
    .unwrap()
}

fn recording_driver(ingress: &GoalIngressContext) -> Arc<RecordingDriver> {
    Arc::new(RecordingDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone(), "fresh-connection")
            .unwrap(),
        binds: AtomicUsize::new(0),
    })
}

#[tokio::test]
async fn supplied_driver_must_match_the_trusted_ingress_before_binding() {
    let ingress = matrix_ingress();
    let driver = Arc::new(RecordingDriver {
        binding: GoalSessionBinding::new(
            GoalSessionKey::zero_code("room:!room:example.org:@alice:example.org").unwrap(),
            "fresh-connection",
        )
        .unwrap(),
        binds: AtomicUsize::new(0),
    });

    let error = GoalExecutionHost::new()
        .submit(ingress, driver.clone(), GoalCommand::Status)
        .await
        .unwrap_err();

    assert!(error.to_string().contains("does not match"));
    assert_eq!(driver.binds.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn exact_driver_binding_and_typed_command_are_preserved() {
    let ingress = matrix_ingress();
    let driver = Arc::new(RecordingDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone(), "fresh-connection")
            .unwrap(),
        binds: AtomicUsize::new(0),
    });
    let command = GoalCommand::Pause;

    let submission = GoalExecutionHost::new()
        .submit(ingress.clone(), driver.clone(), command.clone())
        .await
        .unwrap();

    assert_eq!(submission.ingress(), &ingress);
    assert_eq!(submission.command(), &command);
    assert_eq!(submission.binding(), &driver.binding);
    assert_eq!(driver.binds.load(Ordering::SeqCst), 1);
}

#[test]
fn identical_raw_ids_from_matrix_and_zerocode_are_not_the_same_session() {
    let matrix = GoalSessionKey::matrix("matrix:shared-id").unwrap();
    let zerocode = GoalSessionKey::zero_code("shared-id").unwrap();

    assert_ne!(matrix, zerocode);
    assert_ne!(matrix.durable_id(), zerocode.durable_id());
}

#[tokio::test]
async fn controller_uses_only_a_host_validated_submission_for_lifecycle_transitions() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let controller = GoalController::new(store.clone() as Arc<dyn GoalTaskRegistry>);
    let settings = GoalHostSettings::new(
        true,
        zeroclaw_commands::goal::GoalBudgetLimits {
            token_limit: Some(100),
            cost_limit_usd: Some(1.0),
        },
        42,
        "test-boot",
    )
    .unwrap();
    let ingress = matrix_ingress();
    let driver = Arc::new(RecordingDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone(), "fresh-connection")
            .unwrap(),
        binds: AtomicUsize::new(0),
    });
    let host = GoalExecutionHost::new();

    let start = host
        .submit(
            ingress.clone(),
            driver.clone(),
            GoalCommand::Start {
                budget: zeroclaw_commands::goal::GoalBudgetSelection::Defaults,
                objective: "stop when the implementation is complete".into(),
            },
        )
        .await
        .unwrap();
    let GoalResponse::Started(started) = controller.submit(&settings, start).await.unwrap() else {
        panic!("expected a started Goal");
    };
    assert_eq!(started.execution_epoch, 1);

    let pause = host
        .submit(ingress.clone(), driver.clone(), GoalCommand::Pause)
        .await
        .unwrap();
    let GoalResponse::Paused(paused) = controller.submit(&settings, pause).await.unwrap() else {
        panic!("expected a paused Goal");
    };
    assert_eq!(paused.execution_epoch, 2);

    let resume = host
        .submit(ingress.clone(), driver.clone(), GoalCommand::Resume)
        .await
        .unwrap();
    let GoalResponse::Resumed(resumed) = controller.submit(&settings, resume).await.unwrap() else {
        panic!("expected a resumed Goal");
    };
    assert_eq!(resumed.execution_epoch, 3);

    let cancel = host
        .submit(ingress, driver, GoalCommand::Cancel)
        .await
        .unwrap();
    assert!(matches!(
        controller.submit(&settings, cancel).await.unwrap(),
        GoalResponse::Cancelled(_)
    ));
}

#[tokio::test]
async fn same_zerocode_session_retains_goal_control_after_agent_alias_refresh() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let controller = GoalController::new(store.clone() as Arc<dyn GoalTaskRegistry>);
    let settings = GoalHostSettings::new(
        true,
        zeroclaw_commands::goal::GoalBudgetLimits {
            token_limit: Some(100),
            cost_limit_usd: None,
        },
        42,
        "test-boot",
    )
    .unwrap();
    let host = GoalExecutionHost::new();
    let first_ingress = zerocode_ingress("alpha");
    let first_driver = recording_driver(&first_ingress);

    let start = host
        .submit(
            first_ingress,
            first_driver,
            GoalCommand::Start {
                budget: zeroclaw_commands::goal::GoalBudgetSelection::Defaults,
                objective: "finish the assigned task".into(),
            },
        )
        .await
        .unwrap();
    assert!(matches!(
        controller.submit(&settings, start).await.unwrap(),
        GoalResponse::Started(_)
    ));

    let refreshed_ingress = zerocode_ingress("beta");
    let refreshed_driver = recording_driver(&refreshed_ingress);
    let status = host
        .submit(
            refreshed_ingress.clone(),
            refreshed_driver.clone(),
            GoalCommand::Status,
        )
        .await
        .unwrap();
    assert!(matches!(
        controller.submit(&settings, status).await.unwrap(),
        GoalResponse::Status(_)
    ));

    let cancel = host
        .submit(
            refreshed_ingress.clone(),
            refreshed_driver.clone(),
            GoalCommand::Cancel,
        )
        .await
        .unwrap();
    assert!(matches!(
        controller.submit(&settings, cancel).await.unwrap(),
        GoalResponse::Cancelled(_)
    ));

    let replacement = host
        .submit(
            refreshed_ingress,
            refreshed_driver,
            GoalCommand::Start {
                budget: zeroclaw_commands::goal::GoalBudgetSelection::Defaults,
                objective: "finish the reassigned task".into(),
            },
        )
        .await
        .unwrap();
    assert!(matches!(
        controller.submit(&settings, replacement).await.unwrap(),
        GoalResponse::Started(_)
    ));
}

#[tokio::test]
async fn session_lease_blocks_reconnect_until_controller_returns() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let controller = GoalController::new(store as Arc<dyn GoalTaskRegistry>);
    let settings = GoalHostSettings::new(
        false,
        zeroclaw_commands::goal::GoalBudgetLimits {
            token_limit: None,
            cost_limit_usd: None,
        },
        42,
        "test-boot",
    )
    .unwrap();
    let ingress = matrix_ingress();
    let driver = Arc::new(LeaseDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone(), "fresh-connection")
            .unwrap(),
        available_for_reconnect: Arc::new(AtomicUsize::new(1)),
    });

    let submission = GoalExecutionHost::new()
        .submit(ingress, driver.clone(), GoalCommand::Status)
        .await
        .unwrap();

    assert!(
        !driver.reconnect_is_allowed(),
        "the host must retain the driver's session lease"
    );
    assert!(matches!(
        controller.submit(&settings, submission).await.unwrap(),
        GoalResponse::Disabled
    ));
    assert!(
        driver.reconnect_is_allowed(),
        "the lease may release only after the controller has finished"
    );
}

#[tokio::test]
async fn session_lease_stays_held_during_a_guarded_lifecycle_mutation() {
    let ingress = matrix_ingress();
    let reconnect_available = Arc::new(AtomicUsize::new(1));
    let registry = Arc::new(LeaseObservingRegistry::running_for(
        &ingress,
        reconnect_available.clone(),
    ));
    let controller = GoalController::new(registry.clone() as Arc<dyn GoalTaskRegistry>);
    let settings = GoalHostSettings::new(
        true,
        zeroclaw_commands::goal::GoalBudgetLimits {
            token_limit: None,
            cost_limit_usd: None,
        },
        42,
        "test-boot",
    )
    .unwrap();
    let driver = Arc::new(LeaseDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone(), "fresh-connection")
            .unwrap(),
        available_for_reconnect: reconnect_available,
    });

    let submission = GoalExecutionHost::new()
        .submit(ingress, driver.clone(), GoalCommand::Pause)
        .await
        .unwrap();

    let GoalResponse::Paused(_) = controller.submit(&settings, submission).await.unwrap() else {
        panic!("expected a paused Goal");
    };
    assert_eq!(registry.pause_observed.load(Ordering::SeqCst), 1);
    assert!(
        driver.reconnect_is_allowed(),
        "the lease may release after the guarded lifecycle mutation returns"
    );
}
