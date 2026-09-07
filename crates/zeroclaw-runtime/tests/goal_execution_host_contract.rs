use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use async_trait::async_trait;
use tempfile::TempDir;
use tokio::{
    sync::Notify,
    time::{Duration, timeout},
};
use zeroclaw_commands::goal::GoalCommand;
use zeroclaw_config::cost::CostTracker;
use zeroclaw_runtime::control_plane::{
    GoalAccountingState, GoalPauseReason, GoalPauseState, GoalTaskRecord, GoalTaskRegistry,
    GoalTransitionResult, SqliteTaskStore, TaskContinuationContext, TaskRecord, TaskRegistry,
    TaskStatus,
};
use zeroclaw_runtime::goal_mode::{
    GoalController, GoalExecutionHost, GoalExecutionRestartCoordinator, GoalExecutionScope,
    GoalExecutionSupervisor, GoalHostSettings, GoalIngressContext, GoalIngressPrincipal,
    GoalOperationScope, GoalParentTurn, GoalParentTurnResult, GoalResponse, GoalRuntime,
    GoalSessionBinding, GoalSessionDriver, GoalSessionExecutionLease, GoalSessionKey,
    GoalSessionLease, GoalSurface, GoalVerifierTurn,
};

struct RecordingDriver {
    binding: GoalSessionBinding,
    binds: AtomicUsize,
    execution_acquires: AtomicUsize,
}

struct RecordingExecutionLease {
    delivered: Arc<AtomicUsize>,
    complete_verifier: bool,
    continue_once: bool,
    parent_histories: Arc<Mutex<Vec<Vec<zeroclaw_api::model_provider::ChatMessage>>>>,
}

#[async_trait]
impl GoalSessionExecutionLease for RecordingExecutionLease {
    fn canonical_history(&self) -> anyhow::Result<Vec<zeroclaw_api::model_provider::ChatMessage>> {
        Ok(Vec::new())
    }

    async fn run_parent_turn(
        &mut self,
        _operation: &GoalOperationScope,
        turn: GoalParentTurn,
    ) -> anyhow::Result<GoalParentTurnResult> {
        self.parent_histories
            .lock()
            .unwrap()
            .push(turn.working_history.clone());
        Ok(GoalParentTurnResult {
            candidate: format!("parent:{}", turn.objective),
            working_history: turn.working_history,
        })
    }

    async fn run_verifier(
        &mut self,
        _operation: &GoalOperationScope,
        turn: GoalVerifierTurn,
    ) -> anyhow::Result<String> {
        if self.continue_once {
            self.continue_once = false;
            return Ok(r#"{"decision":"continue","reason":"add the missing detail"}"#.to_owned());
        }
        if self.complete_verifier {
            Ok(
                r#"{"decision":"complete","reason":"candidate satisfies the objective"}"#
                    .to_owned(),
            )
        } else {
            Ok(format!("verifier:{}", turn.candidate))
        }
    }

    async fn append_verified_candidate(&mut self, _candidate: String) -> anyhow::Result<()> {
        self.delivered.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn publish_goal_notice(
        &mut self,
        _notice: zeroclaw_runtime::goal_mode::GoalExecutionNotice,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

struct ExecutionDriver {
    binding: GoalSessionBinding,
    execution_acquires: AtomicUsize,
    delivered: Arc<AtomicUsize>,
    complete_verifier: bool,
    continue_once: bool,
    parent_histories: Arc<Mutex<Vec<Vec<zeroclaw_api::model_provider::ChatMessage>>>>,
}

struct PausingExecutionLease {
    parent_started: Arc<Notify>,
    release_parent: Arc<Notify>,
    verifier_calls: Arc<AtomicUsize>,
}

#[async_trait]
impl GoalSessionExecutionLease for PausingExecutionLease {
    fn canonical_history(&self) -> anyhow::Result<Vec<zeroclaw_api::model_provider::ChatMessage>> {
        Ok(Vec::new())
    }

    async fn run_parent_turn(
        &mut self,
        _operation: &GoalOperationScope,
        _turn: GoalParentTurn,
    ) -> anyhow::Result<GoalParentTurnResult> {
        self.parent_started.notify_one();
        self.release_parent.notified().await;
        Ok(GoalParentTurnResult {
            candidate: "candidate that settled before pause".to_owned(),
            working_history: Vec::new(),
        })
    }

    async fn run_verifier(
        &mut self,
        _operation: &GoalOperationScope,
        _turn: GoalVerifierTurn,
    ) -> anyhow::Result<String> {
        self.verifier_calls.fetch_add(1, Ordering::SeqCst);
        Ok(r#"{"decision":"complete","reason":"done"}"#.to_owned())
    }

    async fn publish_goal_notice(
        &mut self,
        _notice: zeroclaw_runtime::goal_mode::GoalExecutionNotice,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    async fn append_verified_candidate(&mut self, _candidate: String) -> anyhow::Result<()> {
        anyhow::bail!("a paused Goal must not deliver a candidate")
    }
}

struct PausingExecutionDriver {
    binding: GoalSessionBinding,
    parent_started: Arc<Notify>,
    release_parent: Arc<Notify>,
    verifier_calls: Arc<AtomicUsize>,
}

struct FailingExecutionDriver {
    binding: GoalSessionBinding,
}

#[async_trait]
impl GoalSessionDriver for FailingExecutionDriver {
    fn surface(&self) -> GoalSurface {
        self.binding.surface()
    }

    fn session_key(&self) -> &GoalSessionKey {
        self.binding.session_key()
    }

    async fn bind(&self, _ingress: &GoalIngressContext) -> anyhow::Result<GoalSessionLease> {
        Ok(GoalSessionLease::new(self.binding.clone(), ()))
    }

    async fn acquire_execution(
        &self,
        _ingress: &GoalIngressContext,
        _scope: &GoalExecutionScope,
    ) -> anyhow::Result<Box<dyn GoalSessionExecutionLease>> {
        anyhow::bail!("simulated execution lease acquisition failure")
    }
}

#[async_trait]
impl GoalSessionDriver for PausingExecutionDriver {
    fn surface(&self) -> GoalSurface {
        self.binding.surface()
    }

    fn session_key(&self) -> &GoalSessionKey {
        self.binding.session_key()
    }

    async fn bind(&self, _ingress: &GoalIngressContext) -> anyhow::Result<GoalSessionLease> {
        Ok(GoalSessionLease::new(self.binding.clone(), ()))
    }

    async fn acquire_execution(
        &self,
        _ingress: &GoalIngressContext,
        _scope: &GoalExecutionScope,
    ) -> anyhow::Result<Box<dyn GoalSessionExecutionLease>> {
        Ok(Box::new(PausingExecutionLease {
            parent_started: Arc::clone(&self.parent_started),
            release_parent: Arc::clone(&self.release_parent),
            verifier_calls: Arc::clone(&self.verifier_calls),
        }))
    }
}

#[async_trait]
impl GoalSessionDriver for ExecutionDriver {
    fn surface(&self) -> GoalSurface {
        self.binding.surface()
    }

    fn session_key(&self) -> &GoalSessionKey {
        self.binding.session_key()
    }

    async fn bind(&self, _ingress: &GoalIngressContext) -> anyhow::Result<GoalSessionLease> {
        Ok(GoalSessionLease::new(self.binding.clone(), ()))
    }

    async fn acquire_execution(
        &self,
        _ingress: &GoalIngressContext,
        _scope: &GoalExecutionScope,
    ) -> anyhow::Result<Box<dyn GoalSessionExecutionLease>> {
        self.execution_acquires.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(RecordingExecutionLease {
            delivered: self.delivered.clone(),
            complete_verifier: self.complete_verifier,
            continue_once: self.continue_once,
            parent_histories: Arc::clone(&self.parent_histories),
        }))
    }
}

struct WrongBindingDriver {
    advertised: GoalSessionKey,
    returned_binding: GoalSessionBinding,
    binds: AtomicUsize,
}

struct SurfaceMismatchedDriver {
    session_key: GoalSessionKey,
    binds: AtomicUsize,
    execution_acquires: AtomicUsize,
}

#[async_trait]
impl GoalSessionDriver for SurfaceMismatchedDriver {
    fn surface(&self) -> GoalSurface {
        GoalSurface::ZeroCode
    }

    fn session_key(&self) -> &GoalSessionKey {
        &self.session_key
    }

    async fn bind(&self, _ingress: &GoalIngressContext) -> anyhow::Result<GoalSessionLease> {
        self.binds.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("surface mismatched driver must never bind")
    }

    async fn acquire_execution(
        &self,
        _ingress: &GoalIngressContext,
        _scope: &GoalExecutionScope,
    ) -> anyhow::Result<Box<dyn GoalSessionExecutionLease>> {
        self.execution_acquires.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("surface mismatched driver must never acquire")
    }
}

#[async_trait]
impl GoalSessionDriver for WrongBindingDriver {
    fn surface(&self) -> GoalSurface {
        self.advertised.surface()
    }

    fn session_key(&self) -> &GoalSessionKey {
        &self.advertised
    }

    async fn bind(&self, _ingress: &GoalIngressContext) -> anyhow::Result<GoalSessionLease> {
        self.binds.fetch_add(1, Ordering::SeqCst);
        Ok(GoalSessionLease::new(self.returned_binding.clone(), ()))
    }

    async fn acquire_execution(
        &self,
        _ingress: &GoalIngressContext,
        _scope: &GoalExecutionScope,
    ) -> anyhow::Result<Box<dyn GoalSessionExecutionLease>> {
        anyhow::bail!("wrong binding driver has no execution lease")
    }
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

    async fn acquire_execution(
        &self,
        _ingress: &GoalIngressContext,
        _scope: &zeroclaw_runtime::goal_mode::GoalExecutionScope,
    ) -> anyhow::Result<Box<dyn zeroclaw_runtime::goal_mode::GoalSessionExecutionLease>> {
        self.execution_acquires.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("recording driver has no execution lease")
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

    async fn acquire_execution(
        &self,
        _ingress: &GoalIngressContext,
        _scope: &zeroclaw_runtime::goal_mode::GoalExecutionScope,
    ) -> anyhow::Result<Box<dyn zeroclaw_runtime::goal_mode::GoalSessionExecutionLease>> {
        anyhow::bail!("lease driver has no execution lease")
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

    async fn terminal_reason_for_session_goal(
        &self,
        task_id: &str,
        session_id: &str,
    ) -> anyhow::Result<Option<String>> {
        assert_eq!(task_id, self.task.id);
        assert_eq!(self.task.session_id.as_deref(), Some(session_id));
        Ok(None)
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
    matrix_ingress_with_session("matrix_room__room_example_org__alice_example_org")
}

fn matrix_ingress_with_session(session: &str) -> GoalIngressContext {
    GoalIngressContext::trusted(
        GoalSessionKey::matrix(session).unwrap(),
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

fn host_settings(enabled: bool) -> GoalHostSettings {
    GoalHostSettings::new(
        enabled,
        zeroclaw_commands::goal::GoalBudgetLimits {
            token_limit: None,
            cost_limit_usd: None,
        },
        42,
        "test-boot",
    )
    .unwrap()
}

fn recording_driver(ingress: &GoalIngressContext) -> Arc<RecordingDriver> {
    Arc::new(RecordingDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        binds: AtomicUsize::new(0),
        execution_acquires: AtomicUsize::new(0),
    })
}

#[tokio::test]
async fn supplied_driver_must_match_the_trusted_ingress_before_binding() {
    let ingress = matrix_ingress();
    let driver = Arc::new(RecordingDriver {
        binding: GoalSessionBinding::new(GoalSessionKey::zero_code("wrong-session").unwrap()),
        binds: AtomicUsize::new(0),
        execution_acquires: AtomicUsize::new(0),
    });

    let error = GoalExecutionHost::new()
        .submit(ingress, driver.clone(), GoalCommand::Status)
        .await
        .unwrap_err();

    assert!(error.to_string().contains("does not match"));
    assert_eq!(driver.binds.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn supplied_driver_surface_mismatch_is_rejected_before_binding() {
    let ingress = matrix_ingress();
    let driver = Arc::new(SurfaceMismatchedDriver {
        session_key: ingress.session_key().clone(),
        binds: AtomicUsize::new(0),
        execution_acquires: AtomicUsize::new(0),
    });

    let error = GoalExecutionHost::new()
        .submit(ingress, driver.clone(), GoalCommand::Status)
        .await
        .unwrap_err();

    assert!(error.to_string().contains("trusted ingress"));
    assert_eq!(driver.binds.load(Ordering::SeqCst), 0);
    assert_eq!(driver.execution_acquires.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn execution_scope_mismatch_is_rejected_before_driver_acquisition() {
    let ingress = matrix_ingress();
    let driver = Arc::new(ExecutionDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        execution_acquires: AtomicUsize::new(0),
        delivered: Arc::new(AtomicUsize::new(0)),
        complete_verifier: false,
        continue_once: false,
        parent_histories: Arc::new(Mutex::new(Vec::new())),
    });
    let scope =
        zeroclaw_runtime::goal_mode::GoalExecutionScope::new("goal-1", "rpc_wrong-session", 1)
            .unwrap();
    let settings = host_settings(true);

    let error = match GoalExecutionHost::new()
        .acquire_execution(&settings, &ingress, driver.clone(), &scope)
        .await
    {
        Ok(_) => panic!("mismatched scope must not acquire an execution lease"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("does not match"));
    assert_eq!(driver.execution_acquires.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn execution_driver_key_mismatch_is_rejected_before_driver_acquisition() {
    let ingress = matrix_ingress();
    let other_key = GoalSessionKey::matrix("matrix_other_room").unwrap();
    let driver = Arc::new(RecordingDriver {
        binding: GoalSessionBinding::new(other_key),
        binds: AtomicUsize::new(0),
        execution_acquires: AtomicUsize::new(0),
    });
    let scope = GoalExecutionScope::new("goal-1", ingress.session_key().durable_id(), 1).unwrap();
    let settings = host_settings(true);

    let error = match GoalExecutionHost::new()
        .acquire_execution(&settings, &ingress, driver.clone(), &scope)
        .await
    {
        Ok(_) => panic!("mismatched driver key must not acquire an execution lease"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("session key"));
    assert_eq!(driver.execution_acquires.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn execution_driver_surface_mismatch_is_rejected_before_driver_acquisition() {
    let ingress = matrix_ingress();
    let driver = Arc::new(SurfaceMismatchedDriver {
        session_key: ingress.session_key().clone(),
        binds: AtomicUsize::new(0),
        execution_acquires: AtomicUsize::new(0),
    });
    let scope = GoalExecutionScope::new("goal-1", ingress.session_key().durable_id(), 1).unwrap();
    let settings = host_settings(true);

    let error = match GoalExecutionHost::new()
        .acquire_execution(&settings, &ingress, driver.clone(), &scope)
        .await
    {
        Ok(_) => panic!("mismatched driver surface must not acquire an execution lease"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("surface"));
    assert_eq!(driver.execution_acquires.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn driver_returning_a_different_binding_is_rejected_after_binding() {
    let ingress = matrix_ingress();
    let returned_key = GoalSessionKey::matrix("matrix_other_room").unwrap();
    let driver = Arc::new(WrongBindingDriver {
        advertised: ingress.session_key().clone(),
        returned_binding: GoalSessionBinding::new(returned_key),
        binds: AtomicUsize::new(0),
    });

    let error = GoalExecutionHost::new()
        .submit(ingress, driver.clone(), GoalCommand::Status)
        .await
        .unwrap_err();

    assert!(error.to_string().contains("binding does not match"));
    assert_eq!(driver.binds.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn matching_execution_scope_returns_a_working_session_lease() {
    let ingress = matrix_ingress();
    let delivered = Arc::new(AtomicUsize::new(0));
    let driver = Arc::new(ExecutionDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        execution_acquires: AtomicUsize::new(0),
        delivered: delivered.clone(),
        complete_verifier: false,
        continue_once: false,
        parent_histories: Arc::new(Mutex::new(Vec::new())),
    });
    let scope = GoalExecutionScope::new("goal-1", ingress.session_key().durable_id(), 1).unwrap();
    let operation = GoalOperationScope::new(scope.clone());
    let settings = host_settings(true);

    let mut lease = GoalExecutionHost::new()
        .acquire_execution(&settings, &ingress, driver.clone(), &scope)
        .await
        .unwrap();

    assert!(lease.canonical_history().unwrap().is_empty());
    assert_eq!(
        lease
            .run_parent_turn(
                &operation,
                GoalParentTurn {
                    objective: "finish the task".into(),
                    working_history: Vec::new(),
                }
            )
            .await
            .unwrap()
            .candidate,
        "parent:finish the task"
    );
    assert_eq!(
        lease
            .run_verifier(
                &operation,
                GoalVerifierTurn {
                    objective: "finish the task".into(),
                    candidate: "candidate".into(),
                }
            )
            .await
            .unwrap(),
        "verifier:candidate"
    );
    lease
        .append_verified_candidate("candidate".into())
        .await
        .unwrap();

    assert_eq!(driver.execution_acquires.load(Ordering::SeqCst), 1);
    assert_eq!(delivered.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn disabled_goal_mode_cannot_acquire_an_execution_lease() {
    let ingress = matrix_ingress();
    let driver = recording_driver(&ingress);
    let scope = GoalExecutionScope::new("goal-1", ingress.session_key().durable_id(), 1).unwrap();
    let settings = host_settings(false);

    let error = match GoalExecutionHost::new()
        .acquire_execution(&settings, &ingress, driver.clone(), &scope)
        .await
    {
        Ok(_) => panic!("disabled Goal Mode must not acquire an execution lease"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("disabled"));
    assert_eq!(driver.execution_acquires.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn exact_driver_binding_and_typed_command_are_preserved() {
    let ingress = matrix_ingress();
    let driver = Arc::new(RecordingDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        binds: AtomicUsize::new(0),
        execution_acquires: AtomicUsize::new(0),
    });
    let command = GoalCommand::Pause;

    let submission = GoalExecutionHost::new()
        .submit(ingress.clone(), driver.clone(), command.clone())
        .await
        .unwrap();

    assert_eq!(submission.ingress(), &ingress);
    assert_eq!(submission.command(), &command);
    assert_eq!(submission.binding(), &driver.binding);
    let retained: Arc<dyn GoalSessionDriver> = driver.clone();
    assert!(
        Arc::ptr_eq(submission.driver(), &retained),
        "the executor must receive the exact driver validated at admission"
    );
    assert_eq!(driver.binds.load(Ordering::SeqCst), 1);
}

#[test]
fn identical_raw_ids_from_matrix_and_zerocode_are_not_the_same_session() {
    let matrix = GoalSessionKey::matrix("matrix_shared-id").unwrap();
    let zerocode = GoalSessionKey::zero_code("shared-id").unwrap();

    assert_ne!(matrix, zerocode);
    assert_ne!(matrix.durable_id(), zerocode.durable_id());
}

#[test]
fn matrix_history_key_must_already_use_the_canonical_session_form() {
    let raw = "matrix_room:!room:example.org:@alice:example.org";

    assert!(GoalSessionKey::matrix(raw).is_err());
    assert!(GoalSessionKey::matrix(" matrix_room").is_err());
    assert!(GoalSessionKey::zero_code(" same-session").is_err());
}

#[test]
fn execution_scope_rejects_blank_ids_and_nonpositive_epochs() {
    assert!(GoalExecutionScope::new("", "matrix_session", 1).is_err());
    assert!(GoalExecutionScope::new("goal-1", "", 1).is_err());
    assert!(GoalExecutionScope::new("goal-1", "matrix_session", 0).is_err());
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
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        binds: AtomicUsize::new(0),
        execution_acquires: AtomicUsize::new(0),
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
    let (response, start) = controller
        .submit(&settings, start)
        .await
        .unwrap()
        .into_parts();
    let GoalResponse::Started(started) = response else {
        panic!("expected a started Goal");
    };
    assert_eq!(started.execution_epoch, 1);
    assert!(
        Arc::ptr_eq(
            start.driver(),
            &(driver.clone() as Arc<dyn GoalSessionDriver>)
        ),
        "the controller must not consume or replace the validated execution driver"
    );

    let pause = host
        .submit(ingress.clone(), driver.clone(), GoalCommand::Pause)
        .await
        .unwrap();
    let GoalResponse::Paused(paused) = controller
        .submit(&settings, pause)
        .await
        .unwrap()
        .into_parts()
        .0
    else {
        panic!("expected a paused Goal");
    };
    assert_eq!(paused.execution_epoch, 2);

    let resume = host
        .submit(ingress.clone(), driver.clone(), GoalCommand::Resume)
        .await
        .unwrap();
    let GoalResponse::Resumed(resumed) = controller
        .submit(&settings, resume)
        .await
        .unwrap()
        .into_parts()
        .0
    else {
        panic!("expected a resumed Goal");
    };
    assert_eq!(resumed.execution_epoch, 3);

    let cancel = host
        .submit(ingress, driver, GoalCommand::Cancel)
        .await
        .unwrap();
    assert!(matches!(
        controller
            .submit(&settings, cancel)
            .await
            .unwrap()
            .into_parts()
            .0,
        GoalResponse::Cancelled(_)
    ));
}

#[tokio::test]
async fn runtime_returns_exact_execution_handoff_only_for_a_running_goal() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let runtime = GoalRuntime::new(store as Arc<dyn GoalTaskRegistry>);
    let settings = host_settings(true);
    let ingress = matrix_ingress();
    let driver = Arc::new(ExecutionDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        execution_acquires: AtomicUsize::new(0),
        delivered: Arc::new(AtomicUsize::new(0)),
        complete_verifier: false,
        continue_once: false,
        parent_histories: Arc::new(Mutex::new(Vec::new())),
    });

    let submission = runtime
        .submit(
            &settings,
            ingress.clone(),
            driver.clone(),
            GoalCommand::Start {
                budget: zeroclaw_commands::goal::GoalBudgetSelection::Defaults,
                objective: "finish the task".into(),
            },
        )
        .await
        .unwrap();
    let (response, execution) = submission.into_parts();
    let GoalResponse::Started(started) = response else {
        panic!("expected a started Goal");
    };
    let execution = execution.expect("a started Goal must launch exactly one execution request");
    assert_eq!(execution.scope().task_id(), started.task_id);
    assert_eq!(
        execution.scope().session_id(),
        ingress.session_key().durable_id()
    );
    assert_eq!(execution.scope().execution_epoch(), started.execution_epoch);
    let lease = runtime
        .acquire_execution(&settings, &execution)
        .await
        .unwrap();
    assert!(lease.canonical_history().unwrap().is_empty());
    assert_eq!(driver.execution_acquires.load(Ordering::SeqCst), 1);
    assert!(Arc::ptr_eq(
        execution.driver(),
        &(driver as Arc<dyn GoalSessionDriver>)
    ));
}

#[tokio::test]
async fn execution_engine_completes_through_the_exact_runtime_handoff() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let runtime = GoalRuntime::new(store.clone() as Arc<dyn GoalTaskRegistry>);
    let settings = host_settings(true);
    let ingress = matrix_ingress();
    let delivered = Arc::new(AtomicUsize::new(0));
    let parent_histories = Arc::new(Mutex::new(Vec::new()));
    let driver = Arc::new(ExecutionDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        execution_acquires: AtomicUsize::new(0),
        delivered: delivered.clone(),
        complete_verifier: true,
        continue_once: true,
        parent_histories: Arc::clone(&parent_histories),
    });
    let request = runtime
        .submit(
            &settings,
            ingress,
            driver,
            GoalCommand::Start {
                budget: zeroclaw_commands::goal::GoalBudgetSelection::Defaults,
                objective: "finish the task".into(),
            },
        )
        .await
        .unwrap()
        .into_parts()
        .1
        .expect("start must yield an execution request");
    let directory = TempDir::new().unwrap();
    let tracker = Arc::new(
        CostTracker::new(
            zeroclaw_config::schema::CostConfig {
                enabled: false,
                ..Default::default()
            },
            directory.path(),
        )
        .unwrap(),
    );
    let engine = runtime
        .execution_engine(tracker, "main", Arc::default())
        .unwrap();

    assert_eq!(
        engine.run(&settings, &request).await.unwrap(),
        zeroclaw_runtime::goal_mode::GoalExecutionOutcome::Completed
    );
    assert_eq!(delivered.load(Ordering::SeqCst), 1);
    {
        let histories = parent_histories.lock().unwrap();
        assert_eq!(histories.len(), 2, "Continue must run a second parent turn");
        assert!(histories[1].iter().any(|message| {
            message.role == "assistant" && message.content == "parent:finish the task"
        }));
        assert!(
            histories[1]
                .iter()
                .any(|message| message.content.contains("add the missing detail"))
        );
    }
    assert_eq!(
        store
            .current_goal_for_session(request.scope().session_id())
            .await
            .unwrap()
            .unwrap()
            .status,
        TaskStatus::Completed
    );
}

#[tokio::test]
async fn execution_supervisor_retains_one_exact_epoch_until_it_drains() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let runtime = GoalRuntime::new(store.clone() as Arc<dyn GoalTaskRegistry>);
    let settings = host_settings(true);
    let ingress = matrix_ingress();
    let delivered = Arc::new(AtomicUsize::new(0));
    let driver = Arc::new(ExecutionDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        execution_acquires: AtomicUsize::new(0),
        delivered: delivered.clone(),
        complete_verifier: true,
        continue_once: false,
        parent_histories: Arc::new(Mutex::new(Vec::new())),
    });
    let request = runtime
        .submit(
            &settings,
            ingress,
            driver,
            GoalCommand::Start {
                budget: zeroclaw_commands::goal::GoalBudgetSelection::Defaults,
                objective: "finish the task".into(),
            },
        )
        .await
        .unwrap()
        .into_parts()
        .1
        .expect("start must yield an execution request");
    let directory = TempDir::new().unwrap();
    let tracker = Arc::new(
        CostTracker::new(
            zeroclaw_config::schema::CostConfig {
                enabled: false,
                ..Default::default()
            },
            directory.path(),
        )
        .unwrap(),
    );
    let supervisor = GoalExecutionSupervisor::new(Arc::new(
        runtime
            .execution_engine(tracker, "main", Arc::default())
            .unwrap(),
    ));
    let scope = request.scope().clone();

    supervisor.launch(settings.clone(), request).await.unwrap();
    assert!(
        supervisor.owns_scope(&scope).await,
        "an accepted epoch remains supervisor-owned until it is drained"
    );
    assert_eq!(
        supervisor.drain(&scope).await.unwrap(),
        zeroclaw_runtime::goal_mode::GoalExecutionOutcome::Completed
    );
    assert!(!supervisor.owns_scope(&scope).await);
    assert_eq!(delivered.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn execution_supervisor_terminalizes_an_unexpected_worker_failure() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let runtime = GoalRuntime::new(store.clone() as Arc<dyn GoalTaskRegistry>);
    let settings = host_settings(true);
    let ingress = matrix_ingress();
    let driver = Arc::new(FailingExecutionDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
    });
    let request = runtime
        .submit(
            &settings,
            ingress.clone(),
            driver,
            GoalCommand::Start {
                budget: zeroclaw_commands::goal::GoalBudgetSelection::Defaults,
                objective: "finish the task".into(),
            },
        )
        .await
        .unwrap()
        .into_parts()
        .1
        .expect("start must yield an execution request");
    let directory = TempDir::new().unwrap();
    let tracker = Arc::new(
        CostTracker::new(
            zeroclaw_config::schema::CostConfig {
                enabled: false,
                ..Default::default()
            },
            directory.path(),
        )
        .unwrap(),
    );
    let supervisor = GoalExecutionSupervisor::new(Arc::new(
        runtime
            .execution_engine(tracker, "main", Arc::default())
            .unwrap(),
    ));
    let scope = request.scope().clone();

    supervisor.launch(settings, request).await.unwrap();
    assert!(supervisor.drain(&scope).await.is_err());
    let current = store
        .current_goal_for_session(&ingress.session_key().durable_id())
        .await
        .unwrap()
        .expect("failed Goal remains the current terminal Goal");
    assert_eq!(current.status, TaskStatus::Failed);
    assert_eq!(
        store
            .get_snapshot(&current.id)
            .await
            .unwrap()
            .expect("failed Goal remains readable")
            .error
            .as_deref(),
        Some("executor_failed")
    );
}

#[tokio::test]
async fn supervisor_pause_waits_for_the_admitted_parent_to_settle() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let runtime = GoalRuntime::new(store.clone() as Arc<dyn GoalTaskRegistry>);
    let settings = host_settings(true);
    let ingress = matrix_ingress();
    let parent_started = Arc::new(Notify::new());
    let release_parent = Arc::new(Notify::new());
    let verifier_calls = Arc::new(AtomicUsize::new(0));
    let driver = Arc::new(PausingExecutionDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        parent_started: Arc::clone(&parent_started),
        release_parent: Arc::clone(&release_parent),
        verifier_calls: Arc::clone(&verifier_calls),
    });
    let directory = TempDir::new().unwrap();
    let tracker = Arc::new(
        CostTracker::new(
            zeroclaw_config::schema::CostConfig {
                enabled: false,
                ..Default::default()
            },
            directory.path(),
        )
        .unwrap(),
    );
    let supervisor = Arc::new(GoalExecutionSupervisor::new(Arc::new(
        runtime
            .execution_engine(tracker, "main", Arc::default())
            .unwrap(),
    )));

    assert!(matches!(
        supervisor
            .submit(
                settings.clone(),
                ingress.clone(),
                driver.clone(),
                GoalCommand::Start {
                    budget: zeroclaw_commands::goal::GoalBudgetSelection::Defaults,
                    objective: "finish the task".into(),
                },
            )
            .await
            .unwrap()
            .into_response(),
        GoalResponse::Started(_)
    ));
    parent_started.notified().await;

    let pause_supervisor = Arc::clone(&supervisor);
    let pause_settings = settings.clone();
    let pause_ingress = ingress.clone();
    let pause_driver = driver.clone();
    let mut pause = zeroclaw_spawn::spawn!(async move {
        pause_supervisor
            .submit(
                pause_settings,
                pause_ingress,
                pause_driver,
                GoalCommand::Pause,
            )
            .await
    });
    assert!(
        timeout(Duration::from_millis(20), &mut pause)
            .await
            .is_err(),
        "pause must drain the admitted operation instead of interrupting it"
    );

    release_parent.notify_one();
    assert!(matches!(
        pause.await.unwrap().unwrap().into_response(),
        GoalResponse::Paused(_)
    ));
    assert_eq!(verifier_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        store
            .current_goal_for_session(&ingress.session_key().durable_id())
            .await
            .unwrap()
            .expect("paused Goal remains current")
            .status,
        TaskStatus::Paused
    );
}

#[tokio::test]
async fn cancelled_drain_keeps_the_worker_owned_for_a_later_lifecycle_retry() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let runtime = GoalRuntime::new(store.clone() as Arc<dyn GoalTaskRegistry>);
    let settings = host_settings(true);
    let ingress = matrix_ingress();
    let parent_started = Arc::new(Notify::new());
    let release_parent = Arc::new(Notify::new());
    let driver = Arc::new(PausingExecutionDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        parent_started: Arc::clone(&parent_started),
        release_parent: Arc::clone(&release_parent),
        verifier_calls: Arc::new(AtomicUsize::new(0)),
    });
    let directory = TempDir::new().unwrap();
    let tracker = Arc::new(
        CostTracker::new(
            zeroclaw_config::schema::CostConfig {
                enabled: false,
                ..Default::default()
            },
            directory.path(),
        )
        .unwrap(),
    );
    let supervisor = Arc::new(GoalExecutionSupervisor::new(Arc::new(
        runtime
            .execution_engine(tracker, "main", Arc::default())
            .unwrap(),
    )));

    supervisor
        .submit(
            settings,
            ingress.clone(),
            driver,
            GoalCommand::Start {
                budget: zeroclaw_commands::goal::GoalBudgetSelection::Defaults,
                objective: "finish the task".into(),
            },
        )
        .await
        .unwrap();
    parent_started.notified().await;

    let current = store
        .current_goal_for_session(&ingress.session_key().durable_id())
        .await
        .unwrap()
        .expect("started Goal remains current while the parent turn is active");
    let scope = GoalExecutionScope::new(
        current.id,
        ingress.session_key().durable_id(),
        current.execution_epoch,
    )
    .unwrap();

    let interrupted_supervisor = Arc::clone(&supervisor);
    let interrupted_scope = scope.clone();
    let mut interrupted =
        zeroclaw_spawn::spawn!(
            async move { interrupted_supervisor.drain(&interrupted_scope).await }
        );
    assert!(
        timeout(Duration::from_millis(20), &mut interrupted)
            .await
            .is_err(),
        "the first drain must wait for the admitted operation"
    );
    interrupted.abort();
    let _ = interrupted.await;

    assert!(
        supervisor.owns_scope(&scope).await,
        "cancelling a lifecycle waiter must not drop the worker handle"
    );

    let retry_supervisor = Arc::clone(&supervisor);
    let retry_scope = scope.clone();
    let mut retry =
        zeroclaw_spawn::spawn!(async move { retry_supervisor.drain(&retry_scope).await });
    assert!(
        timeout(Duration::from_millis(20), &mut retry)
            .await
            .is_err(),
        "the retained worker must still wait for its admitted operation"
    );
    release_parent.notify_one();
    assert!(
        retry.await.unwrap().is_err(),
        "the test driver refuses delivery after the worker has settled"
    );
    assert!(
        !supervisor.owns_scope(&scope).await,
        "only the successful drain consumes the retained worker handle"
    );
}

#[tokio::test]
async fn concurrent_drains_consume_one_worker_without_repolling_its_join_handle() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let runtime = GoalRuntime::new(store.clone() as Arc<dyn GoalTaskRegistry>);
    let settings = host_settings(true);
    let ingress = matrix_ingress();
    let parent_started = Arc::new(Notify::new());
    let release_parent = Arc::new(Notify::new());
    let driver = Arc::new(PausingExecutionDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        parent_started: Arc::clone(&parent_started),
        release_parent: Arc::clone(&release_parent),
        verifier_calls: Arc::new(AtomicUsize::new(0)),
    });
    let directory = TempDir::new().unwrap();
    let tracker = Arc::new(
        CostTracker::new(
            zeroclaw_config::schema::CostConfig {
                enabled: false,
                ..Default::default()
            },
            directory.path(),
        )
        .unwrap(),
    );
    let supervisor = Arc::new(GoalExecutionSupervisor::new(Arc::new(
        runtime
            .execution_engine(tracker, "main", Arc::default())
            .unwrap(),
    )));

    supervisor
        .submit(
            settings,
            ingress.clone(),
            driver,
            GoalCommand::Start {
                budget: zeroclaw_commands::goal::GoalBudgetSelection::Defaults,
                objective: "finish the task".into(),
            },
        )
        .await
        .unwrap();
    parent_started.notified().await;

    let current = store
        .current_goal_for_session(&ingress.session_key().durable_id())
        .await
        .unwrap()
        .unwrap();
    let scope = GoalExecutionScope::new(
        current.id,
        ingress.session_key().durable_id(),
        current.execution_epoch,
    )
    .unwrap();

    let first_supervisor = Arc::clone(&supervisor);
    let first_scope = scope.clone();
    let first = zeroclaw_spawn::spawn!(async move { first_supervisor.drain(&first_scope).await });
    let second_supervisor = Arc::clone(&supervisor);
    let second_scope = scope.clone();
    let second =
        zeroclaw_spawn::spawn!(async move { second_supervisor.drain(&second_scope).await });

    release_parent.notify_one();
    assert!(first.await.unwrap().is_err());
    assert!(second.await.unwrap().is_err());
    assert!(
        !supervisor.owns_scope(&scope).await,
        "both lifecycle waiters must observe the same completion without retaining the worker"
    );
}

#[tokio::test]
async fn supervisor_restart_pause_waits_for_the_admitted_parent_to_settle() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let runtime = GoalRuntime::new(store.clone() as Arc<dyn GoalTaskRegistry>);
    let settings = host_settings(true);
    let ingress = matrix_ingress();
    let parent_started = Arc::new(Notify::new());
    let release_parent = Arc::new(Notify::new());
    let verifier_calls = Arc::new(AtomicUsize::new(0));
    let driver = Arc::new(PausingExecutionDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        parent_started: Arc::clone(&parent_started),
        release_parent: Arc::clone(&release_parent),
        verifier_calls: Arc::clone(&verifier_calls),
    });
    let directory = TempDir::new().unwrap();
    let tracker = Arc::new(
        CostTracker::new(
            zeroclaw_config::schema::CostConfig {
                enabled: false,
                ..Default::default()
            },
            directory.path(),
        )
        .unwrap(),
    );
    let coordinator = GoalExecutionRestartCoordinator::new();
    let supervisor = coordinator
        .new_supervisor(Arc::new(
            runtime
                .execution_engine(tracker, "main", Arc::default())
                .unwrap(),
        ))
        .await;

    assert!(matches!(
        supervisor
            .submit(
                settings,
                ingress.clone(),
                driver,
                GoalCommand::Start {
                    budget: zeroclaw_commands::goal::GoalBudgetSelection::Defaults,
                    objective: "finish the task".into(),
                },
            )
            .await
            .unwrap()
            .into_response(),
        GoalResponse::Started(_)
    ));
    parent_started.notified().await;

    let restart_coordinator = Arc::new(coordinator);
    let mut restart = {
        let restart_coordinator = Arc::clone(&restart_coordinator);
        zeroclaw_spawn::spawn!(async move { restart_coordinator.quiesce_for_restart().await })
    };
    assert!(
        timeout(Duration::from_millis(20), &mut restart)
            .await
            .is_err(),
        "restart pause must let the admitted operation settle before it returns"
    );

    release_parent.notify_one();
    restart.await.unwrap().unwrap();
    assert_eq!(verifier_calls.load(Ordering::SeqCst), 0);

    let current = store
        .current_goal_for_session(&ingress.session_key().durable_id())
        .await
        .unwrap()
        .expect("restart-paused Goal remains current");
    let goal = store.get_goal_task(&current.id).await.unwrap().unwrap();
    assert_eq!(current.status, TaskStatus::Paused);
    assert_eq!(goal.pause_reason, Some(GoalPauseReason::DaemonRestart));
    assert!(goal.pending_call_id.is_none());
}

#[tokio::test]
async fn restart_coordinator_fences_every_host_before_waiting_for_any_one_host() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let runtime = GoalRuntime::new(store.clone() as Arc<dyn GoalTaskRegistry>);
    let settings = host_settings(true);
    let coordinator = Arc::new(GoalExecutionRestartCoordinator::new());
    let directory = TempDir::new().unwrap();
    let tracker = Arc::new(
        CostTracker::new(
            zeroclaw_config::schema::CostConfig {
                enabled: false,
                ..Default::default()
            },
            directory.path(),
        )
        .unwrap(),
    );
    let supervisor_a = coordinator
        .new_supervisor(Arc::new(
            runtime
                .execution_engine(Arc::clone(&tracker), "main", Arc::default())
                .unwrap(),
        ))
        .await;
    let supervisor_b = coordinator
        .new_supervisor(Arc::new(
            runtime
                .execution_engine(tracker, "main", Arc::default())
                .unwrap(),
        ))
        .await;
    let ingress_a = matrix_ingress_with_session("matrix_room__room_example_org__alice_a");
    let ingress_b = matrix_ingress_with_session("matrix_room__room_example_org__alice_b");
    let a_started = Arc::new(Notify::new());
    let a_release = Arc::new(Notify::new());
    let b_started = Arc::new(Notify::new());
    let b_release = Arc::new(Notify::new());
    let b_verifier_calls = Arc::new(AtomicUsize::new(0));
    let driver_a = Arc::new(PausingExecutionDriver {
        binding: GoalSessionBinding::new(ingress_a.session_key().clone()),
        parent_started: Arc::clone(&a_started),
        release_parent: Arc::clone(&a_release),
        verifier_calls: Arc::new(AtomicUsize::new(0)),
    });
    let driver_b = Arc::new(PausingExecutionDriver {
        binding: GoalSessionBinding::new(ingress_b.session_key().clone()),
        parent_started: Arc::clone(&b_started),
        release_parent: Arc::clone(&b_release),
        verifier_calls: Arc::clone(&b_verifier_calls),
    });
    for (supervisor, ingress, driver) in [
        (Arc::clone(&supervisor_a), ingress_a.clone(), driver_a),
        (Arc::clone(&supervisor_b), ingress_b.clone(), driver_b),
    ] {
        supervisor
            .submit(
                settings.clone(),
                ingress,
                driver,
                GoalCommand::Start {
                    budget: zeroclaw_commands::goal::GoalBudgetSelection::Defaults,
                    objective: "finish the task".into(),
                },
            )
            .await
            .unwrap();
    }
    a_started.notified().await;
    b_started.notified().await;

    let restarting = Arc::clone(&coordinator);
    let mut restart = zeroclaw_spawn::spawn!(async move { restarting.quiesce_for_restart().await });
    assert!(
        timeout(Duration::from_millis(20), &mut restart)
            .await
            .is_err(),
        "host A must still be draining"
    );

    b_release.notify_one();
    tokio::task::yield_now().await;
    let goal_b = store
        .current_goal_for_session(&ingress_b.session_key().durable_id())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(goal_b.status, TaskStatus::Paused);
    assert_eq!(
        b_verifier_calls.load(Ordering::SeqCst),
        0,
        "a globally fenced host must not admit a verifier while another host drains"
    );

    a_release.notify_one();
    restart.await.unwrap().unwrap();
}

#[tokio::test]
async fn restart_coordinator_refuses_new_execution_after_quiescence_begins() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let runtime = GoalRuntime::new(store.clone() as Arc<dyn GoalTaskRegistry>);
    let ingress = matrix_ingress();
    let directory = TempDir::new().unwrap();
    let tracker = Arc::new(
        CostTracker::new(
            zeroclaw_config::schema::CostConfig {
                enabled: false,
                ..Default::default()
            },
            directory.path(),
        )
        .unwrap(),
    );
    let coordinator = GoalExecutionRestartCoordinator::new();
    let supervisor = coordinator
        .new_supervisor(Arc::new(
            runtime
                .execution_engine(tracker, "main", Arc::default())
                .unwrap(),
        ))
        .await;
    let driver = Arc::new(ExecutionDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        execution_acquires: AtomicUsize::new(0),
        delivered: Arc::new(AtomicUsize::new(0)),
        complete_verifier: true,
    });

    coordinator.quiesce_for_restart().await.unwrap();
    let error = supervisor
        .submit(
            host_settings(true),
            ingress.clone(),
            driver,
            GoalCommand::Start {
                budget: zeroclaw_commands::goal::GoalBudgetSelection::Defaults,
                objective: "finish the task".into(),
            },
        )
        .await
        .expect_err("restart quiescence must reject a new execution admission");
    assert!(error.to_string().contains("quiescing"));
    assert!(
        store
            .current_goal_for_session(&ingress.session_key().durable_id())
            .await
            .unwrap()
            .is_none(),
        "a rejected admission must not create durable Goal state"
    );
}

#[tokio::test]
async fn supervisor_disposal_drains_and_classifies_a_pending_operation_before_deletion() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let runtime = GoalRuntime::new(store.clone() as Arc<dyn GoalTaskRegistry>);
    let settings = host_settings(true);
    let ingress = matrix_ingress();
    let parent_started = Arc::new(Notify::new());
    let release_parent = Arc::new(Notify::new());
    let verifier_calls = Arc::new(AtomicUsize::new(0));
    let driver = Arc::new(PausingExecutionDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        parent_started: Arc::clone(&parent_started),
        release_parent: Arc::clone(&release_parent),
        verifier_calls: Arc::clone(&verifier_calls),
    });
    let directory = TempDir::new().unwrap();
    let tracker = Arc::new(
        CostTracker::new(
            zeroclaw_config::schema::CostConfig {
                enabled: false,
                ..Default::default()
            },
            directory.path(),
        )
        .unwrap(),
    );
    let supervisor = Arc::new(GoalExecutionSupervisor::new(Arc::new(
        runtime
            .execution_engine(tracker, "main", Arc::default())
            .unwrap(),
    )));

    supervisor
        .submit(
            settings.clone(),
            ingress.clone(),
            driver,
            GoalCommand::Start {
                budget: zeroclaw_commands::goal::GoalBudgetSelection::Defaults,
                objective: "finish the task".into(),
            },
        )
        .await
        .unwrap();
    parent_started.notified().await;
    let current = store
        .current_goal_for_session(&ingress.session_key().durable_id())
        .await
        .unwrap()
        .expect("started Goal is current");
    assert_eq!(
        store
            .admit_pending_operation(
                &current.id,
                &ingress.session_key().durable_id(),
                current.execution_epoch,
                "disposal-pending-operation",
            )
            .await
            .unwrap(),
        GoalTransitionResult::Applied
    );

    let dispose_supervisor = Arc::clone(&supervisor);
    let session_id = ingress.session_key().durable_id();
    let mut dispose =
        zeroclaw_spawn::spawn!(
            async move { dispose_supervisor.dispose_session(&session_id).await }
        );
    assert!(
        timeout(Duration::from_millis(20), &mut dispose)
            .await
            .is_err(),
        "disposal must drain an admitted operation before deleting Goal state"
    );

    release_parent.notify_one();
    assert_eq!(
        dispose.await.unwrap().unwrap(),
        GoalTransitionResult::Applied
    );
    assert!(
        store
            .current_goal_for_session(&ingress.session_key().durable_id())
            .await
            .unwrap()
            .is_none(),
        "session disposal hard-deletes Goal control state only after pending classification"
    );
    assert_eq!(verifier_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn supervisor_pause_fails_closed_when_the_drained_operation_remains_pending() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let runtime = GoalRuntime::new(store.clone() as Arc<dyn GoalTaskRegistry>);
    let settings = host_settings(true);
    let ingress = matrix_ingress();
    let parent_started = Arc::new(Notify::new());
    let release_parent = Arc::new(Notify::new());
    let driver = Arc::new(PausingExecutionDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        parent_started: Arc::clone(&parent_started),
        release_parent: Arc::clone(&release_parent),
        verifier_calls: Arc::new(AtomicUsize::new(0)),
    });
    let directory = TempDir::new().unwrap();
    let tracker = Arc::new(
        CostTracker::new(
            zeroclaw_config::schema::CostConfig {
                enabled: false,
                ..Default::default()
            },
            directory.path(),
        )
        .unwrap(),
    );
    let supervisor = Arc::new(GoalExecutionSupervisor::new(Arc::new(
        runtime
            .execution_engine(tracker, "main", Arc::default())
            .unwrap(),
    )));

    supervisor
        .submit(
            settings.clone(),
            ingress.clone(),
            driver.clone(),
            GoalCommand::Start {
                budget: zeroclaw_commands::goal::GoalBudgetSelection::Defaults,
                objective: "finish the task".into(),
            },
        )
        .await
        .unwrap();
    parent_started.notified().await;
    let current = store
        .current_goal_for_session(&ingress.session_key().durable_id())
        .await
        .unwrap()
        .expect("started Goal is current");
    store
        .admit_pending_operation(
            &current.id,
            &ingress.session_key().durable_id(),
            current.execution_epoch,
            "pause-pending-operation",
        )
        .await
        .unwrap();

    let pause_supervisor = Arc::clone(&supervisor);
    let pause_settings = settings.clone();
    let pause_ingress = ingress.clone();
    let mut pause = zeroclaw_spawn::spawn!(async move {
        pause_supervisor
            .submit(pause_settings, pause_ingress, driver, GoalCommand::Pause)
            .await
    });
    timeout(Duration::from_millis(20), &mut pause)
        .await
        .expect_err("pause must wait for the admitted operation to drain");
    release_parent.notify_one();

    let response = pause.await.unwrap().unwrap().into_response();
    assert!(matches!(response, GoalResponse::Terminal(_)));
    let current = store
        .current_goal_for_session(&ingress.session_key().durable_id())
        .await
        .unwrap()
        .expect("failed Goal remains current for audit");
    let goal = store.get_goal_task(&current.id).await.unwrap().unwrap();
    assert_eq!(current.status, TaskStatus::Failed);
    assert_eq!(goal.accounting_state, GoalAccountingState::OutcomeUnknown);
    assert!(goal.pending_call_id.is_none());
}

#[tokio::test]
async fn operator_pause_drains_the_admitted_parent_without_starting_a_verifier() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let runtime = GoalRuntime::new(store.clone() as Arc<dyn GoalTaskRegistry>);
    let settings = host_settings(true);
    let ingress = matrix_ingress();
    let parent_started = Arc::new(Notify::new());
    let release_parent = Arc::new(Notify::new());
    let verifier_calls = Arc::new(AtomicUsize::new(0));
    let driver = Arc::new(PausingExecutionDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        parent_started: Arc::clone(&parent_started),
        release_parent: Arc::clone(&release_parent),
        verifier_calls: Arc::clone(&verifier_calls),
    });
    let request = runtime
        .submit(
            &settings,
            ingress.clone(),
            driver.clone(),
            GoalCommand::Start {
                budget: zeroclaw_commands::goal::GoalBudgetSelection::Defaults,
                objective: "finish the task".into(),
            },
        )
        .await
        .unwrap()
        .into_parts()
        .1
        .expect("start must yield an execution request");
    let directory = TempDir::new().unwrap();
    let tracker = Arc::new(
        CostTracker::new(
            zeroclaw_config::schema::CostConfig {
                enabled: false,
                ..Default::default()
            },
            directory.path(),
        )
        .unwrap(),
    );
    let engine = runtime
        .execution_engine(tracker, "main", Arc::default())
        .unwrap();

    let run = zeroclaw_spawn::spawn!(async move { engine.run(&settings, &request).await });
    parent_started.notified().await;

    let pause = runtime
        .submit(&host_settings(true), ingress, driver, GoalCommand::Pause)
        .await
        .unwrap()
        .into_parts()
        .0;
    assert!(matches!(pause, GoalResponse::Paused(_)));

    release_parent.notify_one();
    assert!(run.await.unwrap().is_err());
    assert_eq!(
        verifier_calls.load(Ordering::SeqCst),
        0,
        "a pause may drain an admitted parent operation but cannot admit a verifier"
    );
}

#[tokio::test]
async fn runtime_does_not_create_execution_for_read_only_goal_commands() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let runtime = GoalRuntime::new(store as Arc<dyn GoalTaskRegistry>);
    let settings = host_settings(true);
    let ingress = matrix_ingress();
    let driver = Arc::new(ExecutionDriver {
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        execution_acquires: AtomicUsize::new(0),
        delivered: Arc::new(AtomicUsize::new(0)),
        complete_verifier: false,
        continue_once: false,
        parent_histories: Arc::new(Mutex::new(Vec::new())),
    });

    let (response, execution) = runtime
        .submit(&settings, ingress, driver.clone(), GoalCommand::Status)
        .await
        .unwrap()
        .into_parts();
    assert!(matches!(response, GoalResponse::NoCurrentGoal));
    assert!(execution.is_none());
    assert_eq!(driver.execution_acquires.load(Ordering::SeqCst), 0);
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
        controller
            .submit(&settings, start)
            .await
            .unwrap()
            .into_parts()
            .0,
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
        controller
            .submit(&settings, status)
            .await
            .unwrap()
            .into_parts()
            .0,
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
        controller
            .submit(&settings, cancel)
            .await
            .unwrap()
            .into_parts()
            .0,
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
        controller
            .submit(&settings, replacement)
            .await
            .unwrap()
            .into_parts()
            .0,
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
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
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
    let completed = controller.submit(&settings, submission).await.unwrap();
    assert!(matches!(completed.response(), GoalResponse::Disabled));
    assert!(
        !driver.reconnect_is_allowed(),
        "the lease remains held until the host has completed lifecycle cleanup"
    );
    drop(completed);
    assert!(driver.reconnect_is_allowed());
}

#[tokio::test]
async fn host_submission_is_consumed_by_the_controller_transition() {
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let controller = GoalController::new(store as Arc<dyn GoalTaskRegistry>);
    let settings = host_settings(false);
    let ingress = matrix_ingress();
    let driver = recording_driver(&ingress);
    let submission = GoalExecutionHost::new()
        .submit(ingress, driver, GoalCommand::Status)
        .await
        .unwrap();

    assert!(matches!(
        controller
            .submit(&settings, submission)
            .await
            .unwrap()
            .response(),
        GoalResponse::Disabled
    ));
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
        binding: GoalSessionBinding::new(ingress.session_key().clone()),
        available_for_reconnect: reconnect_available,
    });

    let submission = GoalExecutionHost::new()
        .submit(ingress, driver.clone(), GoalCommand::Pause)
        .await
        .unwrap();

    let completed = controller.submit(&settings, submission).await.unwrap();
    let GoalResponse::Paused(_) = completed.response() else {
        panic!("expected a paused Goal");
    };
    assert_eq!(registry.pause_observed.load(Ordering::SeqCst), 1);
    assert!(
        !driver.reconnect_is_allowed(),
        "the lease must cover the host's post-transition cleanup"
    );
    drop(completed);
    assert!(driver.reconnect_is_allowed());
}
