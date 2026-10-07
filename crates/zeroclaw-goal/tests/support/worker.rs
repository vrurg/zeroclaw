//! VT86 exercises the real executor checkpoint with scripted, non-provider work.

use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::sync::{OwnedMutexGuard, Semaphore};
use zeroclaw_api::model_provider::ChatMessage;
use zeroclaw_goal::{
    GoalAccountingState, GoalExecutionNotice, GoalOperationScope, GoalParentTurn,
    GoalParentTurnResult, GoalVerifierResult, GoalVerifierTurn,
    accounting::{
        GoalOperationAccounting, GoalOperationAccountingFactory, GoalOperationRequest,
        GoalOperationSettlement,
    },
};

struct Script {
    entered: Semaphore,
    release: Semaphore,
    published: AtomicUsize,
    verified: AtomicUsize,
    notices: AtomicUsize,
    hold_verifier: AtomicBool,
    verifier_entered: Semaphore,
    verifier_release: Semaphore,
    fail_launch_bind: AtomicBool,
    fail_notice: AtomicBool,
    fail_notice_once: AtomicBool,
    fail_parent: AtomicBool,
    reject_execution_incarnation: AtomicBool,
    publication_entered: Semaphore,
    publication_gate: std::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    acquisitions: AtomicUsize,
    events: std::sync::Mutex<Vec<FailureEvent>>,
}

#[derive(Debug, PartialEq)]
enum FailureEvent {
    Original(String),
    Notice(GoalExecutionNotice),
}

impl Default for Script {
    fn default() -> Self {
        Self {
            entered: Semaphore::new(0),
            release: Semaphore::new(0),
            published: AtomicUsize::new(0),
            verified: AtomicUsize::new(0),
            notices: AtomicUsize::new(0),
            hold_verifier: AtomicBool::new(false),
            verifier_entered: Semaphore::new(0),
            verifier_release: Semaphore::new(0),
            fail_launch_bind: AtomicBool::new(false),
            fail_notice: AtomicBool::new(false),
            fail_notice_once: AtomicBool::new(false),
            fail_parent: AtomicBool::new(false),
            reject_execution_incarnation: AtomicBool::new(false),
            publication_entered: Semaphore::new(0),
            publication_gate: std::sync::Mutex::new(None),
            acquisitions: AtomicUsize::new(0),
            events: std::sync::Mutex::new(Vec::new()),
        }
    }
}

/// There are no physical provider attempts in this script. Accounting admission
/// or settlement would therefore be a fixture error, not a fabricated usage pass.
struct NoPhysicalAttempts;

#[async_trait]
impl GoalOperationAccounting for NoPhysicalAttempts {
    async fn admit(&self, _request: GoalOperationRequest) -> Result<()> {
        bail!("script unexpectedly admitted a physical provider attempt")
    }

    async fn settle(&self, _settlement: GoalOperationSettlement) -> Result<()> {
        bail!("script unexpectedly settled a physical provider attempt")
    }

    async fn pricing_complete(&self) -> Result<bool> {
        Ok(true)
    }
}

impl GoalOperationAccountingFactory for NoPhysicalAttempts {
    fn for_scope(&self, _scope: GoalExecutionScope) -> Arc<dyn GoalOperationAccounting> {
        Arc::new(Self)
    }
}

struct WorkerDriver {
    ordinary: Arc<SerializedBoundaryDriver>,
    script: Arc<Script>,
}

#[async_trait]
impl GoalSessionDriver for WorkerDriver {
    fn session_key(&self) -> &GoalSessionKey {
        self.ordinary.session_key()
    }

    async fn bind(&self, ingress: &GoalIngressContext) -> Result<GoalSessionLease> {
        if self.script.fail_launch_bind.swap(false, Ordering::SeqCst) {
            bail!("fixture launch rebind unavailable");
        }
        self.ordinary.bind(ingress).await
    }

    async fn try_turn_boundary(
        &self,
        ingress: &GoalIngressContext,
    ) -> Result<Option<GoalTurnBoundary>> {
        self.ordinary.try_turn_boundary(ingress).await
    }

    async fn await_turn_boundary(
        &self,
        ingress: &GoalIngressContext,
        checkpoint: &GoalParentTurnCheckpoint,
    ) -> Result<GoalTurnBoundary> {
        self.ordinary.await_turn_boundary(ingress, checkpoint).await
    }

    async fn acquire_execution(
        &self,
        _ingress: &GoalIngressContext,
        _scope: &GoalExecutionScope,
        checkpoint: &GoalParentTurnCheckpoint,
    ) -> Result<Box<dyn GoalSessionExecutionLease>> {
        if self
            .script
            .reject_execution_incarnation
            .load(Ordering::SeqCst)
        {
            bail!("fixture session incarnation was replaced");
        }
        let gate = self.script.publication_gate.lock().unwrap().take();
        if let Some(gate) = gate {
            self.script.publication_entered.add_permits(1);
            tokio::select! {
                biased;
                () = checkpoint.wait_closed() => bail!("script publication withdrawn"),
                result = gate => result?,
            }
        }
        let guard = tokio::select! {
            biased;
            () = checkpoint.wait_closed() => bail!("script execution admission closed"),
            guard = self.ordinary.boundary.clone().lock_owned() => guard,
        };
        self.script.acquisitions.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(WorkerLease {
            key: self.session_key().clone(),
            script: self.script.clone(),
            _guard: guard,
        }))
    }
}

struct WorkerLease {
    key: GoalSessionKey,
    script: Arc<Script>,
    _guard: OwnedMutexGuard<()>,
}

#[async_trait]
impl GoalSessionExecutionLease for WorkerLease {
    fn session_key(&self) -> &GoalSessionKey {
        &self.key
    }

    fn canonical_history(&self) -> Result<Vec<ChatMessage>> {
        Ok(vec![])
    }

    async fn run_parent_turn(
        &mut self,
        _scope: &GoalOperationScope,
        _turn: GoalParentTurn,
    ) -> Result<GoalParentTurnResult> {
        self.script.entered.add_permits(1);
        self.script.release.acquire().await?.forget();
        if self.script.fail_parent.load(Ordering::SeqCst) {
            bail!("fixture parent failed after its lifecycle fence");
        }
        let candidate = "The bounded fixture is complete.";
        GoalParentTurnResult::from_completed_agent_turn(
            candidate.into(),
            vec![ChatMessage::assistant(candidate)],
        )
    }

    async fn append_parent_candidate(&mut self, _candidate: String) -> Result<()> {
        self.script.published.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn run_verifier(
        &mut self,
        _scope: &GoalOperationScope,
        _turn: GoalVerifierTurn,
    ) -> Result<GoalVerifierResult> {
        self.script.verified.fetch_add(1, Ordering::SeqCst);
        if self.script.hold_verifier.load(Ordering::SeqCst) {
            self.script.verifier_entered.add_permits(1);
            self.script.verifier_release.acquire().await?.forget();
        }
        Ok(GoalVerifierResult {
            response: r#"{"decision":"complete","reason":"The declared criterion is satisfied.","blockers":[]}"#.into(),
            attribution: None,
        })
    }

    async fn publish_goal_notice(&mut self, notice: GoalExecutionNotice) -> Result<()> {
        self.script.notices.fetch_add(1, Ordering::SeqCst);
        if self.script.fail_notice.load(Ordering::SeqCst)
            || self.script.fail_notice_once.swap(false, Ordering::SeqCst)
        {
            bail!("fixture notice delivery unavailable");
        }
        self.script
            .events
            .lock()
            .unwrap()
            .push(FailureEvent::Notice(notice));
        Ok(())
    }

    async fn publish_goal_core_error(&mut self, error: &anyhow::Error) -> Result<()> {
        self.script
            .events
            .lock()
            .unwrap()
            .push(FailureEvent::Original(format!("{error:#}")));
        Ok(())
    }
}

async fn running_worker() -> (
    Arc<WorkerDriver>,
    GoalIngressContext,
    Arc<Registry>,
    Arc<GoalExecutionSupervisor>,
) {
    running_worker_on(None).await
}

/// Use the real command and boundary leases without launching a parent turn.
/// A failed admission audit must publish through this same execution driver.
fn unlaunched_worker_fixture() -> (
    Arc<WorkerDriver>,
    GoalIngressContext,
    Arc<Registry>,
    Arc<GoalExecutionSupervisor>,
) {
    let (ordinary, ingress) = serialized_fixture();
    let driver = Arc::new(WorkerDriver {
        ordinary,
        script: Arc::new(Script::default()),
    });
    let registry = Arc::new(Registry::default());
    let runtime = GoalRuntime::new(registry.clone());
    let supervisor = Arc::new(GoalExecutionSupervisor::new(Arc::new(
        runtime.execution_engine(Arc::new(NoPhysicalAttempts)),
    )));
    (driver, ingress, registry, supervisor)
}

/// Reach the audit fault only after the queued control owns its real boundary.
async fn deferred_audit_fixture() -> (
    Arc<WorkerDriver>,
    GoalIngressContext,
    Arc<Registry>,
    Arc<GoalExecutionSupervisor>,
    zeroclaw_goal::GoalDeferredReady,
) {
    let (driver, ingress, registry, supervisor) = unlaunched_worker_fixture();
    let ordinary = driver.ordinary.boundary.clone().lock_owned().await;
    let deferred = queued(
        supervisor
            .admit_control(
                settings(),
                ingress.clone(),
                driver.clone(),
                launch_command(false),
            )
            .await
            .unwrap(),
        GoalQueuedControl::Start,
    );
    drop(ordinary);
    let ready = tokio::time::timeout(Duration::from_secs(1), deferred.wait())
        .await
        .unwrap()
        .unwrap();
    assert!(driver.ordinary.boundary.try_lock().is_err());
    registry.0.lock().unwrap().fail_settlement = true;
    (driver, ingress, registry, supervisor, ready)
}

#[tokio::test]
async fn deferred_audit_failure_does_not_self_wait_on_held_boundary() {
    let (driver, _, registry, supervisor, ready) = deferred_audit_fixture().await;
    let result = tokio::time::timeout(Duration::from_secs(1), ready.commit(settings())).await;
    registry.0.lock().unwrap().fail_settlement = false;
    let error = result
        .expect("audit-failure publication must release the held boundary")
        .unwrap_err();
    assert!(error.is::<zeroclaw_goal::GoalLaunchFailureReported>());
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    assert_eq!(driver.script.acquisitions.load(Ordering::SeqCst), 1);
    assert_eq!(
        *driver.script.events.lock().unwrap(),
        [FailureEvent::Notice(GoalExecutionNotice::Failed(
            zeroclaw_goal::GoalFailureReason::ExecutorStartFailed
        ))]
    );
    {
        let state = registry.0.lock().unwrap();
        assert_eq!(state.task.as_ref().unwrap().status, TaskStatus::Failed);
        assert_eq!(
            state.terminal_reason.as_deref(),
            Some("executor_start_failed")
        );
        assert_eq!(state.transitions, ["start", "finish"]);
    }
    assert!(driver.ordinary.command.try_lock().is_ok());
    assert!(driver.ordinary.boundary.try_lock().is_ok());
    assert!(matches!(
        wait_for_settlement(&registry).await,
        IntentState::Settled(_)
    ));
    assert!(retire_without_waiting(&supervisor).await);
}

#[tokio::test]
async fn unlaunched_failure_rebind_competing_start_waits_for_notice() {
    let (driver, ingress, registry, supervisor) = unlaunched_worker_fixture();
    let GoalControlAdmission::Direct(mut submission) = supervisor
        .admit_control(
            settings(),
            ingress.clone(),
            driver.clone(),
            launch_command(false),
        )
        .await
        .unwrap()
    else {
        panic!("idle Start must apply directly");
    };
    let predecessor = registry.0.lock().unwrap().task.as_ref().unwrap().id.clone();
    driver.script.fail_launch_bind.store(true, Ordering::SeqCst);
    let release = hold_publication(&driver.script);
    let failure = {
        let supervisor = supervisor.clone();
        zeroclaw_spawn::spawn!(async move {
            supervisor
                .launch_after_response(settings(), &mut submission)
                .await
        })
    };
    await_publication(&driver.script).await;
    let admission = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.admit_control(settings(), ingress, driver.clone(), launch_command(false)),
    )
    .await
    .unwrap()
    .unwrap();
    let deferred = match admission {
        GoalControlAdmission::Queued { response, deferred } => {
            assert_eq!(response, GoalResponse::Queued(GoalQueuedControl::Start));
            deferred
        }
        GoalControlAdmission::Direct(submission) => {
            // Clean up the owned task before reporting the planned pre-fix RED.
            drop(submission);
            failure.abort();
            let _ = failure.await;
            drop(release);
            panic!("a Start must not overtake an unlaunched failure notice");
        }
    };
    {
        let state = registry.0.lock().unwrap();
        assert_eq!(state.task.as_ref().unwrap().id, predecessor);
        assert_eq!(state.task.as_ref().unwrap().status, TaskStatus::Failed);
        assert_eq!(
            state.terminal_reason.as_deref(),
            Some("executor_start_failed")
        );
    }
    let successor = zeroclaw_spawn::spawn!(async move { deferred.wait().await });
    release.send(()).unwrap();
    let error = tokio::time::timeout(Duration::from_secs(1), failure)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(error.is::<zeroclaw_goal::GoalLaunchFailureReported>());
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    assert_eq!(
        *driver.script.events.lock().unwrap(),
        [FailureEvent::Notice(GoalExecutionNotice::Failed(
            zeroclaw_goal::GoalFailureReason::ExecutorStartFailed
        ))]
    );
    let ready = tokio::time::timeout(Duration::from_secs(1), successor)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    let committed = tokio::time::timeout(Duration::from_secs(1), ready.commit(settings()))
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(committed.response(), GoalResponse::Started(_)));
    assert_ne!(
        registry.0.lock().unwrap().task.as_ref().unwrap().id,
        predecessor
    );
    drop(committed);
}

#[tokio::test]
async fn deferred_audit_failure_releases_command_and_boundary_before_notice() {
    let (driver, ingress, registry, supervisor, ready) = deferred_audit_fixture().await;
    let release = hold_publication(&driver.script);
    let committing = zeroclaw_spawn::spawn!(async move { ready.commit(settings()).await });
    await_publication(&driver.script).await;
    let authority_free =
        driver.ordinary.command.try_lock().is_ok() && driver.ordinary.boundary.try_lock().is_ok();
    if !authority_free {
        // Release the fixture gate and consume its task before the red assertion.
        drop(release);
        tokio::time::timeout(Duration::from_secs(1), committing)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        registry.0.lock().unwrap().fail_settlement = false;
        assert!(
            authority_free,
            "audit-failure publication must not own command authority or the ordinary boundary"
        );
        return;
    }
    {
        let state = registry.0.lock().unwrap();
        assert_eq!(state.task.as_ref().unwrap().status, TaskStatus::Failed);
        assert_eq!(
            state.terminal_reason.as_deref(),
            Some("executor_start_failed")
        );
    }
    let status = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.admit_control(settings(), ingress, driver.clone(), GoalCommand::Status),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(matches!(status, GoalControlAdmission::Direct(_)));
    drop(status);
    release.send(()).unwrap();
    let error = tokio::time::timeout(Duration::from_secs(1), committing)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    registry.0.lock().unwrap().fail_settlement = false;
    assert!(error.is::<zeroclaw_goal::GoalLaunchFailureReported>());
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    assert_eq!(driver.script.acquisitions.load(Ordering::SeqCst), 1);
    assert!(driver.ordinary.command.try_lock().is_ok());
    assert!(driver.ordinary.boundary.try_lock().is_ok());
}

#[tokio::test]
async fn deferred_audit_failure_notice_delivery_failure_is_still_reported() {
    let (driver, _, registry, _, ready) = deferred_audit_fixture().await;
    driver.script.fail_notice.store(true, Ordering::SeqCst);
    let result = tokio::time::timeout(Duration::from_secs(1), ready.commit(settings())).await;
    registry.0.lock().unwrap().fail_settlement = false;
    let error = result.unwrap().unwrap_err();
    assert!(error.is::<zeroclaw_goal::GoalLaunchFailureReported>());
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    assert!(driver.script.events.lock().unwrap().is_empty());
    let state = registry.0.lock().unwrap();
    assert_eq!(state.task.as_ref().unwrap().status, TaskStatus::Failed);
    assert_eq!(
        state.terminal_reason.as_deref(),
        Some("executor_start_failed")
    );
    drop(state);
    assert!(driver.ordinary.command.try_lock().is_ok());
    assert!(driver.ordinary.boundary.try_lock().is_ok());
}

#[tokio::test]
async fn deferred_audit_failure_terminal_commit_failure_reports_command_error() {
    let (driver, _, registry, _, ready) = deferred_audit_fixture().await;
    registry.0.lock().unwrap().fail_finish_once = true;
    let error = tokio::time::timeout(Duration::from_secs(1), ready.commit(settings()))
        .await
        .unwrap()
        .unwrap_err();
    registry.0.lock().unwrap().fail_settlement = false;
    assert!(!error.is::<zeroclaw_goal::GoalLaunchFailureReported>());
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 0);
    assert_eq!(driver.script.acquisitions.load(Ordering::SeqCst), 0);
    assert!(driver.ordinary.command.try_lock().is_ok());
    assert!(driver.ordinary.boundary.try_lock().is_ok());
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let finished = {
                let state = registry.0.lock().unwrap();
                state.task.as_ref().unwrap().status == TaskStatus::Failed && state.finish_calls == 2
            };
            if finished {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        registry.0.lock().unwrap().terminal_reason.as_deref(),
        Some("executor_start_failed")
    );
}

#[tokio::test]
async fn deferred_audit_failure_abort_during_notice_releases_authority() {
    let (driver, _, registry, supervisor, ready) = deferred_audit_fixture().await;
    let release = hold_publication(&driver.script);
    let committing = zeroclaw_spawn::spawn!(async move { ready.commit(settings()).await });
    await_publication(&driver.script).await;
    registry.0.lock().unwrap().fail_settlement = false;
    committing.abort();
    assert!(committing.await.unwrap_err().is_cancelled());
    drop(release);
    assert!(driver.ordinary.command.try_lock().is_ok());
    assert!(driver.ordinary.boundary.try_lock().is_ok());
    tokio::time::timeout(Duration::from_secs(1), async {
        while !supervisor.try_retire().await {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let state = registry.0.lock().unwrap();
    assert_eq!(state.task.as_ref().unwrap().status, TaskStatus::Failed);
    assert_eq!(
        state.terminal_reason.as_deref(),
        Some("executor_start_failed")
    );
    assert_eq!(state.transitions, ["start", "finish"]);
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn deferred_audit_failure_abort_during_audit_pauses_unlaunched_epoch() {
    let (driver, _, registry, _, ready) = deferred_audit_fixture().await;
    let entered = Arc::new(Semaphore::new(0));
    let wait = Arc::new(Semaphore::new(0));
    {
        let mut state = registry.0.lock().unwrap();
        state.fail_settlement = false;
        state.settlement_entered = Some(entered.clone());
        state.settlement_wait = Some(wait.clone());
    }
    let committing = zeroclaw_spawn::spawn!(async move { ready.commit(settings()).await });
    tokio::time::timeout(Duration::from_secs(1), entered.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    let task = registry.0.lock().unwrap().task.clone().unwrap();
    assert_eq!(task.status, TaskStatus::Running);
    committing.abort();
    assert!(committing.await.unwrap_err().is_cancelled());
    assert_unlaunched_epoch_paused(&registry, &task.id, task.execution_epoch).await;
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 0);
    assert!(driver.ordinary.command.try_lock().is_ok());
    assert!(driver.ordinary.boundary.try_lock().is_ok());
    wait.add_permits(1);
    wait_for_settlement(&registry).await;
}

#[tokio::test]
async fn deferred_audit_failure_worker_ticket_controls_skip_reconciliation() {
    for (command, kind) in [
        (
            GoalCommand::Pause { immediate: false },
            GoalQueuedControl::Pause,
        ),
        (
            GoalCommand::SetBudget(GoalBudgetSelection::Unlimited),
            GoalQueuedControl::Budget,
        ),
    ] {
        let (driver, ingress, registry, supervisor) = running_worker().await;
        let scope = {
            let state = registry.0.lock().unwrap();
            let task = state.task.as_ref().unwrap();
            GoalExecutionScope::new(
                task.id.clone(),
                task.session_id.clone().unwrap(),
                task.execution_epoch,
            )
            .unwrap()
        };
        let deferred = queued(
            supervisor
                .admit_control(settings(), ingress, driver.clone(), command)
                .await
                .unwrap(),
            kind,
        );
        let wait = zeroclaw_spawn::spawn!(async move { deferred.wait().await });
        driver.script.release.add_permits(1);
        let ready = tokio::time::timeout(Duration::from_secs(1), wait)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(driver.script.published.load(Ordering::SeqCst), 1);
        assert_eq!(driver.script.verified.load(Ordering::SeqCst), 0);
        registry.0.lock().unwrap().fail_settlement = true;
        let error = tokio::time::timeout(Duration::from_secs(1), ready.commit(settings()))
            .await
            .unwrap()
            .unwrap_err();
        registry.0.lock().unwrap().fail_settlement = false;
        assert!(!error.is::<zeroclaw_goal::GoalLaunchFailureReported>());
        assert_eq!(driver.script.notices.load(Ordering::SeqCst), 0);
        assert!(driver.ordinary.command.try_lock().is_ok());
        if kind == GoalQueuedControl::Pause {
            assert_eq!(
                registry.0.lock().unwrap().task.as_ref().unwrap().status,
                TaskStatus::Paused
            );
        } else {
            assert_eq!(
                registry
                    .0
                    .lock()
                    .unwrap()
                    .goal
                    .as_ref()
                    .unwrap()
                    .effective_token_limit,
                None
            );
        }
        tokio::time::timeout(Duration::from_secs(1), supervisor.drain(&scope))
            .await
            .unwrap()
            .unwrap();
    }
}

#[tokio::test]
async fn deferred_audit_failure_competing_start_waits_for_notice() {
    let (driver, ingress, registry, supervisor, ready) = deferred_audit_fixture().await;
    let release = hold_publication(&driver.script);
    let failure = zeroclaw_spawn::spawn!(async move { ready.commit(settings()).await });
    await_publication(&driver.script).await;
    registry.0.lock().unwrap().fail_settlement = false;
    let predecessor = registry.0.lock().unwrap().task.as_ref().unwrap().id.clone();
    let admission = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.admit_control(settings(), ingress, driver.clone(), launch_command(false)),
    )
    .await;
    let deferred = match admission {
        Ok(Ok(GoalControlAdmission::Queued { response, deferred })) => {
            assert_eq!(response, GoalResponse::Queued(GoalQueuedControl::Start));
            deferred
        }
        other => {
            // Consume all owned work before exposing the planned pre-fix RED.
            drop(other);
            failure.abort();
            let _ = failure.await;
            drop(release);
            panic!("a Start must not overtake an unlaunched failure notice");
        }
    };
    {
        let state = registry.0.lock().unwrap();
        assert_eq!(state.task.as_ref().unwrap().id, predecessor);
        assert_eq!(state.task.as_ref().unwrap().status, TaskStatus::Failed);
        assert_eq!(
            state.terminal_reason.as_deref(),
            Some("executor_start_failed")
        );
    }
    let successor = zeroclaw_spawn::spawn!(async move { deferred.wait().await });
    release.send(()).unwrap();
    let error = tokio::time::timeout(Duration::from_secs(1), failure)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(error.is::<zeroclaw_goal::GoalLaunchFailureReported>());
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    assert_eq!(
        *driver.script.events.lock().unwrap(),
        [FailureEvent::Notice(GoalExecutionNotice::Failed(
            zeroclaw_goal::GoalFailureReason::ExecutorStartFailed
        ))]
    );
    let ready = tokio::time::timeout(Duration::from_secs(1), successor)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    let committed = tokio::time::timeout(Duration::from_secs(1), ready.commit(settings()))
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(committed.response(), GoalResponse::Started(_)));
    assert_ne!(
        registry.0.lock().unwrap().task.as_ref().unwrap().id,
        predecessor
    );
    drop(committed);
}

#[tokio::test]
async fn deferred_audit_failure_competing_start_disposal_withdraws_both() {
    let (driver, ingress, registry, supervisor, ready) = deferred_audit_fixture().await;
    let release = hold_publication(&driver.script);
    let failure = zeroclaw_spawn::spawn!(async move { ready.commit(settings()).await });
    await_publication(&driver.script).await;
    registry.0.lock().unwrap().fail_settlement = false;
    let session = ingress.session_key().durable_id();
    let admission = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.admit_control(settings(), ingress, driver.clone(), launch_command(false)),
    )
    .await;
    let deferred = match admission {
        Ok(Ok(GoalControlAdmission::Queued { response, deferred })) => {
            assert_eq!(response, GoalResponse::Queued(GoalQueuedControl::Start));
            deferred
        }
        other => {
            drop(other);
            failure.abort();
            let _ = failure.await;
            drop(release);
            panic!("disposal fixture requires a Start queued behind the failure notice");
        }
    };
    let successor = zeroclaw_spawn::spawn!(async move { deferred.wait().await });
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), supervisor.dispose_session(&session))
            .await
            .unwrap()
            .unwrap(),
        zeroclaw_goal::GoalTransitionResult::Applied
    );
    assert!(
        tokio::time::timeout(Duration::from_secs(1), successor)
            .await
            .unwrap()
            .unwrap()
            .is_err()
    );
    release.send(()).unwrap();
    let error = tokio::time::timeout(Duration::from_secs(1), failure)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(error.is::<zeroclaw_goal::GoalLaunchFailureReported>());
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 0);
    assert!(driver.script.events.lock().unwrap().is_empty());
    assert!(registry.0.lock().unwrap().task.is_none());
    assert!(driver.ordinary.command.try_lock().is_ok());
    assert!(driver.ordinary.boundary.try_lock().is_ok());
}

#[tokio::test]
async fn deferred_audit_failure_competing_start_proceeds_after_teardown() {
    let (driver, ingress, registry, supervisor, ready) = deferred_audit_fixture().await;
    let release = hold_publication(&driver.script);
    let failure = zeroclaw_spawn::spawn!(async move { ready.commit(settings()).await });
    await_publication(&driver.script).await;
    registry.0.lock().unwrap().fail_settlement = false;
    let admission = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.admit_control(settings(), ingress, driver.clone(), launch_command(false)),
    )
    .await;
    let deferred = match admission {
        Ok(Ok(GoalControlAdmission::Queued { response, deferred })) => {
            assert_eq!(response, GoalResponse::Queued(GoalQueuedControl::Start));
            deferred
        }
        other => {
            drop(other);
            failure.abort();
            let _ = failure.await;
            drop(release);
            panic!("teardown fixture requires a Start queued behind the failure notice");
        }
    };
    // Once ready, B legitimately owns both leases. Observe A's release before
    // allowing the already-registered successor to acquire that authority.
    let (begin_wait, waiting) = tokio::sync::oneshot::channel();
    let successor = zeroclaw_spawn::spawn!(async move {
        waiting.await.unwrap();
        deferred.wait().await
    });
    failure.abort();
    assert!(failure.await.unwrap_err().is_cancelled());
    drop(release);
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 0);
    assert!(driver.ordinary.command.try_lock().is_ok());
    assert!(driver.ordinary.boundary.try_lock().is_ok());
    begin_wait.send(()).unwrap();
    let ready = tokio::time::timeout(Duration::from_secs(1), successor)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 0);
    let committed = tokio::time::timeout(Duration::from_secs(1), ready.commit(settings()))
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(committed.response(), GoalResponse::Started(_)));
    drop(committed);
}

#[tokio::test]
async fn unlaunched_failure_rebind_published_then_start_is_direct() {
    let (driver, ingress, _, supervisor) = unlaunched_worker_fixture();
    let GoalControlAdmission::Direct(mut submission) = supervisor
        .admit_control(
            settings(),
            ingress.clone(),
            driver.clone(),
            launch_command(false),
        )
        .await
        .unwrap()
    else {
        panic!("idle Start must apply directly");
    };
    driver.script.fail_launch_bind.store(true, Ordering::SeqCst);
    let error = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.launch_after_response(settings(), &mut submission),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(error.is::<zeroclaw_goal::GoalLaunchFailureReported>());
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    drop(submission);
    let admission = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.admit_control(settings(), ingress, driver, launch_command(false)),
    )
    .await
    .unwrap()
    .unwrap();
    let GoalControlAdmission::Direct(submission) = admission else {
        panic!("a settled failure notice must not defer Start");
    };
    assert!(matches!(submission.response(), GoalResponse::Started(_)));
    drop(submission);
}

async fn running_worker_on(
    coordinator: Option<&zeroclaw_goal::GoalExecutionRestartCoordinator>,
) -> (
    Arc<WorkerDriver>,
    GoalIngressContext,
    Arc<Registry>,
    Arc<GoalExecutionSupervisor>,
) {
    let (ordinary, ingress) = serialized_fixture();
    let driver = Arc::new(WorkerDriver {
        ordinary,
        script: Arc::new(Script::default()),
    });
    let registry = Arc::new(Registry::default());
    let runtime = GoalRuntime::new(registry.clone());
    let engine = Arc::new(runtime.execution_engine(Arc::new(NoPhysicalAttempts)));
    let supervisor = match coordinator {
        Some(coordinator) => coordinator.new_supervisor(engine).await,
        None => Arc::new(GoalExecutionSupervisor::new(engine)),
    };
    let mut submission = supervisor
        .submit(
            settings(),
            ingress.clone(),
            driver.clone(),
            GoalCommand::Start {
                budget: GoalBudgetSelection::Defaults,
                objective: "Complete the bounded fixture".into(),
            },
        )
        .await
        .unwrap();
    supervisor
        .launch_after_response(settings(), &mut submission)
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(1), driver.script.entered.acquire())
        .await
        .expect("actual parent operation must have started")
        .unwrap()
        .forget();
    (driver, ingress, registry, supervisor)
}

#[tokio::test]
async fn fenced_parent_failure_delivers_notice_before_late_terminal_response() {
    assert_fenced_parent_failure_delivery(false).await;
}

#[tokio::test]
async fn fenced_unsettled_call_delivers_accounting_notice_before_late_terminal_response() {
    assert_fenced_parent_failure_delivery(true).await;
}

fn seed_unsettled_call(registry: &Registry) {
    let mut state = registry.0.lock().unwrap();
    let epoch = state.task.as_ref().unwrap().execution_epoch;
    let goal = state.goal.as_mut().unwrap();
    // Seed only durable evidence of an outcome-unknown operation. This is
    // not a fabricated physical attempt or a claim that usage was settled.
    goal.pending_call_id = Some("fixture-unsettled-operation".into());
    goal.pending_call_epoch = Some(epoch);
}

async fn assert_fenced_parent_failure_delivery(unsettled: bool) {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    if unsettled {
        seed_unsettled_call(&registry);
    }
    driver.script.fail_parent.store(true, Ordering::SeqCst);
    let mut pause = supervisor
        .submit(
            settings(),
            ingress,
            driver.clone(),
            GoalCommand::Pause { immediate: true },
        )
        .await
        .unwrap();
    assert!(matches!(
        pause.response(),
        GoalResponse::TransitionPending(_)
    ));
    assert_eq!(
        registry.0.lock().unwrap().task.as_ref().unwrap().status,
        TaskStatus::Paused
    );
    driver.script.release.add_permits(1);
    let late = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.launch_after_response(settings(), &mut pause),
    )
    .await
    .expect("classified parent must settle without another model call")
    .unwrap();
    assert!(matches!(late, Some(GoalResponse::Terminal(_))));
    assert_eq!(
        registry.0.lock().unwrap().task.as_ref().unwrap().status,
        TaskStatus::Failed
    );
    assert_eq!(driver.script.verified.load(Ordering::SeqCst), 0);
    assert_eq!(driver.script.published.load(Ordering::SeqCst), 0);
    assert_eq!(
        registry.0.lock().unwrap().terminal_reason.as_deref(),
        Some(if unsettled {
            "accounting_outcome_unknown"
        } else {
            "executor_failed"
        })
    );
    assert_eq!(
        driver.script.notices.load(Ordering::SeqCst),
        1,
        "the closed admission checkpoint must not swallow the drainer's failure notice"
    );
    assert_eq!(
        driver.script.events.lock().unwrap().last(),
        Some(&FailureEvent::Notice(GoalExecutionNotice::Failed(
            if unsettled {
                zeroclaw_goal::GoalFailureReason::OperationOutcomeUnknown
            } else {
                zeroclaw_goal::GoalFailureReason::ExecutorFailed
            }
        )))
    );
    let state = registry.0.lock().unwrap();
    if unsettled {
        let goal = state.goal.as_ref().unwrap();
        assert!(goal.pending_call_id.is_none());
        assert!(goal.pending_call_epoch.is_none());
        assert_eq!(goal.accounting_state, GoalAccountingState::OutcomeUnknown);
        assert_eq!(
            state.pending_settlements,
            [GoalAccountingState::OutcomeUnknown]
        );
    } else {
        assert!(state.pending_settlements.is_empty());
    }
}

fn hold_publication(script: &Script) -> tokio::sync::oneshot::Sender<()> {
    let (release, gate) = tokio::sync::oneshot::channel();
    *script.publication_gate.lock().unwrap() = Some(gate);
    release
}

async fn await_publication(script: &Script) {
    tokio::time::timeout(Duration::from_secs(1), script.publication_entered.acquire())
        .await
        .expect("publication must reach the pre-FIFO gate")
        .unwrap()
        .forget();
}

fn assert_original_then_executor_notice(script: &Script) {
    let events = script.events.lock().unwrap();
    assert_eq!(
        events.len(),
        2,
        "one original and one notice, without duplication"
    );
    assert!(matches!(&events[0], FailureEvent::Original(error)
        if error.contains("fixture lifecycle settlement unavailable")));
    assert_eq!(
        events[1],
        FailureEvent::Notice(GoalExecutionNotice::Failed(
            zeroclaw_goal::GoalFailureReason::ExecutorFailed
        ))
    );
}

#[tokio::test]
async fn failed_tail_before_fifo_queues_start_until_original_and_notice_settle() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    driver.script.fail_parent.store(true, Ordering::SeqCst);
    registry.0.lock().unwrap().fail_finish_once = true;
    let release = hold_publication(&driver.script);
    driver.script.release.add_permits(1);
    await_publication(&driver.script).await;
    let predecessor = registry.0.lock().unwrap().task.as_ref().unwrap().clone();
    assert_eq!(predecessor.status, TaskStatus::Failed);
    assert_eq!(
        registry.0.lock().unwrap().terminal_reason.as_deref(),
        Some("executor_failed")
    );
    let snapshot = serde_json::to_value(&predecessor).unwrap();
    for command in [
        GoalCommand::Status,
        GoalCommand::Help,
        GoalCommand::Budget,
        GoalCommand::Resume { response: None },
        GoalCommand::Pause { immediate: false },
        GoalCommand::SetBudget(GoalBudgetSelection::Unlimited),
        GoalCommand::Cancel,
        GoalCommand::Pause { immediate: true },
    ] {
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            supervisor.admit_control(settings(), ingress.clone(), driver.clone(), command),
        )
        .await
        .expect("unsettled publication must not defer other controls")
        .unwrap();
        assert!(matches!(result, GoalControlAdmission::Direct(_)));
        drop(result);
        assert_eq!(
            serde_json::to_value(registry.0.lock().unwrap().task.as_ref().unwrap()).unwrap(),
            snapshot
        );
        assert!(registry.0.lock().unwrap().intents.is_empty());
        assert_eq!(driver.script.acquisitions.load(Ordering::SeqCst), 1);
    }
    // The publication has not joined the real foreground FIFO. Admission must
    // therefore consult publication custody, not just an available turn permit.
    let deferred = queued(
        supervisor
            .admit_control(
                settings(),
                ingress.clone(),
                driver.clone(),
                GoalCommand::Start {
                    budget: GoalBudgetSelection::Defaults,
                    objective: "bounded successor".into(),
                },
            )
            .await
            .unwrap(),
        GoalQueuedControl::Start,
    );
    assert_eq!(
        registry.0.lock().unwrap().task.as_ref().unwrap().id,
        predecessor.id
    );
    let mut wait = Box::pin(deferred.wait());
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut wait)
            .await
            .is_err()
    );
    assert_eq!(driver.script.acquisitions.load(Ordering::SeqCst), 1);
    assert!(driver.script.events.lock().unwrap().is_empty());
    release.send(()).unwrap();
    let ready = tokio::time::timeout(Duration::from_secs(1), wait)
        .await
        .unwrap()
        .unwrap();
    assert_original_then_executor_notice(&driver.script);
    let committed = ready.commit(settings()).await.unwrap();
    assert!(matches!(committed.response(), GoalResponse::Started(_)));
    let successor = registry.0.lock().unwrap().task.as_ref().unwrap().clone();
    assert_ne!(successor.id, predecessor.id);
    assert_eq!(
        registry.0.lock().unwrap().transitions,
        ["start", "finish", "start"]
    );
    let mut launch = committed.release();
    driver.script.fail_parent.store(false, Ordering::SeqCst);
    supervisor
        .launch_after_response(settings(), &mut launch)
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(1), driver.script.entered.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    assert_eq!(driver.script.acquisitions.load(Ordering::SeqCst), 3);
    let mut cancel = supervisor
        .submit(settings(), ingress, driver.clone(), GoalCommand::Cancel)
        .await
        .unwrap();
    driver.script.release.add_permits(1);
    supervisor
        .launch_after_response(settings(), &mut cancel)
        .await
        .unwrap();
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    assert_original_then_executor_notice(&driver.script);
}

#[tokio::test]
async fn drainer_delivery_before_fifo_queues_start_without_holding_command_authority() {
    assert_drainer_delivery_queues_start(false).await;
}

#[tokio::test]
async fn unsettled_drainer_delivery_before_fifo_queues_start_until_accounting_notice() {
    assert_drainer_delivery_queues_start(true).await;
}

#[tokio::test]
async fn ack_failure_predecessor_delivery_releases_command_authority() {
    assert_ack_failure_releases_command_authority(false).await;
}

#[tokio::test]
async fn unsettled_ack_failure_predecessor_delivery_releases_command_authority() {
    assert_ack_failure_releases_command_authority(true).await;
}

/// A failed acknowledgement must relinquish command authority before waiting
/// for native failure publication, just like the delivered-response path.
async fn assert_ack_failure_releases_command_authority(unsettled: bool) {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    let admitted = registry.0.lock().unwrap().task.as_ref().unwrap().clone();
    let predecessor_scope = GoalExecutionScope::new(
        admitted.id,
        admitted.session_id.unwrap(),
        admitted.execution_epoch,
    )
    .unwrap();
    if unsettled {
        seed_unsettled_call(&registry);
    }
    driver.script.fail_parent.store(true, Ordering::SeqCst);
    let release = hold_publication(&driver.script);
    let mut pause = supervisor
        .submit(
            settings(),
            ingress.clone(),
            driver.clone(),
            GoalCommand::Pause { immediate: true },
        )
        .await
        .unwrap();
    assert!(matches!(
        pause.response(),
        GoalResponse::TransitionPending(_)
    ));
    driver.script.release.add_permits(1);
    let draining_supervisor = supervisor.clone();
    let draining = zeroclaw_spawn::spawn!(async move {
        draining_supervisor
            .pause_before_launch(
                &mut pause,
                zeroclaw_goal::GoalPauseReason::ExternalDependency,
            )
            .await
    });
    await_publication(&driver.script).await;
    let predecessor = registry.0.lock().unwrap().task.as_ref().unwrap().clone();
    assert_eq!(predecessor.status, TaskStatus::Failed);
    assert!(
        driver.ordinary.command.try_lock().is_ok(),
        "acknowledgement-failure delivery must not own command authority"
    );
    let deferred = queued(
        tokio::time::timeout(
            Duration::from_secs(1),
            supervisor.admit_control(
                settings(),
                ingress,
                driver.clone(),
                GoalCommand::Start {
                    budget: GoalBudgetSelection::Defaults,
                    objective: "bounded successor".into(),
                },
            ),
        )
        .await
        .expect("Start must be acknowledged while failure publication waits")
        .unwrap(),
        GoalQueuedControl::Start,
    );
    assert_eq!(
        registry.0.lock().unwrap().task.as_ref().unwrap().id,
        predecessor.id
    );
    release.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(1), draining)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    assert_eq!(
        driver.script.events.lock().unwrap().last(),
        Some(&FailureEvent::Notice(GoalExecutionNotice::Failed(
            if unsettled {
                zeroclaw_goal::GoalFailureReason::OperationOutcomeUnknown
            } else {
                zeroclaw_goal::GoalFailureReason::ExecutorFailed
            }
        )))
    );
    assert_eq!(
        registry.0.lock().unwrap().pending_settlements.len(),
        usize::from(unsettled)
    );
    assert!(!supervisor.owns_scope(&predecessor_scope).await);
    let ready = tokio::time::timeout(Duration::from_secs(1), deferred.wait())
        .await
        .unwrap()
        .unwrap();
    let committed = ready.commit(settings()).await.unwrap();
    assert!(matches!(committed.response(), GoalResponse::Started(_)));
    assert_ne!(
        registry.0.lock().unwrap().task.as_ref().unwrap().id,
        predecessor.id
    );
    let mut launch = committed.release();
    supervisor
        .pause_before_launch(&mut launch, zeroclaw_goal::GoalPauseReason::OperatorPaused)
        .await
        .unwrap();
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn unseen_terminal_status_consumes_owed_predecessor_notice() {
    assert_unseen_terminal_consumes_owed_notice(GoalCommand::Status).await;
}

#[tokio::test]
async fn ack_failure_direct_boundary_is_released_before_predecessor_delivery() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    let task = registry.0.lock().unwrap().task.as_ref().unwrap().clone();
    let scope =
        GoalExecutionScope::new(task.id, task.session_id.unwrap(), task.execution_epoch).unwrap();
    driver.script.fail_parent.store(true, Ordering::SeqCst);
    registry.0.lock().unwrap().fail_finish = true;
    driver.script.release.add_permits(1);
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let completed = supervisor
                .register_deferred_control(&scope)
                .await
                .unwrap()
                .is_none();
            if completed && driver.ordinary.boundary.try_lock().is_ok() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        registry.0.lock().unwrap().task.as_ref().unwrap().status,
        TaskStatus::Running,
        "VT114(b) requires a completed worker whose durable row remains Running"
    );
    let GoalControlAdmission::Direct(mut submission) = supervisor
        .admit_control(
            settings(),
            ingress,
            driver.clone(),
            GoalCommand::Pause { immediate: false },
        )
        .await
        .unwrap()
    else {
        panic!("VT114(b) requires a direct boundary-bearing admission");
    };
    assert!(
        matches!(submission.response(), GoalResponse::TransitionPending(_)),
        "VT114(b) requires a predecessor obligation after plain pause"
    );
    assert!(driver.ordinary.boundary.try_lock().is_err());
    registry.0.lock().unwrap().fail_finish = false;
    tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.pause_before_launch(
            &mut submission,
            zeroclaw_goal::GoalPauseReason::ExternalDependency,
        ),
    )
    .await
    .expect("acknowledgement failure must release the direct boundary before delivery")
    .unwrap();
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    assert_eq!(
        driver.script.events.lock().unwrap().last(),
        Some(&FailureEvent::Notice(GoalExecutionNotice::Failed(
            zeroclaw_goal::GoalFailureReason::ExecutorFailed
        ),))
    );
    assert!(!supervisor.owns_scope(&scope).await);
    assert!(driver.ordinary.boundary.try_lock().is_ok());
    assert!(driver.ordinary.command.try_lock().is_ok());
}

#[tokio::test]
async fn unseen_terminal_pause_now_consumes_owed_predecessor_notice() {
    assert_unseen_terminal_consumes_owed_notice(GoalCommand::Pause { immediate: true }).await;
}

/// The shared handoff already retries an owed notice without repeating the
/// original error or changing durable state; both adapters must invoke it.
async fn assert_unseen_terminal_consumes_owed_notice(command: GoalCommand) {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    let task = registry.0.lock().unwrap().task.as_ref().unwrap().clone();
    let scope =
        GoalExecutionScope::new(task.id, task.session_id.unwrap(), task.execution_epoch).unwrap();
    driver.script.fail_parent.store(true, Ordering::SeqCst);
    driver.script.fail_notice_once.store(true, Ordering::SeqCst);
    registry.0.lock().unwrap().fail_finish_once = true;
    driver.script.release.add_permits(1);
    tokio::time::timeout(Duration::from_secs(1), async {
        while driver.script.notices.load(Ordering::SeqCst) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(matches!(
        driver.script.events.lock().unwrap().as_slice(),
        [FailureEvent::Original(_)]
    ));
    assert!(supervisor.owns_scope(&scope).await);
    let before = {
        let state = registry.0.lock().unwrap();
        (
            serde_json::to_value(&state.task).unwrap(),
            state.terminal_reason.clone(),
            state.transitions.clone(),
            state.finish_calls,
        )
    };
    let acquisitions = driver.script.acquisitions.load(Ordering::SeqCst);
    let GoalControlAdmission::Direct(mut submission) = supervisor
        .admit_control(settings(), ingress.clone(), driver.clone(), command)
        .await
        .unwrap()
    else {
        panic!("a terminal command must not queue");
    };
    assert!(matches!(submission.response(), GoalResponse::Terminal(_)));
    tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.pause_before_launch(
            &mut submission,
            zeroclaw_goal::GoalPauseReason::ExternalDependency,
        ),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 2);
    assert_original_then_executor_notice(&driver.script);
    assert!(!supervisor.owns_scope(&scope).await);
    {
        let state = registry.0.lock().unwrap();
        assert_eq!(
            (
                serde_json::to_value(&state.task).unwrap(),
                state.terminal_reason.clone(),
                state.transitions.clone(),
                state.finish_calls
            ),
            before,
        );
        assert!(state.pending_settlements.is_empty());
    }
    assert_eq!(
        driver.script.acquisitions.load(Ordering::SeqCst),
        acquisitions + 1
    );
    // The adapter consumes the completed handoff before accepting another
    // command. Preserve that lifetime on the pre-fix control too.
    drop(submission);
    let GoalControlAdmission::Direct(mut status) = supervisor
        .admit_control(settings(), ingress, driver.clone(), GoalCommand::Status)
        .await
        .unwrap()
    else {
        panic!("a settled terminal status must not queue");
    };
    supervisor
        .pause_before_launch(
            &mut status,
            zeroclaw_goal::GoalPauseReason::ExternalDependency,
        )
        .await
        .unwrap();
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 2);
}

async fn assert_drainer_delivery_queues_start(unsettled: bool) {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    if unsettled {
        seed_unsettled_call(&registry);
    }
    driver.script.fail_parent.store(true, Ordering::SeqCst);
    let release = hold_publication(&driver.script);
    let mut pause = supervisor
        .submit(
            settings(),
            ingress.clone(),
            driver.clone(),
            GoalCommand::Pause { immediate: true },
        )
        .await
        .unwrap();
    driver.script.release.add_permits(1);
    let draining_supervisor = Arc::clone(&supervisor);
    let drain = zeroclaw_spawn::spawn!(async move {
        draining_supervisor
            .launch_after_response(settings(), &mut pause)
            .await
    });
    await_publication(&driver.script).await;
    let predecessor = registry.0.lock().unwrap().task.as_ref().unwrap().clone();
    assert_eq!(predecessor.status, TaskStatus::Failed);
    let admission = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.admit_control(
            settings(),
            ingress.clone(),
            driver.clone(),
            GoalCommand::Start {
                budget: GoalBudgetSelection::Defaults,
                objective: "bounded successor".into(),
            },
        ),
    )
    .await
    .expect("delivery must not own command authority")
    .unwrap();
    let predecessor_retained =
        registry.0.lock().unwrap().task.as_ref().unwrap().id == predecessor.id;
    // Release a defective direct admission's lease too, so the baseline reaches
    // the missing-notice observation instead of stopping at the earlier race.
    let deferred = match admission {
        GoalControlAdmission::Queued { response, deferred } => Some((response, deferred)),
        GoalControlAdmission::Direct(direct) => {
            drop(direct);
            None
        }
    };
    let publication_released = release.send(()).is_ok();
    let late = tokio::time::timeout(Duration::from_secs(1), drain)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    assert!(
        publication_released,
        "the pending publication must retain its gate"
    );
    assert!(matches!(late, Some(GoalResponse::Terminal(_))));
    assert_eq!(
        driver.script.events.lock().unwrap().last(),
        Some(&FailureEvent::Notice(GoalExecutionNotice::Failed(
            if unsettled {
                zeroclaw_goal::GoalFailureReason::OperationOutcomeUnknown
            } else {
                zeroclaw_goal::GoalFailureReason::ExecutorFailed
            }
        )))
    );
    assert_eq!(
        registry.0.lock().unwrap().pending_settlements.len(),
        usize::from(unsettled)
    );
    assert!(
        predecessor_retained,
        "Start must retain the failed predecessor"
    );
    let (response, deferred) =
        deferred.expect("Start must be Queued until failure delivery settles");
    assert_eq!(response, GoalResponse::Queued(GoalQueuedControl::Start));
    let ready = tokio::time::timeout(Duration::from_secs(1), deferred.wait())
        .await
        .unwrap()
        .unwrap();
    let committed = ready.commit(settings()).await.unwrap();
    assert!(matches!(committed.response(), GoalResponse::Started(_)));
    let mut launch = committed.release();
    supervisor
        .pause_before_launch(&mut launch, zeroclaw_goal::GoalPauseReason::OperatorPaused)
        .await
        .unwrap();
    assert_eq!(driver.script.acquisitions.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn disposal_withdraws_failed_tail_and_closes_its_queued_start() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    driver.script.fail_parent.store(true, Ordering::SeqCst);
    registry.0.lock().unwrap().fail_finish_once = true;
    let release = hold_publication(&driver.script);
    driver.script.release.add_permits(1);
    await_publication(&driver.script).await;
    let predecessor = registry.0.lock().unwrap().task.as_ref().unwrap().clone();
    assert_eq!(predecessor.status, TaskStatus::Failed);
    let deferred = queued(
        supervisor
            .admit_control(
                settings(),
                ingress.clone(),
                driver.clone(),
                GoalCommand::Start {
                    budget: GoalBudgetSelection::Defaults,
                    objective: "bounded successor".into(),
                },
            )
            .await
            .unwrap(),
        GoalQueuedControl::Start,
    );
    assert_eq!(
        tokio::time::timeout(
            Duration::from_secs(1),
            supervisor.dispose_session(&ingress.session_key().durable_id()),
        )
        .await
        .expect("disposal must withdraw the tail without waiting on its held gate")
        .unwrap(),
        zeroclaw_goal::GoalTransitionResult::Applied
    );
    let closed = tokio::time::timeout(Duration::from_secs(1), deferred.wait())
        .await
        .expect("disposal must close the queued Start")
        .unwrap_err();
    assert!(matches!(closed.response(), GoalResponse::NoCurrentGoal));
    drop(release);
    assert!(registry.0.lock().unwrap().task.is_none());
    assert!(driver.script.events.lock().unwrap().is_empty());
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 0);
    assert_eq!(driver.script.acquisitions.load(Ordering::SeqCst), 1);
    assert!(supervisor.try_retire().await);
}

#[tokio::test]
async fn restart_closes_queued_start_while_failed_tail_publication_is_pending() {
    let coordinator = zeroclaw_goal::GoalExecutionRestartCoordinator::new();
    let (driver, ingress, registry, supervisor) = running_worker_on(Some(&coordinator)).await;
    driver.script.fail_parent.store(true, Ordering::SeqCst);
    registry.0.lock().unwrap().fail_finish_once = true;
    let release = hold_publication(&driver.script);
    driver.script.release.add_permits(1);
    await_publication(&driver.script).await;
    let predecessor = registry.0.lock().unwrap().task.as_ref().unwrap().clone();
    assert_eq!(predecessor.status, TaskStatus::Failed);
    let deferred = queued(
        supervisor
            .admit_control(
                settings(),
                ingress,
                driver.clone(),
                GoalCommand::Start {
                    budget: GoalBudgetSelection::Defaults,
                    objective: "bounded successor".into(),
                },
            )
            .await
            .unwrap(),
        GoalQueuedControl::Start,
    );
    tokio::time::timeout(Duration::from_secs(1), coordinator.quiesce_for_restart())
        .await
        .expect("restart must close queued controls and settle the failed tail")
        .unwrap();
    let closed = tokio::time::timeout(Duration::from_secs(1), deferred.wait())
        .await
        .expect("restart must close the queued Start")
        .unwrap_err();
    assert!(matches!(closed.response(), GoalResponse::Terminal(_)));
    drop(release);
    {
        let state = registry.0.lock().unwrap();
        assert_eq!(state.task.as_ref().unwrap().id, predecessor.id);
        assert_eq!(state.task.as_ref().unwrap().status, TaskStatus::Failed);
    }
    assert_eq!(driver.script.acquisitions.load(Ordering::SeqCst), 2);
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    {
        let events = driver.script.events.lock().unwrap();
        assert!(matches!(
            events.as_slice(),
            [FailureEvent::Original(_), FailureEvent::Notice(_)]
        ));
    }
    assert!(supervisor.try_retire().await);
}

#[tokio::test]
async fn foreground_blocked_failure_delivery_does_not_hold_lifecycle_finalization() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    driver.script.fail_parent.store(true, Ordering::SeqCst);
    let mut pause = supervisor
        .submit(
            settings(),
            ingress.clone(),
            driver.clone(),
            GoalCommand::Pause { immediate: true },
        )
        .await
        .unwrap();
    driver.script.release.add_permits(1);
    let holder = tokio::time::timeout(
        Duration::from_secs(1),
        driver.ordinary.boundary.clone().lock_owned(),
    )
    .await
    .expect("the admitted parent must release the real foreground");
    let draining_supervisor = Arc::clone(&supervisor);
    let mut drain = zeroclaw_spawn::spawn!(async move {
        draining_supervisor
            .launch_after_response(settings(), &mut pause)
            .await
    });
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut drain)
            .await
            .is_err(),
        "classified failure delivery must wait for the actual foreground holder"
    );
    assert_eq!(
        registry.0.lock().unwrap().task.as_ref().unwrap().status,
        TaskStatus::Failed
    );
    assert_eq!(
        tokio::time::timeout(
            Duration::from_secs(1),
            supervisor.dispose_session(&ingress.session_key().durable_id()),
        )
        .await
        .expect("foreground-blocked publication must release lifecycle finalization")
        .unwrap(),
        zeroclaw_goal::GoalTransitionResult::Applied
    );
    drop(holder);
    tokio::time::timeout(Duration::from_secs(1), drain)
        .await
        .expect("withdrawn delivery must terminate after disposal")
        .unwrap()
        .unwrap();
    assert!(registry.0.lock().unwrap().task.is_none());
    assert!(driver.script.events.lock().unwrap().is_empty());
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 0);
    assert!(supervisor.try_retire().await);
}

#[tokio::test]
async fn disposal_withdraws_inflight_drainer_delivery_without_waiting_on_its_latch() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    driver.script.fail_parent.store(true, Ordering::SeqCst);
    let release = hold_publication(&driver.script);
    let mut pause = supervisor
        .submit(
            settings(),
            ingress.clone(),
            driver.clone(),
            GoalCommand::Pause { immediate: true },
        )
        .await
        .unwrap();
    driver.script.release.add_permits(1);
    let draining_supervisor = Arc::clone(&supervisor);
    let drain = zeroclaw_spawn::spawn!(async move {
        draining_supervisor
            .launch_after_response(settings(), &mut pause)
            .await
    });
    await_publication(&driver.script).await;
    assert_eq!(
        registry.0.lock().unwrap().task.as_ref().unwrap().status,
        TaskStatus::Failed
    );
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.dispose_session(&ingress.session_key().durable_id()),
    )
    .await
    .expect("disposal must not wait for the delivery's held latch or lifecycle lock")
    .unwrap();
    assert_eq!(result, zeroclaw_goal::GoalTransitionResult::Applied);
    tokio::time::timeout(Duration::from_secs(1), drain)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    drop(release);
    assert!(registry.0.lock().unwrap().task.is_none());
    assert!(driver.script.events.lock().unwrap().is_empty());
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 0);
    assert!(supervisor.try_retire().await);
}

#[tokio::test]
async fn restart_drains_all_hosts_before_waiting_for_failure_delivery() {
    let coordinator = Arc::new(zeroclaw_goal::GoalExecutionRestartCoordinator::new());
    let (first, _, first_registry, first_supervisor) = running_worker_on(Some(&coordinator)).await;
    let (second, second_ingress, second_registry, second_supervisor) =
        running_worker_on(Some(&coordinator)).await;
    first.script.fail_parent.store(true, Ordering::SeqCst);
    let second_task = second_registry
        .0
        .lock()
        .unwrap()
        .task
        .as_ref()
        .unwrap()
        .clone();
    let second_scope = GoalExecutionScope::new(
        second_task.id,
        second_ingress.session_key().durable_id(),
        second_task.execution_epoch,
    )
    .unwrap();
    let mut holder = Box::pin(first.ordinary.boundary.clone().lock_owned());
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut holder)
            .await
            .is_err()
    );
    let restarting = Arc::clone(&coordinator);
    let mut restart = zeroclaw_spawn::spawn!(async move { restarting.quiesce_for_restart().await });
    tokio::time::timeout(Duration::from_secs(1), async {
        while first_registry
            .0
            .lock()
            .unwrap()
            .task
            .as_ref()
            .unwrap()
            .status
            != TaskStatus::Paused
            || second_registry
                .0
                .lock()
                .unwrap()
                .task
                .as_ref()
                .unwrap()
                .status
                != TaskStatus::Paused
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    first.script.release.add_permits(1);
    second.script.release.add_permits(1);
    let holder = tokio::time::timeout(Duration::from_secs(1), holder)
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while second_supervisor.owns_scope(&second_scope).await {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("host two must drain while host one's delivery is blocked");
    assert_eq!(
        first_registry
            .0
            .lock()
            .unwrap()
            .task
            .as_ref()
            .unwrap()
            .status,
        TaskStatus::Failed
    );
    assert_eq!(
        second_registry
            .0
            .lock()
            .unwrap()
            .task
            .as_ref()
            .unwrap()
            .status,
        TaskStatus::Paused
    );
    assert_eq!(
        second_registry
            .0
            .lock()
            .unwrap()
            .goal
            .as_ref()
            .unwrap()
            .pause_reason,
        Some(zeroclaw_goal::GoalPauseReason::DaemonRestart)
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut restart)
            .await
            .is_err()
    );
    assert_eq!(first.script.notices.load(Ordering::SeqCst), 0);
    drop(holder);
    tokio::time::timeout(Duration::from_secs(1), restart)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(first.script.notices.load(Ordering::SeqCst), 1);
    assert_eq!(second.script.notices.load(Ordering::SeqCst), 0);
    assert!(first_supervisor.try_retire().await);
    assert!(second_supervisor.try_retire().await);
}

#[tokio::test]
async fn failed_tail_notice_is_retried_once_without_repeating_the_parked_original() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    driver.script.fail_parent.store(true, Ordering::SeqCst);
    driver.script.fail_notice_once.store(true, Ordering::SeqCst);
    registry.0.lock().unwrap().fail_finish_once = true;
    let release = hold_publication(&driver.script);
    driver.script.release.add_permits(1);
    await_publication(&driver.script).await;
    let deferred = queued(
        supervisor
            .admit_control(
                settings(),
                ingress,
                driver.clone(),
                GoalCommand::Start {
                    budget: GoalBudgetSelection::Defaults,
                    objective: "bounded retry successor".into(),
                },
            )
            .await
            .unwrap(),
        GoalQueuedControl::Start,
    );
    release.send(()).unwrap();
    let ready = tokio::time::timeout(Duration::from_secs(1), deferred.wait())
        .await
        .unwrap()
        .unwrap();
    assert_original_then_executor_notice(&driver.script);
    assert_eq!(
        driver.script.notices.load(Ordering::SeqCst),
        2,
        "one failed attempt, one successful retry"
    );
    assert_eq!(driver.script.acquisitions.load(Ordering::SeqCst), 3);
    let committed = ready.commit(settings()).await.unwrap();
    assert!(matches!(committed.response(), GoalResponse::Started(_)));
    let mut launch = committed.release();
    supervisor
        .pause_before_launch(&mut launch, zeroclaw_goal::GoalPauseReason::OperatorPaused)
        .await
        .unwrap();
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 2);
    assert_original_then_executor_notice(&driver.script);
    let state = registry.0.lock().unwrap();
    assert_eq!(
        state.finish_calls, 2,
        "one rejected finish, one durable fallback"
    );
    assert!(state.pending_settlements.is_empty());
}

#[tokio::test]
async fn published_failed_tail_is_consumed_once_and_allows_direct_successor() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    let task = registry.0.lock().unwrap().task.as_ref().unwrap().clone();
    let scope = GoalExecutionScope::new(
        task.id,
        ingress.session_key().durable_id(),
        task.execution_epoch,
    )
    .unwrap();
    driver.script.fail_parent.store(true, Ordering::SeqCst);
    driver.script.release.add_permits(1);
    assert!(
        tokio::time::timeout(Duration::from_secs(1), supervisor.drain(&scope))
            .await
            .unwrap()
            .is_err()
    );
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    assert!(!supervisor.owns_scope(&scope).await);
    let GoalControlAdmission::Direct(mut successor) = supervisor
        .admit_control(
            settings(),
            ingress,
            driver.clone(),
            GoalCommand::Start {
                budget: GoalBudgetSelection::Defaults,
                objective: "settled successor".into(),
            },
        )
        .await
        .unwrap()
    else {
        panic!("settled failure must not defer Start");
    };
    assert!(matches!(successor.response(), GoalResponse::Started(_)));
    supervisor
        .pause_before_launch(
            &mut successor,
            zeroclaw_goal::GoalPauseReason::OperatorPaused,
        )
        .await
        .unwrap();
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    assert_eq!(driver.script.acquisitions.load(Ordering::SeqCst), 1);
}

/// Exercise launch reporting through the public handoff, without admitting a
/// provider call or replacing the native failure-notice lease path.
async fn admitted_launch() -> (
    Arc<WorkerDriver>,
    Arc<Registry>,
    Arc<GoalExecutionSupervisor>,
    zeroclaw_goal::GoalExecutionSubmission,
) {
    let (ordinary, ingress) = serialized_fixture();
    let driver = Arc::new(WorkerDriver {
        ordinary,
        script: Arc::new(Script::default()),
    });
    let registry = Arc::new(Registry::default());
    let runtime = GoalRuntime::new(registry.clone());
    let supervisor = Arc::new(GoalExecutionSupervisor::new(Arc::new(
        runtime.execution_engine(Arc::new(NoPhysicalAttempts)),
    )));
    let submission = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.submit(settings(), ingress, driver.clone(), launch_command(false)),
    )
    .await
    .unwrap()
    .unwrap();
    (driver, registry, supervisor, submission)
}

#[tokio::test]
async fn launch_failure_marker_requires_committed_failure_and_one_notice() {
    let (driver, registry, supervisor, mut submission) = admitted_launch().await;
    driver.script.fail_launch_bind.store(true, Ordering::SeqCst);
    let error = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.launch_after_response(settings(), &mut submission),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(error.is::<zeroclaw_goal::GoalLaunchFailureReported>());
    assert_eq!(
        error.to_string(),
        "Goal launch failure was classified and reported"
    );
    assert!(format!("{error:#}").contains("fixture launch rebind unavailable"));
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    assert_eq!(registry.0.lock().unwrap().transitions, ["start", "finish"]);
    assert_eq!(
        registry.0.lock().unwrap().task.as_ref().unwrap().status,
        TaskStatus::Failed
    );
}

#[tokio::test]
async fn launch_failure_marker_records_a_notice_attempt_not_delivery_success() {
    let (driver, registry, supervisor, mut submission) = admitted_launch().await;
    driver.script.fail_launch_bind.store(true, Ordering::SeqCst);
    driver.script.fail_notice.store(true, Ordering::SeqCst);
    let error = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.launch_after_response(settings(), &mut submission),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(error.is::<zeroclaw_goal::GoalLaunchFailureReported>());
    assert!(format!("{error:#}").contains("fixture launch rebind unavailable"));
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    assert_eq!(registry.0.lock().unwrap().transitions, ["start", "finish"]);
}

#[tokio::test]
async fn launch_failure_for_noncurrent_execution_has_no_marker_or_notice() {
    let (driver, registry, supervisor, mut submission) = admitted_launch().await;
    // A competing durable fence wins before the launch bind fails.
    registry
        .0
        .lock()
        .unwrap()
        .task
        .as_mut()
        .unwrap()
        .execution_epoch += 1;
    driver.script.fail_launch_bind.store(true, Ordering::SeqCst);
    let error = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.launch_after_response(settings(), &mut submission),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(!error.is::<zeroclaw_goal::GoalLaunchFailureReported>());
    assert!(format!("{error:#}").contains("fixture launch rebind unavailable"));
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 0);
    assert_eq!(registry.0.lock().unwrap().transitions, ["start"]);
}

#[tokio::test]
async fn launch_failure_reconciliation_error_remains_unmarked() {
    let (driver, registry, supervisor, mut submission) = admitted_launch().await;
    registry.0.lock().unwrap().fail_finish = true;
    driver.script.fail_launch_bind.store(true, Ordering::SeqCst);
    let error = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.launch_after_response(settings(), &mut submission),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(!error.is::<zeroclaw_goal::GoalLaunchFailureReported>());
    assert!(format!("{error:#}").contains("fixture lifecycle settlement unavailable"));
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn transition_pending_drain_error_remains_unmarked() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    let mut submission = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.submit(settings(), ingress, driver.clone(), GoalCommand::Cancel),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(matches!(
        submission.response(),
        GoalResponse::TransitionPending(_)
    ));
    // Remove the extension only after the fence is admitted, so its real
    // predecessor drain fails instead of constructing a malformed command.
    registry.0.lock().unwrap().goal.take();
    driver.script.release.add_permits(1);
    let error = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.launch_after_response(settings(), &mut submission),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(!error.is::<zeroclaw_goal::GoalLaunchFailureReported>());
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn launch_failure_outer_reconciliation_also_marks_a_reported_failure() {
    let (_, registry, supervisor, mut submission) = admitted_launch().await;
    {
        let mut state = registry.0.lock().unwrap();
        state.fail_lookup_at = Some(state.session_lookups + 1);
    }
    // The launch's currentness check fails before either inner reconciliation;
    // only the outer recovery commits and attempts the lifecycle notice.
    let error = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.launch_after_response(settings(), &mut submission),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(error.is::<zeroclaw_goal::GoalLaunchFailureReported>());
    assert!(format!("{error:#}").contains("fixture session lookup unavailable"));
    assert_eq!(registry.0.lock().unwrap().transitions, ["start", "finish"]);
}

#[tokio::test]
async fn launch_failure_outer_reconciliation_error_overrides_the_inner_marker() {
    let (driver, registry, supervisor, mut submission) = admitted_launch().await;
    {
        let mut state = registry.0.lock().unwrap();
        // Inner currentness, two notice checks, then outer reconciliation.
        state.fail_lookup_at = Some(state.session_lookups + 4);
    }
    driver.script.fail_launch_bind.store(true, Ordering::SeqCst);
    let error = tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.launch_after_response(settings(), &mut submission),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(!error.is::<zeroclaw_goal::GoalLaunchFailureReported>());
    assert!(format!("{error:#}").contains("fixture session lookup unavailable"));
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    assert_eq!(registry.0.lock().unwrap().transitions, ["start", "finish"]);
}

#[tokio::test]
async fn custody_completed_worker_remains_owned_until_exact_drain() {
    let (driver, _, registry, supervisor) = running_worker().await;
    let scope = {
        let state = registry.0.lock().unwrap();
        let task = state.task.as_ref().unwrap();
        GoalExecutionScope::new(
            task.id.clone(),
            task.session_id.clone().unwrap(),
            task.execution_epoch,
        )
        .unwrap()
    };
    assert!(!retire_without_waiting(&supervisor).await);
    driver.script.release.add_permits(1);
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if registry.0.lock().unwrap().task.as_ref().unwrap().status == TaskStatus::Completed {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(!retire_without_waiting(&supervisor).await);
    tokio::time::timeout(Duration::from_secs(1), supervisor.drain(&scope))
        .await
        .unwrap()
        .unwrap();
    assert!(retire_without_waiting(&supervisor).await);
}

#[tokio::test]
async fn worker_pause_and_budget_wait_for_published_parent_and_release_before_drain() {
    for (command, kind) in [
        (
            GoalCommand::Pause { immediate: false },
            GoalQueuedControl::Pause,
        ),
        (
            GoalCommand::SetBudget(GoalBudgetSelection::Unlimited),
            GoalQueuedControl::Budget,
        ),
    ] {
        let (driver, ingress, registry, supervisor) = running_worker().await;
        let deferred = queued(
            supervisor
                .admit_control(settings(), ingress.clone(), driver.clone(), command)
                .await
                .unwrap(),
            kind,
        );
        assert_eq!(registry.0.lock().unwrap().transitions, ["start"]);
        let wait = zeroclaw_spawn::spawn!(async move { deferred.wait().await });
        tokio::task::yield_now().await;
        assert!(!wait.is_finished());
        driver.script.release.add_permits(1);
        let ready = tokio::time::timeout(Duration::from_secs(1), wait)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(driver.script.published.load(Ordering::SeqCst), 1);
        assert_eq!(driver.script.verified.load(Ordering::SeqCst), 0);
        let committed = tokio::time::timeout(Duration::from_secs(1), ready.commit(settings()))
            .await
            .expect("commit must not drain the worker waiting on this ticket")
            .unwrap();
        assert!(driver.ordinary.boundary.clone().try_lock_owned().is_err());
        assert_eq!(driver.script.verified.load(Ordering::SeqCst), 0);
        let mut submission = committed.release();
        tokio::time::timeout(
            Duration::from_secs(1),
            supervisor.launch_after_response(settings(), &mut submission),
        )
        .await
        .expect("release must precede predecessor drain")
        .unwrap();
        if kind == GoalQueuedControl::Pause {
            assert_eq!(driver.script.verified.load(Ordering::SeqCst), 0);
            assert_eq!(
                registry.0.lock().unwrap().task.as_ref().unwrap().status,
                TaskStatus::Paused
            );
        } else {
            tokio::time::timeout(Duration::from_secs(1), async {
                while registry.0.lock().unwrap().task.as_ref().unwrap().status
                    != TaskStatus::Completed
                {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert_eq!(driver.script.verified.load(Ordering::SeqCst), 1);
        }
    }
}

#[tokio::test]
async fn running_worker_rejects_start_and_resume_without_creating_intents() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    for command in [
        GoalCommand::Start {
            budget: GoalBudgetSelection::Defaults,
            objective: "Another goal".into(),
        },
        GoalCommand::Resume {
            response: Some("Choice A\nRetain the limits".into()),
        },
    ] {
        let GoalControlAdmission::Direct(submission) = supervisor
            .admit_control(settings(), ingress.clone(), driver.clone(), command)
            .await
            .unwrap()
        else {
            panic!("invalid controls must not queue on the active worker")
        };
        assert!(matches!(submission.response(), GoalResponse::AlreadyActive));
        drop(submission);
    }
    assert!(registry.0.lock().unwrap().intents.is_empty());
    let scope = {
        let state = registry.0.lock().unwrap();
        let task = state.task.as_ref().unwrap();
        GoalExecutionScope::new(
            task.id.clone(),
            task.session_id.clone().unwrap(),
            task.execution_epoch,
        )
        .unwrap()
    };
    driver.script.release.add_permits(1);
    tokio::time::timeout(Duration::from_secs(1), supervisor.drain(&scope))
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn worker_controls_apply_fifo_without_admitting_a_verifier_between_them() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    let first = queued(
        supervisor
            .admit_control(
                settings(),
                ingress.clone(),
                driver.clone(),
                GoalCommand::SetBudget(GoalBudgetSelection::Unlimited),
            )
            .await
            .unwrap(),
        GoalQueuedControl::Budget,
    );
    let second = queued(
        supervisor
            .admit_control(
                settings(),
                ingress.clone(),
                driver.clone(),
                GoalCommand::Pause { immediate: false },
            )
            .await
            .unwrap(),
        GoalQueuedControl::Pause,
    );
    let second_wait = zeroclaw_spawn::spawn!(async move { second.wait().await });
    driver.script.release.add_permits(1);
    let first_ready = tokio::time::timeout(Duration::from_secs(1), first.wait())
        .await
        .unwrap()
        .unwrap();
    assert!(
        !second_wait.is_finished(),
        "registration order, not waiter order, owns FIFO"
    );
    let mut first_submission = first_ready.commit(settings()).await.unwrap().release();
    supervisor
        .launch_after_response(settings(), &mut first_submission)
        .await
        .unwrap();
    let second_ready = tokio::time::timeout(Duration::from_secs(1), second_wait)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(driver.script.verified.load(Ordering::SeqCst), 0);
    let mut second_submission = second_ready.commit(settings()).await.unwrap().release();
    tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.launch_after_response(settings(), &mut second_submission),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(
        registry.0.lock().unwrap().transitions,
        ["start", "budget", "pause"]
    );
    assert_eq!(driver.script.verified.load(Ordering::SeqCst), 0);
    assert!(
        registry
            .0
            .lock()
            .unwrap()
            .intents
            .iter()
            .all(|(_, state)| *state == IntentState::Settled(GoalControlIntentSettlement::Applied))
    );
}

#[tokio::test]
async fn immediate_cancel_closes_a_scheduled_control_before_draining_the_worker() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    let deferred = queued(
        supervisor
            .admit_control(
                settings(),
                ingress.clone(),
                driver.clone(),
                GoalCommand::Pause { immediate: false },
            )
            .await
            .unwrap(),
        GoalQueuedControl::Pause,
    );
    driver.script.release.add_permits(1);
    tokio::time::timeout(Duration::from_secs(1), async {
        while driver.script.published.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    // Keep the deferred owner alive. Cancel must close its issuing checkpoint;
    // waiting for that owner's release while holding the command lock deadlocks.
    let GoalControlAdmission::Direct(mut cancellation) = supervisor
        .admit_control(settings(), ingress, driver.clone(), GoalCommand::Cancel)
        .await
        .unwrap()
    else {
        panic!("cancel must never queue")
    };
    tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.launch_after_response(settings(), &mut cancellation),
    )
    .await
    .expect("immediate fence must close the checkpoint before drain")
    .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_secs(1), deferred.wait())
            .await
            .unwrap()
            .is_err()
    );
    wait_for_settlement(&registry).await;
    assert_eq!(
        registry.0.lock().unwrap().task.as_ref().unwrap().status,
        TaskStatus::Cancelled
    );
    assert_eq!(driver.script.verified.load(Ordering::SeqCst), 0);
    assert_eq!(registry.0.lock().unwrap().transitions, ["start", "finish"]);
    assert!(matches!(
        &registry.0.lock().unwrap().intents[0].1,
        IntentState::Settled(GoalControlIntentSettlement::Superseded { .. })
    ));
}

#[tokio::test]
async fn dropping_a_worker_control_abandons_the_intent_and_releases_execution() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    let deferred = queued(
        supervisor
            .admit_control(
                settings(),
                ingress,
                driver.clone(),
                GoalCommand::Pause { immediate: false },
            )
            .await
            .unwrap(),
        GoalQueuedControl::Pause,
    );
    let scope = {
        let state = registry.0.lock().unwrap();
        let task = state.task.as_ref().unwrap();
        GoalExecutionScope::new(
            task.id.clone(),
            task.session_id.clone().unwrap(),
            task.execution_epoch,
        )
        .unwrap()
    };
    drop(deferred);
    wait_for_settlement(&registry).await;
    driver.script.release.add_permits(1);
    tokio::time::timeout(Duration::from_secs(1), supervisor.drain(&scope))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        registry.0.lock().unwrap().task.as_ref().unwrap().status,
        TaskStatus::Completed
    );
    assert_eq!(driver.script.verified.load(Ordering::SeqCst), 1);
    assert!(matches!(
        &registry.0.lock().unwrap().intents[0].1,
        IntentState::Settled(GoalControlIntentSettlement::Abandoned { .. })
    ));
}

#[tokio::test]
async fn settled_pause_and_cancel_do_not_publish_failure_output() {
    for (command, status) in [
        (GoalCommand::Pause { immediate: true }, TaskStatus::Paused),
        (GoalCommand::Cancel, TaskStatus::Cancelled),
    ] {
        let (driver, ingress, registry, supervisor) = running_worker().await;
        let mut submission = supervisor
            .submit(settings(), ingress, driver.clone(), command)
            .await
            .unwrap();
        driver.script.release.add_permits(1);
        tokio::time::timeout(
            Duration::from_secs(1),
            supervisor.launch_after_response(settings(), &mut submission),
        )
        .await
        .expect("settled immediate controls must retire their worker")
        .unwrap();
        assert_eq!(
            registry.0.lock().unwrap().task.as_ref().unwrap().status,
            status
        );
        assert!(driver.script.events.lock().unwrap().is_empty());
        assert_eq!(driver.script.notices.load(Ordering::SeqCst), 0);
        assert_eq!(driver.script.acquisitions.load(Ordering::SeqCst), 1);
        assert_eq!(driver.script.verified.load(Ordering::SeqCst), 0);
        assert!(supervisor.try_retire().await);
    }
}

#[tokio::test]
async fn foreign_task_epoch_and_session_drains_preserve_the_exact_running_owner() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    seed_unsettled_call(&registry);
    let original = registry.0.lock().unwrap().task.as_ref().unwrap().clone();
    let snapshot = serde_json::to_value(&original).unwrap();
    for scope in [
        GoalExecutionScope::new(
            "foreign-task",
            original.session_id.clone().unwrap(),
            original.execution_epoch,
        )
        .unwrap(),
        GoalExecutionScope::new(
            original.id.clone(),
            original.session_id.clone().unwrap(),
            original.execution_epoch + 2,
        )
        .unwrap(),
        GoalExecutionScope::new(
            original.id.clone(),
            "foreign-session",
            original.execution_epoch,
        )
        .unwrap(),
    ] {
        assert!(
            tokio::time::timeout(Duration::from_secs(1), supervisor.drain(&scope))
                .await
                .expect("a foreign drain must not wait on the admitted parent")
                .is_err()
        );
        let state = registry.0.lock().unwrap();
        assert_eq!(
            serde_json::to_value(state.task.as_ref().unwrap()).unwrap(),
            snapshot
        );
        assert!(state.pending_settlements.is_empty());
        assert_eq!(
            state.goal.as_ref().unwrap().pending_call_id.as_deref(),
            Some("fixture-unsettled-operation")
        );
        assert_eq!(
            state.goal.as_ref().unwrap().pending_call_epoch,
            Some(original.execution_epoch)
        );
        assert!(driver.script.events.lock().unwrap().is_empty());
        assert_eq!(driver.script.acquisitions.load(Ordering::SeqCst), 1);
    }
    let mut cancel = supervisor
        .submit(settings(), ingress, driver.clone(), GoalCommand::Cancel)
        .await
        .unwrap();
    driver.script.release.add_permits(1);
    supervisor
        .launch_after_response(settings(), &mut cancel)
        .await
        .unwrap();
    assert!(driver.script.events.lock().unwrap().is_empty());
    assert!(supervisor.try_retire().await);
}

#[tokio::test]
async fn stale_driver_incarnation_rejects_failure_delivery_without_output() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    driver.script.fail_parent.store(true, Ordering::SeqCst);
    let mut pause = supervisor
        .submit(
            settings(),
            ingress,
            driver.clone(),
            GoalCommand::Pause { immediate: true },
        )
        .await
        .unwrap();
    // Generation checks belong to the ordinary driver. Replace its incarnation
    // after the admitted call, before any publication can acquire a fresh lease.
    driver
        .script
        .reject_execution_incarnation
        .store(true, Ordering::SeqCst);
    driver.script.release.add_permits(1);
    assert!(matches!(
        tokio::time::timeout(
            Duration::from_secs(1),
            supervisor.launch_after_response(settings(), &mut pause)
        )
        .await
        .unwrap()
        .unwrap(),
        Some(GoalResponse::Terminal(_))
    ));
    {
        let state = registry.0.lock().unwrap();
        assert_eq!(state.task.as_ref().unwrap().status, TaskStatus::Failed);
        assert_eq!(state.terminal_reason.as_deref(), Some("executor_failed"));
        assert!(state.pending_settlements.is_empty());
    }
    assert!(driver.script.events.lock().unwrap().is_empty());
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 0);
    assert_eq!(driver.script.acquisitions.load(Ordering::SeqCst), 1);
    assert!(supervisor.try_retire().await);
}

#[tokio::test]
async fn completed_predecessor_allows_direct_start_without_failure_delivery() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    driver.script.release.add_permits(1);
    tokio::time::timeout(Duration::from_secs(1), async {
        while registry.0.lock().unwrap().task.as_ref().unwrap().status != TaskStatus::Completed {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let predecessor = registry.0.lock().unwrap().task.as_ref().unwrap().id.clone();
    let GoalControlAdmission::Direct(mut successor) = supervisor
        .admit_control(
            settings(),
            ingress,
            driver.clone(),
            GoalCommand::Start {
                budget: GoalBudgetSelection::Defaults,
                objective: "next bounded fixture".into(),
            },
        )
        .await
        .unwrap()
    else {
        panic!("a completed predecessor with a free boundary must not queue Start");
    };
    assert!(matches!(successor.response(), GoalResponse::Started(_)));
    {
        let state = registry.0.lock().unwrap();
        assert_ne!(state.task.as_ref().unwrap().id, predecessor);
        assert!(state.intents.is_empty());
    }
    supervisor
        .pause_before_launch(
            &mut successor,
            zeroclaw_goal::GoalPauseReason::OperatorPaused,
        )
        .await
        .unwrap();
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    assert_eq!(driver.script.acquisitions.load(Ordering::SeqCst), 1);
    assert_eq!(driver.script.published.load(Ordering::SeqCst), 1);
    assert_eq!(driver.script.verified.load(Ordering::SeqCst), 1);
    assert!(matches!(
        driver.script.events.lock().unwrap().as_slice(),
        [FailureEvent::Notice(GoalExecutionNotice::Completed)]
    ));
}

#[tokio::test]
async fn completed_worker_falls_back_to_the_ordinary_boundary_without_duplicate_intents() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    driver.script.release.add_permits(1);
    tokio::time::timeout(Duration::from_secs(1), async {
        while registry.0.lock().unwrap().task.as_ref().unwrap().status != TaskStatus::Completed {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let ordinary = tokio::time::timeout(
        Duration::from_secs(1),
        driver.ordinary.boundary.clone().lock_owned(),
    )
    .await
    .unwrap();
    assert_eq!(driver.script.notices.load(Ordering::SeqCst), 1);
    let predecessor = registry.0.lock().unwrap().task.as_ref().unwrap().id.clone();
    let deferred = queued(
        supervisor
            .admit_control(
                settings(),
                ingress,
                driver.clone(),
                GoalCommand::Start {
                    budget: GoalBudgetSelection::Defaults,
                    objective: "Complete the next bounded fixture".into(),
                },
            )
            .await
            .unwrap(),
        GoalQueuedControl::Start,
    );
    {
        let state = registry.0.lock().unwrap();
        assert_eq!(state.intents.len(), 1);
        assert_eq!(
            state.intents[0].0.predecessor_task_id.as_deref(),
            Some(predecessor.as_str())
        );
        assert_eq!(state.task.as_ref().unwrap().status, TaskStatus::Completed);
    }
    drop(ordinary);
    let ready = tokio::time::timeout(Duration::from_secs(1), deferred.wait())
        .await
        .unwrap()
        .unwrap();
    let mut submission = ready.commit(settings()).await.unwrap().release();
    assert!(matches!(submission.response(), GoalResponse::Started(_)));
    // This case proves admission fallback, not a second provider execution.
    supervisor
        .pause_before_launch(
            &mut submission,
            zeroclaw_goal::GoalPauseReason::OperatorPaused,
        )
        .await
        .unwrap();
    assert_eq!(registry.0.lock().unwrap().intents.len(), 1);
    assert_eq!(driver.script.published.load(Ordering::SeqCst), 1);
    assert_eq!(driver.script.verified.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn pause_registered_during_verification_precedes_the_verifiers_completion_transition() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    driver.script.hold_verifier.store(true, Ordering::SeqCst);
    driver.script.release.add_permits(1);
    tokio::time::timeout(
        Duration::from_secs(1),
        driver.script.verifier_entered.acquire(),
    )
    .await
    .unwrap()
    .unwrap()
    .forget();
    let deferred = queued(
        supervisor
            .admit_control(
                settings(),
                ingress,
                driver.clone(),
                GoalCommand::Pause { immediate: false },
            )
            .await
            .unwrap(),
        GoalQueuedControl::Pause,
    );
    assert_eq!(
        registry.0.lock().unwrap().task.as_ref().unwrap().status,
        TaskStatus::Running
    );
    assert_eq!(registry.0.lock().unwrap().transitions, ["start"]);
    let wait = zeroclaw_spawn::spawn!(async move { deferred.wait().await });
    tokio::task::yield_now().await;
    assert!(!wait.is_finished());
    driver.script.verifier_release.add_permits(1);
    let ready = tokio::time::timeout(Duration::from_secs(1), wait)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(
        registry.0.lock().unwrap().task.as_ref().unwrap().status,
        TaskStatus::Running
    );
    let committed = tokio::time::timeout(Duration::from_secs(1), ready.commit(settings()))
        .await
        .unwrap()
        .unwrap();
    let mut submission = committed.release();
    tokio::time::timeout(
        Duration::from_secs(1),
        supervisor.launch_after_response(settings(), &mut submission),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(registry.0.lock().unwrap().transitions, ["start", "pause"]);
    assert_eq!(
        registry.0.lock().unwrap().task.as_ref().unwrap().status,
        TaskStatus::Paused
    );
    assert_eq!(driver.script.published.load(Ordering::SeqCst), 1);
    assert_eq!(driver.script.verified.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn dropping_ready_without_commit_releases_the_ticket_and_abandons_the_intent() {
    let (driver, ingress, registry, supervisor) = running_worker().await;
    let deferred = queued(
        supervisor
            .admit_control(
                settings(),
                ingress,
                driver.clone(),
                GoalCommand::Pause { immediate: false },
            )
            .await
            .unwrap(),
        GoalQueuedControl::Pause,
    );
    let scope = {
        let state = registry.0.lock().unwrap();
        let task = state.task.as_ref().unwrap();
        GoalExecutionScope::new(
            task.id.clone(),
            task.session_id.clone().unwrap(),
            task.execution_epoch,
        )
        .unwrap()
    };
    driver.script.release.add_permits(1);
    let ready = tokio::time::timeout(Duration::from_secs(1), deferred.wait())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(driver.script.verified.load(Ordering::SeqCst), 0);
    drop(ready);
    wait_for_settlement(&registry).await;
    tokio::time::timeout(Duration::from_secs(1), supervisor.drain(&scope))
        .await
        .unwrap()
        .unwrap();
    assert!(driver.ordinary.command.try_lock().is_ok());
    assert_eq!(
        registry.0.lock().unwrap().task.as_ref().unwrap().status,
        TaskStatus::Completed
    );
    assert!(matches!(
        &registry.0.lock().unwrap().intents[0].1,
        IntentState::Settled(GoalControlIntentSettlement::Abandoned { .. })
    ));
}
