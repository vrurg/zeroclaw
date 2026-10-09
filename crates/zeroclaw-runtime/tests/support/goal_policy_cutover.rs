//! Real-store tests of abandoned live cutovers, without a process-global plane.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use async_trait::async_trait;
use tokio::{
    sync::Notify,
    time::{Duration, timeout},
};
use zeroclaw_commands::goal::{GoalBudgetLimits, GoalBudgetSelection, GoalCommand};
use zeroclaw_config::{cost::CostTracker, schema::Config};
use zeroclaw_goal::{
    GoalExecutionNotice, GoalExecutionRestartCoordinator, GoalExecutionScope, GoalHostSettings,
    GoalIngressContext, GoalIngressPrincipal, GoalOperationScope, GoalParentTurn,
    GoalParentTurnCheckpoint, GoalParentTurnResult, GoalResponse, GoalRuntime, GoalSessionBinding,
    GoalSessionDriver, GoalSessionExecutionLease, GoalSessionKey, GoalSessionLease,
    GoalVerifierResult, GoalVerifierTurn,
};

use crate::{
    control_plane::{GoalTaskRegistry, SqliteTaskStore, TaskStatus},
    goal_accounting::{ModelProviderGoalUsagePricer, RuntimeGoalOperationAccountingFactory},
};

struct CutoverDriver {
    binding: GoalSessionBinding,
    started: Arc<Notify>,
    release: Arc<Notify>,
    verifier_calls: Arc<AtomicUsize>,
    notices: Arc<AtomicUsize>,
}

struct CutoverLease {
    session_key: GoalSessionKey,
    started: Arc<Notify>,
    release: Arc<Notify>,
    verifier_calls: Arc<AtomicUsize>,
    notices: Arc<AtomicUsize>,
}

#[async_trait]
impl GoalSessionDriver for CutoverDriver {
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
        _checkpoint: &GoalParentTurnCheckpoint,
    ) -> anyhow::Result<Box<dyn GoalSessionExecutionLease>> {
        Ok(Box::new(CutoverLease {
            session_key: self.binding.session_key().clone(),
            started: self.started.clone(),
            release: self.release.clone(),
            verifier_calls: self.verifier_calls.clone(),
            notices: self.notices.clone(),
        }))
    }
}

#[async_trait]
impl GoalSessionExecutionLease for CutoverLease {
    fn session_key(&self) -> &GoalSessionKey {
        &self.session_key
    }

    fn canonical_history(&self) -> anyhow::Result<Vec<zeroclaw_api::model_provider::ChatMessage>> {
        Ok(Vec::new())
    }

    async fn run_parent_turn(
        &mut self,
        _operation: &GoalOperationScope,
        _turn: GoalParentTurn,
    ) -> anyhow::Result<GoalParentTurnResult> {
        self.started.notify_one();
        self.release.notified().await;
        Ok(GoalParentTurnResult::from_continuation_prefix(
            "settled candidate".into(),
            Vec::new(),
        ))
    }

    async fn run_verifier(
        &mut self,
        _operation: &GoalOperationScope,
        _turn: GoalVerifierTurn,
    ) -> anyhow::Result<GoalVerifierResult> {
        self.verifier_calls.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("a revoked worker must not admit a verifier")
    }

    async fn append_parent_candidate(&mut self, _candidate: String) -> anyhow::Result<()> {
        Ok(())
    }

    async fn publish_goal_notice(&mut self, _notice: GoalExecutionNotice) -> anyhow::Result<()> {
        self.notices.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test]
async fn dropped_live_policy_cutover_reopens_admission_and_drains_cancelled_worker() {
    timeout(Duration::from_secs(5), async {
        let directory = tempfile::tempdir().unwrap();
        let store = Arc::new(SqliteTaskStore::new(directory.path()).unwrap());
        let registry = store.clone() as Arc<dyn GoalTaskRegistry>;
        let runtime = GoalRuntime::new(registry.clone());
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
        let factory = Arc::new(
            RuntimeGoalOperationAccountingFactory::new(
                registry.clone(),
                tracker,
                "main",
                Arc::new(ModelProviderGoalUsagePricer::new(Arc::default())),
            )
            .unwrap(),
        );
        let coordinator = Arc::new(GoalExecutionRestartCoordinator::new());
        let supervisor = coordinator
            .new_supervisor(Arc::new(runtime.execution_engine(factory)))
            .await;
        let ingress = GoalIngressContext::trusted(
            GoalSessionKey::zero_code("abandoned-cutover").unwrap(),
            "main",
            "zerocode:abandoned-cutover",
            GoalIngressPrincipal::ZeroCode {
                tui_id: "test-owner".into(),
            },
        )
        .unwrap();
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let verifier_calls = Arc::new(AtomicUsize::new(0));
        let notices = Arc::new(AtomicUsize::new(0));
        let driver = Arc::new(CutoverDriver {
            binding: GoalSessionBinding::new(ingress.session_key().clone()),
            started: started.clone(),
            release: release.clone(),
            verifier_calls: verifier_calls.clone(),
            notices: notices.clone(),
        });
        let settings = GoalHostSettings::new(
            true,
            GoalBudgetLimits {
                token_limit: Some(100),
                cost_limit_usd: Some(1.0),
            },
            42,
            "test-boot",
        )
        .unwrap();
        let mut execution = supervisor
            .submit(
                settings.clone(),
                ingress.clone(),
                driver.clone(),
                GoalCommand::Start {
                    budget: GoalBudgetSelection::Defaults,
                    objective: "finish the bounded repair".into(),
                },
            )
            .await
            .unwrap();
        supervisor
            .launch_after_response(settings.clone(), &mut execution)
            .await
            .unwrap();
        started.notified().await;
        let before = registry
            .current_goal_for_session(&ingress.session_key().durable_id())
            .await
            .unwrap()
            .unwrap();
        let scope = GoalExecutionScope::new(
            before.id.clone(),
            ingress.session_key().durable_id(),
            before.execution_epoch,
        )
        .unwrap();
        let mut config = Config::default();
        config.goal.enabled = true;
        let mut prospective = config.clone();
        prospective.goal.enabled = false;
        let live = zeroclaw_config::live::LiveConfig::new(config);
        let generation = coordinator.policy_generation();
        let cutover = super::begin_live_goal_policy_cutover_with(
            coordinator.clone(),
            registry.clone(),
            &live.handle(),
            &prospective,
        )
        .await
        .unwrap();
        let cancelled = registry
            .current_goal_for_session(scope.session_id())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(cancelled.status, TaskStatus::Cancelled);
        assert_eq!(cancelled.execution_epoch, before.execution_epoch + 1);
        assert_eq!(
            registry
                .terminal_reason_for_session_goal(scope.task_id(), scope.session_id())
                .await
                .unwrap()
                .as_deref(),
            Some("policy_revoked")
        );
        assert!(
            live.snapshot().goal.enabled,
            "the abandoned request has not swapped config"
        );
        assert_eq!(coordinator.policy_generation(), generation);
        assert!(
            supervisor.owns_scope(&scope).await,
            "an admitted operation must not be interrupted"
        );

        drop(cutover);
        loop {
            if coordinator.policy_generation() > generation {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(supervisor.owns_scope(&scope).await);
        release.notify_one();
        while supervisor.owns_scope(&scope).await {
            tokio::task::yield_now().await;
        }
        assert_eq!(verifier_calls.load(Ordering::SeqCst), 0);
        assert_eq!(notices.load(Ordering::SeqCst), 0);
        let settled = registry
            .current_goal_for_session(scope.session_id())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(settled.status, TaskStatus::Cancelled);
        assert_eq!(settled.execution_epoch, cancelled.execution_epoch);

        // Successful admission, not just a counter increment, proves the gate reopened.
        let successor = supervisor
            .submit(
                settings.clone(),
                ingress.clone(),
                driver.clone(),
                GoalCommand::Start {
                    budget: GoalBudgetSelection::Defaults,
                    objective: "start a new bounded repair".into(),
                },
            )
            .await
            .unwrap();
        assert!(matches!(successor.response(), GoalResponse::Started(_)));
        let cleanup = supervisor
            .submit(settings, ingress, driver, GoalCommand::Cancel)
            .await
            .unwrap();
        assert!(matches!(cleanup.response(), GoalResponse::Cancelled(_)));
        drop(successor);
    })
    .await
    .expect("abandonment must reopen admission and eventually drain the exact worker");
}
