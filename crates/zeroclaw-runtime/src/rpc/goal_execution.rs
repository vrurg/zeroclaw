//! RPC-owned live-session adapter for the transport-neutral Goal executor.
//!
//! This module deliberately owns only the RPC session queue, session binding,
//! and canonical history append. Durable Goal lifecycle and accounting remain
//! in `goal_mode`; Matrix will supply the same adapter contract from its own
//! crate without creating a second executor.

use std::sync::Arc;

use anyhow::{Context as _, Result};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use zeroclaw_api::model_provider::ChatMessage;
use zeroclaw_config::schema::Config;

use crate::{
    agent::agent::Agent,
    goal_mode::{
        GoalExecutionPreflight, GoalExecutionScope, GoalOperationReservation,
        execution::{preflight_reservation, run_isolated_verifier},
        host::{
            GoalHostModelBinding, GoalHostPreflightError, GoalParentTurn, GoalSessionAdapter,
            GoalSessionBinding, GoalSessionLease, GoalVerifiedCandidate, GoalVerifierTurn,
        },
    },
    rpc::session::SessionStore,
};

/// Adapter over the daemon's existing RPC session authority. It has no Goal
/// records or provider factory of its own, which keeps it movable with the RPC
/// surface when the runtime crate is decomposed.
pub struct RpcGoalSessionAdapter {
    sessions: Arc<SessionStore>,
    config: Arc<Config>,
}

impl RpcGoalSessionAdapter {
    pub fn new(sessions: Arc<SessionStore>, config: Arc<Config>) -> Self {
        Self { sessions, config }
    }

    async fn agent_for_binding(
        &self,
        binding: &GoalSessionBinding,
    ) -> Result<Arc<tokio::sync::Mutex<Agent>>> {
        if self
            .sessions
            .get_agent_alias(binding.session_id())
            .await
            .as_deref()
            != Some(binding.agent())
        {
            anyhow::bail!("goal RPC session binding is unavailable");
        }
        self.sessions
            .get_agent(binding.session_id())
            .await
            .context("goal RPC session disappeared")
    }
}

struct RpcGoalSessionLease {
    _queue_guard: zeroclaw_infra::session_queue::SessionGuard,
    agent: tokio::sync::OwnedMutexGuard<Agent>,
    config: Arc<Config>,
}

#[async_trait]
impl GoalSessionAdapter for RpcGoalSessionAdapter {
    fn owns(&self, binding: &GoalSessionBinding) -> bool {
        matches!(binding.route(), "rpc" | "zerocode")
    }

    async fn preflight(
        &self,
        binding: &GoalSessionBinding,
        request: &GoalExecutionPreflight,
    ) -> std::result::Result<GoalOperationReservation, GoalHostPreflightError> {
        let agent = self
            .agent_for_binding(binding)
            .await
            .map_err(|_| GoalHostPreflightError::BindingUnavailable)?;
        let (provider_ref, model) = agent.lock().await.goal_model_binding();
        crate::agent::agent::build_session_model_provider(
            &self.config,
            &self.config.goal.verifier.provider,
            self.config.goal.verifier.model.as_deref(),
        )
        .map_err(|_| GoalHostPreflightError::VerifierUnavailable)?;
        preflight_reservation(
            &self.config,
            request,
            &GoalHostModelBinding {
                provider_ref,
                model,
            },
        )
        .map_err(|_| {
            if request.cost_limit_usd.is_some() {
                GoalHostPreflightError::PricingUnavailable
            } else {
                GoalHostPreflightError::ReservationUnavailable
            }
        })
    }

    async fn model_binding(&self, binding: &GoalSessionBinding) -> Result<GoalHostModelBinding> {
        let agent = self.agent_for_binding(binding).await?;
        let (provider_ref, model) = agent.lock().await.goal_model_binding();
        Ok(GoalHostModelBinding {
            provider_ref,
            model,
        })
    }

    async fn deadline(&self, binding: &GoalSessionBinding) -> Result<Option<tokio::time::Instant>> {
        self.agent_for_binding(binding).await?;
        Ok(None)
    }

    async fn acquire(
        &self,
        scope: &GoalExecutionScope,
        binding: &GoalSessionBinding,
    ) -> Result<Box<dyn GoalSessionLease>> {
        // Capture the concrete live session before queuing. The alias alone is
        // not an identity: a same-ID replacement may retain it while swapping
        // the Agent. We reject that replacement after the queue wait rather
        // than running the old Goal against the new session.
        let expected_agent = self.agent_for_binding(binding).await?;
        let queue_guard = tokio::select! {
            result = self.sessions.session_queue.acquire(scope.session_id.as_str()) => {
                result.context("goal RPC foreground lease is unavailable")?
            }
            () = scope.cancellation.cancelled() => anyhow::bail!("goal RPC foreground lease was cancelled"),
        };
        let agent = self.agent_for_binding(binding).await?;
        if !Arc::ptr_eq(&expected_agent, &agent) {
            anyhow::bail!("goal RPC session was replaced while waiting for its foreground lease");
        }
        let agent = tokio::select! {
            agent = agent.lock_owned() => agent,
            () = scope.cancellation.cancelled() => anyhow::bail!("goal RPC foreground lease was cancelled"),
        };
        Ok(Box::new(RpcGoalSessionLease {
            _queue_guard: queue_guard,
            agent,
            config: Arc::clone(&self.config),
        }))
    }
}

#[async_trait]
impl GoalSessionLease for RpcGoalSessionLease {
    fn working_history(&self) -> Result<Vec<ChatMessage>> {
        self.agent.goal_working_history()
    }

    async fn run_parent_turn(
        &mut self,
        mut turn: GoalParentTurn,
        cancellation: CancellationToken,
    ) -> Result<(String, Vec<ChatMessage>)> {
        turn.working_history
            .push(ChatMessage::user(turn.directive()));
        let response = self
            .agent
            .run_goal_parent_turn(&mut turn.working_history, cancellation)
            .await?;
        Ok((response, turn.working_history))
    }

    async fn run_verifier(
        &mut self,
        turn: GoalVerifierTurn,
        cancellation: CancellationToken,
    ) -> Result<String> {
        run_isolated_verifier(&self.config, &turn, cancellation).await
    }

    async fn append_verified_candidate(&mut self, candidate: GoalVerifiedCandidate) -> Result<()> {
        self.agent
            .append_verified_goal_candidate(candidate.candidate);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::RpcGoalSessionAdapter;
    use crate::{
        agent::{agent::Agent, dispatcher::NativeToolDispatcher},
        goal_mode::{
            GoalExecutionScope, GoalTurnKind,
            host::{GoalSessionAdapter, GoalSessionBinding},
        },
        observability::{NoopObserver, Observer},
        rpc::{
            session::{RpcSession, SessionStore},
            types::ChatMode,
        },
    };
    use tokio::sync::Semaphore;
    use tokio_util::sync::CancellationToken;

    fn test_agent() -> Agent {
        let memory_config = zeroclaw_config::schema::MemoryConfig {
            backend: "none".into(),
            ..zeroclaw_config::schema::MemoryConfig::default()
        };
        let memory = Arc::from(
            zeroclaw_memory::create_memory(&memory_config, &std::env::temp_dir(), None)
                .expect("test memory"),
        );
        Agent::builder()
            .model_provider(Box::new(StubProvider))
            .tools(Vec::new())
            .memory(memory)
            .observer(Arc::new(NoopObserver {}) as Arc<dyn Observer>)
            .tool_dispatcher(Box::new(NativeToolDispatcher))
            .workspace_dir(std::env::temp_dir())
            .build()
            .expect("test agent")
    }

    fn test_session() -> RpcSession {
        RpcSession::new(test_agent(), "main", ".", ChatMode::Chat)
    }

    fn binding() -> GoalSessionBinding {
        GoalSessionBinding::new("session".into(), "main".into(), "rpc".into(), None)
            .expect("test binding")
    }

    fn scope(binding: GoalSessionBinding, cancellation: CancellationToken) -> GoalExecutionScope {
        GoalExecutionScope {
            task_id: "goal".into(),
            session_id: "session".into(),
            binding,
            execution_epoch: 1,
            turn_kind: GoalTurnKind::Start,
            cancellation,
            deadline: None,
            operation_permit: Arc::new(Semaphore::new(1)),
        }
    }

    struct StubProvider;

    #[async_trait::async_trait]
    impl zeroclaw_providers::ModelProvider for StubProvider {
        async fn chat_with_system(
            &self,
            _: Option<&str>,
            _: &str,
            _: &str,
            _: Option<f64>,
        ) -> anyhow::Result<String> {
            Ok(String::new())
        }

        async fn chat(
            &self,
            _: zeroclaw_providers::ChatRequest<'_>,
            _: &str,
            _: Option<f64>,
        ) -> anyhow::Result<zeroclaw_providers::ChatResponse> {
            Ok(zeroclaw_providers::ChatResponse {
                text: Some("stub".into()),
                tool_calls: Vec::new(),
                usage: None,
                reasoning_content: None,
            })
        }
    }

    impl zeroclaw_api::attribution::Attributable for StubProvider {
        fn role(&self) -> zeroclaw_api::attribution::Role {
            zeroclaw_api::attribution::Role::Provider(
                zeroclaw_api::attribution::ProviderKind::Model(
                    zeroclaw_api::attribution::ModelProviderKind::Custom,
                ),
            )
        }

        fn alias(&self) -> &str {
            "stub"
        }
    }

    #[test]
    fn adapter_owns_only_rpc_and_zerocode_bindings() {
        let adapter = RpcGoalSessionAdapter::new(
            Arc::new(SessionStore::new(
                1,
                Arc::new(zeroclaw_infra::session_queue::SessionActorQueue::new(
                    1, 1, 1,
                )),
            )),
            Arc::new(zeroclaw_config::schema::Config::default()),
        );
        let binding = |route: &str| {
            GoalSessionBinding::new("session".into(), "main".into(), route.into(), None)
                .expect("test binding")
        };

        assert!(adapter.owns(&binding("rpc")));
        assert!(adapter.owns(&binding("zerocode")));
        assert!(!adapter.owns(&binding("matrix")));
    }

    #[tokio::test]
    async fn acquire_cancellation_releases_queued_foreground_request() {
        let queue = Arc::new(zeroclaw_infra::session_queue::SessionActorQueue::new(
            2, 60, 1,
        ));
        let sessions = Arc::new(SessionStore::new(1, Arc::clone(&queue)));
        sessions
            .insert("session".into(), test_session())
            .await
            .expect("insert session");
        let adapter = RpcGoalSessionAdapter::new(
            sessions,
            Arc::new(zeroclaw_config::schema::Config::default()),
        );
        let cancellation = CancellationToken::new();
        let binding = binding();
        let scope = scope(binding.clone(), cancellation.clone());
        let held = queue.acquire("session").await.expect("hold session queue");

        let acquire = tokio::spawn(async move { adapter.acquire(&scope, &binding).await });
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while queue.queue_depth("session").await != 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("goal request reaches session queue");

        cancellation.cancel();
        assert!(acquire.await.expect("acquire task joins").is_err());
        assert_eq!(queue.queue_depth("session").await, 1);
        drop(held);
    }

    #[tokio::test]
    async fn acquire_rejects_same_alias_session_replacement_after_queue_wait() {
        let queue = Arc::new(zeroclaw_infra::session_queue::SessionActorQueue::new(
            2, 60, 1,
        ));
        let sessions = Arc::new(SessionStore::new(1, Arc::clone(&queue)));
        sessions
            .insert("session".into(), test_session())
            .await
            .expect("insert original session");
        let adapter = RpcGoalSessionAdapter::new(
            Arc::clone(&sessions),
            Arc::new(zeroclaw_config::schema::Config::default()),
        );
        let binding = binding();
        let held = queue.acquire("session").await.expect("hold session queue");

        let acquire = tokio::spawn(async move {
            adapter
                .acquire(&scope(binding.clone(), CancellationToken::new()), &binding)
                .await
        });
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while queue.queue_depth("session").await != 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("goal request reaches session queue");

        assert!(sessions.remove("session").await);
        sessions
            .insert("session".into(), test_session())
            .await
            .expect("insert same-alias replacement");
        drop(held);

        assert!(acquire.await.expect("acquire task joins").is_err());
        assert_eq!(queue.queue_depth("session").await, 0);
    }

    #[tokio::test]
    async fn acquire_cancellation_releases_queue_while_waiting_for_agent_lock() {
        let queue = Arc::new(zeroclaw_infra::session_queue::SessionActorQueue::new(
            2, 60, 1,
        ));
        let sessions = Arc::new(SessionStore::new(1, Arc::clone(&queue)));
        sessions
            .insert("session".into(), test_session())
            .await
            .expect("insert session");
        let held_agent = sessions
            .get_agent("session")
            .await
            .expect("session agent")
            .lock_owned()
            .await;
        let adapter = RpcGoalSessionAdapter::new(
            Arc::clone(&sessions),
            Arc::new(zeroclaw_config::schema::Config::default()),
        );
        let cancellation = CancellationToken::new();
        let binding = binding();
        let scope = scope(binding.clone(), cancellation.clone());

        let acquire = tokio::spawn(async move { adapter.acquire(&scope, &binding).await });
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while queue.queue_depth("session").await != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("goal acquires the session queue before agent lock");

        cancellation.cancel();
        assert!(acquire.await.expect("acquire task joins").is_err());
        assert_eq!(queue.queue_depth("session").await, 0);
        drop(held_agent);
    }
}
