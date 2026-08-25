//! Transport-neutral live-session boundary for Goal execution.
//!
//! The Goal controller owns durable lifecycle and accounting. A session
//! adapter only recognizes a trusted binding, resolves the configured model,
//! and leases its own foreground state. This keeps RPC and channels outside
//! the extractable Goal execution core.

use std::sync::Arc;

use anyhow::{Result, bail};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use zeroclaw_api::model_provider::ChatMessage;

use super::{GoalExecutionPreflight, GoalExecutionScope, GoalOperationReservation, GoalTurnKind};

/// Canonical facts binding a Goal to adapter-owned live session state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalSessionBinding {
    session_id: String,
    agent: String,
    route: String,
    principal_id: Option<String>,
}

impl GoalSessionBinding {
    pub fn new(
        session_id: String,
        agent: String,
        route: String,
        principal_id: Option<String>,
    ) -> Result<Self> {
        for (field, value) in [
            ("session", &session_id),
            ("agent", &agent),
            ("route", &route),
        ] {
            if value.trim().is_empty() {
                bail!("Goal session binding has a blank {field}");
            }
        }
        Ok(Self {
            session_id,
            agent,
            route,
            principal_id: principal_id.filter(|value| !value.trim().is_empty()),
        })
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }
    pub fn agent(&self) -> &str {
        &self.agent
    }
    pub fn route(&self) -> &str {
        &self.route
    }
    pub fn principal_id(&self) -> Option<&str> {
        self.principal_id.as_deref()
    }
}

/// Configured parent model resolved by the owner of a trusted live session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalHostModelBinding {
    pub provider_ref: String,
    pub model: String,
}

/// Controller-owned input for one parent turn. The lease receives only the
/// durable declaration and its process-local working history; it never owns
/// Goal lifecycle or accounting facts.
#[derive(Debug, Clone)]
pub struct GoalParentTurn {
    pub objective: String,
    pub success_criteria: String,
    pub turn_kind: GoalTurnKind,
    pub working_history: Vec<ChatMessage>,
}

/// Isolated verifier input. Implementations must use this exact data and the
/// fixed verifier schema; session history and transport authority are absent.
#[derive(Debug, Clone)]
pub struct GoalVerifierTurn {
    pub objective: String,
    pub success_criteria: String,
    pub candidate: String,
}

/// Channel- or RPC-specific bridge supplied to the generic execution host.
///
/// Implementations must not mutate Goal lifecycle, accounting, or authority.
#[async_trait]
pub trait GoalSessionAdapter: Send + Sync {
    fn owns(&self, binding: &GoalSessionBinding) -> bool;

    /// Resolve one exact preflight reservation from trusted live session
    /// context, without beginning a model operation.
    async fn preflight(
        &self,
        binding: &GoalSessionBinding,
        request: &GoalExecutionPreflight,
    ) -> Result<GoalOperationReservation>;

    async fn model_binding(&self, binding: &GoalSessionBinding) -> Result<GoalHostModelBinding>;

    async fn acquire(
        &self,
        scope: &GoalExecutionScope,
        binding: &GoalSessionBinding,
    ) -> Result<Box<dyn GoalSessionLease>>;
}

/// One exclusive foreground lease for a Goal execution epoch.
#[async_trait]
pub trait GoalSessionLease: Send {
    fn working_history(&self) -> Result<Vec<ChatMessage>>;

    async fn run_parent_turn(
        &mut self,
        turn: GoalParentTurn,
        cancellation: CancellationToken,
    ) -> Result<(String, Vec<ChatMessage>)>;

    /// Execute the configured verifier through the normal provider path and
    /// return its raw protocol response for strict controller parsing.
    async fn run_verifier(
        &mut self,
        turn: GoalVerifierTurn,
        cancellation: CancellationToken,
    ) -> Result<String>;

    fn append_verified_candidate(&mut self, candidate: &str);
}

/// Generic owner of adapter selection and foreground lease acquisition.
///
/// Its exact-one routing rule prevents a Goal binding from silently moving to
/// another transport implementation. Provider invocation and command
/// transport remain separate composition concerns.
pub struct GoalExecutionHost {
    adapters: Vec<Arc<dyn GoalSessionAdapter>>,
}

impl GoalExecutionHost {
    pub fn new(adapters: Vec<Arc<dyn GoalSessionAdapter>>) -> Self {
        Self { adapters }
    }

    fn adapter(&self, binding: &GoalSessionBinding) -> Result<Arc<dyn GoalSessionAdapter>> {
        let mut matches = self.adapters.iter().filter(|adapter| adapter.owns(binding));
        let Some(adapter) = matches.next() else {
            bail!("no Goal session adapter owns the trusted binding");
        };
        if matches.next().is_some() {
            bail!("multiple Goal session adapters own the trusted binding");
        }
        Ok(Arc::clone(adapter))
    }

    pub async fn model_binding(
        &self,
        binding: &GoalSessionBinding,
    ) -> Result<GoalHostModelBinding> {
        self.adapter(binding)?.model_binding(binding).await
    }

    pub async fn preflight(
        &self,
        binding: &GoalSessionBinding,
        request: &GoalExecutionPreflight,
    ) -> Result<GoalOperationReservation> {
        self.adapter(binding)?.preflight(binding, request).await
    }

    pub async fn acquire(
        &self,
        scope: &GoalExecutionScope,
        binding: &GoalSessionBinding,
    ) -> Result<Box<dyn GoalSessionLease>> {
        if scope.session_id != binding.session_id() {
            bail!("Goal execution scope does not match the trusted session binding");
        }
        self.adapter(binding)?.acquire(scope, binding).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::bail;
    use async_trait::async_trait;

    use super::{
        GoalExecutionHost, GoalHostModelBinding, GoalParentTurn, GoalSessionAdapter,
        GoalSessionBinding, GoalSessionLease, GoalVerifierTurn,
    };
    use crate::goal_mode::{
        GoalExecutionPreflight, GoalExecutionScope, GoalOperationReservation, GoalTurnKind,
    };

    struct Adapter {
        route: &'static str,
        binding_error: bool,
    }

    struct Lease {
        appended: bool,
    }

    #[async_trait]
    impl GoalSessionAdapter for Adapter {
        fn owns(&self, binding: &GoalSessionBinding) -> bool {
            binding.route() == self.route
        }

        async fn preflight(
            &self,
            _binding: &GoalSessionBinding,
            _request: &GoalExecutionPreflight,
        ) -> anyhow::Result<GoalOperationReservation> {
            Ok(GoalOperationReservation {
                token_count: Some(1),
                cost_usd: Some(0.0),
            })
        }

        async fn model_binding(
            &self,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<GoalHostModelBinding> {
            if self.binding_error {
                bail!("provider resolution failed")
            }
            Ok(GoalHostModelBinding {
                provider_ref: "provider.test".into(),
                model: "model-test".into(),
            })
        }

        async fn acquire(
            &self,
            _scope: &GoalExecutionScope,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<Box<dyn GoalSessionLease>> {
            Ok(Box::new(Lease { appended: false }))
        }
    }

    #[async_trait]
    impl GoalSessionLease for Lease {
        fn working_history(
            &self,
        ) -> anyhow::Result<Vec<zeroclaw_api::model_provider::ChatMessage>> {
            Ok(Vec::new())
        }

        async fn run_parent_turn(
            &mut self,
            _turn: GoalParentTurn,
            _cancellation: tokio_util::sync::CancellationToken,
        ) -> anyhow::Result<(String, Vec<zeroclaw_api::model_provider::ChatMessage>)> {
            Ok(("candidate".into(), Vec::new()))
        }

        async fn run_verifier(
            &mut self,
            _turn: GoalVerifierTurn,
            _cancellation: tokio_util::sync::CancellationToken,
        ) -> anyhow::Result<String> {
            Ok(r#"{"decision":"complete","reason":"accepted"}"#.into())
        }

        fn append_verified_candidate(&mut self, _candidate: &str) {
            self.appended = true;
        }
    }

    fn binding(route: &str) -> GoalSessionBinding {
        GoalSessionBinding::new(
            "session".into(),
            "agent".into(),
            route.into(),
            Some("principal".into()),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn host_requires_exactly_one_adapter_and_preserves_binding_errors() {
        let none = GoalExecutionHost::new(Vec::new());
        assert!(none.model_binding(&binding("rpc")).await.is_err());

        let ambiguous = GoalExecutionHost::new(vec![
            Arc::new(Adapter {
                route: "rpc",
                binding_error: false,
            }),
            Arc::new(Adapter {
                route: "rpc",
                binding_error: false,
            }),
        ]);
        assert!(ambiguous.model_binding(&binding("rpc")).await.is_err());

        let unavailable = GoalExecutionHost::new(vec![Arc::new(Adapter {
            route: "rpc",
            binding_error: true,
        })]);
        assert!(unavailable.model_binding(&binding("rpc")).await.is_err());
    }

    #[tokio::test]
    async fn host_rejects_a_scope_from_another_session_before_adapter_acquisition() {
        let host = GoalExecutionHost::new(vec![Arc::new(Adapter {
            route: "rpc",
            binding_error: false,
        })]);
        let scope = GoalExecutionScope {
            task_id: "task".into(),
            session_id: "other-session".into(),
            binding: binding("rpc"),
            execution_epoch: 1,
            turn_kind: GoalTurnKind::Start,
            cancellation: tokio_util::sync::CancellationToken::new(),
            operation_permit: Arc::new(tokio::sync::Semaphore::new(1)),
        };
        assert!(host.acquire(&scope, &binding("rpc")).await.is_err());
    }

    #[tokio::test]
    async fn host_leases_only_the_bound_session_and_keeps_live_state_adapter_owned() {
        let host = GoalExecutionHost::new(vec![Arc::new(Adapter {
            route: "rpc",
            binding_error: false,
        })]);
        let binding = binding("rpc");
        assert_eq!(binding.agent(), "agent");
        assert_eq!(binding.principal_id(), Some("principal"));
        let scope = GoalExecutionScope {
            task_id: "task".into(),
            session_id: binding.session_id().into(),
            binding: binding.clone(),
            execution_epoch: 1,
            turn_kind: GoalTurnKind::Start,
            cancellation: tokio_util::sync::CancellationToken::new(),
            operation_permit: Arc::new(tokio::sync::Semaphore::new(1)),
        };

        let resolved = host.model_binding(&binding).await.unwrap();
        assert_eq!(resolved.provider_ref, "provider.test");
        assert_eq!(resolved.model, "model-test");
        let reservation = host
            .preflight(
                &binding,
                &GoalExecutionPreflight {
                    task_id: scope.task_id.clone(),
                    session_id: scope.session_id.clone(),
                    binding: binding.clone(),
                    execution_epoch: scope.execution_epoch,
                    turn_kind: scope.turn_kind,
                    agent: binding.agent().into(),
                    objective: "objective".into(),
                    success_criteria: "criteria".into(),
                    token_limit: Some(10),
                    cost_limit_usd: None,
                    known_token_usage: 0,
                    known_cost_usd: 0.0,
                    token_accounting_complete: true,
                    cost_accounting_complete: true,
                },
            )
            .await
            .unwrap();
        assert_eq!(reservation.token_count, Some(1));
        let mut lease = host.acquire(&scope, &binding).await.unwrap();
        assert!(lease.working_history().unwrap().is_empty());
        let (candidate, transcript) = lease
            .run_parent_turn(
                GoalParentTurn {
                    objective: "objective".into(),
                    success_criteria: "criteria".into(),
                    turn_kind: GoalTurnKind::Start,
                    working_history: Vec::new(),
                },
                scope.cancellation.child_token(),
            )
            .await
            .unwrap();
        assert_eq!(candidate, "candidate");
        assert!(transcript.is_empty());
        assert_eq!(
            lease
                .run_verifier(
                    GoalVerifierTurn {
                        objective: "objective".into(),
                        success_criteria: "criteria".into(),
                        candidate: candidate.clone(),
                    },
                    scope.cancellation.child_token(),
                )
                .await
                .unwrap(),
            r#"{"decision":"complete","reason":"accepted"}"#,
        );
        lease.append_verified_candidate(&candidate);
    }
}
