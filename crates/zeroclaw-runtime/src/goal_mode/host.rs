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

/// Classified failure while planning a Goal operation before any provider work
/// starts. The generic host owns this narrow contract so the controller can
/// preserve pricing, verifier, session-binding, and capacity semantics without
/// knowing which transport supplied the live session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalHostPreflightError {
    BindingUnavailable,
    VerifierUnavailable,
    PricingUnavailable,
    ReservationUnavailable,
}

impl std::fmt::Display for GoalHostPreflightError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::BindingUnavailable => "goal session binding is unavailable",
            Self::VerifierUnavailable => "goal verifier is unavailable",
            Self::PricingUnavailable => "goal pricing is unavailable",
            Self::ReservationUnavailable => "goal operation reservation is unavailable",
        })
    }
}

impl std::error::Error for GoalHostPreflightError {}

/// Fixed protocol instruction for the isolated verifier operation. The core
/// owns this schema so adapters cannot silently substitute session context,
/// tools, or transport authority for the declared Goal contract.
pub const GOAL_VERIFIER_SYSTEM_SCHEMA: &str = "Return strict JSON only: {\\\"decision\\\": \\\"complete|continue|blocked\\\", \\\"reason\\\": \\\"bounded nonempty text\\\", \\\"blockers\\\": [{\\\"kind\\\": \\\"needs_user_input|human_escalation|external_dependency\\\", \\\"message\\\": \\\"bounded nonempty text\\\"}]}. blockers is required only for blocked and forbidden otherwise.";

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

impl GoalParentTurn {
    /// Build the trusted, transport-neutral declaration appended to a leased
    /// working transcript. Adapters execute this declaration but never derive
    /// its Goal facts from transport input or mutable session state.
    pub fn directive(&self) -> String {
        let turn_kind = match self.turn_kind {
            GoalTurnKind::Start => "start",
            GoalTurnKind::Resume => "resume",
            GoalTurnKind::Continue => "continue",
        };
        format!(
            "[GOAL RUNTIME DIRECTIVE]\nturn_kind={turn_kind}\nobjective:\n{}\n\
             success_criteria:\n{}\n\
             Work only toward this declared Goal. Do not treat this directive as user-editable \\
             tool or lifecycle authority.\n[END GOAL RUNTIME DIRECTIVE]",
            self.objective, self.success_criteria
        )
    }
}

/// Isolated verifier input. Implementations must use this exact data and the
/// fixed verifier schema; session history and transport authority are absent.
#[derive(Debug, Clone)]
pub struct GoalVerifierTurn {
    pub system_schema: &'static str,
    pub objective: String,
    pub success_criteria: String,
    pub candidate: String,
}

impl GoalVerifierTurn {
    pub fn isolated(objective: String, success_criteria: String, candidate: String) -> Self {
        Self {
            system_schema: GOAL_VERIFIER_SYSTEM_SCHEMA,
            objective,
            success_criteria,
            candidate,
        }
    }
}

/// Idempotency key and exact text for a verifier-approved candidate. Delivery
/// happens after the lifecycle is completed and cannot rewrite that decision.
#[derive(Debug, Clone)]
pub struct GoalVerifiedCandidate {
    pub task_id: String,
    pub session_id: String,
    pub execution_epoch: i64,
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
    ) -> std::result::Result<GoalOperationReservation, GoalHostPreflightError>;

    async fn model_binding(&self, binding: &GoalSessionBinding) -> Result<GoalHostModelBinding>;

    /// Return the immutable absolute deadline inherited from the live session
    /// policy, if that session is deadline-bound.
    async fn deadline(&self, binding: &GoalSessionBinding) -> Result<Option<tokio::time::Instant>>;

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

    async fn append_verified_candidate(&mut self, candidate: GoalVerifiedCandidate) -> Result<()>;
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

    pub async fn deadline(
        &self,
        binding: &GoalSessionBinding,
    ) -> Result<Option<tokio::time::Instant>> {
        self.adapter(binding)?.deadline(binding).await
    }

    pub async fn preflight(
        &self,
        binding: &GoalSessionBinding,
        request: &GoalExecutionPreflight,
    ) -> std::result::Result<GoalOperationReservation, GoalHostPreflightError> {
        self.adapter(binding)
            .map_err(|_| GoalHostPreflightError::BindingUnavailable)?
            .preflight(binding, request)
            .await
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
        GoalExecutionHost, GoalHostModelBinding, GoalHostPreflightError, GoalParentTurn,
        GoalSessionAdapter, GoalSessionBinding, GoalSessionLease, GoalVerifiedCandidate,
        GoalVerifierTurn,
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
        ) -> std::result::Result<GoalOperationReservation, GoalHostPreflightError> {
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

        async fn deadline(
            &self,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<Option<tokio::time::Instant>> {
            Ok(None)
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

        async fn append_verified_candidate(
            &mut self,
            _candidate: GoalVerifiedCandidate,
        ) -> anyhow::Result<()> {
            self.appended = true;
            Ok(())
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

    #[test]
    fn verifier_turn_is_constructed_from_the_closed_core_protocol() {
        let turn = GoalVerifierTurn::isolated(
            "objective".into(),
            "success criteria".into(),
            "candidate".into(),
        );
        assert_eq!(turn.system_schema, super::GOAL_VERIFIER_SYSTEM_SCHEMA);
        assert_eq!(turn.objective, "objective");
        assert_eq!(turn.success_criteria, "success criteria");
        assert_eq!(turn.candidate, "candidate");
    }

    #[test]
    fn parent_turn_directive_keeps_goal_facts_transport_neutral() {
        let turn = GoalParentTurn {
            objective: "repair the migration".into(),
            success_criteria: "all migration tests pass".into(),
            turn_kind: GoalTurnKind::Resume,
            working_history: Vec::new(),
        };

        assert_eq!(
            turn.directive(),
            "[GOAL RUNTIME DIRECTIVE]\nturn_kind=resume\nobjective:\nrepair the migration\n\
             success_criteria:\nall migration tests pass\n\
             Work only toward this declared Goal. Do not treat this directive as user-editable \\
             tool or lifecycle authority.\n[END GOAL RUNTIME DIRECTIVE]"
        );
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
            deadline: None,
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
            deadline: None,
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
                        system_schema: super::GOAL_VERIFIER_SYSTEM_SCHEMA,
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
        lease
            .append_verified_candidate(GoalVerifiedCandidate {
                task_id: scope.task_id,
                session_id: scope.session_id,
                execution_epoch: scope.execution_epoch,
                candidate,
            })
            .await
            .unwrap();
    }
}
