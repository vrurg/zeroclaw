//! Transport-neutral session ownership for Goal execution.

use std::sync::Arc;

use anyhow::{Result, bail};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use zeroclaw_api::model_provider::ChatMessage;
use zeroclaw_providers::dispatch::AccountedCallReport;

use super::{GoalExecutionPreflight, GoalExecutionScope, GoalOperationReservation, GoalTurnKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalHostPreflightError {
    BindingUnavailable,
    VerifierUnavailable,
    PricingUnavailable,
    ReservationUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalSessionBinding {
    pub session_id: String,
    pub agent: String,
    pub route: String,
    pub principal_id: Option<String>,
}

impl GoalSessionBinding {
    pub fn new(
        session_id: String,
        agent: String,
        route: String,
        principal_id: Option<String>,
    ) -> Result<Self> {
        if session_id.trim().is_empty() || agent.trim().is_empty() || route.trim().is_empty() {
            bail!("Goal session binding has a blank required field");
        }
        Ok(Self {
            session_id,
            agent,
            route,
            principal_id: principal_id.filter(|value| !value.trim().is_empty()),
        })
    }
}

/// One parent turn rebuilt from durable Goal declaration and process-local state.
///
/// The request intentionally carries neither controller identity nor provider
/// configuration. The lease owns its normal session/provider context, while the
/// execution driver owns admission, accounting, and lifecycle transitions.
#[derive(Debug)]
pub struct GoalParentTurn {
    pub objective: String,
    pub success_criteria: String,
    pub turn_kind: GoalTurnKind,
    pub working_transcript: Vec<ChatMessage>,
}

/// The parent candidate and the process-local transcript to use for the next turn.
#[derive(Debug)]
pub struct GoalParentTurnResult {
    pub candidate: String,
    pub working_transcript: Vec<ChatMessage>,
}

/// Isolated verifier input. Session history, tools, and transport authority
/// are intentionally absent from this controller-owned protocol.
#[derive(Debug)]
pub struct GoalVerifierTurn {
    pub objective: String,
    pub success_criteria: String,
    pub candidate: String,
}

/// The final result of one accounted Goal model operation.
///
/// The report belongs to the same high-level operation as `result`, including
/// ordinary Reliable retry, fallback, and stream recovery. An error result does
/// not let the caller discard the report: Goal accounting must settle or fail
/// closed before it decides how to handle the operation failure.
#[derive(Debug)]
pub struct GoalOperationOutcome<T> {
    pub result: Result<T>,
    pub accounting: AccountedCallReport,
}

pub type GoalParentTurnOutcome = GoalOperationOutcome<GoalParentTurnResult>;
pub type GoalVerifierTurnOutcome = GoalOperationOutcome<String>;

/// An adapter-owned exclusive foreground lease for one Goal execution epoch.
///
/// A lease holds the channel/session mechanics that ordinary turns already use.
/// It is deliberately opaque to Goal control code: adapters do not mutate Goal
/// lifecycle or accounting, and Goal code does not construct a provider path.
#[async_trait]
pub trait GoalSessionLease: Send {
    async fn run_parent_turn(
        &mut self,
        turn: GoalParentTurn,
        cancellation: CancellationToken,
    ) -> GoalParentTurnOutcome;

    async fn run_verifier(
        &mut self,
        turn: GoalVerifierTurn,
        cancellation: CancellationToken,
    ) -> GoalVerifierTurnOutcome;
}

#[async_trait]
pub trait GoalSessionAdapter: Send + Sync {
    fn owns(&self, binding: &GoalSessionBinding) -> bool;
    async fn preflight(
        &self,
        binding: &GoalSessionBinding,
        request: &GoalExecutionPreflight,
    ) -> std::result::Result<GoalOperationReservation, GoalHostPreflightError>;
    async fn acquire(
        &self,
        scope: &GoalExecutionScope,
        binding: &GoalSessionBinding,
    ) -> Result<Box<dyn GoalSessionLease>>;
}

/// Selects exactly one channel/session implementation for a trusted binding.
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
            bail!("no Goal adapter owns this binding");
        };
        if matches.next().is_some() {
            bail!("multiple Goal adapters own this binding");
        }
        Ok(Arc::clone(adapter))
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
        if scope.session_id != binding.session_id {
            bail!("Goal scope does not match session binding");
        }
        self.adapter(binding)?.acquire(scope, binding).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal_mode::{GoalExecutionPreflight, GoalTurnKind};

    struct Adapter {
        owns: bool,
    }
    struct Lease;
    #[async_trait]
    impl GoalSessionLease for Lease {
        async fn run_parent_turn(
            &mut self,
            turn: GoalParentTurn,
            _: CancellationToken,
        ) -> GoalParentTurnOutcome {
            GoalParentTurnOutcome {
                result: Ok(GoalParentTurnResult {
                    candidate: "candidate".into(),
                    working_transcript: turn.working_transcript,
                }),
                accounting: AccountedCallReport::default(),
            }
        }

        async fn run_verifier(
            &mut self,
            _: GoalVerifierTurn,
            _: CancellationToken,
        ) -> GoalVerifierTurnOutcome {
            GoalVerifierTurnOutcome {
                result: Ok(r#"{"decision":"complete","reason":"ok"}"#.into()),
                accounting: AccountedCallReport::default(),
            }
        }
    }
    #[async_trait]
    impl GoalSessionAdapter for Adapter {
        fn owns(&self, _: &GoalSessionBinding) -> bool {
            self.owns
        }
        async fn preflight(
            &self,
            _: &GoalSessionBinding,
            _: &GoalExecutionPreflight,
        ) -> std::result::Result<GoalOperationReservation, GoalHostPreflightError> {
            Ok(GoalOperationReservation::default())
        }
        async fn acquire(
            &self,
            _: &GoalExecutionScope,
            _: &GoalSessionBinding,
        ) -> Result<Box<dyn GoalSessionLease>> {
            Ok(Box::new(Lease))
        }
    }
    fn binding() -> GoalSessionBinding {
        GoalSessionBinding::new("s".into(), "a".into(), "r".into(), None).unwrap()
    }
    fn preflight() -> GoalExecutionPreflight {
        GoalExecutionPreflight {
            task_id: "t".into(),
            session_id: "s".into(),
            execution_epoch: 1,
            turn_kind: GoalTurnKind::Start,
            agent: "a".into(),
            objective: "o".into(),
            success_criteria: "c".into(),
            token_limit: None,
            cost_limit_usd: None,
            known_token_usage: 0,
            known_cost_usd: 0.0,
            token_accounting_complete: true,
            cost_accounting_complete: true,
        }
    }
    #[tokio::test]
    async fn host_requires_exactly_one_owner() {
        let none = GoalExecutionHost::new(vec![]);
        assert_eq!(
            none.preflight(&binding(), &preflight()).await,
            Err(GoalHostPreflightError::BindingUnavailable)
        );
        let many = GoalExecutionHost::new(vec![
            Arc::new(Adapter { owns: true }),
            Arc::new(Adapter { owns: true }),
        ]);
        assert_eq!(
            many.preflight(&binding(), &preflight()).await,
            Err(GoalHostPreflightError::BindingUnavailable)
        );
        let one = GoalExecutionHost::new(vec![Arc::new(Adapter { owns: true })]);
        assert!(one.preflight(&binding(), &preflight()).await.is_ok());
        let scope = GoalExecutionScope {
            task_id: "t".into(),
            session_id: "s".into(),
            binding: binding(),
            operation_permit: Arc::new(tokio::sync::Semaphore::new(1)),
            execution_epoch: 1,
            turn_kind: GoalTurnKind::Start,
            cancellation: tokio_util::sync::CancellationToken::new(),
        };
        let mut lease = one.acquire(&scope, &binding()).await.unwrap();
        let outcome = lease
            .run_parent_turn(
                GoalParentTurn {
                    objective: "o".into(),
                    success_criteria: "c".into(),
                    turn_kind: GoalTurnKind::Start,
                    working_transcript: vec![ChatMessage::user("goal context")],
                },
                scope.cancellation.clone(),
            )
            .await;
        assert!(outcome.accounting.attempts().is_empty());
        let result = outcome.result.unwrap();
        assert_eq!(result.candidate, "candidate");
        assert_eq!(result.working_transcript.len(), 1);
        let verifier = lease
            .run_verifier(
                GoalVerifierTurn {
                    objective: "o".into(),
                    success_criteria: "c".into(),
                    candidate: result.candidate,
                },
                scope.cancellation.clone(),
            )
            .await;
        assert_eq!(verifier.result.unwrap(), r#"{"decision":"complete","reason":"ok"}"#);
    }
}
