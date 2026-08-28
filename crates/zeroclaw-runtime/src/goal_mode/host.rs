//! Transport-neutral session ownership for Goal execution.

use std::sync::Arc;

use anyhow::{Result, bail};
use async_trait::async_trait;

use super::{GoalExecutionPreflight, GoalExecutionScope, GoalOperationReservation};

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

#[async_trait]
pub trait GoalSessionAdapter: Send + Sync {
    fn owns(&self, binding: &GoalSessionBinding) -> bool;
    async fn preflight(
        &self,
        binding: &GoalSessionBinding,
        request: &GoalExecutionPreflight,
    ) -> std::result::Result<GoalOperationReservation, GoalHostPreflightError>;
    async fn acquire(&self, scope: &GoalExecutionScope, binding: &GoalSessionBinding)
    -> Result<()>;
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
    ) -> Result<()> {
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
        async fn acquire(&self, _: &GoalExecutionScope, _: &GoalSessionBinding) -> Result<()> {
            Ok(())
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
    }
}
