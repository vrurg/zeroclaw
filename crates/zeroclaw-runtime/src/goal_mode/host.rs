//! Live-session boundary for the transport-neutral Goal execution host.
//!
//! Durable Goal lifecycle and accounting remain in the control plane. A
//! session adapter only recognizes a trusted binding and supplies an exclusive
//! lease over its own live session state; it never creates or mutates Goal
//! records.

use std::sync::Arc;

use anyhow::{Result, bail};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use zeroclaw_api::model_provider::ChatMessage;

use super::GoalExecutionScope;

/// Canonical facts that bind a Goal to one adapter-owned live session.
///
/// The runtime constructs this only from durable task facts and trusted ingress
/// facts. It intentionally carries no display name, hook output, or mutable
/// transport message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalSessionBinding {
    session_id: String,
    agent: String,
    route: String,
    principal_id: Option<String>,
}

impl GoalSessionBinding {
    pub(crate) fn new(
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

/// Resolved configured parent route supplied by the adapter that owns a live
/// Goal session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalHostModelBinding {
    pub provider_ref: String,
    pub model: String,
}

/// Supplies live-session capabilities to the generic Goal execution owner.
///
/// The adapter is deliberately unable to mutate Goal lifecycle, accounting, or
/// authorization. It only recognizes an already trusted binding, resolves the
/// parent model, and leases its canonical session state.
#[async_trait]
pub trait GoalSessionAdapter: Send + Sync {
    fn owns(&self, binding: &GoalSessionBinding) -> bool;

    async fn model_binding(&self, binding: &GoalSessionBinding) -> Result<GoalHostModelBinding>;

    async fn acquire(
        &self,
        scope: &GoalExecutionScope,
        binding: &GoalSessionBinding,
    ) -> Result<Box<dyn GoalSessionLease>>;
}

/// Runtime-private exact-one router over the daemon generation's adapters.
pub(super) struct GoalSessionAdapterRouter {
    adapters: Vec<Arc<dyn GoalSessionAdapter>>,
}

impl GoalSessionAdapterRouter {
    pub(super) fn new(adapters: Vec<Arc<dyn GoalSessionAdapter>>) -> Self {
        Self { adapters }
    }

    fn matching_adapter(
        &self,
        binding: &GoalSessionBinding,
    ) -> Result<Arc<dyn GoalSessionAdapter>> {
        let mut adapters = self.adapters.iter().filter(|adapter| adapter.owns(binding));
        let Some(adapter) = adapters.next() else {
            bail!("no Goal session adapter owns the trusted binding");
        };
        if adapters.next().is_some() {
            bail!("multiple Goal session adapters own the trusted binding");
        }
        Ok(Arc::clone(adapter))
    }
}

#[async_trait]
impl GoalSessionAdapter for GoalSessionAdapterRouter {
    fn owns(&self, binding: &GoalSessionBinding) -> bool {
        self.matching_adapter(binding).is_ok()
    }

    async fn model_binding(&self, binding: &GoalSessionBinding) -> Result<GoalHostModelBinding> {
        self.matching_adapter(binding)?.model_binding(binding).await
    }

    async fn acquire(
        &self,
        scope: &GoalExecutionScope,
        binding: &GoalSessionBinding,
    ) -> Result<Box<dyn GoalSessionLease>> {
        self.matching_adapter(binding)?
            .acquire(scope, binding)
            .await
    }
}

/// An exclusive lease over one adapter-owned Goal session epoch.
#[async_trait]
pub trait GoalSessionLease: Send {
    /// A detached prefix of canonical session history. Goal intermediate turns
    /// never write to that history.
    fn working_history(&self) -> Result<Vec<ChatMessage>>;

    /// Execute one ordinary parent turn using the caller's process-local
    /// transcript and Goal-scoped cancellation context.
    async fn run_parent_turn(
        &mut self,
        working_history: Vec<ChatMessage>,
        cancellation: CancellationToken,
    ) -> Result<(String, Vec<ChatMessage>)>;

    /// Append only a controller-verified candidate to canonical history.
    fn append_verified_candidate(&mut self, candidate: &str);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::bail;
    use async_trait::async_trait;

    use super::{
        GoalHostModelBinding, GoalSessionAdapter, GoalSessionAdapterRouter, GoalSessionBinding,
        GoalSessionLease,
    };
    use crate::goal_mode::GoalExecutionScope;

    struct Adapter {
        route: &'static str,
        binding_error: bool,
    }

    #[async_trait]
    impl GoalSessionAdapter for Adapter {
        fn owns(&self, binding: &GoalSessionBinding) -> bool {
            binding.route() == self.route
        }

        async fn model_binding(
            &self,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<GoalHostModelBinding> {
            if self.binding_error {
                bail!("provider resolution failed")
            }
            Ok(GoalHostModelBinding {
                provider_ref: "provider.test".to_string(),
                model: "model-test".to_string(),
            })
        }

        async fn acquire(
            &self,
            _scope: &GoalExecutionScope,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<Box<dyn GoalSessionLease>> {
            bail!("not needed for routing tests")
        }
    }

    fn binding(route: &str) -> GoalSessionBinding {
        GoalSessionBinding::new(
            "session".to_string(),
            "agent".to_string(),
            route.to_string(),
            None,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn router_requires_exactly_one_owner_and_preserves_binding_errors() {
        let none = GoalSessionAdapterRouter::new(Vec::new());
        assert!(none.model_binding(&binding("rpc")).await.is_err());

        let ambiguous = GoalSessionAdapterRouter::new(vec![
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

        let unavailable = GoalSessionAdapterRouter::new(vec![Arc::new(Adapter {
            route: "rpc",
            binding_error: true,
        })]);
        assert!(unavailable.owns(&binding("rpc")));
        assert!(unavailable.model_binding(&binding("rpc")).await.is_err());
    }
}
