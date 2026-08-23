//! Narrow live-session contract used by the shared Goal execution driver.
//!
//! Durable Goal ownership, accounting, and lifecycle transitions remain in
//! the control plane. A host only supplies an exclusive foreground lease over
//! its already-authoritative live session state.

use std::sync::Arc;

use anyhow::{Result, bail};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use zeroclaw_api::model_provider::ChatMessage;

use super::GoalExecutionScope;

/// Trusted canonical binding between one Goal and its live session adapter.
///
/// This value carries no transport message or mutable hook state. The
/// controller builds it from authenticated ingress before admission and from
/// the canonical task record for an executing Goal epoch.
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
    ) -> Self {
        Self {
            session_id,
            agent,
            route,
            principal_id,
        }
    }

    /// Canonical session identifier.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Server-derived owning agent alias.
    pub fn agent(&self) -> &str {
        &self.agent
    }

    /// Canonical transport route.
    pub fn route(&self) -> &str {
        &self.route
    }

    /// Durable principal identity, when the transport has one.
    pub fn principal_id(&self) -> Option<&str> {
        self.principal_id.as_deref()
    }
}

/// Resolved parent model binding supplied by a live Goal execution host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalHostModelBinding {
    /// Complete configured provider reference for the parent operation.
    pub provider_ref: String,
    /// Resolved model identifier for the parent operation.
    pub model: String,
}

/// Supplies a live, transport-owned foreground session to [`GoalExecutionHost`].
///
/// Implementations retain authority over their own session identity, history,
/// concurrency lease, and parent turn execution. They must not create or
/// mutate durable Goal records.
#[async_trait]
pub trait GoalSessionAdapter: Send + Sync {
    /// Return whether this adapter owns the trusted live-session binding.
    ///
    /// This predicate is deliberately pure: provider/model resolution belongs
    /// to [`Self::model_binding`] so router selection cannot hide a broken
    /// adapter as a different session owner.
    fn owns(&self, binding: &GoalSessionBinding) -> bool;

    /// Resolve the exact parent binding for a session already bound to an
    /// expected configured agent.
    async fn model_binding(&self, binding: &GoalSessionBinding) -> Result<GoalHostModelBinding>;

    /// Acquire an exclusive foreground lease for the exact Goal scope.
    async fn acquire(
        &self,
        scope: &GoalExecutionScope,
        binding: &GoalSessionBinding,
    ) -> Result<Box<dyn GoalSessionLease>>;
}

/// Runtime-private router over the daemon generation's live-session adapters.
///
/// The Goal host owns this router so individual adapters cannot select one
/// another or mutate the daemon's ownership topology after startup.
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
        let mut matches = self.adapters.iter().filter(|adapter| adapter.owns(binding));
        let Some(adapter) = matches.next() else {
            bail!("no Goal session adapter owns the trusted binding");
        };
        if matches.next().is_some() {
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

/// Exclusive live-session lease retained throughout one Goal execution epoch.
#[async_trait]
pub trait GoalSessionLease: Send {
    /// Return a detached provider-visible prefix of canonical session history.
    fn working_history(&self) -> Result<Vec<ChatMessage>>;

    /// Run one Goal parent turn against the process-local working transcript.
    async fn run_parent_turn(
        &mut self,
        working_history: Vec<ChatMessage>,
        cancellation: CancellationToken,
    ) -> Result<(String, Vec<ChatMessage>)>;

    /// Append the exact verified completion after the controller commits it.
    fn append_verified_candidate(&mut self, candidate: &str) -> Result<()>;

    /// Deliver the exact verified candidate through the owning live surface.
    ///
    /// A delivery error never reverses the durable `Completed` transition.
    async fn deliver_verified_candidate(&mut self, candidate: &str) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::{
        GoalHostModelBinding, GoalSessionAdapter, GoalSessionAdapterRouter, GoalSessionBinding,
        GoalSessionLease,
    };
    use crate::goal_mode::GoalExecutionScope;
    use anyhow::bail;
    use async_trait::async_trait;

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
    }

    #[tokio::test]
    async fn router_requires_one_owner_and_preserves_owner_readiness_errors() {
        let none = GoalSessionAdapterRouter::new(Vec::new());
        assert!(none.model_binding(&binding("matrix")).await.is_err());

        let ambiguous = GoalSessionAdapterRouter::new(vec![
            std::sync::Arc::new(Adapter {
                route: "matrix",
                binding_error: false,
            }),
            std::sync::Arc::new(Adapter {
                route: "matrix",
                binding_error: false,
            }),
        ]);
        assert!(ambiguous.model_binding(&binding("matrix")).await.is_err());

        let unavailable = GoalSessionAdapterRouter::new(vec![std::sync::Arc::new(Adapter {
            route: "matrix",
            binding_error: true,
        })]);
        assert!(unavailable.owns(&binding("matrix")));
        assert!(unavailable.model_binding(&binding("matrix")).await.is_err());
    }
}
