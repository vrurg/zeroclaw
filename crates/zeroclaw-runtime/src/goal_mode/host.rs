//! Narrow live-session contract used by the shared Goal execution driver.
//!
//! Durable Goal ownership, accounting, and lifecycle transitions remain in
//! the control plane. A host only supplies an exclusive foreground lease over
//! its already-authoritative live session state.

use anyhow::Result;
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
    /// Resolve the exact parent binding for a session already bound to an
    /// expected configured agent.
    async fn model_binding(&self, binding: &GoalSessionBinding) -> Option<GoalHostModelBinding>;

    /// Acquire an exclusive foreground lease for the exact Goal scope.
    async fn acquire(
        &self,
        scope: &GoalExecutionScope,
        binding: &GoalSessionBinding,
    ) -> Result<Box<dyn GoalSessionLease>>;
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
    fn append_verified_candidate(&mut self, candidate: String) -> Result<()>;
}
