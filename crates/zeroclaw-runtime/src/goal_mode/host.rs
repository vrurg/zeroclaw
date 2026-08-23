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

/// Resolved parent model binding supplied by a live Goal execution host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalHostModelBinding {
    /// Complete configured provider reference for the parent operation.
    pub provider_ref: String,
    /// Resolved model identifier for the parent operation.
    pub model: String,
}

/// Supplies a live, host-owned foreground session to the Goal driver.
///
/// Implementations retain authority over their own session identity, history,
/// concurrency lease, and parent turn execution. They must not create or
/// mutate durable Goal records.
#[async_trait]
pub trait GoalExecutionHost: Send + Sync {
    /// Resolve the exact parent binding for a session already bound to an
    /// expected configured agent.
    async fn model_binding(
        &self,
        session_id: &str,
        expected_agent: &str,
    ) -> Option<GoalHostModelBinding>;

    /// Acquire an exclusive foreground lease for the exact Goal scope.
    async fn acquire(
        &self,
        scope: &GoalExecutionScope,
        expected_agent: &str,
    ) -> Result<Box<dyn GoalExecutionSession>>;
}

/// Exclusive live-session lease retained throughout one Goal execution epoch.
#[async_trait]
pub trait GoalExecutionSession: Send {
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
