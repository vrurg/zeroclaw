//! RPC composition for Goal Mode's transport-neutral session-host contract.
//!
//! The Goal execution core owns no RPC types. This adapter is the RPC
//! runtime's implementation of that small contract and can be replaced by a
//! Matrix or other channel adapter without changing Goal control or execution.

use std::sync::Arc;

use anyhow::Context as _;
use async_trait::async_trait;

use crate::{
    agent::agent::Agent,
    goal_mode::{
        GoalExecutionScope, GoalHostModelBinding, GoalSessionAdapter, GoalSessionBinding,
        GoalSessionLease,
    },
    rpc::session::SessionStore,
};

/// Adapter that leases the existing RPC session actor. It owns no Goal
/// lifecycle state and does not expose a Goal RPC method.
pub(crate) struct RpcGoalSessionAdapter {
    sessions: Arc<SessionStore>,
}

impl RpcGoalSessionAdapter {
    pub(crate) fn new(sessions: Arc<SessionStore>) -> Self {
        Self { sessions }
    }
}

struct RpcGoalSessionLease {
    _queue_guard: zeroclaw_infra::session_queue::SessionGuard,
    agent: tokio::sync::OwnedMutexGuard<Agent>,
}

#[async_trait]
impl GoalSessionAdapter for RpcGoalSessionAdapter {
    fn owns(&self, binding: &GoalSessionBinding) -> bool {
        binding.route() == "rpc"
    }

    async fn model_binding(
        &self,
        binding: &GoalSessionBinding,
    ) -> anyhow::Result<GoalHostModelBinding> {
        if self
            .sessions
            .get_agent_alias(binding.session_id())
            .await
            .as_deref()
            != Some(binding.agent())
        {
            anyhow::bail!("Goal RPC session agent binding is unavailable");
        }
        let agent = self
            .sessions
            .get_agent(binding.session_id())
            .await
            .context("Goal RPC session is unavailable")?;
        let (provider_ref, model) = agent.lock().await.goal_model_binding();
        if provider_ref.trim().is_empty() || model.trim().is_empty() {
            anyhow::bail!("Goal RPC session model binding is unavailable");
        }
        Ok(GoalHostModelBinding {
            provider_ref,
            model,
        })
    }

    async fn acquire(
        &self,
        scope: &GoalExecutionScope,
        binding: &GoalSessionBinding,
    ) -> anyhow::Result<Box<dyn GoalSessionLease>> {
        if binding.session_id() != scope.session_id || binding.agent().trim().is_empty() {
            anyhow::bail!("Goal RPC lease binding does not match the execution scope");
        }
        let queue_guard = self
            .sessions
            .session_queue
            .acquire(&scope.session_id)
            .await
            .context("Goal RPC foreground lease is unavailable")?;
        if self
            .sessions
            .get_agent_alias(&scope.session_id)
            .await
            .as_deref()
            != Some(binding.agent())
        {
            anyhow::bail!("Goal RPC session agent binding changed while waiting for lease");
        }
        let agent = self
            .sessions
            .get_agent(&scope.session_id)
            .await
            .context("Goal RPC session disappeared while waiting for lease")?;
        Ok(Box::new(RpcGoalSessionLease {
            _queue_guard: queue_guard,
            agent: agent.lock_owned().await,
        }))
    }
}

#[async_trait]
impl GoalSessionLease for RpcGoalSessionLease {
    fn working_history(&self) -> anyhow::Result<Vec<zeroclaw_api::model_provider::ChatMessage>> {
        self.agent.goal_working_history()
    }

    async fn run_parent_turn(
        &mut self,
        working_history: Vec<zeroclaw_api::model_provider::ChatMessage>,
        cancellation: tokio_util::sync::CancellationToken,
    ) -> anyhow::Result<(String, Vec<zeroclaw_api::model_provider::ChatMessage>)> {
        self.agent
            .run_goal_parent_turn(working_history, cancellation)
            .await
    }

    async fn append_verified_candidate(&mut self, candidate: &str) -> anyhow::Result<()> {
        self.agent
            .append_verified_goal_candidate(candidate.to_owned());
        Ok(())
    }
}
