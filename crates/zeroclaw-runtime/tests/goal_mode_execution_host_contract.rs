//! Public contract coverage for Goal execution hosts.
//!
//! A future channel host must use this narrow boundary without importing RPC
//! sessions, agents, or Goal accounting internals.

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use zeroclaw_api::model_provider::ChatMessage;
use zeroclaw_runtime::goal_mode::{
    GoalExecutionScope, GoalHostModelBinding, GoalSessionAdapter, GoalSessionBinding,
    GoalSessionLease,
};

struct FakeGoalSessionAdapter;

struct FakeGoalSessionLease;

#[async_trait]
impl GoalSessionAdapter for FakeGoalSessionAdapter {
    async fn model_binding(&self, _binding: &GoalSessionBinding) -> Option<GoalHostModelBinding> {
        Some(GoalHostModelBinding {
            provider_ref: "test.provider".into(),
            model: "test-model".into(),
        })
    }

    async fn acquire(
        &self,
        _scope: &GoalExecutionScope,
        _binding: &GoalSessionBinding,
    ) -> anyhow::Result<Box<dyn GoalSessionLease>> {
        Ok(Box::new(FakeGoalSessionLease))
    }
}

#[async_trait]
impl GoalSessionLease for FakeGoalSessionLease {
    fn working_history(&self) -> anyhow::Result<Vec<ChatMessage>> {
        Ok(vec![ChatMessage::system("test")])
    }

    async fn run_parent_turn(
        &mut self,
        working_history: Vec<ChatMessage>,
        _cancellation: CancellationToken,
    ) -> anyhow::Result<(String, Vec<ChatMessage>)> {
        Ok(("candidate".into(), working_history))
    }

    fn append_verified_candidate(&mut self, _candidate: String) -> anyhow::Result<()> {
        Ok(())
    }
}

#[test]
fn external_adapters_can_implement_the_narrow_goal_execution_contract() {
    let _adapter: Box<dyn GoalSessionAdapter> = Box::new(FakeGoalSessionAdapter);
}
