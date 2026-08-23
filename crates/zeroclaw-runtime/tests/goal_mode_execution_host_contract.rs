//! Public contract coverage for Goal execution hosts.
//!
//! A future channel host must use this narrow boundary without importing RPC
//! sessions, agents, or Goal accounting internals.

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use zeroclaw_api::model_provider::ChatMessage;
use zeroclaw_runtime::goal_mode::{
    GoalExecutionHost, GoalExecutionScope, GoalExecutionSession, GoalHostModelBinding,
};

struct FakeGoalHost;

struct FakeGoalSession;

#[async_trait]
impl GoalExecutionHost for FakeGoalHost {
    async fn model_binding(
        &self,
        _session_id: &str,
        _expected_agent: &str,
    ) -> Option<GoalHostModelBinding> {
        Some(GoalHostModelBinding {
            provider_ref: "test.provider".into(),
            model: "test-model".into(),
        })
    }

    async fn acquire(
        &self,
        _scope: &GoalExecutionScope,
        _expected_agent: &str,
    ) -> anyhow::Result<Box<dyn GoalExecutionSession>> {
        Ok(Box::new(FakeGoalSession))
    }
}

#[async_trait]
impl GoalExecutionSession for FakeGoalSession {
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
fn external_hosts_can_implement_the_narrow_goal_execution_contract() {
    let _host: Box<dyn GoalExecutionHost> = Box::new(FakeGoalHost);
}
