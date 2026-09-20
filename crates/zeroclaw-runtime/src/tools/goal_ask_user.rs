//! Goal-aware interruption wrapper for the ordinary `ask_user` tool.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use zeroclaw_api::attribution::{Attributable, Role, ToolProvenance};
use zeroclaw_api::tool::{OptionDomain, Tool, ToolResult, ToolSpec};
use zeroclaw_config::policy::{SecurityPolicy, ToolOperation};

use crate::agent::goal_child_fence::{GoalTurnScope, goal_turn_scope};
use crate::agent::goal_user_input::{
    GoalUserInputRequest, MAX_GOAL_BLOCKER_MESSAGE_CHARS, RecordGoalUserInput,
    format_goal_user_input_request, record_goal_user_input,
};

const GOAL_PARENT_DESCRIPTION: &str = "Record a question for the current Goal session. The completed parent turn is presented, then the Goal pauses until the user sends /goal resume RESPONSE. Do not set channel or timeout_secs.";

/// Uses ordinary `ask_user` unchanged outside a Goal.  In a Goal parent turn,
/// it records a typed interruption so the controller can pause after the
/// normal turn presentation and resume with a durable request context.
pub(crate) struct GoalAwareAskUser {
    inner: Arc<dyn Tool>,
    security: Arc<SecurityPolicy>,
}

impl GoalAwareAskUser {
    pub(crate) fn new(inner: Arc<dyn Tool>, security: Arc<SecurityPolicy>) -> Self {
        Self { inner, security }
    }

    fn request_from_args(&self, args: &serde_json::Value) -> Result<GoalUserInputRequest> {
        let question = args
            .get("question")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::Error::msg("ask_user requires a non-empty bounded question"))?
            .to_owned();
        let choices = args
            .get("choices")
            .and_then(serde_json::Value::as_array)
            .map(|choices| {
                choices
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(str::trim)
                    .filter(|choice| !choice.is_empty())
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        anyhow::ensure!(
            format_goal_user_input_request(&question, &choices)
                .chars()
                .count()
                <= MAX_GOAL_BLOCKER_MESSAGE_CHARS,
            "ask_user question and choices exceed the Goal pause-message limit"
        );
        Ok(GoalUserInputRequest { question, choices })
    }
}

impl Attributable for GoalAwareAskUser {
    fn role(&self) -> Role {
        self.inner.role()
    }
    fn alias(&self) -> &str {
        self.inner.alias()
    }
    fn tool_provenance(&self) -> ToolProvenance {
        self.inner.tool_provenance()
    }
}

#[async_trait]
impl Tool for GoalAwareAskUser {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn description(&self) -> &str {
        match goal_turn_scope() {
            Some(GoalTurnScope::Parent) => GOAL_PARENT_DESCRIPTION,
            None | Some(GoalTurnScope::Child) => self.inner.description(),
        }
    }
    fn parameters_schema(&self) -> serde_json::Value {
        let mut schema = self.inner.parameters_schema();
        if matches!(goal_turn_scope(), Some(GoalTurnScope::Parent))
            && let Some(properties) = schema
                .get_mut("properties")
                .and_then(serde_json::Value::as_object_mut)
        {
            properties.remove("channel");
            properties.remove("timeout_secs");
        }
        schema
    }
    fn output_schema(&self) -> Option<serde_json::Value> {
        self.inner.output_schema()
    }
    fn param_domains(&self) -> Vec<(&'static str, OptionDomain)> {
        self.inner.param_domains()
    }
    fn spec(&self) -> ToolSpec {
        self.inner.spec()
    }
    fn invocation_triggers(&self) -> Vec<String> {
        self.inner.invocation_triggers()
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult> {
        match goal_turn_scope() {
            None => self.inner.execute(args).await,
            Some(GoalTurnScope::Child) => Ok(ToolResult::err(
                "ask_user is unavailable to a Goal child; return the needed information to its parent",
            )),
            Some(GoalTurnScope::Parent) => {
                if let Err(error) = self
                    .security
                    .enforce_tool_operation(ToolOperation::Act, "ask_user")
                {
                    return Ok(ToolResult::err(format!("Action blocked: {error}")));
                }
                let request = match self.request_from_args(&args) {
                    Ok(request) => request,
                    Err(error) => return Ok(ToolResult::err(error.to_string())),
                };
                match record_goal_user_input(request).await {
                    Some(RecordGoalUserInput::Recorded) => Ok(ToolResult::ok(
                        "User input request recorded. State the question clearly in your response; the Goal will pause after this turn.",
                    )),
                    Some(RecordGoalUserInput::AlreadyRequested) => Ok(ToolResult::err(
                        "A Goal user-input request is already pending; do not ask another question in this turn.",
                    )),
                    None => Ok(ToolResult::err(
                        "Goal user-input capture is unavailable for this parent turn; do not wait for a channel reply. Finish the response with the required Goal blocker certificate instead.",
                    )),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::GoalAwareAskUser;
    use crate::agent::goal_child_fence::{scope_goal_child, scope_goal_parent};
    use crate::agent::goal_user_input::{scope_goal_user_input, take_goal_user_input};
    use async_trait::async_trait;
    use serde_json::json;
    use zeroclaw_api::attribution::{Attributable, Role, ToolKind};
    use zeroclaw_api::tool::{Tool, ToolResult};
    use zeroclaw_config::policy::SecurityPolicy;

    struct InnerAskUser {
        calls: AtomicUsize,
    }

    impl Attributable for InnerAskUser {
        fn role(&self) -> Role {
            Role::Tool(ToolKind::Plugin)
        }

        fn alias(&self) -> &str {
            "test"
        }
    }

    #[async_trait]
    impl Tool for InnerAskUser {
        fn name(&self) -> &str {
            "ask_user"
        }

        fn description(&self) -> &str {
            "test ask_user"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "choices": {"type": "array"},
                    "channel": {"type": "string"},
                    "timeout_secs": {"type": "integer"}
                }
            })
        }

        async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolResult::ok("ordinary response"))
        }
    }

    #[tokio::test]
    async fn ordinary_turn_delegates_to_the_existing_ask_user_tool() {
        let inner = std::sync::Arc::new(InnerAskUser {
            calls: AtomicUsize::new(0),
        });
        let tool = GoalAwareAskUser::new(
            inner.clone(),
            std::sync::Arc::new(SecurityPolicy::default()),
        );

        let result = tool
            .execute(json!({"question": "Continue?"}))
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(inner.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn goal_parent_records_input_without_opening_another_channel_wait() {
        let inner = std::sync::Arc::new(InnerAskUser {
            calls: AtomicUsize::new(0),
        });
        let tool = GoalAwareAskUser::new(
            inner.clone(),
            std::sync::Arc::new(SecurityPolicy::default()),
        );

        scope_goal_user_input(scope_goal_parent(async {
            let result = tool
                .execute(json!({
                    "question": "Which implementation should I use?",
                    "choices": ["A", "B"]
                }))
                .await
                .unwrap();
            assert!(result.success);
            assert_eq!(inner.calls.load(Ordering::SeqCst), 0);
            let request = take_goal_user_input().await.unwrap();
            assert_eq!(request.question, "Which implementation should I use?");
            assert_eq!(request.choices, ["A", "B"]);
        }))
        .await;
    }

    #[tokio::test]
    async fn goal_parent_without_the_capture_scope_never_falls_back_to_channel_wait() {
        let inner = std::sync::Arc::new(InnerAskUser {
            calls: AtomicUsize::new(0),
        });
        let tool = GoalAwareAskUser::new(
            inner.clone(),
            std::sync::Arc::new(SecurityPolicy::default()),
        );

        let result = scope_goal_parent(async {
            tool.execute(json!({"question": "Which implementation should I use?"}))
                .await
                .unwrap()
        })
        .await;

        assert!(!result.success);
        assert!(
            result
                .error
                .unwrap_or_default()
                .contains("capture is unavailable")
        );
        assert_eq!(inner.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn goal_parent_exposes_session_pause_not_blocking_tool_semantics() {
        let tool = GoalAwareAskUser::new(
            std::sync::Arc::new(InnerAskUser {
                calls: AtomicUsize::new(0),
            }),
            std::sync::Arc::new(SecurityPolicy::default()),
        );

        scope_goal_parent(async {
            assert!(tool.description().contains("Goal session"));
            let schema = tool.parameters_schema();
            let properties = schema["properties"]
                .as_object()
                .expect("ask_user schema has properties");
            assert!(!properties.contains_key("channel"));
            assert!(!properties.contains_key("timeout_secs"));
        })
        .await;
    }

    #[test]
    fn goal_parent_rejects_a_request_that_exceeds_the_durable_pause_message_limit() {
        let tool = GoalAwareAskUser::new(
            std::sync::Arc::new(InnerAskUser {
                calls: AtomicUsize::new(0),
            }),
            std::sync::Arc::new(SecurityPolicy::default()),
        );

        let error = tool
            .request_from_args(&json!({
                "question": "x".repeat(super::MAX_GOAL_BLOCKER_MESSAGE_CHARS),
                "choices": ["A"],
            }))
            .unwrap_err();

        assert!(error.to_string().contains("pause-message limit"));
    }

    #[test]
    fn goal_parent_rejects_when_the_canonical_rendered_request_exceeds_the_limit() {
        let tool = GoalAwareAskUser::new(
            std::sync::Arc::new(InnerAskUser {
                calls: AtomicUsize::new(0),
            }),
            std::sync::Arc::new(SecurityPolicy::default()),
        );
        let question = "x".repeat(1_990);

        let error = tool
            .request_from_args(&json!({"question": question, "choices": ["ABCDE"]}))
            .unwrap_err();

        assert!(error.to_string().contains("pause-message limit"));
    }

    #[tokio::test]
    async fn goal_child_cannot_open_a_user_input_request() {
        let inner = std::sync::Arc::new(InnerAskUser {
            calls: AtomicUsize::new(0),
        });
        let tool = std::sync::Arc::new(GoalAwareAskUser::new(
            inner.clone(),
            std::sync::Arc::new(SecurityPolicy::default()),
        ));

        scope_goal_user_input(scope_goal_parent(async move {
            let tool = std::sync::Arc::clone(&tool);
            let result = scope_goal_child(Box::pin(async move {
                tool.execute(json!({"question": "Which option?"}))
                    .await
                    .unwrap()
            }))
            .await;
            assert!(!result.success);
            assert_eq!(inner.calls.load(Ordering::SeqCst), 0);
            assert!(take_goal_user_input().await.is_none());
        }))
        .await;
    }
}
