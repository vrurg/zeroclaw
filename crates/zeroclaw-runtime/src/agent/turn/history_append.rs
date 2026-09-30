//! History append for one tool round: the assistant message plus per-call
//! `role=tool` messages (native) or a `[Tool results]` user message (prompt
//! mode).

use zeroclaw_providers::{ChatMessage, ToolCall};

use super::results_collect::CollectedToolResult;
use crate::agent::prompt::SESSION_PROMPT_HISTORY_RESULT_KEY;

pub(crate) fn append_tool_round_to_history(
    history: &mut Vec<ChatMessage>,
    assistant_history_content: String,
    native_tool_calls: &[ToolCall],
    individual_results: &[CollectedToolResult],
    tool_results: &str,
    use_native_tools: bool,
) {
    history.push(ChatMessage::assistant(assistant_history_content));
    if native_tool_calls.is_empty() {
        let all_results_have_ids = use_native_tools
            && !individual_results.is_empty()
            && individual_results
                .iter()
                .all(|result| result.tool_call_id.is_some());
        if all_results_have_ids {
            for result in individual_results {
                history.push(native_result_message(
                    result,
                    result.tool_call_id.as_deref(),
                ));
            }
        } else {
            let content = format!("[Tool results]\n{tool_results}");
            let content = if individual_results
                .iter()
                .any(|result| result.sensitive_session_prompt)
            {
                // The host owns this result envelope. Keep the provider's full
                // explicit result, but carry executed-identity sensitivity to
                // every later export even when the assistant named another tool.
                let mut envelope = serde_json::json!({"content": content});
                envelope[SESSION_PROMPT_HISTORY_RESULT_KEY] = serde_json::json!(true);
                envelope.to_string()
            } else {
                content
            };
            history.push(ChatMessage::user(content));
        }
    } else {
        // `zip` would drop trailing results on any length divergence,
        // leaving a native tool_use id with no matching tool_result.
        // Pair on each result's own id instead.
        for (idx, result) in individual_results.iter().enumerate() {
            let resolved_id = result
                .tool_call_id
                .as_deref()
                .or_else(|| native_tool_calls.get(idx).map(|call| call.id.as_str()));
            history.push(native_result_message(result, resolved_id));
        }
    }
}

fn native_result_message(result: &CollectedToolResult, tool_call_id: Option<&str>) -> ChatMessage {
    let mut envelope = serde_json::json!({
        "tool_call_id": tool_call_id,
        "content": result.output,
    });
    if result.sensitive_session_prompt {
        envelope[SESSION_PROMPT_HISTORY_RESULT_KEY] = serde_json::json!(true);
    }
    ChatMessage::tool(envelope.to_string())
}

#[cfg(test)]
mod tests {
    use super::CollectedToolResult;
    use super::append_tool_round_to_history;
    use crate::agent::prompt::redact_session_prompt_tool_exchanges_for_export;
    use zeroclaw_providers::ChatMessage;

    #[test]
    fn json_tool_calls_text_fallback_result_is_redacted_only_in_export_copies() {
        let marker = "session-prompt-private-marker";
        let assistant =
            format!(r#"{{"tool_calls":[{{"name":"session_prompt_list","arguments":{{}}}}]}}"#);
        let result = format!("<tool_result name=\"session_prompt_list\">{marker}</tool_result>");
        let mut history: Vec<ChatMessage> = Vec::new();

        append_tool_round_to_history(
            &mut history,
            assistant,
            &[],
            &[CollectedToolResult {
                tool_call_id: None,
                output: result.clone(),
                sensitive_session_prompt: true,
            }],
            &result,
            false,
        );

        assert!(
            history[1].content.contains(marker),
            "the provider's working history keeps the explicit list result"
        );
        let exported = redact_session_prompt_tool_exchanges_for_export(&history);
        assert!(
            exported
                .iter()
                .all(|message| !message.content.contains(marker)),
            "generic exports must not retain the opaque attachment body"
        );
    }

    #[test]
    fn session_prompt_executed_identity_redaction_preserves_ordinary_results_and_native_ids() {
        let calls = [zeroclaw_providers::ToolCall {
            id: "fallback-id".into(),
            name: "ordinary_tool".into(),
            arguments: "{}".into(),
            extra_content: None,
        }];
        let results = [
            CollectedToolResult {
                tool_call_id: None,
                output: "synthetic-private-prompt".into(),
                sensitive_session_prompt: true,
            },
            CollectedToolResult {
                tool_call_id: Some("ordinary-id".into()),
                output: "ordinary-output".into(),
                sensitive_session_prompt: false,
            },
        ];
        let mut history = Vec::new();
        append_tool_round_to_history(
            &mut history,
            r#"{"tool_calls":[{"name":"ordinary_tool","arguments":{}}]}"#.into(),
            &calls,
            &results,
            "",
            true,
        );
        for (message, id, body) in [
            (&history[1], "fallback-id", "synthetic-private-prompt"),
            (&history[2], "ordinary-id", "ordinary-output"),
        ] {
            let value: serde_json::Value = serde_json::from_str(&message.content).unwrap();
            assert_eq!(value["tool_call_id"], id);
            assert_eq!(value["content"], body);
        }
        let exported = redact_session_prompt_tool_exchanges_for_export(&history);
        assert!(exported[1].content.contains("omitted from export"));
        assert_eq!(exported[2].role, history[2].role);
        assert_eq!(exported[2].content, history[2].content);

        let mut text_history = Vec::new();
        append_tool_round_to_history(
            &mut text_history,
            "ordinary assistant call".into(),
            &[],
            &results[1..],
            "ordinary-output",
            false,
        );
        assert_eq!(text_history[1].content, "[Tool results]\nordinary-output");
        let exported_text = redact_session_prompt_tool_exchanges_for_export(&text_history);
        assert_eq!(exported_text.len(), text_history.len());
        for (exported, original) in exported_text.iter().zip(&text_history) {
            assert_eq!(exported.role, original.role);
            assert_eq!(exported.content, original.content);
        }
    }
}
