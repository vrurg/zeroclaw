use super::history::canonicalize_tool_result_media_markers_for;
use crate::tools::{Tool, ToolSpec};
use serde::Deserialize;
use serde_json::Value;
use std::fmt::Write;
use zeroclaw_providers::{
    ChatMessage, ChatResponse, ConversationMessage, ToolCall, ToolResultMessage,
};

/// Tool transcript shape accepted by the provider serving the next request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolTranscriptProtocol {
    Native,
    Text,
    None,
}

#[derive(Deserialize)]
struct NativeAssistantHistory {
    #[serde(default)]
    content: Option<String>,
    tool_calls: Vec<ToolCall>,
}

#[derive(Deserialize)]
struct NativeToolResultHistory {
    tool_call_id: String,
    content: String,
}

fn escape_xml_attribute(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn complete_native_tool_exchange(
    history: &[ChatMessage],
    assistant_index: usize,
) -> Option<(NativeAssistantHistory, Vec<NativeToolResultHistory>, usize)> {
    let assistant = history.get(assistant_index)?;
    if assistant.role != "assistant" {
        return None;
    }
    let envelope = serde_json::from_str::<NativeAssistantHistory>(&assistant.content).ok()?;
    if envelope.tool_calls.is_empty() {
        return None;
    }

    let mut end = assistant_index + 1;
    let mut results = Vec::new();
    while let Some(message) = history.get(end).filter(|message| message.role == "tool") {
        results.push(serde_json::from_str::<NativeToolResultHistory>(&message.content).ok()?);
        end += 1;
    }

    let complete = results.len() == envelope.tool_calls.len()
        && envelope.tool_calls.iter().all(|call| {
            results
                .iter()
                .filter(|result| result.tool_call_id == call.id)
                .count()
                == 1
        })
        && results.iter().all(|result| {
            envelope
                .tool_calls
                .iter()
                .any(|call| call.id == result.tool_call_id)
        });
    complete.then_some((envelope, results, end))
}

fn render_text_tool_exchange(
    envelope: &NativeAssistantHistory,
    results: &[NativeToolResultHistory],
) -> [ChatMessage; 2] {
    let mut assistant = envelope.content.clone().unwrap_or_default();
    for call in &envelope.tool_calls {
        if !assistant.is_empty() && !assistant.ends_with('\n') {
            assistant.push('\n');
        }
        let arguments = serde_json::from_str::<Value>(&call.arguments)
            .unwrap_or_else(|_| Value::String(call.arguments.clone()));
        let payload = serde_json::json!({
            "name": call.name,
            "arguments": arguments,
        });
        let _ = writeln!(assistant, "<tool_call>\n{payload}\n</tool_call>");
    }

    let mut tool_results = String::from("[Tool results]\n");
    for result in results {
        let name = envelope
            .tool_calls
            .iter()
            .find(|call| call.id == result.tool_call_id)
            .map_or("unknown", |call| call.name.as_str());
        let name = escape_xml_attribute(name);
        let _ = writeln!(
            tool_results,
            "<tool_result name=\"{name}\">\n{}\n</tool_result>",
            result.content
        );
    }

    [
        ChatMessage::assistant(assistant),
        ChatMessage::user(tool_results),
    ]
}

fn render_tool_free_exchange(
    envelope: &NativeAssistantHistory,
    results: &[NativeToolResultHistory],
) -> [ChatMessage; 2] {
    let mut assistant = envelope.content.clone().unwrap_or_default();
    if !assistant.is_empty() && !assistant.ends_with('\n') {
        assistant.push('\n');
    }
    let names = envelope
        .tool_calls
        .iter()
        .map(|call| call.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let _ = write!(
        assistant,
        "Completed tool calls before the model switch: {names}."
    );

    let mut tool_results = String::from("[Tool results from before the model switch]\n");
    for result in results {
        let name = envelope
            .tool_calls
            .iter()
            .find(|call| call.id == result.tool_call_id)
            .map_or("unknown", |call| call.name.as_str());
        let _ = writeln!(tool_results, "\n{name}:\n{}", result.content);
    }

    [
        ChatMessage::assistant(assistant),
        ChatMessage::user(tool_results),
    ]
}

/// Materialize a provider-only view of completed tool exchanges.
///
/// The input remains the canonical/durable transcript. Native exchanges are
/// rewritten only when the target cannot accept `role=tool`; incomplete or
/// malformed exchanges are kept byte-for-byte rather than partially rewritten.
/// Text exchanges remain ordinary assistant/user messages when switching to a
/// native provider: fabricating native call IDs would lose provider metadata.
#[must_use]
pub fn project_provider_visible_tool_history(
    history: &[ChatMessage],
    target: ToolTranscriptProtocol,
) -> Vec<ChatMessage> {
    if target == ToolTranscriptProtocol::Native {
        return history.to_vec();
    }

    let mut projected = Vec::with_capacity(history.len());
    let mut index = 0;
    while index < history.len() {
        let Some((envelope, results, end)) = complete_native_tool_exchange(history, index) else {
            projected.push(history[index].clone());
            index += 1;
            continue;
        };

        let rendered = match target {
            ToolTranscriptProtocol::Text => render_text_tool_exchange(&envelope, &results),
            ToolTranscriptProtocol::None => render_tool_free_exchange(&envelope, &results),
            ToolTranscriptProtocol::Native => unreachable!(),
        };
        projected.extend(rendered);
        index = end;
    }
    projected
}

#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: Value,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ToolExecutionResult {
    pub name: String,
    pub output: String,
    pub success: bool,
    pub tool_call_id: Option<String>,
}

pub trait ToolDispatcher: Send + Sync {
    fn parse_response(&self, response: &ChatResponse) -> (String, Vec<ParsedToolCall>);
    fn format_results(&self, results: &[ToolExecutionResult]) -> ConversationMessage;
    fn prompt_instructions(&self, tools: &[Box<dyn Tool>]) -> String;
    fn to_provider_messages(&self, history: &[ConversationMessage]) -> Vec<ChatMessage>;
    fn should_send_tool_specs(&self) -> bool;
}

#[derive(Default)]
pub struct XmlToolDispatcher;

impl XmlToolDispatcher {
    fn parse_xml_tool_calls(response: &str) -> (String, Vec<ParsedToolCall>) {
        // Strip `<think>...</think>` blocks before parsing tool calls.
        // Qwen and other reasoning models may embed chain-of-thought inline.
        let cleaned = Self::strip_think_tags(response);
        let mut text_parts = Vec::new();
        let mut calls = Vec::new();
        let mut remaining = cleaned.as_str();

        while let Some(start) = remaining.find("<tool_call>") {
            let before = &remaining[..start];
            if !before.trim().is_empty() {
                text_parts.push(before.trim().to_string());
            }

            if let Some(end) = remaining[start..].find("</tool_call>") {
                let inner = &remaining[start + 11..start + end];
                match serde_json::from_str::<Value>(inner.trim()) {
                    Ok(parsed) => {
                        let name = parsed
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        if name.is_empty() {
                            remaining = &remaining[start + end + 12..];
                            continue;
                        }
                        let arguments = parsed
                            .get("arguments")
                            .cloned()
                            .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
                        calls.push(ParsedToolCall {
                            name,
                            arguments,
                            tool_call_id: None,
                        });
                    }
                    Err(e) => {
                        ::zeroclaw_log::record!(
                            WARN,
                            ::zeroclaw_log::Event::new(
                                module_path!(),
                                ::zeroclaw_log::Action::Note
                            )
                            .with_category(::zeroclaw_log::EventCategory::Agent)
                            .with_outcome(::zeroclaw_log::EventOutcome::Unknown)
                            .with_attrs(::serde_json::json!({"error": format!("{}", e)})),
                            "Malformed <tool_call> JSON"
                        );
                    }
                }
                remaining = &remaining[start + end + 12..];
            } else {
                break;
            }
        }

        if !remaining.trim().is_empty() {
            text_parts.push(remaining.trim().to_string());
        }

        (text_parts.join("\n"), calls)
    }

    /// Remove `<think>...</think>` blocks from model output.
    fn strip_think_tags(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        let mut rest = s;
        loop {
            if let Some(start) = rest.find("<think>") {
                result.push_str(&rest[..start]);
                if let Some(end) = rest[start..].find("</think>") {
                    rest = &rest[start + end + "</think>".len()..];
                } else {
                    break;
                }
            } else {
                result.push_str(rest);
                break;
            }
        }
        result
    }

    pub fn tool_specs(tools: &[Box<dyn Tool>]) -> Vec<ToolSpec> {
        tools.iter().map(|tool| tool.spec()).collect()
    }
}

impl ToolDispatcher for XmlToolDispatcher {
    fn parse_response(&self, response: &ChatResponse) -> (String, Vec<ParsedToolCall>) {
        let text = response.text_or_empty();
        Self::parse_xml_tool_calls(text)
    }

    fn format_results(&self, results: &[ToolExecutionResult]) -> ConversationMessage {
        let mut content = String::new();
        for result in results {
            let status = if result.success { "ok" } else { "error" };
            let output = canonicalize_tool_result_media_markers_for(&result.name, &result.output);
            let _ = writeln!(
                content,
                "<tool_result name=\"{}\" status=\"{}\">\n{}\n</tool_result>",
                result.name, status, output
            );
        }
        ConversationMessage::Chat(ChatMessage::user(format!("[Tool results]\n{content}")))
    }

    fn prompt_instructions(&self, tools: &[Box<dyn Tool>]) -> String {
        if tools.is_empty() {
            return String::new();
        }

        let mut instructions = String::new();
        instructions.push_str("## Tool Use Protocol\n\n");
        instructions
            .push_str("To use a tool, wrap a JSON object in <tool_call></tool_call> tags:\n\n");
        instructions.push_str(
            "```\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n```\n\n",
        );

        instructions
    }

    fn to_provider_messages(&self, history: &[ConversationMessage]) -> Vec<ChatMessage> {
        history
            .iter()
            .flat_map(|msg| match msg {
                ConversationMessage::Chat(chat) => vec![chat.clone()],
                ConversationMessage::AssistantToolCalls { text, .. } => {
                    vec![ChatMessage::assistant(text.clone().unwrap_or_default())]
                }
                ConversationMessage::ToolResults(results) => {
                    let mut content = String::new();
                    for result in results {
                        let output = canonicalize_tool_result_media_markers_for(
                            &result.tool_name,
                            &result.content,
                        );
                        let _ = writeln!(
                            content,
                            "<tool_result id=\"{}\">\n{}\n</tool_result>",
                            result.tool_call_id, output
                        );
                    }
                    vec![ChatMessage::user(format!("[Tool results]\n{content}"))]
                }
            })
            .collect()
    }

    fn should_send_tool_specs(&self) -> bool {
        false
    }
}

pub struct NativeToolDispatcher;

impl ToolDispatcher for NativeToolDispatcher {
    fn parse_response(&self, response: &ChatResponse) -> (String, Vec<ParsedToolCall>) {
        let text = response.text.clone().unwrap_or_default();
        let calls = response
            .tool_calls
            .iter()
            .map(|tc| ParsedToolCall {
                name: tc.name.clone(),
                arguments: serde_json::from_str(&tc.arguments).unwrap_or_else(|e| {
                    ::zeroclaw_log::record!(WARN, ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note).with_category(::zeroclaw_log::EventCategory::Tool).with_outcome(::zeroclaw_log::EventOutcome::Unknown).with_attrs(::serde_json::json!({"tool": tc.name, "error": format!("{}", e)})), "Failed to parse native tool call arguments as JSON; defaulting to empty object");
                    Value::Object(serde_json::Map::new())
                }),
                tool_call_id: Some(tc.id.clone()),
            })
            .collect();
        (text, calls)
    }

    fn format_results(&self, results: &[ToolExecutionResult]) -> ConversationMessage {
        let messages = results
            .iter()
            .map(|result| ToolResultMessage {
                tool_call_id: result
                    .tool_call_id
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                // Retain the producing tool name so the read path
                // (`to_provider_messages`) can re-canonicalize provenance-aware
                // instead of blindly re-promoting a search/listing path back
                // into an `[IMAGE:...]` marker
                tool_name: result.name.clone(),
                // Provenance-gated see the XML dispatcher above.
                content: canonicalize_tool_result_media_markers_for(&result.name, &result.output),
            })
            .collect();
        ConversationMessage::ToolResults(messages)
    }

    fn prompt_instructions(&self, _tools: &[Box<dyn Tool>]) -> String {
        String::new()
    }

    fn to_provider_messages(&self, history: &[ConversationMessage]) -> Vec<ChatMessage> {
        history
            .iter()
            .flat_map(|msg| match msg {
                ConversationMessage::Chat(chat) => vec![chat.clone()],
                ConversationMessage::AssistantToolCalls {
                    text,
                    tool_calls,
                    reasoning_content,
                } => {
                    let mut payload = serde_json::json!({
                        "content": text,
                        "tool_calls": tool_calls,
                    });
                    if let Some(rc) = reasoning_content {
                        payload["reasoning_content"] = serde_json::json!(rc);
                    }
                    vec![ChatMessage::assistant(payload.to_string())]
                }
                ConversationMessage::ToolResults(results) => results
                    .iter()
                    .map(|result| {
                        ChatMessage::tool(
                            serde_json::json!({
                                "tool_call_id": result.tool_call_id,
                                "content": canonicalize_tool_result_media_markers_for(
                                    &result.tool_name,
                                    &result.content,
                                ),
                            })
                            .to_string(),
                        )
                    })
                    .collect(),
            })
            .collect()
    }

    fn should_send_tool_specs(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xml_dispatcher_parses_tool_calls() {
        let response = ChatResponse {
            text: Some(
                "Checking\n<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}</tool_call>"
                    .into(),
            ),
            tool_calls: vec![],
            usage: None,
            reasoning_content: None,
        };
        let dispatcher = XmlToolDispatcher;
        let (_, calls) = dispatcher.parse_response(&response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
    }

    #[test]
    fn xml_dispatcher_strips_think_before_tool_call() {
        let response = ChatResponse {
            text: Some(
                "<think>I should list files</think>\n<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}</tool_call>"
                    .into(),
            ),
            tool_calls: vec![],
            usage: None,
            reasoning_content: None,
        };
        let dispatcher = XmlToolDispatcher;
        let (text, calls) = dispatcher.parse_response(&response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert!(
            !text.contains("<think>"),
            "think tags should be stripped from text"
        );
    }

    #[test]
    fn xml_dispatcher_think_only_returns_no_calls() {
        let response = ChatResponse {
            text: Some("<think>Just thinking</think>".into()),
            tool_calls: vec![],
            usage: None,
            reasoning_content: None,
        };
        let dispatcher = XmlToolDispatcher;
        let (_, calls) = dispatcher.parse_response(&response);
        assert!(calls.is_empty());
    }

    #[test]
    fn native_dispatcher_roundtrip() {
        let response = ChatResponse {
            text: Some("ok".into()),
            tool_calls: vec![zeroclaw_providers::ToolCall {
                id: "tc1".into(),
                name: "file_read".into(),
                arguments: "{\"path\":\"a.txt\"}".into(),
                extra_content: None,
            }],
            usage: None,
            reasoning_content: None,
        };
        let dispatcher = NativeToolDispatcher;
        let (_, calls) = dispatcher.parse_response(&response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_call_id.as_deref(), Some("tc1"));

        let msg = dispatcher.format_results(&[ToolExecutionResult {
            name: "file_read".into(),
            output: "hello".into(),
            success: true,
            tool_call_id: Some("tc1".into()),
        }]);
        match msg {
            ConversationMessage::ToolResults(results) => {
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].tool_call_id, "tc1");
            }
            _ => panic!("expected tool results"),
        }
    }

    #[test]
    fn xml_format_results_contains_tool_result_tags() {
        let dispatcher = XmlToolDispatcher;
        let msg = dispatcher.format_results(&[ToolExecutionResult {
            name: "shell".into(),
            output: "ok".into(),
            success: true,
            tool_call_id: None,
        }]);
        let rendered = match msg {
            ConversationMessage::Chat(chat) => chat.content,
            _ => String::new(),
        };
        assert!(rendered.contains("<tool_result"));
        assert!(rendered.contains("shell"));
    }

    #[test]
    fn native_format_results_keeps_tool_call_id() {
        let dispatcher = NativeToolDispatcher;
        let msg = dispatcher.format_results(&[ToolExecutionResult {
            name: "shell".into(),
            output: "ok".into(),
            success: true,
            tool_call_id: Some("tc-1".into()),
        }]);

        match msg {
            ConversationMessage::ToolResults(results) => {
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].tool_call_id, "tc-1");
            }
            _ => panic!("expected ToolResults variant"),
        }
    }

    /// Write a throwaway PNG and return its absolute path string. An existing
    /// local image path is required for canonicalization to fire at all.
    fn write_temp_image(dir: &std::path::Path, name: &str) -> String {
        let image = dir.join(name);
        std::fs::write(&image, [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n']).unwrap();
        image.display().to_string()
    }

    fn xml_format_results_text(
        dispatcher: &XmlToolDispatcher,
        result: ToolExecutionResult,
    ) -> String {
        match dispatcher.format_results(&[result]) {
            ConversationMessage::Chat(chat) => chat.content,
            _ => panic!("XmlToolDispatcher::format_results must return a Chat message"),
        }
    }

    fn native_format_results_content(
        dispatcher: &NativeToolDispatcher,
        result: ToolExecutionResult,
    ) -> String {
        match dispatcher.format_results(&[result]) {
            ConversationMessage::ToolResults(results) => results[0].content.clone(),
            _ => panic!("NativeToolDispatcher::format_results must return ToolResults"),
        }
    }

    #[test]
    fn xml_format_results_does_not_promote_search_tool_image_paths() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp_image(dir.path(), "hit.png");
        let xml = XmlToolDispatcher;

        for tool in ["content_search", "glob_search"] {
            let rendered = xml_format_results_text(
                &xml,
                ToolExecutionResult {
                    name: tool.into(),
                    output: format!("match: {path}"),
                    success: true,
                    tool_call_id: None,
                },
            );
            assert!(
                !rendered.contains("[IMAGE:"),
                "{tool} output must not be promoted to an image marker"
            );
            assert!(
                rendered.contains(&path),
                "{tool} output must still carry the literal path text"
            );
        }
    }

    #[test]
    fn native_format_results_does_not_promote_search_tool_image_paths() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp_image(dir.path(), "hit.png");
        let native = NativeToolDispatcher;

        for tool in ["content_search", "glob_search"] {
            let content = native_format_results_content(
                &native,
                ToolExecutionResult {
                    name: tool.into(),
                    output: format!("found: {path}"),
                    success: true,
                    tool_call_id: Some("tc1".into()),
                },
            );
            assert!(
                !content.contains("[IMAGE:"),
                "{tool} output must not be promoted to an image marker"
            );
            assert!(content.contains(&path));
        }
    }

    #[test]
    fn format_results_still_promotes_image_producing_tool_paths() {
        // Default-allow: a genuinely image-producing tool keeps canonicalization
        // in BOTH dispatchers, so real tool-produced images still route to a
        // vision provider.
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp_image(dir.path(), "generated.png");
        let expected = format!("[IMAGE:{path}]");

        let xml = XmlToolDispatcher;
        let rendered = xml_format_results_text(
            &xml,
            ToolExecutionResult {
                name: "image_gen".into(),
                output: format!("saved to {path}"),
                success: true,
                tool_call_id: None,
            },
        );
        assert!(
            rendered.contains(&expected),
            "image_gen output must be canonicalized into a marker (XML)"
        );

        let native = NativeToolDispatcher;
        let content = native_format_results_content(
            &native,
            ToolExecutionResult {
                name: "image_gen".into(),
                output: format!("saved to {path}"),
                success: true,
                tool_call_id: Some("tc1".into()),
            },
        );
        assert!(
            content.contains(&expected),
            "image_gen output must be canonicalized into a marker (native)"
        );
    }

    fn native_tool_message_content(message: &ChatMessage) -> String {
        let payload: serde_json::Value = serde_json::from_str(&message.content).unwrap();
        payload["content"].as_str().unwrap().to_owned()
    }

    fn native_round_trip_tool_content(
        dispatcher: &NativeToolDispatcher,
        result: ToolExecutionResult,
    ) -> String {
        let stored = dispatcher.format_results(&[result]);
        let messages = dispatcher.to_provider_messages(&[stored]);
        assert_eq!(messages.len(), 1, "one tool result -> one provider message");
        assert_eq!(messages[0].role, "tool");
        native_tool_message_content(&messages[0])
    }

    #[test]
    fn native_search_path_survives_format_then_to_provider_messages() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp_image(dir.path(), "hit.png");
        let native = NativeToolDispatcher;

        for tool in ["content_search", "glob_search"] {
            let rendered = native_round_trip_tool_content(
                &native,
                ToolExecutionResult {
                    name: tool.into(),
                    output: format!("found: {path}"),
                    success: true,
                    tool_call_id: Some("tc1".into()),
                },
            );
            assert!(
                !rendered.contains("[IMAGE:"),
                "{tool} path must not be re-promoted on the read side"
            );
            assert!(
                rendered.contains(&path),
                "{tool} provider-visible content must keep the literal path"
            );
        }
    }

    #[test]
    fn native_image_gen_path_still_promotes_through_to_provider_messages() {
        // Default-allow preserved across the round trip: a real generated image
        // still becomes an `[IMAGE:...]` marker, so it routes to a vision
        // provider as before.
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp_image(dir.path(), "generated.png");
        let native = NativeToolDispatcher;

        let rendered = native_round_trip_tool_content(
            &native,
            ToolExecutionResult {
                name: "image_gen".into(),
                output: format!("saved to {path}"),
                success: true,
                tool_call_id: Some("tc1".into()),
            },
        );
        assert!(
            rendered.contains(&format!("[IMAGE:{path}]")),
            "image_gen image must still canonicalize through the round trip"
        );
    }

    #[test]
    fn native_unknown_provenance_still_promotes_on_read() {
        // contract preserved: a tool result stored WITHOUT provenance
        // (empty `tool_name`, e.g. reconstructed from a provider-wire message)
        // still has a genuine image path canonicalized on the read side.
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp_image(dir.path(), "history.png");
        let native = NativeToolDispatcher;

        let history = vec![ConversationMessage::ToolResults(vec![ToolResultMessage {
            tool_call_id: "tc1".into(),
            content: format!("Saved image to {path}"),
            tool_name: String::new(),
        }])];
        let messages = native.to_provider_messages(&history);
        assert_eq!(messages.len(), 1);
        assert!(
            native_tool_message_content(&messages[0]).contains(&format!("[IMAGE:{path}]")),
            "unknown-provenance result must still promote a real image path"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // reasoning_content pass-through tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn native_to_provider_messages_includes_reasoning_content() {
        let dispatcher = NativeToolDispatcher;
        let history = vec![ConversationMessage::AssistantToolCalls {
            text: Some("answer".into()),
            tool_calls: vec![zeroclaw_providers::ToolCall {
                id: "tc_1".into(),
                name: "shell".into(),
                arguments: "{}".into(),
                extra_content: None,
            }],
            reasoning_content: Some("thinking step".into()),
        }];

        let messages = dispatcher.to_provider_messages(&history);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "assistant");

        let payload: serde_json::Value = serde_json::from_str(&messages[0].content).unwrap();
        assert_eq!(payload["reasoning_content"].as_str(), Some("thinking step"));
        assert_eq!(payload["content"].as_str(), Some("answer"));
        assert!(payload["tool_calls"].is_array());
    }

    #[test]
    fn native_to_provider_messages_omits_reasoning_content_when_none() {
        let dispatcher = NativeToolDispatcher;
        let history = vec![ConversationMessage::AssistantToolCalls {
            text: Some("answer".into()),
            tool_calls: vec![zeroclaw_providers::ToolCall {
                id: "tc_1".into(),
                name: "shell".into(),
                arguments: "{}".into(),
                extra_content: None,
            }],
            reasoning_content: None,
        }];

        let messages = dispatcher.to_provider_messages(&history);
        assert_eq!(messages.len(), 1);

        let payload: serde_json::Value = serde_json::from_str(&messages[0].content).unwrap();
        assert!(payload.get("reasoning_content").is_none());
    }

    #[test]
    fn xml_to_provider_messages_ignores_reasoning_content() {
        let dispatcher = XmlToolDispatcher;
        let history = vec![ConversationMessage::AssistantToolCalls {
            text: Some("answer".into()),
            tool_calls: vec![zeroclaw_providers::ToolCall {
                id: "tc_1".into(),
                name: "shell".into(),
                arguments: "{}".into(),
                extra_content: None,
            }],
            reasoning_content: Some("should be ignored".into()),
        }];

        let messages = dispatcher.to_provider_messages(&history);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "assistant");
        // XmlToolDispatcher returns text only, not JSON payload
        assert_eq!(messages[0].content, "answer");
        assert!(!messages[0].content.contains("reasoning_content"));
    }

    fn native_exchange() -> Vec<ChatMessage> {
        vec![
            ChatMessage::user("do it"),
            ChatMessage::assistant(
                serde_json::json!({
                    "content": "Switching",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "model_switch",
                            "arguments": "{\"model\":\"gemini\"}",
                            "extra_content": {"google": {"thought_signature": "opaque"}}
                        },
                        {
                            "id": "call_2",
                            "name": "status<&\"'",
                            "arguments": "{}"
                        }
                    ],
                    "reasoning_content": "opaque reasoning"
                })
                .to_string(),
            ),
            ChatMessage::tool(
                serde_json::json!({
                    "tool_call_id": "call_1",
                    "content": "switch queued"
                })
                .to_string(),
            ),
            ChatMessage::tool(
                serde_json::json!({
                    "tool_call_id": "call_2",
                    "content": "ready"
                })
                .to_string(),
            ),
        ]
    }

    #[test]
    fn provider_history_projection_renders_complete_native_exchange_for_text_target() {
        let canonical = native_exchange();
        let projected =
            project_provider_visible_tool_history(&canonical, ToolTranscriptProtocol::Text);

        assert_eq!(projected.len(), 3);
        assert!(projected.iter().all(|message| message.role != "tool"));
        assert_eq!(projected[1].role, "assistant");
        assert!(projected[1].content.starts_with("Switching\n"));
        assert_eq!(projected[1].content.matches("<tool_call>").count(), 2);
        assert!(projected[1].content.contains("\"name\":\"model_switch\""));
        assert_eq!(projected[2].role, "user");
        assert!(
            projected[2]
                .content
                .contains("<tool_result name=\"model_switch\">\nswitch queued")
        );
        assert!(
            projected[2]
                .content
                .contains("<tool_result name=\"status&lt;&amp;&quot;&apos;\">\nready")
        );

        assert_eq!(
            canonical.len(),
            4,
            "projection must not mutate canonical history"
        );
        assert_eq!(canonical[2].role, "tool");
        assert!(canonical[1].content.contains("opaque reasoning"));
        assert!(canonical[1].content.contains("thought_signature"));
    }

    #[test]
    fn provider_history_projection_keeps_native_target_byte_stable() {
        let canonical = native_exchange();
        let projected =
            project_provider_visible_tool_history(&canonical, ToolTranscriptProtocol::Native);
        assert_eq!(
            serde_json::to_value(&projected).unwrap(),
            serde_json::to_value(&canonical).unwrap()
        );
    }

    #[test]
    fn provider_history_projection_removes_tool_protocol_for_none_target() {
        let canonical = native_exchange();
        let projected =
            project_provider_visible_tool_history(&canonical, ToolTranscriptProtocol::None);

        assert_eq!(projected.len(), 3);
        assert!(projected.iter().all(|message| message.role != "tool"));
        assert!(projected.iter().all(|message| {
            !message.content.contains("<tool_call>") && !message.content.contains("<tool_result")
        }));
        assert!(
            projected[1]
                .content
                .contains("Completed tool calls before the model switch")
        );
        assert!(
            projected[2]
                .content
                .contains("Tool results from before the model switch")
        );

        assert_eq!(canonical.len(), 4);
        assert_eq!(canonical[2].role, "tool");
    }

    #[test]
    fn provider_history_projection_does_not_partially_rewrite_incomplete_exchange() {
        let mut canonical = native_exchange();
        canonical.pop();
        let projected =
            project_provider_visible_tool_history(&canonical, ToolTranscriptProtocol::Text);
        assert_eq!(
            serde_json::to_value(&projected).unwrap(),
            serde_json::to_value(&canonical).unwrap()
        );
    }

    #[test]
    fn provider_history_projection_does_not_fabricate_native_calls_from_text() {
        let canonical = vec![
            ChatMessage::assistant(
                "<tool_call>\n{\"name\":\"shell\",\"arguments\":{}}\n</tool_call>",
            ),
            ChatMessage::user("[Tool results]\n<tool_result name=\"shell\">ok</tool_result>"),
        ];
        let projected =
            project_provider_visible_tool_history(&canonical, ToolTranscriptProtocol::Native);
        assert_eq!(
            serde_json::to_value(&projected).unwrap(),
            serde_json::to_value(&canonical).unwrap()
        );
        assert!(projected.iter().all(|message| message.role != "tool"));
    }
}
