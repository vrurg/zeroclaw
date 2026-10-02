//! Receipt presentation remains outside verified and durable Goal content.

use super::*;
use zeroclaw_api::tool::{Tool, ToolResult};

const TEXT: &str = "The bounded echo check is complete.";
const TOOL_NAME: &str = "fixture_echo";

pub(super) struct ReceiptTool;
zeroclaw_api::mock_tool_attribution!(ReceiptTool);

#[async_trait::async_trait]
impl Tool for ReceiptTool {
    fn name(&self) -> &str {
        TOOL_NAME
    }
    fn description(&self) -> &str {
        "Return fixed receipt-test evidence without external effects."
    }
    fn parameters_schema(&self) -> Value {
        json!({"type":"object","properties":{}})
    }
    async fn execute(&self, _: Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult {
            success: true,
            output: "fixed echo evidence".into(),
            error: None,
        })
    }
}

/// Fixed verdicts isolate presentation and persistence from model judgment.
pub(super) struct ReceiptResponses;

impl Respond for ReceiptResponses {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        let verifier = body["model"] == "goal-verifier";
        let tool_seen = body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["role"] == "tool");
        let message = if verifier {
            json!({"role":"assistant","content":json!({"decision":"complete","reason":"The bounded echo check succeeded.","blockers":[]}).to_string()})
        } else if tool_seen {
            json!({"role":"assistant","content":TEXT})
        } else {
            json!({"role":"assistant","content":null,"tool_calls":[{"id":"fixture-call","type":"function","function":{"name":TOOL_NAME,"arguments":"{}"}}]})
        };
        let finish_reason = if !verifier && !tool_seen {
            "tool_calls"
        } else {
            "stop"
        };
        let usage = json!({"prompt_tokens":10,"completion_tokens":2,"total_tokens":12});
        if body["stream"] == true {
            let mut delta = message.clone();
            if let Some(calls) = delta["tool_calls"].as_array_mut() {
                for (index, call) in calls.iter_mut().enumerate() {
                    call["index"] = json!(index);
                }
            }
            let chunk = json!({"choices":[{"index":0,"delta":delta,"finish_reason":null}]});
            let finish = json!({"choices":[{"index":0,"delta":{},"finish_reason":finish_reason}]});
            let totals = json!({"choices":[],"usage":usage});
            return ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {chunk}\n\ndata: {finish}\n\ndata: {totals}\n\ndata: [DONE]\n\n"
                ));
        }
        ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"message":message,"finish_reason":if !verifier && !tool_seen {"tool_calls"} else {"stop"}}],
            "usage":usage
        }))
    }
}

pub(super) async fn assert_receipt_scenario(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
    remote: bool,
) {
    let tui_sig = fixture.context.tui_registry.sign(TUI_ID).unwrap();
    transport.send("initialize", json!({"protocol_version":1,"tui_id":TUI_ID,"tui_sig":tui_sig,"auth_token":remote.then_some("zc_goal_test")}), 1).await;
    let initialized = transport.receive().await;
    assert_eq!(initialized["id"], 1);
    assert!(initialized["error"].is_null(), "{initialized}");
    transport.send("session/goal", json!({"session_id":SESSION_ID,"command":"/goal start --unlimited Complete the bounded echo check."}), 2).await;
    let started = transport.receive().await;
    assert_eq!(started["id"], 2, "admission must precede events");
    assert!(started["error"].is_null(), "{started}");
    let mut chunks = String::new();
    let mut receipt_chunks = 0;
    let mut candidate_seen = false;
    loop {
        let frame = transport.receive().await;
        assert!(frame["error"].is_null(), "{frame}");
        match frame["method"].as_str() {
            Some("session/update") => {
                let update: SessionUpdateEvent =
                    serde_json::from_value(frame["params"].clone()).unwrap();
                if let SessionUpdateEvent::AgentMessageChunk { text, .. } = update {
                    if text.contains("Tool receipts:") {
                        receipt_chunks += 1;
                        assert_eq!(chunks, TEXT, "receipt block must follow ordinary text");
                        assert!(text.starts_with("\n\n"));
                        assert!(text.contains("zc-receipt-"));
                    }
                    chunks.push_str(&text);
                }
            }
            Some("session/goal_update") => {
                let update: SessionGoalUpdate =
                    serde_json::from_value(frame["params"].clone()).unwrap();
                match update {
                    SessionGoalUpdate::VerifiedCandidate { candidate, .. } => {
                        assert_eq!(candidate, TEXT);
                        candidate_seen = true;
                    }
                    SessionGoalUpdate::Completed { .. } => {
                        assert!(candidate_seen);
                        break;
                    }
                    other => panic!("unexpected Goal outcome: {other:?}"),
                }
            }
            other => panic!("unexpected receipt transport event: {other:?}: {frame}"),
        }
    }
    assert_eq!(receipt_chunks, 1);
    assert_eq!(chunks.matches("Tool receipts:").count(), 1);
    let history = fixture.backend.load(&format!("rpc_{SESSION_ID}"));
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].role, "assistant");
    assert_eq!(history[0].content, TEXT);
    let requests = fixture.server.received_requests().await.unwrap();
    let bodies: Vec<Value> = requests
        .iter()
        .map(|request| serde_json::from_slice(&request.body).unwrap())
        .collect();
    assert_eq!(bodies.len(), 3, "two worker calls and one verifier");
    let tool = bodies[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["role"] == "tool")
        .unwrap();
    assert!(
        tool["content"]
            .as_str()
            .unwrap()
            .contains("[receipt: zc-receipt-")
    );
    let verifier = bodies
        .iter()
        .find(|body| body["model"] == "goal-verifier")
        .unwrap();
    assert!(verifier["messages"].to_string().contains(TEXT));
    assert!(!verifier["messages"].to_string().contains("Tool receipts:"));
    assert!(!verifier["messages"].to_string().contains("zc-receipt-"));
}
