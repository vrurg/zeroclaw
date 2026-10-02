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

/// Reject an empty, fully accounted primary completion; retain the verifier route.
pub(super) struct FallbackReceiptResponses;

pub(super) struct SwitchReceiptResponses;

impl Respond for SwitchReceiptResponses {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_ne!(
            body["stream"], true,
            "switch fixture uses a configured buffered route"
        );
        let message = match body["model"].as_str().unwrap() {
            "goal-worker" => {
                json!({"role":"assistant","content":null,"tool_calls":[{"id":"switch-call","type":"function","function":{"name":"model_switch","arguments":json!({"action":"set","model_provider":"openai.goal-target","model":"goal-target"}).to_string()}}]})
            }
            "goal-target" => {
                let echo_seen = body["messages"].as_array().unwrap().iter().any(|row| {
                    row["role"] == "tool"
                        && row["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains("fixed echo evidence")
                });
                if echo_seen {
                    json!({"role":"assistant","content":TEXT})
                } else {
                    json!({"role":"assistant","content":null,"tool_calls":[{"id":"target-echo-call","type":"function","function":{"name":TOOL_NAME,"arguments":"{}"}}]})
                }
            }
            "goal-verifier" => {
                json!({"role":"assistant","content":json!({"decision":"complete","reason":"The bounded echo check succeeded.","blockers":[]}).to_string()})
            }
            other => panic!("unexpected switch fixture model: {other}"),
        };
        ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"finish_reason":if message["tool_calls"].is_array() {"tool_calls"} else {"stop"},"message":message}],
            "usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}
        }))
    }
}

impl Respond for FallbackReceiptResponses {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        if body["model"] == "goal-worker" {
            assert_ne!(
                body["stream"], true,
                "the configured buffered fixture must not stream"
            );
            return ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
            }));
        }
        ReceiptResponses.respond(request)
    }
}

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
    assert_receipt_presentation(fixture, transport, remote, ReceiptScenario::Ordinary).await;
}

pub(super) async fn assert_fallback_receipt_scenario(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
    remote: bool,
) {
    assert_receipt_presentation(fixture, transport, remote, ReceiptScenario::Fallback).await;
}

pub(super) async fn assert_switch_receipt_scenario(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
    remote: bool,
) {
    assert_receipt_presentation(fixture, transport, remote, ReceiptScenario::Switch).await;
}

pub(super) async fn assert_ordinary_switch_control(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
    remote: bool,
) {
    assert_ordinary_receipt_control(fixture, transport, remote, false).await;
}

pub(super) async fn assert_ordinary_fallback_control(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
    remote: bool,
) {
    assert_ordinary_receipt_control(fixture, transport, remote, true).await;
}

async fn assert_ordinary_receipt_control(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
    remote: bool,
    fallback: bool,
) {
    let tui_sig = fixture.context.tui_registry.sign(TUI_ID).unwrap();
    transport.send("initialize", json!({"protocol_version":1,"tui_id":TUI_ID,"tui_sig":tui_sig,"auth_token":remote.then_some("zc_goal_test")}), 1).await;
    let initialized = transport.receive().await;
    assert_eq!(initialized["id"], 1);
    assert!(initialized["error"].is_null());
    transport
        .send(
            "session/prompt",
            json!({"session_id":SESSION_ID,"prompt":"Complete the bounded echo check."}),
            2,
        )
        .await;
    let mut acknowledged = false;
    let mut finished = false;
    let mut text = String::new();
    let mut terminal_text = None;
    let mut calls = [0, 0];
    let mut results = [0, 0];
    while !(acknowledged && finished) {
        let frame = transport.receive().await;
        assert!(frame["error"].is_null(), "{frame}");
        match frame["method"].as_str() {
            Some("session/update") => {
                let index = match frame["params"]["name"].as_str() {
                    Some("model_switch") => Some(0),
                    Some(TOOL_NAME) => Some(1),
                    _ => None,
                };
                if let Some(index) = index {
                    match frame["params"]["type"].as_str() {
                        Some("tool_call") => calls[index] += 1,
                        Some("tool_result") => results[index] += 1,
                        _ => {}
                    }
                }
                let update: SessionUpdateEvent =
                    serde_json::from_value(frame["params"].clone()).unwrap();
                match update {
                    SessionUpdateEvent::AgentMessageChunk { text: chunk, .. } => {
                        text.push_str(&chunk)
                    }
                    SessionUpdateEvent::TurnComplete {
                        outcome, content, ..
                    } => {
                        assert_eq!(serde_json::to_value(outcome).unwrap(), "completed");
                        assert!(content.starts_with(TEXT));
                        terminal_text = Some(content);
                        finished = true;
                    }
                    _ => {}
                }
            }
            None if frame["id"] == 2 => {
                assert!(frame["result"].is_object());
                acknowledged = true;
            }
            other => panic!("unexpected ordinary switch event: {other:?}: {frame}"),
        }
    }
    assert_eq!(calls, [usize::from(!fallback), 1]);
    assert_eq!(results, calls);
    let expected_notice = crate::i18n::get_required_cli_string_with_args(
        "turn-model-fallback-notice",
        &[
            ("requested_model", "goal-worker"),
            ("requested_provider", "openai"),
            ("actual_model", "goal-fallback"),
            ("actual_provider", "openai"),
        ],
    );
    assert_eq!(
        text,
        if fallback {
            format!("{TEXT}\n\n{expected_notice}")
        } else {
            TEXT.into()
        }
    );
    let terminal_text = terminal_text.unwrap();
    assert!(
        terminal_text
            .strip_prefix(TEXT)
            .unwrap()
            .starts_with("\n\n---\nTool receipts:")
    );
    assert_eq!(terminal_text.matches("Tool receipts:").count(), 1);
    assert_eq!(
        terminal_text.matches("model_switch: zc-receipt-").count(),
        usize::from(!fallback)
    );
    assert_eq!(
        terminal_text.matches("fixture_echo: zc-receipt-").count(),
        1
    );
    let requests = fixture.server.received_requests().await.unwrap();
    let bodies: Vec<Value> = requests
        .iter()
        .map(|request| serde_json::from_slice(&request.body).unwrap())
        .collect();
    assert_eq!(bodies.len(), if fallback { 4 } else { 3 });
    assert_eq!(
        bodies
            .iter()
            .map(|body| body["model"].as_str().unwrap())
            .collect::<Vec<_>>(),
        if fallback {
            vec![
                "goal-worker",
                "goal-fallback",
                "goal-worker",
                "goal-fallback",
            ]
        } else {
            vec!["goal-worker", "goal-target", "goal-target"]
        }
    );
    for body in &bodies {
        let rows = body["messages"].as_array().unwrap();
        assert_eq!(rows.iter().filter(|row| row["role"] == "system").count(), 1);
        assert!(
            !rows[0]["content"]
                .as_str()
                .unwrap()
                .contains("Untrusted user-declared success criterion follows.")
        );
    }
    assert_eq!(
        terminal_text.matches(&expected_notice).count(),
        usize::from(fallback)
    );
    if fallback {
        assert!(terminal_text.ends_with(&expected_notice));
    }
    let echo = bodies.last().unwrap()["messages"]
        .as_array()
        .unwrap()
        .iter()
        .rev()
        .find(|row| {
            row["role"] == "tool"
                && row["content"]
                    .as_str()
                    .unwrap()
                    .contains("fixed echo evidence")
        })
        .unwrap();
    assert!(
        echo["content"]
            .as_str()
            .unwrap()
            .contains("[receipt: zc-receipt-")
    );
}

enum ReceiptScenario {
    Ordinary,
    Fallback,
    Switch,
}

async fn assert_receipt_presentation(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
    remote: bool,
    scenario: ReceiptScenario,
) {
    let fallback = matches!(scenario, ReceiptScenario::Fallback);
    let switch = matches!(scenario, ReceiptScenario::Switch);
    let tui_sig = fixture.context.tui_registry.sign(TUI_ID).unwrap();
    transport.send("initialize", json!({"protocol_version":1,"tui_id":TUI_ID,"tui_sig":tui_sig,"auth_token":remote.then_some("zc_goal_test")}), 1).await;
    let initialized = transport.receive().await;
    assert_eq!(initialized["id"], 1);
    assert!(initialized["error"].is_null(), "{initialized}");
    transport.send("session/goal", json!({"session_id":SESSION_ID,"command":"/goal start --unlimited Complete the bounded echo check."}), 2).await;
    let started = transport.receive().await;
    assert_eq!(started["id"], 2, "admission must precede events");
    assert!(started["error"].is_null(), "{started}");
    let task_id = started["result"]["response"]["status"]["task_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let mut chunks = String::new();
    let mut receipt_chunks = 0;
    let expected_notice = crate::i18n::get_required_cli_string_with_args(
        "turn-model-fallback-notice",
        &[
            ("requested_model", "goal-worker"),
            ("requested_provider", "openai"),
            ("actual_model", "goal-fallback"),
            ("actual_provider", "openai"),
        ],
    );
    let mut fallback_chunks = 0;
    let mut candidate_seen = false;
    let mut switch_calls = 0;
    let mut switch_results = 0;
    let mut switch_error_visible = false;
    loop {
        let frame = transport.receive().await;
        assert!(frame["error"].is_null(), "{frame}");
        match frame["method"].as_str() {
            Some("session/update") => {
                if switch && frame["params"]["name"] == "model_switch" {
                    match frame["params"]["type"].as_str() {
                        Some("tool_call") => switch_calls += 1,
                        Some("tool_result") => switch_results += 1,
                        _ => {}
                    }
                }
                if switch && frame["params"]["type"] == "turn_complete" {
                    switch_error_visible |= frame["params"]["content"]
                        .as_str()
                        .unwrap_or_default()
                        .contains("model switch requested to openai.goal-target goal-target");
                }
                let update: SessionUpdateEvent =
                    serde_json::from_value(frame["params"].clone()).unwrap();
                if let SessionUpdateEvent::AgentMessageChunk { text, .. } = update {
                    if text == format!("\n\n{expected_notice}") {
                        fallback_chunks += 1;
                        assert_eq!(receipt_chunks, 1, "fallback must follow the receipt chunk");
                    }
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
                    SessionGoalUpdate::Failed { reason, .. } if switch => {
                        assert_eq!((switch_calls, switch_results), (1, 1));
                        assert!(
                            switch_error_visible,
                            "the original switch error precedes the Goal failure"
                        );
                        let records: Vec<zeroclaw_config::cost::types::CostRecord> =
                            std::fs::read_to_string(
                                fixture
                                    .context
                                    .config
                                    .read()
                                    .data_dir
                                    .join("state/costs.jsonl"),
                            )
                            .unwrap()
                            .lines()
                            .map(|line| serde_json::from_str(line).unwrap())
                            .filter(|row: &zeroclaw_config::cost::types::CostRecord| {
                                row.task_id.as_deref() == Some(task_id.as_str())
                            })
                            .collect();
                        assert_eq!(records.len(), 1);
                        assert_eq!(records[0].provider_ref.as_deref(), Some("openai.goal-test"));
                        assert_eq!(records[0].usage.model, "goal-worker");
                        assert_eq!(records[0].usage.total_tokens, 12);
                        assert_eq!(fixture.server.received_requests().await.unwrap().len(), 1);
                        assert!(
                            fixture
                                .backend
                                .load(&format!("rpc_{SESSION_ID}"))
                                .is_empty()
                        );
                        panic!(
                            "a permitted switch must continue the same parent turn, not fail: {reason:?}"
                        );
                    }
                    other => panic!("unexpected Goal outcome: {other:?}"),
                }
            }
            other => panic!("unexpected receipt transport event: {other:?}: {frame}"),
        }
    }
    assert_eq!(receipt_chunks, 1);
    if switch {
        assert_eq!((switch_calls, switch_results), (1, 1));
        assert!(!switch_error_visible);
    }
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
    let successful: Vec<_> = bodies
        .iter()
        .filter(|body| !fallback || body["model"] != "goal-worker")
        .collect();
    assert_eq!(
        successful.len(),
        if switch { 4 } else { 3 },
        "all accepted worker calls and one verifier"
    );
    if fallback {
        assert_eq!(
            bodies.len(),
            5,
            "two rejected primary completions, two fallbacks and one verifier"
        );
        assert_eq!(
            bodies
                .iter()
                .filter(|body| body["model"] == "goal-worker")
                .count(),
            2
        );
        assert_eq!(
            bodies
                .iter()
                .filter(|body| body["model"] == "goal-fallback")
                .count(),
            2
        );
    }
    let tool = successful[if switch { 2 } else { 1 }]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| {
            row["role"] == "tool"
                && (!switch
                    || row["content"]
                        .as_str()
                        .unwrap_or_default()
                        .contains("fixed echo evidence"))
        })
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
    if switch {
        for request in bodies.iter().filter(|body| body["model"] == "goal-target") {
            let messages = request["messages"].as_array().unwrap();
            assert_eq!(
                messages
                    .iter()
                    .filter(|row| row["role"] == "system")
                    .count(),
                1
            );
            let system = messages[0]["content"].as_str().unwrap();
            assert!(system.contains("Untrusted user-declared success criterion follows."));
            assert!(system.contains("Complete the bounded echo check."));
        }
    }
    assert!(verifier["messages"].to_string().contains(TEXT));
    assert!(!verifier["messages"].to_string().contains("Tool receipts:"));
    assert!(!verifier["messages"].to_string().contains("zc-receipt-"));
    assert!(!verifier["messages"].to_string().contains(&expected_notice));
    transport
        .send(
            "session/goal",
            json!({"session_id":SESSION_ID,"command":"/goal status"}),
            30,
        )
        .await;
    let status = transport.receive().await;
    let result: SessionGoalResult = serde_json::from_value(status["result"].clone()).unwrap();
    let GoalResponse::Terminal(projection) = result.response else {
        panic!("completion must be terminal: {result:?}")
    };
    assert_eq!(projection.status, TaskStatus::Completed);
    assert_eq!(projection.accounting_state, GoalAccountingState::Complete);
    assert_eq!(
        projection.usage_totals.unwrap().total_tokens,
        if fallback {
            60
        } else if switch {
            48
        } else {
            36
        }
    );
    let ledger_path = fixture
        .context
        .config
        .read()
        .data_dir
        .join("state/costs.jsonl");
    let records: Vec<zeroclaw_config::cost::types::CostRecord> =
        std::fs::read_to_string(ledger_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
    let expected_routes = if fallback {
        vec![
            ("openai.goal-test", "goal-worker"),
            ("openai.goal-fallback", "goal-fallback"),
            ("openai.goal-test", "goal-worker"),
            ("openai.goal-fallback", "goal-fallback"),
            ("openai.goal-test", "goal-verifier"),
        ]
    } else if switch {
        vec![
            ("openai.goal-test", "goal-worker"),
            ("openai.goal-target", "goal-target"),
            ("openai.goal-target", "goal-target"),
            ("openai.goal-test", "goal-verifier"),
        ]
    } else {
        vec![
            ("openai.goal-test", "goal-worker"),
            ("openai.goal-test", "goal-worker"),
            ("openai.goal-test", "goal-verifier"),
        ]
    };
    assert_eq!(
        records.len(),
        expected_routes.len(),
        "no missing or duplicate attempt rows"
    );
    assert_eq!(
        records
            .iter()
            .map(|row| &row.id)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        records.len()
    );
    for (record, (provider, model)) in records.iter().zip(expected_routes) {
        assert_eq!(record.task_id.as_deref(), Some(projection.task_id.as_str()));
        assert_eq!(record.provider_ref.as_deref(), Some(provider));
        assert_eq!(record.usage.model, model);
        assert_eq!(
            (
                record.usage.input_tokens,
                record.usage.output_tokens,
                record.usage.total_tokens
            ),
            (10, 2, 12)
        );
    }
    assert_eq!(fallback_chunks, usize::from(fallback));
    assert_eq!(
        chunks.matches(&expected_notice).count(),
        usize::from(fallback)
    );
}
