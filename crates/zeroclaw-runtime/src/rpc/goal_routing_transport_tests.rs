//! ZeroCode deliberately has no channel reply-routing scope.

use super::*;

const TEXT: &str = "The bounded routing control is complete.";

pub(super) struct RoutingResponses;

impl Respond for RoutingResponses {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_ne!(
            body["stream"], true,
            "routing control uses a buffered route"
        );
        let message = if body["model"] == "goal-verifier" {
            json!({"role":"assistant","content":json!({
                "decision":"complete", "reason":"The routing control is complete.", "blockers":[]
            }).to_string()})
        } else if body["messages"].as_array().unwrap().last().unwrap()["role"] == "tool" {
            json!({"role":"assistant","content":TEXT})
        } else {
            json!({"role":"assistant","content":null,"tool_calls":[{
                "id":"routing-control-call","type":"function","function":{
                    "name":"send_via","arguments":"{\"modality\":\"text\"}"
                }
            }]})
        };
        ResponseTemplate::new(200).set_body_json(json!({
            "choices":[{"finish_reason":if message["tool_calls"].is_array() {"tool_calls"} else {"stop"},"message":message}],
            "usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}
        }))
    }
}

pub(super) async fn assert_routing_control(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
    remote: bool,
) {
    let tui_sig = fixture.context.tui_registry.sign(TUI_ID).unwrap();
    transport.send("initialize", json!({"protocol_version":1,"tui_id":TUI_ID,"tui_sig":tui_sig,"auth_token":remote.then_some("zc_goal_test")}), 1).await;
    let initialized = transport.receive().await;
    assert_eq!(initialized["id"], 1);
    assert!(initialized["error"].is_null());
    let mut ordinary_history = Vec::new();
    for goal in [false, true] {
        let request_start = fixture.server.received_requests().await.unwrap().len();
        if goal {
            transport.send("session/goal", json!({"session_id":SESSION_ID,"command":"/goal start --unlimited Complete the bounded routing control."}), 3).await;
            let admitted = transport.receive().await;
            assert_eq!(admitted["id"], 3);
            assert!(admitted["error"].is_null(), "{admitted}");
        } else {
            transport.send("session/prompt", json!({"session_id":SESSION_ID,"prompt":"Complete the bounded routing control."}), 2).await;
        }
        let mut chunks = String::new();
        let mut tool_calls = 0;
        let mut tool_results = 0;
        let mut verified = false;
        let mut ordinary_finished = false;
        let mut ordinary_acknowledged = false;
        loop {
            let frame = transport.receive().await;
            assert!(frame["error"].is_null(), "{frame}");
            match frame["method"].as_str() {
                Some("session/update") => {
                    if frame["params"]["name"] == "send_via" {
                        match frame["params"]["type"].as_str() {
                            Some("tool_call") => tool_calls += 1,
                            Some("tool_result") => tool_results += 1,
                            _ => {}
                        }
                    }
                    let update: SessionUpdateEvent =
                        serde_json::from_value(frame["params"].clone()).unwrap();
                    match update {
                        SessionUpdateEvent::AgentMessageChunk { text, .. } => {
                            chunks.push_str(&text)
                        }
                        SessionUpdateEvent::TurnComplete { outcome, .. } if !goal => {
                            assert_eq!(serde_json::to_value(outcome).unwrap(), "completed");
                            ordinary_finished = true;
                        }
                        _ => {}
                    }
                }
                Some("session/goal_update") if goal => {
                    let update: SessionGoalUpdate =
                        serde_json::from_value(frame["params"].clone()).unwrap();
                    match update {
                        SessionGoalUpdate::VerifiedCandidate { candidate, .. } => {
                            assert_eq!(candidate, TEXT);
                            verified = true;
                        }
                        SessionGoalUpdate::Completed { .. } => {
                            assert!(verified);
                            break;
                        }
                        other => panic!("unexpected routing Goal outcome: {other:?}"),
                    }
                }
                None if !goal && frame["id"] == 2 => {
                    assert!(frame["result"].is_object());
                    ordinary_acknowledged = true;
                }
                other => panic!("unexpected routing control frame: {other:?}: {frame}"),
            }
            if ordinary_finished && ordinary_acknowledged {
                break;
            }
        }
        assert_eq!(chunks, TEXT);
        assert_eq!((tool_calls, tool_results), (1, 1));
        let requests = fixture.server.received_requests().await.unwrap();
        let bodies: Vec<Value> = requests[request_start..]
            .iter()
            .map(|request| serde_json::from_slice(&request.body).unwrap())
            .collect();
        assert_eq!(bodies.len(), if goal { 3 } else { 2 });
        let rows = bodies[1]["messages"].as_array().unwrap();
        let output = rows.iter().rev().find(|row| row["role"] == "tool").unwrap()["content"]
            .as_str()
            .unwrap();
        let mut result: Value =
            serde_json::from_str(output.strip_prefix("Error: ").unwrap_or(output)).unwrap();
        if let Some(content) = result["content"].as_str() {
            result =
                serde_json::from_str(content.strip_prefix("Error: ").unwrap_or(content)).unwrap();
        }
        assert_eq!(result["status"], "ignored", "{result}");
        if goal {
            let data: Value = serde_json::from_str(
                bodies[2]["messages"][1]["content"]
                    .as_str()
                    .unwrap()
                    .split_once('\n')
                    .unwrap()
                    .1,
            )
            .unwrap();
            assert_eq!(data["candidate"], TEXT);
        } else {
            ordinary_history = fixture.backend.load(&format!("rpc_{SESSION_ID}"));
        }
    }
    let history = fixture.backend.load(&format!("rpc_{SESSION_ID}"));
    let ordinary_candidates = ordinary_history
        .iter()
        .filter(|row| row.role == "assistant" && row.content == TEXT)
        .count();
    assert_eq!(
        history
            .iter()
            .filter(|row| row.role == "assistant" && row.content == TEXT)
            .count(),
        ordinary_candidates + 1
    );
    assert_eq!(history.len(), ordinary_history.len() + 1);
    assert_eq!(
        serde_json::to_value(&history[..ordinary_history.len()]).unwrap(),
        serde_json::to_value(ordinary_history).unwrap()
    );
    assert_eq!(history.last().unwrap().role, "assistant");
    assert_eq!(history.last().unwrap().content, TEXT);
}
