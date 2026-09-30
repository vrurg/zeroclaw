//! Verifier diagnostic egress through the production ZeroCode engine factory.

use super::*;

const NONCE: &str = "diagnostic-sibling-rpc";
const SAFE_TEXT: &str = "Slovakia locale tests pass and the api_key field is documented.";
const CANDIDATE: &str = "The bounded diagnostic fixture is complete.";

fn token() -> String {
    ["sk", "_live_", "0123456789abcdefghijklmnopqrstuv"].concat()
}

fn reason(nonce: &str) -> String {
    let jwt = [
        "eyJ",
        "fixtureHeader.eyJ",
        "fixturePayload.fixtureSignature",
    ]
    .concat();
    format!("{nonce}: {SAFE_TEXT} {} {jwt}", token())
}

struct DiagnosticResponses {
    reason: String,
}

impl Respond for DiagnosticResponses {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        let text = if body["model"] == "goal-verifier" {
            json!({"decision":"complete","reason":self.reason,"blockers":[]}).to_string()
        } else {
            CANDIDATE.into()
        };
        let usage = json!({"prompt_tokens":10,"completion_tokens":2,"total_tokens":12});
        if body["stream"] == true {
            let chunk =
                json!({"choices":[{"index":0,"delta":{"content":text},"finish_reason":null}]});
            let finish = json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]});
            let totals = json!({"choices":[],"usage":usage});
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {chunk}\n\ndata: {finish}\n\ndata: {totals}\n\ndata: [DONE]\n\n"
                ))
        } else {
            ResponseTemplate::new(200).set_body_json(json!({
                "choices":[{"finish_reason":"stop","message":{"role":"assistant","content":text}}],
                "usage":usage
            }))
        }
    }
}

pub(super) async fn assert_diagnostics(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
    remote: bool,
) {
    zeroclaw_log::try_install_capture_subscriber();
    drop(zeroclaw_log::subscribe_or_install());
    let mut logs = zeroclaw_log::subscribe().unwrap();
    // Keep correlation distinct without making it a credential-like entropy token.
    let nonce = format!(
        "{NONCE}.{}",
        uuid::Uuid::new_v4().to_string().replace('-', ".")
    );
    fixture.server.reset().await;
    Mock::given(method("POST"))
        .respond_with(DiagnosticResponses {
            reason: reason(&nonce),
        })
        .mount(&fixture.server)
        .await;
    let tui_sig = fixture.context.tui_registry.sign(TUI_ID).unwrap();
    transport.send("initialize", json!({"protocol_version":1,"tui_id":TUI_ID,"tui_sig":tui_sig,"auth_token":remote.then_some("zc_goal_test")}), 1).await;
    let initialized = transport.receive().await;
    assert_eq!(initialized["id"], 1);
    assert!(initialized["error"].is_null());
    transport.send("session/goal", json!({"session_id":SESSION_ID,"command":"/goal start --unlimited Complete the bounded diagnostic fixture."}), 2).await;
    let admitted = transport.receive().await;
    assert_eq!(admitted["id"], 2, "start notice precedes worker events");
    assert!(admitted["error"].is_null());
    let result: SessionGoalResult = serde_json::from_value(admitted["result"].clone()).unwrap();
    let GoalResponse::Started(projection) = result.response else {
        panic!("expected started Goal");
    };
    let mut candidate_seen = false;
    let mut chunks = String::new();
    loop {
        let frame = transport.receive().await;
        assert!(frame["error"].is_null());
        match frame["method"].as_str() {
            Some("session/update") => {
                let update: SessionUpdateEvent =
                    serde_json::from_value(frame["params"].clone()).unwrap();
                if let SessionUpdateEvent::AgentMessageChunk { text, .. } = update {
                    chunks.push_str(&text);
                }
            }
            Some("session/goal_update") => {
                let update: SessionGoalUpdate =
                    serde_json::from_value(frame["params"].clone()).unwrap();
                match update {
                    SessionGoalUpdate::VerifiedCandidate { candidate, .. } => {
                        assert_eq!(candidate, CANDIDATE);
                        candidate_seen = true;
                    }
                    SessionGoalUpdate::Completed { .. } => {
                        assert!(candidate_seen);
                        break;
                    }
                    other => panic!("unexpected diagnostic lifecycle: {other:?}"),
                }
            }
            other => panic!("unexpected diagnostic frame: {other:?}"),
        }
    }
    assert_eq!(chunks, CANDIDATE);
    let mut diagnostics = vec![];
    let mut observed_task_frames = vec![];
    tokio::time::timeout(Duration::from_secs(5), async {
        while diagnostics.len() < 2 {
            let event = logs.recv().await.unwrap();
            if event["message"] == "Goal verifier decision and controller action"
                && event["attributes"]["task_id"] == projection.task_id
            {
                observed_task_frames.push(event["attributes"].clone());
                if event["attributes"]["reason"]
                    .as_str()
                    .is_some_and(|text| text.contains(&nonce))
                {
                    diagnostics.push(event["attributes"].clone());
                }
            }
        }
    })
    .await
    .unwrap_or_else(|error| {
        panic!("both real engine diagnostic frames must arrive: {error}; task frames: {observed_task_frames:?}")
    });
    assert_eq!(diagnostics[0]["phase"], "decision_validated");
    assert_eq!(diagnostics[1]["phase"], "after_controller_action");
    for diagnostic in diagnostics {
        assert_eq!(diagnostic["validated_decision"], "complete");
        assert_eq!(diagnostic["verifier_provider_ref"], "openai.goal-test");
        assert_eq!(diagnostic["verifier_model"], "goal-verifier");
        let text = diagnostic["reason"].as_str().unwrap();
        assert!(
            text.contains(&nonce) && text.contains(SAFE_TEXT),
            "valid diagnostic lost: {text}"
        );
        assert!(
            !text.contains(&token()),
            "credential reached host diagnostic"
        );
        assert!(!text.contains("eyJfixtureHeader"));
        assert!(text.contains("REDACTED"));
    }
    let requests = fixture.server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    let verifier: Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(verifier["model"], "goal-verifier");
}
