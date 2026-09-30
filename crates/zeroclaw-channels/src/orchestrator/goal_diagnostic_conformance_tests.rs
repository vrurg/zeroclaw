//! Verifier diagnostic egress through ordinary Matrix ingress and streaming.

use super::*;

const NONCE: &str = "diagnostic-sibling-matrix";
const SAFE_TEXT: &str = "Slovakia locale tests pass and the api_key field is documented.";
const CANDIDATE: &str = "The bounded diagnostic fixture is complete.";

fn token() -> String {
    ["sk", "_live_", "0123456789abcdefghijklmnopqrstuv"].concat()
}

struct DiagnosticResponses {
    nonce: String,
}

impl Respond for DiagnosticResponses {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        let text = if body["model"] == "goal-verifier" {
            let jwt = [
                "eyJ",
                "fixtureHeader.eyJ",
                "fixturePayload.fixtureSignature",
            ]
            .concat();
            json!({"decision":"complete","reason":format!("{}: {SAFE_TEXT} {} {jwt}", self.nonce, token()),"blockers":[]}).to_string()
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

#[tokio::test]
async fn matrix_goal_diagnostics_guard_single_message() {
    const TEST: &str = "orchestrator::goal_conformance_tests::diagnostics::matrix_goal_diagnostics_guard_single_message";
    if isolated(TEST).await {
        return;
    }
    zeroclaw_log::try_install_capture_subscriber();
    drop(zeroclaw_log::subscribe_or_install());
    let tmp = tempfile::tempdir().unwrap();
    let server = MockServer::start().await;
    // Keep correlation distinct without making it a credential-like entropy token.
    let nonce = format!(
        "{NONCE}.{}",
        uuid::Uuid::new_v4().to_string().replace('-', ".")
    );
    Mock::given(method("POST"))
        .respond_with(DiagnosticResponses {
            nonce: nonce.clone(),
        })
        .mount(&server)
        .await;
    let mut config = Config {
        data_dir: tmp.path().to_path_buf(),
        config_path: tmp.path().join("config.toml"),
        ..Default::default()
    };
    config.goal.enabled = true;
    config.heartbeat.enabled = false;
    config.scheduler.enabled = false;
    config.goal.verifier.model_provider = "openai.goal-test".into();
    config.goal.verifier.model = Some("goal-verifier".into());
    let slot = config
        .providers
        .models
        .ensure("openai", "goal-test")
        .unwrap();
    slot.api_key = Some("test-key".into());
    slot.uri = Some(server.uri());
    slot.kind = Some("openai-compatible".into());
    slot.wire_api = Some(WireApi::ChatCompletions);
    slot.model = Some("goal-worker".into());
    slot.live_pricing = false;
    let agent_config = AliasedAgentConfig {
        enabled: true,
        model_provider: "openai.goal-test".into(),
        ..Default::default()
    };
    config.agents.insert("test".into(), agent_config.clone());
    let mode = MatrixStreamMode::SingleMessage;
    config.channels.matrix.insert(
        "goal-test".into(),
        MatrixConfig {
            enabled: true,
            stream_mode: mode,
            draft_update_interval_ms: 50,
            ..Default::default()
        },
    );
    let daemon = boot_daemon(config.clone()).await;
    // Daemon boot installs its own broadcast owner; subscribe after that cutover.
    let mut logs = zeroclaw_log::subscribe().unwrap();
    let (provider, provider_ref, model, _) =
        zeroclaw_runtime::agent::agent::build_session_model_provider(
            &config,
            "openai.goal-test",
            Some("goal-worker"),
        )
        .unwrap();
    let backend = Arc::new(SqliteSessionBackend::new(tmp.path()).unwrap());
    let channel = Arc::new(RecordingMatrix {
        mode,
        events: Default::default(),
    });
    let mut context = super::super::test_channel_ctx_with_backend_channel_and_provider(
        backend.clone(),
        channel.clone(),
        provider.into(),
        0,
    );
    let ctx = Arc::get_mut(&mut context).unwrap();
    ctx.model_provider_ref = Arc::new(provider_ref);
    ctx.model = Arc::new(model);
    ctx.agent_cfg = Arc::new(agent_config);
    ctx.workspace_dir = Arc::new(tmp.path().to_path_buf());
    ctx.prompt_config = Arc::new(config.clone());
    *ctx.live_config.write() = config;
    let start = message(
        "diagnostic-start",
        "/goal start --unlimited Complete the bounded diagnostic fixture.",
    );
    let key = super::super::runtime_conversation_history_key(&context, &start);
    super::super::process_channel_message(context.clone(), start, CancellationToken::new()).await;
    wait_for_notice(&channel, "goal-mode-completed").await;
    assert_turn_delivery(
        &channel,
        mode,
        "goal-mode-started",
        CANDIDATE,
        "goal-mode-completed",
    )
    .await;
    let plane = zeroclaw_runtime::control_plane::control_plane().unwrap();
    let task = plane
        .goal_store()
        .unwrap()
        .current_goal_for_session(&key)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(task.status, TaskStatus::Completed);
    let mut diagnostics = vec![];
    tokio::time::timeout(Duration::from_secs(5), async {
        while diagnostics.len() < 2 {
            let event = logs.recv().await.unwrap();
            if event["message"] == "Goal verifier decision and controller action"
                && event["attributes"]["task_id"] == task.id
                && event["attributes"]["reason"]
                    .as_str()
                    .is_some_and(|text| text.contains(&nonce))
            {
                diagnostics.push(event["attributes"].clone());
            }
        }
    })
    .await
    .expect("both production Matrix diagnostic frames must arrive");
    assert_eq!(diagnostics[0]["phase"], "decision_validated");
    assert_eq!(diagnostics[1]["phase"], "after_controller_action");
    for diagnostic in diagnostics {
        assert_eq!(diagnostic["validated_decision"], "complete");
        let text = diagnostic["reason"].as_str().unwrap();
        assert!(
            text.contains(&nonce) && text.contains(SAFE_TEXT),
            "valid diagnostic lost: {text}"
        );
        assert!(
            !text.contains(&token()),
            "credential reached Matrix diagnostic"
        );
        assert!(!text.contains("eyJfixtureHeader"));
        assert!(text.contains("REDACTED"));
    }
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
    let history = backend.load(&key);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].role, "assistant");
    assert_eq!(history[0].content, CANDIDATE);
    // Only the isolated fixture child receives this cooperative shutdown signal.
    let status = tokio::process::Command::new("kill")
        .args(["-TERM", &std::process::id().to_string()])
        .status()
        .await
        .unwrap();
    assert!(status.success());
    let exit = tokio::time::timeout(Duration::from_secs(10), daemon)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(
        exit,
        zeroclaw_runtime::daemon::DaemonExit::Shutdown
    ));
}
