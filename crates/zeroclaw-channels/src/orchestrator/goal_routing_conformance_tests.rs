//! Queued routing preserves the exact candidate through delivery and verification.

use super::*;
use std::{collections::HashMap, sync::Mutex};
use zeroclaw_config::multi_agent::{OutputModality, PeerGroupConfig, PeerUsername};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouteEventKind {
    Message,
    Draft,
    Edit,
    Final,
    Cancel,
}

#[derive(Debug)]
struct RouteEvent {
    alias: &'static str,
    kind: RouteEventKind,
    message: SendMessage,
}

/// A shared sink preserves ordering across the origin and redirected channel.
struct RoutingMatrix {
    alias: &'static str,
    events: Arc<Mutex<Vec<RouteEvent>>>,
}

impl RoutingMatrix {
    fn record(&self, kind: RouteEventKind, message: SendMessage) {
        self.events.lock().unwrap().push(RouteEvent {
            alias: self.alias,
            kind,
            message,
        });
    }
}

impl Attributable for RoutingMatrix {
    fn role(&self) -> Role {
        Role::Channel(ChannelKind::Matrix)
    }
    fn alias(&self) -> &str {
        self.alias
    }
}

#[async_trait]
impl Channel for RoutingMatrix {
    fn name(&self) -> &str {
        "matrix"
    }
    async fn send(&self, message: &SendMessage) -> anyhow::Result<()> {
        self.record(RouteEventKind::Message, message.clone());
        Ok(())
    }
    async fn send_final(&self, message: &SendMessage) -> anyhow::Result<()> {
        self.record(RouteEventKind::Final, message.clone());
        Ok(())
    }
    async fn listen(&self, _: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()> {
        Ok(())
    }
    fn supports_draft_updates(&self) -> bool {
        true
    }
    async fn send_draft(&self, message: &SendMessage) -> anyhow::Result<Option<String>> {
        self.record(RouteEventKind::Draft, message.clone());
        Ok(Some("routing-draft".into()))
    }
    async fn update_draft(&self, recipient: &str, _: &str, text: &str) -> anyhow::Result<()> {
        self.record(RouteEventKind::Edit, SendMessage::new(text, recipient));
        Ok(())
    }
    async fn finalize_draft(
        &self,
        recipient: &str,
        _: &str,
        text: &str,
        suppress_voice: bool,
    ) -> anyhow::Result<()> {
        let mut message = SendMessage::new(text, recipient);
        message.suppress_voice = suppress_voice;
        self.record(RouteEventKind::Final, message);
        Ok(())
    }
    async fn cancel_draft(&self, recipient: &str, id: &str) -> anyhow::Result<()> {
        self.record(RouteEventKind::Cancel, SendMessage::new(id, recipient));
        Ok(())
    }
}

struct RoutingResponses {
    arguments: Value,
}

impl Respond for RoutingResponses {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        if body["model"] == "goal-verifier" {
            return model_response(
                &json!({"decision":"blocked","reason":"User choice is required.",
                    "blockers":[{"kind":"needs_user_input","message":QUESTION}]})
                .to_string(),
                body["stream"] == true,
            );
        }
        if body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["role"] == "tool")
        {
            return model_response(QUESTION, body["stream"] == true);
        }
        let call = json!({"id":"routing-call","type":"function",
            "function":{"name":"send_via","arguments":self.arguments.to_string()}});
        let usage = json!({"prompt_tokens":10,"completion_tokens":2,"total_tokens":12});
        if body["stream"] == true {
            let mut streamed_call = call;
            streamed_call["index"] = json!(0);
            let chunk = json!({"choices":[{"index":0,"delta":{"tool_calls":[streamed_call]},"finish_reason":null}]});
            let finish = json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]});
            let totals = json!({"choices":[],"usage":usage});
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {chunk}\n\ndata: {finish}\n\ndata: {totals}\n\ndata: [DONE]\n\n"
                ))
        } else {
            ResponseTemplate::new(200).set_body_json(json!({"choices":[{"index":0,
                "message":{"role":"assistant","content":null,"tool_calls":[call]},
                "finish_reason":"tool_calls"}],"usage":usage}))
        }
    }
}

async fn scenario(test_name: &str, modality: OutputModality, redirect: bool) {
    if isolated(test_name).await {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let server = MockServer::start().await;
    let arguments = if redirect {
        json!({"target":"routing-group"})
    } else {
        json!({"modality":if modality == OutputModality::Voice {"voice"} else {"text"}})
    };
    Mock::given(method("POST"))
        .respond_with(RoutingResponses { arguments })
        .mount(&server)
        .await;
    let mut config = Config {
        data_dir: tmp.path().to_owned(),
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
        risk_profile: "routing-test".into(),
        ..Default::default()
    };
    config.agents.insert("test".into(), agent_config.clone());
    config.channels.matrix.insert(
        "goal-test".into(),
        MatrixConfig {
            enabled: true,
            stream_mode: MatrixStreamMode::SingleMessage,
            draft_update_interval_ms: 50,
            ..Default::default()
        },
    );
    let group = PeerGroupConfig {
        channel: "matrix.peer".into(),
        external_peers: vec![PeerUsername::new("routing-recipient")],
        output_modality: modality,
        ..Default::default()
    };
    config
        .peer_groups
        .insert("routing-group".into(), group.clone());
    let risk = zeroclaw_config::schema::RiskProfileConfig {
        auto_approve: vec!["send_via".into()],
        ..Default::default()
    };
    let approvals =
        Arc::new(zeroclaw_runtime::approval::ApprovalManager::for_non_interactive(&risk));
    config.risk_profiles.insert("routing-test".into(), risk);
    let daemon = boot_daemon(config.clone()).await;
    let (provider, provider_ref, model, _) =
        zeroclaw_runtime::agent::agent::build_session_model_provider(
            &config,
            "openai.goal-test",
            Some("goal-worker"),
        )
        .unwrap();
    let backend = Arc::new(SqliteSessionBackend::new(tmp.path()).unwrap());
    let events = Arc::new(Mutex::new(Vec::new()));
    let origin = Arc::new(RoutingMatrix {
        alias: "goal-test",
        events: events.clone(),
    });
    let peer = Arc::new(RoutingMatrix {
        alias: "peer",
        events: events.clone(),
    });
    let channels: HashMap<String, Arc<dyn Channel>> = [
        ("matrix".into(), origin.clone() as Arc<dyn Channel>),
        (
            "matrix.goal-test".into(),
            origin.clone() as Arc<dyn Channel>,
        ),
        ("matrix.peer".into(), peer as Arc<dyn Channel>),
    ]
    .into_iter()
    .collect();
    let mut context = super::super::test_channel_ctx_with_backend_channel_and_provider(
        backend.clone(),
        origin,
        provider.into(),
        0,
    );
    let ctx = Arc::get_mut(&mut context).unwrap();
    ctx.model_provider_ref = Arc::new(provider_ref);
    ctx.model = Arc::new(model);
    ctx.agent_cfg = Arc::new(agent_config);
    ctx.workspace_dir = Arc::new(tmp.path().to_owned());
    ctx.channels_by_name = Arc::new(channels.clone());
    ctx.approval_manager = approvals;
    ctx.tools_registry = Arc::new(
        zeroclaw_runtime::tools::scoped::ScopedToolRegistry::from_raw_for_test(vec![Box::new(
            zeroclaw_runtime::tools::SendViaTool::new(
                Arc::new(zeroclaw_runtime::security::SecurityPolicy::default()),
                Arc::new(parking_lot::RwLock::new(channels)),
                Arc::new(move || {
                    [("routing-group".into(), group.clone())]
                        .into_iter()
                        .collect()
                }),
            ),
        )]),
    );
    ctx.prompt_config = Arc::new(config.clone());
    *ctx.live_config.write() = config;
    let start = message("start", &format!("/goal start --unlimited {OBJECTIVE}"));
    let history_key = super::super::runtime_conversation_history_key(&context, &start);
    super::super::process_channel_message(context.clone(), start, CancellationToken::new()).await;
    let paused_notice = zeroclaw_runtime::i18n::get_required_cli_string("goal-mode-paused-blocked");
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            if events
                .lock()
                .unwrap()
                .iter()
                .any(|event| event.message.content == paused_notice)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("routed question must be followed by the blocked notice");
    {
        let events = events.lock().unwrap();
        let finals: Vec<_> = events
            .iter()
            .enumerate()
            .filter(|(_, event)| event.kind == RouteEventKind::Final)
            .collect();
        assert_eq!(finals.len(), 1, "{events:?}");
        let (candidate_index, candidate) = finals[0];
        assert_eq!(candidate.message.content, QUESTION);
        assert_eq!(candidate.alias, if redirect { "peer" } else { "goal-test" });
        assert_eq!(
            candidate.message.recipient,
            if redirect {
                "routing-recipient"
            } else {
                "!goal-test:example.org"
            }
        );
        assert_eq!(
            candidate.message.suppress_voice,
            modality == OutputModality::Text
        );
        assert_eq!(
            candidate.message.force_voice,
            modality == OutputModality::Voice
        );
        let pause_index = events
            .iter()
            .position(|event| event.message.content == paused_notice)
            .unwrap();
        assert!(candidate_index < pause_index, "{events:?}");
        assert_eq!(events[pause_index].alias, "goal-test");
        let started = zeroclaw_runtime::i18n::get_required_cli_string("goal-mode-started");
        assert!(events[0].message.content.contains(&started), "{events:?}");
        assert_eq!(events[0].alias, "goal-test");
        assert!(
            events
                .iter()
                .any(|event| event.kind == RouteEventKind::Draft)
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event.kind == RouteEventKind::Cancel)
                .count(),
            usize::from(redirect || modality == OutputModality::Voice)
        );
    }
    let history = backend.load(&history_key);
    assert_eq!(history.len(), 1, "{history:?}");
    assert_eq!(
        (history[0].role.as_str(), history[0].content.as_str()),
        ("assistant", QUESTION)
    );
    let requests: Vec<Value> = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|request| request.method.as_str() == "POST")
        .map(|request| serde_json::from_slice(&request.body).unwrap())
        .collect();
    assert_eq!(requests.len(), 3, "no successor after a validated blocker");
    let tool_rows: Vec<_> = requests[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["role"] == "tool")
        .collect();
    assert_eq!(tool_rows.len(), 1);
    assert_eq!(tool_rows[0]["tool_call_id"], "routing-call");
    let output: Value = serde_json::from_str(tool_rows[0]["content"].as_str().unwrap()).unwrap();
    assert_eq!(output["status"], "queued");
    let verifier = requests
        .iter()
        .find(|request| request["model"] == "goal-verifier")
        .unwrap();
    let expected = zeroclaw_goal::GoalVerifierTurn {
        objective: OBJECTIVE.into(),
        candidate: QUESTION.into(),
    }
    .to_chat_messages();
    assert_eq!(
        verifier["messages"].as_array().unwrap().len(),
        expected.len()
    );
    for (row, expected) in verifier["messages"]
        .as_array()
        .unwrap()
        .iter()
        .zip(expected)
    {
        assert_eq!(row["role"], expected.role);
        assert_eq!(row["content"], expected.content);
    }
    let plane = zeroclaw_runtime::control_plane::control_plane().unwrap();
    let task = plane
        .goal_store()
        .unwrap()
        .current_goal_for_session(&history_key)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(task.status, TaskStatus::Paused);
    let query = super::super::goal_execution::submit_matrix_goal(
        context,
        history_key,
        message("status", "/goal status"),
        zeroclaw_commands::goal::GoalCommand::Status,
    )
    .await
    .unwrap();
    let GoalResponse::Status(status) = query.response() else {
        panic!("expected paused status")
    };
    assert_eq!(status.pause_reason, Some(GoalPauseReason::NeedsUserInput));
    assert_eq!(status.accounting_state, GoalAccountingState::Complete);
    assert_eq!(status.usage_totals.as_ref().unwrap().total_tokens, 36);
    drop(query);
    assert!(
        tokio::process::Command::new("kill")
            .args(["-TERM", &std::process::id().to_string()])
            .status()
            .await
            .unwrap()
            .success()
    );
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(10), daemon)
            .await
            .unwrap()
            .unwrap()
            .unwrap(),
        zeroclaw_runtime::daemon::DaemonExit::Shutdown
    ));
}

#[tokio::test]
async fn matrix_goal_routed_question_precedes_blocked_notice() {
    scenario(
        "orchestrator::goal_conformance_tests::routing::matrix_goal_routed_question_precedes_blocked_notice",
        OutputModality::Text,
        true,
    )
    .await;
}

#[tokio::test]
async fn matrix_goal_text_question_preserves_verifier_candidate() {
    scenario(
        "orchestrator::goal_conformance_tests::routing::matrix_goal_text_question_preserves_verifier_candidate",
        OutputModality::Text,
        false,
    )
    .await;
}

#[tokio::test]
async fn matrix_goal_voice_question_precedes_blocked_notice() {
    scenario(
        "orchestrator::goal_conformance_tests::routing::matrix_goal_voice_question_precedes_blocked_notice",
        OutputModality::Voice,
        false,
    )
    .await;
}
