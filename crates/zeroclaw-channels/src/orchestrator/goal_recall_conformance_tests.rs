//! Recall lifetime across real Matrix admission, pause, resume and continuation.

use super::*;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};
use zeroclaw_api::memory_traits::{Memory, MemoryCategory, MemoryEntry};

const PROGRESS: &str = "Option A is selected; one declared criterion remains.";

/// Fixed verdicts exercise the transport/controller, not live model judgment.
#[derive(Default)]
struct RecallResponses(AtomicUsize);

impl Respond for RecallResponses {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        let text = if body["model"] == "goal-verifier" {
            let messages = body["messages"].to_string();
            if messages.contains(COMPLETION) {
                json!({"decision":"complete","reason":"The declared criterion is met.","blockers":[]}).to_string()
            } else if messages.contains(PROGRESS) {
                json!({"decision":"continue","reason":"Finish the remaining declared criterion.","blockers":[]}).to_string()
            } else {
                json!({"decision":"blocked","reason":"User choice is required.","blockers":[{"kind":"needs_user_input","message":QUESTION}]}).to_string()
            }
        } else {
            match self.0.fetch_add(1, Ordering::SeqCst) {
                0 => QUESTION.to_owned(),
                1 => PROGRESS.to_owned(),
                _ => COMPLETION.to_owned(),
            }
        };
        model_response(&text, body["stream"] == true)
    }
}

/// Recall is observable; any attempted memory mutation fails the fixture.
#[derive(Default)]
struct RecallMemory {
    queries: Mutex<Vec<(String, Option<String>)>>,
}

impl Attributable for RecallMemory {
    fn role(&self) -> Role {
        Role::Memory(zeroclaw_api::attribution::MemoryKind::InMemory)
    }
    fn alias(&self) -> &str {
        "goal-recall-fixture"
    }
}

#[async_trait]
impl Memory for RecallMemory {
    fn name(&self) -> &str {
        "goal-recall-fixture"
    }
    async fn store(
        &self,
        _: &str,
        _: &str,
        _: MemoryCategory,
        _: Option<&str>,
    ) -> anyhow::Result<()> {
        panic!("Goal recall must not write memory")
    }
    async fn store_with_agent(
        &self,
        _: &str,
        _: &str,
        _: MemoryCategory,
        _: Option<&str>,
        _: Option<&str>,
        _: Option<f64>,
        _: Option<&str>,
    ) -> anyhow::Result<()> {
        panic!("Goal recall must not write memory")
    }
    async fn recall(
        &self,
        query: &str,
        _: usize,
        session: Option<&str>,
        _: Option<&str>,
        _: Option<&str>,
    ) -> anyhow::Result<Vec<MemoryEntry>> {
        self.queries
            .lock()
            .unwrap()
            .push((query.to_owned(), session.map(str::to_owned)));
        Ok(vec![
            serde_json::from_value(json!({
                "id":"goal-recall-entry", "key":"repair-guidance",
                "content":"Keep the bounded repair within its declared criterion.",
                "category":"core", "timestamp":chrono::Utc::now().to_rfc3339(),
                "session_id":session, "score":null
            }))
            .unwrap(),
        ])
    }
    async fn recall_for_agents(
        &self,
        _: &[&str],
        query: &str,
        limit: usize,
        session: Option<&str>,
        since: Option<&str>,
        until: Option<&str>,
    ) -> anyhow::Result<Vec<MemoryEntry>> {
        // The fixture entry is legacy/unattributed, admitted by every allowlist.
        self.recall(query, limit, session, since, until).await
    }
    async fn get(&self, _: &str) -> anyhow::Result<Option<MemoryEntry>> {
        Ok(None)
    }
    async fn list(
        &self,
        _: Option<&MemoryCategory>,
        _: Option<&str>,
    ) -> anyhow::Result<Vec<MemoryEntry>> {
        Ok(Vec::new())
    }
    async fn forget(&self, _: &str) -> anyhow::Result<bool> {
        panic!("read-only fixture")
    }
    async fn forget_for_agent(&self, _: &str, _: &str) -> anyhow::Result<bool> {
        panic!("read-only fixture")
    }
    async fn count(&self) -> anyhow::Result<usize> {
        Ok(1)
    }
    async fn health_check(&self) -> bool {
        true
    }
}

async fn scenario(test_name: &str, recalled: bool) {
    if isolated(test_name).await {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(RecallResponses::default())
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
        ..Default::default()
    };
    config.agents.insert("test".into(), agent_config.clone());
    config.channels.matrix.insert(
        "goal-test".into(),
        MatrixConfig {
            enabled: true,
            stream_mode: MatrixStreamMode::Off,
            ..Default::default()
        },
    );
    let daemon = boot_daemon(config.clone()).await;
    let (provider, provider_ref, model, _) =
        zeroclaw_runtime::agent::agent::build_session_model_provider(
            &config,
            "openai.goal-test",
            Some("goal-worker"),
        )
        .unwrap();
    let backend = Arc::new(SqliteSessionBackend::new(tmp.path()).unwrap());
    let channel = Arc::new(RecordingMatrix {
        mode: MatrixStreamMode::Off,
        events: Default::default(),
    });
    let memory = Arc::new(RecallMemory::default());
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
    ctx.workspace_dir = Arc::new(tmp.path().to_owned());
    ctx.prompt_config = Arc::new(config.clone());
    *ctx.live_config.write() = config;
    if recalled {
        ctx.memory = memory.clone();
    }
    let start_command = format!("/goal start --unlimited {OBJECTIVE}");
    let start = message("start", &start_command);
    let history_key = super::super::runtime_conversation_history_key(&context, &start);
    let mut allowed_scopes = super::super::sender_memory_session_ids(&start, &history_key);
    if super::super::is_group_reply_target(&start.reply_target) {
        allowed_scopes.push(history_key.clone());
    }
    super::super::process_channel_message(context.clone(), start, CancellationToken::new()).await;
    wait_for_notice(&channel, "goal-mode-paused-blocked").await;
    assert_turn_delivery(
        &channel,
        MatrixStreamMode::Off,
        "goal-mode-started",
        QUESTION,
        "goal-mode-paused-blocked",
    )
    .await;
    channel.events.lock().await.clear();
    let resume_command = format!("/goal resume {ANSWER}");
    super::super::process_channel_message(
        context.clone(),
        message("resume", &resume_command),
        CancellationToken::new(),
    )
    .await;
    wait_for_notice(&channel, "goal-mode-completed").await;
    assert_turn_delivery(
        &channel,
        MatrixStreamMode::Off,
        "goal-mode-resumed",
        COMPLETION,
        "goal-mode-completed",
    )
    .await;
    let events = channel.events.lock().await;
    let finals: Vec<_> = events
        .iter()
        .filter(|(kind, _)| *kind == DeliveryKind::Final)
        .map(|(_, text)| text.as_str())
        .collect();
    assert_eq!(
        finals,
        vec![PROGRESS, COMPLETION],
        "Continue must preserve both ordinary replies"
    );
    drop(events);
    let history = backend.load(&history_key);
    let expected = [
        ("assistant", QUESTION),
        ("user", ANSWER),
        ("assistant", PROGRESS),
        ("assistant", COMPLETION),
    ];
    assert_eq!(history.len(), expected.len());
    for (row, (role, content)) in history.iter().zip(expected) {
        assert_eq!((row.role.as_str(), row.content.as_str()), (role, content));
    }
    let requests = server.received_requests().await.unwrap();
    assert_eq!(
        requests.len(),
        6,
        "one verifier per parent and no successor after completion"
    );
    for (index, request) in requests.iter().enumerate() {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        let rows = body["messages"].as_array().unwrap();
        let blocks: usize = rows
            .iter()
            .map(|row| {
                row["content"]
                    .as_str()
                    .unwrap_or_default()
                    .matches(zeroclaw_memory::MEMORY_CONTEXT_OPEN)
                    .count()
            })
            .sum();
        assert_eq!(
            blocks,
            usize::from(recalled && index % 2 == 0),
            "provider-only recall count in request {index}"
        );
        if index % 2 == 0 {
            let last_user = rows.iter().rev().find(|row| row["role"] == "user").unwrap();
            assert_eq!(
                last_user["content"]
                    .as_str()
                    .unwrap()
                    .starts_with(zeroclaw_memory::MEMORY_CONTEXT_OPEN),
                recalled
            );
            if index == 2 {
                assert!(last_user["content"].as_str().unwrap().ends_with(ANSWER));
            }
            if index == 4 {
                assert!(
                    rows.iter()
                        .any(|row| row["role"] == "user" && row["content"] == ANSWER),
                    "continuation preserves the original answer without stale recall"
                );
            }
            if index > 0 {
                assert!(
                    rows.iter()
                        .any(|row| row["role"] == "assistant" && row["content"] == QUESTION)
                );
            }
        }
    }
    if recalled {
        let queries = memory.queries.lock().unwrap();
        assert!(!queries.is_empty());
        assert!(queries.iter().all(|(query, scope)| {
            (query == &start_command || query == &resume_command)
                && scope
                    .as_ref()
                    .is_some_and(|scope| allowed_scopes.contains(scope))
        }));
        assert!(queries.iter().any(|(query, _)| query == &start_command));
        assert!(queries.iter().any(|(query, _)| query == &resume_command));
    } else {
        assert!(memory.queries.lock().unwrap().is_empty());
    }
    let status = tokio::process::Command::new("kill")
        .args(["-TERM", &std::process::id().to_string()])
        .status()
        .await
        .unwrap();
    assert!(status.success());
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
async fn matrix_goal_recall_conformance_preserves_question_answer_and_continuation() {
    scenario("orchestrator::goal_conformance_tests::recall::matrix_goal_recall_conformance_preserves_question_answer_and_continuation", true).await;
}

#[tokio::test]
async fn matrix_goal_recall_conformance_none_backend_control() {
    scenario("orchestrator::goal_conformance_tests::recall::matrix_goal_recall_conformance_none_backend_control", false).await;
}

#[tokio::test]
async fn matrix_ordinary_recall_fixture_control() {
    if isolated(
        "orchestrator::goal_conformance_tests::recall::matrix_ordinary_recall_fixture_control",
    )
    .await
    {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(RecallResponses::default())
        .mount(&server)
        .await;
    let mut config = Config::default();
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
    let (provider, _, _, _) = zeroclaw_runtime::agent::agent::build_session_model_provider(
        &config,
        "openai.goal-test",
        Some("goal-worker"),
    )
    .unwrap();
    let backend = Arc::new(SqliteSessionBackend::new(tmp.path()).unwrap());
    let channel = Arc::new(RecordingMatrix {
        mode: MatrixStreamMode::Off,
        events: Default::default(),
    });
    let memory = Arc::new(RecallMemory::default());
    let mut context = super::super::test_channel_ctx_with_backend_channel_and_provider(
        backend.clone(),
        channel,
        provider.into(),
        0,
    );
    Arc::get_mut(&mut context).unwrap().memory = memory.clone();
    let query = "Ordinary recall control.";
    let ordinary = message("ordinary", query);
    let durable_user = super::super::timestamped_channel_user_history_content(
        &ordinary,
        super::super::WHATSAPP_CURRENT_GROUP_MESSAGE_LABEL,
    );
    let history_key = super::super::runtime_conversation_history_key(&context, &ordinary);
    super::super::process_channel_message(
        context.clone(),
        ordinary,
        tokio_util::sync::CancellationToken::new(),
    )
    .await;
    let requests = server.received_requests().await.unwrap();
    assert_eq!(
        requests.len(),
        2,
        "ordinary reply-intent precheck and parent"
    );
    let precheck: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert!(
        !precheck["messages"]
            .to_string()
            .contains(zeroclaw_memory::MEMORY_CONTEXT_OPEN)
    );
    let body: Value = serde_json::from_slice(&requests[1].body).unwrap();
    let rows = body["messages"].as_array().unwrap();
    let user = rows.iter().rev().find(|row| row["role"] == "user").unwrap();
    let content = user["content"].as_str().unwrap();
    assert!(content.starts_with(zeroclaw_memory::MEMORY_CONTEXT_OPEN));
    assert!(content.ends_with(query));
    let history = backend.load(&history_key);
    assert!(
        history
            .iter()
            .any(|row| row.role == "user" && row.content == durable_user)
    );
    assert!(
        history
            .iter()
            .all(|row| !row.content.contains(zeroclaw_memory::MEMORY_CONTEXT_OPEN))
    );
    let queries = memory.queries.lock().unwrap();
    assert!(!queries.is_empty());
    assert!(queries.iter().all(|(text, _)| text == query));
}
