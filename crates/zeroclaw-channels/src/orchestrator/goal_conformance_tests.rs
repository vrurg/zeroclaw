//! Goal continuity through the ordinary Matrix command and draft pipeline.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate, matchers::method};
use zeroclaw_api::{
    attribution::{Attributable, ChannelKind, Role},
    channel::{Channel, ChannelMessage, ProgressEvent, SendMessage},
    task::TaskStatus,
};
use zeroclaw_config::schema::{
    AliasedAgentConfig, Config, MatrixConfig, MatrixStreamMode, WireApi,
};
use zeroclaw_goal::{GoalAccountingState, GoalPauseReason, GoalResponse};
use zeroclaw_infra::{session_backend::SessionBackend, session_sqlite::SqliteSessionBackend};

const OBJECTIVE: &str = "Apply the selected option after asking the user to choose.";
const QUESTION: &str = "Please choose option A or B before I continue.";
const ANSWER: &str = "Choose A.\nKeep the second line unchanged.";
const COMPLETION: &str = "Option A is applied; the declared Goal is complete.";
const PREVIOUS_RESPONSE: &str = "The earlier request is complete.";

#[path = "goal_cold_restore_tests.rs"]
mod cold_restore;

#[derive(Clone, Copy, PartialEq, Eq)]
enum SessionScenario {
    Answered,
    PlainThenAnswered,
    AnswerMatchesObjective,
    PrefixMatchesObjective,
}

impl SessionScenario {
    fn objective(self) -> &'static str {
        if self == Self::AnswerMatchesObjective {
            ANSWER
        } else {
            OBJECTIVE
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeliveryKind {
    Message,
    Draft,
    Edit,
    Progress,
    Final,
}

/// Record native channel calls without interpreting or filtering their text.
struct RecordingMatrix {
    mode: MatrixStreamMode,
    events: tokio::sync::Mutex<Vec<(DeliveryKind, String)>>,
}

impl Attributable for RecordingMatrix {
    fn role(&self) -> Role {
        Role::Channel(ChannelKind::Matrix)
    }

    fn alias(&self) -> &str {
        "goal-test"
    }
}

#[async_trait]
impl Channel for RecordingMatrix {
    fn name(&self) -> &str {
        "matrix"
    }

    async fn send(&self, message: &SendMessage) -> anyhow::Result<()> {
        self.events
            .lock()
            .await
            .push((DeliveryKind::Message, message.content.clone()));
        Ok(())
    }

    async fn send_final(&self, message: &SendMessage) -> anyhow::Result<()> {
        self.events
            .lock()
            .await
            .push((DeliveryKind::Final, message.content.clone()));
        Ok(())
    }

    async fn listen(&self, _tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()> {
        Ok(())
    }

    fn supports_draft_updates(&self) -> bool {
        self.mode != MatrixStreamMode::Off
    }

    fn supports_multi_message_streaming(&self) -> bool {
        self.mode == MatrixStreamMode::MultiMessage
    }

    async fn send_draft(&self, message: &SendMessage) -> anyhow::Result<Option<String>> {
        self.events
            .lock()
            .await
            .push((DeliveryKind::Draft, message.content.clone()));
        Ok(Some("goal-draft".to_owned()))
    }

    async fn update_draft(&self, _recipient: &str, _id: &str, text: &str) -> anyhow::Result<()> {
        self.events
            .lock()
            .await
            .push((DeliveryKind::Edit, text.to_owned()));
        Ok(())
    }

    async fn update_draft_progress(
        &self,
        _recipient: &str,
        _id: &str,
        text: &str,
    ) -> anyhow::Result<()> {
        self.events
            .lock()
            .await
            .push((DeliveryKind::Progress, text.to_owned()));
        Ok(())
    }

    async fn update_draft_lifecycle(
        &self,
        _recipient: &str,
        _id: &str,
        event: ProgressEvent,
    ) -> anyhow::Result<()> {
        self.events
            .lock()
            .await
            .push((DeliveryKind::Progress, format!("{event:?}")));
        Ok(())
    }

    async fn finalize_draft(
        &self,
        _recipient: &str,
        _id: &str,
        text: &str,
        _suppress_voice: bool,
    ) -> anyhow::Result<()> {
        self.events
            .lock()
            .await
            .push((DeliveryKind::Final, text.to_owned()));
        Ok(())
    }
}

/// Script decisions solely for boundary testing, not live verifier evaluation.
struct ModelResponses;

impl Respond for ModelResponses {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        let messages = body["messages"].as_array().unwrap();
        let answered = messages
            .iter()
            .any(|row| row["role"] == "assistant" && row["content"] == QUESTION)
            && messages
                .iter()
                .any(|row| row["role"] == "user" && row["content"] == ANSWER);
        let text = if body["model"] == "goal-verifier" {
            if messages.iter().any(|row| {
                row["content"]
                    .as_str()
                    .is_some_and(|text| text.contains(COMPLETION))
            }) {
                json!({"decision":"complete","reason":"The declared criterion is met.","blockers":[]}).to_string()
            } else {
                json!({"decision":"blocked","reason":"User choice is required.","blockers":[{"kind":"needs_user_input","message":QUESTION}]}).to_string()
            }
        } else if answered {
            COMPLETION.to_owned()
        } else {
            QUESTION.to_owned()
        };
        model_response(&text, body["stream"] == true)
    }
}

fn model_response(text: &str, streaming: bool) -> ResponseTemplate {
    let usage = json!({"prompt_tokens":10,"completion_tokens":2,"total_tokens":12});
    if streaming {
        let chunk = json!({"choices":[{"index":0,"delta":{"content":text},"finish_reason":null}]});
        let finish = json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]});
        let totals = json!({"choices":[],"usage":usage});
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(format!(
                "data: {chunk}\n\ndata: {finish}\n\ndata: {totals}\n\ndata: [DONE]\n\n"
            ))
    } else {
        ResponseTemplate::new(200).set_body_json(json!({
                "choices":[{"message":{"role":"assistant","content":text},"finish_reason":"stop"}],"usage":usage
            }))
    }
}

async fn isolated(test_name: &str) -> bool {
    const CHILD_ENV: &str = "ZEROCLAW_MATRIX_GOAL_CONFORMANCE_CHILD";
    if std::env::var(CHILD_ENV).as_deref() == Ok(test_name) {
        return false;
    }
    let mut child = tokio::process::Command::new(std::env::current_exe().unwrap())
        .arg(test_name)
        .arg("--exact")
        .arg("--test-threads=1")
        .env(CHILD_ENV, test_name)
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let status = tokio::time::timeout(Duration::from_secs(60), child.wait())
        .await
        .unwrap()
        .unwrap();
    assert!(
        status.success(),
        "isolated Matrix conformance failed: {status}"
    );
    true
}

async fn wait_for_notice(channel: &RecordingMatrix, key: &str) {
    let notice = zeroclaw_runtime::i18n::get_required_cli_string(key);
    let delivered = tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            if channel
                .events
                .lock()
                .await
                .iter()
                .any(|(kind, text)| *kind == DeliveryKind::Message && text == &notice)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(
        delivered.is_ok(),
        "Matrix did not deliver {key}; recorded events: {:?}",
        channel.events.lock().await
    );
}

fn message(id: &str, content: &str) -> ChannelMessage {
    let mut message = ChannelMessage::new(
        id,
        "@goal-test:example.org",
        "!goal-test:example.org",
        content,
        "matrix",
        1,
    );
    message.channel_alias = Some("goal-test".to_owned());
    message
}

/// Keep recovery authority in the existing daemon owner, even in this fixture.
async fn boot_daemon(
    config: Config,
) -> tokio::task::JoinHandle<anyhow::Result<zeroclaw_runtime::daemon::DaemonExit>> {
    let mut registry = zeroclaw_runtime::daemon::DaemonRegistry::new();
    registry.register_gateway(Box::new(|_, _, _, _, _, controls, _, _, _| {
        Box::pin(async move {
            let mut shutdown = controls.unwrap().shutdown_tx.subscribe();
            while !*shutdown.borrow_and_update() {
                if shutdown.changed().await.is_err() {
                    break;
                }
            }
            Ok(())
        })
    }));
    // The registry starts no network server. All daemon state is temporary;
    // only this isolated child installs signal handlers and global owners.
    let daemon = zeroclaw_spawn::spawn!(Box::pin(zeroclaw_runtime::daemon::run(
        config,
        "127.0.0.1".to_owned(),
        0,
        registry,
        false,
        false,
    )));
    tokio::time::timeout(Duration::from_secs(10), async {
        while zeroclaw_runtime::control_plane::control_plane().is_none() {
            assert!(!daemon.is_finished(), "fixture daemon ended before boot");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("fixture daemon must initialize its canonical control plane");
    daemon
}

async fn scenario(test_name: &str, mode: MatrixStreamMode, scenario: SessionScenario) {
    if isolated(test_name).await {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ModelResponses)
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
    let mut context = super::test_channel_ctx_with_backend_channel_and_provider(
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

    let objective = scenario.objective();
    let start = message("start", &format!("/goal start --unlimited {objective}"));
    let history_key = super::runtime_conversation_history_key(&context, &start);
    assert!(backend.load(&history_key).is_empty());
    if scenario == SessionScenario::PrefixMatchesObjective {
        // Seed the completed turn through the ordinary canonical append path,
        // which updates both durable history and the live conversation cache.
        for row in [
            zeroclaw_api::model_provider::ChatMessage::user(objective),
            zeroclaw_api::model_provider::ChatMessage::assistant(PREVIOUS_RESPONSE),
        ] {
            super::append_sender_turn_durably(&context, &history_key, row).unwrap();
        }
    }
    super::process_channel_message(context.clone(), start, CancellationToken::new()).await;
    wait_for_notice(&channel, "goal-mode-paused-blocked").await;
    assert_turn_delivery(
        &channel,
        mode,
        "goal-mode-started",
        QUESTION,
        "goal-mode-paused-blocked",
    )
    .await;
    let plane = zeroclaw_runtime::control_plane::control_plane().unwrap();
    let paused = plane
        .goal_store()
        .unwrap()
        .current_goal_for_session(&history_key)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(paused.status, TaskStatus::Paused);
    let task_id = paused.id;
    let query = super::goal_execution::submit_matrix_goal(
        context.clone(),
        history_key.clone(),
        message("status", "/goal status"),
        zeroclaw_commands::goal::GoalCommand::Status,
    )
    .await
    .unwrap();
    let GoalResponse::Status(status) = query.response() else {
        panic!("{query_response:?}", query_response = query.response())
    };
    assert_eq!(status.pause_reason, Some(GoalPauseReason::NeedsUserInput));
    assert_eq!(status.accounting_state, GoalAccountingState::Complete);
    assert_eq!(status.usage_totals.as_ref().unwrap().total_tokens, 24);
    drop(query);

    if scenario == SessionScenario::PlainThenAnswered {
        channel.events.lock().await.clear();
        super::process_channel_message(
            context.clone(),
            message("plain-resume", "/goal resume"),
            CancellationToken::new(),
        )
        .await;
        wait_for_notice(&channel, "goal-mode-paused-blocked").await;
        assert_turn_delivery(
            &channel,
            mode,
            "goal-mode-resumed",
            QUESTION,
            "goal-mode-paused-blocked",
        )
        .await;
        let history = backend.load(&history_key);
        assert_eq!(history.len(), 2, "{history:?}");
        assert!(
            history
                .iter()
                .all(|row| { row.role == "assistant" && row.content == QUESTION }),
            "plain resume must retain both real replies and invent no user response: {history:?}"
        );
        let paused_again = plane
            .goal_store()
            .unwrap()
            .current_goal_for_session(&history_key)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(paused_again.id, task_id);
        assert_eq!(paused_again.status, TaskStatus::Paused);
    }

    channel.events.lock().await.clear();
    super::process_channel_message(
        context.clone(),
        message("resume", &format!("/goal resume {ANSWER}")),
        CancellationToken::new(),
    )
    .await;
    wait_for_notice(&channel, "goal-mode-completed").await;
    assert_turn_delivery(
        &channel,
        mode,
        "goal-mode-resumed",
        COMPLETION,
        "goal-mode-completed",
    )
    .await;
    let completed = plane
        .goal_store()
        .unwrap()
        .current_goal_for_session(&history_key)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(completed.id, task_id);
    assert_eq!(completed.status, TaskStatus::Completed);
    let query = super::goal_execution::submit_matrix_goal(
        context,
        history_key.clone(),
        message("status", "/goal status"),
        zeroclaw_commands::goal::GoalCommand::Status,
    )
    .await
    .unwrap();
    let GoalResponse::Terminal(status) = query.response() else {
        panic!("unexpected completed status")
    };
    assert_eq!(status.accounting_state, GoalAccountingState::Complete);
    let plain_resume = scenario == SessionScenario::PlainThenAnswered;
    assert_eq!(
        status.usage_totals.as_ref().unwrap().total_tokens,
        if plain_resume { 72 } else { 48 }
    );
    drop(query);

    let history = backend.load(&history_key);
    let mut expected = Vec::new();
    if scenario == SessionScenario::PrefixMatchesObjective {
        expected.push(("user", objective));
        expected.push(("assistant", PREVIOUS_RESPONSE));
    }
    expected.push(("assistant", QUESTION));
    if plain_resume {
        expected.push(("assistant", QUESTION));
    }
    expected.extend([("user", ANSWER), ("assistant", COMPLETION)]);
    assert_eq!(history.len(), expected.len(), "{history:?}");
    for (row, (role, text)) in history.iter().zip(expected) {
        assert_eq!((row.role.as_str(), row.content.as_str()), (role, text));
    }
    assert!(history.iter().all(|row| row.role != "system"));
    assert_eq!(
        history
            .iter()
            .filter(|row| row.role == "user" && row.content == ANSWER)
            .count(),
        1
    );
    let requests = server.received_requests().await.unwrap();
    assert_eq!(
        requests.len(),
        if plain_resume { 6 } else { 4 },
        "no successor after verified completion"
    );
    let first: Value = serde_json::from_slice(&requests[0].body).unwrap();
    let rows = first["messages"].as_array().unwrap();
    assert_eq!(rows.iter().filter(|row| row["role"] == "system").count(), 1);
    assert_eq!(rows[0]["role"], "system");
    assert!(rows[0]["content"].as_str().unwrap().contains(objective));
    let user = rows.iter().find(|row| row["role"] != "system").unwrap();
    assert_eq!(user["role"], "user");
    assert!(user["content"].as_str().unwrap().ends_with(objective));
    if scenario == SessionScenario::PrefixMatchesObjective {
        assert_eq!(rows[2]["role"], "user");
        assert_eq!(rows[2]["content"], objective);
    }
    for index in (1..requests.len()).step_by(2) {
        let verifier: Value = serde_json::from_slice(&requests[index].body).unwrap();
        assert_eq!(verifier["model"], "goal-verifier");
    }
    if plain_resume {
        let plain: Value = serde_json::from_slice(&requests[2].body).unwrap();
        let rows = plain["messages"].as_array().unwrap();
        assert!(
            rows.iter()
                .any(|row| row["role"] == "assistant" && row["content"] == QUESTION)
        );
        assert!(
            !rows
                .iter()
                .any(|row| row["role"] == "user" && row["content"] == ANSWER),
            "plain resume must not fabricate an answer"
        );
    }
    let resumed: Value = serde_json::from_slice(&requests[requests.len() - 2].body).unwrap();
    let rows = resumed["messages"].as_array().unwrap();
    assert!(
        rows.iter()
            .any(|row| row["role"] == "assistant" && row["content"] == QUESTION)
    );
    assert!(
        rows.iter()
            .any(|row| row["role"] == "user" && row["content"] == ANSWER)
    );
    // This runs only in the isolated Unix child; it cannot signal the parent
    // test harness or any operator daemon. Exercise cooperative teardown.
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

async fn assert_turn_delivery(
    channel: &RecordingMatrix,
    mode: MatrixStreamMode,
    admission: &str,
    candidate: &str,
    notice: &str,
) {
    let events = channel.events.lock().await;
    let admitted = zeroclaw_runtime::i18n::get_required_cli_string(admission);
    assert_eq!(events[0].0, DeliveryKind::Message);
    assert!(events[0].1.contains(&admitted), "{events:?}");
    let final_index = events
        .iter()
        .position(|(kind, text)| *kind == DeliveryKind::Final && text == candidate)
        .unwrap_or_else(|| panic!("missing original candidate: {events:?}"));
    assert_eq!(
        events
            .iter()
            .filter(|(kind, text)| *kind == DeliveryKind::Final && text == candidate)
            .count(),
        1
    );
    let notice = zeroclaw_runtime::i18n::get_required_cli_string(notice);
    let notice_index = events
        .iter()
        .position(|(kind, text)| *kind == DeliveryKind::Message && text == &notice)
        .unwrap();
    assert!(final_index < notice_index, "{events:?}");
    if mode == MatrixStreamMode::Off {
        assert!(
            events
                .iter()
                .all(|(kind, _)| !matches!(kind, DeliveryKind::Draft | DeliveryKind::Edit))
        );
    } else {
        let draft_index = events
            .iter()
            .position(|(kind, _)| *kind == DeliveryKind::Draft)
            .unwrap();
        assert!(draft_index > 0 && draft_index < final_index, "{events:?}");
        assert!(
            events
                .iter()
                .any(|(kind, _)| matches!(kind, DeliveryKind::Edit | DeliveryKind::Progress)),
            "configured draft updates must survive: {events:?}"
        );
    }
}

#[tokio::test]
async fn matrix_goal_preserves_question_and_multiline_resume_without_streaming() {
    scenario("orchestrator::goal_conformance_tests::matrix_goal_preserves_question_and_multiline_resume_without_streaming", MatrixStreamMode::Off, SessionScenario::Answered).await;
}

#[tokio::test]
async fn matrix_goal_preserves_question_and_multiline_resume_with_partial_draft() {
    scenario("orchestrator::goal_conformance_tests::matrix_goal_preserves_question_and_multiline_resume_with_partial_draft", MatrixStreamMode::Partial, SessionScenario::Answered).await;
}

#[tokio::test]
async fn matrix_goal_preserves_question_and_multiline_resume_with_single_message() {
    scenario("orchestrator::goal_conformance_tests::matrix_goal_preserves_question_and_multiline_resume_with_single_message", MatrixStreamMode::SingleMessage, SessionScenario::Answered).await;
}

#[tokio::test]
async fn matrix_goal_preserves_question_and_multiline_resume_with_multi_message() {
    scenario("orchestrator::goal_conformance_tests::matrix_goal_preserves_question_and_multiline_resume_with_multi_message", MatrixStreamMode::MultiMessage, SessionScenario::Answered).await;
}

#[tokio::test]
async fn matrix_goal_preserves_plain_resume_without_answer() {
    scenario(
        "orchestrator::goal_conformance_tests::matrix_goal_preserves_plain_resume_without_answer",
        MatrixStreamMode::SingleMessage,
        SessionScenario::PlainThenAnswered,
    )
    .await;
}

#[tokio::test]
async fn matrix_goal_preserves_an_answer_equal_to_the_objective() {
    scenario("orchestrator::goal_conformance_tests::matrix_goal_preserves_an_answer_equal_to_the_objective", MatrixStreamMode::SingleMessage, SessionScenario::AnswerMatchesObjective).await;
}

#[tokio::test]
async fn matrix_goal_preserves_a_canonical_user_prefix_equal_to_the_objective() {
    scenario("orchestrator::goal_conformance_tests::matrix_goal_preserves_a_canonical_user_prefix_equal_to_the_objective", MatrixStreamMode::SingleMessage, SessionScenario::PrefixMatchesObjective).await;
}
