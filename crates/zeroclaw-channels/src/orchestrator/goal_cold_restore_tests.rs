//! Observe cold restoration at the real parent model-request boundary.

use super::*;
use std::{
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};
use zeroclaw_api::{
    channel::ChannelConversationScope, model_provider::ChatMessage, task::TaskRecord,
};
use zeroclaw_goal::{GoalBlockerKind, GoalTaskRecord};

const EARLIER_QUESTION: &str = "Which setting should the earlier request use?";
const EARLIER_ANSWER: &str = "Use the earlier setting.";
const ORPHAN_CLOSURE: &str = "[Session interrupted — not continuing this request]";

const P2_PHASE_ONE: &str = "blocked-question-phase-one";
const P2_PHASE_TWO: &str = "blocked-question-phase-two";
const P2_SNAPSHOT: &str = "p2-blocked-snapshot.json";
const P2_BOOTED: &str = "p2-phase-two-booted";
const P2_BOOT_ACK: &str = "p2-phase-two-boot-ack";
const P2_FOREIGN_DONE: &str = "p2-foreign-resume-done";
const P2_FOREIGN_ACK: &str = "p2-foreign-resume-ack";

const P3_PHASE_ONE: &str = "graceful-shutdown-phase-one";
const P3_PHASE_TWO: &str = "graceful-shutdown-phase-two";
const P3_SNAPSHOT: &str = "p3-daemon-restart-snapshot.json";
const P3_PROVIDER_HELD: &str = "p3-provider-held";
const P3_SIGNAL_PROCESSED: &str = "p3-signal-processed";
const P3_STILL_HELD: &str = "p3-responder-still-held";
const P3_RELEASE: &str = "p3-release-response";
const P3_RESPONSE_RETURNED: &str = "p3-response-returned";
const P3_BOOTED: &str = "p3-phase-two-booted";
const P3_BOOT_ACK: &str = "p3-phase-two-boot-ack";

struct ColdPhaseFixture {
    daemon: tokio::task::JoinHandle<anyhow::Result<zeroclaw_runtime::daemon::DaemonExit>>,
    backend: Arc<SqliteSessionBackend>,
    channel: Arc<RecordingMatrix>,
    context: Arc<super::super::ChannelRuntimeContext>,
    history_key: String,
}

fn phase_inputs(test_name: &str) -> Option<(String, PathBuf, String)> {
    if std::env::var(CHILD_ENV).as_deref() != Ok(test_name) {
        return None;
    }
    Some((
        std::env::var(PHASE_ENV).expect("phase child must receive its phase"),
        PathBuf::from(
            std::env::var_os(PHASE_DATA_DIR_ENV)
                .expect("phase child must receive its durable data directory"),
        ),
        std::env::var(PHASE_MOCK_URI_ENV).expect("phase child must receive its mock-provider URI"),
    ))
}

fn marker(data_dir: &Path, name: &str) -> PathBuf {
    data_dir.join(name)
}

fn write_marker(data_dir: &Path, name: &str) {
    std::fs::write(marker(data_dir, name), b"ready").unwrap();
}

async fn wait_for_marker(data_dir: &Path, name: &str) {
    let path = marker(data_dir, name);
    tokio::time::timeout(Duration::from_secs(15), async {
        while !path.exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for marker {}", path.display()));
}

fn shared_room_message(id: &str, content: &str) -> ChannelMessage {
    let mut message = message(id, content);
    message.conversation_scope = ChannelConversationScope::ReplyTarget;
    message
}

async fn cold_phase_fixture(
    data_dir: &Path,
    mock_uri: &str,
    shared_room: bool,
) -> ColdPhaseFixture {
    let mut config = Config {
        data_dir: data_dir.to_path_buf(),
        config_path: data_dir.join("config.toml"),
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
    slot.uri = Some(mock_uri.to_owned());
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
    let backend = Arc::new(SqliteSessionBackend::new(data_dir).unwrap());
    let channel = Arc::new(RecordingMatrix {
        mode: MatrixStreamMode::Off,
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
    ctx.workspace_dir = Arc::new(data_dir.to_path_buf());
    ctx.prompt_config = Arc::new(config.clone());
    ctx.live_config = zeroclaw_config::live::LiveConfig::new(config).handle();
    let probe = if shared_room {
        shared_room_message("phase-probe", OBJECTIVE)
    } else {
        message("phase-probe", OBJECTIVE)
    };
    let history_key = super::super::runtime_conversation_history_key(&context, &probe);
    ColdPhaseFixture {
        daemon,
        backend,
        channel,
        context,
        history_key,
    }
}

async fn current_goal(fixture: &ColdPhaseFixture) -> (TaskRecord, GoalTaskRecord) {
    let store = zeroclaw_runtime::control_plane::control_plane()
        .unwrap()
        .goal_store()
        .unwrap();
    let task = store
        .current_goal_for_session(&fixture.history_key)
        .await
        .unwrap()
        .unwrap();
    let goal = store.get_goal_task(&task.id).await.unwrap().unwrap();
    (task, goal)
}

fn assert_settled(goal: &GoalTaskRecord) {
    assert_eq!(goal.accounting_state, GoalAccountingState::Complete);
    assert!(goal.pending_call_id.is_none());
    assert!(goal.pending_call_epoch.is_none());
    assert!(goal.pending_tool_batch_id.is_none());
    assert!(goal.pending_tool_epoch.is_none());
}

fn assert_request_models(requests: &[Request], expected: &[&str]) {
    assert_eq!(
        requests.len(),
        expected.len(),
        "unexpected provider requests"
    );
    for (request, model) in requests.iter().zip(expected) {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["model"], *model);
    }
}

struct HoldFirstWorkerResponse {
    data_dir: PathBuf,
    calls: AtomicUsize,
}

impl Respond for HoldFirstWorkerResponse {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        if self.calls.fetch_add(1, Ordering::SeqCst) != 0 {
            return if body["model"] == "goal-worker" {
                model_response(COMPLETION, body["stream"] == true)
            } else {
                ModelResponses.respond(request)
            };
        }
        assert_eq!(body["model"], "goal-worker");
        write_marker(&self.data_dir, P3_PROVIDER_HELD);
        let release = marker(&self.data_dir, P3_RELEASE);
        let deadline = Instant::now() + Duration::from_secs(30);
        while !release.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            release.exists(),
            "held provider response was never released"
        );
        write_marker(&self.data_dir, P3_RESPONSE_RETURNED);
        model_response(COMPLETION, body["stream"] == true)
    }
}

struct ReleaseResponseOnDrop(PathBuf);

impl Drop for ReleaseResponseOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::write(&self.0, b"release");
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendCall {
    Load,
    TryLoad,
    GetProvenance,
    Append,
    Remove,
    Replace,
    SetProvenance,
}

struct RecordingBackend {
    inner: SqliteSessionBackend,
    calls: Mutex<Vec<BackendCall>>,
    read_failure: Option<ReadFailure>,
    fail_answer: bool,
    answer_failures: AtomicUsize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadFailure {
    Transcript,
    Provenance,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ColdEntry {
    Goal,
    GoalCapped { cap: usize, oversized: bool },
    Ordinary,
}

impl RecordingBackend {
    fn record(&self, call: BackendCall) {
        self.calls.lock().unwrap().push(call);
    }
}

impl SessionBackend for RecordingBackend {
    fn load(&self, key: &str) -> Vec<ChatMessage> {
        self.record(BackendCall::Load);
        self.inner.load(key)
    }

    fn try_load(&self, key: &str) -> std::io::Result<Vec<ChatMessage>> {
        self.record(BackendCall::TryLoad);
        if self.read_failure == Some(ReadFailure::Transcript) {
            return Err(std::io::Error::other("fixture transcript unavailable"));
        }
        self.inner.try_load(key)
    }

    fn append(&self, key: &str, message: &ChatMessage) -> std::io::Result<()> {
        self.record(BackendCall::Append);
        if self.fail_answer && message.role == "user" && message.content == ANSWER {
            self.answer_failures.fetch_add(1, Ordering::SeqCst);
            return Err(std::io::Error::other(
                "fixture resume persistence unavailable",
            ));
        }
        self.inner.append(key, message)
    }

    fn remove_last(&self, key: &str) -> std::io::Result<bool> {
        self.record(BackendCall::Remove);
        self.inner.remove_last(key)
    }

    fn list_sessions(&self) -> Vec<String> {
        self.inner.list_sessions()
    }

    fn rewrite_messages(&self, key: &str, rows: &[ChatMessage]) -> std::io::Result<()> {
        self.record(BackendCall::Replace);
        self.inner.rewrite_messages(key, rows)
    }

    fn replace_conversation_state(
        &self,
        key: &str,
        rows: &[ChatMessage],
        provenance: bool,
    ) -> std::io::Result<()> {
        self.record(BackendCall::Replace);
        self.inner.replace_conversation_state(key, rows, provenance)
    }

    fn get_session_trim_breadcrumb(&self, key: &str) -> std::io::Result<Option<bool>> {
        self.record(BackendCall::GetProvenance);
        if self.read_failure == Some(ReadFailure::Provenance) {
            return Err(std::io::Error::other("fixture provenance unavailable"));
        }
        self.inner.get_session_trim_breadcrumb(key)
    }

    fn set_session_trim_breadcrumb(&self, key: &str, value: bool) -> std::io::Result<()> {
        self.record(BackendCall::SetProvenance);
        self.inner.set_session_trim_breadcrumb(key, value)
    }
}

struct RestoreObservation {
    request: Value,
    durable: Value,
    provenance: Option<bool>,
    cached_history: bool,
    cached_provenance: bool,
    calls: Vec<BackendCall>,
}

struct ColdModelResponses {
    context: Arc<super::super::ChannelRuntimeContext>,
    backend: Arc<RecordingBackend>,
    key: String,
    plain_resume: bool,
    parents: AtomicUsize,
    observations: Mutex<Vec<RestoreObservation>>,
}

impl Respond for ColdModelResponses {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        if body["model"] == "goal-verifier" {
            return ModelResponses.respond(request);
        }
        let parent = self.parents.fetch_add(1, Ordering::SeqCst);
        let streaming = body["stream"] == true;
        let reply_precheck = body["messages"][1]["content"].as_str().is_some_and(|text| {
            text.starts_with("Decide whether the assistant should send any visible reply")
        });
        self.observations.lock().unwrap().push(RestoreObservation {
            request: body,
            durable: serde_json::to_value(self.backend.inner.try_load(&self.key).unwrap()).unwrap(),
            provenance: self
                .backend
                .inner
                .get_session_trim_breadcrumb(&self.key)
                .unwrap(),
            cached_history: self
                .context
                .conversation_histories
                .lock()
                .unwrap()
                .peek(&self.key)
                .is_some(),
            cached_provenance: self
                .context
                .history_crumb_flags
                .lock()
                .unwrap()
                .peek(&self.key)
                .is_some(),
            calls: self.backend.calls.lock().unwrap().clone(),
        });
        let text = if reply_precheck {
            "REPLY"
        } else if self.plain_resume && parent == 0 {
            QUESTION
        } else {
            COMPLETION
        };
        model_response(text, streaming)
    }
}

fn assert_read_only_restore(observation: &RestoreObservation, expected: &[ChatMessage]) {
    assert_eq!(observation.durable, serde_json::to_value(expected).unwrap());
    assert_eq!(observation.provenance, Some(false));
    assert!(!observation.cached_history);
    assert!(!observation.cached_provenance);
    assert_eq!(
        observation.calls,
        [BackendCall::TryLoad, BackendCall::GetProvenance]
    );
    let rows = observation.request["messages"].as_array().unwrap();
    assert_eq!(rows.iter().filter(|row| row["role"] == "system").count(), 1);
    assert_eq!(rows[0]["role"], "system");
    assert_eq!(rows[1]["role"], "user");
    assert!(rows[1]["content"].as_str().unwrap().ends_with(OBJECTIVE));
    let restored: Vec<_> = rows[2..]
        .iter()
        .map(|row| {
            (
                row["role"].as_str().unwrap(),
                row["content"].as_str().unwrap(),
            )
        })
        .collect();
    let expected: Vec<_> = expected
        .iter()
        .map(|row| (row.role.as_str(), row.content.as_str()))
        .collect();
    assert_eq!(
        restored, expected,
        "the complete cold prefix must reach the model unchanged"
    );
}

fn assert_capped_restore(observation: &RestoreObservation, durable: &[ChatMessage], cap: usize) {
    assert_eq!(observation.durable, serde_json::to_value(durable).unwrap());
    assert_eq!(observation.provenance, Some(true));
    assert!(!observation.cached_history && !observation.cached_provenance);
    assert_eq!(
        observation.calls,
        [BackendCall::TryLoad, BackendCall::GetProvenance]
    );
    let limit = cap.min(super::super::MAX_CHANNEL_HISTORY);
    let prefix = &durable[durable.len().saturating_sub(limit)..];
    let rows = observation.request["messages"].as_array().unwrap();
    assert_eq!(rows.len(), prefix.len() + 2);
    assert_eq!(rows.iter().filter(|row| row["role"] == "system").count(), 1);
    assert_eq!(rows[0]["role"], "system");
    let objective_index = if prefix[0].role == "user" { 2 } else { 1 };
    assert_eq!(rows[objective_index]["role"], "user");
    assert!(
        rows[objective_index]["content"]
            .as_str()
            .unwrap()
            .ends_with(OBJECTIVE)
    );
    let restored: Vec<_> = rows
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != 0 && *index != objective_index)
        .map(|(_, row)| {
            (
                row["role"].as_str().unwrap(),
                row["content"].as_str().unwrap(),
            )
        })
        .collect();
    let expected: Vec<_> = prefix
        .iter()
        .map(|row| (row.role.as_str(), row.content.as_str()))
        .collect();
    assert_eq!(
        restored, expected,
        "restore caps must preserve exact rows without an orphan closure"
    );
}

async fn cold_scenario(test_name: &str, plain_resume: bool, read_failure: Option<ReadFailure>) {
    cold_scenario_inner(
        test_name,
        plain_resume,
        read_failure,
        false,
        ColdEntry::Goal,
    )
    .await;
}

async fn cold_scenario_inner(
    test_name: &str,
    plain_resume: bool,
    read_failure: Option<ReadFailure>,
    fail_answer: bool,
    entry: ColdEntry,
) {
    if isolated(test_name).await {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let server = MockServer::start().await;
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
    let mut agent_config = AliasedAgentConfig {
        enabled: true,
        model_provider: "openai.goal-test".into(),
        ..Default::default()
    };
    if entry == ColdEntry::Ordinary {
        agent_config.resolved.max_history_messages = super::super::MAX_CHANNEL_HISTORY;
    } else if let ColdEntry::GoalCapped { cap, .. } = entry {
        agent_config.resolved.max_history_messages = cap;
    }
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
    let backend = Arc::new(RecordingBackend {
        inner: SqliteSessionBackend::new(tmp.path()).unwrap(),
        calls: Mutex::new(Vec::new()),
        read_failure,
        fail_answer,
        answer_failures: AtomicUsize::new(0),
    });
    let channel = Arc::new(RecordingMatrix {
        mode: MatrixStreamMode::Off,
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
    ctx.live_config = zeroclaw_config::live::LiveConfig::new(config).handle();
    let start = if entry == ColdEntry::Ordinary {
        message("ordinary", OBJECTIVE)
    } else {
        message("start", &format!("/goal start --unlimited {OBJECTIVE}"))
    };
    let key = super::super::runtime_conversation_history_key(&context, &start);
    let mut expected = vec![
        ChatMessage::assistant(EARLIER_QUESTION),
        ChatMessage::user(EARLIER_ANSWER),
    ];
    if entry == ColdEntry::Ordinary
        || matches!(
            entry,
            ColdEntry::GoalCapped {
                oversized: true,
                ..
            }
        )
    {
        expected = vec![ChatMessage::user(
            zeroclaw_runtime::agent::history::HISTORY_TRIM_BREADCRUMB_CANONICAL,
        )];
        expected.extend(
            (0..super::super::MAX_CHANNEL_HISTORY)
                .map(|index| ChatMessage::assistant(format!("Earlier response {index}."))),
        );
        expected.push(ChatMessage::user(EARLIER_ANSWER));
    } else if matches!(entry, ColdEntry::GoalCapped { .. }) {
        expected.insert(
            0,
            ChatMessage::user(zeroclaw_runtime::agent::history::HISTORY_TRIM_BREADCRUMB_CANONICAL),
        );
    }
    // Direct backend seeding must not warm either live cache.
    for row in &expected {
        backend.inner.append(&key, row).unwrap();
    }
    backend
        .inner
        .set_session_trim_breadcrumb(&key, entry != ColdEntry::Goal)
        .unwrap();
    let responses = Arc::new(ColdModelResponses {
        context: context.clone(),
        backend: backend.clone(),
        key: key.clone(),
        plain_resume: plain_resume || fail_answer,
        parents: AtomicUsize::new(0),
        observations: Mutex::new(Vec::new()),
    });
    Mock::given(method("POST"))
        .respond_with({
            let responses = responses.clone();
            move |request: &Request| responses.respond(request)
        })
        .mount(&server)
        .await;
    let input_time_min = chrono::Local::now().timestamp();
    super::super::process_channel_message(context.clone(), start, CancellationToken::new()).await;
    let input_time_max = chrono::Local::now().timestamp();
    if entry == ColdEntry::Ordinary {
        {
            let observations = responses.observations.lock().unwrap();
            assert_eq!(
                observations.len(),
                2,
                "ordinary admissions: {:?}",
                observations
                    .iter()
                    .map(|observation| {
                        let request = &observation.request;
                        (
                            request["model"].clone(),
                            request["stream"].clone(),
                            request["messages"].as_array().unwrap().len(),
                            request["messages"][0]["content"]
                                .as_str()
                                .unwrap()
                                .contains("summar"),
                        )
                    })
                    .collect::<Vec<_>>()
            );
            assert!(
                observations[0].request["messages"][1]["content"]
                    .as_str()
                    .unwrap()
                    .starts_with("Decide whether the assistant should send any visible reply")
            );
            assert!(
                !observations[1].request["messages"][1]["content"]
                    .as_str()
                    .unwrap()
                    .starts_with("Decide whether the assistant should send any visible reply")
            );
            let capped = expected[expected.len() - super::super::MAX_CHANNEL_HISTORY..].to_vec();
            let inbound = observations[0].durable.as_array().unwrap().last().unwrap();
            assert_eq!(inbound["role"], "user");
            let inbound_content = inbound["content"].as_str().unwrap();
            let (timestamp, content) = inbound_content
                .strip_prefix('[')
                .unwrap()
                .split_once("] ")
                .unwrap();
            assert_eq!(content, OBJECTIVE);
            let input_time = chrono::DateTime::parse_from_str(timestamp, "%Y-%m-%d %H:%M:%S %:z")
                .unwrap()
                .timestamp();
            assert!((input_time_min..=input_time_max).contains(&input_time));
            let inbound = ChatMessage::user(inbound_content);
            expected = capped;
            expected.push(ChatMessage::assistant(ORPHAN_CLOSURE));
            expected.push(inbound);
            for observation in observations.iter() {
                assert_eq!(
                    observation.durable,
                    serde_json::to_value(&expected).unwrap()
                );
                assert_eq!(observation.provenance, Some(false));
                assert!(observation.cached_history && observation.cached_provenance);
                assert_eq!(
                    observation.calls,
                    [
                        BackendCall::TryLoad,
                        BackendCall::GetProvenance,
                        BackendCall::Append,
                        BackendCall::Replace,
                        BackendCall::Append
                    ]
                );
            }
        }
        expected.push(ChatMessage::assistant(COMPLETION));
        assert_eq!(
            serde_json::to_value(backend.inner.try_load(&key).unwrap()).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
        assert_eq!(
            serde_json::to_value(context.conversation_histories.lock().unwrap().peek(&key))
                .unwrap(),
            serde_json::to_value(Some(
                &expected[expected.len() - super::super::MAX_CHANNEL_HISTORY..]
            ))
            .unwrap()
        );
        assert_eq!(
            context.history_crumb_flags.lock().unwrap().peek(&key),
            Some(&false)
        );
        assert_eq!(
            backend.inner.get_session_trim_breadcrumb(&key).unwrap(),
            Some(false)
        );
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
        let registry = zeroclaw_runtime::control_plane::control_plane()
            .unwrap()
            .goal_store()
            .unwrap();
        assert!(
            registry
                .current_goal_for_session(&key)
                .await
                .unwrap()
                .is_none()
        );
        let events = channel.events.lock().await;
        assert_eq!(
            events
                .iter()
                .filter(|(kind, text)| *kind == DeliveryKind::Final && text == COMPLETION)
                .count(),
            1
        );
        for notice in [
            "goal-mode-started",
            "goal-mode-resumed",
            "goal-mode-completed",
            "goal-mode-paused-blocked",
        ] {
            let heading = zeroclaw_runtime::i18n::get_required_cli_string(notice);
            assert!(
                !events
                    .iter()
                    .any(|(_, text)| text.lines().next() == Some(heading.as_str())),
                "ordinary work must not emit Goal notices: {events:?}"
            );
        }
        drop(events);
        stop_fixture_daemon(daemon).await;
        return;
    }
    if let Some(read_failure) = read_failure {
        wait_for_notice(&channel, "goal-mode-failure-executor").await;
        assert!(server.received_requests().await.unwrap().is_empty());
        assert_eq!(
            serde_json::to_value(backend.inner.try_load(&key).unwrap()).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
        assert_eq!(
            backend.inner.get_session_trim_breadcrumb(&key).unwrap(),
            Some(false)
        );
        assert!(
            context
                .conversation_histories
                .lock()
                .unwrap()
                .peek(&key)
                .is_none()
        );
        assert!(
            context
                .history_crumb_flags
                .lock()
                .unwrap()
                .peek(&key)
                .is_none()
        );
        let expected_calls = match read_failure {
            ReadFailure::Transcript => vec![BackendCall::TryLoad],
            ReadFailure::Provenance => vec![BackendCall::TryLoad, BackendCall::GetProvenance],
        };
        assert_eq!(*backend.calls.lock().unwrap(), expected_calls);
        let plane = zeroclaw_runtime::control_plane::control_plane().unwrap();
        let registry = plane.goal_store().unwrap();
        let goal = registry
            .current_goal_for_session(&key)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(goal.status, TaskStatus::Failed);
        assert_eq!(
            registry
                .terminal_reason_for_session_goal(&goal.id, &key)
                .await
                .unwrap()
                .as_deref(),
            Some("executor_failed")
        );
        let events = channel.events.lock().await.clone();
        stop_fixture_daemon(daemon).await;
        assert!(
            events
                .iter()
                .any(|(_, text)| text.contains("Goal canonical-history restore")),
            "the original Goal restore error must be visible, not only its terminal category: {events:?}"
        );
        let originals: Vec<_> = events
            .iter()
            .enumerate()
            .filter(|(_, (_, text))| text.contains("Goal canonical-history restore"))
            .map(|(index, _)| index)
            .collect();
        assert_eq!(
            originals.len(),
            1,
            "restore error must appear once: {events:?}"
        );
        let started = zeroclaw_runtime::i18n::get_required_cli_string("goal-mode-started");
        let failed = zeroclaw_runtime::i18n::get_required_cli_string("goal-mode-failure-executor");
        let started_at = events
            .iter()
            .position(|(kind, text)| {
                *kind == DeliveryKind::Message && text.lines().next() == Some(started.as_str())
            })
            .unwrap_or_else(|| panic!("missing start response: {events:?}"));
        let failed_at = events
            .iter()
            .position(|(kind, text)| *kind == DeliveryKind::Message && text == &failed)
            .unwrap_or_else(|| panic!("missing failure notice: {events:?}"));
        assert!(
            started_at < originals[0] && originals[0] < failed_at,
            "original error must precede the failure notice: {events:?}"
        );
        let hint = zeroclaw_providers::reliable::transient_error_hint(&anyhow::Error::msg(
            "fixture unavailable",
        ))
        .expect("ordinary unavailable error has a provider hint");
        assert!(
            !events.iter().any(|(_, text)| text.contains(hint)),
            "a provider hint must not replace the restore error: {events:?}"
        );
        return;
    }
    let notice = if plain_resume || fail_answer {
        "goal-mode-paused-blocked"
    } else {
        "goal-mode-completed"
    };
    wait_for_notice(&channel, notice).await;
    assert_turn_delivery(
        &channel,
        MatrixStreamMode::Off,
        "goal-mode-started",
        if plain_resume || fail_answer {
            QUESTION
        } else {
            COMPLETION
        },
        notice,
    )
    .await;
    if let ColdEntry::GoalCapped { cap, oversized } = entry {
        assert_capped_restore(&responses.observations.lock().unwrap()[0], &expected, cap);
        if oversized {
            expected = expected[expected.len() - super::super::MAX_CHANNEL_HISTORY..].to_vec();
        }
    } else {
        assert_read_only_restore(&responses.observations.lock().unwrap()[0], &expected);
    }
    expected.push(ChatMessage::assistant(ORPHAN_CLOSURE));
    expected.push(ChatMessage::assistant(if plain_resume || fail_answer {
        QUESTION
    } else {
        COMPLETION
    }));
    assert_eq!(
        serde_json::to_value(backend.inner.try_load(&key).unwrap()).unwrap(),
        serde_json::to_value(&expected).unwrap()
    );
    assert!(
        context
            .conversation_histories
            .lock()
            .unwrap()
            .peek(&key)
            .is_some()
    );
    if let ColdEntry::GoalCapped { cap, oversized } = entry {
        let cache = expected[expected.len().saturating_sub(cap)..].to_vec();
        assert_eq!(
            serde_json::to_value(context.conversation_histories.lock().unwrap().peek(&key))
                .unwrap(),
            serde_json::to_value(Some(&cache)).unwrap()
        );
        assert_eq!(
            context.history_crumb_flags.lock().unwrap().peek(&key),
            Some(&(!oversized && cap >= expected.len()))
        );
        assert_eq!(
            backend.inner.get_session_trim_breadcrumb(&key).unwrap(),
            Some(!oversized)
        );
        let mut calls = vec![
            BackendCall::TryLoad,
            BackendCall::GetProvenance,
            BackendCall::TryLoad,
            BackendCall::GetProvenance,
            BackendCall::Append,
        ];
        if oversized {
            calls.push(BackendCall::Replace);
        }
        calls.push(BackendCall::Append);
        assert_eq!(*backend.calls.lock().unwrap(), calls);
    }
    assert!(
        context
            .history_crumb_flags
            .lock()
            .unwrap()
            .peek(&key)
            .is_some()
    );
    if fail_answer {
        // Exercise resume on a cold session with the actual ordinary append
        // owner; only the answer append faults, not the controller or store.
        context.conversation_histories.lock().unwrap().pop(&key);
        context.history_crumb_flags.lock().unwrap().pop(&key);
        backend.calls.lock().unwrap().clear();
        channel.events.lock().await.clear();
        super::super::process_channel_message(
            context.clone(),
            message("resume-answer", &format!("/goal resume {ANSWER}")),
            CancellationToken::new(),
        )
        .await;
        wait_for_notice(&channel, "goal-mode-failure-executor").await;
        assert_eq!(backend.answer_failures.load(Ordering::SeqCst), 1);
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            2,
            "no model call after failed answer persistence"
        );
        assert_eq!(
            serde_json::to_value(backend.inner.try_load(&key).unwrap()).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
        let registry = zeroclaw_runtime::control_plane::control_plane()
            .unwrap()
            .goal_store()
            .unwrap();
        let task = registry
            .current_goal_for_session(&key)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(
            registry
                .terminal_reason_for_session_goal(&task.id, &key)
                .await
                .unwrap()
                .as_deref(),
            Some("executor_failed")
        );
        let events = channel.events.lock().await.clone();
        stop_fixture_daemon(daemon).await;
        let original =
            "cannot resume Goal because the Matrix session transcript could not be persisted";
        let errors: Vec<_> = events
            .iter()
            .enumerate()
            .filter(|(_, (_, text))| text.contains(original))
            .map(|(index, _)| index)
            .collect();
        assert_eq!(
            errors.len(),
            1,
            "resume-persistence error must appear once: {events:?}"
        );
        let resumed = zeroclaw_runtime::i18n::get_required_cli_string("goal-mode-resumed");
        let failed = zeroclaw_runtime::i18n::get_required_cli_string("goal-mode-failure-executor");
        let resumed_at = events
            .iter()
            .position(|(kind, text)| {
                *kind == DeliveryKind::Message && text.lines().next() == Some(resumed.as_str())
            })
            .unwrap_or_else(|| panic!("missing resume response: {events:?}"));
        let failed_at = events
            .iter()
            .position(|(kind, text)| *kind == DeliveryKind::Message && text == &failed)
            .unwrap_or_else(|| panic!("missing failure notice: {events:?}"));
        assert!(
            resumed_at < errors[0] && errors[0] < failed_at,
            "resume error must precede failure notice: {events:?}"
        );
        return;
    }
    if plain_resume {
        // Simulate process-local eviction without changing canonical history.
        context.conversation_histories.lock().unwrap().pop(&key);
        context.history_crumb_flags.lock().unwrap().pop(&key);
        backend.calls.lock().unwrap().clear();
        channel.events.lock().await.clear();
        super::super::process_channel_message(
            context.clone(),
            message("resume", "/goal resume"),
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
        assert_read_only_restore(&responses.observations.lock().unwrap()[1], &expected);
        expected.push(ChatMessage::assistant(COMPLETION));
    }
    assert_eq!(
        serde_json::to_value(backend.inner.try_load(&key).unwrap()).unwrap(),
        serde_json::to_value(&expected).unwrap()
    );
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        if plain_resume { 4 } else { 2 }
    );
    let plane = zeroclaw_runtime::control_plane::control_plane().unwrap();
    let goal = plane
        .goal_store()
        .unwrap()
        .current_goal_for_session(&key)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(goal.status, TaskStatus::Completed);
    let status = super::super::goal_execution::submit_matrix_goal(
        context,
        key,
        message("status", "/goal status"),
        zeroclaw_commands::goal::GoalCommand::Status,
    )
    .await
    .unwrap();
    let GoalResponse::Terminal(status) = status.response() else {
        panic!("completed Goal must return terminal status");
    };
    assert_eq!(status.accounting_state, GoalAccountingState::Complete);
    assert_eq!(
        status.usage_totals.as_ref().unwrap().total_tokens,
        if plain_resume { 48 } else { 24 }
    );
    stop_fixture_daemon(daemon).await;
}

async fn stop_fixture_daemon(
    daemon: tokio::task::JoinHandle<anyhow::Result<zeroclaw_runtime::daemon::DaemonExit>>,
) {
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

async fn blocked_question_phase_one(data_dir: &Path, mock_uri: &str) {
    let fixture = cold_phase_fixture(data_dir, mock_uri, true).await;
    super::super::process_channel_message(
        fixture.context.clone(),
        shared_room_message(
            "restart-blocked-start",
            &format!("/goal start --unlimited {OBJECTIVE}"),
        ),
        CancellationToken::new(),
    )
    .await;
    wait_for_notice(&fixture.channel, "goal-mode-paused-blocked").await;
    let (task, goal) = current_goal(&fixture).await;
    assert_eq!(task.status, TaskStatus::Paused);
    assert_eq!(goal.pause_reason, Some(GoalPauseReason::NeedsUserInput));
    assert_eq!(goal.blockers.len(), 1);
    assert_eq!(goal.blockers[0].kind, GoalBlockerKind::NeedsUserInput);
    assert_eq!(goal.blockers[0].message, QUESTION);
    assert!(goal.blockers[0].payload.is_none());
    assert_settled(&goal);
    let history = fixture.backend.load(&fixture.history_key);
    assert_eq!(history.len(), 1, "blocked history: {history:?}");
    assert_eq!(
        (history[0].role.as_str(), history[0].content.as_str()),
        ("assistant", QUESTION)
    );
    std::fs::write(
        marker(data_dir, P2_SNAPSHOT),
        serde_json::to_vec(&json!({
            "task_id": task.id,
            "pause_reason": goal.pause_reason,
            "blockers": goal.blockers,
        }))
        .unwrap(),
    )
    .unwrap();
    stop_fixture_daemon(fixture.daemon).await;
}

async fn blocked_question_phase_two(data_dir: &Path, mock_uri: &str) {
    let fixture = cold_phase_fixture(data_dir, mock_uri, true).await;
    let snapshot: Value =
        serde_json::from_slice(&std::fs::read(marker(data_dir, P2_SNAPSHOT)).unwrap()).unwrap();
    let (task, goal) = current_goal(&fixture).await;
    assert_eq!(task.id, snapshot["task_id"].as_str().unwrap());
    assert_eq!(task.status, TaskStatus::Paused);
    assert_eq!(goal.pause_reason, Some(GoalPauseReason::NeedsUserInput));
    assert_ne!(goal.pause_reason, Some(GoalPauseReason::DaemonRestart));
    assert_eq!(
        serde_json::to_value(goal.pause_reason).unwrap(),
        snapshot["pause_reason"]
    );
    assert_eq!(
        serde_json::to_value(&goal.blockers).unwrap(),
        snapshot["blockers"]
    );
    assert_settled(&goal);
    let blocked_history = fixture.backend.load(&fixture.history_key);
    assert_eq!(
        blocked_history.len(),
        1,
        "blocked history: {blocked_history:?}"
    );
    assert_eq!(
        (
            blocked_history[0].role.as_str(),
            blocked_history[0].content.as_str()
        ),
        ("assistant", QUESTION)
    );
    write_marker(data_dir, P2_BOOTED);
    wait_for_marker(data_dir, P2_BOOT_ACK).await;

    let mut foreign =
        shared_room_message("restart-blocked-foreign", &format!("/goal resume {ANSWER}"));
    foreign.sender = "@foreign-goal-test:example.org".to_owned();
    super::super::process_channel_message(
        fixture.context.clone(),
        foreign,
        CancellationToken::new(),
    )
    .await;
    assert_eq!(
        serde_json::to_value(fixture.backend.load(&fixture.history_key)).unwrap(),
        serde_json::to_value(&blocked_history).unwrap()
    );
    let (foreign_task, foreign_goal) = current_goal(&fixture).await;
    assert_eq!(foreign_task.id, task.id);
    assert_eq!(foreign_task.status, TaskStatus::Paused);
    assert_eq!(foreign_goal.pause_reason, goal.pause_reason);
    assert_eq!(foreign_goal.blockers, goal.blockers);
    write_marker(data_dir, P2_FOREIGN_DONE);
    wait_for_marker(data_dir, P2_FOREIGN_ACK).await;

    fixture.channel.events.lock().await.clear();
    super::super::process_channel_message(
        fixture.context.clone(),
        shared_room_message("restart-blocked-owner", &format!("/goal resume {ANSWER}")),
        CancellationToken::new(),
    )
    .await;
    wait_for_notice(&fixture.channel, "goal-mode-completed").await;
    assert_turn_delivery(
        &fixture.channel,
        MatrixStreamMode::Off,
        "goal-mode-resumed",
        COMPLETION,
        "goal-mode-completed",
    )
    .await;
    let (completed, completed_goal) = current_goal(&fixture).await;
    assert_eq!(completed.id, task.id);
    assert_eq!(completed.status, TaskStatus::Completed);
    assert_settled(&completed_goal);
    let history = fixture.backend.load(&fixture.history_key);
    assert_eq!(
        history
            .iter()
            .map(|row| (row.role.as_str(), row.content.as_str()))
            .collect::<Vec<_>>(),
        [
            ("assistant", QUESTION),
            ("user", ANSWER),
            ("assistant", COMPLETION),
        ]
    );
    assert_eq!(
        history
            .iter()
            .filter(|row| row.role == "user" && row.content == ANSWER)
            .count(),
        1
    );
    let status = super::super::goal_execution::submit_matrix_goal(
        fixture.context,
        fixture.history_key,
        shared_room_message("restart-blocked-status", "/goal status"),
        zeroclaw_commands::goal::GoalCommand::Status,
    )
    .await
    .unwrap();
    let GoalResponse::Terminal(status) = status.response() else {
        panic!("answered restart Goal must be terminal")
    };
    assert_eq!(status.accounting_state, GoalAccountingState::Complete);
    assert_eq!(status.usage_totals.as_ref().unwrap().total_tokens, 48);
    stop_fixture_daemon(fixture.daemon).await;
}

#[tokio::test]
async fn matrix_goal_blocked_question_survives_process_restart_then_answered_resume() {
    const TEST: &str = "orchestrator::goal_conformance_tests::cold_restore::matrix_goal_blocked_question_survives_process_restart_then_answered_resume";
    if let Some((phase, data_dir, mock_uri)) = phase_inputs(TEST) {
        match phase.as_str() {
            P2_PHASE_ONE => blocked_question_phase_one(&data_dir, &mock_uri).await,
            P2_PHASE_TWO => blocked_question_phase_two(&data_dir, &mock_uri).await,
            other => panic!("unknown blocked-restart phase {other}"),
        }
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ModelResponses)
        .mount(&server)
        .await;
    isolated_phase(TEST, P2_PHASE_ONE, tmp.path(), &server.uri(), async {}).await;
    let first_phase = server.received_requests().await.unwrap();
    assert_request_models(&first_phase, &["goal-worker", "goal-verifier"]);
    isolated_phase(TEST, P2_PHASE_TWO, tmp.path(), &server.uri(), async {
        wait_for_marker(tmp.path(), P2_BOOTED).await;
        assert_request_models(
            &server.received_requests().await.unwrap(),
            &["goal-worker", "goal-verifier"],
        );
        write_marker(tmp.path(), P2_BOOT_ACK);
        wait_for_marker(tmp.path(), P2_FOREIGN_DONE).await;
        assert_request_models(
            &server.received_requests().await.unwrap(),
            &["goal-worker", "goal-verifier"],
        );
        write_marker(tmp.path(), P2_FOREIGN_ACK);
    })
    .await;
    let requests = server.received_requests().await.unwrap();
    assert_request_models(
        &requests,
        &[
            "goal-worker",
            "goal-verifier",
            "goal-worker",
            "goal-verifier",
        ],
    );
    let resumed: Value = serde_json::from_slice(&requests[2].body).unwrap();
    let rows = resumed["messages"].as_array().unwrap();
    let question_at = rows
        .iter()
        .position(|row| row["role"] == "assistant" && row["content"] == QUESTION)
        .expect("resumed worker request must retain the question");
    let answers = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row["role"] == "user" && row["content"] == ANSWER)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    assert_eq!(
        answers.len(),
        1,
        "resume answer must appear byte-for-byte once"
    );
    assert!(question_at < answers[0]);
}

async fn graceful_shutdown_phase_one(data_dir: &Path, mock_uri: &str) {
    let fixture = cold_phase_fixture(data_dir, mock_uri, false).await;
    super::super::process_channel_message(
        fixture.context.clone(),
        message(
            "graceful-shutdown-start",
            &format!("/goal start --unlimited {OBJECTIVE}"),
        ),
        CancellationToken::new(),
    )
    .await;
    wait_for_marker(data_dir, P3_PROVIDER_HELD).await;
    let task_id = tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            let (task, goal) = current_goal(&fixture).await;
            if goal.pending_call_id.is_some() && goal.pending_call_epoch.is_some() {
                break task.id;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("provider operation must become durably pending");
    let signal = tokio::process::Command::new("kill")
        .args(["-TERM", &std::process::id().to_string()])
        .status()
        .await
        .unwrap();
    assert!(signal.success());
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            let snapshot = zeroclaw_runtime::health::snapshot();
            if snapshot
                .components
                .get("daemon")
                .and_then(|component| component.last_error.as_deref())
                == Some("shutdown requested")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("daemon must publish the processed shutdown signal");
    write_marker(data_dir, P3_SIGNAL_PROCESSED);
    wait_for_marker(data_dir, P3_STILL_HELD).await;
    let finished_while_held = fixture.daemon.is_finished();
    write_marker(data_dir, P3_RELEASE);
    assert!(
        !finished_while_held,
        "daemon returned before the admitted provider response was released"
    );
    let exit = tokio::time::timeout(Duration::from_secs(10), fixture.daemon)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(
        exit,
        zeroclaw_runtime::daemon::DaemonExit::Shutdown
    ));
    wait_for_marker(data_dir, P3_RESPONSE_RETURNED).await;
    let store = zeroclaw_runtime::control_plane::control_plane()
        .unwrap()
        .goal_store()
        .unwrap();
    let task = store
        .current_goal_for_session(&fixture.history_key)
        .await
        .unwrap()
        .unwrap();
    let goal = store.get_goal_task(&task.id).await.unwrap().unwrap();
    assert_eq!(task.id, task_id);
    assert_eq!(task.status, TaskStatus::Paused);
    assert_eq!(goal.pause_reason, Some(GoalPauseReason::DaemonRestart));
    assert_settled(&goal);
    assert!(
        fixture
            .backend
            .load(&fixture.history_key)
            .iter()
            .any(|row| row.role == "assistant" && row.content == COMPLETION),
        "settled parent response must be durable before shutdown returns"
    );
    std::fs::write(
        marker(data_dir, P3_SNAPSHOT),
        serde_json::to_vec(&json!({"task_id": task.id})).unwrap(),
    )
    .unwrap();
}

async fn graceful_shutdown_phase_two(data_dir: &Path, mock_uri: &str) {
    let fixture = cold_phase_fixture(data_dir, mock_uri, false).await;
    let snapshot: Value =
        serde_json::from_slice(&std::fs::read(marker(data_dir, P3_SNAPSHOT)).unwrap()).unwrap();
    let (task, goal) = current_goal(&fixture).await;
    assert_eq!(task.id, snapshot["task_id"].as_str().unwrap());
    assert_eq!(task.status, TaskStatus::Paused);
    assert_eq!(goal.pause_reason, Some(GoalPauseReason::DaemonRestart));
    assert_settled(&goal);
    write_marker(data_dir, P3_BOOTED);
    wait_for_marker(data_dir, P3_BOOT_ACK).await;
    super::super::process_channel_message(
        fixture.context.clone(),
        message("graceful-shutdown-resume", "/goal resume"),
        CancellationToken::new(),
    )
    .await;
    wait_for_notice(&fixture.channel, "goal-mode-completed").await;
    assert_turn_delivery(
        &fixture.channel,
        MatrixStreamMode::Off,
        "goal-mode-resumed",
        COMPLETION,
        "goal-mode-completed",
    )
    .await;
    let (completed, completed_goal) = current_goal(&fixture).await;
    assert_eq!(completed.id, task.id);
    assert_eq!(completed.status, TaskStatus::Completed);
    assert_settled(&completed_goal);
    let status = super::super::goal_execution::submit_matrix_goal(
        fixture.context,
        fixture.history_key,
        message("graceful-shutdown-status", "/goal status"),
        zeroclaw_commands::goal::GoalCommand::Status,
    )
    .await
    .unwrap();
    let GoalResponse::Terminal(status) = status.response() else {
        panic!("resumed shutdown Goal must be terminal")
    };
    assert_eq!(status.accounting_state, GoalAccountingState::Complete);
    assert_eq!(status.usage_totals.as_ref().unwrap().total_tokens, 36);
    stop_fixture_daemon(fixture.daemon).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn matrix_goal_graceful_shutdown_drains_admitted_parent_then_restart_keeps_it_resumable() {
    const TEST: &str = "orchestrator::goal_conformance_tests::cold_restore::matrix_goal_graceful_shutdown_drains_admitted_parent_then_restart_keeps_it_resumable";
    if let Some((phase, data_dir, mock_uri)) = phase_inputs(TEST) {
        match phase.as_str() {
            P3_PHASE_ONE => graceful_shutdown_phase_one(&data_dir, &mock_uri).await,
            P3_PHASE_TWO => graceful_shutdown_phase_two(&data_dir, &mock_uri).await,
            other => panic!("unknown graceful-shutdown phase {other}"),
        }
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(HoldFirstWorkerResponse {
            data_dir: tmp.path().to_path_buf(),
            calls: AtomicUsize::new(0),
        })
        .mount(&server)
        .await;
    let release = ReleaseResponseOnDrop(marker(tmp.path(), P3_RELEASE));
    isolated_phase(TEST, P3_PHASE_ONE, tmp.path(), &server.uri(), async {
        wait_for_marker(tmp.path(), P3_SIGNAL_PROCESSED).await;
        assert!(
            !marker(tmp.path(), P3_RESPONSE_RETURNED).exists(),
            "provider response escaped before shutdown processing was observed"
        );
        write_marker(tmp.path(), P3_STILL_HELD);
        wait_for_marker(tmp.path(), P3_RELEASE).await;
        drop(release);
    })
    .await;
    assert_request_models(&server.received_requests().await.unwrap(), &["goal-worker"]);
    isolated_phase(TEST, P3_PHASE_TWO, tmp.path(), &server.uri(), async {
        wait_for_marker(tmp.path(), P3_BOOTED).await;
        assert_request_models(&server.received_requests().await.unwrap(), &["goal-worker"]);
        write_marker(tmp.path(), P3_BOOT_ACK);
    })
    .await;
    assert_request_models(
        &server.received_requests().await.unwrap(),
        &["goal-worker", "goal-worker", "goal-verifier"],
    );
}

#[tokio::test]
async fn matrix_goal_cold_start_restores_without_writes_before_model_request() {
    cold_scenario("orchestrator::goal_conformance_tests::cold_restore::matrix_goal_cold_start_restores_without_writes_before_model_request", false, None).await;
}

#[tokio::test]
async fn matrix_goal_cold_plain_resume_restores_without_fabricated_answer() {
    cold_scenario("orchestrator::goal_conformance_tests::cold_restore::matrix_goal_cold_plain_resume_restores_without_fabricated_answer", true, None).await;
}

#[tokio::test]
async fn matrix_goal_cold_unreadable_transcript_never_calls_model_and_reports_error() {
    cold_scenario("orchestrator::goal_conformance_tests::cold_restore::matrix_goal_cold_unreadable_transcript_never_calls_model_and_reports_error", false, Some(ReadFailure::Transcript)).await;
}

#[tokio::test]
async fn matrix_goal_cold_unreadable_provenance_never_calls_model_and_reports_error() {
    cold_scenario("orchestrator::goal_conformance_tests::cold_restore::matrix_goal_cold_unreadable_provenance_never_calls_model_and_reports_error", false, Some(ReadFailure::Provenance)).await;
}

#[tokio::test]
async fn matrix_goal_cold_answer_persistence_failure_precedes_notice() {
    cold_scenario_inner(
        "orchestrator::goal_conformance_tests::cold_restore::matrix_goal_cold_answer_persistence_failure_precedes_notice",
        false,
        None,
        true,
        ColdEntry::Goal,
    ).await;
}

#[tokio::test]
async fn ordinary_matrix_cold_turn_reconciles_cap_orphan_and_provenance() {
    cold_scenario_inner(
        "orchestrator::goal_conformance_tests::cold_restore::ordinary_matrix_cold_turn_reconciles_cap_orphan_and_provenance",
        false,
        None,
        false,
        ColdEntry::Ordinary,
    ).await;
}

#[tokio::test]
async fn matrix_goal_cold_cap_equal_to_reload_limit_retains_owned_breadcrumb() {
    cold_scenario_inner(
        "orchestrator::goal_conformance_tests::cold_restore::matrix_goal_cold_cap_equal_to_reload_limit_retains_owned_breadcrumb",
        false, None, false,
        ColdEntry::GoalCapped { cap: super::super::MAX_CHANNEL_HISTORY, oversized: false },
    ).await;
}

#[tokio::test]
async fn matrix_goal_cold_configured_cap_removes_owned_breadcrumb_read_only() {
    cold_scenario_inner(
        "orchestrator::goal_conformance_tests::cold_restore::matrix_goal_cold_configured_cap_removes_owned_breadcrumb_read_only",
        false, None, false,
        ColdEntry::GoalCapped { cap: 2, oversized: false },
    ).await;
}

#[tokio::test]
async fn matrix_goal_cold_oversized_transcript_applies_both_caps_before_reconciliation() {
    cold_scenario_inner(
        "orchestrator::goal_conformance_tests::cold_restore::matrix_goal_cold_oversized_transcript_applies_both_caps_before_reconciliation",
        false, None, false,
        ColdEntry::GoalCapped { cap: 2, oversized: true },
    ).await;
}
