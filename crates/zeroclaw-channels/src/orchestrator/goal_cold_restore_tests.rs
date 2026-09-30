//! Observe cold restoration at the real parent model-request boundary.

use super::*;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};
use zeroclaw_api::model_provider::ChatMessage;

const EARLIER_QUESTION: &str = "Which setting should the earlier request use?";
const EARLIER_ANSWER: &str = "Use the earlier setting.";
const ORPHAN_CLOSURE: &str = "[Session interrupted — not continuing this request]";

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
    *ctx.live_config.write() = config;
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
