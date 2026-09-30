//! Shared active-Goal conformance fixture for real ZeroCode transports.

use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, tungstenite::Message};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate, matchers::method};
use zeroclaw_api::{model_provider::ChatMessage, task::TaskStatus};
use zeroclaw_config::schema::{AliasedAgentConfig, Config};
use zeroclaw_goal::{GoalAccountingState, GoalPauseReason, control::GoalResponse};
use zeroclaw_infra::{session_backend::SessionBackend, session_sqlite::SqliteSessionBackend};

use super::{
    context::RpcContext,
    session::{RpcSession, SessionStore},
    types::{ChatMode, SessionGoalResult, SessionGoalUpdate, SessionUpdateEvent},
};

pub(crate) const SESSION_ID: &str = "goal-active-transport";
#[path = "goal_diagnostic_transport_tests.rs"]
mod diagnostics;
pub(crate) const TUI_ID: &str = "goal-active-client";
const QUESTION: &str = "Please choose option A or B before I continue.";
pub(crate) const ANSWER: &str = "Choose A.\nKeep the second line unchanged.";
pub(crate) const OBJECTIVE: &str = "Apply the selected option after asking the user to choose.";
const COMPLETION: &str = "Option A is applied; the declared Goal is complete.";
const PREVIOUS_RESPONSE: &str = "The earlier request is complete.";

/// Exercise answer insertion separately from a resume without user input.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResumeScenario {
    Answered,
    Diagnostics,
    PlainThenAnswered,
    PrefixMatchesObjective,
}

#[derive(Clone, Copy)]
pub(crate) enum GoalTransportFailure {
    AnswerAppend,
    PrefixProjection,
}

/// Boot singletons only in a fresh process dedicated to the named test.
pub(crate) async fn run_isolated(test_name: &str) -> bool {
    const CHILD_ENV: &str = "ZEROCLAW_GOAL_TRANSPORT_TEST_CHILD";
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
        .expect("isolated active Goal transport test must finish")
        .unwrap();
    assert!(
        status.success(),
        "isolated Goal transport test failed: {status}"
    );
    true
}

struct GoalModelResponses;

impl Respond for GoalModelResponses {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        let messages = body["messages"].as_array().unwrap();
        // The epoch's objective may legitimately equal ANSWER. Only a reply
        // following this fixture's question completes its scripted task.
        let question_seen = messages
            .iter()
            .any(|message| message["role"] == "assistant" && message["content"] == QUESTION);
        let answered = question_seen
            && body["messages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|message| message["role"] == "user" && message["content"] == ANSWER);
        let text = if body["model"] == "goal-verifier" {
            let candidate_complete = body["messages"].to_string().contains(COMPLETION);
            if candidate_complete {
                json!({"decision":"complete","reason":"The declared criterion is met.","blockers":[]}).to_string()
            } else {
                json!({"decision":"blocked","reason":"User choice is required.","blockers":[{"kind":"needs_user_input","message":QUESTION}]}).to_string()
            }
        } else if answered {
            COMPLETION.to_owned()
        } else {
            QUESTION.to_owned()
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
                "choices":[{"message":{"role":"assistant","content":text},"finish_reason":"stop"}],
                "usage":usage
            }))
        }
    }
}

pub(crate) struct ActiveGoalFixture {
    pub(crate) context: Arc<RpcContext>,
    backend: Arc<SqliteSessionBackend>,
    server: MockServer,
    answer_failures: Arc<AtomicUsize>,
}

/// Keep the real session store; reject only this fixture's resumed answer.
struct AnswerAppendFailure {
    backend: Arc<SqliteSessionBackend>,
    failures: Arc<AtomicUsize>,
}

impl SessionBackend for AnswerAppendFailure {
    fn load(&self, key: &str) -> Vec<ChatMessage> {
        self.backend.load(key)
    }

    fn try_load(&self, key: &str) -> std::io::Result<Vec<ChatMessage>> {
        self.backend.try_load(key)
    }

    fn append(&self, key: &str, message: &ChatMessage) -> std::io::Result<()> {
        if key == format!("rpc_{SESSION_ID}") && message.role == "user" && message.content == ANSWER
        {
            self.failures.fetch_add(1, Ordering::SeqCst);
            return Err(std::io::Error::other(
                "fixture resume persistence unavailable",
            ));
        }
        self.backend.append(key, message)
    }

    fn remove_last(&self, key: &str) -> std::io::Result<bool> {
        self.backend.remove_last(key)
    }

    fn list_sessions(&self) -> Vec<String> {
        self.backend.list_sessions()
    }

    fn rewrite_messages(&self, key: &str, messages: &[ChatMessage]) -> std::io::Result<()> {
        self.backend.rewrite_messages(key, messages)
    }

    fn replace_conversation_state(
        &self,
        key: &str,
        messages: &[ChatMessage],
        crumb: bool,
    ) -> std::io::Result<()> {
        self.backend
            .replace_conversation_state(key, messages, crumb)
    }

    fn replace_conversation_state_if_exists(
        &self,
        key: &str,
        messages: &[ChatMessage],
        crumb: bool,
    ) -> std::io::Result<bool> {
        self.backend
            .replace_conversation_state_if_exists(key, messages, crumb)
    }

    fn set_session_trim_breadcrumb(&self, key: &str, present: bool) -> std::io::Result<()> {
        self.backend.set_session_trim_breadcrumb(key, present)
    }

    fn get_session_trim_breadcrumb(&self, key: &str) -> std::io::Result<Option<bool>> {
        self.backend.get_session_trim_breadcrumb(key)
    }

    fn get_session_metadata(
        &self,
        key: &str,
    ) -> Option<zeroclaw_infra::session_backend::SessionMetadata> {
        self.backend.get_session_metadata(key)
    }

    fn session_exists(&self, key: &str) -> bool {
        self.backend.session_exists(key)
    }

    fn set_session_principal(&self, key: &str, principal: &str) -> std::io::Result<()> {
        self.backend.set_session_principal(key, principal)
    }
}

struct GoalCapabilityStub;

impl zeroclaw_api::attribution::Attributable for GoalCapabilityStub {
    fn role(&self) -> zeroclaw_api::attribution::Role {
        zeroclaw_api::attribution::Role::Channel(zeroclaw_api::attribution::ChannelKind::Cli)
    }

    fn alias(&self) -> &str {
        "goal-test"
    }
}

#[async_trait::async_trait]
impl zeroclaw_api::channel::Channel for GoalCapabilityStub {
    fn name(&self) -> &str {
        "goal-test"
    }

    async fn send(&self, _: &zeroclaw_api::channel::SendMessage) -> anyhow::Result<()> {
        Ok(())
    }

    async fn listen(
        &self,
        _: tokio::sync::mpsc::Sender<zeroclaw_api::channel::ChannelMessage>,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

impl ActiveGoalFixture {
    pub(crate) async fn with_diagnostics(path: &Path) -> Self {
        // Install capture before native listeners create their execution spans.
        zeroclaw_log::try_install_capture_subscriber();
        drop(zeroclaw_log::subscribe_or_install());
        Self::new(path).await
    }

    /// Add a local capability while preserving any existing RPC channel.
    pub(crate) async fn add_local_capability(&self) {
        let agent = self.context.sessions.get_agent(SESSION_ID).await.unwrap();
        let agent = agent.lock().await;
        let mut reaction = agent.channel_handles().reaction.write();
        reaction
            .entry("rpc".into())
            .or_insert_with(|| Arc::new(GoalCapabilityStub));
        reaction.insert("git.main".into(), Arc::new(GoalCapabilityStub));
    }

    pub(crate) async fn new(path: &Path) -> Self {
        Self::build(path, None).await
    }

    pub(crate) async fn with_answer_append_failure(path: &Path) -> Self {
        Self::build(path, Some(GoalTransportFailure::AnswerAppend)).await
    }

    pub(crate) async fn with_prefix_projection_failure(path: &Path) -> Self {
        Self::build(path, Some(GoalTransportFailure::PrefixProjection)).await
    }

    async fn build(path: &Path, failure: Option<GoalTransportFailure>) -> Self {
        Self::build_with_local_principal(path, failure, None).await
    }

    #[cfg(unix)]
    pub(crate) async fn with_local_execute_grant(path: &Path, uid: u32) -> Self {
        Self::build_with_local_principal(path, None, Some(uid)).await
    }

    async fn build_with_local_principal(
        path: &Path,
        failure: Option<GoalTransportFailure>,
        local_uid: Option<u32>,
    ) -> Self {
        let canonical_workspace = local_uid.map(|_| path.canonicalize().unwrap());
        let path = canonical_workspace.as_deref().unwrap_or(path);
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(GoalModelResponses)
            .mount(&server)
            .await;
        let mut config = Config {
            data_dir: path.to_path_buf(),
            config_path: path.join("config.toml"),
            ..Default::default()
        };
        config.gateway.paired_tokens = vec!["zc_goal_test".to_owned()];
        config.goal.enabled = true;
        config.goal.verifier.model_provider = "openai.goal-test".into();
        config.goal.verifier.model = Some("goal-verifier".into());
        let provider = config
            .providers
            .models
            .ensure("openai", "goal-test")
            .unwrap();
        provider.api_key = Some("test-key".into());
        provider.uri = Some(server.uri());
        provider.kind = Some("openai-compatible".into());
        // This fixture serves Chat Completions JSON/SSE, not Responses API.
        provider.wire_api = Some(zeroclaw_config::schema::WireApi::ChatCompletions);
        provider.model = Some("goal-worker".into());
        if matches!(failure, Some(GoalTransportFailure::PrefixProjection)) {
            // Reach the configured vision fallback rather than the primary's
            // documented image-capability short-circuit.
            provider.vision = Some(false);
        }
        let mut agent_config = AliasedAgentConfig {
            enabled: true,
            model_provider: "openai.goal-test".into(),
            ..Default::default()
        };
        if let Some(uid) = local_uid {
            use zeroclaw_api::grants::{Resource, Verb, WILDCARD};
            use zeroclaw_config::schema::{PermissionProfileConfig, RiskProfileConfig, UserConfig};

            // A native test connection has the daemon's UID. Select the
            // ordinary roster instead of the default shared-operator bypass.
            config.security.trust_daemon_uid = false;
            agent_config.workspace.path = Some(path.to_path_buf());
            agent_config.risk_profile = "goal-local-risk".into();
            config
                .risk_profiles
                .insert("goal-local-risk".into(), RiskProfileConfig::default());
            config.permission_profiles.insert(
                "goal-session-execute".into(),
                PermissionProfileConfig {
                    allowed_agents: vec!["test-agent".into()],
                    allowed_tools: vec![WILDCARD.into()],
                    grants: std::collections::HashMap::from([
                        (
                            Resource::Sessions,
                            vec![Verb::Create, Verb::Read, Verb::Execute],
                        ),
                        (Resource::Tools, vec![Verb::Execute]),
                    ]),
                    ..Default::default()
                },
            );
            config.users.insert(
                "goal-local".into(),
                UserConfig {
                    uid: Some(uid),
                    permission_profiles: vec!["goal-session-execute".into()],
                    ..Default::default()
                },
            );
        }
        config
            .agents
            .insert("test-agent".into(), agent_config.clone());
        let owner = crate::control_plane::boot::ControlPlaneRecoveryOwner::start(path)
            .await
            .unwrap();
        assert!(crate::control_plane::global::init_control_plane(owner));
        let (provider, provider_ref, model, resolver) =
            crate::agent::agent::build_session_model_provider(
                &config,
                "openai.goal-test",
                Some("goal-worker"),
            )
            .unwrap();
        let agent = crate::agent::agent::Agent::builder()
            .model_provider(provider)
            .model_provider_name(provider_ref)
            .model_name(model)
            .model_route_resolver(resolver)
            .config(agent_config)
            .tools(crate::tools::scoped::ScopedToolRegistry::from_raw_for_test(
                vec![],
            ))
            .memory(Arc::new(zeroclaw_memory::NoneMemory::new("none")))
            .observer(Arc::new(crate::observability::noop::NoopObserver))
            .tool_dispatcher(Box::new(crate::agent::dispatcher::NativeToolDispatcher))
            .workspace_dir(path.to_path_buf())
            .build()
            .unwrap();
        let sessions = Arc::new(SessionStore::new(
            4,
            Arc::new(zeroclaw_infra::session_queue::SessionActorQueue::new(
                4, 30, 60,
            )),
        ));
        let backend = Arc::new(SqliteSessionBackend::new(path).unwrap());
        let owner_principal = local_uid.map(|_| "user:goal-local".to_owned());
        if let Some(principal) = owner_principal.as_deref() {
            backend
                .set_session_agent_alias(&format!("rpc_{SESSION_ID}"), "test-agent")
                .unwrap();
            backend
                .set_session_principal(&format!("rpc_{SESSION_ID}"), principal)
                .unwrap();
        }
        let answer_failures = Arc::new(AtomicUsize::new(0));
        let session_backend: Arc<dyn SessionBackend> =
            if matches!(failure, Some(GoalTransportFailure::AnswerAppend)) {
                Arc::new(AnswerAppendFailure {
                    backend: backend.clone(),
                    failures: answer_failures.clone(),
                })
            } else {
                backend.clone()
            };
        let mut context =
            RpcContext::for_persistence_tests(config, sessions, Some(session_backend), None);
        std::fs::write(path.join(".secret_key"), "42".repeat(32)).unwrap();
        Arc::get_mut(&mut context).unwrap().tui_registry =
            Arc::new(super::tui_identity::TuiRegistry::new(path));
        context
            .sessions
            .insert(
                SESSION_ID.to_owned(),
                RpcSession::new(agent, "test-agent", path.to_str().unwrap(), ChatMode::Chat)
                    .with_owner(Some(TUI_ID.to_owned()))
                    .with_owner_principal(owner_principal),
            )
            .await
            .unwrap();
        Self {
            context,
            backend,
            server,
            answer_failures,
        }
    }

    async fn assert_history_and_calls(&self, objective: &str, scenario: ResumeScenario) {
        let history = self.backend.load(&format!("rpc_{SESSION_ID}"));
        let mut expected = Vec::new();
        if scenario == ResumeScenario::PrefixMatchesObjective {
            expected.push(("user", objective));
            expected.push(("assistant", PREVIOUS_RESPONSE));
        }
        expected.push(("assistant", QUESTION));
        if scenario == ResumeScenario::PlainThenAnswered {
            expected.push(("assistant", QUESTION));
        }
        expected.extend([("user", ANSWER), ("assistant", COMPLETION)]);
        assert_eq!(history.len(), expected.len(), "{history:?}");
        for (message, &(role, text)) in history.iter().zip(&expected) {
            assert_eq!(
                (message.role.as_str(), message.content.as_str()),
                (role, text)
            );
            assert_eq!(
                history
                    .iter()
                    .filter(|message| message.role == role && message.content == text)
                    .count(),
                expected
                    .iter()
                    .filter(|&&entry| entry == (role, text))
                    .count(),
                "durable message multiplicity must match the full specification: {history:?}"
            );
        }
        let requests = self.server.received_requests().await.unwrap();
        assert_eq!(
            requests.len(),
            if scenario == ResumeScenario::PlainThenAnswered {
                6
            } else {
                4
            },
            "one verifier per parent; no successor after completion"
        );
        assert!(history.iter().all(|message| message.role != "system"));
        let started: Value = serde_json::from_slice(&requests[0].body).unwrap();
        let messages = started["messages"].as_array().unwrap();
        assert_eq!(
            messages
                .iter()
                .filter(|message| message["role"] == "system")
                .count(),
            1
        );
        assert_eq!(messages[0]["role"], "system");
        assert!(messages[0]["content"].as_str().unwrap().contains(objective));
        let first_user = messages
            .iter()
            .find(|message| message["role"] != "system")
            .unwrap();
        assert_eq!(first_user["role"], "user");
        assert!(first_user["content"].as_str().unwrap().ends_with(objective));
        if scenario == ResumeScenario::PrefixMatchesObjective {
            assert_eq!(messages[2]["role"], "user");
            assert_eq!(messages[2]["content"], objective);
        }
        for index in (1..requests.len()).step_by(2) {
            let verifier: Value = serde_json::from_slice(&requests[index].body).unwrap();
            assert_eq!(verifier["model"], "goal-verifier");
        }
        if scenario == ResumeScenario::PlainThenAnswered {
            let plain: Value = serde_json::from_slice(&requests[2].body).unwrap();
            let messages = plain["messages"].as_array().unwrap();
            assert!(messages.iter().any(|message| {
                message["role"] == "assistant" && message["content"] == QUESTION
            }));
            assert!(
                !messages
                    .iter()
                    .any(|message| { message["role"] == "user" && message["content"] == ANSWER }),
                "plain resume must not fabricate the fixture answer"
            );
        }
        let resumed: Value = serde_json::from_slice(&requests[requests.len() - 2].body).unwrap();
        assert!(
            resumed["messages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|message| message["role"] == "assistant" && message["content"] == QUESTION)
        );
        assert!(
            resumed["messages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|message| message["role"] == "user" && message["content"] == ANSWER)
        );
    }
}

/// Execution-admitting commands whose refusal must preserve durable state.
#[derive(Clone, Copy)]
pub(crate) enum GoalRefusalScenario {
    Start,
    AnsweredResume,
}

/// Compare native control outcomes with and without local capabilities.
pub(crate) async fn assert_remote_goal_control_parity(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
) {
    {
        let agent = fixture
            .context
            .sessions
            .get_agent(SESSION_ID)
            .await
            .unwrap();
        let agent = agent.lock().await;
        agent
            .channel_handles()
            .reaction
            .write()
            .insert("rpc".into(), Arc::new(GoalCapabilityStub));
    }
    let signature = fixture.context.tui_registry.sign(TUI_ID).unwrap();
    transport
        .send(
            "initialize",
            json!({"protocol_version":1,"tui_id":TUI_ID,"tui_sig":signature,"auth_token":"zc_goal_test"}),
            1,
        )
        .await;
    let initialized = transport.receive().await;
    assert_eq!(initialized["id"], 1);
    assert!(initialized["error"].is_null(), "{initialized}");

    let mut baseline = None;
    for with_local_capability in [false, true] {
        let task_id = start_goal_and_wait_for_input(transport).await;
        if with_local_capability {
            fixture.add_local_capability().await;
        }
        let agent = fixture
            .context
            .sessions
            .get_agent(SESSION_ID)
            .await
            .unwrap();
        let agent_guard = agent.lock().await;
        {
            let reaction = agent_guard.channel_handles().reaction.read();
            assert_eq!(reaction.contains_key("git.main"), with_local_capability);
            assert!(reaction.contains_key("rpc"));
        }
        let history_before = serde_json::to_value(agent_guard.history()).unwrap();
        let durable_before = fixture.backend.load(&format!("rpc_{SESSION_ID}"));
        let calls_before = fixture.server.received_requests().await.unwrap().len();
        let mut outcomes = Vec::new();
        for (index, command) in [
            Some("/goal status"),
            Some("/goal budget"),
            Some("/goal budget set --tokens 100"),
            None,
            Some("/goal pause"),
            Some("/goal pause now"),
            Some("/goal cancel"),
        ]
        .into_iter()
        .enumerate()
        {
            let id = 20 + index as u64;
            let (method, params) = match command {
                Some(command) => (
                    "session/goal",
                    json!({"session_id":SESSION_ID,"command":command}),
                ),
                None => ("session/cancel", json!({"session_id":SESSION_ID})),
            };
            transport.send(method, params, id).await;
            let mut frame = tokio::time::timeout(Duration::from_secs(2), transport.receive())
                .await
                .expect("control must respond without acquiring the held Agent mutex");
            assert_eq!(frame["id"], id, "unexpected control event: {frame}");
            if command.is_some() {
                assert!(frame["error"].is_null(), "{frame}");
                assert_eq!(frame["result"]["response"]["status"]["task_id"], task_id);
                // Each phase creates a distinct durable task. No other result
                // field or event is normalized out of this parity comparison.
                frame["result"]["response"]["status"]["task_id"] = json!("phase-task");
            } else if !frame["error"].is_null() {
                assert_eq!(
                    frame["error"]["code"],
                    zeroclaw_api::jsonrpc::error_codes::SESSION_NOT_FOUND
                );
                assert_eq!(frame["error"]["message"], "No active turn for this session");
            } else {
                assert_eq!(
                    frame["result"],
                    json!({"session_id":SESSION_ID,"cancelled":true})
                );
            }
            outcomes.push(frame);
        }
        assert_eq!(
            serde_json::to_value(agent_guard.history()).unwrap(),
            history_before
        );
        assert_eq!(
            serde_json::to_value(fixture.backend.load(&format!("rpc_{SESSION_ID}"))).unwrap(),
            serde_json::to_value(durable_before).unwrap()
        );
        assert_eq!(
            fixture.server.received_requests().await.unwrap().len(),
            calls_before
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(100), transport.receive())
                .await
                .is_err()
        );
        drop(agent_guard);
        let key = zeroclaw_goal::control::GoalSessionKey::zero_code(SESSION_ID).unwrap();
        let registry = crate::control_plane::global::control_plane()
            .unwrap()
            .goal_store()
            .unwrap();
        let task = registry
            .current_goal_for_session(&key.durable_id())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(task.id, task_id);
        assert_eq!(task.status, TaskStatus::Cancelled);
        if let Some(baseline) = &baseline {
            assert_eq!(&outcomes, baseline);
        } else {
            baseline = Some(outcomes);
        }
    }
}

async fn start_goal_and_wait_for_input(transport: &mut GoalTestTransport) -> String {
    transport
        .send(
            "session/goal",
            json!({"session_id":SESSION_ID,"command":format!("/goal start --unlimited {OBJECTIVE}")}),
            10,
        )
        .await;
    let started = transport.receive().await;
    assert_eq!(started["id"], 10);
    assert!(started["error"].is_null(), "Goal start refused: {started}");
    let result: SessionGoalResult = serde_json::from_value(started["result"].clone())
        .unwrap_or_else(|error| panic!("invalid Goal start result: {error}; frame: {started}"));
    let GoalResponse::Started(projection) = result.response else {
        panic!("expected initial Goal start: {result:?}");
    };
    let mut candidate_seen = false;
    let mut chunks = String::new();
    loop {
        let frame = transport.receive().await;
        assert!(frame["error"].is_null(), "{frame}");
        match frame["method"].as_str() {
            Some("session/update") => {
                if let SessionUpdateEvent::AgentMessageChunk { text, .. } =
                    serde_json::from_value(frame["params"].clone()).unwrap()
                {
                    chunks.push_str(&text);
                }
            }
            Some("session/goal_update") => {
                match serde_json::from_value::<SessionGoalUpdate>(frame["params"].clone()).unwrap()
                {
                    SessionGoalUpdate::VerifiedCandidate { candidate, .. } => {
                        assert_eq!(candidate, QUESTION);
                        candidate_seen = true;
                    }
                    SessionGoalUpdate::PausedForBlocker { .. } => {
                        assert!(candidate_seen);
                        assert_eq!(chunks, QUESTION);
                        break;
                    }
                    other => panic!("unexpected pre-refusal lifecycle event: {other:?}"),
                }
            }
            other => panic!("unexpected pre-refusal frame: {other:?}: {frame}"),
        }
    }
    projection.task_id
}

/// Fail the actual durable resume append, retaining native transport framing.
pub(crate) async fn assert_answer_append_failure(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
    remote: bool,
) {
    let signature = fixture.context.tui_registry.sign(TUI_ID).unwrap();
    transport.send("initialize", json!({"protocol_version":1,"tui_id":TUI_ID,"tui_sig":signature,"auth_token":remote.then_some("zc_goal_test")}), 1).await;
    let initialized = transport.receive().await;
    assert_eq!(initialized["id"], 1);
    assert!(initialized["error"].is_null(), "{initialized}");
    let task_id = start_goal_and_wait_for_input(transport).await;
    let key = zeroclaw_goal::control::GoalSessionKey::zero_code(SESSION_ID).unwrap();
    let before_history = serde_json::to_value(fixture.backend.load(&key.durable_id())).unwrap();
    assert_eq!(before_history.as_array().unwrap().len(), 1);
    assert_eq!(before_history[0]["role"], "assistant");
    assert_eq!(before_history[0]["content"], QUESTION);
    assert_eq!(fixture.server.received_requests().await.unwrap().len(), 2);
    let agent = fixture
        .context
        .sessions
        .get_agent(SESSION_ID)
        .await
        .unwrap();
    let before_agent_history = serde_json::to_value(agent.lock().await.history()).unwrap();

    transport
        .send(
            "session/goal",
            json!({"session_id":SESSION_ID,"command":format!("/goal resume {ANSWER}")}),
            11,
        )
        .await;
    let resumed = transport.receive().await;
    assert_eq!(resumed["id"], 11);
    let response: SessionGoalResult = serde_json::from_value(resumed["result"].clone()).unwrap();
    let GoalResponse::Resumed(projection) = response.response else {
        panic!("expected answered resume acknowledgement: {response:?}");
    };
    assert_eq!(projection.task_id, task_id);
    let mut ordinary_errors = Vec::new();
    loop {
        let frame = transport.receive().await;
        match frame["method"].as_str() {
            Some("session/update") => {
                let update: SessionUpdateEvent =
                    serde_json::from_value(frame["params"].clone()).unwrap();
                let SessionUpdateEvent::TurnComplete {
                    session_id,
                    outcome,
                    content,
                    ..
                } = update
                else {
                    panic!("no parent work may start after failed answer persistence: {update:?}");
                };
                assert_eq!(session_id, SESSION_ID);
                assert_eq!(outcome, super::types::TurnCompletionOutcome::Failed);
                ordinary_errors.push(content);
            }
            Some("session/goal_update") => {
                let update: SessionGoalUpdate =
                    serde_json::from_value(frame["params"].clone()).unwrap();
                let SessionGoalUpdate::Failed { session_id, reason } = update else {
                    panic!("only the failure notice may follow resume: {update:?}");
                };
                assert_eq!(session_id, SESSION_ID);
                assert_eq!(
                    reason,
                    zeroclaw_goal::control::GoalFailureReason::ExecutorFailed
                );
                break;
            }
            other => panic!("unexpected resume-failure frame: {other:?}: {frame}"),
        }
    }
    let registry = crate::control_plane::global::control_plane()
        .unwrap()
        .goal_store()
        .unwrap();
    let task = registry
        .current_goal_for_session(&key.durable_id())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(task.id, task_id);
    assert_eq!(task.status, TaskStatus::Failed);
    assert_eq!(
        registry
            .terminal_reason_for_session_goal(&task_id, &key.durable_id())
            .await
            .unwrap()
            .as_deref(),
        Some("executor_failed")
    );
    assert_eq!(fixture.answer_failures.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.server.received_requests().await.unwrap().len(), 2);
    assert_eq!(
        serde_json::to_value(fixture.backend.load(&key.durable_id())).unwrap(),
        before_history
    );
    assert_eq!(
        serde_json::to_value(agent.lock().await.history()).unwrap(),
        before_agent_history
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(100), transport.receive())
            .await
            .is_err(),
        "no duplicate failure event is permitted"
    );
    assert_eq!(
        ordinary_errors,
        ["turn failed: Agent turn failed: persist ZeroCode Goal resume response"],
        "the original persistence error must precede the Goal notice exactly once"
    );
}

/// Exercise the actual Agent projection failure before any provider request.
pub(crate) async fn assert_prefix_projection_failure(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
    remote: bool,
) {
    let signature = fixture.context.tui_registry.sign(TUI_ID).unwrap();
    transport.send("initialize", json!({"protocol_version":1,"tui_id":TUI_ID,"tui_sig":signature,"auth_token":remote.then_some("zc_goal_test")}), 1).await;
    let initialized = transport.receive().await;
    assert_eq!(initialized["id"], 1);
    assert!(initialized["error"].is_null(), "{initialized}");
    let key = zeroclaw_goal::control::GoalSessionKey::zero_code(SESSION_ID).unwrap();
    let agent = fixture
        .context
        .sessions
        .get_agent(SESSION_ID)
        .await
        .unwrap();
    let original = {
        let mut agent = agent.lock().await;
        agent.set_multimodal_config(zeroclaw_config::schema::MultimodalConfig {
            vision_model_provider: Some("unknown-fixture-provider".into()),
            ..Default::default()
        });
        let message = ChatMessage::user("inspect [IMAGE:data:image/png;base64,fixture]");
        fixture.backend.append(&key.durable_id(), &message).unwrap();
        agent.seed_history(&[message]);
        agent.isolated_canonical_prefix().await.unwrap_err()
    };
    assert!(
        original
            .to_string()
            .contains("failed to create vision model_provider")
    );
    let before_history = serde_json::to_value(fixture.backend.load(&key.durable_id())).unwrap();
    let before_agent_history = serde_json::to_value(agent.lock().await.history()).unwrap();
    transport.send("session/goal", json!({"session_id":SESSION_ID,"command":format!("/goal start --unlimited {OBJECTIVE}")}), 10).await;
    let started = transport.receive().await;
    assert_eq!(started["id"], 10);
    let result: SessionGoalResult = serde_json::from_value(started["result"].clone()).unwrap();
    let GoalResponse::Started(projection) = result.response else {
        panic!("expected Goal start acknowledgement: {result:?}");
    };
    let registry = crate::control_plane::global::control_plane()
        .unwrap()
        .goal_store()
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let task = registry
                .current_goal_for_session(&key.durable_id())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(task.id, projection.task_id);
            if task.status == TaskStatus::Failed {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the projection failure must terminalize the admitted Goal");
    assert_eq!(
        registry
            .terminal_reason_for_session_goal(&projection.task_id, &key.durable_id())
            .await
            .unwrap()
            .as_deref(),
        Some("executor_failed")
    );
    assert!(fixture.server.received_requests().await.unwrap().is_empty());
    assert_eq!(
        serde_json::to_value(fixture.backend.load(&key.durable_id())).unwrap(),
        before_history
    );
    assert_eq!(
        serde_json::to_value(agent.lock().await.history()).unwrap(),
        before_agent_history
    );

    let mut events = Vec::new();
    while let Ok(frame) = tokio::time::timeout(Duration::from_secs(1), transport.receive()).await {
        match frame["method"].as_str() {
            Some("session/update") => {
                let update: SessionUpdateEvent =
                    serde_json::from_value(frame["params"].clone()).unwrap();
                let SessionUpdateEvent::TurnComplete {
                    session_id,
                    outcome,
                    content,
                    ..
                } = update
                else {
                    panic!("projection failure must not start parent execution: {update:?}");
                };
                assert_eq!(session_id, SESSION_ID);
                assert_eq!(outcome, super::types::TurnCompletionOutcome::Failed);
                assert_eq!(
                    content,
                    format!("turn failed: Agent turn failed: {original}")
                );
                events.push("original_error");
            }
            Some("session/goal_update") => {
                let update: SessionGoalUpdate =
                    serde_json::from_value(frame["params"].clone()).unwrap();
                let SessionGoalUpdate::Failed { session_id, reason } = update else {
                    panic!("expected projection failure notice: {update:?}");
                };
                assert_eq!(session_id, SESSION_ID);
                assert_eq!(
                    reason,
                    zeroclaw_goal::control::GoalFailureReason::ExecutorFailed
                );
                events.push("failure_notice");
            }
            other => panic!("unexpected projection-failure frame: {other:?}: {frame}"),
        }
    }
    assert_eq!(
        events,
        ["original_error", "failure_notice"],
        "native projection error must be rendered exactly once before its notice"
    );
}

/// Exercise command refusal on the real wire with the Agent mutex held.
pub(crate) async fn assert_remote_goal_command_refusal(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
    scenario: GoalRefusalScenario,
) {
    let signature = fixture.context.tui_registry.sign(TUI_ID).unwrap();
    transport
        .send(
            "initialize",
            json!({"protocol_version":1,"tui_id":TUI_ID,"tui_sig":signature,"auth_token":"zc_goal_test"}),
            1,
        )
        .await;
    let initialized = transport.receive().await;
    assert_eq!(initialized["id"], 1);
    assert!(initialized["error"].is_null(), "{initialized}");
    let key = zeroclaw_goal::control::GoalSessionKey::zero_code(SESSION_ID).unwrap();
    let registry = crate::control_plane::global::control_plane()
        .unwrap()
        .goal_store()
        .unwrap();
    if matches!(scenario, GoalRefusalScenario::AnsweredResume) {
        start_goal_and_wait_for_input(transport).await;
    }
    let before_task = registry
        .current_goal_for_session(&key.durable_id())
        .await
        .unwrap();
    match scenario {
        GoalRefusalScenario::Start => assert!(before_task.is_none()),
        GoalRefusalScenario::AnsweredResume => assert!(before_task.is_some()),
    }
    let before_goal = match before_task.as_ref() {
        Some(task) => {
            assert_eq!(task.status, TaskStatus::Paused);
            let goal = registry.get_goal_task(&task.id).await.unwrap().unwrap();
            assert_eq!(goal.pause_reason, Some(GoalPauseReason::NeedsUserInput));
            Some(goal)
        }
        None => None,
    };
    let before_history = serde_json::to_value(fixture.backend.load(&key.durable_id())).unwrap();
    let before_calls = fixture.server.received_requests().await.unwrap().len();
    assert_eq!(
        before_calls,
        if matches!(scenario, GoalRefusalScenario::Start) {
            0
        } else {
            2
        }
    );
    fixture.add_local_capability().await;
    let before = fixture
        .context
        .sessions
        .goal_session_snapshot(SESSION_ID)
        .await
        .unwrap();
    let rpc_channel = Arc::clone(before.reaction_channels.read().get("rpc").unwrap());
    let held_agent = before.agent.lock().await;
    let command = match scenario {
        GoalRefusalScenario::Start => format!("/goal start --unlimited {OBJECTIVE}"),
        GoalRefusalScenario::AnsweredResume => format!("/goal resume {ANSWER}"),
    };
    transport
        .send(
            "session/goal",
            json!({"session_id":SESSION_ID,"command":command}),
            2,
        )
        .await;
    let refusal = tokio::time::timeout(Duration::from_secs(2), transport.receive())
        .await
        .expect("command refusal must not wait for the held Agent");
    assert_eq!(refusal["id"], 2);
    assert_eq!(
        refusal["error"]["code"],
        zeroclaw_api::jsonrpc::error_codes::SESSION_NOT_OWNED
    );
    assert_eq!(
        refusal["error"]["message"],
        "Remote caller cannot use this session's local capabilities"
    );
    let after_task = registry
        .current_goal_for_session(&key.durable_id())
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_value(&after_task).unwrap(),
        serde_json::to_value(&before_task).unwrap()
    );
    let after_goal = match after_task.as_ref() {
        Some(task) => registry.get_goal_task(&task.id).await.unwrap(),
        None => None,
    };
    assert_eq!(after_goal, before_goal);
    assert_eq!(
        fixture.server.received_requests().await.unwrap().len(),
        before_calls
    );
    assert_eq!(
        serde_json::to_value(fixture.backend.load(&key.durable_id())).unwrap(),
        before_history
    );
    assert!(
        fixture
            .backend
            .load(&key.durable_id())
            .iter()
            .all(|message| message.content != ANSWER)
    );
    let after = fixture
        .context
        .sessions
        .goal_session_snapshot(SESSION_ID)
        .await
        .unwrap();
    assert_eq!(after.owner_tui_id, before.owner_tui_id);
    assert_eq!(after.generation, before.generation);
    assert!(Arc::ptr_eq(
        after.reaction_channels.read().get("rpc").unwrap(),
        &rpc_channel
    ));
    assert!(
        tokio::time::timeout(Duration::from_millis(100), transport.receive())
            .await
            .is_err()
    );
    drop(held_agent);
}

/// Prove a late execution refusal is delivered before its durable failure notice.
pub(crate) async fn assert_admitted_goal_capability_refusal(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
    scenario: GoalRefusalScenario,
) {
    assert_admitted_goal_refusal(
        fixture,
        transport,
        scenario,
        AdmissionRefusal::RemoteCapabilities,
    )
    .await;
}

#[cfg(unix)]
pub(crate) async fn assert_admitted_goal_grant_refusal(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
) {
    assert_admitted_goal_refusal(
        fixture,
        transport,
        GoalRefusalScenario::AnsweredResume,
        AdmissionRefusal::ExecuteGrant,
    )
    .await;
}

#[derive(Clone, Copy)]
enum AdmissionRefusal {
    RemoteCapabilities,
    #[cfg(unix)]
    ExecuteGrant,
}

async fn assert_admitted_goal_refusal(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
    scenario: GoalRefusalScenario,
    refusal: AdmissionRefusal,
) {
    let signature = fixture.context.tui_registry.sign(TUI_ID).unwrap();
    let mut initialize = json!({"protocol_version":1,"tui_id":TUI_ID,"tui_sig":signature});
    if matches!(refusal, AdmissionRefusal::RemoteCapabilities) {
        initialize["auth_token"] = json!("zc_goal_test");
    }
    transport.send("initialize", initialize, 1).await;
    let initialized = transport.receive().await;
    assert_eq!(initialized["id"], 1);
    assert!(initialized["error"].is_null(), "{initialized}");
    #[cfg(unix)]
    if matches!(refusal, AdmissionRefusal::ExecuteGrant) {
        assert_eq!(initialized["result"]["principal_id"], "user:goal-local");
    }
    let key = zeroclaw_goal::control::GoalSessionKey::zero_code(SESSION_ID).unwrap();
    let previous_task = if matches!(scenario, GoalRefusalScenario::AnsweredResume) {
        Some(start_goal_and_wait_for_input(transport).await)
    } else {
        None
    };
    let registry = crate::control_plane::global::control_plane()
        .unwrap()
        .goal_store()
        .unwrap();
    if let Some(task_id) = previous_task.as_ref() {
        let task = registry
            .current_goal_for_session(&key.durable_id())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(&task.id, task_id);
        assert_eq!(task.status, TaskStatus::Paused);
        assert_eq!(
            registry
                .get_goal_task(task_id)
                .await
                .unwrap()
                .unwrap()
                .pause_reason,
            Some(GoalPauseReason::NeedsUserInput)
        );
    }
    let agent = fixture
        .context
        .sessions
        .get_agent(SESSION_ID)
        .await
        .unwrap();
    let before_agent_history = serde_json::to_value(agent.lock().await.history()).unwrap();
    let before_history = serde_json::to_value(fixture.backend.load(&key.durable_id())).unwrap();
    let before_calls = fixture.server.received_requests().await.unwrap().len();
    assert_eq!(before_calls, if previous_task.is_some() { 2 } else { 0 });
    let commit_entered = Arc::new(tokio::sync::Notify::new());
    let commit_release = Arc::new(tokio::sync::Notify::new());
    *fixture
        .context
        .goal_runtime
        .test_deferred_commit_pause
        .lock()
        .unwrap() = Some(super::goal::GoalTestDeferredCommitPause {
        session_id: SESSION_ID.to_owned(),
        entered: Arc::clone(&commit_entered),
        release: Arc::clone(&commit_release),
    });
    let permit = fixture
        .context
        .sessions
        .session_queue
        .acquire(SESSION_ID)
        .await
        .unwrap();
    let command = match scenario {
        GoalRefusalScenario::Start => format!("/goal start --unlimited {OBJECTIVE}"),
        GoalRefusalScenario::AnsweredResume => format!("/goal resume {ANSWER}"),
    };
    transport
        .send(
            "session/goal",
            json!({"session_id":SESSION_ID,"command":command}),
            2,
        )
        .await;
    let acknowledged = tokio::time::timeout(Duration::from_secs(2), transport.receive())
        .await
        .expect("Goal command must acknowledge without waiting for the session permit");
    assert_eq!(acknowledged["id"], 2);
    let result: SessionGoalResult = serde_json::from_value(acknowledged["result"].clone()).unwrap();
    let expected_intent = match scenario {
        GoalRefusalScenario::Start => zeroclaw_goal::control::GoalQueuedControl::Start,
        GoalRefusalScenario::AnsweredResume => zeroclaw_goal::control::GoalQueuedControl::Resume,
    };
    assert_eq!(result.response, GoalResponse::Queued(expected_intent));
    tokio::time::timeout(Duration::from_secs(5), async {
        while fixture
            .context
            .sessions
            .session_queue
            .queue_depth(SESSION_ID)
            .await
            < 2
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("queued Goal control must wait behind the held session permit");
    drop(permit);
    tokio::time::timeout(Duration::from_secs(5), commit_entered.notified())
        .await
        .expect("queued control must acquire its final commit boundary");
    // The existing commit barrier proves the control owns its final boundary.
    // Reserve the next ordinary turn only now: an earlier reservation would
    // steal the transient wakeup permit and prevent the control from committing.
    // Keep authority changes after Started/Resumed and before worker execution.
    let next_turn = fixture
        .context
        .sessions
        .session_queue
        .reserve(SESSION_ID)
        .await
        .unwrap();
    commit_release.notify_one();
    let permit = tokio::time::timeout(Duration::from_secs(5), next_turn.acquire())
        .await
        .expect("deferred Goal control must release the ordinary boundary")
        .unwrap();
    let lifecycle = transport.receive().await;
    assert_eq!(lifecycle["method"], "session/goal_update", "{lifecycle}");
    let update: SessionGoalUpdate = serde_json::from_value(lifecycle["params"].clone()).unwrap();
    let SessionGoalUpdate::Lifecycle {
        session_id,
        response,
    } = update
    else {
        panic!("deferred control must publish its transition before execution: {update:?}");
    };
    assert_eq!(session_id, SESSION_ID);
    let admitted = match (scenario, response) {
        (GoalRefusalScenario::Start, GoalResponse::Started(projection))
        | (GoalRefusalScenario::AnsweredResume, GoalResponse::Resumed(projection)) => projection,
        (_, other) => panic!("unexpected deferred transition: {other:?}"),
    };
    if let Some(previous_task) = previous_task {
        assert_eq!(admitted.task_id, previous_task);
    }
    let expected_refusal = match refusal {
        AdmissionRefusal::RemoteCapabilities => {
            fixture.add_local_capability().await;
            "turn refused by daemon: Remote caller cannot use this session's local capabilities"
        }
        #[cfg(unix)]
        AdmissionRefusal::ExecuteGrant => {
            use zeroclaw_api::grants::{Resource, Verb};
            let mut config = fixture.context.config.read().clone();
            config
                .permission_profiles
                .get_mut("goal-session-execute")
                .unwrap()
                .grants
                .insert(Resource::Sessions, vec![Verb::Read]);
            fixture.context.auth.refresh_from_config(&config).unwrap();
            *fixture.context.config.write() = config;
            "turn refused by daemon: Principal is not granted sessions:execute (required by session/goal)"
        }
    };
    drop(permit);

    let refused = transport.receive().await;
    assert_eq!(refused["method"], "session/update", "{refused}");
    let update: SessionUpdateEvent = serde_json::from_value(refused["params"].clone()).unwrap();
    let SessionUpdateEvent::TurnComplete {
        session_id,
        outcome,
        content,
        ..
    } = update
    else {
        panic!("first execution event must be the ordinary refusal: {update:?}");
    };
    assert_eq!(session_id, SESSION_ID);
    assert_eq!(outcome, super::types::TurnCompletionOutcome::Failed);
    assert_eq!(content, expected_refusal);

    let failed = transport.receive().await;
    assert_eq!(failed["method"], "session/goal_update", "{failed}");
    let update: SessionGoalUpdate = serde_json::from_value(failed["params"].clone()).unwrap();
    let SessionGoalUpdate::Failed { session_id, reason } = update else {
        panic!("ordinary refusal must precede one Goal failure notice: {update:?}");
    };
    assert_eq!(session_id, SESSION_ID);
    let (expected_reason, terminal_reason) = match scenario {
        GoalRefusalScenario::Start => (
            zeroclaw_goal::control::GoalFailureReason::AgentOperationFailed,
            "parent_operation_failed",
        ),
        GoalRefusalScenario::AnsweredResume => (
            zeroclaw_goal::control::GoalFailureReason::ExecutorFailed,
            "executor_failed",
        ),
    };
    assert_eq!(reason, expected_reason);
    let task = registry
        .current_goal_for_session(&key.durable_id())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(task.id, admitted.task_id);
    assert_eq!(task.status, TaskStatus::Failed);
    assert_eq!(
        registry
            .terminal_reason_for_session_goal(&task.id, &key.durable_id())
            .await
            .unwrap()
            .as_deref(),
        Some(terminal_reason)
    );
    assert_eq!(
        fixture.server.received_requests().await.unwrap().len(),
        before_calls
    );
    assert_eq!(
        serde_json::to_value(fixture.backend.load(&key.durable_id())).unwrap(),
        before_history
    );
    assert_eq!(
        serde_json::to_value(agent.lock().await.history()).unwrap(),
        before_agent_history
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(100), transport.receive())
            .await
            .is_err(),
        "no duplicate ordinary or Goal failure notification is permitted"
    );
}

/// Keep the same scenario and assertions while retaining native framing.
pub(crate) enum GoalTestTransport {
    #[cfg(unix)]
    Local(
        tokio::io::BufReader<tokio::net::unix::OwnedReadHalf>,
        tokio::net::unix::OwnedWriteHalf,
    ),
    Wss(Box<WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>>),
}

impl GoalTestTransport {
    async fn send(&mut self, method: &str, params: Value, id: u64) {
        let request = json!({"jsonrpc":"2.0","method":method,"params":params,"id":id}).to_string();
        match self {
            #[cfg(unix)]
            Self::Local(_, writer) => writer
                .write_all(format!("{request}\n").as_bytes())
                .await
                .unwrap(),
            Self::Wss(socket) => socket.send(Message::Text(request.into())).await.unwrap(),
        }
    }

    async fn receive(&mut self) -> Value {
        let frame = tokio::time::timeout(Duration::from_secs(15), async {
            match self {
                #[cfg(unix)]
                Self::Local(reader, _) => {
                    let mut line = String::new();
                    assert!(reader.read_line(&mut line).await.unwrap() > 0);
                    serde_json::from_str(&line).unwrap()
                }
                Self::Wss(socket) => loop {
                    let frame = socket.next().await.unwrap().unwrap();
                    if let Message::Text(text) = frame {
                        break serde_json::from_str(&text).unwrap();
                    }
                },
            }
        })
        .await
        .expect("Goal transport must produce the next frame");
        eprintln!("Goal fixture received: {frame}");
        frame
    }
}

pub(crate) async fn assert_active_goal_scenario_with_objective(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
    remote: bool,
    objective: &str,
    scenario: ResumeScenario,
) {
    if scenario == ResumeScenario::Diagnostics {
        diagnostics::assert_diagnostics(fixture, transport, remote).await;
        return;
    }
    let tui_sig = fixture.context.tui_registry.sign(TUI_ID).unwrap();
    transport.send("initialize", json!({"protocol_version":1,"tui_id":TUI_ID,"tui_sig":tui_sig,"auth_token":remote.then_some("zc_goal_test")}), 1).await;
    let initialized = transport.receive().await;
    assert_eq!(initialized["id"], 1);
    assert!(initialized["error"].is_null(), "{initialized}");
    if scenario == ResumeScenario::PrefixMatchesObjective {
        let prefix = [
            ChatMessage::user(objective),
            ChatMessage::assistant(PREVIOUS_RESPONSE),
        ];
        for message in &prefix {
            fixture
                .backend
                .append(&format!("rpc_{SESSION_ID}"), message)
                .unwrap();
        }
        fixture
            .context
            .sessions
            .seed_history(SESSION_ID, &prefix)
            .await;
    }
    let mut task_id = None;
    let mut turns = vec![(
        format!("/goal start --unlimited {objective}"),
        QUESTION,
        true,
    )];
    if scenario == ResumeScenario::PlainThenAnswered {
        turns.push(("/goal resume".to_owned(), QUESTION, true));
    }
    turns.push((format!("/goal resume {ANSWER}"), COMPLETION, false));
    for (turn_index, (command, expected_text, blocked)) in turns.into_iter().enumerate() {
        let id = turn_index as u64 + 2;
        transport
            .send(
                "session/goal",
                json!({"session_id":SESSION_ID,"command":command}),
                id,
            )
            .await;
        let first = transport.receive().await;
        assert_eq!(
            first["id"], id,
            "lifecycle response must precede execution events: {first}"
        );
        assert!(first["error"].is_null(), "{first}");
        let result: SessionGoalResult = serde_json::from_value(first["result"].clone()).unwrap();
        let projection = match result.response {
            GoalResponse::Started(projection) if turn_index == 0 => projection,
            GoalResponse::Resumed(projection) if turn_index > 0 => projection,
            other => panic!("unexpected Goal admission: {other:?}"),
        };
        if let Some(task_id) = task_id.as_ref() {
            assert_eq!(&projection.task_id, task_id);
        } else {
            task_id = Some(projection.task_id);
        }
        let mut chunks = String::new();
        let mut candidate_seen = false;
        loop {
            let frame = transport.receive().await;
            assert!(frame["error"].is_null(), "{frame}");
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
                            assert_eq!(candidate, expected_text);
                            candidate_seen = true;
                        }
                        SessionGoalUpdate::PausedForBlocker { .. } if blocked => {
                            assert!(candidate_seen);
                            break;
                        }
                        SessionGoalUpdate::Completed { .. } if !blocked => {
                            assert!(candidate_seen);
                            break;
                        }
                        other => panic!("unexpected Goal lifecycle update: {other:?}"),
                    }
                }
                other => panic!("unexpected transport event: {other:?}: {frame}"),
            }
        }
        assert_eq!(
            chunks, expected_text,
            "ordinary streaming text must survive Goal verification"
        );
        if blocked {
            transport
                .send(
                    "session/goal",
                    json!({"session_id":SESSION_ID,"command":"/goal status"}),
                    20,
                )
                .await;
            let status = transport.receive().await;
            assert_eq!(status["id"], 20);
            let result: SessionGoalResult =
                serde_json::from_value(status["result"].clone()).unwrap();
            let GoalResponse::Status(paused) = result.response else {
                panic!("blocked Goal must remain queryable: {result:?}")
            };
            assert_eq!(paused.status, TaskStatus::Paused);
            assert_eq!(paused.pause_reason, Some(GoalPauseReason::NeedsUserInput));
            assert_eq!(paused.accounting_state, GoalAccountingState::Complete);
            assert!(paused.resumable);
            assert_eq!(
                paused.usage_totals.unwrap().total_tokens,
                24 * (turn_index as u64 + 1)
            );
            if turn_index == 1 {
                let history = fixture.backend.load(&format!("rpc_{SESSION_ID}"));
                assert_eq!(history.len(), 2, "{history:?}");
                assert!(
                    history.iter().all(|message| {
                        message.role == "assistant" && message.content == QUESTION
                    }),
                    "plain resume must persist both real replies and no fabricated user response"
                );
            }
        }
    }
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
    assert_eq!(projection.task_id, task_id.unwrap());
    assert_eq!(projection.accounting_state, GoalAccountingState::Complete);
    let expected_tokens = if scenario == ResumeScenario::PlainThenAnswered {
        72
    } else {
        48
    };
    assert_eq!(
        projection.usage_totals.unwrap().total_tokens,
        expected_tokens
    );
    fixture.assert_history_and_calls(objective, scenario).await;
}
