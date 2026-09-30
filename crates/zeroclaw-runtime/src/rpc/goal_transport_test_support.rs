//! Shared active-Goal conformance fixture for real ZeroCode transports.

use std::{path::Path, sync::Arc, time::Duration};

use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, tungstenite::Message};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate, matchers::method};
use zeroclaw_api::task::TaskStatus;
use zeroclaw_config::schema::{AliasedAgentConfig, Config};
use zeroclaw_goal::{GoalAccountingState, GoalPauseReason, control::GoalResponse};
use zeroclaw_infra::{session_backend::SessionBackend, session_sqlite::SqliteSessionBackend};

use super::{
    context::RpcContext,
    session::{RpcSession, SessionStore},
    types::{ChatMode, SessionGoalResult, SessionGoalUpdate, SessionUpdateEvent},
};

pub(crate) const SESSION_ID: &str = "goal-active-transport";
pub(crate) const TUI_ID: &str = "goal-active-client";
const QUESTION: &str = "Please choose option A or B before I continue.";
pub(crate) const ANSWER: &str = "Choose A.\nKeep the second line unchanged.";
pub(crate) const OBJECTIVE: &str = "Apply the selected option after asking the user to choose.";
const COMPLETION: &str = "Option A is applied; the declared Goal is complete.";

/// Exercise answer insertion separately from a resume without user input.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResumeScenario {
    Answered,
    PlainThenAnswered,
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
        eprintln!("scripted model request: {}", body["model"]);
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
}

impl ActiveGoalFixture {
    pub(crate) async fn new(path: &Path) -> Self {
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
        let agent_config = AliasedAgentConfig {
            enabled: true,
            model_provider: "openai.goal-test".into(),
            ..Default::default()
        };
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
        let context =
            RpcContext::for_persistence_tests(config, sessions, Some(backend.clone()), None);
        context
            .sessions
            .insert(
                SESSION_ID.to_owned(),
                RpcSession::new(agent, "test-agent", path.to_str().unwrap(), ChatMode::Chat)
                    .with_owner(Some(TUI_ID.to_owned())),
            )
            .await
            .unwrap();
        Self {
            context,
            backend,
            server,
        }
    }

    async fn assert_history_and_calls(&self, objective: &str, scenario: ResumeScenario) {
        let history = self.backend.load(&format!("rpc_{SESSION_ID}"));
        let mut expected = vec![("assistant", QUESTION)];
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

/// Keep the same scenario and assertions while retaining native framing.
pub(crate) enum GoalTestTransport {
    #[cfg(unix)]
    Local(
        tokio::io::BufReader<tokio::net::unix::OwnedReadHalf>,
        tokio::net::unix::OwnedWriteHalf,
    ),
    Wss(WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>),
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
    transport.send("initialize", json!({"protocol_version":1,"tui_id":TUI_ID,"auth_token":remote.then_some("zc_goal_test")}), 1).await;
    let initialized = transport.receive().await;
    assert_eq!(initialized["id"], 1);
    assert!(initialized["error"].is_null(), "{initialized}");
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
