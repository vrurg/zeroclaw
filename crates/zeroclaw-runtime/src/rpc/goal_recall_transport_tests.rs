//! Recall lifetime through native authenticated Goal transports.

use super::*;
use std::sync::Mutex;
use zeroclaw_api::{
    attribution::{Attributable, MemoryKind, Role},
    memory_traits::{Memory, MemoryCategory, MemoryEntry},
};

const PROGRESS: &str = "Option A is selected; one declared criterion remains.";

/// Fixed verdicts test transport and controller behavior, not model judgment.
#[derive(Default)]
pub(super) struct RecallResponses(AtomicUsize);

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
                0 => QUESTION,
                1 => PROGRESS,
                _ => COMPLETION,
            }
            .to_owned()
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

#[derive(Default)]
pub(super) struct RecallMemory {
    queries: Mutex<Vec<(String, Option<String>)>>,
}

impl Attributable for RecallMemory {
    fn role(&self) -> Role {
        Role::Memory(MemoryKind::InMemory)
    }
    fn alias(&self) -> &str {
        "goal-recall-fixture"
    }
}

#[async_trait::async_trait]
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
        // Legacy/unattributed entries use the ordinary allowlist contract.
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

pub(super) async fn assert_recall_scenario(
    fixture: &ActiveGoalFixture,
    transport: &mut GoalTestTransport,
    remote: bool,
    objective: &str,
) {
    let tui_sig = fixture.context.tui_registry.sign(TUI_ID).unwrap();
    transport.send("initialize", json!({"protocol_version":1,"tui_id":TUI_ID,"tui_sig":tui_sig,"auth_token":remote.then_some("zc_goal_test")}), 1).await;
    let initialized = transport.receive().await;
    assert_eq!(initialized["id"], 1);
    assert!(initialized["error"].is_null(), "{initialized}");

    let start = format!("/goal start --unlimited {objective}");
    let resume = format!("/goal resume {ANSWER}");
    let mut task_id = None;
    for (index, (command, candidates)) in [
        (&start, vec![QUESTION]),
        (&resume, vec![PROGRESS, COMPLETION]),
    ]
    .into_iter()
    .enumerate()
    {
        let id = index as u64 + 2;
        transport
            .send(
                "session/goal",
                json!({"session_id":SESSION_ID,"command":command}),
                id,
            )
            .await;
        let first = transport.receive().await;
        assert_eq!(first["id"], id, "notice must precede execution: {first}");
        assert!(first["error"].is_null(), "{first}");
        let result: SessionGoalResult = serde_json::from_value(first["result"].clone()).unwrap();
        let projection = match result.response {
            GoalResponse::Started(projection) if index == 0 => projection,
            GoalResponse::Resumed(projection) if index == 1 => projection,
            other => panic!("unexpected admission: {other:?}"),
        };
        if let Some(previous) = &task_id {
            assert_eq!(&projection.task_id, previous);
        } else {
            task_id = Some(projection.task_id);
        }
        let mut chunks = String::new();
        let mut seen = Vec::new();
        loop {
            let frame = transport.receive().await;
            assert!(frame["error"].is_null(), "{frame}");
            match frame["method"].as_str() {
                Some("session/update") => {
                    let event: SessionUpdateEvent =
                        serde_json::from_value(frame["params"].clone()).unwrap();
                    if let SessionUpdateEvent::AgentMessageChunk { text, .. } = event {
                        chunks.push_str(&text);
                    }
                }
                Some("session/goal_update") => {
                    let event: SessionGoalUpdate =
                        serde_json::from_value(frame["params"].clone()).unwrap();
                    match event {
                        SessionGoalUpdate::VerifiedCandidate { candidate, .. } => {
                            assert_eq!(candidate, candidates[seen.len()]);
                            seen.push(candidate);
                        }
                        SessionGoalUpdate::PausedForBlocker { .. } if index == 0 => break,
                        SessionGoalUpdate::Completed { .. } if index == 1 => break,
                        other => panic!("unexpected lifecycle: {other:?}"),
                    }
                }
                _ => panic!("unexpected frame: {frame}"),
            }
        }
        assert_eq!(seen, candidates, "full candidates precede lifecycle");
        assert_eq!(chunks, candidates.concat(), "ordinary text remains visible");
        transport
            .send(
                "session/goal",
                json!({"session_id":SESSION_ID,"command":"/goal status"}),
                20 + index as u64,
            )
            .await;
        let status = transport.receive().await;
        let result: SessionGoalResult = serde_json::from_value(status["result"].clone()).unwrap();
        let projection = match result.response {
            GoalResponse::Status(projection) if index == 0 => {
                assert_eq!(projection.status, TaskStatus::Paused);
                assert_eq!(
                    projection.pause_reason,
                    Some(GoalPauseReason::NeedsUserInput)
                );
                assert!(projection.resumable);
                projection
            }
            GoalResponse::Terminal(projection) if index == 1 => {
                assert_eq!(projection.status, TaskStatus::Completed);
                projection
            }
            other => panic!("unexpected status: {other:?}"),
        };
        assert_eq!(Some(&projection.task_id), task_id.as_ref());
        assert_eq!(projection.accounting_state, GoalAccountingState::Complete);
        assert_eq!(
            projection.usage_totals.unwrap().total_tokens,
            if index == 0 { 24 } else { 72 }
        );
    }

    let history = fixture.backend.load(&format!("rpc_{SESSION_ID}"));
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
    let requests = fixture.server.received_requests().await.unwrap();
    assert_eq!(
        requests.len(),
        6,
        "one verifier per parent; no extra successor"
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
            usize::from(fixture.recall.is_some() && index % 2 == 0),
            "provider-only recall in request {index}"
        );
        if index % 2 == 0 {
            let user = rows.iter().rev().find(|row| row["role"] == "user").unwrap();
            assert_eq!(
                user["content"]
                    .as_str()
                    .unwrap()
                    .starts_with(zeroclaw_memory::MEMORY_CONTEXT_OPEN),
                fixture.recall.is_some()
            );
            if index == 2 {
                assert!(user["content"].as_str().unwrap().ends_with(ANSWER));
            }
            if index > 0 {
                assert!(
                    rows.iter()
                        .any(|row| row["role"] == "assistant" && row["content"] == QUESTION)
                );
            }
            if index == 4 {
                assert!(
                    rows.iter()
                        .any(|row| row["role"] == "user" && row["content"] == ANSWER)
                );
            }
        }
    }
    let Some(memory) = fixture.recall.as_ref() else {
        return;
    };
    let queries = memory.queries.lock().unwrap();
    assert!(queries.iter().any(|(query, _)| query == &start));
    assert!(queries.iter().any(|(query, _)| query == &resume));
    for (query, scope) in queries.iter() {
        assert!(
            query == &start || query == &resume,
            "unexpected query: {query}"
        );
        assert_eq!(scope.as_deref(), Some(format!("rpc_{SESSION_ID}").as_str()));
    }
}

#[tokio::test]
async fn ordinary_streamed_turn_recall_control() {
    if run_isolated(
        "rpc::goal_transport_test_support::recall::ordinary_streamed_turn_recall_control",
    )
    .await
    {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let fixture = ActiveGoalFixture::with_recall(tmp.path()).await;
    let agent = fixture
        .context
        .sessions
        .get_agent(SESSION_ID)
        .await
        .unwrap();
    let (events, mut receiver) = tokio::sync::mpsc::channel(64);
    let drain = ::zeroclaw_spawn::spawn!(async move { while receiver.recv().await.is_some() {} });
    let query = "Ordinary recall control.";
    let (response, _) = agent
        .lock()
        .await
        .turn_streamed(query, events, None)
        .await
        .unwrap();
    drain.await.unwrap();
    assert_eq!(response, QUESTION);
    let requests = fixture.server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    let rows = body["messages"].as_array().unwrap();
    let user = rows.iter().rev().find(|row| row["role"] == "user").unwrap();
    assert!(
        user["content"]
            .as_str()
            .unwrap()
            .starts_with(zeroclaw_memory::MEMORY_CONTEXT_OPEN)
    );
    assert!(user["content"].as_str().unwrap().ends_with(query));
    let memory = fixture.recall.as_ref().unwrap();
    let queries = memory.queries.lock().unwrap();
    assert!(!queries.is_empty());
    assert!(queries.iter().all(|(text, scope)| text == query
        && scope.as_deref() == Some(format!("rpc_{SESSION_ID}").as_str())));
}
