use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{Value, json};
use zeroclaw_api::attribution::{Attributable, Role, ToolKind};
use zeroclaw_api::channel::{Channel, SendMessage};
use zeroclaw_api::tool::{Tool, ToolResult};
use zeroclaw_config::policy::{SecurityPolicy, ToolOperation};
use zeroclaw_config::schema::Config;
use zeroclaw_tools::ask_user::format_question;
use zeroclaw_tools::escalate::is_valid_urgency_level;

use crate::agent::turn::ToolLoopCancelled;
use crate::control_plane::{
    GoalBlockerKind, TaskContinuationContext, control_plane, current_goal_admission_context,
    current_goal_turn_evaluation_requested, pause_current_goal_for_human_gate,
};

use super::PerToolChannelHandle;

/// Human-gate wrapper behavior selected for a concrete underlying tool.
///
/// Both variants produce a durable goal pause before handing control to a human,
/// but they differ in the blocker kind and delivery mechanism they use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HumanGateKind {
    /// Wrap `ask_user` and persist a "needs user input" pause before delivery.
    AskUser,
    /// Wrap `escalate_to_human` and persist a human-escalation pause before
    /// delivery.
    EscalateToHuman,
}

/// Goal-aware wrapper for tools that intentionally hand control back to a
/// human.
///
/// Outside goal mode this delegates to the wrapped tool unchanged. Inside goal
/// mode it first records a durable pause/blocker on the active goal, then
/// performs the underlying delivery and cancels the current tool loop so the
/// agent does not keep working past a human gate.
pub struct GoalHumanGateTool {
    /// Original user-facing tool implementation.
    inner: Arc<dyn Tool>,
    /// Which pause/blocker semantics to apply to `inner`.
    kind: HumanGateKind,
    /// Runtime policy gate that must still authorize the wrapped action.
    security: Arc<SecurityPolicy>,
    /// Channel handle used by `ask_user` delivery.
    channels: PerToolChannelHandle,
    /// Shared runtime config used to resolve the active goal controller.
    config: Arc<Config>,
}

impl GoalHumanGateTool {
    pub fn ask_user(
        inner: Arc<dyn Tool>,
        security: Arc<SecurityPolicy>,
        channels: PerToolChannelHandle,
        config: Arc<Config>,
    ) -> Self {
        Self {
            inner,
            kind: HumanGateKind::AskUser,
            security,
            channels,
            config,
        }
    }

    pub fn escalate_to_human(
        inner: Arc<dyn Tool>,
        security: Arc<SecurityPolicy>,
        channels: PerToolChannelHandle,
        config: Arc<Config>,
    ) -> Self {
        Self {
            inner,
            kind: HumanGateKind::EscalateToHuman,
            security,
            channels,
            config,
        }
    }

    async fn execute_goal_gate(&self, args: &Value) -> Result<Option<ToolResult>> {
        if !current_goal_turn_evaluation_requested() {
            return Ok(None);
        }
        let ctx = current_goal_admission_context().ok_or_else(|| {
            anyhow::Error::new(ToolLoopCancelled)
                .context("goal human gate has no durable admission context")
        })?;
        if ctx
            .goal_task_id
            .as_deref()
            .is_none_or(|task_id| task_id.trim().is_empty())
        {
            return Err(anyhow::Error::new(ToolLoopCancelled)
                .context("goal human gate has no exact durable task binding"));
        }
        if let Err(error) = self
            .security
            .enforce_tool_operation(ToolOperation::Act, self.inner.name())
        {
            return Ok(Some(ToolResult {
                success: false,
                output: String::new().into(),
                error: Some(format!("Action blocked: {error}")),
            }));
        }

        if self.kind == HumanGateKind::AskUser
            && args
                .get("channel")
                .and_then(Value::as_str)
                .is_some_and(|channel| !channel.trim().is_empty())
        {
            return Ok(Some(ToolResult {
                success: false,
                output: String::new().into(),
                error: Some(crate::i18n::get_required_tool_string(
                    "tool-goal-human-gate-error-channel-override",
                )),
            }));
        }

        let gate = match self.kind {
            HumanGateKind::AskUser => match parse_ask_user_request(args) {
                Some(request) => HumanGatePause::AskUser(request),
                None => return Ok(None),
            },
            HumanGateKind::EscalateToHuman => match parse_escalation_request(args) {
                Some(request) => HumanGatePause::EscalateToHuman(
                    request.scrubbed(&self.config.security.leak_detection),
                ),
                None => return Ok(None),
            },
        };

        let (kind, message, payload) =
            gate.pause_packet(self.inner.name(), &self.config.security.leak_detection);
        let admission = pause_current_goal_for_human_gate(
            &ctx,
            Some(&self.config),
            kind,
            message,
            Some(payload),
        )
        .await?;

        let continuation = match (&gate, admission.task_id.as_deref()) {
            (HumanGatePause::AskUser(_), Some(task_id)) => match control_plane() {
                Some(control_plane) => control_plane
                    .goal_store
                    .get_continuation_context(task_id)
                    .await
                    .with_context(|| {
                        format!("load durable continuation context for goal human gate {task_id}")
                    }),
                None => Err(anyhow::Error::msg(
                    "goal control plane unavailable after human-gate pause",
                )),
            },
            _ => Ok(None),
        };
        finish_goal_human_gate_after_pause(
            &gate,
            &self.channels,
            self.inner.as_ref(),
            continuation,
            &self.config.security.leak_detection,
        )
        .await
    }
}

async fn finish_goal_human_gate_after_pause(
    gate: &HumanGatePause,
    channels: &PerToolChannelHandle,
    inner: &dyn Tool,
    continuation: Result<Option<TaskContinuationContext>>,
    leak_detection: &zeroclaw_config::schema::LeakDetectionConfig,
) -> Result<Option<ToolResult>> {
    // Once the pause commits, notification is best effort. Preparation or
    // transport failure must not escape as an ordinary tool error because the
    // executor could otherwise continue past the durable human gate.
    let delivery = match continuation {
        Ok(continuation) => {
            gate.deliver(channels, inner, continuation.as_ref(), leak_detection)
                .await
        }
        Err(error) => Err(error),
    };
    if let Err(error) = delivery {
        ::zeroclaw_log::record!(
            WARN,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                .with_category(::zeroclaw_log::EventCategory::Tool)
                .with_outcome(::zeroclaw_log::EventOutcome::Unknown)
                .with_attrs(::serde_json::json!({
                    "tool": inner.name(),
                    "error": format!("{error:#}"),
                })),
            "goal human gate notification failed; the durable blocker remains recoverable through goal status and resume"
        );
    }

    Err(ToolLoopCancelled.into())
}

impl Attributable for GoalHumanGateTool {
    fn role(&self) -> Role {
        Role::Tool(ToolKind::Wait)
    }

    fn alias(&self) -> &str {
        self.inner.name()
    }
}

#[async_trait]
impl Tool for GoalHumanGateTool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn parameters_schema(&self) -> Value {
        self.inner.parameters_schema()
    }

    async fn execute(&self, args: Value) -> Result<ToolResult> {
        match self.execute_goal_gate(&args).await {
            Ok(Some(result)) => Ok(result),
            Ok(None) => self.inner.execute(args).await,
            Err(error) => Err(error),
        }
    }
}

/// Parsed human-gate request after tool-specific validation.
///
/// This separates extraction/validation from pause persistence so malformed
/// tool calls can fall back to the original tool behavior instead of creating
/// misleading goal blockers. The enum is transient: durable goal state is the
/// `GoalPauseState` written by `pause_current_goal_for_human_gate`.
enum HumanGatePause {
    /// Validated `ask_user` request.
    AskUser(AskUserRequest),
    /// Validated `escalate_to_human` request.
    EscalateToHuman(EscalationRequest),
}

impl HumanGatePause {
    fn pause_packet(
        &self,
        tool_name: &str,
        leak_detection: &zeroclaw_config::schema::LeakDetectionConfig,
    ) -> (GoalBlockerKind, String, Value) {
        match self {
            Self::AskUser(request) => {
                let question =
                    crate::security::scrub_with_config(&request.question, leak_detection);
                let choices = request.choices.as_ref().map(|choices| {
                    choices
                        .iter()
                        .map(|choice| crate::security::scrub_with_config(choice, leak_detection))
                        .collect::<Vec<_>>()
                });
                (
                    GoalBlockerKind::NeedsUserInput,
                    question.clone(),
                    json!({
                        "tool": tool_name,
                        "question": question,
                        "choices": choices,
                        "delivery": {
                            "route": "durable_goal_continuation",
                            "notification": "best_effort"
                        },
                    }),
                )
            }
            Self::EscalateToHuman(request) => (
                GoalBlockerKind::HumanEscalation,
                request.summary.clone(),
                json!({
                    "tool": tool_name,
                    "summary": request.summary,
                    "context": request.context,
                    "urgency": request.urgency,
                    "wait_for_response_requested": request.wait_for_response,
                }),
            ),
        }
    }

    async fn deliver(
        &self,
        channels: &PerToolChannelHandle,
        inner: &dyn Tool,
        continuation: Option<&TaskContinuationContext>,
        leak_detection: &zeroclaw_config::schema::LeakDetectionConfig,
    ) -> Result<()> {
        match self {
            Self::AskUser(request) => {
                request
                    .deliver(
                        channels,
                        continuation.context(
                            "goal ask_user has no durable continuation delivery context",
                        )?,
                        leak_detection,
                    )
                    .await
            }
            Self::EscalateToHuman(request) => request.deliver(inner).await,
        }
    }
}

/// Validated subset of `ask_user` arguments that matters for goal pausing and
/// channel delivery.
///
/// This is intentionally narrower than the tool schema. It captures only the
/// fields needed to produce an actionable blocker and deliver the same prompt;
/// anything else remains owned by the underlying `ask_user` tool.
struct AskUserRequest {
    /// Prompt to send to the operator. This is model-supplied text and remains
    /// untrusted from the controller's point of view.
    question: String,
    /// Optional model-supplied answer choices.
    choices: Option<Vec<String>>,
}

fn parse_ask_user_request(args: &Value) -> Option<AskUserRequest> {
    let question = args
        .get("question")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())?
        .to_string();
    let choices = args.get("choices").and_then(|value| {
        value.as_array().map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::trim))
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
    });
    Some(AskUserRequest { question, choices })
}

impl AskUserRequest {
    async fn deliver(
        &self,
        channels: &PerToolChannelHandle,
        continuation: &TaskContinuationContext,
        leak_detection: &zeroclaw_config::schema::LeakDetectionConfig,
    ) -> Result<()> {
        let (channel_name, channel) = resolve_trusted_channel(channels, continuation)?;
        let message = crate::security::scrub_with_config(
            &format_question(&self.question, self.choices.as_deref()),
            leak_detection,
        );
        let recipient = continuation.reply_target.trim();
        if recipient.is_empty() {
            anyhow::bail!("goal ask_user durable reply target is empty");
        }
        channel
            .send(&SendMessage::new(message, recipient).in_thread(continuation.thread_ts.clone()))
            .await
            .with_context(|| format!("send goal ask_user prompt to channel {channel_name}"))
    }
}

/// Validated subset of `escalate_to_human` arguments used to pause a goal and
/// send the escalation.
///
/// The wrapper forces the delivered escalation to be non-blocking after the
/// durable pause is recorded. The operator-visible wait intent remains in the
/// blocker payload for context, but it is not a second lifecycle source.
struct EscalationRequest {
    /// Short model-supplied escalation summary.
    summary: String,
    /// Optional model-supplied context included in the escalation payload.
    context: Option<String>,
    /// Validated urgency string accepted by the underlying escalation tool.
    urgency: String,
    /// Whether the model requested waiting. Goal mode always pauses regardless;
    /// this value is retained in the blocker payload for operator context.
    wait_for_response: bool,
}

fn parse_escalation_request(args: &Value) -> Option<EscalationRequest> {
    let summary = args
        .get("summary")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())?
        .to_string();
    let context = args
        .get("context")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let urgency = args
        .get("urgency")
        .and_then(Value::as_str)
        .unwrap_or("medium")
        .trim();
    if !is_valid_urgency_level(urgency) {
        return None;
    }
    let wait_for_response = args
        .get("wait_for_response")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Some(EscalationRequest {
        summary,
        context,
        urgency: urgency.to_string(),
        wait_for_response,
    })
}

impl EscalationRequest {
    fn scrubbed(mut self, leak_detection: &zeroclaw_config::schema::LeakDetectionConfig) -> Self {
        self.summary = crate::security::scrub_with_config(&self.summary, leak_detection);
        self.context = self
            .context
            .as_deref()
            .map(|value| crate::security::scrub_with_config(value, leak_detection));
        self
    }

    async fn deliver(&self, inner: &dyn Tool) -> Result<()> {
        let mut args = json!({
            "summary": self.summary,
            "urgency": self.urgency,
            "wait_for_response": false,
        });
        if let Some(context) = &self.context {
            args["context"] = Value::String(context.clone());
        }
        let result = inner.execute(args).await?;
        if result.success {
            Ok(())
        } else {
            anyhow::bail!(
                "{}",
                result.error.unwrap_or_else(|| result.output.to_string())
            )
        }
    }
}

fn resolve_trusted_channel(
    channels: &PerToolChannelHandle,
    continuation: &TaskContinuationContext,
) -> Result<(String, Arc<dyn Channel>)> {
    let channel_name = continuation
        .channel_key()
        .context("goal ask_user durable channel identity is empty")?;
    let channels = channels.read();
    let channel = channels
        .get(channel_name.as_str())
        .cloned()
        .with_context(|| format!("durable goal channel {channel_name:?} is unavailable"))?;
    Ok((channel_name, channel))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use zeroclaw_api::attribution::{ChannelKind, Role};
    use zeroclaw_api::channel::ChannelMessage;

    /// Minimal wrapped-tool fixture that records whether the underlying tool was
    /// called after goal human-gate handling.
    struct RecordingTool {
        /// Tool name reported through the `Tool` and attribution traits.
        name: &'static str,
        /// Number of times the wrapped implementation actually executed.
        calls: Arc<AtomicUsize>,
        /// Raw argument payloads observed by the wrapped implementation.
        args: Arc<Mutex<Vec<Value>>>,
    }

    impl Attributable for RecordingTool {
        fn role(&self) -> Role {
            Role::Tool(ToolKind::Wait)
        }

        fn alias(&self) -> &str {
            self.name
        }
    }

    #[async_trait]
    impl Tool for RecordingTool {
        fn name(&self) -> &str {
            self.name
        }

        fn description(&self) -> &str {
            "recording tool"
        }

        fn parameters_schema(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }

        async fn execute(&self, args: Value) -> Result<ToolResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.args.lock().unwrap().push(args);
            Ok(ToolResult {
                success: true,
                output: "delegated".into(),
                error: None,
            })
        }
    }

    /// Minimal channel fixture for goal human-gate delivery assertions.
    struct RecordingChannel {
        /// Messages sent through the channel by the wrapper.
        sent: Arc<Mutex<Vec<SendMessage>>>,
        /// Whether delivery should fail after recording the attempt.
        fail: bool,
    }

    impl Attributable for RecordingChannel {
        fn role(&self) -> Role {
            Role::Channel(ChannelKind::Webhook)
        }

        fn alias(&self) -> &str {
            "goal-human-gate-test"
        }
    }

    #[async_trait]
    impl Channel for RecordingChannel {
        fn name(&self) -> &str {
            "goal-human-gate-test"
        }

        async fn send(&self, message: &SendMessage) -> Result<()> {
            self.sent.lock().unwrap().push(message.clone());
            if self.fail {
                anyhow::bail!("synthetic channel delivery failure");
            }
            Ok(())
        }

        async fn listen(&self, _tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> Result<()> {
            Ok(())
        }
    }

    fn empty_channels() -> PerToolChannelHandle {
        Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new()))
    }

    type RecordingToolParts = (Arc<dyn Tool>, Arc<AtomicUsize>, Arc<Mutex<Vec<Value>>>);

    fn recording_tool(name: &'static str) -> RecordingToolParts {
        let calls = Arc::new(AtomicUsize::new(0));
        let args = Arc::new(Mutex::new(Vec::new()));
        (
            Arc::new(RecordingTool {
                name,
                calls: Arc::clone(&calls),
                args: Arc::clone(&args),
            }),
            calls,
            args,
        )
    }

    fn recording_channels() -> (PerToolChannelHandle, Arc<Mutex<Vec<SendMessage>>>) {
        let sent = Arc::new(Mutex::new(Vec::new()));
        let channel: Arc<dyn Channel> = Arc::new(RecordingChannel {
            sent: Arc::clone(&sent),
            fail: false,
        });
        let mut channels = HashMap::new();
        channels.insert("matrix".to_string(), channel);
        (Arc::new(parking_lot::RwLock::new(channels)), sent)
    }

    fn ensure_test_control_plane() -> (
        Arc<dyn crate::control_plane::TaskRegistry>,
        Arc<dyn crate::control_plane::GoalTaskRegistry>,
    ) {
        match crate::control_plane::control_plane() {
            Some(control_plane) => (
                Arc::clone(&control_plane.store),
                Arc::clone(&control_plane.goal_store),
            ),
            None => {
                let sqlite_store =
                    Arc::new(crate::control_plane::SqliteTaskStore::new_in_memory().unwrap());
                let store: Arc<dyn crate::control_plane::TaskRegistry> = sqlite_store.clone();
                let goal_store: Arc<dyn crate::control_plane::GoalTaskRegistry> = sqlite_store;
                let _ = crate::control_plane::init_control_plane(
                    crate::control_plane::ControlPlaneHandle {
                        store: Arc::clone(&store),
                        goal_store: Arc::clone(&goal_store),
                        boot_id: "test-boot".into(),
                        recovered_goal_ids: Arc::new(std::sync::Mutex::new(Vec::new())),
                        data_dir_lock: None,
                    },
                );
                (
                    Arc::clone(&crate::control_plane::control_plane().unwrap().store),
                    Arc::clone(&crate::control_plane::control_plane().unwrap().goal_store),
                )
            }
        }
    }

    async fn create_running_goal(
        agent: &str,
        route: &str,
        principal: &str,
    ) -> (
        Arc<dyn crate::control_plane::TaskRegistry>,
        Arc<dyn crate::control_plane::GoalTaskRegistry>,
        String,
        crate::control_plane::GoalAdmissionContext,
    ) {
        let (store, goal_store) = ensure_test_control_plane();
        let task_id = format!("goal-{}", uuid::Uuid::new_v4());
        let continuation = crate::control_plane::TaskContinuationContext {
            channel: "matrix".into(),
            channel_alias: None,
            reply_target: route.to_string(),
            sender: principal.to_string(),
            transport_principal: Some(principal.to_string()),
            thread_ts: Some("thread-a".into()),
            interruption_scope_id: None,
            conversation_scope:
                crate::control_plane::TaskContinuationConversationScope::ReplyTarget,
        };
        let ctx = crate::control_plane::GoalAdmissionContext::new(agent.to_string())
            .with_channel_type(Some("matrix".into()))
            .with_originator_route(Some(route.to_string()))
            .with_principal_id(Some(principal.to_string()))
            .with_goal_task_id(Some(task_id.clone()))
            .with_continuation_context(Some(continuation.clone()));
        goal_store
            .create_goal(
                crate::control_plane::TaskRecord {
                    id: task_id.clone(),
                    kind: crate::control_plane::TaskKind::Goal,
                    agent: agent.to_string(),
                    status: crate::control_plane::TaskStatus::Running,
                    owner_pid: std::process::id(),
                    owner_boot_id: "test-boot".into(),
                    heartbeat_at: None,
                    depth: 0,
                    parent_id: None,
                    originator_route: Some(route.to_string()),
                    delivered: false,
                    idem_key: None,
                    principal_id: Some(principal.to_string()),
                    started_at: chrono::Utc::now().to_rfc3339(),
                    finished_at: None,
                },
                crate::control_plane::GoalTaskRecord {
                    task_id: task_id.clone(),
                    objective: "wait for a human gate".into(),
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                },
                Some(continuation),
            )
            .await
            .unwrap();
        (store, goal_store, task_id, ctx)
    }

    async fn assert_paused_goal(
        store: &dyn crate::control_plane::TaskRegistry,
        goal_store: &dyn crate::control_plane::GoalTaskRegistry,
        task_id: &str,
        reason: crate::control_plane::GoalPauseReason,
        kind: crate::control_plane::GoalBlockerKind,
        tool_name: &str,
    ) -> crate::control_plane::GoalTaskRecord {
        let task = store.get(task_id).await.unwrap().unwrap();
        assert_eq!(task.status, crate::control_plane::TaskStatus::Paused);
        let goal = goal_store.get_goal_task(task_id).await.unwrap().unwrap();
        assert_eq!(goal.pause_reason, Some(reason));
        assert_eq!(goal.blockers.len(), 1);
        assert_eq!(goal.blockers[0].kind, kind);
        assert_eq!(
            goal.blockers[0].payload.as_ref().unwrap()["tool"],
            tool_name
        );
        goal
    }

    async fn scope_active_goal_work<F>(
        ctx: crate::control_plane::GoalAdmissionContext,
        future: F,
    ) -> F::Output
    where
        F: std::future::Future,
    {
        crate::control_plane::scope_goal_turn_evaluation_marker(
            Some(Arc::new(AtomicBool::new(true))),
            crate::control_plane::scope_goal_admission_context(Some(ctx), future),
        )
        .await
    }

    #[test]
    fn parse_ask_user_request_rejects_missing_question() {
        assert!(parse_ask_user_request(&json!({})).is_none());
        assert!(parse_ask_user_request(&json!({"question": "  "})).is_none());
    }

    #[test]
    fn parse_escalation_request_rejects_invalid_urgency() {
        assert!(parse_escalation_request(&json!({"summary": "help", "urgency": "now"})).is_none());
    }

    #[test]
    fn human_gate_pause_packet_scrubs_sensitive_presentation_content() {
        let request = parse_ask_user_request(&json!({
            "question": "Use sk_test_1234567890abcdefghijklmnop?",
            "choices": ["yes", "password=longsecretvalue"]
        }))
        .unwrap();
        let (_kind, message, payload) = HumanGatePause::AskUser(request).pause_packet(
            "ask_user",
            &zeroclaw_config::schema::LeakDetectionConfig::default(),
        );

        assert!(!message.contains("sk_test_"));
        assert!(!payload.to_string().contains("longsecretvalue"));
        assert!(payload.to_string().contains("REDACTED"));
    }

    #[test]
    fn escalation_delivery_forces_nonblocking_inner_call() {
        let request = parse_escalation_request(&json!({
            "summary": "help",
            "urgency": "high",
            "wait_for_response": true,
        }))
        .unwrap();
        assert!(request.wait_for_response);
        let (_kind, _message, payload) = HumanGatePause::EscalateToHuman(request).pause_packet(
            "escalate_to_human",
            &zeroclaw_config::schema::LeakDetectionConfig::default(),
        );
        assert_eq!(payload["wait_for_response_requested"], true);
    }

    #[tokio::test]
    async fn wrappers_keep_tool_names_and_delegate_without_goal_context() {
        let config = Arc::new(Config::default());
        let security = Arc::new(SecurityPolicy::default());

        let ask_calls = Arc::new(AtomicUsize::new(0));
        let ask_inner: Arc<dyn Tool> = Arc::new(RecordingTool {
            name: "ask_user",
            calls: Arc::clone(&ask_calls),
            args: Arc::new(Mutex::new(Vec::new())),
        });
        let ask = GoalHumanGateTool::ask_user(
            ask_inner,
            Arc::clone(&security),
            empty_channels(),
            Arc::clone(&config),
        );
        assert_eq!(ask.name(), "ask_user");
        let result = ask.execute(json!({"question": "continue?"})).await.unwrap();
        assert!(result.success);
        assert_eq!(ask_calls.load(Ordering::SeqCst), 1);

        let escalate_calls = Arc::new(AtomicUsize::new(0));
        let escalate_inner: Arc<dyn Tool> = Arc::new(RecordingTool {
            name: "escalate_to_human",
            calls: Arc::clone(&escalate_calls),
            args: Arc::new(Mutex::new(Vec::new())),
        });
        let escalate = GoalHumanGateTool::escalate_to_human(
            escalate_inner,
            security,
            empty_channels(),
            config,
        );
        assert_eq!(escalate.name(), "escalate_to_human");
        let result = escalate.execute(json!({"summary": "help"})).await.unwrap();
        assert!(result.success);
        assert_eq!(escalate_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn active_goal_marker_without_binding_context_cancels_without_delivery() {
        let config = Arc::new(Config::default());
        let security = Arc::new(SecurityPolicy::default());
        let (inner, calls, _args) = recording_tool("ask_user");
        let (channels, sent) = recording_channels();
        let tool = GoalHumanGateTool::ask_user(inner, security, channels, config);

        let err = crate::control_plane::scope_goal_turn_evaluation_marker(
            Some(Arc::new(AtomicBool::new(true))),
            tool.execute(json!({"question": "continue?"})),
        )
        .await
        .unwrap_err();

        assert!(err.is::<ToolLoopCancelled>(), "unexpected error: {err:#}");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(sent.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn ask_user_wrapper_pauses_active_goal_and_cancels_tool_loop() {
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let (store, goal_store, task_id, ctx) =
            create_running_goal(&agent, &route, &principal).await;
        let (inner, calls, _args) = recording_tool("ask_user");
        let (channels, sent) = recording_channels();
        let tool = GoalHumanGateTool::ask_user(
            inner,
            Arc::new(SecurityPolicy::default()),
            channels,
            Arc::new(Config::default()),
        );

        let err = scope_active_goal_work(
            ctx,
            tool.execute(json!({
                "question": "continue?",
                "choices": ["yes", "no"]
            })),
        )
        .await
        .unwrap_err();

        assert!(err.is::<ToolLoopCancelled>(), "unexpected error: {err:#}");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "goal ask_user wrapper should deliver directly and not enter the blocking inner tool"
        );
        {
            let sent = sent.lock().unwrap();
            assert_eq!(sent.len(), 1);
            assert!(sent[0].content.contains("continue?"));
            assert!(sent[0].content.contains("1. yes"));
            assert_eq!(sent[0].recipient, route);
            assert_eq!(sent[0].thread_ts.as_deref(), Some("thread-a"));
        }

        let goal = assert_paused_goal(
            store.as_ref(),
            goal_store.as_ref(),
            &task_id,
            crate::control_plane::GoalPauseReason::NeedsUserInput,
            crate::control_plane::GoalBlockerKind::NeedsUserInput,
            "ask_user",
        )
        .await;
        let payload = goal.blockers[0].payload.as_ref().unwrap();
        assert_eq!(payload["question"], "continue?");
        assert_eq!(payload["choices"], json!(["yes", "no"]));
        assert!(payload.get("channel").is_none());
        assert_eq!(payload["delivery"]["route"], "durable_goal_continuation");
    }

    #[tokio::test]
    async fn goal_ask_user_rejects_model_channel_override_before_pause() {
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let (store, _goal_store, task_id, ctx) =
            create_running_goal(&agent, &route, &principal).await;
        let (inner, calls, _args) = recording_tool("ask_user");
        let (channels, sent) = recording_channels();
        let tool = GoalHumanGateTool::ask_user(
            inner,
            Arc::new(SecurityPolicy::default()),
            channels,
            Arc::new(Config::default()),
        );

        let result = scope_active_goal_work(
            ctx,
            tool.execute(json!({
                "question": "send this elsewhere",
                "channel": "attacker-controlled"
            })),
        )
        .await
        .unwrap();

        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("cannot override its channel")
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(sent.lock().unwrap().is_empty());
        assert_eq!(
            store.get(&task_id).await.unwrap().unwrap().status,
            crate::control_plane::TaskStatus::Running
        );
    }

    #[tokio::test]
    async fn goal_ask_user_missing_continuation_stays_paused_without_delivery() {
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let (store, goal_store, task_id, ctx) =
            create_running_goal(&agent, &route, &principal).await;
        goal_store
            .set_continuation_context(&task_id, None)
            .await
            .unwrap();
        let (inner, calls, _args) = recording_tool("ask_user");
        let (channels, sent) = recording_channels();
        let tool = GoalHumanGateTool::ask_user(
            inner,
            Arc::new(SecurityPolicy::default()),
            channels,
            Arc::new(Config::default()),
        );

        let err = scope_active_goal_work(ctx, tool.execute(json!({"question": "continue?"})))
            .await
            .unwrap_err();

        assert!(err.is::<ToolLoopCancelled>(), "unexpected error: {err:#}");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(sent.lock().unwrap().is_empty());
        let goal = assert_paused_goal(
            store.as_ref(),
            goal_store.as_ref(),
            &task_id,
            crate::control_plane::GoalPauseReason::NeedsUserInput,
            crate::control_plane::GoalBlockerKind::NeedsUserInput,
            "ask_user",
        )
        .await;
        assert_eq!(goal.blockers[0].message, "continue?");
    }

    #[tokio::test]
    async fn continuation_lookup_failure_after_pause_still_cancels_tool_loop() {
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let (store, goal_store, task_id, ctx) =
            create_running_goal(&agent, &route, &principal).await;
        let (inner, calls, _args) = recording_tool("ask_user");
        let (channels, sent) = recording_channels();
        let gate = HumanGatePause::AskUser(
            parse_ask_user_request(&json!({"question": "continue?"})).unwrap(),
        );
        let config = Config::default();
        let (kind, message, payload) =
            gate.pause_packet("ask_user", &config.security.leak_detection);
        pause_current_goal_for_human_gate(&ctx, Some(&config), kind, message, Some(payload))
            .await
            .unwrap();

        let err = finish_goal_human_gate_after_pause(
            &gate,
            &channels,
            inner.as_ref(),
            Err(anyhow::Error::msg("synthetic continuation lookup failure")),
            &config.security.leak_detection,
        )
        .await
        .unwrap_err();

        assert!(err.is::<ToolLoopCancelled>(), "unexpected error: {err:#}");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(sent.lock().unwrap().is_empty());
        let goal = assert_paused_goal(
            store.as_ref(),
            goal_store.as_ref(),
            &task_id,
            crate::control_plane::GoalPauseReason::NeedsUserInput,
            crate::control_plane::GoalBlockerKind::NeedsUserInput,
            "ask_user",
        )
        .await;
        assert_eq!(goal.blockers[0].message, "continue?");
    }

    #[tokio::test]
    async fn goal_human_gate_route_mismatch_does_not_pause_or_deliver() {
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let (store, _goal_store, task_id, ctx) =
            create_running_goal(&agent, &route, &principal).await;
        let wrong_route_ctx = ctx.with_originator_route(Some("wrong-route".into()));
        let (inner, calls, _args) = recording_tool("ask_user");
        let (channels, sent) = recording_channels();
        let tool = GoalHumanGateTool::ask_user(
            inner,
            Arc::new(SecurityPolicy::default()),
            channels,
            Arc::new(Config::default()),
        );

        let err = scope_active_goal_work(
            wrong_route_ctx,
            tool.execute(json!({"question": "continue?"})),
        )
        .await
        .unwrap_err();

        assert!(format!("{err:#}").contains("route"));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(sent.lock().unwrap().is_empty());
        assert_eq!(
            store.get(&task_id).await.unwrap().unwrap().status,
            crate::control_plane::TaskStatus::Running
        );
    }

    #[tokio::test]
    async fn goal_ask_user_delivery_failure_leaves_recoverable_durable_blocker() {
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let (store, goal_store, task_id, ctx) =
            create_running_goal(&agent, &route, &principal).await;
        let (inner, calls, _args) = recording_tool("ask_user");
        let sent = Arc::new(Mutex::new(Vec::new()));
        let channel: Arc<dyn Channel> = Arc::new(RecordingChannel {
            sent: Arc::clone(&sent),
            fail: true,
        });
        let channels = Arc::new(parking_lot::RwLock::new(HashMap::from([(
            "matrix".to_string(),
            channel,
        )])));
        let tool = GoalHumanGateTool::ask_user(
            inner,
            Arc::new(SecurityPolicy::default()),
            channels,
            Arc::new(Config::default()),
        );

        let err = scope_active_goal_work(ctx, tool.execute(json!({"question": "continue?"})))
            .await
            .unwrap_err();

        assert!(err.is::<ToolLoopCancelled>(), "unexpected error: {err:#}");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(sent.lock().unwrap().len(), 1);
        let goal = assert_paused_goal(
            store.as_ref(),
            goal_store.as_ref(),
            &task_id,
            crate::control_plane::GoalPauseReason::NeedsUserInput,
            crate::control_plane::GoalBlockerKind::NeedsUserInput,
            "ask_user",
        )
        .await;
        assert_eq!(goal.blockers[0].message, "continue?");
        assert_eq!(
            goal.blockers[0].payload.as_ref().unwrap()["delivery"]["notification"],
            "best_effort"
        );
    }

    #[tokio::test]
    async fn ordinary_goal_admission_context_delegates_without_pausing_goal() {
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let (store, goal_store, task_id, ctx) =
            create_running_goal(&agent, &route, &principal).await;
        let (inner, calls, _args) = recording_tool("ask_user");
        let (channels, sent) = recording_channels();
        let tool = GoalHumanGateTool::ask_user(
            inner,
            Arc::new(SecurityPolicy::default()),
            channels,
            Arc::new(Config::default()),
        );

        let result = crate::control_plane::scope_goal_admission_context(
            Some(ctx),
            tool.execute(json!({
                "question": "continue?",
                "channel": "matrix"
            })),
        )
        .await
        .unwrap();

        assert!(result.success);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "ordinary channel turns may carry goal admission facts, \
             but ask_user must not pause an active goal until the turn is marked as goal work"
        );
        assert!(
            sent.lock().unwrap().is_empty(),
            "ordinary path delegates to the wrapped tool instead of goal wrapper delivery"
        );
        let task = store.get(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, crate::control_plane::TaskStatus::Running);
        let goal = goal_store.get_goal_task(&task_id).await.unwrap().unwrap();
        assert_eq!(goal.pause_reason, None);
        assert!(goal.blockers.is_empty());
    }

    #[tokio::test]
    async fn escalation_wrapper_pauses_active_goal_and_forces_nonblocking_delivery() {
        let agent = format!("agent-{}", uuid::Uuid::new_v4());
        let route = format!("route-{}", uuid::Uuid::new_v4());
        let principal = format!("principal-{}", uuid::Uuid::new_v4());
        let (store, goal_store, task_id, ctx) =
            create_running_goal(&agent, &route, &principal).await;
        let (inner, calls, args) = recording_tool("escalate_to_human");
        let (channels, _sent) = recording_channels();
        let tool = GoalHumanGateTool::escalate_to_human(
            inner,
            Arc::new(SecurityPolicy::default()),
            channels,
            Arc::new(Config::default()),
        );

        let err = scope_active_goal_work(
            ctx,
            tool.execute(json!({
                "summary": "operator approval requires sk_test_1234567890abcdefghijklmnop",
                "context": "password=longsecretvalue",
                "urgency": "high",
                "wait_for_response": true
            })),
        )
        .await
        .unwrap_err();

        assert!(err.is::<ToolLoopCancelled>(), "unexpected error: {err:#}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        {
            let args = args.lock().unwrap();
            assert_eq!(args.len(), 1);
            assert!(!args[0].to_string().contains("sk_test_"));
            assert!(!args[0].to_string().contains("longsecretvalue"));
            assert!(args[0].to_string().contains("REDACTED"));
            assert_eq!(args[0]["urgency"], "high");
            assert_eq!(args[0]["wait_for_response"], false);
        }

        let goal = assert_paused_goal(
            store.as_ref(),
            goal_store.as_ref(),
            &task_id,
            crate::control_plane::GoalPauseReason::HumanEscalation,
            crate::control_plane::GoalBlockerKind::HumanEscalation,
            "escalate_to_human",
        )
        .await;
        let payload = goal.blockers[0].payload.as_ref().unwrap();
        assert!(!payload.to_string().contains("sk_test_"));
        assert!(!payload.to_string().contains("longsecretvalue"));
        assert!(payload.to_string().contains("REDACTED"));
        assert_eq!(payload["urgency"], "high");
        assert_eq!(payload["wait_for_response_requested"], true);
    }
}
