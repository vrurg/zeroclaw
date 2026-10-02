//! Engine-bound switch regression; no live service or model is contacted.

use async_trait::async_trait;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use tokio::{
    sync::Notify,
    time::{Duration, timeout},
};
use zeroclaw_api::{
    attribution::{Attributable, ModelProviderKind, ProviderKind, Role, ToolKind},
    model_provider::{ChatMessage, TokenUsage},
};
use zeroclaw_commands::goal::{GoalBudgetLimits, GoalBudgetSelection, GoalCommand};
use zeroclaw_config::{
    cost::CostTracker,
    schema::{AliasedAgentConfig, Config, RiskProfileConfig},
};
use zeroclaw_goal::{
    GoalExecutionNotice, GoalExecutionScope, GoalExecutionSupervisor, GoalHostSettings,
    GoalIngressContext, GoalIngressPrincipal, GoalOperationScope, GoalParentTurn,
    GoalParentTurnCheckpoint, GoalParentTurnResult, GoalResponse, GoalRuntime, GoalSessionBinding,
    GoalSessionDriver, GoalSessionExecutionLease, GoalSessionKey, GoalSessionLease,
    GoalTaskRegistry, GoalVerifierResult, GoalVerifierTurn,
};
use zeroclaw_runtime::{
    agent::{
        Agent,
        agent::{IsolatedTranscriptSource, ProviderSwitchConfig},
        dispatcher::{NativeToolDispatcher, ToolDispatcher, XmlToolDispatcher},
    },
    control_plane::{SqliteTaskStore, TaskStatus},
    goal_accounting::{ModelProviderGoalUsagePricer, RuntimeGoalOperationAccountingFactory},
    tools::{Tool, ToolResult},
};

#[path = "support/goal_transition_fault.rs"]
mod goal_transition_fault;
use goal_transition_fault::{FinishFaultRegistry, GoalModelAdmissionGate, GoalRegistryObservation};

struct SwitchProvider {
    calls: Arc<AtomicUsize>,
    switch: bool,
    xml: bool,
    requests: Arc<Mutex<Vec<Vec<ChatMessage>>>>,
    interrupted_call: Option<Arc<Notify>>,
}
impl Attributable for SwitchProvider {
    fn role(&self) -> Role {
        Role::Provider(ProviderKind::Model(ModelProviderKind::Custom))
    }
    fn alias(&self) -> &str {
        "switch-fixture"
    }
}
#[async_trait]
impl zeroclaw_providers::ModelProvider for SwitchProvider {
    fn supports_native_tools(&self) -> bool {
        !self.xml
    }
    async fn chat_with_system(
        &self,
        _: Option<&str>,
        _: &str,
        _: &str,
        _: Option<f64>,
    ) -> anyhow::Result<String> {
        anyhow::bail!("fixture requires structured dispatch")
    }
    async fn chat(
        &self,
        request: zeroclaw_providers::ChatRequest<'_>,
        _: &str,
        _: Option<f64>,
    ) -> anyhow::Result<zeroclaw_providers::ChatResponse> {
        let index = self.calls.fetch_add(1, Ordering::SeqCst);
        if index == 1
            && !self.switch
            && let Some(ready) = &self.interrupted_call
        {
            ready.notify_one();
            std::future::pending::<()>().await;
            unreachable!("the fixture interrupts this admitted call");
        }
        assert_eq!(index, 0);
        self.requests
            .lock()
            .unwrap()
            .push(request.messages.to_vec());
        let name = if self.switch { "model_switch" } else { "echo" };
        let arguments = if self.switch {
            serde_json::json!({"action":"set", "model_provider":"custom.target", "model":"target-model"}).to_string()
        } else {
            "{}".into()
        };
        Ok(zeroclaw_providers::ChatResponse {
            text: self.xml.then(|| {
                format!("<tool_call>{{\"name\":\"{name}\",\"arguments\":{arguments}}}</tool_call>")
            }),
            tool_calls: if self.xml {
                Vec::new()
            } else {
                vec![zeroclaw_providers::ToolCall {
                    id: "initial-call".into(),
                    name: name.into(),
                    arguments,
                    extra_content: None,
                }]
            },
            usage: Some(TokenUsage {
                input_tokens: Some(10),
                output_tokens: Some(2),
                ..Default::default()
            }),
            reasoning_content: None,
        })
    }
}

struct CountingTool(Arc<AtomicUsize>);
impl Attributable for CountingTool {
    fn role(&self) -> Role {
        Role::Tool(ToolKind::Plugin)
    }
    fn alias(&self) -> &str {
        "switch-fixture"
    }
}
#[async_trait]
impl Tool for CountingTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "Return the bounded fixture result."
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type":"object"})
    }
    async fn execute(&self, _: serde_json::Value) -> anyhow::Result<ToolResult> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult {
            success: true,
            output: "paired echo".into(),
            error: None,
        })
    }
}

struct SwitchDriver {
    binding: GoalSessionBinding,
    expected_candidate: String,
    recall: bool,
    interruption: Option<Arc<InterruptionProbe>>,
    agent: Mutex<Option<Agent>>,
    isolated: bool,
    observation: Arc<GoalRegistryObservation>,
    backend: Arc<zeroclaw_infra::session_sqlite::SqliteSessionBackend>,
    notices: Arc<Mutex<Vec<GoalExecutionNotice>>>,
    errors: Arc<Mutex<Vec<String>>>,
    published: Arc<Notify>,
}
struct SwitchLease {
    key: GoalSessionKey,
    expected_candidate: String,
    recall: bool,
    interruption: Option<Arc<InterruptionProbe>>,
    agent: Option<Agent>,
    isolated: bool,
    observation: Arc<GoalRegistryObservation>,
    backend: Arc<zeroclaw_infra::session_sqlite::SqliteSessionBackend>,
    notices: Arc<Mutex<Vec<GoalExecutionNotice>>>,
    errors: Arc<Mutex<Vec<String>>>,
    published: Arc<Notify>,
}
#[async_trait]
impl GoalSessionDriver for SwitchDriver {
    fn session_key(&self) -> &GoalSessionKey {
        self.binding.session_key()
    }
    async fn bind(&self, _: &GoalIngressContext) -> anyhow::Result<GoalSessionLease> {
        Ok(GoalSessionLease::new(self.binding.clone(), ()))
    }
    async fn acquire_execution(
        &self,
        _: &GoalIngressContext,
        _: &GoalExecutionScope,
        _: &GoalParentTurnCheckpoint,
    ) -> anyhow::Result<Box<dyn GoalSessionExecutionLease>> {
        Ok(Box::new(SwitchLease {
            key: self.binding.session_key().clone(),
            expected_candidate: self.expected_candidate.clone(),
            recall: self.recall,
            interruption: self.interruption.clone(),
            // A post-drain notice lease must not create another Agent or turn.
            agent: self.agent.lock().unwrap().take(),
            isolated: self.isolated,
            observation: self.observation.clone(),
            backend: self.backend.clone(),
            notices: self.notices.clone(),
            errors: self.errors.clone(),
            published: self.published.clone(),
        }))
    }
}
#[async_trait]
impl GoalSessionExecutionLease for SwitchLease {
    fn session_key(&self) -> &GoalSessionKey {
        &self.key
    }
    fn canonical_history(&self) -> anyhow::Result<Vec<ChatMessage>> {
        use zeroclaw_infra::session_backend::SessionBackend;
        Ok(self.backend.load(&self.key.durable_id()))
    }
    async fn append_resume_response(&mut self, response: String) -> anyhow::Result<()> {
        use zeroclaw_infra::session_backend::SessionBackend;
        self.backend
            .append(&self.key.durable_id(), &ChatMessage::user(response))
            .map_err(Into::into)
    }
    async fn run_parent_turn(
        &mut self,
        scope: &GoalOperationScope,
        turn: GoalParentTurn,
    ) -> anyhow::Result<GoalParentTurnResult> {
        let agent = self
            .agent
            .as_mut()
            .expect("only the original execution lease may run a parent");
        let result = {
            let parent = async {
                if self.isolated {
                    let input = zeroclaw_goal::control::goal_parent_turn_input(&turn);
                    agent
                        .run_isolated_turn(
                            IsolatedTranscriptSource::Canonical {
                                prefix: turn.working_history,
                            },
                            &input,
                            "finish",
                        )
                        .await
                        .and_then(|outcome| {
                            if self.recall {
                                assert!(
                                    !outcome
                                        .working_history
                                        .iter()
                                        .any(|row| row.content.contains(RECALL_TEXT))
                                );
                            }
                            GoalParentTurnResult::from_completed_agent_turn(
                                outcome.response,
                                outcome.working_history,
                            )
                        })
                } else {
                    // The ordinary streamed turn under the same real engine scopes is the
                    // control for tool pairing and per-attempt accounting mechanics.
                    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
                    let (outcome, ()) =
                        tokio::join!(agent.turn_streamed("finish", tx, None), async {
                            while rx.recv().await.is_some() {}
                        });
                    outcome.and_then(|(response, _)| {
                        GoalParentTurnResult::from_completed_agent_turn(
                            response,
                            NativeToolDispatcher.to_provider_messages(agent.history()),
                        )
                    })
                }
            };
            tokio::pin!(parent);
            if let Some(probe) = &self.interruption {
                tokio::select! {
                    result = &mut parent => result,
                    _ = probe.ready.notified() => {
                        let extension = probe.store.get_goal_task(scope.execution().task_id()).await.unwrap().unwrap();
                        probe.pending.lock().unwrap().push((extension.pending_call_id.is_some(), extension.pending_tool_batch_id.is_some()));
                        // Drop the admitted parent future, not a lifecycle command.
                        // This models forced host interruption without assuming that
                        // the unfinished provider operation became safe to resume.
                        Err(anyhow::Error::msg("fixture forced parent interruption"))
                    }
                }
            } else {
                parent.await
            }
        };
        self.observation
            .parent_returned
            .store(true, Ordering::SeqCst);
        if let Err(error) = &result {
            self.errors.lock().unwrap().push(format!("{error:#}"));
        }
        result
    }
    async fn run_verifier(
        &mut self,
        _: &GoalOperationScope,
        turn: GoalVerifierTurn,
    ) -> anyhow::Result<GoalVerifierResult> {
        use zeroclaw_goal::accounting::{
            GOAL_OPERATION_ACCOUNTING, GoalOperationRequest, GoalOperationSettlement,
            GoalUsageEvent,
        };
        assert_eq!(turn.candidate, self.expected_candidate);
        // The verdict is controlled, not evidence of live-model judgment.
        // Its synthetic usage still settles through the real runtime ledger.
        let accountant = GOAL_OPERATION_ACCOUNTING.with(Clone::clone).unwrap();
        accountant
            .admit(GoalOperationRequest::new(
                "fixture.verifier",
                "verifier-model",
            ))
            .await?;
        accountant
            .settle(GoalOperationSettlement {
                accounting_state: zeroclaw_goal::GoalAccountingState::Complete,
                events: vec![GoalUsageEvent {
                    provider_ref: "fixture.verifier".into(),
                    model: "verifier-model".into(),
                    usage: TokenUsage {
                        input_tokens: Some(10),
                        output_tokens: Some(2),
                        ..Default::default()
                    },
                    priceable: true,
                }],
            })
            .await?;
        Ok(GoalVerifierResult {
            response: r#"{"decision":"complete","reason":"bounded work complete","blockers":[]}"#
                .into(),
            attribution: None,
        })
    }
    async fn append_parent_candidate(&mut self, candidate: String) -> anyhow::Result<()> {
        use zeroclaw_infra::session_backend::SessionBackend;
        self.backend
            .append(&self.key.durable_id(), &ChatMessage::assistant(candidate))
            .map_err(Into::into)
    }
    async fn publish_goal_core_error(&mut self, error: &anyhow::Error) -> anyhow::Result<()> {
        self.errors.lock().unwrap().push(format!("{error:#}"));
        Ok(())
    }
    async fn publish_goal_notice(&mut self, notice: GoalExecutionNotice) -> anyhow::Result<()> {
        self.notices.lock().unwrap().push(notice);
        self.published.notify_one();
        Ok(())
    }
}

async fn fixture_agent(
    config: Arc<Config>,
    calls: Arc<AtomicUsize>,
    tool_calls: Arc<AtomicUsize>,
    switch: bool,
    requests: Arc<Mutex<Vec<Vec<ChatMessage>>>>,
    recall: bool,
    interrupted_call: Option<Arc<Notify>>,
) -> Agent {
    use zeroclaw_runtime::tools::{
        AllToolsResult,
        scoped::{ScopedAssembly, ScopedToolRegistry},
    };
    let security = Arc::new(zeroclaw_config::policy::SecurityPolicy::default());
    let risk = &config.risk_profiles["switch-fixture"];
    let registry = ScopedToolRegistry::assemble(ScopedAssembly {
        config: config.as_ref(),
        agent_alias: "main",
        security: &security,
        built: AllToolsResult::from_prebuilt_tools(vec![
            Box::new(zeroclaw_runtime::tools::ModelSwitchTool::new(
                security.clone(),
                config.clone(),
            )),
            Box::new(CountingTool(tool_calls)),
        ]),
        skills: &[],
        runtime: zeroclaw_runtime::platform::create_runtime(&config.runtime)
            .unwrap()
            .into(),
        caller_allowed: None,
        connect_mcp: false,
        connect_peripherals: false,
        exclude_memory: false,
        acp_delivery: false,
        list_deferred_mcp_specs: false,
        emit_assembly_logs: false,
        mcp_registry: None,
    })
    .await
    .registry;
    assert!(registry.iter().any(|tool| tool.name() == "model_switch"));
    assert!(registry.iter().any(|tool| tool.name() == "echo"));
    let xml = config.agents["main"].resolved.tool_dispatcher == "xml";
    let memory: Arc<dyn zeroclaw_memory::Memory> = if recall {
        use zeroclaw_memory::Memory;
        let memory =
            zeroclaw_memory::SqliteMemory::new("switch-fixture", &config.data_dir).unwrap();
        memory
            .store(
                "fixture-guidance",
                RECALL_TEXT,
                zeroclaw_memory::MemoryCategory::Core,
                None,
            )
            .await
            .unwrap();
        Arc::new(memory)
    } else {
        Arc::new(zeroclaw_memory::NoneMemory::new("switch-fixture"))
    };
    let mut agent = Agent::builder()
        .model_provider(Box::new(SwitchProvider {
            calls,
            switch,
            xml,
            requests,
            interrupted_call,
        }))
        .model_provider_name("custom.initial".into())
        .model_name("initial-model".into())
        .config(config.agents["main"].clone())
        .tools(registry)
        .memory(memory)
        .auto_save(false)
        .observer(Arc::new(zeroclaw_runtime::observability::NoopObserver))
        .tool_dispatcher(if xml {
            Box::new(XmlToolDispatcher)
        } else {
            Box::new(NativeToolDispatcher)
        })
        .workspace_dir(config.data_dir.clone())
        .security(security)
        .approval_manager(Some(Arc::new(
            zeroclaw_runtime::approval::ApprovalManager::for_non_interactive(risk),
        )))
        .provider_switch_config(ProviderSwitchConfig {
            config: Some(config),
            live_config: None,
            live: None,
        })
        .build()
        .unwrap();
    if recall {
        agent.seed_history(&recall_prefix());
    }
    agent
}

const RECALL_TEXT: &str = "Finish the bounded fixture using recalled guidance.";

struct InterruptionProbe {
    store: Arc<SqliteTaskStore>,
    ready: Arc<Notify>,
    pending: Mutex<Vec<(bool, bool)>>,
}

fn recall_prefix() -> Vec<ChatMessage> {
    vec![
        ChatMessage::user(format!(
            "{}\nUser-authored fixture text.\n{}",
            zeroclaw_memory::MEMORY_CONTEXT_OPEN,
            zeroclaw_memory::MEMORY_CONTEXT_CLOSE
        )),
        ChatMessage::assistant("acknowledged"),
    ]
}

#[tokio::test]
async fn engine_model_switch_retains_one_marker_and_exact_attempt_accounting() {
    run_engine_switch_fixture(SwitchFixtureCase::Success).await;
}

#[tokio::test]
async fn engine_model_switch_pause_now_denies_retry_like_an_ordinary_tool_round_fence() {
    run_engine_switch_fixture(SwitchFixtureCase::FenceParity).await;
}

#[tokio::test]
async fn engine_operator_pause_remains_resumable_after_a_paired_tool_round() {
    run_engine_switch_fixture(SwitchFixtureCase::OperatorPause).await;
}

#[tokio::test]
async fn engine_model_switch_shares_the_configured_execution_tree_budget() {
    run_engine_switch_fixture(SwitchFixtureCase::Budget).await;
}

#[tokio::test]
async fn engine_model_switch_recall_native_preserves_pairing_and_usage() {
    run_engine_switch_fixture(SwitchFixtureCase::RecallNative).await;
}

#[tokio::test]
async fn engine_model_switch_recall_xml_preserves_pairing_and_usage() {
    run_engine_switch_fixture(SwitchFixtureCase::RecallXml).await;
}

#[tokio::test]
async fn engine_model_switch_forced_interruption_matches_an_ordinary_tool_round() {
    run_engine_switch_fixture(SwitchFixtureCase::ForcedInterruption).await;
}

#[derive(Clone, Copy)]
enum SwitchFixtureCase {
    Success,
    FenceParity,
    OperatorPause,
    Budget,
    RecallNative,
    RecallXml,
    ForcedInterruption,
}

async fn run_engine_switch_fixture(case: SwitchFixtureCase) {
    let fence = matches!(
        case,
        SwitchFixtureCase::FenceParity | SwitchFixtureCase::OperatorPause
    );
    let budget = matches!(case, SwitchFixtureCase::Budget);
    let recall = matches!(
        case,
        SwitchFixtureCase::RecallNative | SwitchFixtureCase::RecallXml
    );
    let xml = matches!(case, SwitchFixtureCase::RecallXml);
    let forced = matches!(case, SwitchFixtureCase::ForcedInterruption);
    let interrupted_call = forced.then(|| Arc::new(Notify::new()));
    use wiremock::{Mock, MockServer, Respond, ResponseTemplate, matchers::method};
    struct TargetReplies {
        calls: AtomicUsize,
        budget: bool,
        xml: bool,
        interrupted_call: Option<Arc<Notify>>,
    }
    impl Respond for TargetReplies {
        fn respond(&self, request: &wiremock::Request) -> ResponseTemplate {
            let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
            assert_eq!(body["model"], "target-model");
            assert_ne!(body["stream"], true);
            if let Some(ready) = &self.interrupted_call {
                ready.notify_one();
                return ResponseTemplate::new(200).set_delay(Duration::from_secs(60));
            }
            let index = self.calls.fetch_add(1, Ordering::SeqCst);
            // The ordinary loop reserves the switch-detection iteration before
            // retrying. Three slots leave only the root's final completion on
            // the target route, not a fresh tool budget.
            if self.budget {
                assert!(
                    body.get("tools")
                        .is_none_or(|tools| tools.as_array().is_some_and(Vec::is_empty))
                );
            }
            let tool_round = !self.budget && index.is_multiple_of(2);
            let message = if tool_round && self.xml {
                serde_json::json!({"role":"assistant","content":"<tool_call>{\"name\":\"echo\",\"arguments\":{}}</tool_call>"})
            } else if tool_round {
                serde_json::json!({"role":"assistant","content":null,"tool_calls":[{"id":"target-echo","type":"function","function":{"name":"echo","arguments":"{}"}}]})
            } else {
                serde_json::json!({"role":"assistant","content":"switched candidate"})
            };
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"choices":[{"message":message,"finish_reason":if tool_round && !self.xml {"tool_calls"} else {"stop"}}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}))
        }
    }
    let tmp = tempfile::tempdir().unwrap();
    let store = Arc::new(SqliteTaskStore::new(tmp.path()).unwrap());
    let tracker = Arc::new(
        CostTracker::new(
            zeroclaw_config::schema::CostConfig {
                enabled: true,
                ..Default::default()
            },
            tmp.path(),
        )
        .unwrap(),
    );
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(TargetReplies {
            calls: AtomicUsize::new(0),
            budget,
            xml,
            interrupted_call: interrupted_call.clone(),
        })
        .mount(&server)
        .await;
    let mut config = Config {
        data_dir: tmp.path().to_path_buf(),
        ..Default::default()
    };
    config.reliability.provider_retries = 0;
    let target = config.providers.models.ensure("custom", "target").unwrap();
    target.uri = Some(server.uri());
    target.api_key = Some("synthetic-key".into());
    // The custom family owns the explicit native-tools opt-out. The generic
    // compatibility kind instead selects the default OpenAI family behavior.
    target.kind = (!xml).then(|| "openai-compatible".into());
    target.wire_api = Some(zeroclaw_config::schema::WireApi::ChatCompletions);
    target.model = Some("target-model".into());
    target.live_pricing = false;
    target.thinking_passthrough = true;
    target.native_tools = Some(!xml);
    config.risk_profiles.insert(
        "switch-fixture".into(),
        RiskProfileConfig {
            auto_approve: vec!["model_switch".into(), "echo".into()],
            ..Default::default()
        },
    );
    let mut agent_config = AliasedAgentConfig {
        risk_profile: "switch-fixture".into(),
        ..Default::default()
    };
    agent_config.resolved.tool_dispatcher = if xml { "xml" } else { "native" }.into();
    if budget {
        agent_config.resolved.max_execution_tree_iterations = Some(3);
    }
    config.agents.insert("main".into(), agent_config);
    let config = Arc::new(config);
    let pricing = Arc::new(std::collections::HashMap::from([
        (
            "custom.initial".into(),
            std::collections::HashMap::from([
                ("initial-model.input".into(), 1.0),
                ("initial-model.output".into(), 1.0),
            ]),
        ),
        (
            "custom.target".into(),
            std::collections::HashMap::from([
                ("target-model.input".into(), 1.0),
                ("target-model.output".into(), 1.0),
            ]),
        ),
        (
            "fixture.verifier".into(),
            std::collections::HashMap::from([
                ("verifier-model.input".into(), 1.0),
                ("verifier-model.output".into(), 1.0),
            ]),
        ),
    ]));
    let variants = match case {
        SwitchFixtureCase::Success
        | SwitchFixtureCase::Budget
        | SwitchFixtureCase::RecallNative
        | SwitchFixtureCase::RecallXml => vec![(false, true), (true, true)],
        SwitchFixtureCase::FenceParity => {
            vec![(false, false), (true, false), (false, true), (true, true)]
        }
        SwitchFixtureCase::OperatorPause => vec![(false, false), (true, false)],
        SwitchFixtureCase::ForcedInterruption => {
            vec![(false, false), (true, false), (false, true), (true, true)]
        }
    };
    let mut no_switch_fences = [None, None];
    let mut no_switch_interruptions = [None, None];
    for (isolated, switch) in variants {
        let gate = fence.then(|| Arc::new(GoalModelAdmissionGate::default()));
        let observation = Arc::new(GoalRegistryObservation {
            before_retry: gate.clone(),
            ..Default::default()
        });
        let registry = Arc::new(FinishFaultRegistry::observing(
            store.clone(),
            observation.clone(),
        ));
        let runtime = GoalRuntime::new(registry.clone());
        let factory = Arc::new(
            RuntimeGoalOperationAccountingFactory::new(
                registry.clone(),
                tracker.clone(),
                "main",
                Arc::new(ModelProviderGoalUsagePricer::new(pricing.clone())),
            )
            .unwrap(),
        );
        let supervisor = GoalExecutionSupervisor::new(Arc::new(runtime.execution_engine(factory)));
        let settings = GoalHostSettings::new(
            true,
            GoalBudgetLimits {
                token_limit: Some(100),
                cost_limit_usd: Some(1.0),
            },
            42,
            "switch-fixture-boot",
        )
        .unwrap();
        let key = GoalSessionKey::zero_code(format!("switch-fixture-{isolated}-{switch}")).unwrap();
        let ingress = GoalIngressContext::trusted(
            key.clone(),
            "main",
            "zerocode:local",
            GoalIngressPrincipal::ZeroCode {
                tui_id: "fixture-connection".into(),
            },
        )
        .unwrap();
        let initial_calls = Arc::new(AtomicUsize::new(0));
        let tool_calls = Arc::new(AtomicUsize::new(0));
        let backend = Arc::new(
            zeroclaw_infra::session_sqlite::SqliteSessionBackend::new(tmp.path()).unwrap(),
        );
        if recall {
            use zeroclaw_infra::session_backend::SessionBackend;
            for row in recall_prefix() {
                backend.append(&key.durable_id(), &row).unwrap();
            }
        }
        let initial_requests = Arc::new(Mutex::new(Vec::new()));
        let interruption = interrupted_call.as_ref().map(|ready| {
            Arc::new(InterruptionProbe {
                store: store.clone(),
                ready: ready.clone(),
                pending: Mutex::new(Vec::new()),
            })
        });
        let notices = Arc::new(Mutex::new(Vec::new()));
        let errors = Arc::new(Mutex::new(Vec::new()));
        let published = Arc::new(Notify::new());
        let expected_candidate = if budget {
            format!(
                "switched candidate\n\n{}",
                zeroclaw_runtime::i18n::get_required_cli_string(
                    "turn-execution-tree-budget-reached"
                )
            )
        } else {
            "switched candidate".into()
        };
        let driver = Arc::new(SwitchDriver {
            binding: GoalSessionBinding::new(key.clone()),
            expected_candidate: expected_candidate.clone(),
            recall,
            interruption: interruption.clone(),
            agent: Mutex::new(Some(
                fixture_agent(
                    config.clone(),
                    initial_calls.clone(),
                    tool_calls.clone(),
                    switch,
                    initial_requests.clone(),
                    recall,
                    interrupted_call.clone(),
                )
                .await,
            )),
            isolated,
            observation: observation.clone(),
            backend: backend.clone(),
            notices: notices.clone(),
            errors: errors.clone(),
            published: published.clone(),
        });
        let mut started = supervisor
            .submit(
                settings.clone(),
                ingress.clone(),
                driver.clone(),
                GoalCommand::Start {
                    budget: GoalBudgetSelection::Defaults,
                    objective: "finish the bounded fixture".into(),
                },
            )
            .await
            .unwrap();
        let projection = match started.response() {
            GoalResponse::Started(projection) => projection,
            response => panic!("unexpected response: {response:?}"),
        };
        let scope = GoalExecutionScope::new(
            projection.task_id.clone(),
            key.durable_id(),
            projection.execution_epoch,
        )
        .unwrap();
        supervisor
            .launch_after_response(settings.clone(), &mut started)
            .await
            .unwrap();
        let (completion, retry_reached) = if let Some(gate) = &gate {
            let reached = timeout(Duration::from_secs(5), async {
                tokio::select! {
                    _ = gate.observed.notified() => true,
                    _ = published.notified() => false,
                }
            })
            .await
            .expect("worker must reach the retry gate or publish its failure");
            if reached {
                let mut pause = supervisor
                    .submit(
                        settings.clone(),
                        ingress,
                        driver,
                        GoalCommand::Pause { immediate: true },
                    )
                    .await
                    .unwrap();
                assert!(matches!(
                    pause.response(),
                    GoalResponse::TransitionPending(zeroclaw_goal::GoalPendingAction::Pause)
                ));
                assert_eq!(
                    store
                        .current_goal_for_session(&key.durable_id())
                        .await
                        .unwrap()
                        .unwrap()
                        .status,
                    TaskStatus::Paused
                );
                gate.released.notify_one();
                let (completion, transition) = timeout(Duration::from_secs(5), async {
                    tokio::join!(biased; supervisor.drain(&scope), supervisor.launch_after_response(settings, &mut pause))
                }).await.expect("a paired, fenced parent must drain without deadlock");
                let transition = transition.unwrap();
                if matches!(case, SwitchFixtureCase::OperatorPause) {
                    assert!(matches!(transition, Some(GoalResponse::Paused(_))));
                }
                (completion, true)
            } else {
                (supervisor.drain(&scope).await, false)
            }
        } else {
            timeout(Duration::from_secs(5), published.notified())
                .await
                .expect("engine must publish a terminal notice");
            (supervisor.drain(&scope).await, true)
        };
        let task = store
            .current_goal_for_session(&key.durable_id())
            .await
            .unwrap()
            .unwrap();
        let extension = store.get_goal_task(&task.id).await.unwrap().unwrap();
        assert_eq!(
            initial_calls.load(Ordering::SeqCst),
            if forced && !switch { 2 } else { 1 }
        );
        assert_eq!(registry.fault_counts(), (0, 0, 0));
        if forced {
            assert_eq!(
                *interruption.as_ref().unwrap().pending.lock().unwrap(),
                vec![(true, true)],
                "the retry must be interrupted after admission, with the original marker retained; isolated={isolated}, switch={switch}"
            );
        }
        let calls = observation.calls.lock().unwrap().clone();
        let admissions: Vec<_> = calls
            .iter()
            .filter(|call| call.kind == "tool_admit")
            .collect();
        let settlements: Vec<_> = calls
            .iter()
            .filter(|call| call.kind == "tool_settle")
            .collect();
        assert_eq!(admissions.len(), 1);
        assert_eq!(settlements.len(), 1);
        assert_eq!(admissions[0].batch_id, settlements[0].batch_id);
        assert_eq!(admissions[0].batch_id, admissions[0].pending_tool_batch_id);
        assert_eq!(admissions[0].batch_id, settlements[0].pending_tool_batch_id);
        assert!(!admissions[0].parent_returned);
        assert!(settlements[0].parent_returned);
        assert!(admissions[0].pending_call_id.is_none());
        assert_eq!(settlements[0].pending_call_id.is_some(), forced);
        assert!(extension.pending_call_id.is_none());
        assert!(extension.pending_tool_batch_id.is_none());
        let ledger = std::fs::read_to_string(tmp.path().join("state/costs.jsonl")).unwrap();
        let records: Vec<zeroclaw_config::cost::CostRecord> = ledger
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .filter(|row: &zeroclaw_config::cost::CostRecord| {
                row.task_id.as_deref() == Some(task.id.as_str())
            })
            .collect();
        let initial_attempts: Vec<_> = records
            .iter()
            .filter(|row| row.provider_ref.as_deref() == Some("custom.initial"))
            .collect();
        assert_eq!(initial_attempts.len(), 1);
        assert_eq!(initial_attempts[0].usage.model, "initial-model");
        assert_eq!(initial_attempts[0].usage.input_tokens, 10);
        assert_eq!(initial_attempts[0].usage.output_tokens, 2);
        if forced {
            assert_eq!(records.len(), 1);
            assert_eq!(tracker.get_usage_totals_for_task(&task.id).unwrap().0, 12);
            assert_eq!(tool_calls.load(Ordering::SeqCst), usize::from(!switch));
            assert!(completion.is_err());
            assert_eq!(task.status, TaskStatus::Failed);
            let snapshot = (
                task.status,
                extension.accounting_state,
                registry
                    .terminal_reason_for_session_goal(&task.id, &key.durable_id())
                    .await
                    .unwrap(),
                notices.lock().unwrap().clone(),
                extension.pending_call_id.is_some(),
                extension.pending_tool_batch_id.is_some(),
            );
            assert_eq!(
                snapshot.1,
                zeroclaw_goal::GoalAccountingState::OutcomeUnknown
            );
            assert_eq!(snapshot.2.as_deref(), Some("parent_operation_failed"));
            assert_eq!(
                snapshot.3,
                vec![GoalExecutionNotice::Failed(
                    zeroclaw_goal::GoalFailureReason::AgentOperationFailed
                )]
            );
            assert!(
                errors
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|error| error.contains("fixture forced parent interruption"))
            );
            if switch {
                assert_eq!(
                    no_switch_interruptions[usize::from(isolated)].as_ref(),
                    Some(&snapshot)
                );
            } else {
                no_switch_interruptions[usize::from(isolated)] = Some(snapshot);
            }
            continue;
        }
        if recall {
            let initial = initial_requests.lock().unwrap();
            assert_eq!(initial.len(), 1);
            assert_eq!(
                initial[0]
                    .iter()
                    .filter(|row| row.content.contains(RECALL_TEXT))
                    .count(),
                1
            );
            assert!(
                initial[0]
                    .iter()
                    .rev()
                    .find(|row| row.role == "user")
                    .unwrap()
                    .content
                    .starts_with(zeroclaw_memory::MEMORY_CONTEXT_OPEN)
            );
            assert!(
                initial[0]
                    .iter()
                    .any(|row| row.role == "user" && row.content == recall_prefix()[0].content)
            );
        }
        if fence {
            assert!(
                retry_reached,
                "switch parent must reach the same admission fence; isolated={isolated}; errors={:?}",
                errors.lock().unwrap()
            );
            assert_eq!(tool_calls.load(Ordering::SeqCst), usize::from(!switch));
            assert!(server.received_requests().await.unwrap().is_empty());
            let model_admissions: Vec<_> = calls
                .iter()
                .filter(|call| call.kind == "model_admit")
                .collect();
            assert_eq!(model_admissions.len(), 2);
            assert!(model_admissions[1].pending_call_id.is_none());
            assert_eq!(
                model_admissions[1].pending_tool_batch_id,
                admissions[0].batch_id
            );
            assert_eq!(tracker.get_usage_totals_for_task(&task.id).unwrap().0, 12);
            assert_eq!(records.len(), 1);
            let snapshot = (
                task.status,
                extension.pause_reason,
                extension.accounting_state,
                registry
                    .terminal_reason_for_session_goal(&task.id, &key.durable_id())
                    .await
                    .unwrap(),
                notices.lock().unwrap().clone(),
                completion.as_ref().ok().copied(),
            );
            if matches!(case, SwitchFixtureCase::OperatorPause) {
                // Preserve the operator-pause contract independently of the
                // switch-parity baseline. Equality cannot waive this failure.
                assert_eq!(snapshot.0, TaskStatus::Paused, "{snapshot:?}");
                assert_eq!(task.execution_epoch, scope.execution_epoch() + 1);
                assert_eq!(
                    snapshot.1,
                    Some(zeroclaw_goal::GoalPauseReason::OperatorPaused)
                );
                assert_eq!(snapshot.2, zeroclaw_goal::GoalAccountingState::Complete);
                assert!(snapshot.3.is_none(), "{snapshot:?}");
                assert!(extension.pending_call_epoch.is_none());
                assert!(extension.pending_tool_epoch.is_none());
                let visible_errors = errors.lock().unwrap();
                assert_eq!(visible_errors.len(), 1, "{visible_errors:?}");
                assert!(
                    visible_errors[0].contains(
                        &zeroclaw_goal::accounting::GoalOperationFenced::ADMISSION_FENCE_LOST
                            .to_string()
                    )
                );
                assert_eq!(
                    snapshot.5,
                    Some(zeroclaw_goal::GoalExecutionOutcome::Fenced)
                );
                assert!(snapshot.4.is_empty(), "{snapshot:?}");
            } else if switch {
                assert_eq!(
                    no_switch_fences[usize::from(isolated)].as_ref(),
                    Some(&snapshot),
                    "a switch must not change the corresponding no-switch fence"
                );
            } else {
                no_switch_fences[usize::from(isolated)] = Some(snapshot);
            }
            continue;
        }
        // Keep the baseline's original error observable before the expected
        // RED; later assertions exercise the full successful contract.
        assert_eq!(
            task.status,
            TaskStatus::Completed,
            "isolated={isolated}; errors={:?}; calls={calls:?}; completion={completion:?}",
            errors.lock().unwrap()
        );
        assert!(completion.is_ok());
        assert!(errors.lock().unwrap().is_empty());
        assert_eq!(
            *notices.lock().unwrap(),
            vec![GoalExecutionNotice::Completed]
        );
        assert_eq!(tool_calls.load(Ordering::SeqCst), usize::from(!budget));
        if recall {
            let requests = server.received_requests().await.unwrap();
            let start = if isolated { 2 } else { 0 };
            assert_eq!(requests.len(), start + 2);
            let mut injection_row = None;
            for (iteration, request) in requests[start..].iter().enumerate() {
                let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
                let rows = body["messages"].as_array().unwrap();
                assert_eq!(rows.iter().filter(|row| row["role"] == "system").count(), 1);
                assert_eq!(body.get("tools").is_some(), !xml);
                // The ordinary XML provider coalesces adjacent user rows.
                // Preserve the authored bytes, not an artificial wire boundary.
                assert!(rows.iter().any(|row| {
                    row["role"] == "user"
                        && row["content"]
                            .as_str()
                            .is_some_and(|text| text.contains(&recall_prefix()[0].content))
                }));
                let count = rows
                    .iter()
                    .filter(|row| {
                        row["content"]
                            .as_str()
                            .unwrap_or_default()
                            .contains(RECALL_TEXT)
                    })
                    .count();
                // RG46 characterizes the ordinary XML owner's stacked recall.
                // Goal must not inherit that defect or change ordinary behavior.
                assert_eq!(count, if xml && !isolated { 2 } else { 1 });
                let last_user = rows.iter().rposition(|row| row["role"] == "user").unwrap();
                if iteration == 0 {
                    assert!(
                        rows[last_user]["content"]
                            .as_str()
                            .unwrap()
                            .starts_with(zeroclaw_memory::MEMORY_CONTEXT_OPEN)
                    );
                    injection_row = Some((last_user, rows[last_user]["content"].clone()));
                } else {
                    let (index, content) = injection_row.as_ref().unwrap();
                    assert_eq!(&rows[*index]["content"], content);
                    assert_eq!(
                        rows[last_user]["content"]
                            .as_str()
                            .unwrap()
                            .starts_with(zeroclaw_memory::MEMORY_CONTEXT_OPEN),
                        !xml
                    );
                }
            }
            use zeroclaw_infra::session_backend::SessionBackend;
            let durable = backend.load(&key.durable_id());
            assert!(
                durable
                    .iter()
                    .any(|row| row.role == "user" && row.content == recall_prefix()[0].content)
            );
            assert!(!durable.iter().any(|row| row.content.contains(RECALL_TEXT)));
        }
        let model_admissions: Vec<_> = calls
            .iter()
            .filter(|call| call.kind == "model_admit")
            .collect();
        let parent_calls = if budget { 2 } else { 3 };
        assert_eq!(model_admissions.len(), parent_calls + 1);
        for admission in &model_admissions[1..parent_calls] {
            assert!(admission.pending_call_id.is_none());
            assert_eq!(admission.pending_tool_batch_id, admissions[0].batch_id);
            assert!(!admission.parent_returned);
        }
        assert!(
            model_admissions[parent_calls]
                .pending_tool_batch_id
                .is_none()
        );
        assert_eq!(records.len(), parent_calls + 1);
        let mut expected_routes = vec![
            ("custom.initial", "initial-model"),
            ("custom.target", "target-model"),
        ];
        if !budget {
            expected_routes.push(("custom.target", "target-model"));
        }
        expected_routes.push(("fixture.verifier", "verifier-model"));
        assert_eq!(
            records
                .iter()
                .map(|row| (
                    row.provider_ref.as_deref().unwrap(),
                    row.usage.model.as_str()
                ))
                .collect::<Vec<_>>(),
            expected_routes
        );
        assert_eq!(
            records
                .iter()
                .map(|row| &row.id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            parent_calls + 1
        );
        assert_eq!(
            tracker.get_usage_totals_for_task(&task.id).unwrap().0,
            (parent_calls as u64 + 1) * 12
        );
        use zeroclaw_infra::session_backend::SessionBackend;
        assert_eq!(
            backend
                .load(&key.durable_id())
                .iter()
                .filter(|row| row.role == "assistant" && row.content == expected_candidate)
                .count(),
            1
        );
    }
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        if forced {
            2
        } else if fence {
            0
        } else if budget {
            2
        } else {
            4
        }
    );
}
