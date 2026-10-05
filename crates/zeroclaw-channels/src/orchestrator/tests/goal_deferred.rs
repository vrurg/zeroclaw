use super::*;

#[derive(Default)]
struct GoalIngressBoundaryProvider {
    requests: std::sync::Mutex<Vec<Vec<zeroclaw_api::model_provider::ChatMessage>>>,
    ordinary_release: tokio::sync::Notify,
    goal_release: tokio::sync::Notify,
    resumed_release: tokio::sync::Notify,
}

impl zeroclaw_api::attribution::Attributable for GoalIngressBoundaryProvider {
    fn role(&self) -> zeroclaw_api::attribution::Role {
        zeroclaw_api::attribution::Role::Provider(zeroclaw_api::attribution::ProviderKind::Model(
            zeroclaw_api::attribution::ModelProviderKind::Custom,
        ))
    }

    fn alias(&self) -> &str {
        "goal-ingress-boundary-fixture"
    }
}

#[async_trait::async_trait]
impl ModelProvider for GoalIngressBoundaryProvider {
    fn supports_native_tools(&self) -> bool {
        true
    }

    async fn chat_with_system(
        &self,
        _: Option<&str>,
        message: &str,
        _: &str,
        _: Option<f64>,
    ) -> anyhow::Result<String> {
        if message.starts_with("Decide whether the assistant should send") {
            return Ok("REPLY".into());
        }
        anyhow::bail!("fixture requires the ordinary structured model boundary")
    }

    async fn chat(
        &self,
        request: zeroclaw_api::model_provider::ChatRequest<'_>,
        _: &str,
        _: Option<f64>,
    ) -> anyhow::Result<zeroclaw_api::model_provider::ChatResponse> {
        let call = {
            let mut requests = self.requests.lock().unwrap();
            requests.push(request.messages.to_vec());
            requests.len()
        };
        let text = match call {
            1 => {
                self.ordinary_release.notified().await;
                "ordinary boundary candidate"
            }
            2 => {
                self.goal_release.notified().await;
                "Goal boundary candidate"
            }
            3 => {
                self.resumed_release.notified().await;
                "resumed budget candidate"
            }
            _ => anyhow::bail!("cancelled fixture must not admit a verifier or successor"),
        };
        Ok(zeroclaw_api::model_provider::ChatResponse {
            text: Some(text.into()),
            tool_calls: Vec::new(),
            usage: Some(zeroclaw_api::model_provider::TokenUsage {
                input_tokens: Some(10),
                output_tokens: Some(2),
                ..Default::default()
            }),
            reasoning_content: None,
        })
    }
}

struct HeldGoalLifecycleChannel {
    inner: SendMessageRecordingChannel,
    send_count: std::sync::atomic::AtomicUsize,
    hold_at: usize,
    entered: tokio::sync::Notify,
    release: tokio::sync::Notify,
}

impl zeroclaw_api::attribution::Attributable for HeldGoalLifecycleChannel {
    fn role(&self) -> zeroclaw_api::attribution::Role {
        zeroclaw_api::attribution::Role::Channel(zeroclaw_api::attribution::ChannelKind::Matrix)
    }

    fn alias(&self) -> &str {
        "held-goal-lifecycle-fixture"
    }
}

#[async_trait::async_trait]
impl Channel for HeldGoalLifecycleChannel {
    fn name(&self) -> &str {
        "matrix"
    }

    async fn send(&self, message: &SendMessage) -> anyhow::Result<()> {
        let ordinal = self
            .send_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        if ordinal == self.hold_at {
            // Hold the native send before it is observable, not the queued
            // acknowledgement or a synthetic pre-commit boundary.
            self.entered.notify_one();
            self.release.notified().await;
        }
        self.inner.send(message).await
    }

    async fn listen(&self, _: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn matrix_goal_aborted_direct_launch_notice_preserves_resumable_identity() {
    assert_matrix_goal_aborted_launch_notice(false, false).await;
    assert_matrix_goal_aborted_launch_notice(true, false).await;
}

#[tokio::test]
async fn matrix_goal_aborted_queued_launch_notice_preserves_resumable_identity() {
    assert_matrix_goal_aborted_launch_notice(false, true).await;
    assert_matrix_goal_aborted_launch_notice(true, true).await;
}

async fn assert_matrix_goal_aborted_launch_notice(resume: bool, queued: bool) {
    use zeroclaw_commands::goal::{GoalBudgetSelection, GoalCommand};
    use zeroclaw_goal::{GoalAccountingState, GoalPauseReason, GoalResponse};
    use zeroclaw_runtime::control_plane::{ControlPlaneHandle, TaskStatus};

    let directory = tempfile::tempdir().unwrap();
    let verifier_server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(wiremock::ResponseTemplate::new(500))
        .expect(0)
        .mount(&verifier_server)
        .await;
    let handle = ControlPlaneHandle::open(directory.path()).unwrap();
    let registry = handle.goal_store().unwrap();
    let channel = Arc::new(HeldGoalLifecycleChannel {
        inner: SendMessageRecordingChannel::matrix(),
        send_count: std::sync::atomic::AtomicUsize::new(0),
        hold_at: if queued { 2 } else { 1 },
        entered: tokio::sync::Notify::new(),
        release: tokio::sync::Notify::new(),
    });
    let provider = Arc::new(GoalIngressBoundaryProvider::default());
    let mut config = matrix_control_test_config();
    config.data_dir = directory.path().to_owned();
    config.goal.enabled = true;
    config.goal.verifier.model_provider =
        zeroclaw_config::providers::ModelProviderRef::new("openai.fixture");
    let verifier = config.providers.models.ensure("openai", "fixture").unwrap();
    verifier.api_key = Some("synthetic-key".into());
    verifier.uri = Some(verifier_server.uri());
    verifier.kind = Some("openai-compatible".into());
    verifier.wire_api = Some(zeroclaw_config::schema::WireApi::ChatCompletions);
    verifier.model = Some("fixture-verifier".into());
    verifier.live_pricing = false;
    let mut context = test_runtime_ctx_with_config_agent_and_provider_ref(
        channel.clone(),
        provider.clone(),
        config,
        zeroclaw_config::schema::AliasedAgentConfig::default(),
        "test-provider",
        None,
    );
    Arc::get_mut(&mut context).unwrap().session_store = Some(Arc::new(
        zeroclaw_infra::session_sqlite::SqliteSessionBackend::new(directory.path()).unwrap(),
    ));
    let message = matrix_msg("@fixture:example.test");
    let key = runtime_conversation_history_key(&context, &message);
    foreground::install_goal_test_control_plane(&context.persist_locks, &key, handle.clone());
    // Keep the fixture's real store and ledger co-located rather than sharing
    // a process-global tracker across unrelated temporary test directories.
    let tracker = Arc::new(
        zeroclaw_runtime::cost::CostTracker::new(
            zeroclaw_config::schema::CostConfig::default(),
            directory.path(),
        )
        .unwrap(),
    );
    let accounting =
        Arc::new(
            zeroclaw_runtime::goal_accounting::RuntimeGoalOperationAccountingFactory::new(
                registry.clone(),
                tracker,
                context.agent_alias.to_string(),
                Arc::new(
                    zeroclaw_runtime::goal_accounting::ModelProviderGoalUsagePricer::new(
                        Arc::default(),
                    ),
                ),
            )
            .unwrap(),
        );
    let supervisor = handle
        .goal_execution_restart()
        .new_supervisor(Arc::new(
            zeroclaw_goal::GoalRuntime::new(registry.clone()).execution_engine(accounting),
        ))
        .await;
    let slot = foreground::goal_supervisor_slot(&context.persist_locks, &key);
    *slot.lock().await = Some(supervisor.clone());
    let start = GoalCommand::Start {
        budget: GoalBudgetSelection::Unlimited,
        objective: "finish the bounded lifecycle fixture".into(),
    };
    let predecessor = if resume {
        let initial = goal_execution::submit_matrix_goal(
            context.clone(),
            key.clone(),
            message.clone(),
            start.clone(),
        )
        .await
        .unwrap();
        initial.response_delivery_failed().await.unwrap();
        *slot.lock().await = Some(supervisor.clone());
        Some(
            registry
                .current_goal_for_session(&key)
                .await
                .unwrap()
                .unwrap(),
        )
    } else {
        None
    };
    let ordinary = if queued {
        Some(
            foreground::foreground_lock(&context.persist_locks, &key)
                .lock_owned()
                .await,
        )
    } else {
        None
    };
    let command = goal_execution::submit_matrix_goal(
        context.clone(),
        key.clone(),
        message.clone(),
        if resume {
            GoalCommand::Resume { response: None }
        } else {
            start
        },
    )
    .await
    .unwrap();
    assert_eq!(
        matches!(command.response(), GoalResponse::Queued(_)),
        queued
    );
    let output = channel.clone();
    let delivery_message = message.clone();
    let delivery = zeroclaw_spawn::spawn!(async move {
        goal_execution::deliver_matrix_goal_command(command, Some(output), delivery_message).await;
    });
    if queued {
        tokio::time::timeout(Duration::from_secs(2), async {
            while channel.inner.sent_messages.lock().await.is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        drop(ordinary);
    }
    tokio::time::timeout(Duration::from_secs(2), channel.entered.notified())
        .await
        .unwrap();
    let admitted = registry
        .current_goal_for_session(&key)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(admitted.status, TaskStatus::Running);
    if let Some(predecessor) = predecessor {
        assert_eq!(admitted.id, predecessor.id);
        assert_eq!(admitted.execution_epoch, predecessor.execution_epoch + 1);
    }
    assert!(provider.requests.lock().unwrap().is_empty());
    delivery.abort();
    assert!(delivery.await.unwrap_err().is_cancelled());
    let paused = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let current = registry
                .current_goal_for_session(&key)
                .await
                .unwrap()
                .unwrap();
            if current.status == TaskStatus::Paused {
                break current;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(paused.id, admitted.id);
    assert_eq!(paused.execution_epoch, admitted.execution_epoch + 1);
    let goal = registry.get_goal_task(&paused.id).await.unwrap().unwrap();
    assert_eq!(goal.pause_reason, Some(GoalPauseReason::ExternalDependency));
    assert_eq!(goal.accounting_state, GoalAccountingState::Complete);
    assert!(goal.pending_call_id.is_none());
    assert!(goal.pending_call_epoch.is_none());
    if queued {
        let audit = rusqlite::Connection::open(directory.path().join("control_plane.db")).unwrap();
        let state: String = audit
            .query_row(
                "SELECT state FROM goal_control_intents WHERE session_id = ?1",
                [&key],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            state, "applied",
            "notice cancellation must not rewrite the committed audit"
        );
    }
    assert!(provider.requests.lock().unwrap().is_empty());
    assert_eq!(
        channel.inner.sent_messages.lock().await.len(),
        usize::from(queued)
    );
    assert!(
        foreground::foreground_lock(&context.persist_locks, &key)
            .try_lock()
            .is_ok()
    );
    assert!(
        foreground::goal_command_lock(&context.persist_locks, &key)
            .try_lock()
            .is_ok()
    );

    let recovery = goal_execution::submit_matrix_goal(
        context,
        key.clone(),
        message,
        GoalCommand::Resume { response: None },
    )
    .await
    .unwrap();
    let GoalResponse::Resumed(recovered) = recovery.response() else {
        panic!("explicit authorized recovery must resume the same Goal");
    };
    assert_eq!(recovered.task_id, paused.id);
    assert_eq!(recovered.execution_epoch, paused.execution_epoch + 1);
    // Do not launch model work merely to test recoverable admission.
    recovery.response_delivery_failed().await.unwrap();
    assert!(provider.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn matrix_goal_start_is_queued_through_actual_ingress_during_ordinary_turn() {
    assert_matrix_goal_deferred_ingress(false, GoalIngressEnding::Cancel).await;
}

#[tokio::test]
async fn matrix_goal_resume_is_queued_through_actual_ingress_with_multiline_response() {
    assert_matrix_goal_deferred_ingress(true, GoalIngressEnding::Cancel).await;
}

#[tokio::test]
async fn matrix_goal_pause_is_queued_through_actual_ingress_until_parent_settlement() {
    assert_matrix_goal_deferred_ingress(false, GoalIngressEnding::DeferredPause).await;
}

#[tokio::test]
async fn matrix_goal_budget_is_queued_and_enforced_after_parent_settlement() {
    assert_matrix_goal_deferred_ingress(false, GoalIngressEnding::DeferredBudget).await;
}

enum GoalIngressEnding {
    Cancel,
    DeferredPause,
    DeferredBudget,
}

async fn assert_matrix_goal_deferred_ingress(resume: bool, ending: GoalIngressEnding) {
    use zeroclaw_goal::GoalRuntime;
    use zeroclaw_infra::session_backend::SessionBackend;
    use zeroclaw_runtime::{
        control_plane::{ControlPlaneHandle, TaskStatus},
        cost::CostTracker,
        goal_accounting::{ModelProviderGoalUsagePricer, RuntimeGoalOperationAccountingFactory},
    };

    let directory = tempfile::tempdir().unwrap();
    let verifier_server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(wiremock::ResponseTemplate::new(500))
        .expect(0)
        .mount(&verifier_server)
        .await;
    let handle = ControlPlaneHandle::open(directory.path()).unwrap();
    let registry = handle.goal_store().unwrap();
    let channel = Arc::new(SendMessageRecordingChannel::matrix());
    let provider = Arc::new(GoalIngressBoundaryProvider::default());
    let mut config = matrix_control_test_config();
    config.data_dir = directory.path().to_owned();
    config.goal.enabled = true;
    config.goal.verifier.model_provider =
        zeroclaw_config::providers::ModelProviderRef::new("openai.fixture");
    let verifier = config.providers.models.ensure("openai", "fixture").unwrap();
    verifier.api_key = Some("synthetic-key".into());
    verifier.uri = Some(verifier_server.uri());
    verifier.kind = Some("openai-compatible".into());
    verifier.wire_api = Some(zeroclaw_config::schema::WireApi::ChatCompletions);
    verifier.model = Some("fixture-verifier".into());
    verifier.live_pricing = false;
    config.channels.debounce_ms = 1000;
    let mut context = test_runtime_ctx_with_config_agent_and_provider_ref(
        channel.clone(),
        provider.clone(),
        config,
        zeroclaw_config::schema::AliasedAgentConfig::default(),
        "test-provider",
        None,
    );
    let backend = Arc::new(
        zeroclaw_infra::session_sqlite::SqliteSessionBackend::new(directory.path()).unwrap(),
    );
    Arc::get_mut(&mut context).unwrap().session_store = Some(backend.clone());
    let mut message = matrix_msg("@fixture:example.test");
    message.content = "ordinary boundary request".into();
    let key = runtime_conversation_history_key(&context, &message);
    foreground::install_goal_test_control_plane(&context.persist_locks, &key, handle.clone());
    let tracker = Arc::new(
        CostTracker::new(
            zeroclaw_config::schema::CostConfig::default(),
            directory.path(),
        )
        .unwrap(),
    );
    let accounting = Arc::new(
        RuntimeGoalOperationAccountingFactory::new(
            registry.clone(),
            tracker.clone(),
            context.agent_alias.to_string(),
            Arc::new(ModelProviderGoalUsagePricer::new(Arc::default())),
        )
        .unwrap(),
    );
    let supervisor = handle
        .goal_execution_restart()
        .new_supervisor(Arc::new(
            GoalRuntime::new(registry.clone()).execution_engine(accounting),
        ))
        .await;
    *foreground::goal_supervisor_slot(&context.persist_locks, &key)
        .lock()
        .await = Some(supervisor.clone());
    let predecessor = if resume {
        let initial = goal_execution::submit_matrix_goal(
            context.clone(),
            key.clone(),
            message.clone(),
            zeroclaw_commands::goal::GoalCommand::Start {
                budget: zeroclaw_commands::goal::GoalBudgetSelection::Unlimited,
                objective: "fixture boundary objective".into(),
            },
        )
        .await
        .unwrap();
        // Prepare a real paused Goal through the existing failed-delivery
        // settlement, without launching a synthetic predecessor worker.
        initial.response_delivery_failed().await.unwrap();
        *foreground::goal_supervisor_slot(&context.persist_locks, &key)
            .lock()
            .await = Some(supervisor.clone());
        backend
            .append(
                &key,
                &zeroclaw_api::model_provider::ChatMessage::user("fixture initial request"),
            )
            .unwrap();
        backend
            .append(
                &key,
                &zeroclaw_api::model_provider::ChatMessage::assistant(
                    "Please choose fixture option A or B.",
                ),
            )
            .unwrap();
        let task = registry
            .current_goal_for_session(&key)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(task.status, TaskStatus::Paused);
        assert_eq!(task.execution_epoch, 2);
        Some((task.id, task.execution_epoch))
    } else {
        None
    };
    let audit = rusqlite::Connection::open(directory.path().join("control_plane.db")).unwrap();
    let intent_state = || {
        audit
            .query_row(
                "SELECT state FROM goal_control_intents WHERE session_id = ?1",
                [&key],
                |row| row.get::<_, String>(0),
            )
            .unwrap()
    };
    let (sender, receiver) = tokio::sync::mpsc::channel(8);
    let router = AgentRouter::single(context.clone());
    let mut dispatch = zeroclaw_spawn::spawn!(run_message_dispatch_loop(receiver, router, 4,));
    sender.send(message.clone()).await.unwrap();
    let mut launched_scope = None;
    let outcome = async {
        tokio::time::timeout(Duration::from_secs(3), async {
            while provider.requests.lock().unwrap().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("ordinary turn must enter before the Goal control");
        let response = "First fixture answer\nSecond fixture line.";
        message.content = if resume {
            format!("/goal resume {response}")
        } else {
            "/goal start --unlimited fixture boundary objective".into()
        };
        sender.send(message.clone()).await.unwrap();
        let queued = channel_runtime_cli_string(if resume {
            "goal-mode-queued-resume"
        } else {
            "goal-mode-queued-start"
        });
        tokio::time::timeout(Duration::from_millis(250), async {
            while !channel
                .sent_messages
                .lock()
                .await
                .iter()
                .any(|reply| reply.content == queued)
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("valid Goal control must acknowledge before the ordinary turn ends");
        let before = registry.current_goal_for_session(&key).await.unwrap();
        if let Some((id, epoch)) = predecessor.as_ref() {
            let task = before.unwrap();
            assert_eq!(&task.id, id);
            assert_eq!(task.status, TaskStatus::Paused);
            assert_eq!(task.execution_epoch, *epoch);
            assert!(backend.load(&key).iter().all(|row| row.content != response));
        } else {
            assert!(before.is_none());
        }
        assert_eq!(intent_state(), "pending");
        assert_eq!(provider.requests.lock().unwrap().len(), 1);
        provider.ordinary_release.notify_one();
        tokio::time::timeout(Duration::from_secs(3), async {
            while provider.requests.lock().unwrap().len() < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("one Goal parent must begin after the ordinary turn");
        let task = registry
            .current_goal_for_session(&key)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(task.status, TaskStatus::Running);
        assert_eq!(
            task.execution_epoch,
            predecessor.as_ref().map_or(1, |(_, epoch)| epoch + 1)
        );
        launched_scope = Some(
            zeroclaw_goal::GoalExecutionScope::new(
                task.id.clone(),
                key.clone(),
                task.execution_epoch,
            )
            .unwrap(),
        );
        if let Some((id, _)) = predecessor.as_ref() {
            assert_eq!(&task.id, id);
            let requests = provider.requests.lock().unwrap();
            assert_eq!(
                requests[1]
                    .iter()
                    .filter(|row| row.role == "user" && row.content == response)
                    .count(),
                1
            );
            assert!(requests[1].iter().any(|row| row.role == "assistant"
                && row.content == "Please choose fixture option A or B."));
            assert_eq!(
                backend
                    .load(&key)
                    .iter()
                    .filter(|row| row.role == "user" && row.content == response)
                    .count(),
                1
            );
        }
        assert_eq!(intent_state(), "applied");
        let replies = channel.sent_messages.lock().await;
        let ordinary = replies
            .iter()
            .position(|reply| reply.content == "ordinary boundary candidate")
            .unwrap();
        let lifecycle = replies
            .iter()
            .position(|reply| {
                reply.content.contains(if resume {
                    "Goal resumed."
                } else {
                    "Goal started."
                })
            })
            .unwrap();
        assert_eq!(
            lifecycle,
            ordinary + 1,
            "started precedes every Goal execution event"
        );
        drop(replies);
        let expected_status = match ending {
            GoalIngressEnding::Cancel => {
                message.content = "/goal cancel".into();
                TaskStatus::Cancelled
            }
            GoalIngressEnding::DeferredPause => {
                message.content = "/goal pause".into();
                TaskStatus::Paused
            }
            GoalIngressEnding::DeferredBudget => {
                message.content = "/goal budget set --tokens 10".into();
                TaskStatus::Paused
            }
        };
        sender.send(message.clone()).await.unwrap();
        if !matches!(ending, GoalIngressEnding::Cancel) {
            let (queued_key, intent_kind) = match ending {
                GoalIngressEnding::DeferredPause => ("goal-mode-queued-pause", "pause"),
                GoalIngressEnding::DeferredBudget => ("goal-mode-queued-budget", "set_budget"),
                GoalIngressEnding::Cancel => unreachable!(),
            };
            let queued_notice = channel_runtime_cli_string(queued_key);
            tokio::time::timeout(Duration::from_millis(250), async {
                while !channel
                    .sent_messages
                    .lock()
                    .await
                    .iter()
                    .any(|reply| reply.content == queued_notice)
                {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("normal control must acknowledge while the Goal parent remains held");
            let held = registry
                .current_goal_for_session(&key)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(held.status, TaskStatus::Running);
            assert_eq!(held.execution_epoch, task.execution_epoch);
            let goal = registry.get_goal_task(&held.id).await.unwrap().unwrap();
            assert!(goal.pause_reason.is_none());
            assert!(goal.effective_token_limit.is_none());
            let intent = audit.query_row(
                "SELECT state, task_id, expected_epoch FROM goal_control_intents WHERE session_id = ?1 AND kind = ?2",
                [&key, intent_kind],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, i64>(2)?)),
            ).unwrap();
            assert_eq!(intent, ("pending".into(), held.id, held.execution_epoch));
            assert_eq!(provider.requests.lock().unwrap().len(), 2);
            provider.goal_release.notify_one();
        }
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if registry
                    .current_goal_for_session(&key)
                    .await
                    .unwrap()
                    .unwrap()
                    .status
                    == expected_status
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the admitted control must reach its specified boundary");
    };
    // Always release both provider gates before surfacing a failed assertion.
    let result = std::panic::AssertUnwindSafe(outcome).catch_unwind().await;
    provider.ordinary_release.notify_one();
    provider.goal_release.notify_one();
    drop(sender);
    let drained = tokio::time::timeout(Duration::from_secs(3), &mut dispatch).await;
    if drained.is_err() {
        dispatch.abort();
    }
    assert!(
        drained.is_ok(),
        "actual ingress must drain after fixture release"
    );
    drained.unwrap().unwrap();
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
    if matches!(ending, GoalIngressEnding::DeferredBudget) {
        // Unlike Pause/Cancel, a budget update does not fence and drain the
        // worker itself. Consume its completion through the real supervisor;
        // owns_scope deliberately remains true for an undrained result.
        let completion = tokio::time::timeout(
            Duration::from_secs(3),
            supervisor.drain(launched_scope.as_ref().unwrap()),
        )
        .await
        .expect("budget-limited worker must finish without verifier admission");
        // The producer's committed budget fence must survive the original
        // worker drain without converting the resumable pause into failure.
        assert_eq!(
            completion.unwrap(),
            zeroclaw_goal::GoalExecutionOutcome::Fenced
        );
        assert_eq!(
            registry
                .current_goal_for_session(&key)
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Paused,
            "RG-55: draining a budget-exhausted worker must not terminalize its resumable pause",
        );
    }
    assert_eq!(provider.requests.lock().unwrap().len(), 2);
    assert!(
        verifier_server
            .received_requests()
            .await
            .unwrap()
            .is_empty()
    );
    if !matches!(ending, GoalIngressEnding::Cancel) {
        let scope = launched_scope.as_ref().unwrap();
        let goal = registry
            .get_goal_task(scope.task_id())
            .await
            .unwrap()
            .unwrap();
        let (reason, intent_kind) = match ending {
            GoalIngressEnding::DeferredPause => {
                (zeroclaw_goal::GoalPauseReason::OperatorPaused, "pause")
            }
            GoalIngressEnding::DeferredBudget => {
                assert_eq!(goal.effective_token_limit, Some(10));
                (
                    zeroclaw_goal::GoalPauseReason::BudgetExhausted,
                    "set_budget",
                )
            }
            GoalIngressEnding::Cancel => unreachable!(),
        };
        assert_eq!(goal.pause_reason, Some(reason));
        assert_eq!(
            goal.accounting_state,
            zeroclaw_goal::GoalAccountingState::Complete
        );
        assert!(goal.pending_call_id.is_none());
        assert!(goal.pending_call_epoch.is_none());
        assert_eq!(
            tracker
                .get_usage_totals_for_task(scope.task_id())
                .unwrap()
                .0,
            12
        );
        assert_eq!(
            audit
                .query_row(
                    "SELECT state FROM goal_control_intents WHERE session_id = ?1 AND kind = ?2",
                    [&key, intent_kind],
                    |row| row.get::<_, String>(0)
                )
                .unwrap(),
            "applied"
        );
        assert_eq!(
            backend
                .load(&key)
                .iter()
                .filter(|row| row.role == "assistant" && row.content == "Goal boundary candidate")
                .count(),
            1
        );
        let replies = channel.sent_messages.lock().await;
        let candidate = replies
            .iter()
            .position(|reply| reply.content == "Goal boundary candidate")
            .unwrap();
        if matches!(ending, GoalIngressEnding::DeferredBudget) {
            let core_error = channel_user_error_message(
                &anyhow::Error::new(
                    zeroclaw_goal::accounting::GoalOperationFenced::BUDGET_EXHAUSTED,
                ),
                "Goal budget is exhausted",
            );
            assert_eq!(
                replies
                    .iter()
                    .filter(|reply| reply.content == core_error)
                    .count(),
                1,
                "the ordinary core error remains visible exactly once"
            );
            assert!(
                replies
                    .iter()
                    .all(|reply| !reply.content.contains("Goal failed."))
            );
            let budget_updated = channel_runtime_cli_string("goal-mode-budget-updated");
            let budget_updates = replies
                .iter()
                .filter(|reply| reply.content == budget_updated)
                .count();
            assert_eq!(budget_updates, 1);
            let final_response = replies
                .iter()
                .position(|reply| reply.content == budget_updated)
                .unwrap();
            assert!(
                candidate < final_response,
                "the budget response follows the settled parent candidate"
            );
        } else {
            let paused = replies
                .iter()
                .position(|reply| reply.content.contains("Goal paused."))
                .unwrap();
            assert!(
                candidate < paused,
                "the original candidate precedes the final pause notice"
            );
        }
    }
    let scope = launched_scope.unwrap();
    assert!(!supervisor.owns_scope(&scope).await);
    if matches!(ending, GoalIngressEnding::DeferredBudget) {
        use zeroclaw_commands::goal::{GoalBudgetSelection, GoalCommand};
        let recovery = async {
            let mut budget = goal_execution::submit_matrix_goal(
                context.clone(),
                key.clone(),
                message.clone(),
                GoalCommand::SetBudget(GoalBudgetSelection::Unlimited),
            )
            .await
            .unwrap();
            assert!(matches!(
                budget.response(),
                zeroclaw_goal::GoalResponse::BudgetUpdated(_)
            ));
            assert!(budget.after_response().await.unwrap().is_none());
            drop(budget);
            let mut resume = goal_execution::submit_matrix_goal(
                context.clone(),
                key.clone(),
                message.clone(),
                GoalCommand::Resume { response: None },
            )
            .await
            .unwrap();
            assert!(matches!(
                resume.response(),
                zeroclaw_goal::GoalResponse::Resumed(_)
            ));
            // Deliver the native lifecycle acknowledgement before the real
            // host handoff, just as the production delivery coordinator does.
            channel
                .send(&SendMessage::reply_to(
                    &message,
                    render_goal_response(resume.response()),
                ))
                .await
                .unwrap();
            assert!(resume.after_response().await.unwrap().is_none());
            drop(resume);
            tokio::time::timeout(Duration::from_secs(3), async {
                while provider.requests.lock().unwrap().len() < 3 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("the budget-paused Goal must resume into a real parent request");
            let resumed = registry
                .current_goal_for_session(&key)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(resumed.id, scope.task_id());
            assert_eq!(resumed.status, TaskStatus::Running);
            assert_eq!(resumed.execution_epoch, scope.execution_epoch() + 2);
            let resumed_scope = zeroclaw_goal::GoalExecutionScope::new(
                resumed.id,
                key.clone(),
                resumed.execution_epoch,
            )
            .unwrap();
            let mut cancel = goal_execution::submit_matrix_goal(
                context.clone(),
                key.clone(),
                message.clone(),
                GoalCommand::Cancel,
            )
            .await
            .unwrap();
            provider.resumed_release.notify_one();
            cancel.after_response().await.unwrap();
            assert!(!supervisor.owns_scope(&resumed_scope).await);
        };
        let result = std::panic::AssertUnwindSafe(recovery).catch_unwind().await;
        provider.resumed_release.notify_one();
        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
        assert_eq!(provider.requests.lock().unwrap().len(), 3);
        assert!(
            verifier_server
                .received_requests()
                .await
                .unwrap()
                .is_empty()
        );
    }
}

enum GoalInboundHook {
    Cancel,
    Rewrite,
}

#[async_trait::async_trait]
impl zeroclaw_runtime::hooks::HookHandler for GoalInboundHook {
    fn name(&self) -> &str {
        "goal-control-ingress-fixture"
    }

    async fn on_message_received(
        &self,
        mut message: ChannelMessage,
    ) -> zeroclaw_runtime::hooks::HookResult<ChannelMessage> {
        match self {
            Self::Cancel => {
                zeroclaw_runtime::hooks::HookResult::Cancel("fixture cancellation".into())
            }
            Self::Rewrite => {
                message.content = "rewritten ordinary request".into();
                message.sender = "@other:example.test".into();
                message.reply_target = "rewritten-room".into();
                zeroclaw_runtime::hooks::HookResult::Continue(message)
            }
        }
    }
}

async fn dispatch_goal_with_hook(hook: GoalInboundHook) -> Vec<SendMessage> {
    let channel = Arc::new(SendMessageRecordingChannel::matrix());
    let provider = Arc::new(GatedProbeModelProvider::default());
    let mut context = test_runtime_ctx_with_config_agent_and_provider_ref(
        channel.clone(),
        provider.clone(),
        matrix_control_test_config(),
        zeroclaw_config::schema::AliasedAgentConfig::default(),
        "test-provider",
        None,
    );
    let mut hooks = zeroclaw_runtime::hooks::HookRunner::new();
    hooks.register(Box::new(hook));
    Arc::get_mut(&mut context).unwrap().hooks = Some(Arc::new(hooks));
    let (sender, receiver) = tokio::sync::mpsc::channel(8);
    let mut message = matrix_msg("@fixture:example.test");
    message.content = "/goal unknown".into();
    sender.send(message).await.unwrap();
    drop(sender);
    tokio::time::timeout(
        Duration::from_secs(3),
        run_message_dispatch_loop(receiver, AgentRouter::single(context), 4),
    )
    .await
    .unwrap();
    assert!(
        provider.entered_ids().is_empty(),
        "the hook must not turn a Goal command into a model prompt"
    );
    channel.sent_messages.lock().await.clone()
}

#[tokio::test]
async fn matrix_goal_ingress_keeps_hook_cancellation_before_admission() {
    assert!(
        dispatch_goal_with_hook(GoalInboundHook::Cancel)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn matrix_goal_ingress_keeps_original_command_and_reply_authority() {
    let replies = dispatch_goal_with_hook(GoalInboundHook::Rewrite).await;
    assert_eq!(replies.len(), 1);
    assert_eq!(
        replies[0].content,
        channel_runtime_cli_string("goal-mode-command-invalid")
    );
    assert_eq!(
        replies[0].recipient,
        matrix_msg("@fixture:example.test").reply_target
    );
}

#[test]
fn matrix_goal_ingress_classification_preserves_surface_and_passive_boundaries() {
    let mut message = matrix_msg("@fixture:example.test");
    for command in [
        "/goal start --unlimited fixture objective",
        "/goal resume first line\nsecond line",
        "/goal pause",
        "/goal pause now",
        "/goal cancel",
        "/goal status",
        "/goal unknown",
    ] {
        message.content = command.into();
        assert!(is_matrix_goal_control(&message), "{command}");
    }
    message.passive_context = true;
    assert!(!is_matrix_goal_control(&message));
    message.passive_context = false;
    message.channel = "cli".into();
    assert!(!is_matrix_goal_control(&message));
    message.channel = "matrix".into();
    message.content = "ordinary text mentioning /goal resume".into();
    assert!(!is_matrix_goal_control(&message));
}

#[test]
fn matrix_goal_queued_acknowledgements_name_the_control_boundary() {
    use zeroclaw_goal::{GoalQueuedControl, GoalResponse};
    for (control, expected) in [
        (
            GoalQueuedControl::Start,
            "Goal start queued; it begins after the current turn.",
        ),
        (
            GoalQueuedControl::Resume,
            "Goal resume queued; it continues after the current turn.",
        ),
        (
            GoalQueuedControl::Pause,
            "Goal pause queued; it applies after the current step.",
        ),
        (
            GoalQueuedControl::Budget,
            "Goal budget change queued; it applies after the current step.",
        ),
    ] {
        assert_eq!(
            render_goal_response(&GoalResponse::Queued(control)),
            expected
        );
    }
}

fn matrix_control_test_config() -> zeroclaw_config::schema::Config {
    let mut config = zeroclaw_config::schema::Config::default();
    config.channels.matrix.insert(
        "default".into(),
        zeroclaw_config::schema::MatrixConfig {
            interrupt_on_new_message: true,
            ..Default::default()
        },
    );
    config
}

fn matrix_control_test_ctx() -> Arc<ChannelRuntimeContext> {
    test_runtime_ctx_with_config_agent_and_provider_ref(
        Arc::new(SendMessageRecordingChannel::matrix()),
        Arc::new(GatedProbeModelProvider::default()),
        matrix_control_test_config(),
        zeroclaw_config::schema::AliasedAgentConfig::default(),
        "test-provider",
        None,
    )
}

async fn assert_matrix_control_does_not_register(content: &str) {
    let ctx = matrix_control_test_ctx();
    let in_flight = Arc::new(Mutex::new(HashMap::new()));
    let sequence = Arc::new(AtomicU64::new(1));
    let generation = CancellationToken::new();
    let ordinary = matrix_msg("@fixture:example.test");
    let registration = register_inbound_turn(&ctx, &ordinary, &in_flight, &sequence, &generation)
        .await
        .unwrap();
    let mut control = ordinary.clone();
    control.content = content.into();
    let control_registration =
        register_inbound_turn(&ctx, &control, &in_flight, &sequence, &generation).await;

    assert!(
        !registration.cancellation.is_cancelled(),
        "a Matrix Goal control must not interrupt the admitted ordinary turn"
    );
    assert!(
        control_registration.is_none(),
        "a Matrix Goal control must not acquire an ordinary interruption slot"
    );
    assert_eq!(sequence.load(Ordering::Relaxed), 2);
    assert_eq!(
        in_flight
            .lock()
            .unwrap()
            .values()
            .map(Vec::len)
            .sum::<usize>(),
        1
    );
}

#[tokio::test]
async fn matrix_goal_normal_controls_do_not_register_or_interrupt() {
    for command in [
        "/goal start --unlimited fixture objective",
        "/goal resume",
        "/goal resume first line\nsecond line",
        "/goal pause",
    ] {
        assert_matrix_control_does_not_register(command).await;
    }
}

#[tokio::test]
async fn matrix_goal_parse_error_does_not_register_or_interrupt() {
    assert_matrix_control_does_not_register("/goal unknown").await;
}

#[tokio::test]
async fn matrix_goal_control_exemption_preserves_ordinary_interruption() {
    let ctx = matrix_control_test_ctx();
    let in_flight = Arc::new(Mutex::new(HashMap::new()));
    let sequence = Arc::new(AtomicU64::new(1));
    let generation = CancellationToken::new();
    let message = matrix_msg("@fixture:example.test");
    let first = register_inbound_turn(&ctx, &message, &in_flight, &sequence, &generation)
        .await
        .unwrap();
    let second = register_inbound_turn(&ctx, &message, &in_flight, &sequence, &generation)
        .await
        .unwrap();
    assert!(first.cancellation.is_cancelled());
    assert!(!second.cancellation.is_cancelled());
    assert_eq!(sequence.load(Ordering::Relaxed), 3);
}

#[tokio::test]
async fn matrix_goal_parse_error_is_acknowledged_during_ordinary_turn_without_debounce() {
    let channel = Arc::new(SendMessageRecordingChannel::matrix());
    let provider = Arc::new(GatedProbeModelProvider::default());
    provider.hold("m1");
    let mut config = matrix_control_test_config();
    config.channels.debounce_ms = 1000;
    let ctx = test_runtime_ctx_with_config_agent_and_provider_ref(
        channel.clone(),
        provider.clone(),
        config,
        zeroclaw_config::schema::AliasedAgentConfig::default(),
        "test-provider",
        None,
    );
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    let mut dispatch =
        zeroclaw_spawn::spawn!(run_message_dispatch_loop(rx, AgentRouter::single(ctx), 4));
    let mut ordinary = matrix_msg("@fixture:example.test");
    ordinary.content = "message_id=m1 ordinary request".into();
    tx.send(ordinary.clone()).await.unwrap();
    let entered = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if provider.entered_ids().contains(&"m1".into()) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await;
    if entered.is_err() {
        provider.release("m1");
        dispatch.abort();
        panic!("the ordinary request must enter the provider before the control");
    }
    let mut control = ordinary;
    control.content = "/goal unknown".into();
    tx.send(control).await.unwrap();
    let expected = channel_runtime_cli_string("goal-mode-command-invalid");
    let acknowledged = tokio::time::timeout(Duration::from_millis(250), async {
        loop {
            if channel
                .sent_messages
                .lock()
                .await
                .iter()
                .any(|message| message.content == expected)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await;

    // Always release the fixture before asserting, including the expected red.
    provider.release("m1");
    drop(tx);
    let drained = tokio::time::timeout(Duration::from_secs(3), &mut dispatch).await;
    if drained.is_err() {
        dispatch.abort();
    }
    assert!(
        drained.is_ok(),
        "the real dispatcher must drain after release"
    );
    drained.unwrap().unwrap();
    assert!(
        acknowledged.is_ok(),
        "a Goal parse error must bypass the one-second debounce and ordinary lane"
    );
    assert!(
        channel
            .sent_messages
            .lock()
            .await
            .iter()
            .any(|message| { message.content.contains("done for message_id=m1") }),
        "the control must leave the ordinary turn's response intact"
    );
    assert_eq!(provider.entered_ids(), ["m1"]);
}
