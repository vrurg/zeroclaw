//! Real-control-plane proof of the verifier diagnostic egress boundary.

use std::sync::{
    Arc, Mutex, Once,
    atomic::{AtomicUsize, Ordering},
};

use async_trait::async_trait;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::time::{Duration, timeout};
use zeroclaw_api::{
    model_provider::{ChatMessage, TokenUsage},
    task::TaskStatus,
};
use zeroclaw_commands::goal::{GoalBudgetLimits, GoalBudgetSelection, GoalCommand};
use zeroclaw_config::{cost::CostTracker, schema::CostConfig};
use zeroclaw_goal::{
    GoalAccountingState, GoalExecutionNotice, GoalExecutionOutcome, GoalExecutionScope,
    GoalExecutionSupervisor, GoalHostSettings, GoalIngressContext, GoalIngressPrincipal,
    GoalOperationScope, GoalParentTurn, GoalParentTurnCheckpoint, GoalParentTurnResult,
    GoalRuntime, GoalSessionBinding, GoalSessionDriver, GoalSessionExecutionLease, GoalSessionKey,
    GoalSessionLease, GoalTaskRegistry, GoalVerifierAttribution, GoalVerifierResult,
    GoalVerifierTurn,
    accounting::{
        GOAL_OPERATION_ACCOUNTING, GoalOperationRequest, GoalOperationSettlement, GoalUsageEvent,
    },
};
use zeroclaw_runtime::{
    control_plane::SqliteTaskStore,
    goal_accounting::{
        ModelProviderGoalUsagePricer, RuntimeGoalDiagnosticTextGuard,
        RuntimeGoalOperationAccountingFactory,
    },
};

#[derive(Clone, Copy)]
enum Verdict {
    Complete,
    Continue,
    Blocked,
    Invalid,
    Error,
}

struct Script {
    reason: String,
    verdict: Verdict,
    parents: AtomicUsize,
    verifiers: AtomicUsize,
    settlements: AtomicUsize,
    inputs: Mutex<Vec<Vec<ChatMessage>>>,
    events: Mutex<Vec<String>>,
}

struct Driver {
    key: GoalSessionKey,
    script: Arc<Script>,
}

struct Lease {
    key: GoalSessionKey,
    script: Arc<Script>,
}

#[async_trait]
impl GoalSessionDriver for Driver {
    fn session_key(&self) -> &GoalSessionKey {
        &self.key
    }

    async fn bind(&self, _: &GoalIngressContext) -> anyhow::Result<GoalSessionLease> {
        Ok(GoalSessionLease::new(
            GoalSessionBinding::new(self.key.clone()),
            (),
        ))
    }

    async fn acquire_execution(
        &self,
        _: &GoalIngressContext,
        _: &GoalExecutionScope,
        _: &GoalParentTurnCheckpoint,
    ) -> anyhow::Result<Box<dyn GoalSessionExecutionLease>> {
        Ok(Box::new(Lease {
            key: self.key.clone(),
            script: self.script.clone(),
        }))
    }
}

impl Lease {
    async fn settle(&self) -> anyhow::Result<()> {
        let accountant = GOAL_OPERATION_ACCOUNTING.with(Clone::clone).unwrap();
        accountant
            .admit(GoalOperationRequest::new("fixture.route", "fixture-model"))
            .await?;
        accountant
            .settle(GoalOperationSettlement {
                accounting_state: GoalAccountingState::Complete,
                events: vec![GoalUsageEvent {
                    provider_ref: "fixture.route".into(),
                    model: "fixture-model".into(),
                    usage: TokenUsage {
                        input_tokens: Some(10),
                        output_tokens: Some(2),
                        cached_input_tokens: Some(0),
                        cache_creation_input_tokens: Some(0),
                    },
                    priceable: true,
                }],
            })
            .await?;
        self.script.settlements.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[async_trait]
impl GoalSessionExecutionLease for Lease {
    fn session_key(&self) -> &GoalSessionKey {
        &self.key
    }
    fn canonical_history(&self) -> anyhow::Result<Vec<ChatMessage>> {
        Ok(vec![])
    }

    async fn run_parent_turn(
        &mut self,
        _: &GoalOperationScope,
        turn: GoalParentTurn,
    ) -> anyhow::Result<GoalParentTurnResult> {
        self.script.parents.fetch_add(1, Ordering::SeqCst);
        self.script
            .inputs
            .lock()
            .unwrap()
            .push(turn.working_history);
        self.settle().await?;
        GoalParentTurnResult::from_completed_agent_turn(
            "bounded candidate".into(),
            vec![ChatMessage::assistant("bounded candidate")],
        )
    }

    async fn run_verifier(
        &mut self,
        _: &GoalOperationScope,
        _: GoalVerifierTurn,
    ) -> anyhow::Result<GoalVerifierResult> {
        let call = self.script.verifiers.fetch_add(1, Ordering::SeqCst);
        self.settle().await?;
        if matches!(self.script.verdict, Verdict::Error) {
            anyhow::bail!("fixture verifier error: {}", self.script.reason);
        }
        let decision = match self.script.verdict {
            Verdict::Continue if call == 0 => "continue",
            Verdict::Blocked => "blocked",
            _ => "complete",
        };
        let blockers = if decision == "blocked" {
            json!([{"kind":"needs_user_input", "message":self.script.reason,
                "payload":{"fixture":stripe_token()}}])
        } else {
            json!([])
        };
        let response = if matches!(self.script.verdict, Verdict::Invalid) {
            format!("not JSON: {}", self.script.reason)
        } else {
            json!({"decision":decision,"reason":self.script.reason,"blockers":blockers}).to_string()
        };
        Ok(GoalVerifierResult {
            response,
            attribution: Some(GoalVerifierAttribution {
                provider_ref: "fixture.route".into(),
                model: "fixture-model".into(),
            }),
        })
    }

    async fn append_parent_candidate(&mut self, candidate: String) -> anyhow::Result<()> {
        self.script.events.lock().unwrap().push(candidate);
        Ok(())
    }

    async fn publish_goal_notice(&mut self, notice: GoalExecutionNotice) -> anyhow::Result<()> {
        self.script
            .events
            .lock()
            .unwrap()
            .push(format!("notice:{notice:?}"));
        Ok(())
    }
}

struct Observation {
    frames: Vec<Value>,
    script: Arc<Script>,
    status: TaskStatus,
}

async fn run_case(reason: String, verdict: Verdict) -> Observation {
    run_case_with_guard(reason, verdict, true).await
}

async fn run_case_with_guard(reason: String, verdict: Verdict, guarded: bool) -> Observation {
    static LOG_INIT: Once = Once::new();
    static CASE: AtomicUsize = AtomicUsize::new(0);
    LOG_INIT.call_once(|| {
        zeroclaw_log::try_install_capture_subscriber();
        drop(zeroclaw_log::subscribe_or_install());
    });
    let mut logs = zeroclaw_log::subscribe().unwrap();
    let key = GoalSessionKey::matrix(format!(
        "matrix_diagnostic_fixture_{}",
        CASE.fetch_add(1, Ordering::SeqCst)
    ))
    .unwrap();
    let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
    let runtime = GoalRuntime::new(store.clone());
    let directory = tempfile::tempdir().unwrap();
    let tracker = Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap());
    let accounting = Arc::new(
        RuntimeGoalOperationAccountingFactory::new(
            store.clone(),
            tracker.clone(),
            "fixture-agent",
            Arc::new(ModelProviderGoalUsagePricer::new(Arc::default())),
        )
        .unwrap(),
    );
    let engine = runtime.execution_engine(accounting);
    let engine = if guarded {
        engine.with_diagnostic_text_guard(Arc::new(RuntimeGoalDiagnosticTextGuard))
    } else {
        engine
    };
    let supervisor = GoalExecutionSupervisor::new(Arc::new(engine));
    let script = Arc::new(Script {
        reason,
        verdict,
        parents: AtomicUsize::new(0),
        verifiers: AtomicUsize::new(0),
        settlements: AtomicUsize::new(0),
        inputs: Mutex::new(vec![]),
        events: Mutex::new(vec![]),
    });
    let driver = Arc::new(Driver {
        key: key.clone(),
        script: script.clone(),
    });
    let ingress = GoalIngressContext::trusted(
        key.clone(),
        "fixture-agent",
        "fixture-origin",
        GoalIngressPrincipal::Matrix {
            raw_mxid: "@fixture:example.invalid".into(),
        },
    )
    .unwrap();
    let settings = GoalHostSettings::new(
        true,
        GoalBudgetLimits {
            token_limit: Some(100),
            cost_limit_usd: None,
        },
        42,
        "fixture-boot",
    )
    .unwrap();
    let mut start = supervisor
        .submit(
            settings.clone(),
            ingress,
            driver,
            GoalCommand::Start {
                budget: GoalBudgetSelection::Defaults,
                objective: "finish bounded work".into(),
            },
        )
        .await
        .unwrap();
    let original = store
        .current_goal_for_session(&key.durable_id())
        .await
        .unwrap()
        .unwrap();
    let scope =
        GoalExecutionScope::new(&original.id, key.durable_id(), original.execution_epoch).unwrap();
    supervisor
        .launch_after_response(settings, &mut start)
        .await
        .unwrap();
    let outcome = timeout(Duration::from_secs(5), supervisor.drain(&scope))
        .await
        .unwrap();
    if matches!(verdict, Verdict::Invalid | Verdict::Error) {
        assert!(outcome.is_err());
    } else {
        assert_eq!(
            outcome.unwrap(),
            if matches!(verdict, Verdict::Blocked) {
                GoalExecutionOutcome::VerifierBlocked
            } else {
                GoalExecutionOutcome::Completed
            }
        );
    }
    let task = store
        .current_goal_for_session(&key.durable_id())
        .await
        .unwrap()
        .unwrap();
    let goal = store.get_goal_task(&task.id).await.unwrap().unwrap();
    assert_eq!(goal.accounting_state, GoalAccountingState::Complete);
    assert!(goal.pending_call_id.is_none());
    assert!(goal.pending_tool_batch_id.is_none());
    assert_eq!(
        tracker.get_usage_totals_for_task(&task.id).unwrap().0,
        script.settlements.load(Ordering::SeqCst) as u64 * 12,
    );
    if matches!(verdict, Verdict::Blocked) {
        assert_eq!(goal.blockers.len(), 1);
        assert_eq!(goal.blockers[0].message, script.reason);
        assert_eq!(
            goal.blockers[0].payload,
            Some(json!({"fixture":stripe_token()}))
        );
    }
    let mut frames = vec![];
    while let Ok(event) = logs.try_recv() {
        if event["message"].as_str() == Some("Goal verifier decision and controller action")
            && event["attributes"]["task_id"] == task.id
        {
            for field in [
                "task_id",
                "execution_epoch",
                "candidate_fingerprint",
                "verifier_provider_ref",
                "verifier_model",
                "validation_outcome",
                "validated_decision",
                "reason",
                "controller_action",
                "phase",
            ] {
                assert!(event["attributes"].get(field).is_some(), "missing {field}");
            }
            assert_eq!(
                event["attributes"]["execution_epoch"],
                original.execution_epoch
            );
            assert_eq!(
                event["attributes"]["candidate_fingerprint"],
                format!("{:x}", Sha256::digest("bounded candidate")),
            );
            if matches!(verdict, Verdict::Error) {
                // A failed lease returns no attribution result to the engine.
                assert!(event["attributes"]["verifier_provider_ref"].is_null());
                assert!(event["attributes"]["verifier_model"].is_null());
            } else {
                assert_eq!(
                    event["attributes"]["verifier_provider_ref"],
                    "fixture.route"
                );
                assert_eq!(event["attributes"]["verifier_model"], "fixture-model");
            }
            assert!(
                !event["attributes"]
                    .to_string()
                    .contains("bounded candidate")
            );
            frames.push(event["attributes"].clone());
        }
    }
    assert!(
        !frames.is_empty(),
        "capture must reach the real diagnostic producer"
    );
    assert_eq!(
        frames.len(),
        if matches!(verdict, Verdict::Continue) {
            4
        } else {
            2
        },
        "each verifier attempt retains its validation and controller-action frames"
    );
    for pair in frames.as_chunks::<2>().0 {
        assert_eq!(
            pair[0]["phase"],
            match verdict {
                Verdict::Invalid => "decision_invalid",
                Verdict::Error => "operation_failed",
                _ => "decision_validated",
            }
        );
        assert_eq!(pair[1]["phase"], "after_controller_action");
    }
    Observation {
        frames,
        script,
        status: task.status,
    }
}

fn stripe_token() -> String {
    ["sk", "_live_", "0123456789abcdefghijklmnopqrstuv"].concat()
}

fn jwt_token() -> String {
    [
        "eyJ",
        "fixtureHeader.eyJ",
        "fixturePayload.fixtureSignature",
    ]
    .concat()
}

/// The old table misses the central detector's Stripe-shaped credential.
#[tokio::test]
async fn verifier_diagnostics_guard_detector_covered_text() {
    let token = stripe_token();
    let jwt = jwt_token();
    let mut observations = Vec::new();
    for verdict in [Verdict::Complete, Verdict::Continue, Verdict::Blocked] {
        let observed = run_case(
            format!("diagnostic fixture criterion met {token} {jwt} safely"),
            verdict,
        )
        .await;
        observations.push((verdict, observed));
    }
    for (verdict, observed) in observations {
        assert_eq!(
            observed.status,
            if matches!(verdict, Verdict::Blocked) {
                TaskStatus::Paused
            } else {
                TaskStatus::Completed
            }
        );
        for frame in observed.frames {
            assert!(
                !frame.to_string().contains(&token) && !frame.to_string().contains(&jwt),
                "credential reached canonical log frame"
            );
            let reason = frame["reason"].as_str().unwrap();
            assert!(reason.contains("REDACTED"));
            assert!(reason.contains("criterion met") && reason.contains("safely"));
        }
    }
}

/// Truncation must not leave a password fragment before the URL terminator.
#[tokio::test]
async fn verifier_diagnostics_guard_before_bound() {
    let password = ["fixture", "Password", "0123456789ABCDEF"].concat();
    let reason = format!(
        "{}postgres://fixture:{password}@example.invalid/db",
        "word ".repeat(45)
    );
    let observed = run_case(reason, Verdict::Complete).await;
    for frame in observed.frames {
        let text = frame["reason"].as_str().unwrap();
        assert!(text.chars().count() <= 256);
        assert!(
            !text.contains("fixturePass"),
            "password prefix survived truncation"
        );
    }
}

/// Removing a NUL joins a token; the scan must follow that mutation.
#[tokio::test]
async fn verifier_diagnostics_guard_after_normalization() {
    let token = stripe_token();
    let reason = format!("criterion met {}\0{}", &token[..2], &token[2..]);
    let observed = run_case(reason, Verdict::Complete).await;
    for frame in observed.frames {
        assert!(!frame.to_string().contains(&token));
    }
}

/// Quoted sensitive keys use the ordinary key/value redactor, not whole-text loss.
#[tokio::test]
async fn verifier_diagnostics_guard_json_password() {
    let password = ["test", "Password", "Value"].concat();
    let observed = run_case(
        format!("criterion met {{\"password\":\"{password}\"}}"),
        Verdict::Complete,
    )
    .await;
    for frame in observed.frames {
        assert!(!frame.to_string().contains(&password));
        assert!(frame["reason"].as_str().unwrap().contains("password"));
    }
}

/// Innocent words matching the old substring list must remain useful evidence.
#[tokio::test]
async fn verifier_diagnostics_preserve_valid_reason() {
    let reason = "The task-level criterion is met: Slovakia locale tests pass and the api_key field is documented; bearer-token auth is unchanged.";
    let observed = run_case(reason.into(), Verdict::Complete).await;
    for frame in observed.frames {
        assert_eq!(frame["reason"], reason);
    }
}

/// Preserve Unicode while removing Cc and normalizing whitespace; bidi is not Cc.
#[tokio::test]
async fn verifier_diagnostics_preserve_unicode_and_normalize_controls() {
    let observed = run_case(
        "Критерій виконано — café 東京\r\n\t\u{202e} x\0 y.".into(),
        Verdict::Complete,
    )
    .await;
    for frame in observed.frames {
        assert_eq!(
            frame["reason"],
            "Критерій виконано — café 東京 \u{202e} x y."
        );
    }
    let long_reason: String = "К🌻\r\n\t\u{001b}[31m\0\u{202e}"
        .repeat(200)
        .chars()
        .take(2000)
        .collect();
    assert_eq!(long_reason.chars().count(), 2000);
    let observed = run_case(long_reason, Verdict::Complete).await;
    for frame in observed.frames {
        let text = frame["reason"].as_str().unwrap();
        assert!(text.chars().count() <= 256);
        assert!(!text.chars().any(char::is_control));
        assert!(text.contains('К') && text.contains('🌻') && text.contains('\u{202e}'));
    }
}

/// Log redaction must not alter raw verifier feedback or operation accounting.
#[tokio::test]
async fn verifier_diagnostics_continue_preserves_feedback_and_counts() {
    let reason = format!("concrete work remains {}", stripe_token());
    let observed = run_case(reason.clone(), Verdict::Continue).await;
    assert_eq!(observed.script.parents.load(Ordering::SeqCst), 2);
    assert_eq!(observed.script.verifiers.load(Ordering::SeqCst), 2);
    assert_eq!(observed.script.settlements.load(Ordering::SeqCst), 4);
    assert!(
        observed.script.inputs.lock().unwrap()[1]
            .iter()
            .any(|row| row.content.contains(&reason))
    );
    assert_eq!(observed.frames[0]["phase"], "decision_validated");
    assert_eq!(
        observed.frames[1]["controller_action"],
        "continuation_prepared"
    );
}

/// Invalid output and operation-error bodies are not diagnostic reason text.
#[tokio::test]
async fn verifier_diagnostics_failures_keep_literal_frames() {
    let token = stripe_token();
    for (verdict, guarded) in [
        (Verdict::Invalid, false),
        (Verdict::Error, false),
        (Verdict::Invalid, true),
        (Verdict::Error, true),
    ] {
        let observed = run_case_with_guard(token.clone(), verdict, guarded).await;
        assert_eq!(observed.status, TaskStatus::Failed);
        for frame in observed.frames {
            assert!(!frame.to_string().contains(&token));
            assert_eq!(
                frame["reason"],
                if matches!(verdict, Verdict::Invalid) {
                    "Verifier response failed strict JSON/schema validation."
                } else {
                    "Verifier operation failed before a decision was available."
                }
            );
        }
    }
}

/// Unwired engines withhold reasons but keep decisions, raw feedback and blockers.
#[tokio::test]
async fn verifier_diagnostics_without_guard_fail_closed_at_every_action() {
    let reason = format!("fixture requires a choice {}", stripe_token());
    for verdict in [Verdict::Complete, Verdict::Continue, Verdict::Blocked] {
        let observed = run_case_with_guard(reason.clone(), verdict, false).await;
        for frame in &observed.frames {
            assert_eq!(
                frame["reason"],
                "[verifier reason withheld: no diagnostic leak guard installed]"
            );
        }
        assert_eq!(observed.frames[0]["phase"], "decision_validated");
        assert_eq!(observed.frames[1]["phase"], "after_controller_action");
        if matches!(verdict, Verdict::Continue) {
            assert!(
                observed.script.inputs.lock().unwrap()[1]
                    .iter()
                    .any(|row| row.content.contains(&reason))
            );
            assert_eq!(observed.script.parents.load(Ordering::SeqCst), 2);
        } else {
            assert_eq!(observed.script.parents.load(Ordering::SeqCst), 1);
        }
    }
}

/// A typed blocked decision preserves the full request before its pause notice.
#[tokio::test]
async fn verifier_diagnostics_blocked_keeps_delivery_order() {
    let observed = run_case("Please choose A or B.".into(), Verdict::Blocked).await;
    assert_eq!(observed.status, TaskStatus::Paused);
    let events = observed.script.events.lock().unwrap();
    assert_eq!(events[0], "bounded candidate");
    assert!(events[1].contains("PausedForBlocker"));
    assert_eq!(observed.script.parents.load(Ordering::SeqCst), 1);
    assert_eq!(observed.script.settlements.load(Ordering::SeqCst), 2);
    assert!(
        observed
            .frames
            .iter()
            .all(|frame| !frame.to_string().contains(&stripe_token()))
    );
}
