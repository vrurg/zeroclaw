use super::*;

async fn boundary_fixture() -> (
    MatrixGoalSessionDriver,
    GoalIngressContext,
    Arc<tokio::sync::Mutex<()>>,
) {
    let context =
        super::super::super::test_channel_ctx_with_backend(Arc::new(FailingAppendSessionBackend));
    let key = "matrix_boundary_fixture";
    let foreground = foreground_lock(&context.persist_locks, key);
    let command_lease = goal_command_lock(&context.persist_locks, key)
        .lock_owned()
        .await;
    let ingress = GoalIngressContext::trusted(
        GoalSessionKey::matrix(key).unwrap(),
        context.agent_alias.to_string(),
        "matrix:fixture:room",
        GoalIngressPrincipal::Matrix {
            raw_mxid: "@fixture:example.test".into(),
        },
    )
    .unwrap();
    let driver = MatrixGoalSessionDriver::new(
        context,
        key.into(),
        "@fixture:example.test".into(),
        "matrix:fixture:room".into(),
        ChannelMessage::new(
            "fixture-event",
            "@fixture:example.test",
            "room",
            "/goal resume",
            "matrix",
            1,
        ),
        command_lease,
    )
    .unwrap();
    (driver, ingress, foreground)
}

#[tokio::test]
async fn matrix_goal_boundary_try_retains_the_real_foreground_guard() {
    let (driver, ingress, foreground) = boundary_fixture().await;
    let ordinary_turn = foreground.clone().lock_owned().await;
    assert!(driver.try_turn_boundary(&ingress).await.unwrap().is_none());
    drop(ordinary_turn);
    let boundary = driver.try_turn_boundary(&ingress).await.unwrap().unwrap();
    assert!(foreground.try_lock().is_err());
    drop(boundary);
    assert!(foreground.try_lock().is_ok());
}

#[tokio::test]
async fn matrix_goal_boundary_waits_for_the_real_ordinary_turn() {
    let (driver, ingress, foreground) = boundary_fixture().await;
    let ordinary_turn = foreground.clone().lock_owned().await;
    let checkpoint = GoalParentTurnCheckpoint::new();
    let mut waiting = Box::pin(driver.await_turn_boundary(&ingress, &checkpoint));
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), &mut waiting)
            .await
            .is_err()
    );
    drop(ordinary_turn);
    let boundary = tokio::time::timeout(std::time::Duration::from_secs(1), waiting)
        .await
        .unwrap()
        .unwrap();
    assert!(foreground.try_lock().is_err());
    drop(boundary);
    assert!(foreground.try_lock().is_ok());
}

#[tokio::test]
async fn matrix_goal_boundary_closed_wait_never_acquires_foreground() {
    let (driver, ingress, foreground) = boundary_fixture().await;
    let checkpoint = GoalParentTurnCheckpoint::new();
    checkpoint.close();
    let error = driver
        .await_turn_boundary(&ingress, &checkpoint)
        .await
        .unwrap_err();
    assert!(error.is::<zeroclaw_goal::GoalCheckpointClosed>());
    assert!(foreground.try_lock().is_ok());
}

#[tokio::test]
async fn matrix_goal_boundary_rejects_stale_principal_on_both_paths() {
    let (driver, ingress, foreground) = boundary_fixture().await;
    let stale = GoalIngressContext::trusted(
        ingress.session_key().clone(),
        ingress.agent(),
        ingress.route(),
        GoalIngressPrincipal::Matrix {
            raw_mxid: "@other:example.test".into(),
        },
    )
    .unwrap();
    assert!(driver.try_turn_boundary(&stale).await.is_err());
    assert!(
        driver
            .await_turn_boundary(&stale, &GoalParentTurnCheckpoint::new())
            .await
            .is_err()
    );
    assert!(foreground.try_lock().is_ok());
}

#[tokio::test]
async fn matrix_goal_queued_acknowledgement_failure_abandons_without_transition() {
    queued_control_abandonment_fixture(QueuedControlExit::UnavailableChannel).await;
}

#[tokio::test]
async fn matrix_goal_queued_send_error_abandons_without_transition() {
    queued_control_abandonment_fixture(QueuedControlExit::SendFailure).await;
}

#[tokio::test]
async fn matrix_goal_dropped_queued_control_abandons_without_transition() {
    queued_control_abandonment_fixture(QueuedControlExit::Dropped).await;
}

#[tokio::test]
async fn matrix_goal_aborted_queued_wait_abandons_without_transition() {
    queued_control_abandonment_fixture(QueuedControlExit::AbortedWait).await;
}

enum QueuedControlExit {
    UnavailableChannel,
    SendFailure,
    Dropped,
    AbortedWait,
}

async fn queued_control_abandonment_fixture(exit: QueuedControlExit) {
    use zeroclaw_commands::goal::{GoalBudgetSelection, GoalCommand};
    use zeroclaw_goal::{GoalQueuedControl, GoalTaskRegistry};
    use zeroclaw_runtime::control_plane::SqliteTaskStore;

    let (driver, ingress, foreground) = boundary_fixture().await;
    let context = Arc::clone(&driver.context);
    let message = driver.message.clone();
    let session = ingress.session_key().durable_id();
    let ordinary_turn = foreground.clone().lock_owned().await;
    let directory = tempfile::tempdir().unwrap();
    let store = Arc::new(SqliteTaskStore::new(directory.path()).unwrap());
    let settings = GoalHostSettings::new(
        true,
        GoalBudgetLimits {
            token_limit: Some(1000),
            cost_limit_usd: None,
        },
        std::process::id(),
        "matrix-control-fixture-boot",
    )
    .unwrap();
    let tracker = Arc::new(
        CostTracker::new(
            zeroclaw_config::schema::CostConfig::default(),
            directory.path(),
        )
        .unwrap(),
    );
    let accounting = Arc::new(
        RuntimeGoalOperationAccountingFactory::new(
            store.clone(),
            tracker,
            context.agent_alias.to_string(),
            Arc::new(ModelProviderGoalUsagePricer::new(Arc::default())),
        )
        .unwrap(),
    );
    let supervisor = Arc::new(GoalExecutionSupervisor::new(Arc::new(
        GoalRuntime::new(store.clone()).execution_engine(accounting),
    )));
    let admission = supervisor
        .admit_control(
            settings.clone(),
            ingress,
            Arc::new(driver),
            GoalCommand::Start {
                budget: GoalBudgetSelection::Defaults,
                objective: "fixture objective must not become a Goal".into(),
            },
        )
        .await
        .unwrap();
    assert!(matches!(
        &admission,
        GoalControlAdmission::Queued {
            response: GoalResponse::Queued(GoalQueuedControl::Start),
            ..
        }
    ));
    assert!(
        store
            .current_goal_for_session(&session)
            .await
            .unwrap()
            .is_none()
    );
    let audit = rusqlite::Connection::open(directory.path().join("control_plane.db")).unwrap();
    let state = || {
        audit
            .query_row(
                "SELECT state, outcome_reason FROM goal_control_intents WHERE session_id = ?1",
                [&session],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
            )
            .unwrap()
    };
    assert_eq!(state(), ("pending".into(), None));
    let slot = Arc::new(tokio::sync::Mutex::new(Some(Arc::clone(&supervisor))));
    let command = MatrixGoalCommandSubmission::Deferred {
        supervisor: Arc::clone(&supervisor),
        supervisor_slot: Arc::clone(&slot),
        settings,
        context: Arc::clone(&context),
        submission: Box::new(admission),
        history_key: session.clone(),
    };

    let expected_reason = match exit {
        QueuedControlExit::UnavailableChannel => {
            // Unavailable native delivery must abandon promptly, not await
            // the ordinary turn or create an unacknowledged Goal.
            tokio::time::timeout(
                std::time::Duration::from_secs(1),
                deliver_matrix_goal_command(command, None, message),
            )
            .await
            .unwrap();
            "delivery_failed"
        }
        QueuedControlExit::SendFailure => {
            let channel = Arc::new(MatrixGoalErrorRecordingChannel {
                fail_sends: true,
                ..Default::default()
            });
            tokio::time::timeout(
                std::time::Duration::from_secs(1),
                deliver_matrix_goal_command(command, Some(channel.clone()), message),
            )
            .await
            .unwrap();
            assert_eq!(
                channel.messages.lock().await.as_slice(),
                [super::super::super::render_goal_response(
                    &GoalResponse::Queued(GoalQueuedControl::Start,)
                )]
            );
            "delivery_failed"
        }
        QueuedControlExit::Dropped => {
            // The tracked task may be dropped before acknowledgement. Its
            // owned continuation must settle the audit intent without ever
            // turning that audit row into lifecycle authority.
            drop(command);
            "control_dropped"
        }
        QueuedControlExit::AbortedWait => {
            let channel = Arc::new(MatrixGoalErrorRecordingChannel::default());
            let delivery_channel = channel.clone();
            let continuation = zeroclaw_spawn::spawn!(async move {
                deliver_matrix_goal_command(command, Some(delivery_channel), message).await;
            });
            tokio::time::timeout(std::time::Duration::from_secs(1), async {
                while channel.messages.lock().await.is_empty() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert_eq!(state(), ("pending".into(), None));
            assert!(
                !continuation.is_finished(),
                "the acknowledged control must still wait for the ordinary turn"
            );
            continuation.abort();
            assert!(continuation.await.unwrap_err().is_cancelled());
            assert_eq!(channel.messages.lock().await.len(), 1);
            "control_dropped"
        }
    };
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            if state() == ("abandoned".into(), Some(expected_reason.into())) {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(
        store
            .current_goal_for_session(&session)
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        foreground.try_lock().is_err(),
        "the ordinary turn still owns its boundary"
    );
    assert!(
        goal_command_lock(&context.persist_locks, &session)
            .try_lock_owned()
            .is_ok()
    );
    assert!(Arc::ptr_eq(
        slot.lock().await.as_ref().unwrap(),
        &supervisor
    ));
    drop(ordinary_turn);
    assert!(foreground.try_lock().is_ok());
}
