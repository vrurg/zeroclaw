//! Tracked Matrix control delivery; waiting never owns command serialization.

use super::*;

pub(in crate::orchestrator) async fn deliver_matrix_goal_command(
    command: MatrixGoalCommandSubmission,
    channel: Option<Arc<dyn Channel>>,
    message: ChannelMessage,
) {
    if let Err(error) = send_response(channel.as_ref(), &message, command.response()).await {
        log_delivery_error(
            "Goal command acknowledgement could not be delivered",
            &error,
        );
        if let Err(error) = command.response_delivery_failed().await {
            log_delivery_error("Goal acknowledgement failure could not be settled", &error);
        }
        return;
    }

    // An idle/immediate command already committed. A queued acknowledgement
    // must be visible before waiting, and that wait must retain no bind lease.
    let MatrixGoalCommandSubmission::Deferred {
        supervisor,
        supervisor_slot,
        mut settings,
        context,
        submission,
        history_key,
    } = command
    else {
        return;
    };
    let submission = match submission {
        GoalControlAdmission::Direct(submission) => GoalControlAdmission::Direct(submission),
        GoalControlAdmission::Queued { deferred, .. } => {
            let ready = match deferred.wait().await {
                Ok(ready) => ready,
                Err(closed) => {
                    // The closed wait's durable response must be delivered,
                    // not interpreted as another lifecycle transition.
                    if let Err(error) =
                        send_response(channel.as_ref(), &message, closed.response()).await
                    {
                        log_delivery_error(
                            "Closed Goal control response could not be delivered",
                            &error,
                        );
                    }
                    log_delivery_error(
                        "Goal deferred control closed before admission",
                        &closed.into(),
                    );
                    return;
                }
            };
            settings = match matrix_goal_settings(context.as_ref(), &history_key) {
                Ok(settings) => settings,
                Err(error) => {
                    drop(ready);
                    log_delivery_error("Goal deferred settings could not be resolved", &error);
                    send_command_failed(channel.as_ref(), &message).await;
                    return;
                }
            };
            let commit = match ready.commit(settings.clone()).await {
                Ok(commit) => commit,
                Err(error) => {
                    log_delivery_error("Goal deferred command could not commit", &error);
                    send_command_failed(channel.as_ref(), &message).await;
                    return;
                }
            };
            let before_release = matches!(
                commit.response(),
                GoalResponse::Started(_) | GoalResponse::Resumed(_)
            );
            let delivery = if before_release {
                send_response(channel.as_ref(), &message, commit.response()).await
            } else {
                Ok(())
            };
            // Release checkpoint and command ownership before draining a
            // predecessor. Budget and refused/stale outcomes publish afterwards.
            let command = MatrixGoalCommandSubmission::Deferred {
                supervisor,
                supervisor_slot,
                settings,
                context,
                submission: GoalControlAdmission::Direct(commit.release()),
                history_key,
            };
            if let Err(error) = delivery {
                log_delivery_error(
                    "Goal deferred launch response could not be delivered",
                    &error,
                );
                if let Err(error) = command.response_delivery_failed().await {
                    log_delivery_error("Goal deferred launch failure could not be settled", &error);
                }
                return;
            }
            finish_after_response(command, channel.as_ref(), &message, !before_release).await;
            return;
        }
    };
    let command = MatrixGoalCommandSubmission::Deferred {
        supervisor,
        supervisor_slot,
        settings,
        context,
        submission,
        history_key,
    };
    finish_after_response(command, channel.as_ref(), &message, false).await;
}

async fn finish_after_response(
    mut command: MatrixGoalCommandSubmission,
    channel: Option<&Arc<dyn Channel>>,
    message: &ChannelMessage,
    publish_initial: bool,
) {
    match command.after_response().await {
        Ok(late) => {
            let response = late
                .as_ref()
                .or_else(|| publish_initial.then(|| command.response()));
            if let Some(response) = response
                && let Err(error) = send_response(channel, message, response).await
            {
                log_delivery_error("Final Goal command response could not be delivered", &error);
                return;
            }
            command.final_response_delivered(late.as_ref()).await;
        }
        Err(error) => {
            log_delivery_error("Goal command launch or drain could not finish", &error);
            if !error.is::<zeroclaw_goal::GoalLaunchFailureReported>() {
                send_command_failed(channel, message).await;
            }
        }
    }
}

async fn send_response(
    channel: Option<&Arc<dyn Channel>>,
    message: &ChannelMessage,
    response: &GoalResponse,
) -> Result<()> {
    let channel = channel.context("Matrix channel is unavailable for Goal response")?;
    channel
        .send(&SendMessage::reply_to(
            message,
            super::super::render_goal_response(response),
        ))
        .await
}

pub(super) async fn send_command_failed(
    channel: Option<&Arc<dyn Channel>>,
    message: &ChannelMessage,
) {
    if let Some(channel) = channel
        && let Err(error) = channel
            .send(&SendMessage::reply_to(
                message,
                super::super::channel_runtime_cli_string("goal-mode-command-failed"),
            ))
            .await
    {
        log_delivery_error(
            "Goal command failure response could not be delivered",
            &error,
        );
    }
}

fn log_delivery_error(context: &str, error: &anyhow::Error) {
    ::zeroclaw_log::record!(
        WARN,
        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
            .with_attrs(serde_json::json!({
                "error": zeroclaw_providers::sanitize_api_error(&RuntimeGoalDiagnosticTextGuard.guard_diagnostic_text(&format!("{error:#}"))),
            })),
        context
    );
}
