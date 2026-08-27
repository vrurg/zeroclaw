//! The per-tool-call approval gate: CLI prompt, channel inline approval, or
//! auto-deny, plus decision recording.

use super::context::TurnCtx;
use super::events::StreamDelta;
use super::redact::scrub_credentials;
use crate::agent::tool_execution::ToolExecutionOutcome;
use crate::approval::{ApprovalRequest, ApprovalRequirement, ApprovalResponse};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::time::Duration;

pub(crate) enum ApprovalGateOutcome {
    Proceed { approved: bool },
    Deny(ToolExecutionOutcome),
    Replace(ToolExecutionOutcome),
}

/// Run the approval flow for one tool call (upstream loop body, approval
/// section): resolve the tool's approval requirement, prompt interactively on
/// CLI or via the channel's inline approval on non-interactive channels
/// (falling back to auto-deny), and record the decision.
pub(crate) async fn gate_tool_approval(
    ctx: &TurnCtx<'_>,
    tool_name: &str,
    tool_args: &serde_json::Value,
    iteration: usize,
) -> ApprovalGateOutcome {
    if is_session_prompt_mutation(tool_name) && ctx.session_prompt_approval_required {
        return gate_session_prompt_approval(ctx, tool_name, tool_args).await;
    }

    let mut approval_requirement = ctx
        .approval
        .map(|mgr| mgr.approval_requirement(tool_name))
        .unwrap_or(ApprovalRequirement::NotRequired);
    if let Some(mgr) = ctx.approval
        && approval_requirement == ApprovalRequirement::Prompt
    {
        let request = ApprovalRequest {
            tool_name: tool_name.to_string(),
            arguments: tool_args.clone(),
        };

        // Interactive CLI: prompt the operator.
        // Non-interactive (channels): try the channel's inline
        // approval (e.g. Telegram inline keyboard) before falling
        // back to auto-deny.
        let (decision, decided_by, unanswerable) = if mgr.is_non_interactive() {
            let attributed = if let Some(ch) = ctx.channel {
                let ch_request = zeroclaw_api::channel::ChannelApprovalRequest {
                    tool_name: request.tool_name.clone(),
                    arguments_summary: crate::approval::summarize_args(&request.arguments),
                    raw_arguments: Some(request.arguments.clone()),
                };
                let recipient = ctx.channel_reply_target.unwrap_or_default();
                match ch.request_approval_attributed(recipient, &ch_request).await {
                    Ok(Some(a)) => Some(a),
                    Ok(None) => None,
                    Err(e) => {
                        ::zeroclaw_log::record!(
                            WARN,
                            ::zeroclaw_log::Event::new(
                                module_path!(),
                                ::zeroclaw_log::Action::Fail
                            )
                            .with_category(::zeroclaw_log::EventCategory::Tool)
                            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                            .with_attrs(::serde_json::json!({"error": format!("{}", e)})),
                            "Channel approval request failed"
                        );
                        None
                    }
                }
            } else {
                None
            };
            // The deciding back-channel (when a fan-out bridge answered) rides
            // back on the response itself, so attribution can't be cross-wired
            // by a concurrent approval on the same channel instance.
            let decided_by = attributed.as_ref().and_then(|a| a.decided_by.clone());
            // Whether an operator actually decided, taken from the response's own
            // provenance rather than inferred.
            //
            // `attributed.is_none()` is NOT sufficient: a fail-closed approval route
            // returns `Some(Deny)` with no decider when the approver is missing,
            // unreachable, silent, or timed out, and a direct channel timeout does the
            // same. Those are runtime denials wearing an operator's clothes. Nor does
            // `decided_by.is_none()` work, since a single non-fan-out channel leaves
            // that `None` for a real human answer.
            let unanswerable = attributed
                .as_ref()
                .map(|a| a.source.is_runtime_fail_closed())
                .unwrap_or(true);
            let decision = match attributed.map(|a| a.response) {
                Some(zeroclaw_api::channel::ChannelApprovalResponse::Approve) => {
                    ApprovalResponse::Yes
                }
                Some(zeroclaw_api::channel::ChannelApprovalResponse::AlwaysApprove) => {
                    ApprovalResponse::Always
                }
                Some(zeroclaw_api::channel::ChannelApprovalResponse::Deny) => ApprovalResponse::No,
                Some(zeroclaw_api::channel::ChannelApprovalResponse::DenyWithEdit {
                    replacement,
                }) => ApprovalResponse::ReplaceWith(replacement),
                // Channel doesn't support approval — auto-deny.
                None => ApprovalResponse::No,
            };
            (decision, decided_by, unanswerable)
        } else {
            (mgr.prompt_cli(&request), None, false)
        };

        let decision_channel = decided_by.unwrap_or_else(|| ctx.channel_name.to_string());
        mgr.record_decision(tool_name, tool_args, &decision, &decision_channel);

        if decision == ApprovalResponse::No {
            // This string is fed back to the MODEL, so it states the outcome and
            // stops there. It deliberately does not name the settings that would
            // permit the call: `auto_approve` bypasses operator approval for that
            // tool and `level = "full"` removes approval gates for every tool and
            // drops workspace-only confinement. Putting that remedy in front of the
            // model invites it to argue for expanding its own privileges, which is a
            // disproportionate response to an approval channel being unavailable.
            // Operators get the actionable advice through the WARN record below and
            // the UI, where changing policy is actually their decision to make.
            let denied = if unanswerable {
                format!(
                    "Tool call not executed: '{tool_name}' requires approval and no operator \
                     decision was available, so the runtime denied it by policy. This was not \
                     a user's decision."
                )
            } else {
                "Denied by user.".to_string()
            };
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                    .with_category(::zeroclaw_log::EventCategory::Tool)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(::serde_json::json!({
                        "model": ctx.model,
                        "iteration": iteration + 1,
                        "tool": tool_name,
                        "arguments": scrub_credentials(&tool_args.to_string()),
                        "result": denied,
                        "trace_id": ctx.turn_id,
                        // Operator-facing only. The remedy lives here rather than
                        // in `result`, which is shown to the model: deciding to
                        // relax an approval policy is the operator's call, and
                        // putting the option in front of the model would invite it
                        // to lobby for its own privilege expansion.
                        "denied_by_runtime": unanswerable,
                        "operator_hint": if unanswerable {
                            Some("No operator could be asked. Check that an approval-capable \
                                  channel is connected and that the agent's approval route names \
                                  a registered, reachable approver. If this tool should run \
                                  unattended, review the agent's risk profile deliberately.")
                        } else {
                            None
                        },
                    })),
                "tool_call_result"
            );
            if let Some(tx) = ctx.on_delta {
                let _ = tx
                    .send(StreamDelta::Status(format!(
                        "\u{274c} {}: {}\n",
                        tool_name, denied
                    )))
                    .await;
            }
            return ApprovalGateOutcome::Deny(ToolExecutionOutcome {
                output: denied.clone(),
                success: false,
                error_reason: Some(denied),
                duration: Duration::ZERO,
                receipt: None,
                output_data: None,
            });
        }

        if let ApprovalResponse::ReplaceWith(replacement) = &decision {
            if let Some(tx) = ctx.on_delta {
                let _ = tx
                    .send(StreamDelta::Status(format!(
                        "\u{270f} {}: replaced by user\n",
                        tool_name
                    )))
                    .await;
            }
            ::zeroclaw_log::record!(
                INFO,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Approve)
                    .with_category(::zeroclaw_log::EventCategory::Tool)
                    .with_outcome(::zeroclaw_log::EventOutcome::Success)
                    .with_attrs(::serde_json::json!({
                        "model": ctx.model,
                        "iteration": iteration + 1,
                        "tool": tool_name,
                        "arguments": scrub_credentials(&tool_args.to_string()),
                        "replaced": true,
                        "output": scrub_credentials(replacement),
                        "trace_id": ctx.turn_id,
                    })),
                "tool_call_result"
            );
            return ApprovalGateOutcome::Replace(ToolExecutionOutcome {
                output: crate::approval::sanitize_tool_replacement(replacement),
                success: true,
                error_reason: None,
                duration: Duration::ZERO,
                receipt: None,
                output_data: None,
            });
        }

        if matches!(decision, ApprovalResponse::Yes | ApprovalResponse::Always) {
            approval_requirement = ApprovalRequirement::Approved;
        }
    }

    ApprovalGateOutcome::Proceed {
        approved: approval_requirement == ApprovalRequirement::Approved,
    }
}

fn is_session_prompt_mutation(tool_name: &str) -> bool {
    matches!(tool_name, "session_prompt_set" | "session_prompt_delete")
}

fn session_prompt_approval_summary(
    tool_name: &str,
    tool_args: &serde_json::Value,
) -> Result<String, &'static str> {
    let session_id = zeroclaw_api::TOOL_LOOP_SESSION_KEY
        .try_with(Clone::clone)
        .ok()
        .flatten()
        .ok_or("no active chat session is available for confirmation")?;
    let raw_id = tool_args
        .get("id")
        .and_then(serde_json::Value::as_str)
        .ok_or("the attachment id is missing")?;
    // Storage accepts only this canonical representation. Validate before the
    // operator sees the binding so approval and execution cannot disagree.
    let id = zeroclaw_infra::session_prompts::validate_prompt_id(raw_id)
        .map_err(|_| "the attachment id is invalid")?;
    let action = if tool_name == "session_prompt_set" {
        "set"
    } else {
        "delete"
    };
    let mut summary = String::from(
        "Approve this one persistent session-prompt mutation. This approval cannot be remembered.\n",
    );
    let _ = writeln!(summary, "action: {action}");
    let _ = writeln!(summary, "storage_domain: sqlite chat session prompts");
    let _ = writeln!(summary, "session_id: {session_id}");
    let _ = writeln!(summary, "attachment_id: {id}");
    if action == "set" {
        let content = tool_args
            .get("content")
            .and_then(serde_json::Value::as_str)
            .ok_or("the prompt content is missing")?;
        let digest = Sha256::digest(content.as_bytes());
        let _ = writeln!(summary, "content_sha256: {digest:x}");
        let _ = writeln!(summary, "content:");
        summary.push_str(content);
    }
    Ok(summary)
}

async fn gate_session_prompt_approval(
    ctx: &TurnCtx<'_>,
    tool_name: &str,
    tool_args: &serde_json::Value,
) -> ApprovalGateOutcome {
    let denied = |reason: &str| {
        ApprovalGateOutcome::Deny(ToolExecutionOutcome {
            output: format!("Session prompt mutation not executed: {reason}"),
            success: false,
            error_reason: Some("session prompt mutation denied".to_string()),
            duration: Duration::ZERO,
            receipt: None,
            output_data: None,
        })
    };
    let Ok(summary) = session_prompt_approval_summary(tool_name, tool_args) else {
        return denied("the runtime could not bind an exact session confirmation");
    };
    let Some(mgr) = ctx.approval else {
        return denied("no approval manager is available");
    };

    let approved = if mgr.is_non_interactive() {
        let Some(channel) = ctx.channel else {
            return denied("no approval-capable channel is available");
        };
        let request = zeroclaw_api::channel::ChannelApprovalRequest {
            tool_name: tool_name.to_string(),
            arguments_summary: summary,
            // Prompt content belongs only on the approval surface, never in the
            // generic structured arguments that downstream event consumers log.
            raw_arguments: None,
        };
        match channel
            .request_approval_attributed(ctx.channel_reply_target.unwrap_or_default(), &request)
            .await
        {
            Ok(Some(attributed)) => is_one_time_session_prompt_approval(&attributed),
            Ok(_) | Err(_) => false,
        }
    } else {
        mgr.prompt_cli_once(
            &crate::i18n::get_required_cli_string("session-prompt-approval-heading"),
            &summary,
        )
    };

    if approved {
        ApprovalGateOutcome::Proceed { approved: true }
    } else {
        denied("a one-time operator approval was not granted")
    }
}

fn is_one_time_session_prompt_approval(
    approval: &zeroclaw_api::channel::AttributedApprovalResponse,
) -> bool {
    !approval.source.is_runtime_fail_closed()
        && matches!(
            approval.response,
            zeroclaw_api::channel::ChannelApprovalResponse::Approve
        )
}

#[cfg(test)]
mod tests {
    use super::{is_one_time_session_prompt_approval, session_prompt_approval_summary};
    use zeroclaw_api::channel::{
        ApprovalSource, AttributedApprovalResponse, ChannelApprovalResponse,
    };

    #[tokio::test]
    async fn session_prompt_confirmation_binds_session_id_content_and_digest() {
        let summary = zeroclaw_api::TOOL_LOOP_SESSION_KEY
            .scope(
                Some("matrix:room:thread".to_string()),
                async {
                    session_prompt_approval_summary(
                        "session_prompt_set",
                        &serde_json::json!({"id": "current-task", "content": "Finish RFC reconciliation."}),
                    )
                    .unwrap()
                },
            )
            .await;

        assert!(summary.contains("action: set"));
        assert!(summary.contains("storage_domain: sqlite chat session prompts"));
        assert!(summary.contains("session_id: matrix:room:thread"));
        assert!(summary.contains("attachment_id: current-task"));
        assert!(summary.contains("content:\nFinish RFC reconciliation."));
        assert!(summary.contains(
            "content_sha256: 16e48f498e379a0e5530eb194069ef5ce3f2133b53b6bdd28c80472425e552de"
        ));
    }

    #[test]
    fn session_prompt_confirmation_rejects_persistent_and_runtime_decisions() {
        assert!(is_one_time_session_prompt_approval(
            &AttributedApprovalResponse::operator(ChannelApprovalResponse::Approve)
        ));
        assert!(!is_one_time_session_prompt_approval(
            &AttributedApprovalResponse::operator(ChannelApprovalResponse::AlwaysApprove)
        ));
        assert!(!is_one_time_session_prompt_approval(
            &AttributedApprovalResponse::from_runtime(
                ChannelApprovalResponse::Approve,
                ApprovalSource::TimedOut,
            )
        ));
    }

    #[tokio::test]
    async fn session_prompt_confirmation_canonicalizes_attachment_ids() {
        let result = zeroclaw_api::TOOL_LOOP_SESSION_KEY
            .scope(Some("matrix:room:thread".to_string()), async {
                session_prompt_approval_summary(
                    "session_prompt_delete",
                    &serde_json::json!({"id": " task "}),
                )
            })
            .await;
        assert!(result.unwrap().contains("attachment_id: task"));
    }
}
