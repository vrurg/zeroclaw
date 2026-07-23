//! The per-tool-call approval gate: CLI prompt, channel inline approval, or
//! auto-deny, plus decision recording.

use super::context::TurnCtx;
use super::events::StreamDelta;
use super::redact::scrub_credentials;
use crate::agent::tool_execution::{ToolExecutionOutcome, scrub_tool_arguments_for_presentation};
use crate::approval::{ApprovalRequest, ApprovalRequirement, ApprovalResponse};
use std::time::Duration;

pub(crate) enum ApprovalGateOutcome {
    Proceed { approved: bool },
    Deny(ToolExecutionOutcome),
    Replace(ToolExecutionOutcome),
}

fn approval_request_for_presentation(
    tool_name: &str,
    tool_args: &serde_json::Value,
    leak_detection: &zeroclaw_config::schema::LeakDetectionConfig,
) -> ApprovalRequest {
    ApprovalRequest {
        tool_name: tool_name.to_string(),
        arguments: scrub_tool_arguments_for_presentation(tool_args, leak_detection),
    }
}

fn approval_decision_for_audit(
    decision: &ApprovalResponse,
    leak_detection: &zeroclaw_config::schema::LeakDetectionConfig,
) -> ApprovalResponse {
    match decision {
        ApprovalResponse::ReplaceWith(replacement) => ApprovalResponse::ReplaceWith(
            crate::security::scrub_with_config(&scrub_credentials(replacement), leak_detection),
        ),
        other => other.clone(),
    }
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
    let mut approval_requirement = ctx
        .approval
        .map(|mgr| mgr.approval_requirement(tool_name))
        .unwrap_or(ApprovalRequirement::NotRequired);
    if let Some(mgr) = ctx.approval
        && approval_requirement == ApprovalRequirement::Prompt
    {
        let request = approval_request_for_presentation(tool_name, tool_args, ctx.leak_detection);

        // Interactive CLI: prompt the operator.
        // Non-interactive (channels): try the channel's inline
        // approval (e.g. Telegram inline keyboard) before falling
        // back to auto-deny.
        let (decision, decided_by) = if mgr.is_non_interactive() {
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
            (decision, decided_by)
        } else {
            (mgr.prompt_cli(&request), None)
        };

        let decision_channel = decided_by.unwrap_or_else(|| ctx.channel_name.to_string());
        let audit_decision = approval_decision_for_audit(&decision, ctx.leak_detection);
        mgr.record_decision(
            tool_name,
            &request.arguments,
            &audit_decision,
            &decision_channel,
        );

        if decision == ApprovalResponse::No {
            let denied = "Denied by user.".to_string();
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                    .with_category(::zeroclaw_log::EventCategory::Tool)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(::serde_json::json!({
                        "model": ctx.model,
                        "iteration": iteration + 1,
                        "tool": tool_name,
                        "arguments": request.arguments,
                        "result": denied,
                        "trace_id": ctx.turn_id,
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
            let ApprovalResponse::ReplaceWith(audit_replacement) = &audit_decision else {
                unreachable!("audit decision must preserve the response variant");
            };
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
                        "arguments": request.arguments,
                        "replaced": true,
                        "output": audit_replacement,
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

#[cfg(test)]
mod tests {
    use super::{approval_decision_for_audit, approval_request_for_presentation};
    use crate::approval::{ApprovalManager, ApprovalResponse};
    use crate::security::AutonomyLevel;
    use zeroclaw_config::schema::{LeakDetectionConfig, RiskProfileConfig};

    #[test]
    fn approval_presentation_scrubs_configured_patterns_before_audit_or_delivery() {
        let credential = "AKIAIOSFODNN7EXAMPLE";
        let request = approval_request_for_presentation(
            "shell",
            &serde_json::json!({
                "command": format!("echo {credential}"),
                "content": credential,
            }),
            &LeakDetectionConfig::default(),
        );
        let serialized = request.arguments.to_string();
        assert!(!serialized.contains(credential));
        assert!(serialized.contains("[REDACTED"));

        let manager = ApprovalManager::from_risk_profile(&RiskProfileConfig {
            level: AutonomyLevel::Supervised,
            ..RiskProfileConfig::default()
        });
        manager.record_decision(
            &request.tool_name,
            &request.arguments,
            &ApprovalResponse::No,
            "test",
        );
        let audit = manager.audit_log();
        assert_eq!(audit.len(), 1);
        assert!(!audit[0].arguments_summary.contains(credential));
        assert!(audit[0].arguments_summary.contains("[REDACT"));
    }

    #[test]
    fn replacement_decision_is_scrubbed_for_audit_without_changing_execution_copy() {
        let credential = "AKIAIOSFODNN7EXAMPLE";
        let decision = ApprovalResponse::ReplaceWith(format!("denied: {credential}"));
        let audit_decision =
            approval_decision_for_audit(&decision, &LeakDetectionConfig::default());

        let ApprovalResponse::ReplaceWith(audit_text) = audit_decision else {
            panic!("replacement decision changed variant");
        };
        assert!(!audit_text.contains(credential));
        assert!(audit_text.contains("[REDACTED"));
        assert_eq!(
            decision,
            ApprovalResponse::ReplaceWith(format!("denied: {credential}"))
        );
    }
}
