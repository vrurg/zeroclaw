//! The per-call preparation loop: `before_tool_call` hook, delivery defaults,
//! the approval gate, the duplicate-call gate, and start logging — producing
//! the executable subset of this round's tool calls.

use super::approval_gate::{ApprovalGateOutcome, gate_tool_approval};
use super::context::TurnCtx;
use super::delivery_defaults::maybe_inject_channel_delivery_defaults;
use super::events::{StreamDelta, emit_tool_call_pair};
use super::outcome::ToolLoopCancelled;
use super::redact::scrub_credentials;
use crate::agent::tool_execution::{
    PreparedToolCall, ToolExecutionOutcome, scrub_tool_arguments_for_presentation,
};
use crate::util::truncate_with_ellipsis;
use anyhow::Result;
use std::collections::HashSet;
use std::time::Duration;
use zeroclaw_tool_call_parser::{ParsedToolCall, canonicalize_json_for_tool_signature};

pub(crate) struct PreparedToolCalls {
    pub(crate) ordered_results: Vec<Option<(String, Option<String>, ToolExecutionOutcome)>>,
    pub(crate) executable_indices: Vec<usize>,
    pub(crate) executable_calls: Vec<PreparedToolCall>,
}

fn tool_call_signature(tool_name: &str, tool_args: &serde_json::Value) -> (String, String) {
    let canonical_args = canonicalize_json_for_tool_signature(tool_args);
    let args_json = serde_json::to_string(&canonical_args).unwrap_or_else(|_| "{}".to_string());
    (tool_name.trim().to_ascii_lowercase(), args_json)
}

fn tool_start_progress(
    tool_name: &str,
    arguments: &serde_json::Value,
    leak_detection: &zeroclaw_config::schema::LeakDetectionConfig,
) -> String {
    let hint = {
        let value = match tool_name {
            "shell" => arguments.get("command").and_then(|value| value.as_str()),
            "file_read" | "file_write" => arguments.get("path").and_then(|value| value.as_str()),
            _ => arguments
                .get("action")
                .and_then(|value| value.as_str())
                .or_else(|| arguments.get("query").and_then(|value| value.as_str())),
        };
        value.map_or_else(String::new, |text| truncate_with_ellipsis(text, 60))
    };
    let progress = if hint.is_empty() {
        format!("\u{23f3} {tool_name}\n")
    } else {
        format!("\u{23f3} {tool_name}: {hint}\n")
    };
    crate::security::scrub_with_config(&scrub_credentials(&progress), leak_detection)
}

async fn record_duplicate_tool_call(
    ctx: &TurnCtx<'_>,
    tool_name: &str,
    tool_args: &serde_json::Value,
    iteration: usize,
) -> ToolExecutionOutcome {
    let duplicate =
        format!("Skipped duplicate tool call '{tool_name}' with identical arguments in this turn.");
    ::zeroclaw_log::record!(
        INFO,
        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Skip)
            .with_category(::zeroclaw_log::EventCategory::Tool)
            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
            .with_attrs(::serde_json::json!({
                "model": ctx.model,
                "iteration": iteration + 1,
                "tool": tool_name,
                "arguments": scrub_tool_arguments_for_presentation(
                    tool_args,
                    ctx.leak_detection
                ).to_string(),
                "result": duplicate,
                "deduplicated": true,
                "trace_id": ctx.turn_id,
            })),
        "tool_call_result"
    );
    if let Some(tx) = ctx.on_delta {
        let _ = tx
            .send(StreamDelta::Status(format!(
                "\u{274c} {}: {}\n",
                tool_name, duplicate
            )))
            .await;
    }
    ToolExecutionOutcome {
        output: duplicate.clone(),
        success: false,
        error_reason: Some(duplicate),
        duration: Duration::ZERO,
        receipt: None,
        output_data: None,
    }
}

/// Run per-call preparation over this round's parsed tool calls (upstream
/// loop body, per-call prep loop).
pub(crate) async fn prepare_tool_calls(
    ctx: &TurnCtx<'_>,
    tool_calls: &[ParsedToolCall],
    seen_tool_signatures: &mut HashSet<(String, String)>,
    prompt_approval_tool_signatures: &mut HashSet<(String, String)>,
    iteration: usize,
    dedup_enabled: bool,
) -> Result<PreparedToolCalls> {
    let mut ordered_results: Vec<Option<(String, Option<String>, ToolExecutionOutcome)>> =
        (0..tool_calls.len()).map(|_| None).collect();
    let mut executable_indices: Vec<usize> = Vec::new();
    let mut executable_calls: Vec<PreparedToolCall> = Vec::new();
    let mut prompt_approval_tool_signatures_this_round: HashSet<(String, String)> = HashSet::new();

    for (idx, call) in tool_calls.iter().enumerate() {
        // ── Hook: before_tool_call (modifying) ──────────
        let mut tool_name = call.name.clone();
        let mut tool_args = call.arguments.clone();
        if let Some(hooks) = ctx.hooks {
            match hooks
                .run_before_tool_call(tool_name.clone(), tool_args.clone())
                .await
            {
                crate::hooks::HookResult::Cancel(reason) => {
                    ::zeroclaw_log::record!(INFO, ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Cancel).with_category(::zeroclaw_log::EventCategory::Tool).with_attrs(::serde_json::json!({"tool": call.name, "reason": reason.to_string()})), "tool call cancelled by hook");
                    let cancelled = format!("Cancelled by hook: {reason}");
                    ::zeroclaw_log::record!(
                        WARN,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Cancel)
                            .with_category(::zeroclaw_log::EventCategory::Tool)
                            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                            .with_attrs(::serde_json::json!({
                                "model": ctx.model,
                                "iteration": iteration + 1,
                                "tool": call.name,
                                "arguments": scrub_tool_arguments_for_presentation(
                                    &tool_args,
                                    ctx.leak_detection
                                ).to_string(),
                                "result": cancelled,
                                "trace_id": ctx.turn_id,
                            })),
                        "tool_call_result"
                    );
                    if let Some(tx) = ctx.on_delta {
                        let _ = tx
                            .send(StreamDelta::Status(format!(
                                "\u{274c} {}: {}\n",
                                call.name,
                                truncate_with_ellipsis(&scrub_credentials(&cancelled), 200)
                            )))
                            .await;
                    }
                    let outcome = ToolExecutionOutcome {
                        output: cancelled,
                        success: false,
                        error_reason: Some(reason),
                        duration: Duration::ZERO,
                        receipt: None,
                        output_data: None,
                    };
                    // Streaming consumers still see the call and its
                    // hook-cancel outcome as a ToolCall/ToolResult pair,
                    // as the direct execution path always emitted.
                    if let Some(tx) = ctx.event_tx {
                        emit_tool_call_pair(tx, call, &outcome, ctx.leak_detection).await;
                    }
                    ordered_results[idx] =
                        Some((call.name.clone(), call.tool_call_id.clone(), outcome));
                    continue;
                }
                crate::hooks::HookResult::Continue((name, args)) => {
                    tool_name = name;
                    tool_args = args;
                }
            }
        }

        maybe_inject_channel_delivery_defaults(
            &tool_name,
            &mut tool_args,
            ctx.channel_name,
            ctx.channel_reply_target,
        );

        crate::agent::set_runtime_approved_arg(&tool_name, &mut tool_args, false);

        let requires_prompt = ctx
            .approval
            .map(|mgr| mgr.needs_approval(&tool_name))
            .unwrap_or(false);
        let reentrant_agent_tool =
            crate::tools::REENTRANT_AGENT_TOOLS.contains(&tool_name.as_str());
        if requires_prompt && tool_name == "shell" && !reentrant_agent_tool {
            let prompt_signature = tool_call_signature(&tool_name, &tool_args);
            if !prompt_approval_tool_signatures_this_round.insert(prompt_signature.clone()) {
                let duplicate =
                    record_duplicate_tool_call(ctx, &tool_name, &tool_args, iteration).await;
                ordered_results[idx] =
                    Some((tool_name.clone(), call.tool_call_id.clone(), duplicate));
                continue;
            }
            if !prompt_approval_tool_signatures.insert(prompt_signature) {
                let repeated = format!(
                    "Agent loop aborted: repeated prompt-required tool call '{tool_name}' with identical arguments before approval."
                );
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Fail)
                        .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                        .with_attrs(::serde_json::json!({
                            "model": ctx.model,
                            "iteration": iteration + 1,
                            "tool": tool_name.clone(),
                            "arguments": scrub_tool_arguments_for_presentation(
                                &tool_args,
                                ctx.leak_detection
                            ).to_string(),
                            "result": repeated,
                            "trace_id": ctx.turn_id,
                        })),
                    "tool_call_result"
                );
                if let Some(tx) = ctx.on_delta {
                    let _ = tx
                        .send(StreamDelta::Status(format!(
                            "\u{274c} {}: {}\n",
                            tool_name, repeated
                        )))
                        .await;
                }
                anyhow::bail!("{repeated}");
            }
        }

        // ── Approval hook ────────────────────────────────
        let approved = match gate_tool_approval(ctx, &tool_name, &tool_args, iteration).await {
            ApprovalGateOutcome::Proceed { approved } => approved,
            ApprovalGateOutcome::Deny(outcome) | ApprovalGateOutcome::Replace(outcome) => {
                // Streaming consumers see the denied/replaced call and its
                // synthesized result (e.g. a DenyWithEdit replacement) as a
                // ToolCall/ToolResult pair, as the direct path always did.
                if let Some(tx) = ctx.event_tx {
                    emit_tool_call_pair(tx, call, &outcome, ctx.leak_detection).await;
                }
                ordered_results[idx] =
                    Some((tool_name.clone(), call.tool_call_id.clone(), outcome));
                continue;
            }
            ApprovalGateOutcome::Cancel => return Err(ToolLoopCancelled.into()),
        };
        crate::agent::set_runtime_approved_arg(&tool_name, &mut tool_args, approved);

        let signature = tool_call_signature(&tool_name, &tool_args);
        let dedup_exempt =
            ctx.dedup_exempt_tools.iter().any(|e| e == &tool_name) || reentrant_agent_tool;
        if dedup_enabled && !dedup_exempt && !seen_tool_signatures.insert(signature) {
            let duplicate =
                record_duplicate_tool_call(ctx, &tool_name, &tool_args, iteration).await;
            ordered_results[idx] = Some((tool_name.clone(), call.tool_call_id.clone(), duplicate));
            continue;
        }

        let presentation_arguments =
            scrub_tool_arguments_for_presentation(&tool_args, ctx.leak_detection);
        ::zeroclaw_log::record!(
            INFO,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Start)
                .with_category(::zeroclaw_log::EventCategory::Tool)
                .with_attrs(::serde_json::json!({
                    "model": ctx.model,
                    "iteration": iteration + 1,
                    "tool": tool_name.clone(),
                    "arguments": presentation_arguments.to_string(),
                    "trace_id": ctx.turn_id,
                })),
            "tool_call_start"
        );

        // ── Progress: tool start ────────────────────────────
        if let Some(tx) = ctx.on_delta {
            let progress =
                tool_start_progress(&tool_name, &presentation_arguments, ctx.leak_detection);
            ::zeroclaw_log::record!(
                DEBUG,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                    .with_category(::zeroclaw_log::EventCategory::Tool)
                    .with_attrs(::serde_json::json!({"tool": tool_name})),
                "Sending progress start to draft"
            );
            let _ = tx.send(StreamDelta::Status(progress)).await;
        }

        executable_indices.push(idx);
        let call_id = super::events::resolve_tool_call_id(&ParsedToolCall {
            name: tool_name.clone(),
            arguments: tool_args.clone(),
            tool_call_id: call.tool_call_id.clone(),
        });
        // Pin the resolved id onto the executable call so the pending ToolCall
        // and the terminal ToolResult (both emitted by the executor at dispatch
        // and completion) share one correlation id, even for id-less
        // text-protocol calls.
        executable_calls.push(PreparedToolCall::new(
            ParsedToolCall {
                name: tool_name,
                arguments: tool_args,
                tool_call_id: Some(call_id),
            },
            presentation_arguments,
        ));
    }

    Ok(PreparedToolCalls {
        ordered_results,
        executable_indices,
        executable_calls,
    })
}

#[cfg(test)]
mod tests {
    use super::tool_start_progress;
    use crate::agent::tool_execution::scrub_tool_arguments_for_presentation;
    use zeroclaw_config::schema::LeakDetectionConfig;

    #[test]
    fn tool_start_progress_scrubs_arguments_before_truncation_and_final_display() {
        let configured_credential = "AKIAIOSFODNN7EXAMPLE";
        let legacy_credential = "token=abcdefghijklmnop";
        let raw = serde_json::json!({
            "command": format!("{configured_credential} {legacy_credential}"),
        });

        let config = LeakDetectionConfig::default();
        let arguments = scrub_tool_arguments_for_presentation(&raw, &config);
        let progress = tool_start_progress("shell", &arguments, &config);

        for output in [arguments.to_string(), progress] {
            assert!(!output.contains(configured_credential));
            assert!(!output.contains("abcdefghijklmnop"));
            assert!(output.contains("[REDACTED"));
        }
    }
}
