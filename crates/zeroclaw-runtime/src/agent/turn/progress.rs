//! Human-facing progress lines for the streamed turn status surface.
//!
//! The raw tool call remains the source of truth. These helpers derive a short
//! display-only summary at emission time so start and completion lines stay
//! correlated without storing duplicate call metadata.

use super::redact::scrub_credentials;
use crate::util::truncate_with_ellipsis;
use serde_json::Value;

const MAX_PROGRESS_VALUE_CHARS: usize = 60;
const TOOL_ARGUMENT_HINT_KEYS: &[&str] = &["command", "path", "action", "query"];

/// Render the start line for a tool call using the same argument hint that
/// completion lines use.
pub(crate) fn render_tool_start_progress(tool: &str, args: &Value) -> String {
    match tool_argument_hint(args) {
        Some(hint) => format!("\u{23f3} {tool}: {hint}\n"),
        None => format!("\u{23f3} {tool}\n"),
    }
}

/// Render the completion line for a tool call. Failures include the same
/// argument hint as successes, plus scrubbed error text when available.
pub(crate) fn render_tool_completion_progress(
    tool: &str,
    args: &Value,
    secs: u64,
    success: bool,
    error_reason: Option<&str>,
) -> String {
    let subject = match tool_argument_hint(args) {
        Some(hint) => format!("{tool}: {hint}"),
        None => tool.to_string(),
    };

    if success {
        format!("\u{2705} {subject} ({secs}s)\n")
    } else if let Some(reason) = error_reason {
        format!(
            "\u{274c} {subject} ({secs}s): {}\n",
            truncate_with_ellipsis(&scrub_and_collapse_display(reason), 200)
        )
    } else {
        format!("\u{274c} {subject} ({secs}s)\n")
    }
}

/// Build a compact, tool-agnostic argument summary from a conservative key
/// allowlist. This keeps start/completion lines useful without publishing
/// arbitrary tool arguments into chat-visible progress.
fn tool_argument_hint(args: &Value) -> Option<String> {
    let Value::Object(map) = args else {
        return None;
    };

    let parts: Vec<String> = TOOL_ARGUMENT_HINT_KEYS
        .iter()
        .filter_map(|key| {
            let value = render_argument_value(map.get(*key)?)?;
            Some(format!(
                "{key}={}",
                truncate_with_ellipsis(&value, MAX_PROGRESS_VALUE_CHARS)
            ))
        })
        .collect();

    (!parts.is_empty()).then(|| parts.join(", "))
}

/// Convert one JSON argument value to compact display text while scrubbing
/// credential-shaped data before it reaches the progress stream.
fn render_argument_value(value: &Value) -> Option<String> {
    let rendered = match value {
        Value::Null => return None,
        Value::String(s) => s.clone(),
        Value::Bool(_) | Value::Number(_) | Value::Array(_) | Value::Object(_) => {
            serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
        }
    };
    let rendered = scrub_and_collapse_display(&rendered);
    if rendered.is_empty() {
        None
    } else {
        Some(rendered)
    }
}

fn scrub_and_collapse_display(value: &str) -> String {
    scrub_credentials(value)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::{render_tool_completion_progress, render_tool_start_progress};
    use serde_json::json;

    #[test]
    fn start_progress_reports_only_allowlisted_arguments() {
        let line = render_tool_start_progress(
            "delegate",
            &json!({
                "agent": "sysadmin",
                "prompt": "Check **service**\nthen report",
                "query": "current state",
                "background": true
            }),
        );

        assert_eq!(line, "\u{23f3} delegate: query=current state\n");
    }

    #[test]
    fn completion_progress_reuses_argument_hint() {
        let line = render_tool_completion_progress(
            "web_fetch",
            &json!({"url": "https://example.test/a?b=c", "method": "GET", "path": "/tmp/out"}),
            3,
            true,
            None,
        );

        assert_eq!(line, "\u{2705} web_fetch: path=/tmp/out (3s)\n");
    }

    #[test]
    fn progress_omits_arguments_without_allowlisted_keys() {
        let line = render_tool_start_progress(
            "delegate",
            &json!({"agent": "sysadmin", "prompt": "private details"}),
        );

        assert_eq!(line, "\u{23f3} delegate\n");
    }

    #[test]
    fn completion_progress_failure_scrubs_error_reason() {
        let line = render_tool_completion_progress(
            "config_read",
            &json!({"path": "/tmp/config.toml"}),
            2,
            false,
            Some("api_key = \"sk-live-abcd1234efgh5678\""),
        );

        assert!(
            line.contains("[REDACTED]"),
            "expected scrubbed line: {line}"
        );
        assert!(
            !line.contains("abcd1234efgh5678"),
            "raw secret leaked: {line}"
        );
        assert!(line.contains("path=/tmp/config.toml"));
    }

    #[test]
    fn completion_progress_failure_keeps_error_reason_on_one_line() {
        let line = render_tool_completion_progress(
            "shell",
            &json!({"command": "printf fail"}),
            1,
            false,
            Some("first line\nsecond line"),
        );

        assert_eq!(
            line,
            "\u{274c} shell: command=printf fail (1s): first line second line\n"
        );
    }

    #[test]
    fn progress_truncates_allowlisted_values_to_sixty_chars() {
        let line = render_tool_start_progress(
            "shell",
            &json!({"command": "012345678901234567890123456789012345678901234567890123456789tail"}),
        );

        assert_eq!(
            line,
            "\u{23f3} shell: command=012345678901234567890123456789012345678901234567890123456789...\n"
        );
    }

    #[test]
    fn progress_omits_runtime_approved_control_arg() {
        let line =
            render_tool_start_progress("shell", &json!({"command": "uname -a", "approved": true}));

        assert_eq!(line, "\u{23f3} shell: command=uname -a\n");
    }
}
