//! Credential redaction for the rendering layer (logs, observer events, and
//! UI-facing turn events). This never runs on the data path: tool results fed
//! back to the model and signed by HMAC receipts always carry raw bytes.

use regex::Regex;
use std::sync::LazyLock;

static SENSITIVE_KV_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(authorization|token|api[_-]?key|password|secret|user[_-]?key|bearer|credential|set[_-]?cookie|cookie)["']?\s*[:=]\s*(?:"([^"]{8,})"|'([^']{8,})'|([a-zA-Z0-9_\-\./+=]{8,}))"#).unwrap()
});

pub fn scrub_credentials(input: &str) -> String {
    SENSITIVE_KV_REGEX
        .replace_all(input, |caps: &regex::Captures| {
            let full_match = &caps[0];
            let key = &caps[1];
            let val = caps
                .get(2)
                .or(caps.get(3))
                .or(caps.get(4))
                .map(|m| m.as_str())
                .unwrap_or("");

            // Preserve first 4 chars for context, then redact.
            // Use char_indices to find the byte offset of the 4th character
            // so we never slice in the middle of a multi-byte UTF-8 sequence.
            let prefix = if val.len() > 4 {
                val.char_indices()
                    .nth(4)
                    .map(|(byte_idx, _)| &val[..byte_idx])
                    .unwrap_or(val)
            } else {
                ""
            };

            if full_match.contains(':') {
                if full_match.contains('"') {
                    format!("\"{}\": \"{}*[REDACTED]\"", key, prefix)
                } else {
                    format!("{}: {}*[REDACTED]", key, prefix)
                }
            } else if full_match.contains('=') {
                if full_match.contains('"') {
                    format!("{}=\"{}*[REDACTED]\"", key, prefix)
                } else {
                    format!("{}={}*[REDACTED]", key, prefix)
                }
            } else {
                format!("{}: {}*[REDACTED]", key, prefix)
            }
        })
        .to_string()
}

static SENSITIVE_KEY_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(authorization|token|api[_-]?key|password|secret|user[_-]?key|bearer|credential|set[_-]?cookie|cookie)"#).unwrap()
});

/// Structured-aware credential scrub for a JSON value bound for a human-facing
/// surface. Object entries whose key names a credential have their string value
/// redacted in place, preserving the key; every other string leaf still runs
/// through the text [`scrub_credentials`] so inline `token=...` patterns inside
/// unrelated fields are caught too. Serialize-then-scrub would corrupt key names
/// that merely contain a sensitive word (e.g. `access_token`), so this walks the
/// value instead. Same rendering-boundary contract as [`scrub_credentials`].
pub fn scrub_credentials_value(value: serde_json::Value) -> serde_json::Value {
    scrub_credentials_value_with(value, &scrub_credentials)
}

pub(crate) fn scrub_credentials_value_with_config(
    value: serde_json::Value,
    config: &zeroclaw_config::schema::LeakDetectionConfig,
) -> serde_json::Value {
    scrub_credentials_value_with(value, &|text| {
        crate::security::scrub_with_config(&scrub_credentials(text), config)
    })
}

fn scrub_credentials_value_with(
    mut value: serde_json::Value,
    scrub_string: &impl Fn(&str) -> String,
) -> serde_json::Value {
    scrub_credentials_value_in_place(&mut value, scrub_string);
    value
}

fn scrub_credentials_value_in_place(
    value: &mut serde_json::Value,
    scrub_string: &impl Fn(&str) -> String,
) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, nested) in map {
                if SENSITIVE_KEY_REGEX.is_match(key)
                    && let serde_json::Value::String(secret) = nested
                {
                    let redacted = redact_credential_string(secret);
                    *secret = scrub_string(&redacted);
                } else {
                    scrub_credentials_value_in_place(nested, scrub_string);
                }
            }
        }
        serde_json::Value::Array(items) => {
            for nested in items {
                scrub_credentials_value_in_place(nested, scrub_string);
            }
        }
        serde_json::Value::String(text) => *text = scrub_string(text),
        _ => {}
    }
}

/// Redact a string sitting under a credential-named key while retaining a
/// short prefix for operator context.
fn redact_credential_string(secret: &str) -> String {
    let prefix = secret
        .char_indices()
        .nth(4)
        .map(|(byte_idx, _)| &secret[..byte_idx])
        .unwrap_or("");
    format!("{prefix}*[REDACTED]")
}

#[cfg(test)]
mod tests {
    use super::{scrub_credentials, scrub_credentials_value, scrub_credentials_value_with};
    use std::cell::Cell;

    #[test]
    fn scrub_credentials_value_redacts_nested_secret_and_keeps_key() {
        let input = serde_json::json!({
            "body": {"access_token": "sk-live-abcdef0123456789", "status": "ok"},
            "count": 3
        });
        let out = scrub_credentials_value(input);
        let token = out["body"]["access_token"].as_str().unwrap();
        assert!(token.contains("[REDACTED]"));
        assert!(!token.contains("abcdef0123456789"));
        assert_eq!(out["body"]["status"], "ok");
        assert_eq!(out["count"], 3);
    }

    #[test]
    fn scrub_credentials_value_redacts_authorization_and_cookie_keys() {
        let input = serde_json::json!({
            "body": {
                "authorization": "Bearer sk-live-abcdef0123456789",
                "cookie": "session=deadbeefcafebabe0123",
                "set-cookie": "sid=9f8e7d6c5b4a3210feed",
                "status": "ok"
            }
        });
        let out = scrub_credentials_value(input);
        let authorization = out["body"]["authorization"].as_str().unwrap();
        assert!(authorization.contains("[REDACTED]"));
        assert!(!authorization.contains("sk-live-abcdef0123456789"));
        let cookie = out["body"]["cookie"].as_str().unwrap();
        assert!(cookie.contains("[REDACTED]"));
        assert!(!cookie.contains("deadbeefcafebabe0123"));
        let set_cookie = out["body"]["set-cookie"].as_str().unwrap();
        assert!(set_cookie.contains("[REDACTED]"));
        assert!(!set_cookie.contains("9f8e7d6c5b4a3210feed"));
        assert_eq!(out["body"]["status"], "ok");
    }

    #[test]
    fn structured_secret_leaf_still_runs_the_followup_scrubber() {
        let calls = Cell::new(0);
        let out = scrub_credentials_value_with(serde_json::json!({"token": "abcdefgh"}), &|text| {
            calls.set(calls.get() + 1);
            format!("{text}|configured")
        });

        assert_eq!(calls.get(), 1);
        assert_eq!(out["token"], "abcd*[REDACTED]|configured");
    }

    #[test]
    fn scrub_credentials_redacts_unquoted_base64_credential_values() {
        let input = "token=QWxh+GRpbjpvcGVu/IHNlc2FtZQ== next=public";
        let scrubbed = scrub_credentials(input);

        assert_eq!(scrubbed, "token=QWxh*[REDACTED] next=public");
        assert!(!scrubbed.contains("IHNlc2FtZQ"));
        assert!(!scrubbed.contains("=="));
    }

    #[test]
    fn scrub_credentials_redacts_quoted_base64_credential_values() {
        let input = r#"secret="QWxhZGRpbjpvcGVu/IHNlc2FtZQ==""#;
        let scrubbed = scrub_credentials(input);

        assert_eq!(scrubbed, r#"secret="QWxh*[REDACTED]""#);
        assert!(!scrubbed.contains("IHNlc2FtZQ"));
        assert!(!scrubbed.contains("=="));
    }
}
