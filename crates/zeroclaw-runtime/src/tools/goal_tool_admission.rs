//! Shared pre-admission for model-callable goal tools.

use zeroclaw_config::schema::Config;

/// Prevalidated transient resources needed after a model goal admission.
///
/// The durable controller chooses the exact task id. This permit validates all
/// fallible task-local and ledger prerequisites before that mutation, so a
/// failed prerequisite cannot create or resume a goal and then pause it again.
pub(super) struct ModelGoalAdmissionPermit {
    binding: crate::control_plane::GoalTaskBindingReservation,
    accounting: crate::agent::cost::PreparedToolLoopGoalAttribution,
}

pub(super) fn prepare(config: &Config) -> anyhow::Result<ModelGoalAdmissionPermit> {
    crate::control_plane::ensure_current_goal_task_binding_available()?;
    let binding = crate::control_plane::reserve_current_goal_task_binding()?;
    let accounting = crate::agent::cost::prepare_current_tool_loop_goal_attribution(config)?;
    Ok(ModelGoalAdmissionPermit {
        binding,
        accounting,
    })
}

/// Record a model-goal admission failure without copying the underlying error
/// into a channel-visible tool result or structured event.
///
/// Admission errors can contain storage paths or provider details. Operators
/// still need to distinguish the fail-closed prerequisite that rejected the
/// tool invocation, so retain only a stable category derived from the trusted
/// local error boundary.
pub(super) fn record_failure(operation: &str, agent_alias: &str, error: &anyhow::Error) {
    ::zeroclaw_log::record!(
        WARN,
        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
            .with_attrs(::serde_json::json!({
                "error_key": "goal_tool_admission.preflight_failed",
                "operation": operation,
                "agent_alias": agent_alias,
                "cause": failure_cause(error),
            })),
        "model goal tool admission preflight failed"
    );
}

fn failure_cause(error: &anyhow::Error) -> &'static str {
    let message = error.to_string();
    if message.contains("goal admission context unavailable") {
        "admission_context_unavailable"
    } else if message.contains("goal admission already has an exact live task binding")
        || message.contains("goal admission task binding is reserved")
    {
        "task_binding_unavailable"
    } else if message.contains("goal accounting context unavailable") {
        "accounting_context_unavailable"
    } else if message.contains("goal accounting tracker unavailable") {
        "accounting_ledger_unavailable"
    } else if message.contains("goal accounting attribution is already bound")
        || message.contains("goal accounting attribution is reserved")
    {
        "accounting_binding_unavailable"
    } else {
        "preflight_unavailable"
    }
}

impl ModelGoalAdmissionPermit {
    /// Commit the already-validated ephemeral bindings for the admitted task.
    pub(super) fn activate(self, task_id: String) {
        self.binding.bind(task_id.clone());
        self.accounting.activate(task_id);
        crate::control_plane::mark_current_goal_turn_for_evaluation();
    }
}

#[cfg(test)]
mod tests {
    use super::failure_cause;

    #[test]
    fn failure_cause_classifies_known_preflight_boundaries_without_raw_error_text() {
        assert_eq!(
            failure_cause(&anyhow::Error::msg("goal accounting context unavailable")),
            "accounting_context_unavailable"
        );
        assert_eq!(
            failure_cause(&anyhow::Error::msg(
                "goal admission task binding is reserved"
            )),
            "task_binding_unavailable"
        );
        assert_eq!(
            failure_cause(&anyhow::Error::msg(
                "goal accounting tracker unavailable: /private/secret/costs.jsonl"
            )),
            "accounting_ledger_unavailable"
        );
        assert_eq!(
            failure_cause(&anyhow::Error::msg("sqlite failure at /private/secret")),
            "preflight_unavailable"
        );
    }
}
