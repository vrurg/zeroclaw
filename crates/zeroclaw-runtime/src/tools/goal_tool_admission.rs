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

impl ModelGoalAdmissionPermit {
    /// Commit the already-validated ephemeral bindings for the admitted task.
    pub(super) fn activate(self, task_id: String) {
        self.binding.bind(task_id.clone());
        self.accounting.activate(task_id);
        crate::control_plane::mark_current_goal_turn_for_evaluation();
    }
}
