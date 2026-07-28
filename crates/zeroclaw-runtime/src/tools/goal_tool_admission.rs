//! Shared pre-admission for model-callable goal tools.

use zeroclaw_config::schema::Config;

/// Prevalidated transient resources needed after a model goal admission.
///
/// The durable controller chooses the exact task id. This permit validates all
/// fallible task-local and ledger prerequisites before that mutation, so a
/// failed prerequisite cannot create or resume a goal and then pause it again.
pub(super) struct ModelGoalAdmissionPermit {
    accounting: crate::agent::cost::PreparedToolLoopGoalAttribution,
}

pub(super) fn prepare(config: &Config) -> anyhow::Result<ModelGoalAdmissionPermit> {
    crate::control_plane::ensure_current_goal_task_binding_available()?;
    let accounting = crate::agent::cost::prepare_current_tool_loop_goal_attribution(config)?;
    Ok(ModelGoalAdmissionPermit { accounting })
}

impl ModelGoalAdmissionPermit {
    /// Commit the already-validated ephemeral bindings for the admitted task.
    pub(super) fn activate(self, task_id: &str) -> anyhow::Result<()> {
        if !crate::control_plane::bind_current_goal_task(task_id) {
            anyhow::bail!("goal admission could not bind its exact live task");
        }
        self.accounting.activate(task_id)?;
        crate::control_plane::mark_current_goal_turn_for_evaluation();
        Ok(())
    }
}
