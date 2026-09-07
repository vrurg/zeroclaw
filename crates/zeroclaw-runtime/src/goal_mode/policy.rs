//! Transport-neutral policy facts for Goal admission and revocation.
//!
//! The control plane owns durable lifecycle state. This module only classifies
//! a canonical task against a prospective runtime configuration; Matrix and
//! ZeroCode retain their own surface-specific facts and checks.

use anyhow::Result;
use zeroclaw_config::schema::Config;

use super::{GoalIngressContext, GoalSurface};
use crate::control_plane::{TaskKind, TaskRecord};

/// Why prospective policy revokes an otherwise nonterminal Goal.
///
/// Reasons are controller-internal. Adapters render typed lifecycle results
/// rather than exposing durable routes or principals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalPolicyRevocationReason {
    GoalDisabled,
    BindingInvalid,
    AgentUnavailable,
    AgentProviderUnavailable,
    VerifierUnavailable,
    SurfaceUnavailable,
}

/// Typed result of prospective Goal policy evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalPolicyDecision {
    Eligible,
    Revoked(GoalPolicyRevocationReason),
}

/// Closed V1 surface extension point for policy-only checks.
///
/// A policy adapter receives durable task bindings or immutable trusted
/// ingress. It must not create lifecycle state, resolve a live session from a
/// route string, or retain a policy snapshot beyond the calling cutover.
pub trait GoalSurfacePolicy: Send + Sync {
    fn surface(&self) -> GoalSurface;

    fn classify_persisted(&self, config: &Config, task: &TaskRecord) -> Result<GoalPolicyDecision>;

    fn authorize_ingress(
        &self,
        config: &Config,
        ingress: &GoalIngressContext,
    ) -> Result<GoalPolicyDecision>;
}

/// Classify policy facts shared by every V1 Goal surface.
///
/// Callers invoke this only for a nonterminal `TaskKind::Goal` row. A terminal
/// row is deliberately left eligible: revocation must not rewrite audit
/// history. Surface-specific route and principal checks belong to a matching
/// [`GoalSurfacePolicy`] implementation.
pub fn classify_common_goal_policy(config: &Config, task: &TaskRecord) -> GoalPolicyDecision {
    if task.kind != TaskKind::Goal || task.status.is_terminal() {
        return GoalPolicyDecision::Eligible;
    }
    if !config.goal.enabled {
        return GoalPolicyDecision::Revoked(GoalPolicyRevocationReason::GoalDisabled);
    }
    if task
        .session_id
        .as_deref()
        .is_none_or(|session_id| session_id.trim().is_empty())
        || task
            .originator_route
            .as_deref()
            .is_none_or(|route| route.trim().is_empty())
    {
        return GoalPolicyDecision::Revoked(GoalPolicyRevocationReason::BindingInvalid);
    }
    let Some(agent) = config.agents.get(&task.agent) else {
        return GoalPolicyDecision::Revoked(GoalPolicyRevocationReason::AgentUnavailable);
    };
    if !agent.is_dispatchable() {
        return GoalPolicyDecision::Revoked(GoalPolicyRevocationReason::AgentUnavailable);
    }
    if config
        .resolved_model_provider_for_agent(&task.agent)
        .is_none()
    {
        return GoalPolicyDecision::Revoked(GoalPolicyRevocationReason::AgentProviderUnavailable);
    }
    let verifier = config.goal.verifier.model_provider.as_str().trim();
    let verifier_resolves =
        verifier
            .split_once('.')
            .is_some_and(|(provider_type, provider_alias)| {
                !provider_type.is_empty()
                    && !provider_alias.is_empty()
                    && config
                        .providers
                        .models
                        .find(provider_type, provider_alias)
                        .is_some()
            });
    if !verifier_resolves {
        return GoalPolicyDecision::Revoked(GoalPolicyRevocationReason::VerifierUnavailable);
    }
    GoalPolicyDecision::Eligible
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_plane::{TaskRecord, TaskStatus};

    fn running_goal() -> TaskRecord {
        TaskRecord {
            id: "goal-policy-test".to_owned(),
            kind: TaskKind::Goal,
            agent: "main".to_owned(),
            status: TaskStatus::Running,
            owner_pid: 1,
            owner_boot_id: "boot".to_owned(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: Some("zerocode:session".to_owned()),
            delivered: false,
            idem_key: None,
            principal_id: None,
            session_id: Some("rpc_session".to_owned()),
            execution_epoch: 1,
            started_at: "2026-01-01T00:00:00Z".to_owned(),
            finished_at: None,
        }
    }

    #[test]
    fn global_goal_disable_revokes_every_nonterminal_surface_before_surface_policy() {
        let config = Config::default();
        assert_eq!(
            classify_common_goal_policy(&config, &running_goal()),
            GoalPolicyDecision::Revoked(GoalPolicyRevocationReason::GoalDisabled)
        );
    }

    #[test]
    fn terminal_goal_remains_audit_history_when_goal_mode_is_disabled() {
        let config = Config::default();
        let mut task = running_goal();
        task.status = TaskStatus::Cancelled;
        assert_eq!(
            classify_common_goal_policy(&config, &task),
            GoalPolicyDecision::Eligible
        );
    }
}
