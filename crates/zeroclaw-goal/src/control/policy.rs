//! Prospective Goal-policy classification for daemon reload cutovers.
//!
//! The classifier is deliberately pure: it translates a durable Goal record
//! plus a fully loaded successor configuration into an exact revocation
//! target.  SQLite remains responsible for the all-or-nothing cancellation;
//! adapters and reload orchestration must not reconstruct policy from route
//! text on their own.

use anyhow::{Result, ensure};
use std::collections::HashMap;

use crate::{
    control::GoalSessionKey,
    task::{GoalPolicyTarget, GoalTaskRegistry, GoalTransitionResult},
};
use zeroclaw_api::task::{TaskKind, TaskRecord, TaskStatus};

/// Why a successor configuration revokes a currently nonterminal Goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalPolicyRevocation {
    GoalModeDisabled,
    AgentUnavailable,
    MatrixChannelUnavailable,
    UnsupportedRoute,
}

/// Config-independent projection of the successor policy used to classify
/// session-bound Goals. The runtime builds this value from its validated
/// configuration; Goal policy never depends on the config schema crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalPolicySnapshot {
    pub goal_mode_enabled: bool,
    pub agents: HashMap<String, GoalAgentPolicySnapshot>,
    pub matrix_channels: HashMap<String, bool>,
}

/// The subset of an agent's validated policy relevant to Goal continuation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalAgentPolicySnapshot {
    pub enabled: bool,
    pub channels: Vec<String>,
}

/// One observed Goal and the policy consequence selected from a prospective
/// configuration. `Keep` never grants new authority; it only means the Goal
/// remains subject to ordinary restart pause and later explicit resume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoalPolicyDecision {
    Keep,
    Revoke {
        target: GoalPolicyTarget,
        reason: GoalPolicyRevocation,
    },
}

/// Classify a nonterminal session-bound Goal against a successor configuration.
///
/// V1 only supports Matrix and ZeroCode ingress. Unknown, malformed, or
/// removed routes fail closed rather than retaining a Goal that a successor
/// runtime cannot safely authorize.
pub fn classify_goal_policy(
    successor: &GoalPolicySnapshot,
    task: &TaskRecord,
) -> Result<GoalPolicyDecision> {
    let target = policy_target(task)?;
    if !successor.goal_mode_enabled {
        return Ok(revoked(target, GoalPolicyRevocation::GoalModeDisabled));
    }
    let Some(agent) = successor.agents.get(&task.agent) else {
        return Ok(revoked(target, GoalPolicyRevocation::AgentUnavailable));
    };
    if !agent.enabled {
        return Ok(revoked(target, GoalPolicyRevocation::AgentUnavailable));
    }
    let Some(route) = task.originator_route.as_deref() else {
        return Ok(revoked(target, GoalPolicyRevocation::UnsupportedRoute));
    };
    if let Some(raw_session_id) = route.strip_prefix("zerocode:") {
        return Ok(match GoalSessionKey::zero_code(raw_session_id) {
            Ok(session_key) if session_key.durable_id() == target.session_id => {
                GoalPolicyDecision::Keep
            }
            Ok(_) | Err(_) => revoked(target, GoalPolicyRevocation::UnsupportedRoute),
        });
    }
    let Some((alias, reply_target)) = matrix_route(route) else {
        return Ok(revoked(target, GoalPolicyRevocation::UnsupportedRoute));
    };
    let matrix_enabled = successor
        .matrix_channels
        .get(alias)
        .copied()
        .unwrap_or(false);
    let required_channel = format!("matrix.{alias}");
    if !matrix_enabled
        || !agent
            .channels
            .iter()
            .any(|channel| channel.as_str() == required_channel)
        || reply_target.is_empty()
    {
        return Ok(revoked(
            target,
            GoalPolicyRevocation::MatrixChannelUnavailable,
        ));
    }
    Ok(GoalPolicyDecision::Keep)
}

/// Cancel every nonterminal Goal revoked by a fully validated successor policy.
///
/// Classification precedes one guarded SQLite transaction. If a record changes
/// after observation, the transaction returns `Stale` and the caller must keep
/// the old runtime and worker ownership intact rather than applying a partial
/// cutover.
pub async fn revoke_goals_under_policy(
    registry: &dyn GoalTaskRegistry,
    successor: &GoalPolicySnapshot,
) -> Result<Vec<GoalPolicyRevocation>> {
    let observed = registry.list_nonterminal_session_goals().await?;
    let mut targets = Vec::new();
    let mut reasons = Vec::new();
    for task in &observed {
        if let GoalPolicyDecision::Revoke { target, reason } =
            classify_goal_policy(successor, task)?
        {
            targets.push(target);
            reasons.push(reason);
        }
    }
    if targets.is_empty() {
        return Ok(Vec::new());
    }
    match registry.cancel_policy_targets(&targets).await? {
        GoalTransitionResult::Applied => Ok(reasons),
        GoalTransitionResult::Stale => anyhow::bail!("Goal policy targets changed during cutover"),
        GoalTransitionResult::Missing => {
            anyhow::bail!("Goal policy target disappeared during cutover")
        }
    }
}

fn revoked(target: GoalPolicyTarget, reason: GoalPolicyRevocation) -> GoalPolicyDecision {
    GoalPolicyDecision::Revoke { target, reason }
}

fn policy_target(task: &TaskRecord) -> Result<GoalPolicyTarget> {
    ensure!(
        matches!(task.status, TaskStatus::Running | TaskStatus::Paused),
        "policy classification requires a nonterminal Goal"
    );
    ensure!(
        task.kind == TaskKind::Goal,
        "policy classification requires a Goal task"
    );
    let session_id = task
        .session_id
        .as_deref()
        .ok_or_else(|| anyhow::Error::msg("Goal policy target is missing a session binding"))?;
    ensure!(
        !session_id.is_empty() && session_id.trim() == session_id,
        "Goal policy target has a noncanonical session binding"
    );
    ensure!(
        task.execution_epoch > 0,
        "Goal policy target has a nonpositive execution epoch"
    );
    Ok(GoalPolicyTarget {
        task_id: task.id.clone(),
        session_id: session_id.to_owned(),
        execution_epoch: task.execution_epoch,
    })
}

fn matrix_route(route: &str) -> Option<(&str, &str)> {
    let mut fields = route.splitn(3, ':');
    (fields.next()? == "matrix").then_some(())?;
    let alias = fields.next()?;
    let reply_target = fields.next()?;
    (!alias.is_empty() && !reply_target.is_empty()).then_some((alias, reply_target))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn task(route: &str) -> TaskRecord {
        let session_id = route
            .strip_prefix("zerocode:")
            .filter(|raw_session_id| !raw_session_id.is_empty())
            .map(|raw_session_id| format!("rpc_{raw_session_id}"))
            .unwrap_or_else(|| "matrix_session".into());
        TaskRecord {
            id: "goal-1".into(),
            kind: TaskKind::Goal,
            agent: "main".into(),
            status: TaskStatus::Running,
            owner_pid: 1,
            owner_boot_id: "boot".into(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: Some(route.into()),
            originator_chain: Vec::new(),
            delivered: false,
            idem_key: None,
            principal_id: None,
            session_id: Some(session_id),
            execution_epoch: 3,
            started_at: "2026-09-01T00:00:00Z".into(),
            finished_at: None,
        }
    }

    fn policy() -> GoalPolicySnapshot {
        GoalPolicySnapshot {
            goal_mode_enabled: true,
            agents: HashMap::from([(
                "main".to_owned(),
                GoalAgentPolicySnapshot {
                    enabled: true,
                    channels: vec!["matrix.main".to_owned()],
                },
            )]),
            matrix_channels: HashMap::from([("main".to_owned(), true)]),
        }
    }

    #[test]
    fn successor_policy_revokes_only_removed_goal_scopes() {
        let mut successor = policy();
        assert_eq!(
            classify_goal_policy(&successor, &task("matrix:main:!room:example")).unwrap(),
            GoalPolicyDecision::Keep
        );
        assert_eq!(
            classify_goal_policy(&successor, &task("zerocode:local-session")).unwrap(),
            GoalPolicyDecision::Keep
        );
        successor.matrix_channels.insert("main".into(), false);
        assert!(matches!(
            classify_goal_policy(&successor, &task("matrix:main:!room:example")).unwrap(),
            GoalPolicyDecision::Revoke {
                reason: GoalPolicyRevocation::MatrixChannelUnavailable,
                ..
            }
        ));
        assert_eq!(
            classify_goal_policy(&successor, &task("zerocode:local-session")).unwrap(),
            GoalPolicyDecision::Keep,
            "removing Matrix must not revoke a ZeroCode Goal"
        );
    }

    #[test]
    fn disabled_goal_mode_and_unknown_routes_fail_closed() {
        let mut successor = policy();
        successor.goal_mode_enabled = false;
        assert!(matches!(
            classify_goal_policy(&successor, &task("zerocode:local-session")).unwrap(),
            GoalPolicyDecision::Revoke {
                reason: GoalPolicyRevocation::GoalModeDisabled,
                ..
            }
        ));

        let successor = policy();
        assert!(matches!(
            classify_goal_policy(&successor, &task("telegram:main:chat")).unwrap(),
            GoalPolicyDecision::Revoke {
                reason: GoalPolicyRevocation::UnsupportedRoute,
                ..
            }
        ));

        for route in [
            "zerocode:",
            "zerocode: padded",
            "zerocode:local-session:extra",
        ] {
            assert!(matches!(
                classify_goal_policy(&successor, &task(route)).unwrap(),
                GoalPolicyDecision::Revoke {
                    reason: GoalPolicyRevocation::UnsupportedRoute,
                    ..
                }
            ));
        }

        let mut mismatched = task("zerocode:local-session");
        mismatched.session_id = Some("rpc_other-session".to_owned());
        assert!(matches!(
            classify_goal_policy(&successor, &mismatched).unwrap(),
            GoalPolicyDecision::Revoke {
                reason: GoalPolicyRevocation::UnsupportedRoute,
                ..
            }
        ));
    }
}
