//! Prospective Goal-policy classification for daemon reload cutovers.
//!
//! The classifier is deliberately pure: it translates a durable Goal record
//! plus a fully loaded successor configuration into an exact revocation
//! target.  SQLite remains responsible for the all-or-nothing cancellation;
//! adapters and reload orchestration must not reconstruct policy from route
//! text on their own.

use anyhow::{Result, ensure};
use zeroclaw_config::schema::Config;

use crate::{
    control_plane::{
        GoalPolicyTarget, GoalTaskRegistry, GoalTransitionResult, TaskRecord, TaskStatus,
    },
    goal_mode::GoalSessionKey,
};

/// Why a successor configuration revokes a currently nonterminal Goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalPolicyRevocation {
    GoalModeDisabled,
    AgentUnavailable,
    MatrixChannelUnavailable,
    UnsupportedRoute,
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
pub fn classify_goal_policy(config: &Config, task: &TaskRecord) -> Result<GoalPolicyDecision> {
    let target = policy_target(task)?;
    if !config.goal.enabled {
        return Ok(revoked(target, GoalPolicyRevocation::GoalModeDisabled));
    }
    let Some(agent) = config.agents.get(&task.agent) else {
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
    let Some(matrix) = config.channels.matrix.get(alias) else {
        return Ok(revoked(
            target,
            GoalPolicyRevocation::MatrixChannelUnavailable,
        ));
    };
    let required_channel = format!("matrix.{alias}");
    if !matrix.enabled
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
    successor: &Config,
) -> Result<Vec<GoalPolicyRevocation>> {
    let observed = registry.list_nonterminal_session_goals().await?;
    let decisions = observed
        .iter()
        .map(|task| classify_goal_policy(successor, task))
        .collect::<Result<Vec<_>>>()?;
    let revoked = decisions
        .iter()
        .filter_map(|decision| match decision {
            GoalPolicyDecision::Keep => None,
            GoalPolicyDecision::Revoke { target, reason } => Some((target.clone(), *reason)),
        })
        .collect::<Vec<_>>();
    if revoked.is_empty() {
        return Ok(Vec::new());
    }
    match registry
        .cancel_policy_targets(
            &revoked
                .iter()
                .map(|(target, _)| target.clone())
                .collect::<Vec<_>>(),
        )
        .await?
    {
        GoalTransitionResult::Applied => {
            Ok(revoked.into_iter().map(|(_, reason)| reason).collect())
        }
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
    let session_id = task
        .session_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let session_id = session_id
        .ok_or_else(|| anyhow::Error::msg("Goal policy target is missing a session binding"))?;
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
    use crate::control_plane::{
        GoalTaskRecord, GoalTaskRegistry, SqliteTaskStore, TaskKind, TaskRegistry, TaskStatus,
    };
    use zeroclaw_config::{
        providers::ChannelRef,
        schema::{AliasedAgentConfig, MatrixConfig},
    };

    fn task(route: &str) -> TaskRecord {
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
            delivered: false,
            idem_key: None,
            principal_id: None,
            session_id: Some(
                route
                    .strip_prefix("zerocode:")
                    .map_or_else(|| "matrix_session".to_owned(), |raw| format!("rpc_{raw}")),
            ),
            execution_epoch: 3,
            started_at: "2026-09-01T00:00:00Z".into(),
            finished_at: None,
        }
    }

    fn config() -> Config {
        let mut config = Config::default();
        config.goal.enabled = true;
        let mut agent = AliasedAgentConfig::default();
        agent.channels.push(ChannelRef::new("matrix.main"));
        config.agents.insert("main".into(), agent);
        config.channels.matrix.insert(
            "main".into(),
            MatrixConfig {
                enabled: true,
                ..MatrixConfig::default()
            },
        );
        config
    }

    #[test]
    fn successor_policy_revokes_only_removed_goal_scopes() {
        let mut successor = config();
        assert_eq!(
            classify_goal_policy(&successor, &task("matrix:main:!room:example")).unwrap(),
            GoalPolicyDecision::Keep
        );
        assert_eq!(
            classify_goal_policy(&successor, &task("zerocode:local-session")).unwrap(),
            GoalPolicyDecision::Keep
        );

        successor.channels.matrix.get_mut("main").unwrap().enabled = false;
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
        let mut successor = config();
        successor.goal.enabled = false;
        assert!(matches!(
            classify_goal_policy(&successor, &task("zerocode:local-session")).unwrap(),
            GoalPolicyDecision::Revoke {
                reason: GoalPolicyRevocation::GoalModeDisabled,
                ..
            }
        ));

        let successor = config();
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

    #[tokio::test]
    async fn prospective_policy_revocation_uses_one_durable_goal_transaction() {
        let store = SqliteTaskStore::new_in_memory().unwrap();
        let active = task("matrix:main:!room:example");
        assert_eq!(
            store
                .create_or_replace_session_goal(
                    active,
                    GoalTaskRecord {
                        task_id: "goal-1".into(),
                        objective: "finish the task".into(),
                        ..GoalTaskRecord::default()
                    },
                )
                .await
                .unwrap(),
            crate::control_plane::GoalTransitionResult::Applied
        );
        let mut successor = config();
        successor.goal.enabled = false;

        assert_eq!(
            revoke_goals_under_policy(&store, &successor).await.unwrap(),
            vec![GoalPolicyRevocation::GoalModeDisabled]
        );
        let cancelled = store.get("goal-1").await.unwrap().unwrap();
        assert_eq!(cancelled.status, TaskStatus::Cancelled);
        assert_eq!(
            cancelled.execution_epoch, 2,
            "new Goals begin at epoch one and policy cancellation fences that epoch"
        );
        assert_eq!(
            store
                .terminal_reason_for_session_goal("goal-1", "matrix_session")
                .await
                .unwrap()
                .as_deref(),
            Some("policy_revoked")
        );
        assert_eq!(
            store
                .resume_session_goal("goal-1", "matrix_session", 2, 1, "replacement-boot")
                .await
                .unwrap(),
            crate::control_plane::GoalTransitionResult::Stale,
            "a later configuration expansion must not revive a policy-cancelled Goal"
        );
    }
}
