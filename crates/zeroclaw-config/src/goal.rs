//! Goal-mode operator policy and control-plane decision record.
//!
//! This section is config-visible only: no runtime path consumes it yet. The
//! goal controller is its first consumer. Limits here are policy inputs;
//! consumption and remaining budget stay derived from the canonical ledger.

use serde::{Deserialize, Serialize};
use zeroclaw_macros::Configurable;

/// V1 command surfaces that may admit `/goal`.
///
/// This is not a configuration type: configuration uses one explicit,
/// default-closed flag per surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalSurface {
    /// Matrix channel ingress.
    Matrix,
    /// ZeroCode local IPC and WSS session control.
    Zerocode,
}

/// Effective budget policy resolved from operator configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GoalBudgetPolicy {
    /// At least one finite, post-call admission limit.
    Limited {
        /// Maximum known token usage before admitting the next operation.
        token_limit: Option<u64>,
        /// Maximum known USD usage before admitting the next operation.
        cost_limit_usd: Option<f64>,
    },
    /// Unlimited operation explicitly authorized by the operator.
    Unlimited,
}

/// Why operator budget configuration cannot admit a goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalBudgetError {
    /// No finite limit is configured and unlimited operation is not authorized.
    NoFiniteLimit,
    /// A token limit is present but cannot bound any work.
    InvalidTokenLimit,
    /// A cost limit is present but is not positive and finite.
    InvalidCostLimit,
}

/// Explicit, default-closed Goal Mode admission inputs.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Configurable)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[prefix = "goal"]
pub struct GoalConfig {
    /// Master opt-in. Every other field is inert while this is false.
    #[serde(default)]
    pub enabled: bool,

    /// Allow `/goal` admission from the Matrix channel surface.
    #[serde(default)]
    pub allow_matrix: bool,

    /// Allow `/goal` admission from ZeroCode's local IPC and WSS session
    /// control. This does not add authentication.
    #[serde(default)]
    pub allow_zerocode: bool,

    /// Agent aliases eligible to own a Goal. An empty list admits no agent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agents: Vec<String>,

    /// Token limit for a new Goal. When both limits are absent, unlimited work
    /// requires `allow_unlimited`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_token_limit: Option<u64>,

    /// USD limit for a new Goal. This is a post-call admission limit, not a
    /// strict spend ceiling: one already-admitted operation may overshoot it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_cost_limit_usd: Option<f64>,

    /// Explicit operator authorization for a Goal with no finite limit.
    #[serde(default)]
    pub allow_unlimited: bool,
}

impl GoalConfig {
    /// Returns true only when Goal Mode and this surface are both enabled.
    #[must_use]
    pub fn surface_enabled(&self, surface: GoalSurface) -> bool {
        self.enabled
            && match surface {
                GoalSurface::Matrix => self.allow_matrix,
                GoalSurface::Zerocode => self.allow_zerocode,
            }
    }

    /// Returns exact allowlist membership after trimming; an empty alias fails.
    #[must_use]
    pub fn agent_eligible(&self, alias: &str) -> bool {
        let alias = alias.trim();
        !alias.is_empty() && self.agents.iter().any(|agent| agent.trim() == alias)
    }

    /// Returns whether all operator-side admission inputs allow this request.
    #[must_use]
    pub fn admission_enabled(&self, surface: GoalSurface, alias: &str) -> bool {
        self.surface_enabled(surface) && self.agent_eligible(alias)
    }

    /// Resolves the effective budget, rejecting unsafe or malformed policy.
    pub fn budget_policy(&self) -> Result<GoalBudgetPolicy, GoalBudgetError> {
        if self.default_token_limit == Some(0) {
            return Err(GoalBudgetError::InvalidTokenLimit);
        }

        if self
            .default_cost_limit_usd
            .is_some_and(|limit| !limit.is_finite() || limit <= 0.0)
        {
            return Err(GoalBudgetError::InvalidCostLimit);
        }

        if self.default_token_limit.is_some() || self.default_cost_limit_usd.is_some() {
            return Ok(GoalBudgetPolicy::Limited {
                token_limit: self.default_token_limit,
                cost_limit_usd: self.default_cost_limit_usd,
            });
        }

        if self.allow_unlimited {
            Ok(GoalBudgetPolicy::Unlimited)
        } else {
            Err(GoalBudgetError::NoFiniteLimit)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GoalBudgetError, GoalBudgetPolicy, GoalConfig, GoalSurface};
    use crate::schema::Config;

    #[test]
    fn defaults_deny_goal_admission_on_every_surface() {
        let config = GoalConfig::default();

        assert!(!config.enabled);
        assert!(!config.allow_matrix);
        assert!(!config.allow_zerocode);
        assert!(config.agents.is_empty());
        assert_eq!(config.default_token_limit, None);
        assert_eq!(config.default_cost_limit_usd, None);
        assert!(!config.allow_unlimited);
        assert!(!config.surface_enabled(GoalSurface::Matrix));
        assert!(!config.surface_enabled(GoalSurface::Zerocode));
        assert!(!config.admission_enabled(GoalSurface::Matrix, "main"));
        assert!(!config.admission_enabled(GoalSurface::Zerocode, "main"));
    }

    #[test]
    fn agent_allowlist_requires_an_exact_nonempty_trimmed_alias() {
        let config = GoalConfig {
            agents: vec![" main ".to_string()],
            ..GoalConfig::default()
        };

        assert!(config.agent_eligible("main"));
        assert!(config.agent_eligible(" main "));
        assert!(!config.agent_eligible("MAIN"));
        assert!(!config.agent_eligible(""));
        assert!(!config.agent_eligible("   "));
    }

    #[test]
    fn admission_requires_global_surface_and_agent_opt_in() {
        let mut config = GoalConfig {
            enabled: true,
            allow_matrix: true,
            agents: vec!["main".to_string()],
            ..GoalConfig::default()
        };

        assert!(config.admission_enabled(GoalSurface::Matrix, "main"));
        assert!(!config.admission_enabled(GoalSurface::Zerocode, "main"));
        assert!(!config.admission_enabled(GoalSurface::Matrix, "other"));

        config.enabled = false;
        assert!(!config.admission_enabled(GoalSurface::Matrix, "main"));
    }

    #[test]
    fn budget_requires_a_finite_limit_without_explicit_unlimited_opt_in() {
        assert_eq!(
            GoalConfig::default().budget_policy(),
            Err(GoalBudgetError::NoFiniteLimit)
        );
        assert_eq!(
            GoalConfig {
                allow_unlimited: true,
                ..GoalConfig::default()
            }
            .budget_policy(),
            Ok(GoalBudgetPolicy::Unlimited)
        );
    }

    #[test]
    fn finite_limits_take_precedence_over_unlimited_opt_in() {
        let config = GoalConfig {
            default_token_limit: Some(10),
            allow_unlimited: true,
            ..GoalConfig::default()
        };

        assert_eq!(
            config.budget_policy(),
            Ok(GoalBudgetPolicy::Limited {
                token_limit: Some(10),
                cost_limit_usd: None,
            })
        );
    }

    #[test]
    fn malformed_limits_are_never_masked_by_unlimited_opt_in() {
        for config in [
            GoalConfig {
                default_token_limit: Some(0),
                allow_unlimited: true,
                ..GoalConfig::default()
            },
            GoalConfig {
                default_cost_limit_usd: Some(0.0),
                allow_unlimited: true,
                ..GoalConfig::default()
            },
            GoalConfig {
                default_cost_limit_usd: Some(f64::NAN),
                ..GoalConfig::default()
            },
            GoalConfig {
                default_cost_limit_usd: Some(f64::INFINITY),
                ..GoalConfig::default()
            },
            GoalConfig {
                default_cost_limit_usd: Some(-1.0),
                ..GoalConfig::default()
            },
        ] {
            let expected = if config.default_token_limit == Some(0) {
                GoalBudgetError::InvalidTokenLimit
            } else {
                GoalBudgetError::InvalidCostLimit
            };
            assert_eq!(config.budget_policy(), Err(expected));
        }
    }

    #[test]
    fn absent_goal_table_uses_defaults_without_a_schema_migration() {
        let config: Config = toml::from_str("").expect("empty config should deserialize");

        assert_eq!(config.goal.enabled, GoalConfig::default().enabled);
        assert_eq!(config.goal.agents, GoalConfig::default().agents);
    }

    #[test]
    fn populated_goal_table_round_trips_and_omits_empty_defaults() {
        let config: Config = toml::from_str(
            r#"
            [goal]
            enabled = true
            allow_matrix = true
            allow_zerocode = true
            agents = ["main"]
            default_token_limit = 100
            default_cost_limit_usd = 2.5
            allow_unlimited = true
            "#,
        )
        .expect("populated goal config should deserialize");

        assert_eq!(
            config.goal.budget_policy(),
            Ok(GoalBudgetPolicy::Limited {
                token_limit: Some(100),
                cost_limit_usd: Some(2.5),
            })
        );

        let serialized = toml::to_string(&config).expect("config should serialize");
        let round_tripped: Config =
            toml::from_str(&serialized).expect("serialized config should deserialize");
        assert_eq!(round_tripped.goal.agents, vec!["main"]);
        assert_eq!(round_tripped.goal.default_token_limit, Some(100));
        assert_eq!(round_tripped.goal.default_cost_limit_usd, Some(2.5));

        let defaults =
            toml::to_string(&GoalConfig::default()).expect("default goal config should serialize");
        assert!(!defaults.contains("agents"));
        assert!(!defaults.contains("default_token_limit"));
        assert!(!defaults.contains("default_cost_limit_usd"));
    }
}
