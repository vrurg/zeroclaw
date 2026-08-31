//! Dormant Goal Mode configuration. Runtime admission is introduced later.

use serde::{Deserialize, Serialize};
use zeroclaw_macros::Configurable;

use crate::providers::ModelProviderRef;

/// Normalized durable limits: `None` is unlimited for that dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GoalBudgetLimits {
    pub token_limit: Option<u64>,
    pub cost_limit_usd: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalConfigError {
    MissingTokenLimit,
    MissingCostLimit,
    MissingVerifierProvider,
    TokenLimitOutOfRange,
    InvalidCostLimit,
    EmptyVerifierModel,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, Configurable)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub struct GoalVerifierConfig {
    /// Existing provider/profile reference used by the mandatory verifier.
    #[serde(default)]
    pub model_provider: ModelProviderRef,
    /// Optional model override for the verifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Default-closed Goal Mode configuration. Raw options retain omission versus
/// explicit zero; [`GoalConfig::effective_limits`] normalizes zero once.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Configurable)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[prefix = "goal"]
pub struct GoalConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_token_limit: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_cost_limit_usd: Option<f64>,
    #[serde(default)]
    #[nested]
    pub verifier: GoalVerifierConfig,
}

impl GoalConfig {
    /// Validate syntax and enabled-only required fields without adding a new
    /// credential or provider-resolution path.
    pub fn validate(&self) -> Result<(), GoalConfigError> {
        if self
            .default_token_limit
            .is_some_and(|value| value > i64::MAX as u64)
        {
            return Err(GoalConfigError::TokenLimitOutOfRange);
        }
        if self
            .default_cost_limit_usd
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(GoalConfigError::InvalidCostLimit);
        }
        if self
            .verifier
            .model
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(GoalConfigError::EmptyVerifierModel);
        }
        if self.enabled {
            if self.default_token_limit.is_none() {
                return Err(GoalConfigError::MissingTokenLimit);
            }
            if self.default_cost_limit_usd.is_none() {
                return Err(GoalConfigError::MissingCostLimit);
            }
            if self.verifier.model_provider.as_str().trim().is_empty() {
                return Err(GoalConfigError::MissingVerifierProvider);
            }
        }
        Ok(())
    }

    /// Normalize explicit zero to the existing unlimited (`None`) storage form.
    pub fn effective_limits(&self) -> Result<GoalBudgetLimits, GoalConfigError> {
        self.validate()?;
        Ok(GoalBudgetLimits {
            token_limit: self.default_token_limit.filter(|value| *value != 0),
            cost_limit_usd: self.default_cost_limit_usd.filter(|value| *value != 0.0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Config;

    #[test]
    fn disabled_config_is_optional_but_rejects_malformed_supplied_values() {
        assert_eq!(
            GoalConfig::default().effective_limits(),
            Ok(GoalBudgetLimits {
                token_limit: None,
                cost_limit_usd: None
            })
        );
        assert_eq!(
            GoalConfig {
                default_cost_limit_usd: Some(-1.0),
                ..GoalConfig::default()
            }
            .validate(),
            Err(GoalConfigError::InvalidCostLimit)
        );
        assert_eq!(
            GoalConfig {
                default_token_limit: Some(i64::MAX as u64 + 1),
                ..GoalConfig::default()
            }
            .validate(),
            Err(GoalConfigError::TokenLimitOutOfRange)
        );
    }

    #[test]
    fn enabled_config_requires_both_defaults_and_a_verifier() {
        let mut config = GoalConfig {
            enabled: true,
            ..GoalConfig::default()
        };
        assert_eq!(config.validate(), Err(GoalConfigError::MissingTokenLimit));
        config.default_token_limit = Some(0);
        assert_eq!(config.validate(), Err(GoalConfigError::MissingCostLimit));
        config.default_cost_limit_usd = Some(0.0);
        assert_eq!(
            config.validate(),
            Err(GoalConfigError::MissingVerifierProvider)
        );
        config.verifier.model_provider = ModelProviderRef::new("openai.default");
        assert_eq!(
            config.effective_limits(),
            Ok(GoalBudgetLimits {
                token_limit: None,
                cost_limit_usd: None
            })
        );
    }

    #[test]
    fn positive_values_remain_finite_limits() {
        let config = GoalConfig {
            enabled: true,
            default_token_limit: Some(42),
            default_cost_limit_usd: Some(1.5),
            verifier: GoalVerifierConfig {
                model_provider: ModelProviderRef::new("openai.default"),
                model: Some("gpt-test".into()),
            },
        };
        assert_eq!(
            config.effective_limits(),
            Ok(GoalBudgetLimits {
                token_limit: Some(42),
                cost_limit_usd: Some(1.5)
            })
        );
    }

    #[test]
    fn enabled_goal_verifier_must_resolve_through_existing_provider_config() {
        let config: Config = toml::from_str(
            r#"
                [goal]
                enabled = true
                default_token_limit = 0
                default_cost_limit_usd = 0.0

                [goal.verifier]
                model_provider = "openai.default"

                [providers.models.openai.default]
                model = "gpt-test"
            "#,
        )
        .expect("configuration parses");
        assert!(config.validate().is_ok());

        let unresolved: Config = toml::from_str(
            r#"
                [goal]
                enabled = true
                default_token_limit = 0
                default_cost_limit_usd = 0.0

                [goal.verifier]
                model_provider = "openai.missing"
            "#,
        )
        .expect("configuration parses");
        assert!(
            unresolved
                .validate()
                .expect_err("unresolved verifier must fail")
                .to_string()
                .contains("goal.verifier.model_provider")
        );
    }
}
