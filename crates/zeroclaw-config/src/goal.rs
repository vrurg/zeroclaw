//! Goal Mode admission, budget, verifier, and channel prompt configuration.

use serde::{Deserialize, Serialize};
use zeroclaw_macros::Configurable;

use crate::providers::ModelProviderRef;

/// Normalized limits used by Goal Mode. `None` means unlimited in a dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GoalBudgetLimits {
    pub token_limit: Option<u64>,
    pub cost_limit_usd: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalConfigError {
    MissingVerifierProvider,
    TokenLimitOutOfRange,
    InvalidCostLimit,
    EmptyVerifierModel,
}

/// The existing model-provider profile selected for the mandatory verifier.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Configurable)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub struct GoalVerifierConfig {
    #[serde(default)]
    pub model_provider: ModelProviderRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Default-closed Goal Mode configuration. Explicit zero defaults mean
/// unlimited. When the `[goal]` section is present, both defaults are required
/// so Goal Mode cannot acquire a hidden hard-coded budget.
#[derive(Debug, Clone, Serialize, Deserialize, Configurable)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[prefix = "goal"]
pub struct GoalConfig {
    #[serde(default)]
    pub enabled: bool,
    pub default_token_limit: u64,
    pub default_cost_limit_usd: f64,
    /// Include ordinary channel turn context in the system message of Matrix
    /// Goal parent turns. Defaults to `true`; `false` narrows ordinary prompt
    /// reuse by omitting that context. ZeroCode and verifier turns are unchanged.
    /// Resolved once per parent turn; reload affects the next parent turn.
    /// Including changing context may affect provider prompt-cache reuse.
    #[serde(default = "default_true")]
    pub include_channel_turn_context: bool,
    #[serde(default)]
    #[nested]
    pub verifier: GoalVerifierConfig,
}

fn default_true() -> bool {
    true
}

impl Default for GoalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_token_limit: 0,
            default_cost_limit_usd: 0.0,
            include_channel_turn_context: true,
            verifier: GoalVerifierConfig::default(),
        }
    }
}

impl GoalConfig {
    /// Validate values locally. `Config::validate` additionally checks that a
    /// configured verifier reference resolves through `providers.models`.
    pub fn validate(&self) -> Result<(), GoalConfigError> {
        if self.default_token_limit > i64::MAX as u64 {
            return Err(GoalConfigError::TokenLimitOutOfRange);
        }
        if !self.default_cost_limit_usd.is_finite() || self.default_cost_limit_usd < 0.0 {
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
        if self.enabled && self.verifier.model_provider.as_str().trim().is_empty() {
            return Err(GoalConfigError::MissingVerifierProvider);
        }
        Ok(())
    }

    /// Normalize explicit zero configuration defaults to unlimited.
    pub fn effective_limits(&self) -> Result<GoalBudgetLimits, GoalConfigError> {
        self.validate()?;
        Ok(GoalBudgetLimits {
            token_limit: (self.default_token_limit != 0).then_some(self.default_token_limit),
            cost_limit_usd: (self.default_cost_limit_usd != 0.0)
                .then_some(self.default_cost_limit_usd),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Config;

    #[test]
    fn channel_turn_context_defaults_and_round_trips() {
        assert!(GoalConfig::default().include_channel_turn_context);
        assert!(Config::default().goal.include_channel_turn_context);
        assert!(
            toml::from_str::<Config>("")
                .unwrap()
                .goal
                .include_channel_turn_context
        );
        let section = "[goal]\ndefault_token_limit = 0\ndefault_cost_limit_usd = 0.0\n";
        assert!(
            toml::from_str::<Config>(section)
                .unwrap()
                .goal
                .include_channel_turn_context
        );
        for include in [false, true] {
            let config: Config = toml::from_str(&format!(
                "{section}include_channel_turn_context = {include}\n"
            ))
            .unwrap();
            assert_eq!(config.goal.include_channel_turn_context, include);
            let encoded = toml::to_string(&config).unwrap();
            let decoded: Config = toml::from_str(&encoded).unwrap();
            assert_eq!(decoded.goal.include_channel_turn_context, include);
        }
        for invalid in ["\"yes\"", "1", "\"false\""] {
            assert!(
                toml::from_str::<Config>(&format!(
                    "{section}include_channel_turn_context = {invalid}\n"
                ))
                .is_err()
            );
        }
    }

    #[test]
    fn channel_turn_context_does_not_change_validation() {
        for enabled in [false, true] {
            let mut config: Config = toml::from_str(&format!(
                "[goal]\nenabled = {enabled}\ndefault_token_limit = 0\n\
                 default_cost_limit_usd = 0.0\n[goal.verifier]\n\
                 model_provider = \"openai.default\"\n\
                 [providers.models.openai.default]\nmodel = \"gpt-test\"\n"
            ))
            .unwrap();
            for include in [false, true] {
                config.goal.include_channel_turn_context = include;
                assert_eq!(config.goal.validate(), Ok(()));
                assert!(config.validate().is_ok());
                config.goal.default_cost_limit_usd = -1.0;
                assert_eq!(
                    config.goal.validate(),
                    Err(GoalConfigError::InvalidCostLimit)
                );
                assert!(config.validate().is_err());
                config.goal.default_cost_limit_usd = 0.0;
            }
        }
    }

    #[test]
    fn enabled_config_requires_a_verifier() {
        let mut config = GoalConfig {
            enabled: true,
            ..GoalConfig::default()
        };
        assert_eq!(
            config.validate(),
            Err(GoalConfigError::MissingVerifierProvider)
        );
        config.verifier.model_provider = ModelProviderRef::new("openai.default");
        assert_eq!(
            config.effective_limits(),
            Ok(GoalBudgetLimits {
                token_limit: None,
                cost_limit_usd: None,
            })
        );
    }

    #[test]
    fn an_explicit_goal_section_requires_both_budget_defaults() {
        let missing_token_limit = toml::from_str::<GoalConfig>(
            r#"
                enabled = true
                default_cost_limit_usd = 0.0
            "#,
        );
        assert!(missing_token_limit.is_err());

        let missing_cost_limit = toml::from_str::<GoalConfig>(
            r#"
                enabled = true
                default_token_limit = 0
            "#,
        );
        assert!(missing_cost_limit.is_err());
    }

    #[test]
    fn rejects_malformed_defaults_even_while_disabled() {
        assert_eq!(
            GoalConfig {
                default_cost_limit_usd: -1.0,
                ..GoalConfig::default()
            }
            .validate(),
            Err(GoalConfigError::InvalidCostLimit)
        );
        assert_eq!(
            GoalConfig {
                default_token_limit: i64::MAX as u64 + 1,
                ..GoalConfig::default()
            }
            .validate(),
            Err(GoalConfigError::TokenLimitOutOfRange)
        );
    }

    #[test]
    fn enabled_verifier_must_resolve_through_existing_provider_config() {
        let valid: Config = toml::from_str(
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
        assert!(valid.validate().is_ok());

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

    #[test]
    fn toml_round_trip_preserves_explicit_unlimited_defaults() {
        let source = r#"
            enabled = true
            default_token_limit = 0
            default_cost_limit_usd = 0.0

            [verifier]
            model_provider = "openai.default"
            model = "gpt-test"
        "#;
        let config: GoalConfig = toml::from_str(source).expect("Goal config parses");
        assert_eq!(config.default_token_limit, 0);
        assert_eq!(config.default_cost_limit_usd, 0.0);
        assert_eq!(
            config.effective_limits(),
            Ok(GoalBudgetLimits {
                token_limit: None,
                cost_limit_usd: None,
            })
        );

        let encoded = toml::to_string(&config).expect("Goal config serializes");
        let decoded: GoalConfig = toml::from_str(&encoded).expect("serialized Goal config parses");
        assert_eq!(decoded.default_token_limit, 0);
        assert_eq!(decoded.default_cost_limit_usd, 0.0);
        assert_eq!(decoded.verifier.model_provider.as_str(), "openai.default");
        assert_eq!(decoded.verifier.model.as_deref(), Some("gpt-test"));
    }
}
