//! Transport-neutral, dormant Goal Mode controller contract.
//!
//! This module owns the shared command grammar and trusted ingress boundary.
//! Concrete command surfaces and provider execution remain later stages.

/// Parsed Goal Mode commands accepted by the V1 grammar.
#[derive(Debug, Clone, PartialEq)]
pub enum GoalCommand {
    Start {
        budget: GoalBudgetRequest,
        success_criteria: String,
        objective: String,
    },
    Status,
    Budget,
    SetBudget(GoalBudgetRequest),
    Pause,
    Resume,
    Cancel,
    Help,
}

/// The complete two-dimensional budget selection carried by a command.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GoalBudgetRequest {
    /// Copy both configured defaults. Defaults may not silently mean unlimited.
    Defaults,
    /// Replace both dimensions; an omitted dimension is unlimited.
    Limits {
        token_limit: Option<u64>,
        cost_limit_usd: Option<f64>,
    },
    /// The explicit, and only, unlimited selection.
    Unlimited,
}

/// Stable parser failure keys. A later adapter renders them through Fluent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalCommandErrorKey {
    InvalidGrammar,
    ObjectiveRequired,
    SuccessCriteriaRequired,
    InvalidBudget,
    InvalidArguments,
}

/// Typed parser error that keeps admission rules independent of presentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalCommandParseError {
    pub key: GoalCommandErrorKey,
}

impl std::fmt::Display for GoalCommandParseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self.key {
            GoalCommandErrorKey::InvalidGrammar => "goal_invalid_grammar",
            GoalCommandErrorKey::ObjectiveRequired => "goal_objective_required",
            GoalCommandErrorKey::SuccessCriteriaRequired => "goal_success_criteria_required",
            GoalCommandErrorKey::InvalidBudget => "goal_invalid_budget",
            GoalCommandErrorKey::InvalidArguments => "goal_invalid_arguments",
        })
    }
}

impl std::error::Error for GoalCommandParseError {}

/// Parse one raw Goal Mode command using the exact V1 grammar.
pub fn parse_goal_command(input: &str) -> Result<GoalCommand, GoalCommandParseError> {
    let command = input
        .trim_start()
        .strip_prefix("/goal")
        .filter(|suffix| {
            suffix.is_empty() || suffix.chars().next().is_some_and(char::is_whitespace)
        })
        .ok_or_else(invalid_grammar)?
        .trim_start();
    let (action, remainder) = take_token(command).ok_or_else(invalid_grammar)?;
    match action {
        "start" => parse_start(remainder),
        "status" if remainder.trim().is_empty() => Ok(GoalCommand::Status),
        "budget" => parse_budget_command(remainder),
        "pause" if remainder.trim().is_empty() => Ok(GoalCommand::Pause),
        "resume" if remainder.trim().is_empty() => Ok(GoalCommand::Resume),
        "cancel" if remainder.trim().is_empty() => Ok(GoalCommand::Cancel),
        "help" if remainder.trim().is_empty() => Ok(GoalCommand::Help),
        _ => Err(invalid_arguments()),
    }
}

fn parse_start(input: &str) -> Result<GoalCommand, GoalCommandParseError> {
    let mut remainder = input;
    let mut flags = Vec::new();
    loop {
        let (token, rest) = take_token(remainder).ok_or_else(success_criteria_required)?;
        if token == "--success-criteria" {
            let Some((criteria, objective)) = split_objective_delimiter(rest) else {
                return Err(objective_required());
            };
            if criteria.trim().is_empty() {
                return Err(success_criteria_required());
            }
            if criteria
                .split_whitespace()
                .any(|part| part.starts_with("--"))
            {
                return Err(invalid_arguments());
            }
            if objective.trim().is_empty() {
                return Err(objective_required());
            }
            return Ok(GoalCommand::Start {
                budget: parse_budget_flags(&flags, true)?,
                success_criteria: criteria.to_owned(),
                objective: objective.to_owned(),
            });
        }
        if token == "--" {
            return Err(success_criteria_required());
        }
        flags.push(token);
        remainder = rest;
    }
}

fn split_objective_delimiter(input: &str) -> Option<(&str, &str)> {
    input.match_indices("--").find_map(|(index, _)| {
        let before = &input[..index];
        let after = &input[index + 2..];
        (before.chars().last().is_none_or(char::is_whitespace)
            && after.chars().next().is_none_or(char::is_whitespace))
        .then_some((before, after))
    })
}

fn parse_budget_command(input: &str) -> Result<GoalCommand, GoalCommandParseError> {
    if input.trim().is_empty() {
        return Ok(GoalCommand::Budget);
    }
    let (action, rest) = take_token(input).ok_or_else(invalid_arguments)?;
    if action != "set" {
        return Err(invalid_arguments());
    }
    Ok(GoalCommand::SetBudget(parse_budget_flags(
        &rest.split_whitespace().collect::<Vec<_>>(),
        false,
    )?))
}

fn parse_budget_flags(
    flags: &[&str],
    allow_defaults: bool,
) -> Result<GoalBudgetRequest, GoalCommandParseError> {
    if flags.is_empty() {
        return allow_defaults
            .then_some(GoalBudgetRequest::Defaults)
            .ok_or_else(invalid_budget);
    }
    let mut index = 0;
    let mut unlimited = false;
    let mut token_limit = None;
    let mut cost_limit_usd = None;
    while index < flags.len() {
        match flags[index] {
            "--unlimited" if !unlimited && token_limit.is_none() && cost_limit_usd.is_none() => {
                unlimited = true;
                index += 1;
            }
            "--tokens" if !unlimited && token_limit.is_none() => {
                token_limit = Some(parse_token_limit(
                    flags.get(index + 1).ok_or_else(invalid_budget)?,
                )?);
                index += 2;
            }
            "--cost-usd" if !unlimited && cost_limit_usd.is_none() => {
                cost_limit_usd = Some(parse_cost_limit(
                    flags.get(index + 1).ok_or_else(invalid_budget)?,
                )?);
                index += 2;
            }
            _ => return Err(invalid_arguments()),
        }
    }
    if unlimited {
        Ok(GoalBudgetRequest::Unlimited)
    } else {
        Ok(GoalBudgetRequest::Limits {
            token_limit,
            cost_limit_usd,
        })
    }
}

fn parse_token_limit(value: &str) -> Result<u64, GoalCommandParseError> {
    let value = value.parse::<u64>().map_err(|_| invalid_budget())?;
    if value == 0 || value > i64::MAX as u64 {
        Err(invalid_budget())
    } else {
        Ok(value)
    }
}

fn parse_cost_limit(value: &str) -> Result<f64, GoalCommandParseError> {
    let value = value.parse::<f64>().map_err(|_| invalid_budget())?;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(invalid_budget())
    }
}

fn take_token(input: &str) -> Option<(&str, &str)> {
    let input = input.trim_start();
    (!input.is_empty()).then(|| {
        let end = input
            .find(|character: char| character.is_whitespace())
            .unwrap_or(input.len());
        (&input[..end], &input[end..])
    })
}

fn error(key: GoalCommandErrorKey) -> GoalCommandParseError {
    GoalCommandParseError { key }
}
fn invalid_grammar() -> GoalCommandParseError {
    error(GoalCommandErrorKey::InvalidGrammar)
}
fn objective_required() -> GoalCommandParseError {
    error(GoalCommandErrorKey::ObjectiveRequired)
}
fn success_criteria_required() -> GoalCommandParseError {
    error(GoalCommandErrorKey::SuccessCriteriaRequired)
}
fn invalid_budget() -> GoalCommandParseError {
    error(GoalCommandErrorKey::InvalidBudget)
}
fn invalid_arguments() -> GoalCommandParseError {
    error(GoalCommandErrorKey::InvalidArguments)
}

/// Trusted, server-derived facts binding a Goal command to one session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalIngressContext {
    session_id: String,
    agent: String,
    route: String,
    principal_id: Option<String>,
}

impl GoalIngressContext {
    /// Construct from facts already derived by trusted server-side ingress.
    pub fn trusted(
        session_id: impl Into<String>,
        agent: impl Into<String>,
        route: impl Into<String>,
        principal_id: Option<String>,
    ) -> Result<Self, GoalIngressError> {
        let session_id = required(session_id.into())?;
        let agent = required(agent.into())?;
        let route = required(route.into())?;
        let principal_id = principal_id.map(required).transpose()?;
        Ok(Self {
            session_id,
            agent,
            route,
            principal_id,
        })
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn agent(&self) -> &str {
        &self.agent
    }

    pub fn route(&self) -> &str {
        &self.route
    }

    pub fn durable_principal_id(&self) -> Option<&str> {
        self.principal_id.as_deref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalIngressError {
    MissingBinding,
}

impl std::fmt::Display for GoalIngressError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("goal_ingress_missing_binding")
    }
}

impl std::error::Error for GoalIngressError {}

fn required(value: String) -> Result<String, GoalIngressError> {
    if value.trim().is_empty() {
        Err(GoalIngressError::MissingBinding)
    } else {
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_exact_start_budget_forms() {
        assert_eq!(
            parse_goal_command(
                "/goal start --tokens 12 --cost-usd 1.25 --success-criteria done -- objective"
            ),
            Ok(GoalCommand::Start {
                budget: GoalBudgetRequest::Limits {
                    token_limit: Some(12),
                    cost_limit_usd: Some(1.25),
                },
                success_criteria: " done ".into(),
                objective: " objective".into(),
            })
        );
        assert_eq!(
            parse_goal_command("/goal start --unlimited --success-criteria done -- objective"),
            Ok(GoalCommand::Start {
                budget: GoalBudgetRequest::Unlimited,
                success_criteria: " done ".into(),
                objective: " objective".into(),
            })
        );
        assert_eq!(
            parse_goal_command("/goal start --success-criteria done -- objective"),
            Ok(GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: " done ".into(),
                objective: " objective".into(),
            })
        );
    }

    #[test]
    fn rejects_ambiguous_or_incomplete_start_forms() {
        for command in [
            "/goal start -- objective",
            "/goal start --success-criteria -- objective",
            "/goal start --success-criteria done --",
            "/goal start --tokens=4 --success-criteria done -- objective",
            "/goal start --tokens 4 --unlimited --success-criteria done -- objective",
            "/goal start --tokens 0 --success-criteria done -- objective",
            "/goal start --cost-usd NaN --success-criteria done -- objective",
        ] {
            assert!(parse_goal_command(command).is_err(), "{command}");
        }
    }

    #[test]
    fn parses_only_exact_non_start_commands() {
        assert_eq!(parse_goal_command("/goal status"), Ok(GoalCommand::Status));
        assert_eq!(parse_goal_command("/goal budget"), Ok(GoalCommand::Budget));
        assert_eq!(parse_goal_command("/goal pause"), Ok(GoalCommand::Pause));
        assert_eq!(parse_goal_command("/goal resume"), Ok(GoalCommand::Resume));
        assert_eq!(parse_goal_command("/goal cancel"), Ok(GoalCommand::Cancel));
        assert_eq!(parse_goal_command("/goal help"), Ok(GoalCommand::Help));
        assert_eq!(
            parse_goal_command("/goal budget set --tokens 9"),
            Ok(GoalCommand::SetBudget(GoalBudgetRequest::Limits {
                token_limit: Some(9),
                cost_limit_usd: None,
            }))
        );
        assert_eq!(
            parse_goal_command("/goal budget set --unlimited"),
            Ok(GoalCommand::SetBudget(GoalBudgetRequest::Unlimited))
        );
        for command in [
            "/goal objective x",
            "/goal status task-id",
            "/goal help now",
            "/goal budget set",
            "/goal budget set --tokens=9",
            "/goals status",
        ] {
            assert!(parse_goal_command(command).is_err(), "{command}");
        }
    }

    #[test]
    fn ingress_accepts_only_server_derived_nonblank_bindings() {
        assert!(GoalIngressContext::trusted("session", "agent", "route", None).is_ok());
        assert!(GoalIngressContext::trusted(" ", "agent", "route", None).is_err());
        assert!(GoalIngressContext::trusted("session", " ", "route", None).is_err());
        assert!(GoalIngressContext::trusted("session", "agent", " ", None).is_err());
        assert!(
            GoalIngressContext::trusted("session", "agent", "route", Some(" ".into())).is_err()
        );
    }
}
