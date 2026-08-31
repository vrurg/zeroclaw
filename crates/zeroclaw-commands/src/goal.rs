//! Transport-neutral, strict `/goal` command grammar.

use crate::CommandSurface;

pub const MAX_GOAL_OBJECTIVE_CHARS: usize = 4096;

/// `Some(0)` is explicit unlimited; `None` means omitted from this command.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GoalBudgetOverrides {
    pub token_limit: Option<u64>,
    pub cost_limit_usd: Option<f64>,
}

impl GoalBudgetOverrides {
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.token_limit.is_none() && self.cost_limit_usd.is_none()
    }
}

/// A parsed command contains no authority, route, task, or lifecycle facts.
#[derive(Debug, Clone, PartialEq)]
pub enum GoalCommand {
    Start {
        overrides: GoalBudgetOverrides,
        objective: String,
    },
    Status,
    Budget {
        set: Option<GoalBudgetOverrides>,
    },
    Pause,
    Resume,
    Cancel,
    Help,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoalCommandParseError {
    NotGoalCommand,
    MissingSubcommand,
    UnknownSubcommand(String),
    MissingObjectiveDelimiter,
    MissingObjective,
    MissingFlagValue(&'static str),
    InvalidTokenLimit(String),
    InvalidCostLimit(String),
    DuplicateFlag(&'static str),
    UnknownFlag(String),
    UnexpectedArguments(&'static str),
    TextTooLong { max: usize },
}

pub fn parse_goal_command(
    content: &str,
    _surface: CommandSurface,
) -> Result<GoalCommand, GoalCommandParseError> {
    // Preserve the opaque start-objective payload. Only command grammar owns
    // leading whitespace before `/goal` and between command tokens.
    let (token, rest) = split_token(content.trim_start());
    if !token.eq_ignore_ascii_case("/goal") {
        return Err(GoalCommandParseError::NotGoalCommand);
    }
    let (subcommand, arguments) = split_token(rest.trim_start());
    if subcommand.is_empty() {
        return Err(GoalCommandParseError::MissingSubcommand);
    }
    match subcommand.to_ascii_lowercase().as_str() {
        "start" => parse_start(arguments),
        "status" => argument_free(arguments, "status", GoalCommand::Status),
        "budget" => parse_budget(arguments),
        "pause" => argument_free(arguments, "pause", GoalCommand::Pause),
        "resume" => argument_free(arguments, "resume", GoalCommand::Resume),
        "cancel" => argument_free(arguments, "cancel", GoalCommand::Cancel),
        "help" => argument_free(arguments, "help", GoalCommand::Help),
        _ => Err(GoalCommandParseError::UnknownSubcommand(
            subcommand.to_ascii_lowercase(),
        )),
    }
}

fn split_token(input: &str) -> (&str, &str) {
    input
        .split_once(char::is_whitespace)
        .map_or((input, ""), |(a, b)| (a, b))
}

fn parse_start(arguments: &str) -> Result<GoalCommand, GoalCommandParseError> {
    let Some(delimiter) = arguments.match_indices("--").find_map(|(index, _)| {
        let before = arguments[..index].chars().next_back();
        let after = arguments[index + 2..].chars().next();
        (before.is_none_or(char::is_whitespace) && after.is_none_or(char::is_whitespace))
            .then_some(index)
    }) else {
        return Err(GoalCommandParseError::MissingObjectiveDelimiter);
    };
    let (flags, objective) = arguments.split_at(delimiter);
    let objective = &objective[2..];
    // The delimiter needs one whitespace separator before an objective. Consume
    // precisely that separator, retaining every other payload byte verbatim.
    let separator_len = objective
        .chars()
        .next()
        .filter(|character| character.is_whitespace())
        .map_or(0, char::len_utf8);
    let objective = &objective[separator_len..];
    if objective.trim().is_empty() {
        return Err(GoalCommandParseError::MissingObjective);
    }
    if objective.chars().count() > MAX_GOAL_OBJECTIVE_CHARS {
        return Err(GoalCommandParseError::TextTooLong {
            max: MAX_GOAL_OBJECTIVE_CHARS,
        });
    }
    Ok(GoalCommand::Start {
        overrides: parse_overrides(flags)?,
        objective: objective.to_string(),
    })
}

fn parse_budget(arguments: &str) -> Result<GoalCommand, GoalCommandParseError> {
    let (subcommand, flags) = split_token(arguments.trim());
    if subcommand.is_empty() {
        return Ok(GoalCommand::Budget { set: None });
    }
    if !subcommand.eq_ignore_ascii_case("set") {
        return Err(GoalCommandParseError::UnexpectedArguments("budget"));
    }
    let overrides = parse_overrides(flags)?;
    if overrides.is_empty() {
        return Err(GoalCommandParseError::UnexpectedArguments("budget set"));
    }
    Ok(GoalCommand::Budget {
        set: Some(overrides),
    })
}

fn argument_free(
    arguments: &str,
    name: &'static str,
    command: GoalCommand,
) -> Result<GoalCommand, GoalCommandParseError> {
    if arguments.trim().is_empty() {
        Ok(command)
    } else {
        Err(GoalCommandParseError::UnexpectedArguments(name))
    }
}

fn parse_overrides(flags: &str) -> Result<GoalBudgetOverrides, GoalCommandParseError> {
    let mut result = GoalBudgetOverrides {
        token_limit: None,
        cost_limit_usd: None,
    };
    let mut words = flags.split_whitespace();
    while let Some(flag) = words.next() {
        let value = words.next().ok_or_else(|| match flag {
            "--tokens" => GoalCommandParseError::MissingFlagValue("--tokens"),
            "--cost-usd" => GoalCommandParseError::MissingFlagValue("--cost-usd"),
            _ => GoalCommandParseError::UnknownFlag(flag.to_string()),
        })?;
        match flag {
            "--tokens" => {
                if result.token_limit.is_some() {
                    return Err(GoalCommandParseError::DuplicateFlag("--tokens"));
                }
                let value: u64 = value
                    .parse()
                    .map_err(|_| GoalCommandParseError::InvalidTokenLimit(value.to_string()))?;
                if value > i64::MAX as u64 {
                    return Err(GoalCommandParseError::InvalidTokenLimit(value.to_string()));
                }
                result.token_limit = Some(value);
            }
            "--cost-usd" => {
                if result.cost_limit_usd.is_some() {
                    return Err(GoalCommandParseError::DuplicateFlag("--cost-usd"));
                }
                let value: f64 = value
                    .parse()
                    .map_err(|_| GoalCommandParseError::InvalidCostLimit(value.to_string()))?;
                if !value.is_finite() || value < 0.0 {
                    return Err(GoalCommandParseError::InvalidCostLimit(value.to_string()));
                }
                result.cost_limit_usd = Some(value);
            }
            _ => return Err(GoalCommandParseError::UnknownFlag(flag.to_string())),
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CommandSurface;

    #[test]
    fn parses_fixed_v1_grammar() {
        let surface = CommandSurface::Channel;
        assert_eq!(
            parse_goal_command("/goal start -- objective", surface),
            Ok(GoalCommand::Start {
                overrides: GoalBudgetOverrides {
                    token_limit: None,
                    cost_limit_usd: None
                },
                objective: "objective".into()
            })
        );
        assert_eq!(
            parse_goal_command(
                "/goal start --tokens 0 --cost-usd 1.5 -- objective",
                surface
            ),
            Ok(GoalCommand::Start {
                overrides: GoalBudgetOverrides {
                    token_limit: Some(0),
                    cost_limit_usd: Some(1.5)
                },
                objective: "objective".into()
            })
        );
        assert_eq!(
            parse_goal_command("/goal budget set --tokens 9", surface),
            Ok(GoalCommand::Budget {
                set: Some(GoalBudgetOverrides {
                    token_limit: Some(9),
                    cost_limit_usd: None
                })
            })
        );
        for command in [
            "/goal status",
            "/goal budget",
            "/goal pause",
            "/goal resume",
            "/goal cancel",
            "/goal help",
        ] {
            assert!(parse_goal_command(command, surface).is_ok(), "{command}");
        }
    }

    #[test]
    fn parses_limits_and_control_commands_without_transport_policy() {
        let surface = CommandSurface::Channel;
        let cases = [
            (
                "\t/GOAL\tstart\t--tokens\t9223372036854775807\t--cost-usd\t-0\t--\tfinish",
                GoalCommand::Start {
                    overrides: GoalBudgetOverrides {
                        token_limit: Some(i64::MAX as u64),
                        cost_limit_usd: Some(-0.0),
                    },
                    objective: "finish".into(),
                },
            ),
            ("/goal budget", GoalCommand::Budget { set: None }),
            (
                "/goal budget set --tokens 0",
                GoalCommand::Budget {
                    set: Some(GoalBudgetOverrides {
                        token_limit: Some(0),
                        cost_limit_usd: None,
                    }),
                },
            ),
            (
                "/goal budget set --cost-usd 2.5",
                GoalCommand::Budget {
                    set: Some(GoalBudgetOverrides {
                        token_limit: None,
                        cost_limit_usd: Some(2.5),
                    }),
                },
            ),
        ];

        for (input, expected) in cases {
            assert_eq!(parse_goal_command(input, surface), Ok(expected), "{input}");
        }
    }

    #[test]
    fn rejects_retired_or_ambiguous_syntax() {
        let surface = CommandSurface::Channel;
        for command in [
            "/goal start objective",
            "/goal start --",
            "/goal objective change",
            "/goal resume note",
            "/goal start --unlimited -- x",
            "/goal start --tokens=1 -- x",
            "/goal start --tokens 1 --tokens 2 -- x",
            "/goal budget set",
            "/goal budget set --cost-usd -1",
        ] {
            assert!(parse_goal_command(command, surface).is_err(), "{command}");
        }
    }

    #[test]
    fn rejects_out_of_range_and_malformed_limit_forms() {
        let surface = CommandSurface::Channel;
        let cases = [
            "/goal start --tokens 9223372036854775808 -- objective",
            "/goal start --tokens 18446744073709551616 -- objective",
            "/goal start --cost-usd NaN -- objective",
            "/goal start --cost-usd inf -- objective",
            "/goal start --cost-usd -inf -- objective",
            "/goal start --cost-usd -0.1 -- objective",
            "/goal start --tokens=1 -- objective",
            "/goal start --cost-usd=1 -- objective",
            "/goal start --tokens 1 --tokens 2 -- objective",
            "/goal start --cost-usd 1 --cost-usd 2 -- objective",
            "/goal start --tokens -- objective",
            "/goal budget set --tokens 1 extra",
            "/goal budget set --tokens=1",
        ];

        for input in cases {
            assert!(parse_goal_command(input, surface).is_err(), "{input}");
        }
    }

    #[test]
    fn preserves_the_exact_objective_payload_after_the_delimiter() {
        let surface = CommandSurface::Channel;
        let command = "/goal start --  \u{2003}finish -- exactly\u{2002}";

        assert_eq!(
            parse_goal_command(command, surface),
            Ok(GoalCommand::Start {
                overrides: GoalBudgetOverrides {
                    token_limit: None,
                    cost_limit_usd: None,
                },
                objective: " \u{2003}finish -- exactly\u{2002}".into(),
            })
        );
    }
}
