//! Transport-neutral `/goal` subcommand grammar.
//!
//! This parser produces data only. It establishes no authority: surface
//! eligibility, operator opt-in, agent eligibility, principal trust, session
//! binding, and lifecycle are decided by the runtime goal controller.

use crate::{BuiltinCommandId, CommandSurface, parse_command_token};

/// Maximum objective or resume-note length in characters, not bytes.
pub const MAX_GOAL_TEXT_CHARS: usize = 4096;

/// Parsed `/goal` subcommand without lifecycle or authority semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoalCommand {
    /// Start a Goal with the durable objective text.
    Start { objective: String },
    /// Parse the currently advertised objective amendment syntax without
    /// assigning it controller semantics.
    Objective { objective: String },
    /// Read the current Goal status.
    Status,
    /// Read the current Goal budget projection.
    Budget,
    /// Pause the current Goal.
    Pause,
    /// Resume the current Goal, optionally carrying the advertised note.
    Resume { note: Option<String> },
    /// Cancel the current Goal.
    Cancel,
    /// Read Goal command help.
    Help,
}

/// Typed failure while parsing a `/goal` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoalCommandParseError {
    /// Input is not `/goal` on the supplied surface.
    NotGoalCommand,
    /// `/goal` has no subcommand.
    MissingSubcommand,
    /// The echoed subcommand is not part of the advertised grammar.
    UnknownSubcommand(String),
    /// `start` or `objective` has no objective text.
    MissingObjective(&'static str),
    /// An argument-free subcommand was given trailing text.
    UnexpectedArguments(&'static str),
    /// Objective or note exceeds [`MAX_GOAL_TEXT_CHARS`].
    TextTooLong { max: usize },
}

/// Parse a raw command string for the supplied catalogue surface.
pub fn parse_goal_command(
    content: &str,
    surface: CommandSurface,
) -> Result<GoalCommand, GoalCommandParseError> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Err(GoalCommandParseError::NotGoalCommand);
    }

    let (token, rest) = split_token(trimmed);
    let Some(parsed) = parse_command_token(token, surface) else {
        return Err(GoalCommandParseError::NotGoalCommand);
    };
    if parsed.command.id != BuiltinCommandId::Goal {
        return Err(GoalCommandParseError::NotGoalCommand);
    }

    let rest = rest.trim();
    if rest.is_empty() {
        return Err(GoalCommandParseError::MissingSubcommand);
    }

    let (subcommand, arguments) = split_token(rest);
    let arguments = arguments.trim();
    match subcommand.to_ascii_lowercase().as_str() {
        "start" => {
            parse_objective(arguments, "start").map(|objective| GoalCommand::Start { objective })
        }
        "objective" => parse_objective(arguments, "objective")
            .map(|objective| GoalCommand::Objective { objective }),
        "status" => parse_argument_free(arguments, "status", GoalCommand::Status),
        "budget" => parse_argument_free(arguments, "budget", GoalCommand::Budget),
        "pause" => parse_argument_free(arguments, "pause", GoalCommand::Pause),
        "resume" => parse_resume(arguments),
        "cancel" => parse_argument_free(arguments, "cancel", GoalCommand::Cancel),
        "help" => parse_argument_free(arguments, "help", GoalCommand::Help),
        _ => Err(GoalCommandParseError::UnknownSubcommand(
            subcommand.to_ascii_lowercase(),
        )),
    }
}

fn split_token(input: &str) -> (&str, &str) {
    input
        .split_once(char::is_whitespace)
        .map_or((input, ""), |(token, rest)| (token, rest))
}

fn parse_objective(
    arguments: &str,
    subcommand: &'static str,
) -> Result<String, GoalCommandParseError> {
    if arguments.is_empty() {
        return Err(GoalCommandParseError::MissingObjective(subcommand));
    }
    parse_bounded_text(arguments)
}

fn parse_argument_free(
    arguments: &str,
    subcommand: &'static str,
    command: GoalCommand,
) -> Result<GoalCommand, GoalCommandParseError> {
    if arguments.is_empty() {
        Ok(command)
    } else {
        Err(GoalCommandParseError::UnexpectedArguments(subcommand))
    }
}

fn parse_resume(arguments: &str) -> Result<GoalCommand, GoalCommandParseError> {
    if arguments.is_empty() {
        Ok(GoalCommand::Resume { note: None })
    } else {
        parse_bounded_text(arguments).map(|note| GoalCommand::Resume { note: Some(note) })
    }
}

fn parse_bounded_text(arguments: &str) -> Result<String, GoalCommandParseError> {
    if arguments.chars().count() > MAX_GOAL_TEXT_CHARS {
        Err(GoalCommandParseError::TextTooLong {
            max: MAX_GOAL_TEXT_CHARS,
        })
    } else {
        Ok(arguments.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{GoalCommand, GoalCommandParseError, MAX_GOAL_TEXT_CHARS, parse_goal_command};
    use crate::CommandSurface;

    #[test]
    fn parses_start_objective_after_catalogue_resolution() {
        assert_eq!(
            parse_goal_command(
                "/goal start keep the interior  spaces",
                CommandSurface::Channel
            ),
            Ok(GoalCommand::Start {
                objective: "keep the interior  spaces".to_string(),
            })
        );
    }

    #[test]
    fn parser_inherits_catalogue_token_normalization() {
        assert_eq!(
            parse_goal_command(
                " /GOAL@zeroclaw_bot START Preserve Case ",
                CommandSurface::Channel
            ),
            Ok(GoalCommand::Start {
                objective: "Preserve Case".to_string(),
            })
        );
    }

    #[test]
    fn parse_rejects_goal_on_tui_and_web_surfaces() {
        for surface in [CommandSurface::Tui, CommandSurface::Web] {
            assert_eq!(
                parse_goal_command("/goal status", surface),
                Err(GoalCommandParseError::NotGoalCommand)
            );
        }
    }

    #[test]
    fn parser_rejects_non_goal_and_missing_subcommand_inputs() {
        assert_eq!(
            parse_goal_command("", CommandSurface::Channel),
            Err(GoalCommandParseError::NotGoalCommand)
        );
        assert_eq!(
            parse_goal_command("/help", CommandSurface::Channel),
            Err(GoalCommandParseError::NotGoalCommand)
        );
        assert_eq!(
            parse_goal_command("/goalx start x", CommandSurface::Channel),
            Err(GoalCommandParseError::NotGoalCommand)
        );
        assert_eq!(
            parse_goal_command("/goal", CommandSurface::Channel),
            Err(GoalCommandParseError::MissingSubcommand)
        );
    }

    #[test]
    fn parser_accepts_all_advertised_subcommands() {
        let surface = CommandSurface::Channel;
        assert_eq!(
            parse_goal_command("/goal status", surface),
            Ok(GoalCommand::Status)
        );
        assert_eq!(
            parse_goal_command("/goal budget", surface),
            Ok(GoalCommand::Budget)
        );
        assert_eq!(
            parse_goal_command("/goal pause", surface),
            Ok(GoalCommand::Pause)
        );
        assert_eq!(
            parse_goal_command("/goal resume", surface),
            Ok(GoalCommand::Resume { note: None })
        );
        assert_eq!(
            parse_goal_command("/goal resume maintenance complete", surface),
            Ok(GoalCommand::Resume {
                note: Some("maintenance complete".to_string()),
            })
        );
        assert_eq!(
            parse_goal_command("/goal cancel", surface),
            Ok(GoalCommand::Cancel)
        );
        assert_eq!(
            parse_goal_command("/goal help", surface),
            Ok(GoalCommand::Help)
        );
        assert_eq!(
            parse_goal_command("/goal objective revised target", surface),
            Ok(GoalCommand::Objective {
                objective: "revised target".to_string(),
            })
        );
    }

    #[test]
    fn parser_rejects_missing_or_unknown_objective_subcommands() {
        let surface = CommandSurface::Channel;
        assert_eq!(
            parse_goal_command("/goal start", surface),
            Err(GoalCommandParseError::MissingObjective("start"))
        );
        assert_eq!(
            parse_goal_command("/goal objective", surface),
            Err(GoalCommandParseError::MissingObjective("objective"))
        );
        assert_eq!(
            parse_goal_command("/goal retry", surface),
            Err(GoalCommandParseError::UnknownSubcommand(
                "retry".to_string()
            ))
        );
    }

    #[test]
    fn pause_rejects_a_trailing_reason() {
        assert_eq!(
            parse_goal_command("/goal pause for maintenance", CommandSurface::Channel),
            Err(GoalCommandParseError::UnexpectedArguments("pause"))
        );
    }

    #[test]
    fn argument_free_subcommands_reject_trailing_text() {
        for command in ["status", "budget", "cancel", "help"] {
            assert_eq!(
                parse_goal_command(
                    &format!("/goal {command} unexpected"),
                    CommandSurface::Channel
                ),
                Err(GoalCommandParseError::UnexpectedArguments(command))
            );
        }
    }

    #[test]
    fn objective_length_bound_counts_characters_not_bytes() {
        let accepted = "界".repeat(MAX_GOAL_TEXT_CHARS);
        let rejected = "界".repeat(MAX_GOAL_TEXT_CHARS + 1);

        assert_eq!(
            parse_goal_command(&format!("/goal start {accepted}"), CommandSurface::Channel),
            Ok(GoalCommand::Start {
                objective: accepted
            })
        );
        assert_eq!(
            parse_goal_command(&format!("/goal start {rejected}"), CommandSurface::Channel),
            Err(GoalCommandParseError::TextTooLong {
                max: MAX_GOAL_TEXT_CHARS,
            })
        );
    }

    #[test]
    fn multiline_objective_preserves_interior_newlines_verbatim() {
        assert_eq!(
            parse_goal_command(
                "/goal start first line\nsecond line",
                CommandSurface::Channel
            ),
            Ok(GoalCommand::Start {
                objective: "first line\nsecond line".to_string(),
            })
        );
    }
}
