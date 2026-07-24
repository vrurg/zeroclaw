//! Shared built-in channel slash command catalogue.

use serde::Serialize;

/// User-facing surface where a command can be advertised or accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandSurface {
    /// Command is available from the local CLI.
    Cli,
    /// Command is available from the Web UI/API surface.
    Web,
    /// Command is available from the terminal UI.
    Tui,
    /// Command is available from message-channel ingress.
    Channel,
}

/// Stable built-in command identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltinCommandId {
    /// Show runtime command help.
    Help,
    /// Clear local conversation history.
    Clear,
    /// Start a fresh conversation/session.
    New,
    /// Stop current work where the owning surface supports it.
    Stop,
    /// Show or change the selected model.
    Model,
    /// List configured/known models.
    Models,
    /// Show runtime config visible to the surface.
    Config,
    /// Show or change model thinking/reasoning effort.
    Thinking,
    /// Manage durable goal-mode work.
    Goal,
}

/// Where command execution is owned today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandExecution {
    /// The client surface handles the command without runtime admission.
    ClientLocal,
    /// The channel/runtime command handler owns the command.
    RuntimeCommand,
    /// The durable goal controller/admission path owns the command.
    GoalAdmission,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CommandSpec {
    /// Stable id for code that should not branch on display text.
    pub id: BuiltinCommandId,
    /// Canonical slash command name without the leading slash.
    pub name: &'static str,
    /// Additional names accepted by the same handler.
    pub aliases: &'static [&'static str],
    /// Human-readable usage shape shown in help.
    pub usage: &'static str,
    /// Fluent key for the localized command description.
    pub description_key: &'static str,
    /// Surfaces where this command may be advertised or accepted.
    pub surfaces: &'static [CommandSurface],
    /// Current owner of command execution.
    pub execution: CommandExecution,
}

impl CommandSpec {
    pub fn supports(self, surface: CommandSurface) -> bool {
        self.surfaces.contains(&surface)
    }

    pub fn token_matches(self, token: &str) -> bool {
        self.name == token || self.aliases.contains(&token)
    }

    /// Whether the shared runtime, rather than a client surface, owns execution.
    pub fn is_runtime_owned(self) -> bool {
        matches!(
            self.execution,
            CommandExecution::RuntimeCommand | CommandExecution::GoalAdmission
        )
    }
}

impl CommandSurface {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cli => "cli",
            Self::Web => "web",
            Self::Tui => "tui",
            Self::Channel => "channel",
        }
    }
}

/// Parsed command token before surface-specific argument handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedCommandToken {
    /// Catalogue entry matched by the leading slash token.
    pub command: CommandSpec,
    /// Explicit bot target following `@`, without the leading marker.
    ///
    /// Ingress must validate this against its trusted channel identity before
    /// executing the command. Keeping it here prevents shared parsing from
    /// silently turning `/goal@other_bot` into an unaddressed `/goal`.
    pub target: Option<String>,
}

/// Shared classification for a leading command token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandTokenClassification {
    /// No catalogue command exists for this token on the requested surface.
    Unknown,
    /// A catalogue command was recognized, but its explicit target is invalid.
    MalformedTarget(CommandSpec),
    /// A valid catalogue command with its optional explicit target preserved.
    Valid(ParsedCommandToken),
}

impl ParsedCommandToken {
    /// Whether this command is bare or explicitly addressed to `self_handle`.
    ///
    /// A suffixed command fails closed when the channel cannot provide a
    /// trusted self handle. Channel adapters own identity discovery; the
    /// catalogue owns only normalization and comparison.
    pub fn targets(&self, self_handle: Option<&str>) -> bool {
        let Some(target) = self.target.as_deref() else {
            return true;
        };
        let Some(self_handle) = self_handle else {
            return false;
        };
        let self_target = self_handle
            .trim()
            .strip_prefix('@')
            .unwrap_or(self_handle.trim());
        !self_target.is_empty() && self_target.eq_ignore_ascii_case(target)
    }
}

const CHANNEL_ONLY: &[CommandSurface] = &[CommandSurface::Channel];

static BUILTIN_COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        id: BuiltinCommandId::Help,
        name: "help",
        aliases: &[],
        usage: "/help",
        description_key: "command-help-description",
        surfaces: CHANNEL_ONLY,
        execution: CommandExecution::ClientLocal,
    },
    CommandSpec {
        id: BuiltinCommandId::Clear,
        name: "clear",
        aliases: &[],
        usage: "/clear",
        description_key: "command-clear-description",
        surfaces: CHANNEL_ONLY,
        execution: CommandExecution::RuntimeCommand,
    },
    CommandSpec {
        id: BuiltinCommandId::New,
        name: "new",
        aliases: &["new-session"],
        usage: "/new",
        description_key: "command-new-description",
        surfaces: CHANNEL_ONLY,
        execution: CommandExecution::RuntimeCommand,
    },
    CommandSpec {
        id: BuiltinCommandId::Stop,
        name: "stop",
        aliases: &[],
        usage: "/stop",
        description_key: "command-stop-description",
        surfaces: CHANNEL_ONLY,
        execution: CommandExecution::RuntimeCommand,
    },
    CommandSpec {
        id: BuiltinCommandId::Model,
        name: "model",
        aliases: &[],
        usage: "/model [--user|--agent] [model]",
        description_key: "command-model-description",
        surfaces: CHANNEL_ONLY,
        execution: CommandExecution::RuntimeCommand,
    },
    CommandSpec {
        id: BuiltinCommandId::Models,
        name: "models",
        aliases: &[],
        usage: "/models [provider]",
        description_key: "command-models-description",
        surfaces: CHANNEL_ONLY,
        execution: CommandExecution::RuntimeCommand,
    },
    CommandSpec {
        id: BuiltinCommandId::Config,
        name: "config",
        aliases: &[],
        usage: "/config",
        description_key: "command-config-description",
        surfaces: CHANNEL_ONLY,
        execution: CommandExecution::RuntimeCommand,
    },
    CommandSpec {
        id: BuiltinCommandId::Thinking,
        name: "thinking",
        aliases: &["think"],
        usage: "/thinking [off|low|medium|high|max|reset]",
        description_key: "command-thinking-description",
        surfaces: CHANNEL_ONLY,
        execution: CommandExecution::RuntimeCommand,
    },
    CommandSpec {
        id: BuiltinCommandId::Goal,
        name: "goal",
        aliases: &[],
        usage: "/goal <start <objective>|objective <objective>|status|budget|pause|resume [reason]|cancel|help> ...",
        description_key: "command-goal-description",
        surfaces: CHANNEL_ONLY,
        execution: CommandExecution::GoalAdmission,
    },
];

pub fn builtin_commands() -> &'static [CommandSpec] {
    BUILTIN_COMMANDS
}

pub fn commands_for_surface(
    surface: CommandSurface,
) -> impl Iterator<Item = CommandSpec> + 'static {
    BUILTIN_COMMANDS
        .iter()
        .copied()
        .filter(move |spec| spec.supports(surface))
}

/// Runtime-owned commands that a surface may advertise for the current policy.
///
/// Parsing remains independent from advertisement so a stale remote menu or a
/// hand-written command still reaches the authoritative runtime admission
/// check. Goal admission is the only policy-conditional command class today.
pub fn advertised_runtime_commands(
    surface: CommandSurface,
    goal_admission_visible: bool,
) -> impl Iterator<Item = CommandSpec> + 'static {
    commands_for_surface(surface).filter(move |spec| {
        matches!(spec.execution, CommandExecution::RuntimeCommand)
            || (goal_admission_visible && matches!(spec.execution, CommandExecution::GoalAdmission))
    })
}

pub fn command_by_name(name: &str) -> Option<CommandSpec> {
    let (normalized, _) = parse_command_name(name)?;
    BUILTIN_COMMANDS
        .iter()
        .copied()
        .find(|spec| spec.token_matches(&normalized))
}

pub fn parse_command_token(token: &str, surface: CommandSurface) -> Option<ParsedCommandToken> {
    match classify_command_token(token, surface) {
        CommandTokenClassification::Valid(parsed) => Some(parsed),
        CommandTokenClassification::Unknown | CommandTokenClassification::MalformedTarget(_) => {
            None
        }
    }
}

pub fn classify_command_token(token: &str, surface: CommandSurface) -> CommandTokenClassification {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return CommandTokenClassification::Unknown;
    }
    let without_slash = trimmed.strip_prefix('/').unwrap_or(trimmed);
    let (name, raw_target) = without_slash
        .split_once('@')
        .map_or((without_slash, None), |(name, target)| (name, Some(target)));
    let normalized = name.trim().to_ascii_lowercase();
    let command = BUILTIN_COMMANDS
        .iter()
        .copied()
        .find(|spec| spec.token_matches(&normalized) && spec.supports(surface));
    let Some(command) = command else {
        return CommandTokenClassification::Unknown;
    };
    let target = match raw_target {
        Some(target) => match normalize_explicit_command_target(target) {
            Some(target) => Some(target),
            None => return CommandTokenClassification::MalformedTarget(command),
        },
        None => None,
    };
    CommandTokenClassification::Valid(ParsedCommandToken { command, target })
}

pub fn normalize_command_name(token: &str) -> Option<String> {
    parse_command_name(token).map(|(name, _target)| name)
}

fn parse_command_name(token: &str) -> Option<(String, Option<String>)> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return None;
    }
    let without_slash = trimmed.strip_prefix('/').unwrap_or(trimmed);
    let (name, target) = match without_slash.split_once('@') {
        Some((name, target)) => (name, Some(normalize_explicit_command_target(target)?)),
        None => (without_slash, None),
    };
    let normalized = name.trim().to_ascii_lowercase();
    (!normalized.is_empty()).then_some((normalized, target))
}

fn normalize_explicit_command_target(target: &str) -> Option<String> {
    let normalized = target.trim();
    if normalized.is_empty() || normalized.contains('@') {
        return None;
    }
    Some(normalized.to_ascii_lowercase())
}

pub fn usage_for_surface(surface: CommandSurface) -> Vec<&'static str> {
    commands_for_surface(surface)
        .map(|spec| spec.usage)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_lookup_accepts_slash_alias_and_bot_suffix() {
        assert_eq!(
            command_by_name("/new@zeroclaw_bot").map(|spec| spec.id),
            Some(BuiltinCommandId::New)
        );
        assert_eq!(
            command_by_name("new-session").map(|spec| spec.id),
            Some(BuiltinCommandId::New)
        );
        assert_eq!(
            command_by_name("/think").map(|spec| spec.id),
            Some(BuiltinCommandId::Thinking)
        );
    }

    #[test]
    fn command_lookup_normalizes_case_whitespace_and_bot_suffix() {
        assert_eq!(
            normalize_command_name("  /MODEL@ZeroClaw_Bot  "),
            Some("model".to_string())
        );
        assert_eq!(
            command_by_name("  /THINK@ZeroClaw_Bot  ").map(|spec| spec.id),
            Some(BuiltinCommandId::Thinking)
        );
        assert_eq!(
            parse_command_token("  /NEW-SESSION@ZeroClaw_Bot  ", CommandSurface::Channel)
                .map(|parsed| parsed.command.id),
            Some(BuiltinCommandId::New)
        );
        let parsed = parse_command_token("/goal@ZeroClaw_Bot", CommandSurface::Channel).unwrap();
        assert_eq!(parsed.target.as_deref(), Some("zeroclaw_bot"));
        assert!(parsed.targets(Some("@ZEROCLAW_BOT")));
        assert!(!parsed.targets(Some("other_bot")));
        assert!(!parsed.targets(None));
        assert!(
            parse_command_token("/goal", CommandSurface::Channel)
                .unwrap()
                .targets(None)
        );
        assert!(parse_command_token("/goal@", CommandSurface::Channel).is_none());
        assert!(parse_command_token("/goal@@other", CommandSurface::Channel).is_none());
        assert!(matches!(
            classify_command_token("/goal@", CommandSurface::Channel),
            CommandTokenClassification::MalformedTarget(spec)
                if spec.id == BuiltinCommandId::Goal
        ));
        assert!(matches!(
            classify_command_token("/unknown@", CommandSurface::Channel),
            CommandTokenClassification::Unknown
        ));
    }

    #[test]
    fn surface_filter_rejects_unavailable_commands() {
        assert!(parse_command_token("/config", CommandSurface::Channel).is_some());
        assert!(parse_command_token("/config", CommandSurface::Web).is_none());
        assert!(parse_command_token("/attach", CommandSurface::Tui).is_none());
        assert!(parse_command_token("/attach", CommandSurface::Channel).is_none());
    }

    #[test]
    fn normalize_command_name_empty_and_whitespace_returns_none() {
        assert_eq!(normalize_command_name(""), None);
        assert_eq!(normalize_command_name("   "), None);
        assert_eq!(normalize_command_name("\t\n"), None);
    }

    #[test]
    fn normalize_command_name_pure_slash_or_at_suffix_returns_none() {
        assert_eq!(normalize_command_name("/"), None);
        assert_eq!(normalize_command_name("@bot"), None);
        assert_eq!(normalize_command_name("/@bot"), None);
        assert_eq!(normalize_command_name("  /  @bot  "), None);
    }

    #[test]
    fn normalize_command_name_unicode_preserved() {
        assert_eq!(normalize_command_name("/新"), Some("新".to_string()));
        assert_eq!(normalize_command_name("/新@my_bot"), Some("新".to_string()));
    }

    #[test]
    fn goal_is_advertised_only_where_admission_is_implemented() {
        assert!(parse_command_token("/goal", CommandSurface::Web).is_none());
        assert!(parse_command_token("/goal", CommandSurface::Tui).is_none());
        assert!(parse_command_token("/goal", CommandSurface::Channel).is_some());
        let goal = command_by_name("/goal").expect("goal command should be registered");
        assert!(
            goal.usage.contains("start <objective>"),
            "goal command usage must advertise the required start objective"
        );
        assert!(
            goal.usage.contains("objective <objective>"),
            "goal command usage must advertise objective amendment syntax"
        );
    }

    #[test]
    fn runtime_advertisement_hides_goal_without_hiding_runtime_commands() {
        let hidden: Vec<_> = advertised_runtime_commands(CommandSurface::Channel, false).collect();
        assert!(
            hidden
                .iter()
                .all(|spec| spec.execution == CommandExecution::RuntimeCommand)
        );
        assert!(hidden.iter().any(|spec| spec.id == BuiltinCommandId::Clear));
        assert!(hidden.iter().all(|spec| spec.id != BuiltinCommandId::Goal));

        let visible: Vec<_> = advertised_runtime_commands(CommandSurface::Channel, true).collect();
        assert!(visible.iter().any(|spec| spec.id == BuiltinCommandId::Goal));
    }
}
