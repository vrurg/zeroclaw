//! Transport-neutral Goal Mode controller contract.
//!
//! This module deliberately has no channel, RPC, or provider integration.
//! Later stages supply those concrete boundaries; this stage owns only the
//! exact command grammar and session-bound controller facade.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use zeroclaw_config::{cost::CostTracker, schema::Config};

use crate::control_plane::{
    GoalAccountingState, GoalBlocker, GoalPauseReason, GoalPauseState, GoalResumeSafety,
    GoalTaskRecord, GoalTaskRegistry, GoalTaskStatus, GoalTransitionResult, TaskKind, TaskRecord,
    TaskStatus,
};

mod execution;
pub(crate) use execution::SessionGoalExecutor;

const INITIAL_GOAL_EXECUTION_EPOCH: i64 = 1;

/// Parsed Goal Mode commands accepted by the V1 grammar.
#[derive(Debug, Clone, PartialEq)]
pub enum GoalCommand {
    /// Start a new session-bound Goal with immutable completion criteria and
    /// objective text.
    Start {
        /// Effective budget selection supplied by the command.
        budget: GoalBudgetRequest,
        /// Immutable operator-declared completion criteria.
        success_criteria: String,
        /// Objective text following the required delimiter.
        objective: String,
    },
    /// Read the current Goal lifecycle projection.
    Status,
    /// Read the current Goal budget projection.
    Budget,
    /// Replace both effective Goal budget dimensions.
    SetBudget(GoalBudgetRequest),
    /// Fence a running Goal without changing its objective or limits.
    Pause,
    /// Request a fresh execution epoch for a paused Goal.
    Resume,
    /// Terminally cancel the current Goal while retaining its audit record.
    Cancel,
    /// Render local grammar help without invoking a model.
    Help,
}

/// The complete two-dimensional budget selection carried by a command.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GoalBudgetRequest {
    /// Resolve both limits from configured defaults.
    Defaults,
    /// Replace both dimensions; omitted dimensions are unlimited.
    Limits {
        /// Optional positive total-token limit.
        token_limit: Option<u64>,
        /// Optional finite positive USD limit.
        cost_limit_usd: Option<f64>,
    },
    /// Explicitly make both dimensions unlimited.
    Unlimited,
}

/// Stable parser failure keys; adapters map these to Fluent messages later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalCommandErrorKey {
    /// The input did not match the `/goal` grammar.
    InvalidGrammar,
    /// `start` omitted the objective delimiter or a nonblank objective.
    ObjectiveRequired,
    /// `start` omitted the criteria delimiter or nonblank criteria.
    SuccessCriteriaRequired,
    /// A budget value was absent, malformed, zero, or non-finite.
    InvalidBudget,
    /// Incompatible flags or unsupported extra arguments were supplied.
    InvalidArguments,
}

/// Typed parser error that keeps stable logic keys separate from presentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalCommandParseError {
    /// Stable English key for tracing and adapter localization.
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
    // Leading command whitespace is syntax; text after the two start delimiters
    // remains operator-supplied data and must not be normalized.
    let input = input.trim_start();
    let command = input
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
            let Some((criteria, objective)) = split_standalone_objective_delimiter(rest) else {
                return Err(objective_required());
            };
            if criteria.trim().is_empty() {
                return Err(success_criteria_required());
            }
            if criteria
                .split_whitespace()
                .any(|token| token.starts_with("--"))
            {
                return Err(invalid_arguments());
            }
            if objective.trim().is_empty() {
                return Err(objective_required());
            }
            return Ok(GoalCommand::Start {
                budget: parse_budget_flags(&flags, true)?,
                success_criteria: criteria.to_string(),
                objective: objective.to_string(),
            });
        }
        if token == "--" {
            return Err(success_criteria_required());
        }

        flags.push(token);
        remainder = rest;
    }
}

/// Split criteria from the objective at the required standalone `--` token.
///
/// The criteria segment intentionally preserves internal whitespace. A token
/// such as `--literal` is not a delimiter and is subsequently rejected as a
/// flag when it appears in the criteria segment.
fn split_standalone_objective_delimiter(input: &str) -> Option<(&str, &str)> {
    input.match_indices("--").find_map(|(index, _)| {
        let before = &input[..index];
        let after = &input[index + 2..];
        let before_is_boundary = before.chars().last().is_none_or(char::is_whitespace);
        let after_is_boundary = after.chars().next().is_none_or(char::is_whitespace);
        (before_is_boundary && after_is_boundary).then_some((before, after))
    })
}

fn parse_budget_command(input: &str) -> Result<GoalCommand, GoalCommandParseError> {
    if input.trim().is_empty() {
        return Ok(GoalCommand::Budget);
    }

    let (action, remainder) = take_token(input).ok_or_else(invalid_arguments)?;
    if action != "set" {
        return Err(invalid_arguments());
    }

    let flags = remainder.split_whitespace().collect::<Vec<_>>();
    Ok(GoalCommand::SetBudget(parse_budget_flags(&flags, false)?))
}

fn parse_budget_flags(
    flags: &[&str],
    allow_defaults: bool,
) -> Result<GoalBudgetRequest, GoalCommandParseError> {
    if flags.is_empty() {
        return if allow_defaults {
            Ok(GoalBudgetRequest::Defaults)
        } else {
            Err(invalid_budget())
        };
    }

    let mut index = 0;
    let mut unlimited = false;
    let mut token_limit = None;
    let mut cost_limit_usd = None;
    while index < flags.len() {
        match flags[index] {
            "--unlimited" => {
                if unlimited || token_limit.is_some() || cost_limit_usd.is_some() {
                    return Err(invalid_arguments());
                }
                unlimited = true;
                index += 1;
            }
            "--tokens" => {
                if unlimited || token_limit.is_some() {
                    return Err(invalid_arguments());
                }
                let value = *flags.get(index + 1).ok_or_else(invalid_budget)?;
                token_limit = Some(parse_token_limit(value)?);
                index += 2;
            }
            "--cost-usd" => {
                if unlimited || cost_limit_usd.is_some() {
                    return Err(invalid_arguments());
                }
                let value = *flags.get(index + 1).ok_or_else(invalid_budget)?;
                cost_limit_usd = Some(parse_cost_limit(value)?);
                index += 2;
            }
            _ => return Err(invalid_arguments()),
        }
    }

    if unlimited {
        Ok(GoalBudgetRequest::Unlimited)
    } else if token_limit.is_some() || cost_limit_usd.is_some() {
        Ok(GoalBudgetRequest::Limits {
            token_limit,
            cost_limit_usd,
        })
    } else {
        Err(invalid_budget())
    }
}

fn parse_token_limit(value: &str) -> Result<u64, GoalCommandParseError> {
    let value = value.parse::<u64>().map_err(|_| invalid_budget())?;
    if value == 0 || value > i64::MAX as u64 {
        return Err(invalid_budget());
    }
    Ok(value)
}

fn parse_cost_limit(value: &str) -> Result<f64, GoalCommandParseError> {
    let value = value.parse::<f64>().map_err(|_| invalid_budget())?;
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid_budget());
    }
    Ok(value)
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

fn invalid_grammar() -> GoalCommandParseError {
    GoalCommandParseError {
        key: GoalCommandErrorKey::InvalidGrammar,
    }
}

fn objective_required() -> GoalCommandParseError {
    GoalCommandParseError {
        key: GoalCommandErrorKey::ObjectiveRequired,
    }
}

fn success_criteria_required() -> GoalCommandParseError {
    GoalCommandParseError {
        key: GoalCommandErrorKey::SuccessCriteriaRequired,
    }
}

fn invalid_budget() -> GoalCommandParseError {
    GoalCommandParseError {
        key: GoalCommandErrorKey::InvalidBudget,
    }
}

fn invalid_arguments() -> GoalCommandParseError {
    GoalCommandParseError {
        key: GoalCommandErrorKey::InvalidArguments,
    }
}

/// Trusted, server-derived facts that bind a Goal command to one session.
///
/// Transport adapters construct this only after authentication and session
/// lookup. Command text never selects any of these facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalIngressContext {
    session_id: String,
    agent: String,
    route: String,
    principal: GoalIngressPrincipal,
}

impl GoalIngressContext {
    /// Construct trusted Matrix ingress from its immutable raw SDK MXID.
    pub fn matrix(
        session_id: impl Into<String>,
        agent: impl Into<String>,
        route: impl Into<String>,
        raw_mxid: impl Into<String>,
    ) -> Result<Self, GoalIngressError> {
        Self::new(
            session_id,
            agent,
            route,
            GoalIngressPrincipal::Matrix {
                raw_mxid: raw_mxid.into(),
            },
        )
    }

    /// Construct trusted ZeroCode ingress from its transient registered TUI ID.
    ///
    /// The TUI ID is deliberately never persisted as a task principal.
    pub fn zero_code(
        session_id: impl Into<String>,
        agent: impl Into<String>,
        route: impl Into<String>,
        tui_id: impl Into<String>,
    ) -> Result<Self, GoalIngressError> {
        Self::new(
            session_id,
            agent,
            route,
            GoalIngressPrincipal::ZeroCode {
                tui_id: tui_id.into(),
            },
        )
    }

    fn new(
        session_id: impl Into<String>,
        agent: impl Into<String>,
        route: impl Into<String>,
        principal: GoalIngressPrincipal,
    ) -> Result<Self, GoalIngressError> {
        let session_id = required_ingress_value(session_id.into())?;
        let agent = required_ingress_value(agent.into())?;
        let route = required_ingress_value(route.into())?;
        match &principal {
            GoalIngressPrincipal::Matrix { raw_mxid } => {
                required_ingress_value(raw_mxid.clone())?;
            }
            GoalIngressPrincipal::ZeroCode { tui_id } => {
                required_ingress_value(tui_id.clone())?;
            }
        }
        Ok(Self {
            session_id,
            agent,
            route,
            principal,
        })
    }

    /// Canonical session key chosen before command parsing.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Server-derived owning agent alias.
    pub fn agent(&self) -> &str {
        &self.agent
    }

    /// Canonical route/reply target selected by the transport.
    pub fn route(&self) -> &str {
        &self.route
    }

    fn durable_principal_id(&self) -> Option<&str> {
        match &self.principal {
            GoalIngressPrincipal::Matrix { raw_mxid } => Some(raw_mxid),
            GoalIngressPrincipal::ZeroCode { .. } => None,
        }
    }
}

/// Transport-specific trusted principal facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoalIngressPrincipal {
    /// Raw Matrix SDK sender identity, never a display name.
    Matrix { raw_mxid: String },
    /// Process-local TUI identity, never a durable task principal.
    ZeroCode { tui_id: String },
}

/// Stable ingress-construction errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalIngressError {
    /// A required server-derived binding was blank or missing.
    MissingBinding,
}

impl std::fmt::Display for GoalIngressError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("goal_ingress_missing_binding")
    }
}

impl std::error::Error for GoalIngressError {}

fn required_ingress_value(value: String) -> Result<String, GoalIngressError> {
    if value.trim().is_empty() {
        Err(GoalIngressError::MissingBinding)
    } else {
        Ok(value)
    }
}

/// Semantic response keys. Adapters render them through their own Fluent
/// catalogue only when concrete transport wiring is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalMessageKey {
    Help,
    Disabled,
    InvalidConfiguration,
    NoCurrentGoal,
    AlreadyActive,
    BudgetRequired,
    PricingUnavailable,
    VerifierUnavailable,
    Started,
    Status,
    Budget,
    Paused,
    AlreadyPaused,
    Resumed,
    Cancelled,
    AlreadyCancelled,
    BudgetUpdated,
    InvalidCommand,
    InvalidBudget,
    SuccessCriteriaRequired,
    ToolPhaseNonResumable,
    Terminal,
    ExecutorStartFailed,
    Stale,
}

/// Controller response that contains no rendered user text.
#[derive(Debug, Clone, PartialEq)]
pub struct GoalResponse {
    pub key: GoalMessageKey,
    pub status: Option<GoalStatusProjection>,
}

impl GoalResponse {
    fn message(key: GoalMessageKey) -> Self {
        Self { key, status: None }
    }

    fn status(key: GoalMessageKey, status: GoalStatusProjection) -> Self {
        Self {
            key,
            status: Some(status),
        }
    }
}

/// Read-only current-Goal projection derived from durable control state and
/// the canonical cost ledger.
#[derive(Debug, Clone, PartialEq)]
pub struct GoalStatusProjection {
    pub task_id: String,
    pub status: TaskStatus,
    pub resumable: bool,
    pub token_limit: Option<u64>,
    pub cost_limit_usd: Option<f64>,
    /// Known durable token usage. When accounting is incomplete, this remains
    /// a lower bound and no remaining budget is advertised.
    pub known_token_usage: u64,
    /// Known durable cost in USD. When pricing is incomplete, this remains a
    /// lower bound and no remaining cost budget is advertised.
    pub known_cost_usd: f64,
    pub token_accounting_complete: bool,
    pub cost_accounting_complete: bool,
    pub remaining_tokens: Option<u64>,
    pub remaining_cost_usd: Option<f64>,
    pub accounting_state: GoalAccountingState,
    pub pending_reserved_tokens: Option<u64>,
    pub pending_reserved_cost_usd: Option<f64>,
    pub resume_safety: GoalResumeSafety,
    pub pause_reason: Option<GoalPauseReason>,
    pub pause_description: Option<String>,
    pub blockers: Vec<GoalBlocker>,
    pub delivered: bool,
    pub terminal_error: Option<String>,
}

/// One process-local transcript, intentionally absent from the control plane
/// and canonical session history.
#[derive(Debug, Default)]
pub struct GoalWorkingTranscript {
    turns: Vec<GoalWorkingTurn>,
}

impl GoalWorkingTranscript {
    /// Add one working-only turn fragment for the future executor.
    pub fn push(&mut self, turn: GoalWorkingTurn) {
        self.turns.push(turn);
    }

    /// Borrow working-only turn fragments.
    pub fn turns(&self) -> &[GoalWorkingTurn] {
        &self.turns
    }
}

/// A process-local parent or verifier continuation fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalWorkingTurn {
    pub turn_kind: GoalTurnKind,
    pub text: String,
}

/// Trusted controller turn kind supplied to the executor seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalTurnKind {
    Start,
    Resume,
}

/// Exact durable identity supplied to the dormant executor seam.
#[derive(Debug, Clone)]
pub struct GoalExecutionScope {
    pub task_id: String,
    pub session_id: String,
    pub execution_epoch: i64,
    pub turn_kind: GoalTurnKind,
    pub cancellation: CancellationToken,
    /// Process-local serialization for one Goal's logical model operations.
    /// The durable pending slot remains the authoritative cross-process fence.
    pub operation_permit: Arc<Semaphore>,
}

/// Preflight input used before a durable Goal is admitted to execution.
#[derive(Debug, Clone)]
pub struct GoalExecutionPreflight {
    pub session_id: String,
    pub agent: String,
    pub token_limit: Option<u64>,
    pub cost_limit_usd: Option<f64>,
}

/// Readiness facts supplied by the dormant executor seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalExecutionReadiness {
    Ready,
    PricingUnavailable,
    VerifierUnavailable,
    BindingUnavailable,
    ExecutorBusy,
}

/// Dormant execution seam. This stage performs no provider work.
#[async_trait]
pub trait GoalExecutor: Send + Sync {
    async fn preflight(&self, request: &GoalExecutionPreflight) -> GoalExecutionReadiness;
    async fn launch(&self, scope: GoalExecutionScope) -> Result<(), ()>;

    /// Signal and quiesce an owned exact executor after the caller has
    /// committed its durable epoch fence. Dormant/fake executors need no work.
    async fn cancel(&self, _task_id: &str, _session_id: &str, _epoch: i64) -> Result<(), ()> {
        Ok(())
    }
}

/// The sole transport-neutral Goal controller facade.
pub struct GoalRuntime {
    config: Arc<Config>,
    goals: Arc<dyn GoalTaskRegistry>,
    cost_tracker: Arc<CostTracker>,
    executor: Arc<dyn GoalExecutor>,
    boot_id: String,
}

impl GoalRuntime {
    /// Build one controller for one canonical runtime configuration generation.
    ///
    /// This dormant stage has no reload entrypoint. Lifecycle cutovers will
    /// replace full controller instances only after fencing affected Goals.
    pub fn new(
        config: Arc<Config>,
        goals: Arc<dyn GoalTaskRegistry>,
        cost_tracker: Arc<CostTracker>,
        executor: Arc<dyn GoalExecutor>,
        boot_id: impl Into<String>,
    ) -> Result<Self, GoalIngressError> {
        Ok(Self {
            config,
            goals,
            cost_tracker,
            executor,
            boot_id: required_ingress_value(boot_id.into())?,
        })
    }

    /// Submit one already-parsed command bound to trusted ingress.
    pub async fn submit(
        &self,
        ingress: GoalIngressContext,
        command: GoalCommand,
    ) -> Result<GoalResponse, GoalRuntimeError> {
        if command == GoalCommand::Help {
            return Ok(GoalResponse::message(GoalMessageKey::Help));
        }
        if !self.config.goal.enabled {
            return Ok(GoalResponse::message(GoalMessageKey::Disabled));
        }
        if self.config.validate().is_err() {
            return Ok(GoalResponse::message(GoalMessageKey::InvalidConfiguration));
        }

        match command {
            GoalCommand::Start {
                budget,
                success_criteria,
                objective,
            } => {
                self.start(ingress, budget, success_criteria, objective)
                    .await
            }
            GoalCommand::Status => self.status(ingress).await,
            GoalCommand::Budget => self.budget(ingress).await,
            GoalCommand::SetBudget(budget) => self.set_budget(ingress, budget).await,
            GoalCommand::Pause => self.pause(ingress).await,
            GoalCommand::Resume => self.resume(ingress).await,
            GoalCommand::Cancel => self.cancel(ingress).await,
            GoalCommand::Help => unreachable!("help returns before configuration admission"),
        }
    }

    async fn start(
        &self,
        ingress: GoalIngressContext,
        request: GoalBudgetRequest,
        success_criteria: String,
        objective: String,
    ) -> Result<GoalResponse, GoalRuntimeError> {
        if objective.trim().is_empty() {
            return Ok(GoalResponse::message(GoalMessageKey::InvalidCommand));
        }
        if success_criteria.trim().is_empty() {
            return Ok(GoalResponse::message(
                GoalMessageKey::SuccessCriteriaRequired,
            ));
        }
        if let Some(current) = self
            .goals
            .current_goal_for_session(ingress.session_id())
            .await
            .map_err(GoalRuntimeError::control_plane)?
        {
            if !task_matches_ingress(&current, &ingress) {
                return Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal));
            }
            if !current.status.is_terminal() {
                return Ok(GoalResponse::message(GoalMessageKey::AlreadyActive));
            }
        }

        let (token_limit, cost_limit_usd) = match self.resolve_limits(request) {
            Ok(limits) => limits,
            Err(GoalRuntimeError::BudgetRequired) => {
                return Ok(GoalResponse::message(GoalMessageKey::BudgetRequired));
            }
            Err(GoalRuntimeError::InvalidBudget) => {
                return Ok(GoalResponse::message(GoalMessageKey::InvalidBudget));
            }
            Err(error) => return Err(error),
        };
        let readiness = self
            .executor
            .preflight(&GoalExecutionPreflight {
                session_id: ingress.session_id.clone(),
                agent: ingress.agent.clone(),
                token_limit,
                cost_limit_usd,
            })
            .await;
        if !readiness_permits_admission(readiness, cost_limit_usd) {
            return Ok(GoalResponse::message(readiness_message_key(readiness)));
        }

        let task_id = Uuid::new_v4().to_string();
        let task = TaskRecord {
            id: task_id.clone(),
            kind: TaskKind::Goal,
            agent: ingress.agent.clone(),
            status: TaskStatus::Running,
            owner_pid: std::process::id(),
            owner_boot_id: self.boot_id.clone(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: Some(ingress.route.clone()),
            delivered: false,
            idem_key: None,
            principal_id: ingress.durable_principal_id().map(str::to_string),
            session_id: Some(ingress.session_id.clone()),
            execution_epoch: 0,
            started_at: Utc::now().to_rfc3339(),
            finished_at: None,
        };
        let goal = GoalTaskRecord {
            task_id: task_id.clone(),
            objective,
            success_criteria,
            effective_token_limit: token_limit,
            effective_cost_limit_usd: cost_limit_usd,
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
            pending_call_id: None,
            pending_call_epoch: None,
            pending_reserved_tokens: None,
            pending_reserved_cost_usd: None,
            accounting_state: GoalAccountingState::Complete,
            resume_safety: GoalResumeSafety::Clean,
        };
        match self
            .goals
            .create_or_replace_session_goal(task, goal)
            .await
            .map_err(GoalRuntimeError::control_plane)?
        {
            GoalTransitionResult::Applied => {
                self.launch_exact(
                    ingress,
                    &task_id,
                    INITIAL_GOAL_EXECUTION_EPOCH,
                    GoalTurnKind::Start,
                )
                .await
            }
            GoalTransitionResult::Stale => Ok(GoalResponse::message(
                if self.current_goal(&ingress).await?.is_some() {
                    GoalMessageKey::AlreadyActive
                } else {
                    GoalMessageKey::NoCurrentGoal
                },
            )),
            GoalTransitionResult::Missing => Ok(GoalResponse::message(GoalMessageKey::Stale)),
        }
    }

    async fn status(&self, ingress: GoalIngressContext) -> Result<GoalResponse, GoalRuntimeError> {
        let Some(current) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal));
        };
        Ok(GoalResponse::status(
            GoalMessageKey::Status,
            self.status_projection(current).await?,
        ))
    }

    async fn budget(&self, ingress: GoalIngressContext) -> Result<GoalResponse, GoalRuntimeError> {
        let Some(current) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal));
        };
        Ok(GoalResponse::status(
            GoalMessageKey::Budget,
            self.status_projection(current).await?,
        ))
    }

    async fn set_budget(
        &self,
        ingress: GoalIngressContext,
        request: GoalBudgetRequest,
    ) -> Result<GoalResponse, GoalRuntimeError> {
        let limits = match request {
            GoalBudgetRequest::Limits {
                token_limit,
                cost_limit_usd,
            } => validate_goal_limits((token_limit, cost_limit_usd)),
            GoalBudgetRequest::Unlimited => Ok((None, None)),
            GoalBudgetRequest::Defaults => {
                return Ok(GoalResponse::message(GoalMessageKey::InvalidCommand));
            }
        };
        let (token_limit, cost_limit_usd) = match limits {
            Ok(limits) => limits,
            Err(GoalRuntimeError::InvalidBudget) => {
                return Ok(GoalResponse::message(GoalMessageKey::InvalidBudget));
            }
            Err(error) => return Err(error),
        };
        let Some(current) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal));
        };
        let task = &current.task;
        if !matches!(task.status, TaskStatus::Running | TaskStatus::Paused) {
            return Ok(GoalResponse::message(GoalMessageKey::Terminal));
        }
        let result = self
            .goals
            .update_session_goal_limits(
                &task.id,
                ingress.session_id(),
                task.execution_epoch,
                token_limit,
                cost_limit_usd,
            )
            .await
            .map_err(GoalRuntimeError::control_plane)?;
        Ok(GoalResponse::message(transition_message(
            result,
            GoalMessageKey::BudgetUpdated,
        )))
    }

    async fn pause(&self, ingress: GoalIngressContext) -> Result<GoalResponse, GoalRuntimeError> {
        let Some(current) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal));
        };
        let task = &current.task;
        if task.status == TaskStatus::Paused {
            return Ok(GoalResponse::message(GoalMessageKey::AlreadyPaused));
        }
        if task.status != TaskStatus::Running {
            return Ok(GoalResponse::message(GoalMessageKey::Terminal));
        }
        let result = self
            .goals
            .pause_session_goal(
                &task.id,
                ingress.session_id(),
                task.execution_epoch,
                GoalPauseState {
                    reason: GoalPauseReason::OperatorPaused,
                    description: None,
                    blockers: Vec::new(),
                },
            )
            .await
            .map_err(GoalRuntimeError::control_plane)?;
        if result == GoalTransitionResult::Applied {
            self.executor
                .cancel(&task.id, ingress.session_id(), task.execution_epoch)
                .await
                .map_err(|_| {
                    GoalRuntimeError::control_plane(anyhow::Error::msg(
                        "goal executor did not quiesce after pause",
                    ))
                })?;
        }
        Ok(GoalResponse::message(transition_message(
            result,
            GoalMessageKey::Paused,
        )))
    }

    async fn resume(&self, ingress: GoalIngressContext) -> Result<GoalResponse, GoalRuntimeError> {
        let Some(current) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal));
        };
        let task = &current.task;
        let goal = &current.goal;
        if task.status != TaskStatus::Paused {
            return Ok(GoalResponse::message(if task.status.is_terminal() {
                GoalMessageKey::Terminal
            } else {
                GoalMessageKey::Stale
            }));
        }
        if goal.resume_safety == GoalResumeSafety::ToolExecuted {
            return Ok(GoalResponse::message(GoalMessageKey::ToolPhaseNonResumable));
        }
        let readiness = self
            .executor
            .preflight(&GoalExecutionPreflight {
                session_id: ingress.session_id.clone(),
                agent: ingress.agent.clone(),
                token_limit: goal.effective_token_limit,
                cost_limit_usd: goal.effective_cost_limit_usd,
            })
            .await;
        if !readiness_permits_admission(readiness, goal.effective_cost_limit_usd) {
            return Ok(GoalResponse::message(readiness_message_key(readiness)));
        }
        match self
            .goals
            .resume_session_goal(
                &task.id,
                ingress.session_id(),
                task.execution_epoch,
                std::process::id(),
                &self.boot_id,
            )
            .await
            .map_err(GoalRuntimeError::control_plane)?
        {
            GoalTransitionResult::Applied => {
                self.launch_exact(
                    ingress,
                    &task.id,
                    task.execution_epoch.saturating_add(1),
                    GoalTurnKind::Resume,
                )
                .await
            }
            GoalTransitionResult::Stale => Ok(GoalResponse::message(GoalMessageKey::Stale)),
            GoalTransitionResult::Missing => {
                Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal))
            }
        }
    }

    async fn cancel(&self, ingress: GoalIngressContext) -> Result<GoalResponse, GoalRuntimeError> {
        let Some(current) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal));
        };
        let task = &current.task;
        if task.status == TaskStatus::Cancelled {
            return Ok(GoalResponse::message(GoalMessageKey::AlreadyCancelled));
        }
        if task.status.is_terminal() {
            return Ok(GoalResponse::message(GoalMessageKey::Terminal));
        }
        let result = self
            .goals
            .finish_session_goal(
                &task.id,
                ingress.session_id(),
                task.execution_epoch,
                TaskStatus::Cancelled,
                None,
            )
            .await
            .map_err(GoalRuntimeError::control_plane)?;
        if result == GoalTransitionResult::Applied {
            self.executor
                .cancel(&task.id, ingress.session_id(), task.execution_epoch)
                .await
                .map_err(|_| {
                    GoalRuntimeError::control_plane(anyhow::Error::msg(
                        "goal executor did not quiesce after cancellation",
                    ))
                })?;
        }
        Ok(GoalResponse::message(transition_message(
            result,
            GoalMessageKey::Cancelled,
        )))
    }

    async fn launch_exact(
        &self,
        ingress: GoalIngressContext,
        task_id: &str,
        expected_epoch: i64,
        turn_kind: GoalTurnKind,
    ) -> Result<GoalResponse, GoalRuntimeError> {
        let Some(current) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::Stale));
        };
        let task = &current.task;
        if task.id != task_id
            || task.execution_epoch != expected_epoch
            || task.status != TaskStatus::Running
        {
            return Ok(GoalResponse::message(GoalMessageKey::Stale));
        }
        let scope = GoalExecutionScope {
            task_id: task.id.clone(),
            session_id: ingress.session_id.clone(),
            execution_epoch: task.execution_epoch,
            turn_kind,
            cancellation: CancellationToken::new(),
            operation_permit: Arc::new(Semaphore::new(1)),
        };
        if self.executor.launch(scope).await.is_ok() {
            return Ok(GoalResponse::message(match turn_kind {
                GoalTurnKind::Start => GoalMessageKey::Started,
                GoalTurnKind::Resume => GoalMessageKey::Resumed,
            }));
        }
        let result = self
            .goals
            .finish_session_goal(
                &task.id,
                ingress.session_id(),
                task.execution_epoch,
                TaskStatus::Failed,
                Some("executor_start_failed".into()),
            )
            .await
            .map_err(GoalRuntimeError::control_plane)?;
        Ok(GoalResponse::message(transition_message(
            result,
            GoalMessageKey::ExecutorStartFailed,
        )))
    }

    fn resolve_limits(
        &self,
        request: GoalBudgetRequest,
    ) -> Result<(Option<u64>, Option<f64>), GoalRuntimeError> {
        match request {
            GoalBudgetRequest::Defaults => {
                let limits = (
                    self.config.goal.default_token_limit,
                    self.config.goal.default_cost_limit_usd,
                );
                if limits.0.is_none() && limits.1.is_none() {
                    Err(GoalRuntimeError::BudgetRequired)
                } else {
                    validate_goal_limits(limits)
                }
            }
            GoalBudgetRequest::Limits {
                token_limit,
                cost_limit_usd,
            } => validate_goal_limits((token_limit, cost_limit_usd)),
            GoalBudgetRequest::Unlimited => Ok((None, None)),
        }
    }

    async fn current_goal(
        &self,
        ingress: &GoalIngressContext,
    ) -> Result<Option<GoalTaskStatus>, GoalRuntimeError> {
        let Some(current) = self
            .goals
            .current_goal_status_for_session(ingress.session_id())
            .await
            .map_err(GoalRuntimeError::control_plane)?
        else {
            return Ok(None);
        };
        if !task_matches_ingress(&current.task, ingress) {
            return Ok(None);
        }
        Ok(Some(current))
    }

    async fn status_projection(
        &self,
        current: GoalTaskStatus,
    ) -> Result<GoalStatusProjection, GoalRuntimeError> {
        let task_id = current.task.id.clone();
        let tracker = Arc::clone(&self.cost_tracker);
        let totals =
            tokio::task::spawn_blocking(move || tracker.get_strict_usage_totals_for_task(&task_id))
                .await
                .map_err(GoalRuntimeError::accounting)?
                .map_err(GoalRuntimeError::accounting)?;
        Ok(status_projection(&current, totals))
    }
}

/// Non-user-facing controller errors. Adapters receive typed responses only.
#[derive(Debug, thiserror::Error)]
pub enum GoalRuntimeError {
    #[error("goal_control_plane")]
    ControlPlane(#[source] anyhow::Error),
    #[error("goal_budget_required")]
    BudgetRequired,
    #[error("goal_invalid_budget")]
    InvalidBudget,
    #[error("goal_accounting_unavailable")]
    Accounting(#[source] anyhow::Error),
}

impl GoalRuntimeError {
    fn control_plane(error: anyhow::Error) -> Self {
        Self::ControlPlane(error)
    }

    fn accounting(error: impl Into<anyhow::Error>) -> Self {
        Self::Accounting(error.into())
    }
}

fn readiness_message_key(readiness: GoalExecutionReadiness) -> GoalMessageKey {
    match readiness {
        GoalExecutionReadiness::Ready => GoalMessageKey::Started,
        GoalExecutionReadiness::PricingUnavailable => GoalMessageKey::PricingUnavailable,
        GoalExecutionReadiness::VerifierUnavailable => GoalMessageKey::VerifierUnavailable,
        GoalExecutionReadiness::BindingUnavailable | GoalExecutionReadiness::ExecutorBusy => {
            GoalMessageKey::Stale
        }
    }
}

fn readiness_permits_admission(
    readiness: GoalExecutionReadiness,
    cost_limit_usd: Option<f64>,
) -> bool {
    readiness == GoalExecutionReadiness::Ready
        || (readiness == GoalExecutionReadiness::PricingUnavailable && cost_limit_usd.is_none())
}

fn transition_message(result: GoalTransitionResult, applied: GoalMessageKey) -> GoalMessageKey {
    match result {
        GoalTransitionResult::Applied => applied,
        GoalTransitionResult::Stale => GoalMessageKey::Stale,
        GoalTransitionResult::Missing => GoalMessageKey::NoCurrentGoal,
    }
}

fn status_projection(
    current: &GoalTaskStatus,
    totals: zeroclaw_config::cost::StrictTaskUsageTotals,
) -> GoalStatusProjection {
    let task = &current.task;
    let goal = &current.goal;
    let settled = goal.accounting_state == GoalAccountingState::Complete
        && goal.pending_call_id.is_none()
        && goal.pending_call_epoch.is_none()
        && goal.pending_reserved_tokens.is_none()
        && goal.pending_reserved_cost_usd.is_none();
    let token_accounting_complete = settled;
    let cost_accounting_complete = settled && totals.pricing_available;
    GoalStatusProjection {
        task_id: task.id.clone(),
        status: task.status,
        resumable: task.status == TaskStatus::Paused
            && task.execution_epoch < i64::MAX
            && goal.accounting_state == GoalAccountingState::Complete
            && goal.pending_call_id.is_none()
            && goal.pending_call_epoch.is_none()
            && goal.pending_reserved_tokens.is_none()
            && goal.pending_reserved_cost_usd.is_none()
            && goal.resume_safety == GoalResumeSafety::Clean,
        token_limit: goal.effective_token_limit,
        cost_limit_usd: goal.effective_cost_limit_usd,
        known_token_usage: totals.total_tokens,
        known_cost_usd: totals.cost_usd,
        token_accounting_complete,
        cost_accounting_complete,
        remaining_tokens: token_accounting_complete
            .then_some(goal.effective_token_limit)
            .flatten()
            .map(|limit| limit.saturating_sub(totals.total_tokens)),
        remaining_cost_usd: cost_accounting_complete
            .then_some(goal.effective_cost_limit_usd)
            .flatten()
            .map(|limit| (limit - totals.cost_usd).max(0.0)),
        accounting_state: goal.accounting_state,
        pending_reserved_tokens: goal.pending_reserved_tokens,
        pending_reserved_cost_usd: goal.pending_reserved_cost_usd,
        resume_safety: goal.resume_safety,
        pause_reason: goal.pause_reason,
        pause_description: goal.pause_description.clone(),
        blockers: goal.blockers.clone(),
        delivered: task.delivered,
        terminal_error: current.terminal_error.clone(),
    }
}

fn task_matches_ingress(task: &TaskRecord, ingress: &GoalIngressContext) -> bool {
    task.agent == ingress.agent
        && task.originator_route.as_deref() == Some(ingress.route())
        && task.principal_id.as_deref() == ingress.durable_principal_id()
}

fn validate_goal_limits(
    limits: (Option<u64>, Option<f64>),
) -> Result<(Option<u64>, Option<f64>), GoalRuntimeError> {
    let (token_limit, cost_limit_usd) = limits;
    if (token_limit.is_none() && cost_limit_usd.is_none())
        || token_limit.is_some_and(|limit| limit == 0 || limit > i64::MAX as u64)
        || cost_limit_usd.is_some_and(|limit| !limit.is_finite() || limit <= 0.0)
    {
        return Err(GoalRuntimeError::InvalidBudget);
    }
    Ok((token_limit, cost_limit_usd))
}
