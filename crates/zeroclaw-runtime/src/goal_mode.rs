//! Transport-neutral Goal Mode command contract.
//!
//! This module owns the strict command grammar and controller facade used by
//! later Matrix and ZeroCode adapters. It deliberately performs no provider
//! work and exposes no channel-specific authority surface.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use zeroclaw_config::schema::Config;
use zeroclaw_control_plane::{
    GoalAccountingState, GoalPauseReason, GoalPauseState, GoalTaskRecord, GoalTaskRegistry,
    GoalTransitionResult, TaskKind, TaskRecord, TaskStatus,
};

const INITIAL_GOAL_EXECUTION_EPOCH: i64 = 1;

/// Parsed Goal Mode commands accepted by the shared command grammar.
#[derive(Debug, Clone, PartialEq)]
pub enum GoalCommand {
    /// Start a new session-bound goal with an immutable objective.
    Start {
        /// Effective budget selection supplied by the command.
        budget: GoalBudgetRequest,
        /// Untrusted objective text, retained only as the goal objective.
        objective: String,
    },
    /// Read the current goal lifecycle projection.
    Status,
    /// Read the current goal budget projection.
    Budget,
    /// Replace both effective goal budget dimensions.
    SetBudget(GoalBudgetRequest),
    /// Fence a running goal without changing its objective or limits.
    Pause,
    /// Request a fresh execution epoch for a paused goal.
    Resume,
    /// Terminally cancel the current goal while retaining its audit record.
    Cancel,
    /// Render local grammar help without invoking a model.
    Help,
}

/// The complete two-dimensional budget selection carried by a command.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GoalBudgetRequest {
    /// Resolve both limits from the configured defaults.
    Defaults,
    /// Replace both dimensions: omitted dimensions are unlimited.
    Limits {
        /// Optional positive total-token limit.
        token_limit: Option<u64>,
        /// Optional finite positive USD limit.
        cost_limit_usd: Option<f64>,
    },
    /// Explicitly make both dimensions unlimited.
    Unlimited,
}

/// Stable parser failure keys; adapters map these to Fluent messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalCommandErrorKey {
    /// The command did not match the exact `/goal` grammar.
    InvalidGrammar,
    /// `start` omitted the mandatory objective delimiter or usable objective.
    ObjectiveRequired,
    /// A budget number was absent, malformed, zero, or non-finite.
    InvalidBudget,
    /// Incompatible flags or unsupported extra arguments were supplied.
    InvalidArguments,
}

/// Typed parser failure that keeps stable logic keys separate from user text.
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
            GoalCommandErrorKey::InvalidBudget => "goal_invalid_budget",
            GoalCommandErrorKey::InvalidArguments => "goal_invalid_arguments",
        })
    }
}

impl std::error::Error for GoalCommandParseError {}

/// Parse one raw Goal Mode command using the RFC V1 grammar.
pub fn parse_goal_command(input: &str) -> Result<GoalCommand, GoalCommandParseError> {
    let input = input.trim();
    let command = input
        .strip_prefix("/goal")
        .filter(|suffix| {
            suffix.is_empty()
                || suffix
                    .chars()
                    .next()
                    .is_some_and(|character| character.is_whitespace())
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
        let (token, rest) = take_token(remainder).ok_or_else(objective_required)?;
        if token == "--" {
            let objective = rest.trim();
            if objective.is_empty() {
                return Err(objective_required());
            }
            return Ok(GoalCommand::Start {
                budget: parse_budget_flags(&flags, true)?,
                objective: objective.to_string(),
            });
        }

        flags.push(token);
        remainder = rest;
    }
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

/// Trusted, server-derived facts that bind a Goal Mode command to one session.
///
/// Callers construct this only after transport authentication and session
/// lookup. Command text never selects these facts.
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

    /// Construct trusted ZeroCode ingress from the transient registered TUI id.
    ///
    /// The `tui_id` is deliberately never exposed as a durable task principal.
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
        let session_id = required_ingress_value("session_id", session_id.into())?;
        let agent = required_ingress_value("agent", agent.into())?;
        let route = required_ingress_value("route", route.into())?;
        match &principal {
            GoalIngressPrincipal::Matrix { raw_mxid } => {
                required_ingress_value("raw_mxid", raw_mxid.clone())?;
            }
            GoalIngressPrincipal::ZeroCode { tui_id } => {
                required_ingress_value("tui_id", tui_id.clone())?;
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
    /// Process-local TUI identity. It is not a durable task principal.
    ZeroCode { tui_id: String },
}

/// Stable ingress-construction error keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalIngressError {
    /// A server-derived binding fact was blank or missing.
    MissingBinding,
}

impl std::fmt::Display for GoalIngressError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("goal_ingress_missing_binding")
    }
}

impl std::error::Error for GoalIngressError {}

fn required_ingress_value(_name: &str, value: String) -> Result<String, GoalIngressError> {
    if value.trim().is_empty() {
        Err(GoalIngressError::MissingBinding)
    } else {
        Ok(value)
    }
}

/// Semantic response keys; adapters render them through their owning Fluent catalogue.
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
    Terminal,
    ExecutorStartFailed,
    Stale,
}

/// Controller result that stays transport-neutral and contains no rendered text.
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

/// Read-only current-goal projection. Usage totals stay absent until PR3
/// derives them from the canonical ledger rather than inventing a counter.
#[derive(Debug, Clone, PartialEq)]
pub struct GoalStatusProjection {
    pub task_id: String,
    pub status: TaskStatus,
    pub resumable: bool,
    pub token_limit: Option<u64>,
    pub cost_limit_usd: Option<f64>,
    pub accounting_state: GoalAccountingState,
    pub pause_reason: Option<GoalPauseReason>,
}

/// One process-local transcript. It is intentionally never written to the
/// control plane or canonical session history.
#[derive(Debug, Default)]
pub struct GoalWorkingTranscript {
    turns: Vec<GoalWorkingTurn>,
}

impl GoalWorkingTranscript {
    /// Add a process-local turn fragment for a running executor.
    pub fn push(&mut self, turn: GoalWorkingTurn) {
        self.turns.push(turn);
    }

    /// Borrow the working-only turn fragments.
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

/// Trusted controller turn kind supplied to an executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalTurnKind {
    Start,
    Resume,
}

/// Exact durable identity supplied to the dormant execution seam.
#[derive(Debug, Clone)]
pub struct GoalExecutionScope {
    pub task_id: String,
    pub session_id: String,
    pub execution_epoch: i64,
    pub turn_kind: GoalTurnKind,
    pub cancellation: CancellationToken,
}

/// Preflight input used before a durable goal is admitted to execution.
#[derive(Debug, Clone)]
pub struct GoalExecutionPreflight {
    pub session_id: String,
    pub agent: String,
    pub token_limit: Option<u64>,
    pub cost_limit_usd: Option<f64>,
}

/// Narrow fake-execution result. PR3 replaces its ready path with the exact
/// provider, verifier, and ledger admission protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalExecutionReadiness {
    Ready,
    PricingUnavailable,
    VerifierUnavailable,
}

/// Dormant execution seam. It performs no model call in this PR.
#[async_trait]
pub trait GoalExecutor: Send + Sync {
    async fn preflight(&self, request: &GoalExecutionPreflight) -> GoalExecutionReadiness;
    async fn launch(&self, scope: GoalExecutionScope) -> Result<(), ()>;
}

/// The sole transport-neutral Goal Mode controller facade.
pub struct GoalRuntime {
    config: Arc<Config>,
    goals: Arc<dyn GoalTaskRegistry>,
    executor: Arc<dyn GoalExecutor>,
    boot_id: String,
}

impl GoalRuntime {
    /// Build a controller for one runtime configuration.
    ///
    /// This dormant controller deliberately has no hot-reload surface. PR4
    /// owns the atomic policy-revocation cutover and will replace complete
    /// runtime instances only after it has fenced affected goals.
    pub fn new(
        config: Arc<Config>,
        goals: Arc<dyn GoalTaskRegistry>,
        executor: Arc<dyn GoalExecutor>,
        boot_id: impl Into<String>,
    ) -> Result<Self, GoalIngressError> {
        let boot_id = required_ingress_value("boot_id", boot_id.into())?;
        Ok(Self {
            config,
            goals,
            executor,
            boot_id,
        })
    }

    /// Submit exactly one already-parsed command bound to trusted ingress.
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
            GoalCommand::Start { budget, objective } => {
                self.start(ingress, budget, objective).await
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
        objective: String,
    ) -> Result<GoalResponse, GoalRuntimeError> {
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
            effective_token_limit: token_limit,
            effective_cost_limit_usd: cost_limit_usd,
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
            pending_call_id: None,
            pending_call_epoch: None,
            accounting_state: GoalAccountingState::Complete,
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
        let Some((task, goal)) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal));
        };
        Ok(GoalResponse::status(
            GoalMessageKey::Status,
            status_projection(&task, &goal),
        ))
    }

    async fn budget(&self, ingress: GoalIngressContext) -> Result<GoalResponse, GoalRuntimeError> {
        let Some((task, goal)) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal));
        };
        Ok(GoalResponse::status(
            GoalMessageKey::Budget,
            status_projection(&task, &goal),
        ))
    }

    async fn set_budget(
        &self,
        ingress: GoalIngressContext,
        request: GoalBudgetRequest,
    ) -> Result<GoalResponse, GoalRuntimeError> {
        match request {
            GoalBudgetRequest::Limits {
                token_limit,
                cost_limit_usd,
            } => {
                self.update_budget_for_current(ingress, token_limit, cost_limit_usd)
                    .await
            }
            GoalBudgetRequest::Unlimited => {
                self.update_budget_for_current(ingress, None, None).await
            }
            // The parser never produces this shape for `budget set`; keep the
            // controller strict if an internal caller constructs it directly.
            GoalBudgetRequest::Defaults => {
                Ok(GoalResponse::message(GoalMessageKey::InvalidCommand))
            }
        }
    }

    async fn update_budget_for_current(
        &self,
        ingress: GoalIngressContext,
        token_limit: Option<u64>,
        cost_limit_usd: Option<f64>,
    ) -> Result<GoalResponse, GoalRuntimeError> {
        let (token_limit, cost_limit_usd) =
            match validate_goal_limits((token_limit, cost_limit_usd)) {
                Ok(limits) => limits,
                Err(GoalRuntimeError::InvalidBudget) => {
                    return Ok(GoalResponse::message(GoalMessageKey::InvalidBudget));
                }
                Err(error) => return Err(error),
            };
        let Some((task, _)) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal));
        };
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
        let Some((task, _)) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal));
        };
        if task.status == TaskStatus::Paused {
            return Ok(GoalResponse::message(GoalMessageKey::AlreadyPaused));
        }
        if task.status != TaskStatus::Running {
            return Ok(GoalResponse::message(GoalMessageKey::Terminal));
        }
        let pause = GoalPauseState {
            reason: GoalPauseReason::OperatorPaused,
            description: None,
            blockers: Vec::new(),
        };
        let result = self
            .goals
            .pause_session_goal(&task.id, ingress.session_id(), task.execution_epoch, pause)
            .await
            .map_err(GoalRuntimeError::control_plane)?;
        Ok(GoalResponse::message(transition_message(
            result,
            GoalMessageKey::Paused,
        )))
    }

    async fn resume(&self, ingress: GoalIngressContext) -> Result<GoalResponse, GoalRuntimeError> {
        let Some((task, goal)) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal));
        };
        if task.status != TaskStatus::Paused {
            return Ok(GoalResponse::message(if task.status.is_terminal() {
                GoalMessageKey::Terminal
            } else {
                GoalMessageKey::Stale
            }));
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
        let result = self
            .goals
            .resume_session_goal(
                &task.id,
                ingress.session_id(),
                task.execution_epoch,
                std::process::id(),
                &self.boot_id,
            )
            .await
            .map_err(GoalRuntimeError::control_plane)?;
        match result {
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
        let Some((task, _)) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal));
        };
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
        let Some((task, _)) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::Stale));
        };
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
                let config = &self.config;
                let limits = (
                    config.goal.default_token_limit,
                    config.goal.default_cost_limit_usd,
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
    ) -> Result<Option<(TaskRecord, GoalTaskRecord)>, GoalRuntimeError> {
        let Some(task) = self
            .goals
            .current_goal_for_session(ingress.session_id())
            .await
            .map_err(GoalRuntimeError::control_plane)?
        else {
            return Ok(None);
        };
        if !task_matches_ingress(&task, ingress) {
            // A canonical session key is necessary but not a licence to bind
            // a different trusted transport context to its durable goal.
            return Ok(None);
        }
        let goal = self
            .goals
            .get_goal_task(&task.id)
            .await
            .map_err(GoalRuntimeError::control_plane)?
            .ok_or(GoalRuntimeError::MissingExtension)?;
        Ok(Some((task, goal)))
    }
}

/// Non-user-facing controller errors. Adapters only receive typed responses.
#[derive(Debug, thiserror::Error)]
pub enum GoalRuntimeError {
    #[error("goal_control_plane")]
    ControlPlane(#[source] anyhow::Error),
    #[error("goal_budget_required")]
    BudgetRequired,
    #[error("goal_invalid_budget")]
    InvalidBudget,
    #[error("goal_missing_extension")]
    MissingExtension,
}

impl GoalRuntimeError {
    fn control_plane(error: anyhow::Error) -> Self {
        Self::ControlPlane(error)
    }
}

fn readiness_message_key(readiness: GoalExecutionReadiness) -> GoalMessageKey {
    match readiness {
        GoalExecutionReadiness::Ready => GoalMessageKey::Started,
        GoalExecutionReadiness::PricingUnavailable => GoalMessageKey::PricingUnavailable,
        GoalExecutionReadiness::VerifierUnavailable => GoalMessageKey::VerifierUnavailable,
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

fn status_projection(task: &TaskRecord, goal: &GoalTaskRecord) -> GoalStatusProjection {
    GoalStatusProjection {
        task_id: task.id.clone(),
        status: task.status,
        resumable: task.status == TaskStatus::Paused
            && goal.accounting_state == GoalAccountingState::Complete
            && goal.pending_call_id.is_none()
            && goal.pending_call_epoch.is_none(),
        token_limit: goal.effective_token_limit,
        cost_limit_usd: goal.effective_cost_limit_usd,
        accounting_state: goal.accounting_state,
        pause_reason: goal.pause_reason,
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
    if token_limit.is_some_and(|limit| limit == 0 || limit > i64::MAX as u64)
        || cost_limit_usd.is_some_and(|limit| !limit.is_finite() || limit <= 0.0)
    {
        return Err(GoalRuntimeError::InvalidBudget);
    }
    Ok((token_limit, cost_limit_usd))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use parking_lot::Mutex;
    use zeroclaw_config::schema::Config;
    use zeroclaw_control_plane::{
        GoalAccountingState, GoalPauseReason, GoalPauseState, GoalTaskRegistry,
        GoalTransitionResult, SqliteTaskStore, TaskStatus,
    };

    use super::{
        GoalBudgetRequest, GoalCommand, GoalCommandErrorKey, GoalExecutionPreflight,
        GoalExecutionReadiness, GoalExecutionScope, GoalExecutor, GoalIngressContext,
        GoalMessageKey, GoalResponse, GoalRuntime, GoalTurnKind, parse_goal_command,
    };

    #[derive(Clone)]
    enum LaunchAction {
        Succeed,
        Fail,
        FenceThenFail(Arc<SqliteTaskStore>),
    }

    struct FakeExecutor {
        readiness: Mutex<GoalExecutionReadiness>,
        action: Mutex<LaunchAction>,
        launches: Mutex<Vec<GoalExecutionScope>>,
        preflights: Mutex<Vec<GoalExecutionPreflight>>,
    }

    impl FakeExecutor {
        fn new(readiness: GoalExecutionReadiness, action: LaunchAction) -> Self {
            Self {
                readiness: Mutex::new(readiness),
                action: Mutex::new(action),
                launches: Mutex::new(Vec::new()),
                preflights: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl GoalExecutor for FakeExecutor {
        async fn preflight(&self, request: &GoalExecutionPreflight) -> GoalExecutionReadiness {
            self.preflights.lock().push(request.clone());
            *self.readiness.lock()
        }

        async fn launch(&self, scope: GoalExecutionScope) -> Result<(), ()> {
            self.launches.lock().push(scope.clone());
            let action = { self.action.lock().clone() };
            match action {
                LaunchAction::Succeed => Ok(()),
                LaunchAction::Fail => Err(()),
                LaunchAction::FenceThenFail(store) => {
                    let result = store
                        .pause_session_goal(
                            &scope.task_id,
                            &scope.session_id,
                            scope.execution_epoch,
                            GoalPauseState {
                                reason: GoalPauseReason::OperatorPaused,
                                description: None,
                                blockers: Vec::new(),
                            },
                        )
                        .await
                        .expect("test fence persists");
                    assert_eq!(result, GoalTransitionResult::Applied);
                    Err(())
                }
            }
        }
    }

    fn enabled_config() -> Arc<Config> {
        let mut config = Config::default();
        config.goal.enabled = true;
        config.goal.default_token_limit = Some(64);
        config.goal.verifier.provider = "openai.verifier".into();
        let _ = config.providers.models.ensure("openai", "verifier");
        Arc::new(config)
    }

    fn runtime(
        config: Arc<Config>,
        store: Arc<SqliteTaskStore>,
        executor: Arc<FakeExecutor>,
    ) -> GoalRuntime {
        let goals: Arc<dyn GoalTaskRegistry> = store;
        GoalRuntime::new(config, goals, executor, "test-boot").expect("valid controller")
    }

    fn matrix(session_id: &str) -> GoalIngressContext {
        GoalIngressContext::matrix(session_id, "main", "matrix.room", "@alice:example.test")
            .expect("trusted matrix context")
    }

    #[test]
    fn start_requires_the_delimiter_and_preserves_the_objective_verbatim() {
        let command = parse_goal_command(
            "/goal start --tokens 250 --cost-usd 1.25 -- Ship the release -- safely",
        )
        .expect("the RFC start grammar should parse");

        assert_eq!(
            command,
            GoalCommand::Start {
                budget: GoalBudgetRequest::Limits {
                    token_limit: Some(250),
                    cost_limit_usd: Some(1.25),
                },
                objective: "Ship the release -- safely".into(),
            }
        );
    }

    #[test]
    fn every_permitted_non_start_command_has_one_exact_shape() {
        let cases = [
            ("/goal status", GoalCommand::Status),
            ("/goal budget", GoalCommand::Budget),
            ("/goal pause", GoalCommand::Pause),
            ("/goal resume", GoalCommand::Resume),
            ("/goal cancel", GoalCommand::Cancel),
            ("/goal help", GoalCommand::Help),
            (
                "/goal budget set --tokens 42",
                GoalCommand::SetBudget(GoalBudgetRequest::Limits {
                    token_limit: Some(42),
                    cost_limit_usd: None,
                }),
            ),
            (
                "/goal budget set --unlimited",
                GoalCommand::SetBudget(GoalBudgetRequest::Unlimited),
            ),
            (
                "/goal start --unlimited -- retained objective",
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Unlimited,
                    objective: "retained objective".into(),
                },
            ),
            (
                "/goal start --tokens 9223372036854775807 -- largest valid limit",
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Limits {
                        token_limit: Some(i64::MAX as u64),
                        cost_limit_usd: None,
                    },
                    objective: "largest valid limit".into(),
                },
            ),
            (
                "/goal start -- defaulted objective",
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    objective: "defaulted objective".into(),
                },
            ),
        ];

        for (input, expected) in cases {
            assert_eq!(parse_goal_command(input).unwrap(), expected, "{input}");
        }
    }

    #[test]
    fn rejected_forms_never_gain_an_alternate_authority_or_budget_meaning() {
        let cases = [
            ("/goal", GoalCommandErrorKey::InvalidGrammar),
            ("/goal@bot status", GoalCommandErrorKey::InvalidGrammar),
            (
                "/goal start objective without delimiter",
                GoalCommandErrorKey::ObjectiveRequired,
            ),
            ("/goal start --   ", GoalCommandErrorKey::ObjectiveRequired),
            (
                "/goal start --tokens=42 -- objective",
                GoalCommandErrorKey::InvalidArguments,
            ),
            (
                "/goal start --tokens 0 -- objective",
                GoalCommandErrorKey::InvalidBudget,
            ),
            (
                "/goal start --tokens 9223372036854775808 -- objective",
                GoalCommandErrorKey::InvalidBudget,
            ),
            (
                "/goal start --cost-usd NaN -- objective",
                GoalCommandErrorKey::InvalidBudget,
            ),
            (
                "/goal start --unlimited --tokens 42 -- objective",
                GoalCommandErrorKey::InvalidArguments,
            ),
            ("/goal budget set", GoalCommandErrorKey::InvalidBudget),
            (
                "/goal budget set --cost 1",
                GoalCommandErrorKey::InvalidArguments,
            ),
            ("/goal status task-1", GoalCommandErrorKey::InvalidArguments),
            ("/goal resume task-1", GoalCommandErrorKey::InvalidArguments),
            (
                "/goal objective rewrite",
                GoalCommandErrorKey::InvalidArguments,
            ),
        ];

        for (input, expected) in cases {
            assert_eq!(
                parse_goal_command(input).unwrap_err().key,
                expected,
                "{input}"
            );
        }
    }

    #[tokio::test]
    async fn disabled_mode_fails_closed_while_help_remains_a_local_grammar_response() {
        let config = Arc::new(Config::default());
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let executor = Arc::new(FakeExecutor::new(
            GoalExecutionReadiness::Ready,
            LaunchAction::Succeed,
        ));
        let runtime = runtime(config, store, executor);

        assert_eq!(
            runtime
                .submit(matrix("s-1"), GoalCommand::Help)
                .await
                .unwrap()
                .key,
            GoalMessageKey::Help
        );
        assert_eq!(
            runtime
                .submit(matrix("s-1"), GoalCommand::Status)
                .await
                .unwrap()
                .key,
            GoalMessageKey::Disabled
        );
    }

    #[tokio::test]
    async fn start_without_an_explicit_or_configured_limit_is_rejected_without_a_goal() {
        let mut config = Config::default();
        config.goal.enabled = true;
        config.goal.verifier.provider = "openai.verifier".into();
        let _ = config.providers.models.ensure("openai", "verifier");
        let config = Arc::new(config);
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let executor = Arc::new(FakeExecutor::new(
            GoalExecutionReadiness::Ready,
            LaunchAction::Succeed,
        ));
        let runtime = runtime(config, Arc::clone(&store), executor);

        assert_eq!(
            runtime
                .submit(
                    matrix("no-default-limit"),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Defaults,
                        objective: "must have a budget".into(),
                    },
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::BudgetRequired
        );
        assert!(
            store
                .current_goal_for_session("no-default-limit")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn controller_transition_table_keeps_the_current_terminal_goal_visible_until_replaced() {
        let config = enabled_config();
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let executor = Arc::new(FakeExecutor::new(
            GoalExecutionReadiness::Ready,
            LaunchAction::Succeed,
        ));
        let runtime = runtime(config, Arc::clone(&store), Arc::clone(&executor));
        let ingress = matrix("s-1");

        let started = runtime
            .submit(
                ingress.clone(),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    objective: "ship the controller".into(),
                },
            )
            .await
            .unwrap();
        assert_eq!(started.key, GoalMessageKey::Started);
        assert_eq!(executor.launches.lock().len(), 1);
        let first = store
            .current_goal_for_session("s-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first.execution_epoch, 1);

        let status = runtime
            .submit(ingress.clone(), GoalCommand::Status)
            .await
            .unwrap();
        assert_eq!(status.status.unwrap().status, TaskStatus::Running);
        assert_eq!(
            runtime
                .submit(
                    ingress.clone(),
                    GoalCommand::SetBudget(GoalBudgetRequest::Limits {
                        token_limit: None,
                        cost_limit_usd: Some(3.0),
                    }),
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::BudgetUpdated
        );
        assert_eq!(
            runtime
                .submit(
                    ingress.clone(),
                    GoalCommand::SetBudget(GoalBudgetRequest::Defaults),
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::InvalidCommand
        );
        assert_eq!(
            runtime
                .submit(ingress.clone(), GoalCommand::Pause)
                .await
                .unwrap()
                .key,
            GoalMessageKey::Paused
        );
        assert_eq!(
            runtime
                .submit(ingress.clone(), GoalCommand::Pause)
                .await
                .unwrap()
                .key,
            GoalMessageKey::AlreadyPaused
        );
        assert_eq!(
            runtime
                .submit(ingress.clone(), GoalCommand::Resume)
                .await
                .unwrap()
                .key,
            GoalMessageKey::Resumed
        );
        let resumed = store
            .current_goal_for_session("s-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resumed.execution_epoch, 3);
        assert_eq!(executor.launches.lock()[1].turn_kind, GoalTurnKind::Resume);

        assert_eq!(
            runtime
                .submit(ingress.clone(), GoalCommand::Cancel)
                .await
                .unwrap()
                .key,
            GoalMessageKey::Cancelled
        );
        assert_eq!(
            runtime
                .submit(ingress.clone(), GoalCommand::Cancel)
                .await
                .unwrap()
                .key,
            GoalMessageKey::AlreadyCancelled
        );
        let terminal = runtime
            .submit(ingress.clone(), GoalCommand::Status)
            .await
            .unwrap();
        assert_eq!(terminal.status.unwrap().status, TaskStatus::Cancelled);

        assert_eq!(
            runtime
                .submit(
                    ingress,
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Unlimited,
                        objective: "replacement".into(),
                    },
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::Started
        );
        let replacement = store
            .current_goal_for_session("s-1")
            .await
            .unwrap()
            .unwrap();
        assert_ne!(replacement.id, first.id);
        assert_eq!(replacement.status, TaskStatus::Running);
    }

    #[tokio::test]
    async fn preflight_fails_closed_for_limited_pricing_but_not_for_token_only_or_unlimited_goals()
    {
        let config = enabled_config();
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let executor = Arc::new(FakeExecutor::new(
            GoalExecutionReadiness::PricingUnavailable,
            LaunchAction::Succeed,
        ));
        let runtime = runtime(config, Arc::clone(&store), executor);

        assert_eq!(
            runtime
                .submit(
                    matrix("cost"),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Limits {
                            token_limit: None,
                            cost_limit_usd: Some(1.0),
                        },
                        objective: "cost limited".into(),
                    },
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::PricingUnavailable
        );
        assert!(
            store
                .current_goal_for_session("cost")
                .await
                .unwrap()
                .is_none()
        );

        for (session, budget) in [
            (
                "tokens",
                GoalBudgetRequest::Limits {
                    token_limit: Some(12),
                    cost_limit_usd: None,
                },
            ),
            ("unlimited", GoalBudgetRequest::Unlimited),
        ] {
            assert_eq!(
                runtime
                    .submit(
                        matrix(session),
                        GoalCommand::Start {
                            budget,
                            objective: "no cost limit".into(),
                        },
                    )
                    .await
                    .unwrap()
                    .key,
                GoalMessageKey::Started
            );
        }
    }

    #[tokio::test]
    async fn launch_failure_is_terminal_and_a_stale_launch_result_cannot_overwrite_a_newer_fence() {
        let config = enabled_config();
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let failed_executor = Arc::new(FakeExecutor::new(
            GoalExecutionReadiness::Ready,
            LaunchAction::Fail,
        ));
        let failed_runtime = runtime(Arc::clone(&config), Arc::clone(&store), failed_executor);
        assert_eq!(
            failed_runtime
                .submit(
                    matrix("fail"),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Unlimited,
                        objective: "will not launch".into(),
                    },
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::ExecutorStartFailed
        );
        assert_eq!(
            store
                .current_goal_for_session("fail")
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Failed
        );

        let fenced_store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let fenced_executor = Arc::new(FakeExecutor::new(
            GoalExecutionReadiness::Ready,
            LaunchAction::FenceThenFail(Arc::clone(&fenced_store)),
        ));
        let fenced_runtime = runtime(config, Arc::clone(&fenced_store), fenced_executor);
        assert_eq!(
            fenced_runtime
                .submit(
                    matrix("fence"),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Unlimited,
                        objective: "fenced launch".into(),
                    },
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::Stale
        );
        assert_eq!(
            fenced_store
                .current_goal_for_session("fence")
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Paused
        );
    }

    #[tokio::test]
    async fn ingress_keeps_sessions_isolated_and_never_persists_a_zerocode_tui_id() {
        let config = enabled_config();
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let executor = Arc::new(FakeExecutor::new(
            GoalExecutionReadiness::Ready,
            LaunchAction::Succeed,
        ));
        let runtime = runtime(config, Arc::clone(&store), executor);

        for ingress in [
            GoalIngressContext::matrix("matrix", "main", "same-route", "@alice:example.test")
                .unwrap(),
            GoalIngressContext::zero_code("tui", "main", "same-route", "transient-tui-id").unwrap(),
        ] {
            assert_eq!(
                runtime
                    .submit(
                        ingress,
                        GoalCommand::Start {
                            budget: GoalBudgetRequest::Unlimited,
                            objective: "isolated".into(),
                        },
                    )
                    .await
                    .unwrap()
                    .key,
                GoalMessageKey::Started
            );
        }
        assert_eq!(
            store
                .current_goal_for_session("matrix")
                .await
                .unwrap()
                .unwrap()
                .principal_id
                .as_deref(),
            Some("@alice:example.test")
        );
        assert_eq!(
            runtime
                .submit(
                    GoalIngressContext::matrix(
                        "matrix",
                        "main",
                        "same-route",
                        "@mallory:example.test",
                    )
                    .unwrap(),
                    GoalCommand::Status,
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::NoCurrentGoal
        );
        assert!(
            store
                .current_goal_for_session("tui")
                .await
                .unwrap()
                .unwrap()
                .principal_id
                .is_none()
        );
    }

    #[tokio::test]
    async fn mismatched_ingress_cannot_expose_or_replace_a_session_goal() {
        let config = enabled_config();
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let executor = Arc::new(FakeExecutor::new(
            GoalExecutionReadiness::Ready,
            LaunchAction::Succeed,
        ));
        let runtime = runtime(config, Arc::clone(&store), Arc::clone(&executor));
        let owner = matrix("shared-session");
        let mismatched = GoalIngressContext::matrix(
            "shared-session",
            "main",
            "matrix.room",
            "@mallory:example.test",
        )
        .unwrap();

        assert_eq!(
            runtime
                .submit(
                    owner.clone(),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Defaults,
                        objective: "owner goal".into(),
                    },
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::Started
        );
        let active_id = store
            .current_goal_for_session("shared-session")
            .await
            .unwrap()
            .unwrap()
            .id;
        assert_eq!(
            runtime
                .submit(
                    mismatched.clone(),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Unlimited,
                        objective: "must not discover active goal".into(),
                    },
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::NoCurrentGoal
        );
        assert_eq!(
            store
                .current_goal_for_session("shared-session")
                .await
                .unwrap()
                .unwrap()
                .id,
            active_id
        );

        assert_eq!(
            runtime
                .submit(owner, GoalCommand::Cancel)
                .await
                .unwrap()
                .key,
            GoalMessageKey::Cancelled
        );
        assert_eq!(
            runtime
                .submit(
                    mismatched,
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Unlimited,
                        objective: "must not replace terminal goal".into(),
                    },
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::NoCurrentGoal
        );
        let terminal = store
            .current_goal_for_session("shared-session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(terminal.id, active_id);
        assert_eq!(terminal.status, TaskStatus::Cancelled);
        assert_eq!(
            terminal.principal_id.as_deref(),
            Some("@alice:example.test")
        );
        assert_eq!(executor.launches.lock().len(), 1);
        assert_eq!(executor.preflights.lock().len(), 1);
    }

    #[tokio::test]
    async fn status_only_marks_paused_goals_resumable_when_pr1_can_resume_them() {
        let config = enabled_config();
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let executor = Arc::new(FakeExecutor::new(
            GoalExecutionReadiness::Ready,
            LaunchAction::Succeed,
        ));
        let runtime = runtime(config, Arc::clone(&store), executor);
        let ingress = matrix("resume-eligibility");
        assert_eq!(
            runtime
                .submit(
                    ingress.clone(),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Defaults,
                        objective: "verify resume eligibility".into(),
                    },
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::Started
        );
        let task = store
            .current_goal_for_session("resume-eligibility")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            store
                .admit_pending_call(&task.id, "resume-eligibility", 1, "pending-call")
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            runtime
                .submit(ingress.clone(), GoalCommand::Pause)
                .await
                .unwrap()
                .key,
            GoalMessageKey::Paused
        );
        let pending = runtime
            .submit(ingress.clone(), GoalCommand::Status)
            .await
            .unwrap()
            .status
            .unwrap();
        assert!(!pending.resumable);

        assert_eq!(
            store
                .settle_pending_call(
                    &task.id,
                    "resume-eligibility",
                    1,
                    "pending-call",
                    GoalAccountingState::Missing,
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        let incomplete = runtime
            .submit(ingress, GoalCommand::Status)
            .await
            .unwrap()
            .status
            .unwrap();
        assert!(!incomplete.resumable);
    }

    #[tokio::test]
    async fn typed_start_rejects_limits_outside_the_sqlite_integer_domain() {
        let config = enabled_config();
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let executor = Arc::new(FakeExecutor::new(
            GoalExecutionReadiness::Ready,
            LaunchAction::Succeed,
        ));
        let runtime = runtime(config, Arc::clone(&store), executor);

        assert_eq!(
            runtime
                .submit(
                    matrix("too-large-token-limit"),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Limits {
                            token_limit: Some(i64::MAX as u64 + 1),
                            cost_limit_usd: None,
                        },
                        objective: "must fail before persistence".into(),
                    },
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::InvalidBudget
        );
        assert!(
            store
                .current_goal_for_session("too-large-token-limit")
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(
            runtime
                .submit(
                    matrix("invalid-cost-limit"),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Limits {
                            token_limit: None,
                            cost_limit_usd: Some(f64::INFINITY),
                        },
                        objective: "must fail before persistence".into(),
                    },
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::InvalidBudget
        );
        assert!(
            store
                .current_goal_for_session("invalid-cost-limit")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn typed_budget_update_rejects_invalid_limits_without_mutating_the_goal() {
        let config = enabled_config();
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let executor = Arc::new(FakeExecutor::new(
            GoalExecutionReadiness::Ready,
            LaunchAction::Succeed,
        ));
        let runtime = runtime(config, Arc::clone(&store), executor);
        let ingress = matrix("typed-budget-validation");

        assert_eq!(
            runtime
                .submit(
                    ingress.clone(),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Defaults,
                        objective: "retain the valid limits".into(),
                    },
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::Started
        );
        let task = store
            .current_goal_for_session("typed-budget-validation")
            .await
            .unwrap()
            .unwrap();
        let original = store.get_goal_task(&task.id).await.unwrap().unwrap();

        for request in [
            GoalBudgetRequest::Limits {
                token_limit: Some(0),
                cost_limit_usd: None,
            },
            GoalBudgetRequest::Limits {
                token_limit: None,
                cost_limit_usd: Some(0.0),
            },
            GoalBudgetRequest::Limits {
                token_limit: Some(i64::MAX as u64 + 1),
                cost_limit_usd: None,
            },
            GoalBudgetRequest::Limits {
                token_limit: None,
                cost_limit_usd: Some(f64::INFINITY),
            },
        ] {
            let response = runtime
                .submit(ingress.clone(), GoalCommand::SetBudget(request))
                .await;
            assert!(
                matches!(
                    &response,
                    Ok(GoalResponse {
                        key: GoalMessageKey::InvalidBudget,
                        ..
                    })
                ),
                "invalid direct budget must yield the typed result: {response:?}",
            );
            let persisted = store.get_goal_task(&task.id).await.unwrap().unwrap();
            assert_eq!(
                persisted.effective_token_limit,
                original.effective_token_limit
            );
            assert_eq!(
                persisted.effective_cost_limit_usd,
                original.effective_cost_limit_usd
            );
        }
    }

    #[tokio::test]
    async fn stale_launch_cannot_start_or_fail_a_same_binding_successor() {
        let config = enabled_config();
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let executor = Arc::new(FakeExecutor::new(
            GoalExecutionReadiness::Ready,
            LaunchAction::Succeed,
        ));
        let runtime = runtime(config, Arc::clone(&store), Arc::clone(&executor));
        let ingress = matrix("stale-launch");

        assert_eq!(
            runtime
                .submit(
                    ingress.clone(),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Defaults,
                        objective: "predecessor".into(),
                    },
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::Started
        );
        let predecessor = store
            .current_goal_for_session("stale-launch")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            runtime
                .submit(ingress.clone(), GoalCommand::Cancel)
                .await
                .unwrap()
                .key,
            GoalMessageKey::Cancelled
        );
        assert_eq!(
            runtime
                .submit(
                    ingress.clone(),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Defaults,
                        objective: "successor".into(),
                    },
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::Started
        );
        let successor = store
            .current_goal_for_session("stale-launch")
            .await
            .unwrap()
            .unwrap();
        executor.launches.lock().clear();

        assert_eq!(
            runtime
                .launch_exact(
                    ingress,
                    &predecessor.id,
                    predecessor.execution_epoch,
                    GoalTurnKind::Start,
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::Stale
        );
        assert!(executor.launches.lock().is_empty());
        let current = store
            .current_goal_for_session("stale-launch")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(current.id, successor.id);
        assert_eq!(current.status, TaskStatus::Running);
        assert_ne!(current.id, predecessor.id);
    }
}
