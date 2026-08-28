//! Transport-neutral, dormant Goal Mode controller contract.
//!
//! This module owns the shared command grammar and trusted ingress boundary.
//! Concrete command surfaces and provider execution remain later stages.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use zeroclaw_config::{cost::CostTracker, schema::Config};

use crate::control_plane::{
    GoalAccountingState, GoalBlocker, GoalPauseReason, GoalPauseState, GoalTaskRecord,
    GoalTaskRegistry, GoalToolPhase, GoalTransitionResult, TaskKind, TaskRecord, TaskStatus,
};

pub mod execution;
pub mod host;

/// Bound durable, user-supplied Goal text before it can reach SQLite or a
/// future verifier prompt. This is the same 20k-character envelope used for
/// other durable user-authored prompt material, but is kept local to this
/// transport-neutral module so the controller can later move unchanged into a
/// focused execution crate.
const MAX_GOAL_DECLARATION_CHARS: usize = 20_000;

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
    InputTooLong,
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
            GoalCommandErrorKey::InputTooLong => "goal_input_too_long",
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
            validate_goal_declaration(criteria)?;
            validate_goal_declaration(objective)?;
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
fn input_too_long() -> GoalCommandParseError {
    error(GoalCommandErrorKey::InputTooLong)
}
fn invalid_budget() -> GoalCommandParseError {
    error(GoalCommandErrorKey::InvalidBudget)
}
fn invalid_arguments() -> GoalCommandParseError {
    error(GoalCommandErrorKey::InvalidArguments)
}

fn validate_goal_declaration(value: &str) -> Result<(), GoalCommandParseError> {
    (value.chars().count() <= MAX_GOAL_DECLARATION_CHARS)
        .then_some(())
        .ok_or_else(input_too_long)
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

/// Semantic response keys. Concrete transports render these through Fluent
/// only after their adapters are introduced in a later stage.
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
    InputTooLong,
    BudgetExhausted,
    AccountingIncomplete,
    ReservationUnavailable,
    ReservationExceedsBudget,
    ToolPhaseNonResumable,
    Terminal,
    ExecutorStartFailed,
    Stale,
}

/// A transport-neutral semantic result with no rendered text.
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

/// Current Goal status derived from canonical task rows and the usage ledger.
#[derive(Debug, Clone, PartialEq)]
pub struct GoalStatusProjection {
    pub task_id: String,
    pub status: TaskStatus,
    pub resumable: bool,
    pub token_limit: Option<u64>,
    pub cost_limit_usd: Option<f64>,
    pub known_token_usage: u64,
    pub known_cost_usd: f64,
    pub token_accounting_complete: bool,
    pub cost_accounting_complete: bool,
    pub remaining_tokens: Option<u64>,
    pub remaining_cost_usd: Option<f64>,
    pub accounting_state: GoalAccountingState,
    pub tool_phase: GoalToolPhase,
    pub pause_reason: Option<GoalPauseReason>,
    pub pause_description: Option<String>,
    pub blockers: Vec<GoalBlocker>,
    pub delivered: bool,
    pub terminal_error: Option<String>,
}

/// Process-local transcript contract. It deliberately has no durable store.
#[derive(Debug, Default)]
pub struct GoalWorkingTranscript {
    turns: Vec<GoalWorkingTurn>,
}

impl GoalWorkingTranscript {
    pub fn push(&mut self, turn: GoalWorkingTurn) {
        self.turns.push(turn);
    }

    pub fn turns(&self) -> &[GoalWorkingTurn] {
        &self.turns
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalWorkingTurn {
    pub turn_kind: GoalTurnKind,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalTurnKind {
    Start,
    Resume,
    Continue,
}

/// Exact durable identity submitted to the dormant execution seam.
#[derive(Debug, Clone)]
pub struct GoalExecutionScope {
    pub task_id: String,
    pub session_id: String,
    /// Immutable ingress facts selected before any adapter or model work.
    pub binding: host::GoalSessionBinding,
    /// Shared serialization authority for every logical model operation in this epoch.
    pub operation_permit: Arc<Semaphore>,
    pub execution_epoch: i64,
    pub turn_kind: GoalTurnKind,
    pub cancellation: CancellationToken,
}

/// Read-only facts an execution host uses to plan one exact Goal turn.
///
/// The controller supplies only Goal-owned declaration, binding, and accounting
/// facts. Provider selection and request construction stay inside the concrete
/// host, keeping this transport-neutral boundary extractable.
#[derive(Debug, Clone)]
pub struct GoalExecutionPreflight {
    /// The task and epoch that the prospective execution scope must own.
    pub task_id: String,
    pub session_id: String,
    /// Immutable trusted ingress selected before adapter or model work.
    pub binding: host::GoalSessionBinding,
    pub execution_epoch: i64,
    pub turn_kind: GoalTurnKind,
    pub agent: String,
    /// Immutable durable declaration from which the host constructs its turn.
    pub objective: String,
    pub success_criteria: String,
    pub token_limit: Option<u64>,
    pub cost_limit_usd: Option<f64>,
    /// Durable ledger totals before the next logical operation.
    pub known_token_usage: u64,
    pub known_cost_usd: f64,
    /// Completeness is explicit because a lower-bound total cannot safely
    /// admit a new operation under a finite limit.
    pub token_accounting_complete: bool,
    pub cost_accounting_complete: bool,
}

/// Conservative capacity required by one logical Goal operation.
///
/// Concrete execution hosts derive it from PR1's provider/model operation
/// plan. The controller deliberately receives only the resulting dimensions,
/// preserving its transport- and provider-neutral extraction boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GoalOperationReservation {
    pub token_count: Option<u64>,
    pub cost_usd: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GoalExecutionReadiness {
    Ready(GoalOperationReservation),
    PricingUnavailable,
    VerifierUnavailable,
}

/// Dormant execution seam. This stage performs no provider work.
#[async_trait]
pub trait GoalExecutor: Send + Sync {
    async fn preflight(&self, request: &GoalExecutionPreflight) -> GoalExecutionReadiness;
    async fn launch(&self, scope: GoalExecutionScope) -> Result<(), ()>;
}

struct CurrentGoal {
    task: TaskRecord,
    goal: GoalTaskRecord,
}

/// The complete, controller-owned answer to whether a paused Goal may resume.
/// Keeping this decision independent of transport rendering makes `status` and
/// `resume` consume the same durable and runtime facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GoalResumeReadiness {
    Ready,
    NotPaused,
    AccountingIncomplete,
    ToolPhase,
    BudgetExhausted,
    ReservationUnavailable,
    ReservationExceedsBudget,
    PricingUnavailable,
    VerifierUnavailable,
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
            boot_id: required(boot_id.into())?,
        })
    }

    /// Submit an already-parsed command bound to trusted ingress.
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
            GoalCommand::Status => self.status(ingress, GoalMessageKey::Status).await,
            GoalCommand::Budget => self.status(ingress, GoalMessageKey::Budget).await,
            GoalCommand::SetBudget(request) => self.set_budget(ingress, request).await,
            GoalCommand::Pause => self.pause(ingress).await,
            GoalCommand::Resume => self.resume(ingress).await,
            GoalCommand::Cancel => self.cancel(ingress).await,
            GoalCommand::Help => unreachable!("handled before configuration admission"),
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
        if validate_goal_declaration(&objective).is_err()
            || validate_goal_declaration(&success_criteria).is_err()
        {
            return Ok(GoalResponse::message(GoalMessageKey::InputTooLong));
        }
        if let Some(current) = self.current_goal(&ingress).await?
            && !current.task.status.is_terminal()
        {
            return Ok(GoalResponse::message(GoalMessageKey::AlreadyActive));
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
        let id = Uuid::new_v4().to_string();
        match self
            .preflight(
                &ingress,
                &id,
                1,
                GoalTurnKind::Start,
                &objective,
                &success_criteria,
                token_limit,
                cost_limit_usd,
                (0, 0.0, true),
            )
            .await
        {
            GoalExecutionReadiness::Ready(reservation)
                if reservation_is_available(token_limit, cost_limit_usd, reservation)
                    && reservation_fits_budget(
                        0,
                        0.0,
                        token_limit,
                        cost_limit_usd,
                        reservation,
                    ) => {}
            GoalExecutionReadiness::Ready(reservation)
                if !reservation_is_available(token_limit, cost_limit_usd, reservation) =>
            {
                return Ok(GoalResponse::message(
                    GoalMessageKey::ReservationUnavailable,
                ));
            }
            GoalExecutionReadiness::Ready(_) => {
                return Ok(GoalResponse::message(
                    GoalMessageKey::ReservationExceedsBudget,
                ));
            }
            GoalExecutionReadiness::PricingUnavailable if cost_limit_usd.is_some() => {
                return Ok(GoalResponse::message(GoalMessageKey::PricingUnavailable));
            }
            GoalExecutionReadiness::PricingUnavailable => {}
            GoalExecutionReadiness::VerifierUnavailable => {
                return Ok(GoalResponse::message(GoalMessageKey::VerifierUnavailable));
            }
        }
        let task = TaskRecord {
            id: id.clone(),
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
            principal_id: ingress.durable_principal_id().map(str::to_owned),
            session_id: Some(ingress.session_id.clone()),
            execution_epoch: 0,
            started_at: Utc::now().to_rfc3339(),
            finished_at: None,
        };
        let goal = GoalTaskRecord {
            task_id: id.clone(),
            objective,
            success_criteria,
            effective_token_limit: token_limit,
            effective_cost_limit_usd: cost_limit_usd,
            ..GoalTaskRecord::default()
        };
        match self
            .goals
            .create_or_replace_session_goal(task, goal)
            .await
            .map_err(GoalRuntimeError::control_plane)?
        {
            GoalTransitionResult::Applied => {
                self.launch_exact(ingress, &id, 1, GoalTurnKind::Start)
                    .await
            }
            GoalTransitionResult::Stale => Ok(GoalResponse::message(GoalMessageKey::AlreadyActive)),
            GoalTransitionResult::Missing => Ok(GoalResponse::message(GoalMessageKey::Stale)),
        }
    }

    async fn status(
        &self,
        ingress: GoalIngressContext,
        key: GoalMessageKey,
    ) -> Result<GoalResponse, GoalRuntimeError> {
        let Some(current) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal));
        };
        let totals = self.usage_totals(&current.task.id).await?;
        let terminal_reason = if current.task.status.is_terminal() {
            self.goals
                .terminal_reason_for_session_goal(&current.task.id, ingress.session_id())
                .await
                .map_err(GoalRuntimeError::control_plane)?
                .as_deref()
                .map(sanitize_terminal_reason)
        } else {
            None
        };
        Ok(GoalResponse::status(
            key,
            status_projection(
                &current,
                totals,
                terminal_reason,
                self.resume_readiness(&ingress, &current, totals).await,
            ),
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
            } => validate_limits(token_limit, cost_limit_usd),
            GoalBudgetRequest::Unlimited => Ok((None, None)),
            GoalBudgetRequest::Defaults => Err(GoalRuntimeError::InvalidBudget),
        };
        let (tokens, cost) = match limits {
            Ok(value) => value,
            Err(GoalRuntimeError::InvalidBudget) => {
                return Ok(GoalResponse::message(GoalMessageKey::InvalidBudget));
            }
            Err(error) => return Err(error),
        };
        let Some(current) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal));
        };
        if !matches!(
            current.task.status,
            TaskStatus::Running | TaskStatus::Paused
        ) {
            return Ok(GoalResponse::message(GoalMessageKey::Terminal));
        }
        let result = self
            .goals
            .update_session_goal_limits(
                &current.task.id,
                ingress.session_id(),
                current.task.execution_epoch,
                tokens,
                cost,
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
        if current.task.status == TaskStatus::Paused {
            return Ok(GoalResponse::message(GoalMessageKey::AlreadyPaused));
        }
        if current.task.status != TaskStatus::Running {
            return Ok(GoalResponse::message(GoalMessageKey::Terminal));
        }
        let result = self
            .goals
            .pause_session_goal(
                &current.task.id,
                ingress.session_id(),
                current.task.execution_epoch,
                GoalPauseState {
                    reason: GoalPauseReason::OperatorPaused,
                    description: None,
                    blockers: Vec::new(),
                },
            )
            .await
            .map_err(GoalRuntimeError::control_plane)?;
        Ok(GoalResponse::message(transition_message(
            result,
            GoalMessageKey::Paused,
        )))
    }

    async fn resume(&self, ingress: GoalIngressContext) -> Result<GoalResponse, GoalRuntimeError> {
        let Some(current) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::NoCurrentGoal));
        };
        if current.task.status != TaskStatus::Paused {
            return Ok(GoalResponse::message(
                if current.task.status.is_terminal() {
                    GoalMessageKey::Terminal
                } else {
                    GoalMessageKey::Stale
                },
            ));
        }
        let totals = self.usage_totals(&current.task.id).await?;
        match self.resume_readiness(&ingress, &current, totals).await {
            GoalResumeReadiness::Ready => {}
            GoalResumeReadiness::ToolPhase => {
                return Ok(GoalResponse::message(GoalMessageKey::ToolPhaseNonResumable));
            }
            GoalResumeReadiness::BudgetExhausted => {
                return Ok(GoalResponse::message(GoalMessageKey::BudgetExhausted));
            }
            GoalResumeReadiness::ReservationUnavailable => {
                return Ok(GoalResponse::message(
                    GoalMessageKey::ReservationUnavailable,
                ));
            }
            GoalResumeReadiness::ReservationExceedsBudget => {
                return Ok(GoalResponse::message(
                    GoalMessageKey::ReservationExceedsBudget,
                ));
            }
            GoalResumeReadiness::AccountingIncomplete => {
                return Ok(GoalResponse::message(GoalMessageKey::AccountingIncomplete));
            }
            GoalResumeReadiness::PricingUnavailable => {
                return Ok(GoalResponse::message(GoalMessageKey::PricingUnavailable));
            }
            GoalResumeReadiness::VerifierUnavailable => {
                return Ok(GoalResponse::message(GoalMessageKey::VerifierUnavailable));
            }
            GoalResumeReadiness::NotPaused => {
                return Ok(GoalResponse::message(GoalMessageKey::Stale));
            }
        }
        match self
            .goals
            .resume_session_goal(
                &current.task.id,
                ingress.session_id(),
                current.task.execution_epoch,
                std::process::id(),
                &self.boot_id,
            )
            .await
            .map_err(GoalRuntimeError::control_plane)?
        {
            GoalTransitionResult::Applied => {
                self.launch_exact(
                    ingress,
                    &current.task.id,
                    current.task.execution_epoch + 1,
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
        if current.task.status == TaskStatus::Cancelled {
            return Ok(GoalResponse::message(GoalMessageKey::AlreadyCancelled));
        }
        if current.task.status.is_terminal() {
            return Ok(GoalResponse::message(GoalMessageKey::Terminal));
        }
        let result = self
            .goals
            .finish_session_goal(
                &current.task.id,
                ingress.session_id(),
                current.task.execution_epoch,
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
        epoch: i64,
        turn_kind: GoalTurnKind,
    ) -> Result<GoalResponse, GoalRuntimeError> {
        let Some(current) = self.current_goal(&ingress).await? else {
            return Ok(GoalResponse::message(GoalMessageKey::Stale));
        };
        if current.task.id != task_id
            || current.task.execution_epoch != epoch
            || current.task.status != TaskStatus::Running
        {
            return Ok(GoalResponse::message(GoalMessageKey::Stale));
        }
        let scope = GoalExecutionScope {
            task_id: task_id.to_owned(),
            session_id: ingress.session_id.clone(),
            binding: host::GoalSessionBinding::new(
                ingress.session_id.clone(),
                ingress.agent.clone(),
                ingress.route.clone(),
                ingress.principal_id.clone(),
            )
            .map_err(|_| {
                GoalRuntimeError::control_plane(anyhow::Error::msg("invalid Goal binding"))
            })?,
            operation_permit: Arc::new(Semaphore::new(1)),
            execution_epoch: epoch,
            turn_kind,
            cancellation: CancellationToken::new(),
        };
        if self.executor.launch(scope).await.is_ok() {
            return Ok(GoalResponse::message(if turn_kind == GoalTurnKind::Start {
                GoalMessageKey::Started
            } else {
                GoalMessageKey::Resumed
            }));
        }
        let result = self
            .goals
            .finish_session_goal(
                task_id,
                ingress.session_id(),
                epoch,
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

    async fn preflight(
        &self,
        ingress: &GoalIngressContext,
        task_id: &str,
        execution_epoch: i64,
        turn_kind: GoalTurnKind,
        objective: &str,
        success_criteria: &str,
        token_limit: Option<u64>,
        cost_limit_usd: Option<f64>,
        totals: (u64, f64, bool),
    ) -> GoalExecutionReadiness {
        let binding = match host::GoalSessionBinding::new(
            ingress.session_id.clone(),
            ingress.agent.clone(),
            ingress.route.clone(),
            ingress.principal_id.clone(),
        ) {
            Ok(binding) => binding,
            Err(_) => return GoalExecutionReadiness::VerifierUnavailable,
        };
        self.executor
            .preflight(&GoalExecutionPreflight {
                task_id: task_id.to_owned(),
                session_id: ingress.session_id.clone(),
                binding,
                execution_epoch,
                turn_kind,
                agent: ingress.agent.clone(),
                objective: objective.to_owned(),
                success_criteria: success_criteria.to_owned(),
                token_limit,
                cost_limit_usd,
                known_token_usage: totals.0,
                known_cost_usd: totals.1,
                token_accounting_complete: true,
                cost_accounting_complete: totals.2,
            })
            .await
    }

    async fn usage_totals(&self, task_id: &str) -> Result<(u64, f64, bool), GoalRuntimeError> {
        let task_id = task_id.to_owned();
        let tracker = Arc::clone(&self.cost_tracker);
        tokio::task::spawn_blocking(move || {
            tracker.get_strict_usage_totals_for_task_with_pricing(&task_id)
        })
        .await
        .map_err(GoalRuntimeError::accounting)?
        .map_err(GoalRuntimeError::accounting)
    }

    async fn resume_readiness(
        &self,
        ingress: &GoalIngressContext,
        current: &CurrentGoal,
        totals: (u64, f64, bool),
    ) -> GoalResumeReadiness {
        if current.task.status != TaskStatus::Paused || current.task.execution_epoch == i64::MAX {
            return GoalResumeReadiness::NotPaused;
        }
        let settled = current.goal.accounting_state == GoalAccountingState::Complete
            && current.goal.pending_call_id.is_none()
            && current.goal.pending_call_epoch.is_none();
        if !settled {
            return GoalResumeReadiness::AccountingIncomplete;
        }
        if current.goal.effective_cost_limit_usd.is_some() && !totals.2 {
            return GoalResumeReadiness::AccountingIncomplete;
        }
        if current.goal.tool_phase != GoalToolPhase::Clean {
            return GoalResumeReadiness::ToolPhase;
        }
        match self
            .preflight(
                ingress,
                &current.task.id,
                current.task.execution_epoch + 1,
                GoalTurnKind::Resume,
                &current.goal.objective,
                &current.goal.success_criteria,
                current.goal.effective_token_limit,
                current.goal.effective_cost_limit_usd,
                totals,
            )
            .await
        {
            GoalExecutionReadiness::Ready(_reservation)
                if budget_is_exhausted(
                    totals.0,
                    totals.1,
                    current.goal.effective_token_limit,
                    current.goal.effective_cost_limit_usd,
                ) =>
            {
                GoalResumeReadiness::BudgetExhausted
            }
            GoalExecutionReadiness::Ready(reservation)
                if !reservation_is_available(
                    current.goal.effective_token_limit,
                    current.goal.effective_cost_limit_usd,
                    reservation,
                ) =>
            {
                GoalResumeReadiness::ReservationUnavailable
            }
            GoalExecutionReadiness::Ready(reservation)
                if !reservation_fits_budget(
                    totals.0,
                    totals.1,
                    current.goal.effective_token_limit,
                    current.goal.effective_cost_limit_usd,
                    reservation,
                ) =>
            {
                GoalResumeReadiness::ReservationExceedsBudget
            }
            GoalExecutionReadiness::Ready(_) => GoalResumeReadiness::Ready,
            GoalExecutionReadiness::PricingUnavailable
                if current.goal.effective_cost_limit_usd.is_some() =>
            {
                GoalResumeReadiness::PricingUnavailable
            }
            GoalExecutionReadiness::PricingUnavailable => GoalResumeReadiness::Ready,
            GoalExecutionReadiness::VerifierUnavailable => GoalResumeReadiness::VerifierUnavailable,
        }
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
                    validate_limits(limits.0, limits.1)
                }
            }
            GoalBudgetRequest::Limits {
                token_limit,
                cost_limit_usd,
            } => validate_limits(token_limit, cost_limit_usd),
            GoalBudgetRequest::Unlimited => Ok((None, None)),
        }
    }

    async fn current_goal(
        &self,
        ingress: &GoalIngressContext,
    ) -> Result<Option<CurrentGoal>, GoalRuntimeError> {
        let Some(task) = self
            .goals
            .current_goal_for_session(ingress.session_id())
            .await
            .map_err(GoalRuntimeError::control_plane)?
        else {
            return Ok(None);
        };
        if task.agent != ingress.agent()
            || task.originator_route.as_deref() != Some(ingress.route())
            || task.principal_id.as_deref() != ingress.durable_principal_id()
        {
            return Ok(None);
        }
        let Some(goal) = self
            .goals
            .get_goal_task(&task.id)
            .await
            .map_err(GoalRuntimeError::control_plane)?
        else {
            return Ok(None);
        };
        Ok(Some(CurrentGoal { task, goal }))
    }
}

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

fn validate_limits(
    token_limit: Option<u64>,
    cost_limit_usd: Option<f64>,
) -> Result<(Option<u64>, Option<f64>), GoalRuntimeError> {
    if (token_limit.is_none() && cost_limit_usd.is_none())
        || token_limit.is_some_and(|limit| limit == 0 || limit > i64::MAX as u64)
        || cost_limit_usd.is_some_and(|limit| !limit.is_finite() || limit <= 0.0)
    {
        Err(GoalRuntimeError::InvalidBudget)
    } else {
        Ok((token_limit, cost_limit_usd))
    }
}

fn transition_message(result: GoalTransitionResult, applied: GoalMessageKey) -> GoalMessageKey {
    match result {
        GoalTransitionResult::Applied => applied,
        GoalTransitionResult::Stale => GoalMessageKey::Stale,
        GoalTransitionResult::Missing => GoalMessageKey::NoCurrentGoal,
    }
}

fn budget_is_exhausted(
    known_tokens: u64,
    known_cost_usd: f64,
    token_limit: Option<u64>,
    cost_limit_usd: Option<f64>,
) -> bool {
    token_limit.is_some_and(|limit| known_tokens >= limit)
        || cost_limit_usd
            .is_some_and(|limit| !known_cost_usd.is_finite() || known_cost_usd >= limit)
}

fn reservation_is_available(
    token_limit: Option<u64>,
    cost_limit_usd: Option<f64>,
    reservation: GoalOperationReservation,
) -> bool {
    (token_limit.is_none() || reservation.token_count.is_some())
        && (cost_limit_usd.is_none()
            || reservation
                .cost_usd
                .is_some_and(|cost| cost.is_finite() && cost >= 0.0))
}

fn reservation_fits_budget(
    known_tokens: u64,
    known_cost_usd: f64,
    token_limit: Option<u64>,
    cost_limit_usd: Option<f64>,
    reservation: GoalOperationReservation,
) -> bool {
    if budget_is_exhausted(known_tokens, known_cost_usd, token_limit, cost_limit_usd) {
        return false;
    }
    let token_fit = match token_limit {
        Some(limit) => reservation
            .token_count
            .and_then(|reserved| known_tokens.checked_add(reserved))
            .is_some_and(|projected| projected <= limit),
        None => true,
    };
    let cost_fit = match cost_limit_usd {
        Some(limit) => reservation.cost_usd.is_some_and(|reserved| {
            reserved.is_finite()
                && reserved >= 0.0
                && known_cost_usd.is_finite()
                && (known_cost_usd + reserved).is_finite()
                && known_cost_usd + reserved <= limit
        }),
        None => true,
    };
    token_fit && cost_fit
}

fn status_projection(
    current: &CurrentGoal,
    totals: (u64, f64, bool),
    terminal_error: Option<String>,
    resume_readiness: GoalResumeReadiness,
) -> GoalStatusProjection {
    let settled = current.goal.accounting_state == GoalAccountingState::Complete
        && current.goal.pending_call_id.is_none()
        && current.goal.pending_call_epoch.is_none();
    let token_complete = settled;
    let cost_complete = settled && totals.2;
    GoalStatusProjection {
        task_id: current.task.id.clone(),
        status: current.task.status,
        resumable: resume_readiness == GoalResumeReadiness::Ready,
        token_limit: current.goal.effective_token_limit,
        cost_limit_usd: current.goal.effective_cost_limit_usd,
        known_token_usage: totals.0,
        known_cost_usd: totals.1,
        token_accounting_complete: token_complete,
        cost_accounting_complete: cost_complete,
        remaining_tokens: token_complete
            .then_some(current.goal.effective_token_limit)
            .flatten()
            .map(|limit| limit.saturating_sub(totals.0)),
        remaining_cost_usd: cost_complete
            .then_some(current.goal.effective_cost_limit_usd)
            .flatten()
            .map(|limit| (limit - totals.1).max(0.0)),
        accounting_state: current.goal.accounting_state,
        tool_phase: current.goal.tool_phase,
        pause_reason: current.goal.pause_reason,
        pause_description: current.goal.pause_description.clone(),
        blockers: current.goal.blockers.clone(),
        delivered: current.task.delivered,
        terminal_error,
    }
}

fn sanitize_terminal_reason(reason: &str) -> String {
    match reason {
        "executor_start_failed"
        | "legacy_goal_missing_session"
        | "accounting_missing_or_invalid"
        | "accounting_outcome_unknown"
        | "pricing_unavailable"
        | "candidate_empty" => reason.to_owned(),
        _ => "goal_terminal_failure".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use zeroclaw_config::{cost::TokenUsage, schema::CostConfig};

    use crate::control_plane::SqliteTaskStore;

    struct FakeExecutor {
        readiness: Mutex<GoalExecutionReadiness>,
        preflights: Mutex<Vec<GoalExecutionPreflight>>,
        launches: Mutex<Vec<GoalExecutionScope>>,
        fail_launch: bool,
    }

    #[async_trait]
    impl GoalExecutor for FakeExecutor {
        async fn preflight(&self, request: &GoalExecutionPreflight) -> GoalExecutionReadiness {
            self.preflights.lock().push(request.clone());
            *self.readiness.lock()
        }

        async fn launch(&self, scope: GoalExecutionScope) -> Result<(), ()> {
            if self.fail_launch {
                return Err(());
            }
            self.launches.lock().push(scope);
            Ok(())
        }
    }

    fn enabled_config() -> Arc<Config> {
        let mut config = Config::default();
        config.goal.enabled = true;
        config.goal.default_token_limit = Some(8);
        config.goal.verifier.provider = "openai.verifier".into();
        let _ = config.providers.models.ensure("openai", "verifier");
        Arc::new(config)
    }

    fn controller(
        readiness: GoalExecutionReadiness,
        fail_launch: bool,
    ) -> (
        GoalRuntime,
        Arc<SqliteTaskStore>,
        Arc<FakeExecutor>,
        tempfile::TempDir,
    ) {
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let directory = tempfile::tempdir().unwrap();
        let tracker = Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap());
        let executor = Arc::new(FakeExecutor {
            readiness: Mutex::new(readiness),
            preflights: Mutex::new(Vec::new()),
            launches: Mutex::new(Vec::new()),
            fail_launch,
        });
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        (
            GoalRuntime::new(enabled_config(), goals, tracker, executor.clone(), "boot").unwrap(),
            store,
            executor,
            directory,
        )
    }

    fn ingress() -> GoalIngressContext {
        GoalIngressContext::trusted("session", "main", "room", Some("principal".into())).unwrap()
    }

    fn start_command() -> GoalCommand {
        GoalCommand::Start {
            budget: GoalBudgetRequest::Defaults,
            success_criteria: "the reported answer meets the stated objective".into(),
            objective: "produce the answer".into(),
        }
    }

    struct DerivingExecutor;

    #[async_trait]
    impl GoalExecutor for DerivingExecutor {
        async fn preflight(&self, request: &GoalExecutionPreflight) -> GoalExecutionReadiness {
            let turn_overhead = match request.turn_kind {
                GoalTurnKind::Start => 1,
                GoalTurnKind::Resume => 2,
                GoalTurnKind::Continue => 3,
            };
            let declaration_len = request
                .objective
                .chars()
                .count()
                .checked_add(request.success_criteria.chars().count())
                .and_then(|count| count.checked_add(turn_overhead))
                .and_then(|count| u64::try_from(count).ok());
            GoalExecutionReadiness::Ready(GoalOperationReservation {
                token_count: declaration_len,
                cost_usd: Some(0.0),
            })
        }

        async fn launch(&self, _scope: GoalExecutionScope) -> Result<(), ()> {
            Ok(())
        }
    }

    fn ready() -> GoalExecutionReadiness {
        GoalExecutionReadiness::Ready(GoalOperationReservation {
            token_count: Some(1),
            cost_usd: Some(0.0),
        })
    }

    fn preflight_for(
        objective: &str,
        success_criteria: &str,
        turn_kind: GoalTurnKind,
    ) -> GoalExecutionPreflight {
        GoalExecutionPreflight {
            task_id: "task".into(),
            session_id: "session".into(),
            binding: host::GoalSessionBinding::new(
                "session".into(),
                "main".into(),
                "route".into(),
                None,
            )
            .unwrap(),
            execution_epoch: 1,
            turn_kind,
            agent: "main".into(),
            objective: objective.into(),
            success_criteria: success_criteria.into(),
            token_limit: Some(100),
            cost_limit_usd: None,
            known_token_usage: 0,
            known_cost_usd: 0.0,
            token_accounting_complete: true,
            cost_accounting_complete: true,
        }
    }

    fn record_goal_usage(tracker: &CostTracker, task_id: String, total_tokens: u64) {
        tracker
            .record_scoped_usage_with_owned_task_attribution(
                TokenUsage::new("test-model", total_tokens, 0, 0, 0.0, 0.0, 0.0),
                Some("main"),
                Some(task_id),
            )
            .unwrap();
    }

    fn record_unpriced_goal_usage(tracker: &CostTracker, task_id: String) {
        tracker
            .record_scoped_usage_with_owned_task_attribution(
                TokenUsage {
                    model: "test-model".into(),
                    input_tokens: 1,
                    output_tokens: 0,
                    cached_input_tokens: 0,
                    total_tokens: 1,
                    cost_usd: 0.0,
                    pricing_available: false,
                    timestamp: Utc::now(),
                },
                Some("main"),
                Some(task_id),
            )
            .unwrap();
    }

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
    fn bounds_objective_and_success_criteria_in_the_parser() {
        let over_limit = "x".repeat(MAX_GOAL_DECLARATION_CHARS + 1);
        for command in [
            format!("/goal start --success-criteria done -- {over_limit}"),
            format!("/goal start --success-criteria {over_limit} -- objective"),
        ] {
            assert_eq!(
                parse_goal_command(&command),
                Err(GoalCommandParseError {
                    key: GoalCommandErrorKey::InputTooLong,
                })
            );
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

    #[tokio::test]
    async fn start_persists_immutable_criteria_and_launches_only_exact_epoch() {
        let (runtime, store, executor, _directory) = controller(ready(), false);

        assert_eq!(
            runtime
                .submit(ingress(), start_command())
                .await
                .unwrap()
                .key,
            GoalMessageKey::Started
        );
        let task = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(task.status, TaskStatus::Running);
        assert_eq!(task.execution_epoch, 1);
        let goal = store.get_goal_task(&task.id).await.unwrap().unwrap();
        assert_eq!(goal.objective, "produce the answer");
        assert_eq!(
            goal.success_criteria,
            "the reported answer meets the stated objective"
        );
        {
            let launches = executor.launches.lock();
            assert_eq!(launches.len(), 1);
            assert_eq!(launches[0].binding.session_id, "session");
            assert_eq!(launches[0].binding.agent, "main");
            assert_eq!(launches[0].binding.route, "room");
            assert_eq!(
                launches[0].binding.principal_id.as_deref(),
                Some("principal")
            );
            assert_eq!(launches[0].operation_permit.available_permits(), 1);
        }
        assert_eq!(
            runtime
                .submit(ingress(), start_command())
                .await
                .unwrap()
                .key,
            GoalMessageKey::AlreadyActive
        );
        assert_eq!(executor.launches.lock().len(), 1);
    }

    #[tokio::test]
    async fn start_preflight_receives_the_exact_durable_declaration_and_binding() {
        let (runtime, store, executor, _directory) = controller(ready(), false);
        let command = GoalCommand::Start {
            budget: GoalBudgetRequest::Defaults,
            success_criteria: "validate the exact declaration".into(),
            objective: "plan this specific goal".into(),
        };

        assert_eq!(
            runtime.submit(ingress(), command).await.unwrap().key,
            GoalMessageKey::Started
        );
        let task = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        let preflights = executor.preflights.lock();
        assert_eq!(preflights.len(), 1);
        let preflight = &preflights[0];
        assert_eq!(preflight.task_id, task.id);
        assert_eq!(preflight.session_id, "session");
        assert_eq!(preflight.execution_epoch, task.execution_epoch);
        assert_eq!(preflight.turn_kind, GoalTurnKind::Start);
        assert_eq!(preflight.objective, "plan this specific goal");
        assert_eq!(preflight.success_criteria, "validate the exact declaration");
    }

    #[tokio::test]
    async fn reservation_derivation_uses_declaration_and_turn_shape() {
        let executor = DerivingExecutor;
        let start = executor
            .preflight(&preflight_for("objective", "criteria", GoalTurnKind::Start))
            .await;
        let changed_declaration = executor
            .preflight(&preflight_for(
                "longer objective",
                "criteria",
                GoalTurnKind::Start,
            ))
            .await;
        let resume = executor
            .preflight(&preflight_for(
                "objective",
                "criteria",
                GoalTurnKind::Resume,
            ))
            .await;

        assert_eq!(
            start,
            GoalExecutionReadiness::Ready(GoalOperationReservation {
                token_count: Some(18),
                cost_usd: Some(0.0),
            })
        );
        assert_ne!(start, changed_declaration);
        assert_ne!(start, resume);
    }

    #[tokio::test]
    async fn budget_update_keeps_a_paused_goal_paused_until_explicit_resume() {
        let (runtime, store, executor, _directory) = controller(ready(), false);
        runtime.submit(ingress(), start_command()).await.unwrap();
        assert_eq!(
            runtime
                .submit(ingress(), GoalCommand::Pause)
                .await
                .unwrap()
                .key,
            GoalMessageKey::Paused
        );
        assert_eq!(
            runtime
                .submit(
                    ingress(),
                    GoalCommand::SetBudget(GoalBudgetRequest::Limits {
                        token_limit: Some(3),
                        cost_limit_usd: None,
                    }),
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::BudgetUpdated
        );
        let paused = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(paused.status, TaskStatus::Paused);
        assert_eq!(executor.launches.lock().len(), 1);
        assert_eq!(
            runtime
                .submit(ingress(), GoalCommand::Resume)
                .await
                .unwrap()
                .key,
            GoalMessageKey::Resumed
        );
        let resumed = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resumed.status, TaskStatus::Running);
        assert_eq!(resumed.execution_epoch, paused.execution_epoch + 1);
        assert_eq!(executor.launches.lock().len(), 2);
        let preflights = executor.preflights.lock();
        let resume_preflight = preflights.last().unwrap();
        assert_eq!(resume_preflight.task_id, resumed.id);
        assert_eq!(resume_preflight.execution_epoch, resumed.execution_epoch);
        assert_eq!(resume_preflight.turn_kind, GoalTurnKind::Resume);
    }

    #[tokio::test]
    async fn status_and_resume_reject_an_exhausted_budget_before_launch() {
        let (runtime, store, executor, directory) = controller(ready(), false);
        runtime.submit(ingress(), start_command()).await.unwrap();
        let task = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        let tracker = CostTracker::new(CostConfig::default(), directory.path()).unwrap();
        record_goal_usage(&tracker, task.id.clone(), 8);
        assert_eq!(
            store
                .pause_session_goal(
                    &task.id,
                    "session",
                    task.execution_epoch,
                    GoalPauseState {
                        reason: GoalPauseReason::OperatorPaused,
                        description: None,
                        blockers: Vec::new(),
                    },
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        let projection = runtime
            .submit(ingress(), GoalCommand::Status)
            .await
            .unwrap()
            .status
            .unwrap();
        assert!(!projection.resumable);
        assert_eq!(projection.remaining_tokens, Some(0));
        assert_eq!(
            runtime
                .submit(ingress(), GoalCommand::Resume)
                .await
                .unwrap()
                .key,
            GoalMessageKey::BudgetExhausted
        );
        assert_eq!(executor.launches.lock().len(), 1);
    }

    #[tokio::test]
    async fn a_new_cost_limit_rejects_historical_unpriced_usage() {
        let (runtime, store, executor, directory) = controller(ready(), false);
        let token_only = GoalCommand::Start {
            budget: GoalBudgetRequest::Limits {
                token_limit: Some(8),
                cost_limit_usd: None,
            },
            success_criteria: "done".into(),
            objective: "objective".into(),
        };
        runtime.submit(ingress(), token_only).await.unwrap();
        let task = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        let tracker = CostTracker::new(CostConfig::default(), directory.path()).unwrap();
        record_unpriced_goal_usage(&tracker, task.id.clone());
        assert_eq!(
            store
                .pause_session_goal(
                    &task.id,
                    "session",
                    task.execution_epoch,
                    GoalPauseState {
                        reason: GoalPauseReason::OperatorPaused,
                        description: None,
                        blockers: Vec::new(),
                    },
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            runtime
                .submit(
                    ingress(),
                    GoalCommand::SetBudget(GoalBudgetRequest::Limits {
                        token_limit: None,
                        cost_limit_usd: Some(1.0),
                    }),
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::BudgetUpdated
        );
        assert!(
            !runtime
                .submit(ingress(), GoalCommand::Status)
                .await
                .unwrap()
                .status
                .unwrap()
                .resumable
        );
        assert_eq!(
            runtime
                .submit(ingress(), GoalCommand::Resume)
                .await
                .unwrap()
                .key,
            GoalMessageKey::AccountingIncomplete
        );
        assert_eq!(executor.launches.lock().len(), 1);
    }

    #[tokio::test]
    async fn resume_requires_conservative_reservation_headroom() {
        let (runtime, store, executor, directory) = controller(ready(), false);
        runtime.submit(ingress(), start_command()).await.unwrap();
        let task = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        let tracker = CostTracker::new(CostConfig::default(), directory.path()).unwrap();
        record_goal_usage(&tracker, task.id.clone(), 6);
        assert_eq!(
            store
                .pause_session_goal(
                    &task.id,
                    "session",
                    task.execution_epoch,
                    GoalPauseState {
                        reason: GoalPauseReason::OperatorPaused,
                        description: None,
                        blockers: Vec::new(),
                    },
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        *executor.readiness.lock() = GoalExecutionReadiness::Ready(GoalOperationReservation {
            token_count: Some(3),
            cost_usd: Some(0.0),
        });
        assert_eq!(
            runtime
                .submit(ingress(), GoalCommand::Resume)
                .await
                .unwrap()
                .key,
            GoalMessageKey::ReservationExceedsBudget
        );
        assert_eq!(executor.launches.lock().len(), 1);
    }

    #[tokio::test]
    async fn resume_accepts_an_exact_conservative_reservation_fit() {
        let (runtime, store, executor, directory) = controller(ready(), false);
        runtime.submit(ingress(), start_command()).await.unwrap();
        let task = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        let tracker = CostTracker::new(CostConfig::default(), directory.path()).unwrap();
        record_goal_usage(&tracker, task.id.clone(), 7);
        assert_eq!(
            store
                .pause_session_goal(
                    &task.id,
                    "session",
                    task.execution_epoch,
                    GoalPauseState {
                        reason: GoalPauseReason::OperatorPaused,
                        description: None,
                        blockers: Vec::new(),
                    },
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            runtime
                .submit(ingress(), GoalCommand::Resume)
                .await
                .unwrap()
                .key,
            GoalMessageKey::Resumed
        );
        assert_eq!(executor.launches.lock().len(), 2);
    }

    #[tokio::test]
    async fn status_and_resume_share_pricing_readiness() {
        let (runtime, store, executor, _directory) = controller(ready(), false);
        let command = GoalCommand::Start {
            budget: GoalBudgetRequest::Limits {
                token_limit: None,
                cost_limit_usd: Some(1.0),
            },
            success_criteria: "done".into(),
            objective: "objective".into(),
        };
        runtime.submit(ingress(), command).await.unwrap();
        let task = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            store
                .pause_session_goal(
                    &task.id,
                    "session",
                    task.execution_epoch,
                    GoalPauseState {
                        reason: GoalPauseReason::OperatorPaused,
                        description: None,
                        blockers: Vec::new(),
                    },
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        *executor.readiness.lock() = GoalExecutionReadiness::PricingUnavailable;
        assert!(
            !runtime
                .submit(ingress(), GoalCommand::Status)
                .await
                .unwrap()
                .status
                .unwrap()
                .resumable
        );
        assert_eq!(
            runtime
                .submit(ingress(), GoalCommand::Resume)
                .await
                .unwrap()
                .key,
            GoalMessageKey::PricingUnavailable
        );
        assert_eq!(executor.launches.lock().len(), 1);
    }

    #[tokio::test]
    async fn disabled_mode_fails_closed_except_for_local_help() {
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let directory = tempfile::tempdir().unwrap();
        let executor = Arc::new(FakeExecutor {
            readiness: Mutex::new(ready()),
            preflights: Mutex::new(Vec::new()),
            launches: Mutex::new(Vec::new()),
            fail_launch: false,
        });
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        let runtime = GoalRuntime::new(
            Arc::new(Config::default()),
            goals,
            Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap()),
            executor.clone(),
            "boot",
        )
        .unwrap();

        assert_eq!(
            runtime
                .submit(ingress(), GoalCommand::Status)
                .await
                .unwrap()
                .key,
            GoalMessageKey::Disabled
        );
        assert_eq!(
            runtime
                .submit(ingress(), GoalCommand::Help)
                .await
                .unwrap()
                .key,
            GoalMessageKey::Help
        );
        assert!(
            store
                .current_goal_for_session("session")
                .await
                .unwrap()
                .is_none()
        );
        assert!(executor.launches.lock().is_empty());
    }

    #[tokio::test]
    async fn launch_failure_becomes_a_sanitized_terminal_state() {
        let (runtime, store, _executor, _directory) = controller(ready(), true);
        assert_eq!(
            runtime
                .submit(ingress(), start_command())
                .await
                .unwrap()
                .key,
            GoalMessageKey::ExecutorStartFailed
        );
        let current = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(current.status, TaskStatus::Failed);
        let status = runtime
            .submit(ingress(), GoalCommand::Status)
            .await
            .unwrap()
            .status
            .unwrap();
        assert_eq!(
            status.terminal_error.as_deref(),
            Some("executor_start_failed")
        );
    }

    #[tokio::test]
    async fn a_different_trusted_binding_cannot_select_or_mutate_the_goal() {
        let (runtime, store, executor, _directory) = controller(ready(), false);
        runtime.submit(ingress(), start_command()).await.unwrap();
        let other =
            GoalIngressContext::trusted("session", "other", "room", Some("principal".into()))
                .unwrap();
        assert_eq!(
            runtime
                .submit(other, GoalCommand::Status)
                .await
                .unwrap()
                .key,
            GoalMessageKey::NoCurrentGoal
        );
        assert_eq!(executor.launches.lock().len(), 1);
        assert_eq!(
            store
                .current_goal_for_session("session")
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Running
        );
    }

    #[tokio::test]
    async fn status_projects_only_the_current_goal_and_sanitizes_unknown_failures() {
        let (runtime, store, _executor, _directory) = controller(ready(), false);
        runtime.submit(ingress(), start_command()).await.unwrap();
        let task = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            store
                .finish_session_goal(
                    &task.id,
                    "session",
                    task.execution_epoch,
                    TaskStatus::Failed,
                    Some("private provider response".into()),
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        let projection = runtime
            .submit(ingress(), GoalCommand::Status)
            .await
            .unwrap()
            .status
            .unwrap();
        assert_eq!(projection.task_id, task.id);
        assert_eq!(projection.status, TaskStatus::Failed);
        assert_eq!(
            projection.terminal_error.as_deref(),
            Some("goal_terminal_failure")
        );
    }

    #[tokio::test]
    async fn unresolved_tool_phase_is_reported_and_cannot_resume() {
        let (runtime, store, executor, _directory) = controller(ready(), false);
        runtime.submit(ingress(), start_command()).await.unwrap();
        let task = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            store
                .begin_goal_tool_phase(&task.id, "session", task.execution_epoch)
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            store
                .pause_session_goal(
                    &task.id,
                    "session",
                    task.execution_epoch,
                    GoalPauseState {
                        reason: GoalPauseReason::OperatorPaused,
                        description: None,
                        blockers: Vec::new(),
                    },
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        let projection = runtime
            .submit(ingress(), GoalCommand::Status)
            .await
            .unwrap()
            .status
            .unwrap();
        assert!(!projection.resumable);
        assert_eq!(projection.tool_phase, GoalToolPhase::InFlight);
        assert_eq!(
            runtime
                .submit(ingress(), GoalCommand::Resume)
                .await
                .unwrap()
                .key,
            GoalMessageKey::ToolPhaseNonResumable
        );
        assert_eq!(executor.launches.lock().len(), 1);
    }

    #[tokio::test]
    async fn pricing_readiness_blocks_only_cost_limited_admission() {
        let (runtime, store, executor, _directory) =
            controller(GoalExecutionReadiness::PricingUnavailable, false);
        let token_only = GoalCommand::Start {
            budget: GoalBudgetRequest::Limits {
                token_limit: Some(8),
                cost_limit_usd: None,
            },
            success_criteria: "done".into(),
            objective: "objective".into(),
        };
        assert_eq!(
            runtime.submit(ingress(), token_only).await.unwrap().key,
            GoalMessageKey::Started
        );
        assert_eq!(executor.launches.lock().len(), 1);
        let task = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            store
                .finish_session_goal(
                    &task.id,
                    "session",
                    task.execution_epoch,
                    TaskStatus::Cancelled,
                    None,
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        let cost_limited = GoalCommand::Start {
            budget: GoalBudgetRequest::Limits {
                token_limit: None,
                cost_limit_usd: Some(1.0),
            },
            success_criteria: "done".into(),
            objective: "objective".into(),
        };
        assert_eq!(
            runtime.submit(ingress(), cost_limited).await.unwrap().key,
            GoalMessageKey::PricingUnavailable
        );
        assert_eq!(executor.launches.lock().len(), 1);
    }

    #[tokio::test]
    async fn typed_invalid_start_and_budget_values_fail_without_state_change() {
        let (runtime, store, executor, _directory) = controller(ready(), false);
        for command in [
            GoalCommand::Start {
                budget: GoalBudgetRequest::Limits {
                    token_limit: None,
                    cost_limit_usd: None,
                },
                success_criteria: "done".into(),
                objective: "objective".into(),
            },
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: " ".into(),
                objective: "objective".into(),
            },
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "done".into(),
                objective: " ".into(),
            },
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "done".into(),
                objective: "x".repeat(MAX_GOAL_DECLARATION_CHARS + 1),
            },
            GoalCommand::Start {
                budget: GoalBudgetRequest::Defaults,
                success_criteria: "x".repeat(MAX_GOAL_DECLARATION_CHARS + 1),
                objective: "objective".into(),
            },
        ] {
            assert!(matches!(
                runtime.submit(ingress(), command).await.unwrap().key,
                GoalMessageKey::InvalidBudget
                    | GoalMessageKey::SuccessCriteriaRequired
                    | GoalMessageKey::InvalidCommand
                    | GoalMessageKey::InputTooLong
            ));
        }
        assert_eq!(
            runtime
                .submit(
                    ingress(),
                    GoalCommand::SetBudget(GoalBudgetRequest::Defaults)
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::InvalidBudget
        );
        assert!(
            store
                .current_goal_for_session("session")
                .await
                .unwrap()
                .is_none()
        );
        assert!(executor.launches.lock().is_empty());
    }

    #[tokio::test]
    async fn invalid_enabled_configuration_fails_closed_before_any_launch() {
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let directory = tempfile::tempdir().unwrap();
        let executor = Arc::new(FakeExecutor {
            readiness: Mutex::new(ready()),
            preflights: Mutex::new(Vec::new()),
            launches: Mutex::new(Vec::new()),
            fail_launch: false,
        });
        let mut config = Config::default();
        config.goal.enabled = true;
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        let runtime = GoalRuntime::new(
            Arc::new(config),
            goals,
            Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap()),
            executor.clone(),
            "boot",
        )
        .unwrap();

        assert_eq!(
            runtime
                .submit(ingress(), start_command())
                .await
                .unwrap()
                .key,
            GoalMessageKey::InvalidConfiguration
        );
        assert!(
            store
                .current_goal_for_session("session")
                .await
                .unwrap()
                .is_none()
        );
        assert!(executor.launches.lock().is_empty());
    }

    #[tokio::test]
    async fn cancel_is_idempotent_and_terminal_states_do_not_resume_or_mutate() {
        let (runtime, store, executor, _directory) = controller(ready(), false);
        runtime.submit(ingress(), start_command()).await.unwrap();
        assert_eq!(
            runtime
                .submit(ingress(), GoalCommand::Cancel)
                .await
                .unwrap()
                .key,
            GoalMessageKey::Cancelled
        );
        assert_eq!(
            runtime
                .submit(ingress(), GoalCommand::Cancel)
                .await
                .unwrap()
                .key,
            GoalMessageKey::AlreadyCancelled
        );
        assert_eq!(
            runtime
                .submit(ingress(), GoalCommand::Resume)
                .await
                .unwrap()
                .key,
            GoalMessageKey::Terminal
        );
        assert_eq!(
            runtime
                .submit(
                    ingress(),
                    GoalCommand::SetBudget(GoalBudgetRequest::Unlimited)
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::Terminal
        );
        assert_eq!(executor.launches.lock().len(), 1);
        assert_eq!(
            store
                .current_goal_for_session("session")
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Cancelled
        );
    }

    #[tokio::test]
    async fn stale_executor_launch_is_discarded_after_an_epoch_is_fenced() {
        let (runtime, store, executor, _directory) = controller(ready(), false);
        runtime.submit(ingress(), start_command()).await.unwrap();
        let preflight = executor.preflights.lock()[0].clone();
        let task = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            store
                .finish_session_goal(
                    &task.id,
                    "session",
                    task.execution_epoch,
                    TaskStatus::Cancelled,
                    None,
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            runtime
                .launch_exact(
                    ingress(),
                    &preflight.task_id,
                    preflight.execution_epoch,
                    preflight.turn_kind,
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::Stale
        );
        assert_eq!(executor.launches.lock().len(), 1);
    }
}
