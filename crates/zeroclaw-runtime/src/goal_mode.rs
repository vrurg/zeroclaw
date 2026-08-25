//! Transport-neutral, dormant Goal Mode command contract.
//!
//! This module deliberately does not wire a channel, RPC method, provider, or
//! executor. Those boundaries are introduced only by later Goal Mode stages.

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

mod execution;
mod host;

pub use execution::GoalExecutionHost;
pub use host::{GoalHostModelBinding, GoalSessionAdapter, GoalSessionBinding, GoalSessionLease};

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
    Defaults,
    Limits {
        token_limit: Option<u64>,
        cost_limit_usd: Option<f64>,
    },
    Unlimited,
}

/// Stable parser failure keys; rendering is provided by a later Fluent-bound
/// transport stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalCommandErrorKey {
    InvalidGrammar,
    ObjectiveRequired,
    SuccessCriteriaRequired,
    InvalidBudget,
    InvalidArguments,
}

/// Typed parser error that keeps stable logic keys separate from presentation.
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
    /// Construct from facts already derived by a trusted server-side ingress.
    ///
    /// Surface adapters decide whether their identity is durable. For example,
    /// an adapter may supply a durable principal or `None`; the controller does
    /// not identify or branch on the transport that supplied those facts.
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("goal_ingress_missing_binding")
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
/// only when they are wired in a later stage.
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

/// Current Goal status derived from the canonical task rows and usage ledger.
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
#[derive(Clone)]
pub struct GoalExecutionScope {
    pub task_id: String,
    pub session_id: String,
    pub binding: GoalSessionBinding,
    pub execution_epoch: i64,
    pub turn_kind: GoalTurnKind,
    pub cancellation: CancellationToken,
    /// One permit shared by every Goal-owned logical provider operation for
    /// this exact executor scope. Child propagation retains this same permit
    /// in GM-05; session ownership is supplied by the generic host in GM-06.
    pub operation_permit: Arc<Semaphore>,
    operation_accounting: Option<Arc<dyn crate::agent::cost::GoalOperationAccounting>>,
}

impl std::fmt::Debug for GoalExecutionScope {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GoalExecutionScope")
            .field("task_id", &self.task_id)
            .field("session_id", &self.session_id)
            .field("execution_epoch", &self.execution_epoch)
            .field("turn_kind", &self.turn_kind)
            .finish_non_exhaustive()
    }
}

impl GoalExecutionScope {
    fn with_operation_accounting(
        mut self,
        accounting: Arc<dyn crate::agent::cost::GoalOperationAccounting>,
    ) -> Self {
        self.operation_accounting = Some(accounting);
        self
    }
}

#[derive(Debug, Clone)]
pub struct GoalExecutionPreflight {
    pub binding: GoalSessionBinding,
    pub token_limit: Option<u64>,
    pub cost_limit_usd: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalExecutionReadiness {
    Ready,
    PricingUnavailable,
    VerifierUnavailable,
}

/// Dormant execution seam. This stage performs no provider work.
#[async_trait]
pub trait GoalExecutor: Send + Sync {
    async fn preflight(&self, request: &GoalExecutionPreflight) -> GoalExecutionReadiness;
    async fn launch(&self, scope: GoalExecutionScope) -> Result<(), ()>;

    /// Quiesce an exact fenced executor after its durable state transition.
    /// Dormant/fake executors need no work.
    async fn cancel(&self, _task_id: &str, _session_id: &str, _epoch: i64) -> Result<(), ()> {
        Ok(())
    }
}

struct CurrentGoal {
    task: TaskRecord,
    goal: GoalTaskRecord,
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
        match self
            .preflight(&ingress, token_limit, cost_limit_usd)
            .await?
        {
            GoalExecutionReadiness::Ready => {}
            GoalExecutionReadiness::PricingUnavailable if cost_limit_usd.is_some() => {
                return Ok(GoalResponse::message(GoalMessageKey::PricingUnavailable));
            }
            GoalExecutionReadiness::PricingUnavailable => {}
            GoalExecutionReadiness::VerifierUnavailable => {
                return Ok(GoalResponse::message(GoalMessageKey::VerifierUnavailable));
            }
        }
        let id = Uuid::new_v4().to_string();
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
        let task_id = current.task.id.clone();
        let tracker = Arc::clone(&self.cost_tracker);
        let totals =
            tokio::task::spawn_blocking(move || tracker.get_strict_usage_totals_for_task(&task_id))
                .await
                .map_err(GoalRuntimeError::accounting)?
                .map_err(GoalRuntimeError::accounting)?;
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
                current,
                (
                    totals.total_tokens,
                    totals.cost_usd,
                    totals.pricing_available,
                ),
                terminal_reason,
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
        let pause = GoalPauseState {
            reason: GoalPauseReason::OperatorPaused,
            description: None,
            blockers: Vec::new(),
        };
        let result = self
            .goals
            .pause_session_goal(
                &current.task.id,
                ingress.session_id(),
                current.task.execution_epoch,
                pause,
            )
            .await
            .map_err(GoalRuntimeError::control_plane)?;
        if result == GoalTransitionResult::Applied {
            let _ = self
                .executor
                .cancel(
                    &current.task.id,
                    ingress.session_id(),
                    current.task.execution_epoch,
                )
                .await;
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
        if current.task.status != TaskStatus::Paused {
            return Ok(GoalResponse::message(
                if current.task.status.is_terminal() {
                    GoalMessageKey::Terminal
                } else {
                    GoalMessageKey::Stale
                },
            ));
        }
        if current.goal.tool_phase != GoalToolPhase::Clean {
            return Ok(GoalResponse::message(GoalMessageKey::ToolPhaseNonResumable));
        }
        match self
            .preflight(
                &ingress,
                current.goal.effective_token_limit,
                current.goal.effective_cost_limit_usd,
            )
            .await?
        {
            GoalExecutionReadiness::Ready => {}
            GoalExecutionReadiness::PricingUnavailable
                if current.goal.effective_cost_limit_usd.is_some() =>
            {
                return Ok(GoalResponse::message(GoalMessageKey::PricingUnavailable));
            }
            GoalExecutionReadiness::PricingUnavailable => {}
            GoalExecutionReadiness::VerifierUnavailable => {
                return Ok(GoalResponse::message(GoalMessageKey::VerifierUnavailable));
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
        if result == GoalTransitionResult::Applied {
            let _ = self
                .executor
                .cancel(
                    &current.task.id,
                    ingress.session_id(),
                    current.task.execution_epoch,
                )
                .await;
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
            binding: GoalSessionBinding::new(
                ingress.session_id.clone(),
                ingress.agent.clone(),
                ingress.route.clone(),
                ingress.principal_id.clone(),
            )
            .map_err(GoalRuntimeError::control_plane)?,
            execution_epoch: epoch,
            turn_kind,
            cancellation: CancellationToken::new(),
            operation_permit: Arc::new(Semaphore::new(1)),
            operation_accounting: None,
        };
        let accounting: Arc<dyn crate::agent::cost::GoalOperationAccounting> =
            Arc::new(execution::GoalOperationAccountant::new(
                Arc::clone(&self.goals),
                Arc::clone(&self.config),
                Arc::clone(&self.cost_tracker),
                scope.clone(),
                current.task.agent.clone(),
            ));
        let scope = scope.with_operation_accounting(accounting);
        let failure_scope = scope.clone();
        if self.executor.launch(scope).await.is_ok() {
            return Ok(GoalResponse::message(if turn_kind == GoalTurnKind::Start {
                GoalMessageKey::Started
            } else {
                GoalMessageKey::Resumed
            }));
        }
        let applied = execution::finish_goal_or_retain_execution(
            &self.goals,
            &failure_scope,
            TaskStatus::Failed,
            Some("executor_start_failed"),
        )
        .await;
        Ok(GoalResponse::message(if applied {
            GoalMessageKey::ExecutorStartFailed
        } else {
            GoalMessageKey::Stale
        }))
    }

    async fn preflight(
        &self,
        ingress: &GoalIngressContext,
        token_limit: Option<u64>,
        cost_limit_usd: Option<f64>,
    ) -> Result<GoalExecutionReadiness, GoalRuntimeError> {
        let binding = GoalSessionBinding::new(
            ingress.session_id.clone(),
            ingress.agent.clone(),
            ingress.route.clone(),
            ingress.principal_id.clone(),
        )
        .map_err(GoalRuntimeError::control_plane)?;
        Ok(self
            .executor
            .preflight(&GoalExecutionPreflight {
                binding,
                token_limit,
                cost_limit_usd,
            })
            .await)
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
fn status_projection(
    current: CurrentGoal,
    totals: (u64, f64, bool),
    terminal_error: Option<String>,
) -> GoalStatusProjection {
    let settled = current.goal.accounting_state == GoalAccountingState::Complete
        && current.goal.pending_call_id.is_none()
        && current.goal.pending_call_epoch.is_none();
    let token_complete = settled;
    let cost_complete = settled && totals.2;
    GoalStatusProjection {
        task_id: current.task.id,
        status: current.task.status,
        resumable: current.task.status == TaskStatus::Paused
            && current.task.execution_epoch < i64::MAX
            && settled
            && current.goal.tool_phase == GoalToolPhase::Clean,
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
        pause_reason: current.goal.pause_reason,
        pause_description: current.goal.pause_description,
        blockers: current.goal.blockers,
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
    use zeroclaw_config::schema::{Config, CostConfig};

    use crate::control_plane::SqliteTaskStore;

    struct FakeExecutor {
        readiness: GoalExecutionReadiness,
        launches: Mutex<Vec<GoalExecutionScope>>,
    }

    #[async_trait]
    impl GoalExecutor for FakeExecutor {
        async fn preflight(&self, _: &GoalExecutionPreflight) -> GoalExecutionReadiness {
            self.readiness
        }

        async fn launch(&self, scope: GoalExecutionScope) -> Result<(), ()> {
            self.launches.lock().push(scope);
            Ok(())
        }
    }

    struct FailingExecutor;

    #[async_trait]
    impl GoalExecutor for FailingExecutor {
        async fn preflight(&self, _: &GoalExecutionPreflight) -> GoalExecutionReadiness {
            GoalExecutionReadiness::Ready
        }

        async fn launch(&self, _: GoalExecutionScope) -> Result<(), ()> {
            Err(())
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
    ) -> (
        GoalRuntime,
        Arc<SqliteTaskStore>,
        Arc<FakeExecutor>,
        tempfile::TempDir,
    ) {
        controller_with_config(enabled_config(), readiness)
    }

    fn controller_with_config(
        config: Arc<Config>,
        readiness: GoalExecutionReadiness,
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
            readiness,
            launches: Mutex::new(Vec::new()),
        });
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        (
            GoalRuntime::new(config, goals, tracker, executor.clone(), "boot").unwrap(),
            store,
            executor,
            directory,
        )
    }

    fn ingress() -> GoalIngressContext {
        GoalIngressContext::trusted("session", "main", "room", Some("principal".into())).unwrap()
    }

    #[test]
    fn parses_exact_start_grammar_and_preserves_operator_text() {
        assert_eq!(
            parse_goal_command(" /goal start --tokens 8 --cost-usd 1.5 --success-criteria  done safely  --  ship it  ").unwrap(),
            GoalCommand::Start { budget: GoalBudgetRequest::Limits { token_limit: Some(8), cost_limit_usd: Some(1.5) }, success_criteria: "  done safely  ".into(), objective: "  ship it  ".into() }
        );
    }

    #[tokio::test]
    async fn launched_scope_settles_one_task_attributed_provider_operation() {
        let (runtime, store, executor, _directory) = controller(GoalExecutionReadiness::Ready);
        let response = runtime
            .submit(
                ingress(),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Unlimited,
                    success_criteria: "deliver the result".into(),
                    objective: "complete the scoped operation".into(),
                },
            )
            .await
            .unwrap();
        assert_eq!(response.key, GoalMessageKey::Started);
        let scope = executor.launches.lock().pop().unwrap();
        let accountant = scope.operation_accounting.clone().unwrap();
        crate::agent::cost::GoalOperationAccounting::admit(
            accountant.as_ref(),
            crate::agent::cost::GoalOperationRequest {
                model_provider: "openai".into(),
                model: "verifier".into(),
                temperature: None,
                thinking: None,
                messages: vec![zeroclaw_api::model_provider::ChatMessage::user("prompt")],
                tools: Vec::new(),
            },
        )
        .await
        .unwrap();
        crate::agent::cost::GoalOperationAccounting::settle(
            accountant.as_ref(),
            crate::agent::cost::GoalOperationSettlement {
                accounting_state: GoalAccountingState::Complete,
                pricing_complete: true,
                usage_events: vec![crate::agent::cost::GoalUsageEvent {
                    model_provider: "openai".into(),
                    model: "verifier".into(),
                    usage: zeroclaw_config::cost::TokenUsage::new(
                        "verifier", 3, 2, 0, 0.0, 0.0, 0.0,
                    ),
                }],
            },
        )
        .await
        .unwrap();

        let current = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        let goal = store.get_goal_task(&current.id).await.unwrap().unwrap();
        assert_eq!(goal.accounting_state, GoalAccountingState::Complete);
        assert!(goal.pending_call_id.is_none());
        assert!(goal.pending_call_epoch.is_none());
        let status = runtime
            .submit(ingress(), GoalCommand::Status)
            .await
            .unwrap()
            .status
            .unwrap();
        assert_eq!(status.known_token_usage, 5);
    }

    #[tokio::test]
    async fn first_budgeted_operation_is_admitted_even_when_its_reservation_overshoots() {
        let mut config = (*enabled_config()).clone();
        config.goal.default_token_limit = Some(1);
        let (runtime, _store, executor, _directory) =
            controller_with_config(Arc::new(config), GoalExecutionReadiness::Ready);
        let response = runtime
            .submit(
                ingress(),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    success_criteria: "deliver the result".into(),
                    objective: "complete the scoped operation".into(),
                },
            )
            .await
            .unwrap();
        assert_eq!(response.key, GoalMessageKey::Started);
        let scope = executor.launches.lock().pop().unwrap();
        let accountant = scope.operation_accounting.clone().unwrap();
        crate::agent::cost::GoalOperationAccounting::admit(
            accountant.as_ref(),
            crate::agent::cost::GoalOperationRequest {
                model_provider: "openai".into(),
                model: "verifier".into(),
                temperature: None,
                thinking: None,
                messages: vec![zeroclaw_api::model_provider::ChatMessage::user("prompt")],
                tools: Vec::new(),
            },
        )
        .await
        .expect("one operation may overshoot a finite admission limit");
        crate::agent::cost::GoalOperationAccounting::settle(
            accountant.as_ref(),
            crate::agent::cost::GoalOperationSettlement {
                accounting_state: GoalAccountingState::Complete,
                pricing_complete: true,
                usage_events: vec![crate::agent::cost::GoalUsageEvent {
                    model_provider: "openai".into(),
                    model: "verifier".into(),
                    usage: zeroclaw_config::cost::TokenUsage::new(
                        "verifier", 1, 1, 0, 0.0, 0.0, 0.0,
                    ),
                }],
            },
        )
        .await
        .unwrap();
        let second = crate::agent::cost::GoalOperationAccounting::admit(
            accountant.as_ref(),
            crate::agent::cost::GoalOperationRequest {
                model_provider: "openai".into(),
                model: "verifier".into(),
                temperature: None,
                thinking: None,
                messages: vec![zeroclaw_api::model_provider::ChatMessage::user("prompt")],
                tools: Vec::new(),
            },
        )
        .await;
        assert!(second.is_err());
    }

    #[tokio::test]
    async fn complete_settlement_without_usage_is_durably_classified_missing() {
        let (runtime, store, executor, _directory) = controller(GoalExecutionReadiness::Ready);
        runtime
            .submit(
                ingress(),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Unlimited,
                    success_criteria: "deliver the result".into(),
                    objective: "complete the scoped operation".into(),
                },
            )
            .await
            .unwrap();
        let scope = executor.launches.lock().pop().unwrap();
        let accountant = scope.operation_accounting.clone().unwrap();
        crate::agent::cost::GoalOperationAccounting::admit(
            accountant.as_ref(),
            crate::agent::cost::GoalOperationRequest {
                model_provider: "openai".into(),
                model: "verifier".into(),
                temperature: None,
                thinking: None,
                messages: vec![zeroclaw_api::model_provider::ChatMessage::user("prompt")],
                tools: Vec::new(),
            },
        )
        .await
        .unwrap();
        assert!(
            crate::agent::cost::GoalOperationAccounting::settle(
                accountant.as_ref(),
                crate::agent::cost::GoalOperationSettlement {
                    accounting_state: GoalAccountingState::Complete,
                    pricing_complete: true,
                    usage_events: Vec::new(),
                },
            )
            .await
            .is_err()
        );
        let current = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        let goal = store.get_goal_task(&current.id).await.unwrap().unwrap();
        assert_eq!(goal.accounting_state, GoalAccountingState::Missing);
        assert!(goal.pending_call_id.is_none());
    }

    #[tokio::test]
    async fn malformed_usage_is_durably_classified_invalid_without_a_ledger_row() {
        let (runtime, store, executor, _directory) = controller(GoalExecutionReadiness::Ready);
        runtime
            .submit(
                ingress(),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Unlimited,
                    success_criteria: "deliver the result".into(),
                    objective: "complete the scoped operation".into(),
                },
            )
            .await
            .unwrap();
        let scope = executor.launches.lock().pop().unwrap();
        let accountant = scope.operation_accounting.clone().unwrap();
        crate::agent::cost::GoalOperationAccounting::admit(
            accountant.as_ref(),
            crate::agent::cost::GoalOperationRequest {
                model_provider: "openai".into(),
                model: "verifier".into(),
                temperature: None,
                thinking: None,
                messages: vec![zeroclaw_api::model_provider::ChatMessage::user("prompt")],
                tools: Vec::new(),
            },
        )
        .await
        .unwrap();
        let mut usage = zeroclaw_config::cost::TokenUsage::new("verifier", 3, 2, 0, 0.0, 0.0, 0.0);
        usage.total_tokens = 0;
        assert!(
            crate::agent::cost::GoalOperationAccounting::settle(
                accountant.as_ref(),
                crate::agent::cost::GoalOperationSettlement {
                    accounting_state: GoalAccountingState::Complete,
                    pricing_complete: true,
                    usage_events: vec![crate::agent::cost::GoalUsageEvent {
                        model_provider: "openai".into(),
                        model: "verifier".into(),
                        usage,
                    }],
                },
            )
            .await
            .is_err()
        );
        let current = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        let goal = store.get_goal_task(&current.id).await.unwrap().unwrap();
        assert_eq!(goal.accounting_state, GoalAccountingState::Invalid);
        let status = runtime
            .submit(ingress(), GoalCommand::Status)
            .await
            .unwrap()
            .status
            .unwrap();
        assert_eq!(status.known_token_usage, 0);
    }

    #[test]
    fn rejects_legacy_or_ambiguous_grammar() {
        for input in [
            "/goal objective x",
            "/goal start --tokens=8 --success-criteria c -- o",
            "/goal start -- o",
            "/goal status x",
            "/goal budget set --unlimited --tokens 1",
        ] {
            assert!(parse_goal_command(input).is_err(), "{input}");
        }
    }

    #[test]
    fn accepts_every_v1_command_form() {
        for input in [
            "/goal start --success-criteria done -- objective",
            "/goal start --tokens 1 --success-criteria done -- objective",
            "/goal start --cost-usd 0.5 --success-criteria done -- objective",
            "/goal start --unlimited --success-criteria done -- objective",
            "/goal status",
            "/goal budget",
            "/goal budget set --tokens 1",
            "/goal budget set --cost-usd 0.5",
            "/goal budget set --unlimited",
            "/goal pause",
            "/goal resume",
            "/goal cancel",
            "/goal help",
        ] {
            assert!(parse_goal_command(input).is_ok(), "{input}");
        }
    }

    #[test]
    fn rejects_every_disallowed_budget_and_start_shape() {
        for input in [
            "/goal start --success-criteria done --",
            "/goal start --success-criteria -- objective",
            "/goal start --tokens 0 --success-criteria done -- objective",
            "/goal start --tokens 1 --tokens 2 --success-criteria done -- objective",
            "/goal start --unlimited --cost-usd 1 --success-criteria done -- objective",
            "/goal start --cost-usd NaN --success-criteria done -- objective",
            "/goal start --tokens=1 --success-criteria done -- objective",
            "/goal budget set",
            "/goal budget set --tokens 0",
            "/goal budget set --unlimited --tokens 1",
            "/goal budget set --cost-usd=1",
            "/goal pause extra",
        ] {
            assert!(parse_goal_command(input).is_err(), "{input}");
        }
    }

    #[test]
    fn trusted_ingress_rejects_blank_bindings_and_allows_no_durable_principal() {
        assert!(
            GoalIngressContext::trusted("session", "agent", "route", Some(" ".into())).is_err()
        );
        let ingress = GoalIngressContext::trusted("session", "agent", "route", None).unwrap();
        assert_eq!(ingress.durable_principal_id(), None);
    }

    #[tokio::test]
    async fn controller_replaces_terminal_goal_and_keeps_verifier_failure_distinct() {
        let (runtime, store, executor, _directory) = controller(GoalExecutionReadiness::Ready);
        let start = GoalCommand::Start {
            budget: GoalBudgetRequest::Defaults,
            success_criteria: "done".into(),
            objective: "first".into(),
        };
        assert_eq!(
            runtime.submit(ingress(), start.clone()).await.unwrap().key,
            GoalMessageKey::Started
        );
        let first = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            store
                .finish_session_goal(&first.id, "session", 1, TaskStatus::Completed, None)
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            runtime
                .submit(
                    ingress(),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Defaults,
                        success_criteria: "done".into(),
                        objective: "second".into(),
                    }
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::Started
        );
        assert_eq!(executor.launches.lock().len(), 2);

        let (runtime, _store, _executor, _directory) =
            controller(GoalExecutionReadiness::VerifierUnavailable);
        assert_eq!(
            runtime
                .submit(
                    ingress(),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Defaults,
                        success_criteria: "done".into(),
                        objective: "blocked".into()
                    }
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::VerifierUnavailable
        );
    }

    #[tokio::test]
    async fn pricing_unavailability_only_blocks_cost_limited_admission() {
        let (runtime, _store, executor, _directory) =
            controller(GoalExecutionReadiness::PricingUnavailable);
        assert_eq!(
            runtime
                .submit(
                    ingress(),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Unlimited,
                        success_criteria: "done".into(),
                        objective: "tokenless".into(),
                    }
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::Started
        );
        assert_eq!(executor.launches.lock().len(), 1);

        let (runtime, _store, _executor, _directory) =
            controller(GoalExecutionReadiness::PricingUnavailable);
        assert_eq!(
            runtime
                .submit(
                    ingress(),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Limits {
                            token_limit: None,
                            cost_limit_usd: Some(1.0)
                        },
                        success_criteria: "done".into(),
                        objective: "priced".into(),
                    }
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::PricingUnavailable
        );
    }

    #[tokio::test]
    async fn disabled_mode_fails_closed_except_for_local_help() {
        let (mut config, store, executor, directory) = {
            let mut config = Config::default();
            config.goal.default_token_limit = Some(8);
            let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
            let directory = tempfile::tempdir().unwrap();
            let executor = Arc::new(FakeExecutor {
                readiness: GoalExecutionReadiness::Ready,
                launches: Mutex::new(Vec::new()),
            });
            (config, store, executor, directory)
        };
        config.goal.enabled = false;
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
    async fn state_commands_preserve_paused_state_and_require_explicit_resume() {
        let (runtime, store, executor, _directory) = controller(GoalExecutionReadiness::Ready);
        let start = GoalCommand::Start {
            budget: GoalBudgetRequest::Defaults,
            success_criteria: "done".into(),
            objective: "objective".into(),
        };
        assert_eq!(
            runtime.submit(ingress(), start).await.unwrap().key,
            GoalMessageKey::Started
        );
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
                .submit(ingress(), GoalCommand::Pause)
                .await
                .unwrap()
                .key,
            GoalMessageKey::AlreadyPaused
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
    }

    #[tokio::test]
    async fn typed_unlimited_start_is_the_only_unlimited_admission() {
        let (runtime, store, _executor, _directory) = controller(GoalExecutionReadiness::Ready);
        assert_eq!(
            runtime
                .submit(
                    ingress(),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Unlimited,
                        success_criteria: "done".into(),
                        objective: "objective".into(),
                    },
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::Started
        );
        let goal = store
            .get_goal_task(
                &store
                    .current_goal_for_session("session")
                    .await
                    .unwrap()
                    .unwrap()
                    .id,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(goal.effective_token_limit, None);
        assert_eq!(goal.effective_cost_limit_usd, None);

        let (mut config, store, executor, directory) = {
            let config = Config::default();
            let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
            let directory = tempfile::tempdir().unwrap();
            let executor = Arc::new(FakeExecutor {
                readiness: GoalExecutionReadiness::Ready,
                launches: Mutex::new(Vec::new()),
            });
            (config, store, executor, directory)
        };
        config.goal.enabled = true;
        config.goal.verifier.provider = "openai.verifier".into();
        let _ = config.providers.models.ensure("openai", "verifier");
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        let runtime = GoalRuntime::new(
            Arc::new(config),
            goals,
            Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap()),
            executor,
            "boot",
        )
        .unwrap();
        assert_eq!(
            runtime
                .submit(
                    ingress(),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Defaults,
                        success_criteria: "done".into(),
                        objective: "objective".into(),
                    },
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::BudgetRequired
        );
    }

    #[tokio::test]
    async fn typed_start_rejects_blank_input_and_implicit_unlimited_limits() {
        let (runtime, store, _executor, _directory) = controller(GoalExecutionReadiness::Ready);
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
        ] {
            let response = runtime.submit(ingress(), command).await.unwrap();
            assert!(
                matches!(
                    response.key,
                    GoalMessageKey::InvalidBudget
                        | GoalMessageKey::SuccessCriteriaRequired
                        | GoalMessageKey::InvalidCommand
                ),
                "{response:?}"
            );
        }
        assert!(
            store
                .current_goal_for_session("session")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn unfinished_tool_or_accounting_state_is_never_resumable() {
        let (runtime, store, executor, _directory) = controller(GoalExecutionReadiness::Ready);
        runtime
            .submit(
                ingress(),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    success_criteria: "done".into(),
                    objective: "objective".into(),
                },
            )
            .await
            .unwrap();
        let current = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            store
                .begin_goal_tool_phase(&current.id, "session", current.execution_epoch)
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            store
                .pause_session_goal(
                    &current.id,
                    "session",
                    current.execution_epoch,
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
        let response = runtime
            .submit(ingress(), GoalCommand::Status)
            .await
            .unwrap();
        assert!(!response.status.unwrap().resumable);
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
    async fn pending_accounting_is_non_resumable_and_prevents_a_second_launch() {
        let (runtime, store, executor, _directory) = controller(GoalExecutionReadiness::Ready);
        runtime
            .submit(
                ingress(),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    success_criteria: "done".into(),
                    objective: "objective".into(),
                },
            )
            .await
            .unwrap();
        let current = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            store
                .admit_pending_call_with_reservation(
                    &current.id,
                    "session",
                    current.execution_epoch,
                    "pending",
                    Some(1),
                    None,
                )
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            store
                .pause_session_goal(
                    &current.id,
                    "session",
                    current.execution_epoch,
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
        let status = runtime
            .submit(ingress(), GoalCommand::Status)
            .await
            .unwrap()
            .status
            .unwrap();
        assert!(!status.resumable);
        assert!(!status.token_accounting_complete);
        assert!(!status.cost_accounting_complete);
        assert_eq!(
            runtime
                .submit(ingress(), GoalCommand::Resume)
                .await
                .unwrap()
                .key,
            GoalMessageKey::Stale
        );
        assert_eq!(executor.launches.lock().len(), 1);
    }

    #[tokio::test]
    async fn typed_budget_and_terminal_command_matrix_fails_closed() {
        let (runtime, store, _executor, _directory) = controller(GoalExecutionReadiness::Ready);
        for command in [
            GoalCommand::SetBudget(GoalBudgetRequest::Defaults),
            GoalCommand::SetBudget(GoalBudgetRequest::Limits {
                token_limit: Some(0),
                cost_limit_usd: None,
            }),
            GoalCommand::SetBudget(GoalBudgetRequest::Limits {
                token_limit: Some(i64::MAX as u64 + 1),
                cost_limit_usd: None,
            }),
            GoalCommand::SetBudget(GoalBudgetRequest::Limits {
                token_limit: None,
                cost_limit_usd: Some(f64::NAN),
            }),
            GoalCommand::SetBudget(GoalBudgetRequest::Limits {
                token_limit: None,
                cost_limit_usd: Some(-1.0),
            }),
        ] {
            assert_eq!(
                runtime.submit(ingress(), command).await.unwrap().key,
                GoalMessageKey::InvalidBudget
            );
        }
        for command in [
            GoalCommand::Pause,
            GoalCommand::Resume,
            GoalCommand::Cancel,
            GoalCommand::SetBudget(GoalBudgetRequest::Unlimited),
        ] {
            assert_eq!(
                runtime.submit(ingress(), command).await.unwrap().key,
                GoalMessageKey::NoCurrentGoal
            );
        }
        runtime
            .submit(
                ingress(),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    success_criteria: "done".into(),
                    objective: "objective".into(),
                },
            )
            .await
            .unwrap();
        let task = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        store
            .finish_session_goal(&task.id, "session", 1, TaskStatus::Completed, None)
            .await
            .unwrap();
        for command in [
            GoalCommand::Pause,
            GoalCommand::Resume,
            GoalCommand::Cancel,
            GoalCommand::SetBudget(GoalBudgetRequest::Unlimited),
        ] {
            assert_eq!(
                runtime.submit(ingress(), command).await.unwrap().key,
                GoalMessageKey::Terminal
            );
        }
        let (runtime, _store, _executor, _directory) = controller(GoalExecutionReadiness::Ready);
        runtime
            .submit(
                ingress(),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    success_criteria: "done".into(),
                    objective: "objective".into(),
                },
            )
            .await
            .unwrap();
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
    }

    #[tokio::test]
    async fn fenced_predecessor_cannot_reach_the_dormant_executor() {
        let (runtime, store, executor, _directory) = controller(GoalExecutionReadiness::Ready);
        runtime
            .submit(
                ingress(),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    success_criteria: "done".into(),
                    objective: "objective".into(),
                },
            )
            .await
            .unwrap();
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
                    &task.id,
                    task.execution_epoch,
                    GoalTurnKind::Start
                )
                .await
                .unwrap()
                .key,
            GoalMessageKey::Stale
        );
        assert_eq!(executor.launches.lock().len(), 1);
    }

    #[tokio::test]
    async fn saturated_epoch_is_never_projected_as_resumable() {
        let (runtime, store, _executor, _directory) = controller(GoalExecutionReadiness::Ready);
        runtime
            .submit(
                ingress(),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    success_criteria: "done".into(),
                    objective: "objective".into(),
                },
            )
            .await
            .unwrap();
        let mut task = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        let goal = store.get_goal_task(&task.id).await.unwrap().unwrap();
        task.status = TaskStatus::Paused;
        task.execution_epoch = i64::MAX;
        let status = status_projection(CurrentGoal { task, goal }, (0, 0.0, true), None);
        assert!(!status.resumable);
    }

    #[tokio::test]
    async fn executor_launch_failure_is_durably_terminal_and_status_is_sanitized() {
        let store = Arc::new(SqliteTaskStore::new_in_memory().unwrap());
        let directory = tempfile::tempdir().unwrap();
        let goals: Arc<dyn GoalTaskRegistry> = store.clone();
        let runtime = GoalRuntime::new(
            enabled_config(),
            goals,
            Arc::new(CostTracker::new(CostConfig::default(), directory.path()).unwrap()),
            Arc::new(FailingExecutor),
            "boot",
        )
        .unwrap();

        assert_eq!(
            runtime
                .submit(
                    ingress(),
                    GoalCommand::Start {
                        budget: GoalBudgetRequest::Defaults,
                        success_criteria: "done".into(),
                        objective: "objective".into(),
                    },
                )
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
            .unwrap();
        assert_eq!(
            status.status.unwrap().terminal_error.as_deref(),
            Some("executor_start_failed")
        );
    }

    #[tokio::test]
    async fn status_uses_only_a_sanitized_exact_terminal_reason() {
        let (runtime, store, _executor, _directory) = controller(GoalExecutionReadiness::Ready);
        runtime
            .submit(
                ingress(),
                GoalCommand::Start {
                    budget: GoalBudgetRequest::Defaults,
                    success_criteria: "done".into(),
                    objective: "objective".into(),
                },
            )
            .await
            .unwrap();
        let task = store
            .current_goal_for_session("session")
            .await
            .unwrap()
            .unwrap();
        store
            .finish_session_goal(
                &task.id,
                "session",
                1,
                TaskStatus::Failed,
                Some("provider returned private prompt text".into()),
            )
            .await
            .unwrap();
        let response = runtime
            .submit(ingress(), GoalCommand::Status)
            .await
            .unwrap();
        assert_eq!(
            response.status.unwrap().terminal_error.as_deref(),
            Some("goal_terminal_failure")
        );
    }
}
