//! Transport-neutral Goal execution boundary.
//!
//! This module deliberately does not select a transport driver from route
//! text. A trusted Matrix or ZeroCode adapter must supply the exact live
//! session driver with the typed ingress it derived before mutable hooks or
//! prompt handling. The host owns typed admission and the controller owns
//! guarded durable lifecycle transitions; later stages add execution,
//! accounting, and transport adapters behind those boundaries.

use std::sync::Arc;

use anyhow::{Result, bail};
use async_trait::async_trait;
use chrono::Utc;
use uuid::Uuid;
use zeroclaw_api::{model_provider::ChatMessage, session_keys::sanitize_session_key};
use zeroclaw_commands::goal::{GoalBudgetLimits, GoalBudgetSelection, GoalCommand};

use crate::control_plane::{
    GoalAccountingState, GoalPauseReason, GoalPauseState, GoalTaskRecord, GoalTaskRegistry,
    GoalTransitionResult, TaskKind, TaskRecord, TaskStatus,
};

/// The only V1 surfaces permitted to admit a Goal command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalSurface {
    Matrix,
    ZeroCode,
}

/// A canonical session key owned by the trusted ingress surface.
///
/// Matrix supplies its existing canonical conversation-history key. ZeroCode
/// supplies the raw RPC session identifier, which is namespaced as `rpc_` for
/// the durable task-plane binding. The variants prevent an adapter from
/// mistaking a route-shaped string for a different transport's session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoalSessionKey {
    Matrix { history_key: String },
    ZeroCode { raw_session_id: String },
}

impl GoalSessionKey {
    pub fn matrix(history_key: impl Into<String>) -> Result<Self> {
        let history_key = required("Matrix history key", history_key.into())?;
        if sanitize_session_key(&history_key) != history_key {
            bail!("Matrix Goal history key is not canonical");
        }
        Ok(Self::Matrix { history_key })
    }

    pub fn zero_code(raw_session_id: impl Into<String>) -> Result<Self> {
        Ok(Self::ZeroCode {
            raw_session_id: required("ZeroCode session id", raw_session_id.into())?,
        })
    }

    pub const fn surface(&self) -> GoalSurface {
        match self {
            Self::Matrix { .. } => GoalSurface::Matrix,
            Self::ZeroCode { .. } => GoalSurface::ZeroCode,
        }
    }

    /// Canonical task-plane session binding. This deliberately reuses the
    /// existing session namespaces rather than creating a Goal identity store.
    pub fn durable_id(&self) -> String {
        match self {
            Self::Matrix { history_key } => history_key.clone(),
            Self::ZeroCode { raw_session_id } => format!("rpc_{raw_session_id}"),
        }
    }
}

/// Immutable source identity captured by the trusted adapter. It is never a
/// hook-mutable generic channel field and `tui_id` is never durable identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoalIngressPrincipal {
    Matrix { raw_mxid: String },
    ZeroCode { tui_id: String },
}

impl GoalIngressPrincipal {
    const fn surface(&self) -> GoalSurface {
        match self {
            Self::Matrix { .. } => GoalSurface::Matrix,
            Self::ZeroCode { .. } => GoalSurface::ZeroCode,
        }
    }

    fn validate(&self) -> Result<()> {
        let value = match self {
            Self::Matrix { raw_mxid } => raw_mxid,
            Self::ZeroCode { tui_id } => tui_id,
        };
        required("Goal ingress principal", value.clone()).map(|_| ())
    }
}

/// Trusted authority supplied alongside a parsed [`GoalCommand`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalIngressContext {
    session_key: GoalSessionKey,
    agent: String,
    route: String,
    principal: GoalIngressPrincipal,
}

impl GoalIngressContext {
    pub fn trusted(
        session_key: GoalSessionKey,
        agent: impl Into<String>,
        route: impl Into<String>,
        principal: GoalIngressPrincipal,
    ) -> Result<Self> {
        if session_key.surface() != principal.surface() {
            bail!("Goal ingress principal does not match the session surface");
        }
        principal.validate()?;
        Ok(Self {
            session_key,
            agent: required("Goal ingress agent", agent.into())?,
            route: required("Goal ingress route", route.into())?,
            principal,
        })
    }

    pub fn session_key(&self) -> &GoalSessionKey {
        &self.session_key
    }

    pub const fn surface(&self) -> GoalSurface {
        self.session_key.surface()
    }

    pub fn agent(&self) -> &str {
        &self.agent
    }

    pub fn route(&self) -> &str {
        &self.route
    }

    pub fn principal(&self) -> &GoalIngressPrincipal {
        &self.principal
    }

    fn durable_principal_id(&self) -> Option<&str> {
        match &self.principal {
            GoalIngressPrincipal::Matrix { raw_mxid } => Some(raw_mxid),
            GoalIngressPrincipal::ZeroCode { .. } => None,
        }
    }
}

/// Metadata about a live session returned by an already-selected driver.
///
/// `freshness_token` is diagnostic metadata for the surface; it is not an
/// execution fence. The paired [`GoalSessionLease`] owns the only guard that
/// keeps the session authoritative through a controller transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalSessionBinding {
    session_key: GoalSessionKey,
    freshness_token: String,
}

impl GoalSessionBinding {
    pub fn new(session_key: GoalSessionKey, freshness_token: impl Into<String>) -> Result<Self> {
        Ok(Self {
            session_key,
            freshness_token: required("Goal session freshness token", freshness_token.into())?,
        })
    }

    pub fn session_key(&self) -> &GoalSessionKey {
        &self.session_key
    }

    pub const fn surface(&self) -> GoalSurface {
        self.session_key.surface()
    }

    pub fn freshness_token(&self) -> &str {
        &self.freshness_token
    }
}

/// Surface-owned live-session mechanics used by the shared Goal host.
///
/// The adapter chooses this object from trusted live-session state. The host
/// verifies its typed surface and canonical key before asking it to bind; it
/// never probes a registry using user-controlled route or session text.
#[async_trait]
pub trait GoalSessionDriver: Send + Sync {
    fn surface(&self) -> GoalSurface;
    fn session_key(&self) -> &GoalSessionKey;

    /// Revalidate the exact live session and acquire its validity lease.
    ///
    /// The returned lease must keep the driver's authoritative live-session
    /// guard held until it is dropped.  In particular, a reconnect or session
    /// replacement must not become effective while a submitted Goal command
    /// still owns this lease. The guard remains held across durable controller
    /// I/O, so it must be async-safe and must not be a lock that a Goal
    /// lifecycle path needs to acquire again.
    async fn bind(&self, ingress: &GoalIngressContext) -> Result<GoalSessionLease>;

    /// Acquire the exact session's foreground execution lease.
    ///
    /// The returned lease must retain the driver's authoritative live-session
    /// guard until it is dropped, exclusively serialize foreground work for
    /// this session, and tolerate being held across the executor's durable
    /// lifecycle and accounting I/O. It must not re-enter a lock that the
    /// Goal lifecycle needs. The executor treats `ingress` as admission-time
    /// audit data only; live policy stays with the driver and is revalidated
    /// there. `canonical_history` is read-only, parent and verifier calls
    /// mutate only process-local working state, and `append_verified_candidate`
    /// appends and delivers exactly one already-verified final candidate.
    /// The object is the sole transport bridge used by the later Goal executor.
    async fn acquire_execution(
        &self,
        ingress: &GoalIngressContext,
        scope: &GoalExecutionScope,
    ) -> Result<Box<dyn GoalSessionExecutionLease>>;
}

/// Controller-owned facts carried for one fenced Goal execution epoch.
///
/// The host validates the session binding only. The later executor performs
/// the durable task and epoch fence before it asks a driver for this lease.
#[derive(Debug, Clone)]
pub struct GoalExecutionScope {
    task_id: String,
    session_id: String,
    execution_epoch: i64,
}

impl GoalExecutionScope {
    pub fn new(
        task_id: impl Into<String>,
        session_id: impl Into<String>,
        execution_epoch: i64,
    ) -> Result<Self> {
        if execution_epoch <= 0 {
            bail!("Goal execution epoch must be positive");
        }
        Ok(Self {
            task_id: required("Goal execution task id", task_id.into())?,
            session_id: required("Goal execution session id", session_id.into())?,
            execution_epoch,
        })
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub const fn execution_epoch(&self) -> i64 {
        self.execution_epoch
    }
}

/// Trusted input for a parent Goal turn.
#[derive(Debug, Clone)]
pub struct GoalParentTurn {
    pub objective: String,
    pub working_history: Vec<ChatMessage>,
}

/// Trusted, isolated input for the mandatory Goal verifier.
#[derive(Debug, Clone)]
pub struct GoalVerifierTurn {
    pub objective: String,
    pub candidate: String,
}

/// Surface-owned foreground execution bridge. It retains the live-session
/// guard and one foreground slot for its lifetime, but has no lifecycle or
/// ledger authority; the later executor owns both around these calls.
#[async_trait]
pub trait GoalSessionExecutionLease: Send {
    fn canonical_history(&self) -> Result<Vec<ChatMessage>>;
    async fn run_parent_turn(&mut self, turn: GoalParentTurn) -> Result<String>;
    async fn run_verifier(&mut self, turn: GoalVerifierTurn) -> Result<String>;
    async fn append_verified_candidate(&mut self, candidate: String) -> Result<()>;
}

/// A driver's owned proof that a live session remains authoritative.
///
/// The guard is intentionally opaque to the Goal controller.  Its producer
/// owns the surface-specific reconnect and replacement mechanics, while the
/// host keeps it alive for the complete lifecycle transition.
pub struct GoalSessionLease {
    binding: GoalSessionBinding,
    validity_guard: Box<dyn Send>,
}

impl std::fmt::Debug for GoalSessionLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GoalSessionLease")
            .field("binding", &self.binding)
            .field(
                "validity_guard_type",
                &std::any::type_name_of_val(self.validity_guard.as_ref()),
            )
            .finish_non_exhaustive()
    }
}

impl GoalSessionLease {
    /// Construct a lease from a validated binding and surface-owned guard.
    ///
    /// `validity_guard` may be an owned lock, generation lease, or another
    /// surface-specific resource whose lifetime prevents a stale session from
    /// mutating Goal lifecycle state. The controller retains it across durable
    /// I/O, so the resource must not require a Goal lifecycle path to acquire
    /// the same lock re-entrantly.
    pub fn new<G>(binding: GoalSessionBinding, validity_guard: G) -> Self
    where
        G: Send + 'static,
    {
        Self {
            binding,
            validity_guard: Box::new(validity_guard),
        }
    }

    pub fn binding(&self) -> &GoalSessionBinding {
        &self.binding
    }
}

/// A validated Goal command whose live-session lease has not yet been settled.
///
/// This is intentionally non-cloneable: duplicating it could allow a stale
/// authority proof to outlive the session transition it protects.
#[derive(Debug)]
pub struct GoalSubmission {
    ingress: GoalIngressContext,
    command: GoalCommand,
    lease: GoalSessionLease,
}

impl GoalSubmission {
    pub fn ingress(&self) -> &GoalIngressContext {
        &self.ingress
    }

    pub fn command(&self) -> &GoalCommand {
        &self.command
    }

    pub fn binding(&self) -> &GoalSessionBinding {
        self.lease.binding()
    }
}

/// The transport-neutral authority boundary for Goal admission.
#[derive(Debug, Default)]
pub struct GoalExecutionHost;

impl GoalExecutionHost {
    pub const fn new() -> Self {
        Self
    }

    /// Validate a typed command against the exact trusted driver supplied by
    /// the caller. It does not mutate lifecycle state or start model work;
    /// [`GoalController`] consumes the returned submission for durable
    /// transitions, while execution and accounting remain later-stage work.
    pub async fn submit(
        &self,
        ingress: GoalIngressContext,
        driver: Arc<dyn GoalSessionDriver>,
        command: GoalCommand,
    ) -> Result<GoalSubmission> {
        if driver.surface() != ingress.surface() || driver.session_key() != ingress.session_key() {
            bail!("Goal session driver does not match trusted ingress");
        }

        let lease = driver.bind(&ingress).await?;
        if lease.binding().surface() != ingress.surface()
            || lease.binding().session_key() != ingress.session_key()
        {
            bail!("Goal session binding does not match trusted ingress");
        }

        Ok(GoalSubmission {
            ingress,
            command,
            lease,
        })
    }

    /// Acquire execution only through the same exact trusted driver boundary
    /// used for command admission. This never selects a driver from route text
    /// and refuses while Goal Mode is disabled.
    pub async fn acquire_execution(
        &self,
        settings: &GoalHostSettings,
        ingress: &GoalIngressContext,
        driver: Arc<dyn GoalSessionDriver>,
        scope: &GoalExecutionScope,
    ) -> Result<Box<dyn GoalSessionExecutionLease>> {
        if !settings.enabled {
            bail!("Goal Mode is disabled");
        }
        if driver.surface() != ingress.surface() {
            bail!("Goal execution driver surface does not match trusted ingress");
        }
        if driver.session_key() != ingress.session_key() {
            bail!("Goal execution driver session key does not match trusted ingress");
        }
        if scope.session_id() != ingress.session_key().durable_id() {
            bail!("Goal execution scope session does not match trusted ingress");
        }
        driver.acquire_execution(ingress, scope).await
    }
}

fn required(name: &str, value: String) -> Result<String> {
    if value.trim().is_empty() {
        bail!("{name} is blank");
    }
    Ok(value.trim().to_owned())
}

/// Process-local settings already validated from the active Goal configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct GoalHostSettings {
    pub enabled: bool,
    pub default_limits: GoalBudgetLimits,
    pub owner_pid: u32,
    pub owner_boot_id: String,
}

impl GoalHostSettings {
    pub fn new(
        enabled: bool,
        default_limits: GoalBudgetLimits,
        owner_pid: u32,
        owner_boot_id: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            enabled,
            default_limits,
            owner_pid,
            owner_boot_id: required("Goal owner boot id", owner_boot_id.into())?,
        })
    }
}

/// Controller response before a transport renders it through Fluent.
#[derive(Debug, Clone, PartialEq)]
pub enum GoalResponse {
    Help,
    Disabled,
    Started(GoalStatusProjection),
    Status(GoalStatusProjection),
    Budget(GoalStatusProjection),
    BudgetUpdated(GoalStatusProjection),
    Paused(GoalStatusProjection),
    AlreadyPaused(GoalStatusProjection),
    Resumed(GoalStatusProjection),
    Cancelled(GoalStatusProjection),
    AlreadyCancelled(GoalStatusProjection),
    NoCurrentGoal,
    AlreadyActive,
    Terminal(GoalStatusProjection),
    Stale,
}

/// Durable state visible to the future Matrix and ZeroCode renderers.
#[derive(Debug, Clone, PartialEq)]
pub struct GoalStatusProjection {
    pub task_id: String,
    pub status: TaskStatus,
    pub execution_epoch: i64,
    pub token_limit: Option<u64>,
    pub cost_limit_usd: Option<f64>,
    pub accounting_state: GoalAccountingState,
    pub pause_reason: Option<GoalPauseReason>,
    pub resumable: bool,
}

impl GoalStatusProjection {
    fn from_parts(task: &TaskRecord, goal: &GoalTaskRecord) -> Self {
        Self {
            task_id: task.id.clone(),
            status: task.status,
            execution_epoch: task.execution_epoch,
            token_limit: goal.effective_token_limit,
            cost_limit_usd: goal.effective_cost_limit_usd,
            accounting_state: goal.accounting_state,
            pause_reason: goal.pause_reason,
            resumable: task.status == TaskStatus::Paused
                && goal.accounting_state == GoalAccountingState::Complete
                && goal.pending_operation_id.is_none()
                && goal.tool_phase == crate::control_plane::GoalToolPhase::Clean,
        }
    }
}

/// Transport-neutral lifecycle controller. Its input is opaque outside this
/// module, so callers must pass through [`GoalExecutionHost`] first.
pub struct GoalController {
    registry: Arc<dyn GoalTaskRegistry>,
}

impl GoalController {
    pub fn new(registry: Arc<dyn GoalTaskRegistry>) -> Self {
        Self { registry }
    }

    pub async fn submit(
        &self,
        settings: &GoalHostSettings,
        submission: GoalSubmission,
    ) -> Result<GoalResponse> {
        let GoalSubmission {
            ingress,
            command,
            lease,
        } = submission;
        // Keep the surface-owned lease through every guarded transition below.
        // Dropping it any earlier would permit a reconnect to invalidate the
        // binding between host admission and durable mutation.
        let _lease = lease;

        if matches!(command, GoalCommand::Help) {
            return Ok(GoalResponse::Help);
        }
        if !settings.enabled {
            return Ok(GoalResponse::Disabled);
        }

        match command {
            GoalCommand::Start { budget, objective } => {
                self.start(settings, &ingress, budget, objective).await
            }
            GoalCommand::Status => self.status(&ingress, false).await,
            GoalCommand::Budget => self.status(&ingress, true).await,
            GoalCommand::SetBudget(selection) => self.set_budget(&ingress, selection).await,
            GoalCommand::Pause => self.pause(&ingress).await,
            GoalCommand::Resume => self.resume(settings, &ingress).await,
            GoalCommand::Cancel => self.cancel(&ingress).await,
            GoalCommand::Help => unreachable!("handled before controller dispatch"),
        }
    }

    async fn start(
        &self,
        settings: &GoalHostSettings,
        ingress: &GoalIngressContext,
        selection: GoalBudgetSelection,
        objective: String,
    ) -> Result<GoalResponse> {
        let session_id = ingress.session_key().durable_id();
        if let Some(current) = self.registry.current_goal_for_session(&session_id).await?
            && !current.status.is_terminal()
        {
            return Ok(GoalResponse::AlreadyActive);
        }
        let limits = select_limits(settings.default_limits, selection)?;
        let task = TaskRecord {
            id: Uuid::new_v4().to_string(),
            kind: TaskKind::Goal,
            agent: ingress.agent().to_owned(),
            status: TaskStatus::Running,
            owner_pid: settings.owner_pid,
            owner_boot_id: settings.owner_boot_id.clone(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: Some(ingress.route().to_owned()),
            delivered: false,
            idem_key: None,
            principal_id: ingress.durable_principal_id().map(ToOwned::to_owned),
            session_id: Some(session_id),
            execution_epoch: 1,
            started_at: Utc::now().to_rfc3339(),
            finished_at: None,
        };
        let goal = GoalTaskRecord {
            task_id: task.id.clone(),
            objective,
            effective_token_limit: limits.token_limit,
            effective_cost_limit_usd: limits.cost_limit_usd,
            ..GoalTaskRecord::default()
        };
        match self
            .registry
            .create_or_replace_session_goal(task.clone(), goal)
            .await?
        {
            GoalTransitionResult::Applied => self
                .projection_for(ingress, &task.id)
                .await?
                .map_or(Ok(GoalResponse::Stale), |projection| {
                    Ok(GoalResponse::Started(projection))
                }),
            GoalTransitionResult::Stale => Ok(GoalResponse::AlreadyActive),
            GoalTransitionResult::Missing => Ok(GoalResponse::Stale),
        }
    }

    async fn status(&self, ingress: &GoalIngressContext, budget: bool) -> Result<GoalResponse> {
        let Some(task) = self.current(ingress).await? else {
            return Ok(GoalResponse::NoCurrentGoal);
        };
        let projection = self.project(&task).await?;
        if task.status.is_terminal() {
            Ok(GoalResponse::Terminal(projection))
        } else if budget {
            Ok(GoalResponse::Budget(projection))
        } else {
            Ok(GoalResponse::Status(projection))
        }
    }

    async fn set_budget(
        &self,
        ingress: &GoalIngressContext,
        selection: GoalBudgetSelection,
    ) -> Result<GoalResponse> {
        let Some(task) = self.current(ingress).await? else {
            return Ok(GoalResponse::NoCurrentGoal);
        };
        if task.status.is_terminal() {
            return self.project(&task).await.map(GoalResponse::Terminal);
        }
        let limits = select_budget_update_limits(selection)?;
        match self
            .registry
            .update_session_goal_limits(
                &task.id,
                &ingress.session_key().durable_id(),
                task.execution_epoch,
                limits.token_limit,
                limits.cost_limit_usd,
            )
            .await?
        {
            GoalTransitionResult::Applied => self
                .projection_for(ingress, &task.id)
                .await?
                .map_or(Ok(GoalResponse::Stale), |projection| {
                    Ok(GoalResponse::BudgetUpdated(projection))
                }),
            GoalTransitionResult::Stale | GoalTransitionResult::Missing => Ok(GoalResponse::Stale),
        }
    }

    async fn pause(&self, ingress: &GoalIngressContext) -> Result<GoalResponse> {
        let Some(task) = self.current(ingress).await? else {
            return Ok(GoalResponse::NoCurrentGoal);
        };
        if task.status == TaskStatus::Paused {
            return self.project(&task).await.map(GoalResponse::AlreadyPaused);
        }
        if task.status.is_terminal() {
            return self.project(&task).await.map(GoalResponse::Terminal);
        }
        match self
            .registry
            .pause_session_goal(
                &task.id,
                &ingress.session_key().durable_id(),
                task.execution_epoch,
                GoalPauseState {
                    reason: GoalPauseReason::OperatorPaused,
                    description: None,
                    blockers: Vec::new(),
                },
            )
            .await?
        {
            GoalTransitionResult::Applied => self
                .projection_for(ingress, &task.id)
                .await?
                .map_or(Ok(GoalResponse::Stale), |projection| {
                    Ok(GoalResponse::Paused(projection))
                }),
            GoalTransitionResult::Stale | GoalTransitionResult::Missing => Ok(GoalResponse::Stale),
        }
    }

    async fn resume(
        &self,
        settings: &GoalHostSettings,
        ingress: &GoalIngressContext,
    ) -> Result<GoalResponse> {
        let Some(task) = self.current(ingress).await? else {
            return Ok(GoalResponse::NoCurrentGoal);
        };
        if task.status.is_terminal() {
            return self.project(&task).await.map(GoalResponse::Terminal);
        }
        if task.status != TaskStatus::Paused {
            return Ok(GoalResponse::AlreadyActive);
        }
        match self
            .registry
            .resume_session_goal(
                &task.id,
                &ingress.session_key().durable_id(),
                task.execution_epoch,
                settings.owner_pid,
                &settings.owner_boot_id,
            )
            .await?
        {
            GoalTransitionResult::Applied => self
                .projection_for(ingress, &task.id)
                .await?
                .map_or(Ok(GoalResponse::Stale), |projection| {
                    Ok(GoalResponse::Resumed(projection))
                }),
            GoalTransitionResult::Stale | GoalTransitionResult::Missing => Ok(GoalResponse::Stale),
        }
    }

    async fn cancel(&self, ingress: &GoalIngressContext) -> Result<GoalResponse> {
        let Some(task) = self.current(ingress).await? else {
            return Ok(GoalResponse::NoCurrentGoal);
        };
        if task.status == TaskStatus::Cancelled {
            return self
                .project(&task)
                .await
                .map(GoalResponse::AlreadyCancelled);
        }
        if task.status.is_terminal() {
            return self.project(&task).await.map(GoalResponse::Terminal);
        }
        match self
            .registry
            .finish_session_goal(
                &task.id,
                &ingress.session_key().durable_id(),
                task.execution_epoch,
                TaskStatus::Cancelled,
                None,
            )
            .await?
        {
            GoalTransitionResult::Applied => self
                .projection_for(ingress, &task.id)
                .await?
                .map_or(Ok(GoalResponse::Stale), |projection| {
                    Ok(GoalResponse::Cancelled(projection))
                }),
            GoalTransitionResult::Stale | GoalTransitionResult::Missing => Ok(GoalResponse::Stale),
        }
    }

    async fn current(&self, ingress: &GoalIngressContext) -> Result<Option<TaskRecord>> {
        // Session binding is the lifecycle authority. Agent, route, and
        // principal remain creation-time audit facts and policy inputs; a
        // valid session can legitimately refresh those live ingress facts.
        self.registry
            .current_goal_for_session(&ingress.session_key().durable_id())
            .await
    }

    async fn projection_for(
        &self,
        ingress: &GoalIngressContext,
        task_id: &str,
    ) -> Result<Option<GoalStatusProjection>> {
        let Some(task) = self.current(ingress).await? else {
            return Ok(None);
        };
        if task.id != task_id {
            return Ok(None);
        }
        self.project(&task).await.map(Some)
    }

    async fn project(&self, task: &TaskRecord) -> Result<GoalStatusProjection> {
        let goal = self
            .registry
            .get_goal_task(&task.id)
            .await?
            .ok_or_else(|| anyhow::Error::msg("Goal task extension is missing"))?;
        Ok(GoalStatusProjection::from_parts(task, &goal))
    }
}

fn select_limits(
    defaults: GoalBudgetLimits,
    selection: GoalBudgetSelection,
) -> Result<GoalBudgetLimits> {
    match selection {
        GoalBudgetSelection::Defaults => Ok(defaults),
        GoalBudgetSelection::Limits(limits) => Ok(limits),
        GoalBudgetSelection::Unlimited => Ok(GoalBudgetLimits {
            token_limit: None,
            cost_limit_usd: None,
        }),
    }
}

fn select_budget_update_limits(selection: GoalBudgetSelection) -> Result<GoalBudgetLimits> {
    match selection {
        GoalBudgetSelection::Defaults => bail!("Goal budget set cannot use configured defaults"),
        GoalBudgetSelection::Limits(limits) => Ok(limits),
        GoalBudgetSelection::Unlimited => Ok(GoalBudgetLimits {
            token_limit: None,
            cost_limit_usd: None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_budget_update_cannot_smuggle_configured_defaults() {
        assert!(select_budget_update_limits(GoalBudgetSelection::Defaults).is_err());
    }

    #[test]
    fn matrix_key_must_keep_the_existing_matrix_history_namespace() {
        assert!(GoalSessionKey::matrix("rpc_shared").is_err());
        assert!(GoalSessionKey::matrix("matrix_primary_room").is_ok());
    }
}
