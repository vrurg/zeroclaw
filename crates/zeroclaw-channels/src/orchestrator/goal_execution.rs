//! Matrix implementation of the transport-neutral Goal session driver.
//!
//! This module owns only live Matrix session mechanics. Durable lifecycle,
//! accounting, and execution fencing remain in `zeroclaw-runtime`.

use std::sync::Arc;

use anyhow::{Context as _, Result, bail, ensure};
use async_trait::async_trait;
use zeroclaw_api::{
    channel::ChannelMessage,
    model_provider::{ChatMessage, ChatRequest},
};
use zeroclaw_runtime::{
    agent::cost::build_type_level_model_provider_pricing,
    agent::loop_::{LoopKnobs, ToolLoop, run_tool_call_loop, scope_session_key, scope_thread_id},
    control_plane::{
        GoalPolicyRevocationTarget, GoalTransitionResult, TaskKind, TaskRecord, control_plane,
    },
    cost::CostTracker,
    goal_mode::{
        GoalExecutionNotice, GoalExecutionScope, GoalIngressContext, GoalOperationScope,
        GoalParentTurn, GoalParentTurnKind, GoalParentTurnResult, GoalSessionBinding,
        GoalSessionDriver, GoalSessionExecutionLease, GoalSessionKey, GoalSessionLease,
        GoalSurface, GoalVerifierTurn, dispose_unowned_session_goal,
    },
};

const DEFAULT_MATRIX_ALIAS: &str = "default";

/// Canonical Matrix route stored with a Goal. The room id is intentionally
/// split only once: Matrix room ids themselves contain a colon.
#[derive(Debug, Clone, PartialEq, Eq)]
struct MatrixGoalRoute {
    alias: String,
    room_id: String,
}

impl MatrixGoalRoute {
    fn from_ingress(alias: Option<&str>, room_id: &str) -> Result<Self> {
        let alias = alias
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(DEFAULT_MATRIX_ALIAS);
        ensure!(
            !room_id.trim().is_empty(),
            "Matrix Goal route requires a room id"
        );
        Ok(Self {
            alias: alias.to_owned(),
            room_id: room_id.to_owned(),
        })
    }

    fn parse(route: &str) -> Option<Self> {
        let (alias, room_id) = route.strip_prefix("matrix:")?.split_once(':')?;
        let alias = if alias.trim().is_empty() {
            DEFAULT_MATRIX_ALIAS
        } else {
            alias
        };
        (!room_id.trim().is_empty()).then(|| Self {
            alias: alias.to_owned(),
            room_id: room_id.to_owned(),
        })
    }

    fn encode(&self) -> String {
        format!("matrix:{}:{}", self.alias, self.room_id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatrixGoalPolicyEligibility {
    Eligible,
    Revoked(&'static str),
}

/// Classify a persisted Matrix Goal against the prospective canonical config.
/// It consumes only durable task bindings and shared Matrix policy predicates;
/// hooks and current inbound message data are deliberately not authorities.
fn classify_matrix_goal_policy(
    config: &zeroclaw_config::schema::Config,
    task: &TaskRecord,
) -> MatrixGoalPolicyEligibility {
    if task.kind != TaskKind::Goal || task.status.is_terminal() {
        return MatrixGoalPolicyEligibility::Eligible;
    }
    let Some(raw_route) = task.originator_route.as_deref() else {
        return MatrixGoalPolicyEligibility::Eligible;
    };
    if !raw_route.starts_with("matrix:") {
        return MatrixGoalPolicyEligibility::Eligible;
    }
    let Some(route) = MatrixGoalRoute::parse(raw_route) else {
        return MatrixGoalPolicyEligibility::Revoked("matrix_binding_invalid");
    };
    if !config.goal.enabled {
        return MatrixGoalPolicyEligibility::Revoked("goal_disabled");
    }
    let Some(principal) = task
        .principal_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    else {
        return MatrixGoalPolicyEligibility::Revoked("matrix_binding_invalid");
    };
    if task
        .session_id
        .as_deref()
        .is_none_or(|value| value.trim().is_empty())
    {
        return MatrixGoalPolicyEligibility::Revoked("matrix_binding_invalid");
    }
    let Some(agent) = config.agents.get(&task.agent) else {
        return MatrixGoalPolicyEligibility::Revoked("agent_unavailable");
    };
    if !agent.is_dispatchable() {
        return MatrixGoalPolicyEligibility::Revoked("agent_unavailable");
    }
    let Some(matrix) = config.channels.matrix.get(&route.alias) else {
        return MatrixGoalPolicyEligibility::Revoked("matrix_channel_unavailable");
    };
    if !matrix.enabled
        || config.agent_for_channel(&format!("matrix.{}", route.alias)) != Some(task.agent.as_str())
    {
        return MatrixGoalPolicyEligibility::Revoked("matrix_channel_unavailable");
    }
    if !crate::allowlist::is_matrix_room_allowed(&matrix.allowed_rooms, &route.room_id) {
        return MatrixGoalPolicyEligibility::Revoked("matrix_room_revoked");
    }
    let peers = config.channel_external_peers("matrix", &route.alias);
    if !crate::allowlist::is_matrix_user_allowed(&peers, principal) {
        return MatrixGoalPolicyEligibility::Revoked("matrix_principal_revoked");
    }
    if config
        .resolved_model_provider_for_agent(&task.agent)
        .is_none()
    {
        return MatrixGoalPolicyEligibility::Revoked("agent_provider_unavailable");
    }
    if config
        .goal
        .verifier
        .model_provider
        .as_str()
        .trim()
        .is_empty()
    {
        return MatrixGoalPolicyEligibility::Revoked("verifier_unavailable");
    }
    MatrixGoalPolicyEligibility::Eligible
}

fn policy_revocation_target(task: TaskRecord) -> Result<GoalPolicyRevocationTarget> {
    let session_id = task
        .session_id
        .filter(|value| !value.trim().is_empty())
        .context("Matrix Goal policy target lost its session binding")?;
    Ok(GoalPolicyRevocationTarget {
        task_id: task.id,
        session_id,
        expected_epoch: task.execution_epoch,
    })
}

pub(super) async fn revoke_matrix_goals_for_policy(
    context: &ChannelRuntimeContext,
    next_config: &zeroclaw_config::schema::Config,
) -> Result<()> {
    let control_plane = control_plane().context("Goal control plane is unavailable")?;
    let registry = control_plane.goal_store()?;
    let targets = registry
        .list_nonterminal_session_goals_for_agent(context.agent_alias.as_str())
        .await?
        .into_iter()
        .filter(|task| {
            matches!(
                classify_matrix_goal_policy(next_config, task),
                MatrixGoalPolicyEligibility::Revoked(_)
            )
        })
        .map(policy_revocation_target)
        .collect::<Result<Vec<_>>>()?;
    if targets.is_empty() {
        return Ok(());
    }
    match registry
        .cancel_goals_for_policy_revocation(&targets)
        .await?
    {
        GoalTransitionResult::Applied => {}
        GoalTransitionResult::Stale | GoalTransitionResult::Missing => {
            bail!("Goal policy changed during revocation cutover")
        }
    }
    for target in targets {
        let scope =
            GoalExecutionScope::new(target.task_id, target.session_id, target.expected_epoch)?;
        drain_policy_revoked_goal(&context.persist_locks, &scope).await?;
    }
    Ok(())
}

/// Apply the persisted Matrix policy before channel listeners start. Boot
/// recovery has already classified interrupted pending operations, so there
/// is no resident executor to drain on this path.
pub(super) async fn revoke_matrix_goals_at_startup(
    config: &zeroclaw_config::schema::Config,
) -> Result<()> {
    let Some(control_plane) = control_plane() else {
        return Ok(());
    };
    let registry = control_plane.goal_store()?;
    let targets = registry
        .list_nonterminal_session_goals()
        .await?
        .into_iter()
        .filter(|task| {
            matches!(
                classify_matrix_goal_policy(config, task),
                MatrixGoalPolicyEligibility::Revoked(_)
            )
        })
        .map(policy_revocation_target)
        .collect::<Result<Vec<_>>>()?;
    if targets.is_empty() {
        return Ok(());
    }
    match registry
        .cancel_goals_for_policy_revocation(&targets)
        .await?
    {
        GoalTransitionResult::Applied => Ok(()),
        GoalTransitionResult::Stale | GoalTransitionResult::Missing => {
            bail!("Goal policy changed during startup reconciliation")
        }
    }
}

use super::foreground::{
    drain_policy_revoked_goal, foreground_lock, goal_command_lock, goal_supervisor_slot,
};
use super::{
    ChannelRouteSelection, ChannelRuntimeContext, append_sender_turn,
    build_channel_system_prompt_for_message_with_signal,
    callable_protocol_exposed_for_channel_turn, find_channel_for_message, get_or_create_provider,
    get_route_selection, model_provider_entry_for_ref, outbound_content_format_for_channel,
    runtime_defaults_from_config, runtime_defaults_snapshot,
    sanitize_channel_response_for_format_with_leak_detection, system_prompt_for_channel_turn,
    turn_execution::resolved_channel_execution,
};

/// Immutable Matrix facts captured before mutable inbound hooks run.
#[derive(Clone)]
struct MatrixGoalSessionDriver {
    context: Arc<ChannelRuntimeContext>,
    session_key: GoalSessionKey,
    raw_mxid: String,
    route: String,
    message: ChannelMessage,
}

impl MatrixGoalSessionDriver {
    fn new(
        context: Arc<ChannelRuntimeContext>,
        history_key: String,
        raw_mxid: String,
        route: String,
        message: ChannelMessage,
    ) -> Result<Self> {
        ensure!(
            message.channel.eq_ignore_ascii_case("matrix"),
            "Goal driver is not Matrix"
        );
        ensure!(
            message.sender == raw_mxid,
            "Matrix Goal raw MXID does not match message sender"
        );
        Ok(Self {
            context,
            session_key: GoalSessionKey::matrix(history_key)?,
            raw_mxid,
            route,
            message,
        })
    }

    fn assert_ingress(&self, ingress: &GoalIngressContext) -> Result<()> {
        ensure!(
            ingress.surface() == GoalSurface::Matrix,
            "Goal ingress is not Matrix"
        );
        ensure!(
            ingress.session_key() == &self.session_key,
            "Goal ingress session is stale"
        );
        ensure!(
            ingress.agent() == self.context.agent_alias.as_str(),
            "Goal ingress agent is stale"
        );
        ensure!(ingress.route() == self.route, "Goal ingress route is stale");
        match ingress.principal() {
            zeroclaw_runtime::goal_mode::GoalIngressPrincipal::Matrix { raw_mxid }
                if raw_mxid == &self.raw_mxid =>
            {
                Ok(())
            }
            _ => bail!("Goal ingress principal is stale"),
        }
    }
}

/// Submit an already parsed Goal command using immutable Matrix ingress facts.
pub(super) async fn submit_matrix_goal(
    context: Arc<ChannelRuntimeContext>,
    history_key: String,
    original: ChannelMessage,
    command: zeroclaw_commands::goal::GoalCommand,
) -> Result<zeroclaw_runtime::goal_mode::GoalResponse> {
    let route =
        MatrixGoalRoute::from_ingress(original.channel_alias.as_deref(), &original.reply_target)?
            .encode();
    let driver = Arc::new(MatrixGoalSessionDriver::new(
        Arc::clone(&context),
        history_key.clone(),
        original.sender.clone(),
        route.clone(),
        original,
    )?);
    let _policy_cutover = context.goal_policy_cutover.lock().await;
    super::maybe_apply_runtime_config_update_locked(context.as_ref()).await?;
    let control_plane = control_plane().context("Goal control plane is unavailable")?;
    let registry = control_plane.goal_store()?;
    let defaults = runtime_defaults_snapshot(context.as_ref());
    let configured_limits =
        defaults.config.goal.effective_limits().map_err(|error| {
            anyhow::Error::msg(format!("Goal configuration is invalid: {error:?}"))
        })?;
    let settings = zeroclaw_runtime::goal_mode::GoalHostSettings::new(
        defaults.config.goal.enabled,
        zeroclaw_commands::goal::GoalBudgetLimits {
            token_limit: configured_limits.token_limit,
            cost_limit_usd: configured_limits.cost_limit_usd,
        },
        std::process::id(),
        control_plane.boot_id.clone(),
    )?;
    let current = registry.current_goal_for_session(&history_key).await?;
    let supervisor_slot = goal_supervisor_slot(&context.persist_locks, &history_key);
    let supervisor = {
        let mut slot = supervisor_slot.lock().await;
        if current
            .as_ref()
            .is_some_and(|task| task.status.is_terminal())
        {
            // A completed worker remains in the supervisor until drained. A
            // successor must instead receive a fresh engine resolved from the
            // current runtime configuration.
            *slot = None;
        }
        if let Some(supervisor) = slot.as_ref() {
            Arc::clone(supervisor)
        } else {
            let tracker = CostTracker::get_or_init_global_required(
                defaults.config.cost.clone(),
                &defaults.config.data_dir,
            )?;
            let runtime = zeroclaw_runtime::goal_mode::GoalRuntime::with_tracker(
                Arc::clone(&registry),
                Arc::clone(&tracker),
            );
            let engine = Arc::new(runtime.execution_engine(
                tracker,
                context.agent_alias.to_string(),
                Arc::new(build_type_level_model_provider_pricing(
                    defaults.config.as_ref(),
                )),
            )?);
            let supervisor = Arc::new(zeroclaw_runtime::goal_mode::GoalExecutionSupervisor::new(
                engine,
            ));
            *slot = Some(Arc::clone(&supervisor));
            supervisor
        }
    };
    let ingress = GoalIngressContext::trusted(
        GoalSessionKey::matrix(history_key)?,
        context.agent_alias.to_string(),
        route,
        zeroclaw_runtime::goal_mode::GoalIngressPrincipal::Matrix {
            raw_mxid: driver.raw_mxid.clone(),
        },
    )?;
    let response = supervisor
        .submit(settings, ingress, driver, command)
        .await?;
    if matches!(
        response,
        zeroclaw_runtime::goal_mode::GoalResponse::Paused(_)
            | zeroclaw_runtime::goal_mode::GoalResponse::AlreadyPaused(_)
            | zeroclaw_runtime::goal_mode::GoalResponse::Cancelled(_)
            | zeroclaw_runtime::goal_mode::GoalResponse::AlreadyCancelled(_)
            | zeroclaw_runtime::goal_mode::GoalResponse::Terminal(_)
    ) {
        // Pause and cancellation drain the old worker before the supervisor
        // returns. Rebuilding on an explicit resume picks up live policy and
        // pricing instead of keeping a stale configuration snapshot.
        *supervisor_slot.lock().await = None;
    }
    Ok(response)
}

/// Dispose a Matrix session's Goal before the channel resets its history.
///
/// A resident supervisor performs the required fence-and-drain ordering. If
/// this process has no worker (for example after restart), the durable row is
/// already the sole live authority and can be fenced, classified, and deleted
/// directly without recreating a worker merely to dispose it.
pub(super) async fn dispose_matrix_goal(
    context: &ChannelRuntimeContext,
    history_key: &str,
) -> Result<()> {
    let slot = goal_supervisor_slot(&context.persist_locks, history_key);
    if let Some(supervisor) = slot.lock().await.as_ref().cloned() {
        supervisor.dispose_session(history_key).await?;
        *slot.lock().await = None;
        return Ok(());
    }

    let Some(control_plane) = control_plane() else {
        // Without the canonical control plane no Goal could have been
        // admitted, so ordinary Matrix `/new` keeps its historical behavior.
        return Ok(());
    };
    let registry = control_plane.goal_store()?;
    let _ = dispose_unowned_session_goal(registry.as_ref(), history_key).await?;
    Ok(())
}

#[async_trait]
impl GoalSessionDriver for MatrixGoalSessionDriver {
    fn surface(&self) -> GoalSurface {
        GoalSurface::Matrix
    }

    fn session_key(&self) -> &GoalSessionKey {
        &self.session_key
    }

    async fn bind(&self, ingress: &GoalIngressContext) -> Result<GoalSessionLease> {
        self.assert_ingress(ingress)?;
        let guard = goal_command_lock(&self.context.persist_locks, &self.session_key.durable_id())
            .lock_owned()
            .await;
        Ok(GoalSessionLease::new(
            GoalSessionBinding::new(self.session_key.clone(), self.message.id.clone())?,
            guard,
        ))
    }

    async fn acquire_execution(
        &self,
        ingress: &GoalIngressContext,
        scope: &GoalExecutionScope,
    ) -> Result<Box<dyn GoalSessionExecutionLease>> {
        self.assert_ingress(ingress)?;
        ensure!(
            scope.session_id() == self.session_key.durable_id(),
            "Goal execution session is stale"
        );
        let foreground =
            foreground_lock(&self.context.persist_locks, &self.session_key.durable_id())
                .lock_owned()
                .await;
        Ok(Box::new(MatrixGoalExecutionLease {
            _foreground: foreground,
            context: Arc::clone(&self.context),
            session_key: self.session_key.clone(),
            message: self.message.clone(),
        }))
    }
}

struct MatrixGoalExecutionLease {
    _foreground: tokio::sync::OwnedMutexGuard<()>,
    context: Arc<ChannelRuntimeContext>,
    session_key: GoalSessionKey,
    message: ChannelMessage,
}

impl MatrixGoalExecutionLease {
    fn history(&self) -> Vec<ChatMessage> {
        self.context
            .conversation_histories
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .peek(&self.session_key.durable_id())
            .cloned()
            .unwrap_or_default()
    }

    fn initial_working_history(
        &self,
        provider: &dyn zeroclaw_providers::ModelProvider,
        route: &ChannelRouteSelection,
        canonical_history: Vec<ChatMessage>,
    ) -> Vec<ChatMessage> {
        let excluded_tools: &[String] =
            if self.context.autonomy_level == zeroclaw_config::autonomy::AutonomyLevel::Full {
                &[]
            } else {
                self.context.non_cli_excluded_tools.as_ref()
            };
        let native_tool_specs_present =
            zeroclaw_runtime::agent::loop_::native_tool_specs_present_for_turn(
                provider,
                route.model.as_str(),
                self.context.tools_registry.as_ref(),
                excluded_tools,
                self.context.activated_tools.as_ref(),
            )
            .unwrap_or(false);
        let callable_protocol_exposed = callable_protocol_exposed_for_channel_turn(
            native_tool_specs_present,
            self.context.agent_cfg.resolved.strict_tool_parsing,
            self.context.system_prompt.as_str(),
        );
        let base_system_prompt = system_prompt_for_channel_turn(
            self.context.as_ref(),
            self.context.system_prompt.as_str(),
            true,
            callable_protocol_exposed,
            excluded_tools,
        );
        let target_channel =
            find_channel_for_message(&self.context.channels_by_name, &self.message);
        let system_prompt = build_channel_system_prompt_for_message_with_signal(
            &base_system_prompt,
            &self.message,
            target_channel,
            native_tool_specs_present,
        );
        let mut history = vec![ChatMessage::system(system_prompt)];
        history.extend(canonical_history);
        history
    }
}

#[async_trait]
impl GoalSessionExecutionLease for MatrixGoalExecutionLease {
    fn canonical_history(&self) -> Result<Vec<ChatMessage>> {
        Ok(self.history())
    }

    async fn run_parent_turn(
        &mut self,
        operation: &GoalOperationScope,
        turn: GoalParentTurn,
    ) -> Result<GoalParentTurnResult> {
        let defaults = runtime_defaults_snapshot(self.context.as_ref());
        let route = get_route_selection(
            self.context.as_ref(),
            &self.message,
            &self.session_key.durable_id(),
            &defaults,
        );
        let provider = get_or_create_provider(
            self.context.as_ref(),
            &route.model_provider,
            route.api_key.as_deref(),
            &defaults,
        )
        .await?;
        let directive = turn.prompt_directive();
        let mut history = match turn.kind {
            GoalParentTurnKind::Start | GoalParentTurnKind::Resume => {
                let mut history =
                    self.initial_working_history(provider.as_ref(), &route, turn.working_history);
                history.insert(1, directive);
                history
            }
            GoalParentTurnKind::Continue => {
                let mut history = turn.working_history;
                history.push(directive);
                history
            }
        };
        let turn_id = uuid::Uuid::new_v4().to_string();
        let loop_knobs = LoopKnobs::default();
        let tool_loop = run_tool_call_loop(ToolLoop {
            exec: resolved_channel_execution(
                self.context.as_ref(),
                provider.as_ref(),
                &route,
                self.context.observer.as_ref(),
                &loop_knobs,
                self.context.non_cli_excluded_tools.as_ref(),
                defaults.defaults.temperature,
            ),
            history: &mut history,
            channel_name: "matrix",
            channel_reply_target: Some(self.message.reply_target.as_str()),
            cancellation_token: None,
            on_delta: None,
            shared_budget: None,
            channel: None,
            collected_receipts: None,
            event_tx: None,
            steering: None,
            new_messages_out: None,
            image_cache: None,
            ingress: zeroclaw_api::ingress::IngressContext::channel(),
            memory: None,
            agent_alias: Some(self.context.agent_alias.as_str()),
            parent_agent_alias: None,
            turn_id: &turn_id,
            sop_reassembly: None,
        });
        let candidate = scope_session_key(Some(self.session_key.durable_id()), async {
            scope_thread_id(Some(self.message.id.clone()), tool_loop).await
        })
        .await?;
        let _ = operation;
        Ok(GoalParentTurnResult {
            candidate,
            working_history: history,
        })
    }

    async fn run_verifier(
        &mut self,
        _operation: &GoalOperationScope,
        turn: GoalVerifierTurn,
    ) -> Result<String> {
        let defaults = runtime_defaults_snapshot(self.context.as_ref());
        let provider_ref = defaults.config.goal.verifier.model_provider.as_str();
        let (provider_name, entry) =
            model_provider_entry_for_ref(defaults.config.as_ref(), provider_ref)?;
        let model = defaults.config.goal.verifier.model.as_deref().unwrap_or(
            entry
                .model
                .as_deref()
                .context("Goal verifier has no model")?,
        );
        let verifier_defaults =
            runtime_defaults_from_config(defaults.config.as_ref(), &provider_name)?;
        let provider = get_or_create_provider(
            self.context.as_ref(),
            &provider_name,
            entry.api_key.as_deref(),
            &defaults,
        )
        .await?;
        let messages = vec![
            ChatMessage::system(
                "Return only strict JSON: {\\\"decision\\\":\\\"complete|continue|blocked\\\",\\\"reason\\\":\\\"...\\\",\\\"blockers\\\":[...]}.",
            ),
            ChatMessage::user(format!(
                "Objective:\n{}\n\nCandidate:\n{}",
                turn.objective, turn.candidate
            )),
        ];
        let response = resolved_channel_execution(
            self.context.as_ref(),
            provider.as_ref(),
            &ChannelRouteSelection {
                model_provider: provider_name,
                model: model.to_owned(),
                api_key: entry.api_key.clone(),
            },
            self.context.observer.as_ref(),
            &LoopKnobs::default(),
            &[],
            verifier_defaults.temperature,
        )
        .model_access
        .run_model_query(ChatRequest {
            messages: &messages,
            tools: None,
            thinking: None,
        })
        .await?;
        response.text.context("Goal verifier returned no text")
    }

    async fn append_verified_candidate(&mut self, candidate: String) -> Result<()> {
        let delivered = sanitize_channel_response_for_format_with_leak_detection(
            &candidate,
            self.context.tools_registry.as_ref(),
            &self.context.prompt_config.security.leak_detection,
            outbound_content_format_for_channel(&self.message.channel),
        );
        let channel = find_channel_for_message(&self.context.channels_by_name, &self.message)
            .context("Matrix Goal channel is no longer available")?;
        channel
            .send(&zeroclaw_api::channel::SendMessage::reply_to(
                &self.message,
                &delivered,
            ))
            .await
            .context("deliver verified Matrix Goal candidate")?;
        append_sender_turn(
            self.context.as_ref(),
            &self.session_key.durable_id(),
            ChatMessage::assistant(&delivered),
        );
        Ok(())
    }

    async fn publish_goal_notice(&mut self, notice: GoalExecutionNotice) -> Result<()> {
        let key = match notice {
            GoalExecutionNotice::Completed => "goal-mode-completed",
            GoalExecutionNotice::PausedForBlocker => "goal-mode-paused-blocked",
        };
        let channel = find_channel_for_message(&self.context.channels_by_name, &self.message)
            .context("Matrix Goal channel is no longer available")?;
        channel
            .send(&zeroclaw_api::channel::SendMessage::reply_to(
                &self.message,
                zeroclaw_runtime::i18n::get_required_cli_string(key),
            ))
            .await
            .context("deliver Matrix Goal lifecycle notice")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_goal_route_preserves_colons_in_room_ids_and_legacy_empty_aliases() {
        let parsed = MatrixGoalRoute::parse("matrix::!room:example.org").unwrap();
        assert_eq!(parsed.alias, DEFAULT_MATRIX_ALIAS);
        assert_eq!(parsed.room_id, "!room:example.org");
        assert_eq!(parsed.encode(), "matrix:default:!room:example.org");
    }

    #[test]
    fn matrix_cutover_does_not_revoke_non_matrix_goals() {
        let task = TaskRecord {
            id: "goal-zerocode".to_owned(),
            kind: TaskKind::Goal,
            agent: "main".to_owned(),
            status: zeroclaw_runtime::control_plane::TaskStatus::Running,
            owner_pid: 1,
            owner_boot_id: "test".to_owned(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: Some("zerocode:local".to_owned()),
            delivered: false,
            idem_key: None,
            principal_id: Some("local".to_owned()),
            session_id: Some("zc-session".to_owned()),
            execution_epoch: 1,
            started_at: "2026-09-07T00:00:00Z".to_owned(),
            finished_at: None,
        };
        assert_eq!(
            classify_matrix_goal_policy(&zeroclaw_config::schema::Config::default(), &task),
            MatrixGoalPolicyEligibility::Eligible
        );
    }

    #[test]
    fn goal_parent_directive_keeps_lifecycle_phase_out_of_user_control() {
        for (kind, expected) in [
            (GoalParentTurnKind::Start, "start"),
            (GoalParentTurnKind::Resume, "resume"),
            (GoalParentTurnKind::Continue, "continue"),
        ] {
            let directive = GoalParentTurn {
                kind,
                objective: "the declared success criterion".to_owned(),
                working_history: Vec::new(),
            }
            .prompt_directive();
            assert_eq!(directive.role, "system");
            assert!(directive.content.contains("the declared success criterion"));
            assert!(directive.content.contains("untrusted user-declared data"));
            assert!(
                directive
                    .content
                    .contains(&format!("Turn kind: {expected}"))
            );
        }
    }
}
