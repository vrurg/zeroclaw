//! Matrix-specific trusted binding helpers for the Goal session adapter.
//!
//! Goal lifecycle and execution stay in `zeroclaw-runtime`; this module only
//! reconstructs the Matrix facts that must agree with durable task state.

use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use async_trait::async_trait;
use tokio::sync::OwnedMutexGuard;
use tokio_util::sync::CancellationToken;
use zeroclaw_api::session_keys::sanitize_session_key;
use zeroclaw_runtime::{
    agent::loop_::{
        LoopKnobs, ResolvedAgentExecution, ResolvedIo, ResolvedModelAccess, ResolvedRuntimeKnobs,
        ToolLoop, run_tool_call_loop, scope_session_key,
    },
    goal_mode::{
        GoalExecutionPreflight, GoalExecutionScope, GoalOperationReservation,
        execution::{preflight_reservation, run_isolated_verifier},
        host::{
            GoalHostModelBinding, GoalHostPreflightError, GoalParentTurn, GoalSessionAdapter,
            GoalSessionBinding, GoalSessionLease, GoalVerifiedCandidate, GoalVerifierTurn,
        },
    },
};

use super::{
    CHANNEL_MESSAGE_TIMEOUT_SCALE_CAP, ChannelRouteSelection, ChannelRuntimeContext,
    ChannelRuntimeDefaultsSnapshot, channel_message_timeout_budget_secs_with_cap,
    composite_channel_key, foreground_lock, get_or_create_provider, get_route_selection,
    normalize_cached_channel_turns, resolve_channel_thinking, runtime_defaults_snapshot,
    sanitize_verified_goal_delivery, system_prompt_for_channel_turn, wait_for_foreground_lease,
};

const MATRIX_ROUTE_PREFIX: &str = "matrix:";

fn route_is_owned_by_agent(
    config: &zeroclaw_config::schema::Config,
    channel_key: &str,
    agent: &str,
) -> bool {
    config.agent_for_channel(channel_key) == Some(agent)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MatrixGoalRoute {
    alias: String,
    room_id: String,
}

impl MatrixGoalRoute {
    fn parse(route: &str) -> Result<Self> {
        let Some(rest) = route.strip_prefix(MATRIX_ROUTE_PREFIX) else {
            bail!("Goal route is not Matrix");
        };
        let Some((alias, room_id)) = rest.split_once(':') else {
            bail!("Matrix Goal route is malformed");
        };
        if alias.trim().is_empty()
            || alias.trim() != alias
            || alias.contains('\n')
            || room_id.trim().is_empty()
            || room_id.trim() != room_id
            || room_id.contains('\n')
        {
            bail!("Matrix Goal route has blank or unsafe fields");
        }
        Ok(Self {
            alias: alias.into(),
            room_id: room_id.into(),
        })
    }

    fn canonical_session_id(&self, raw_mxid: &str) -> Result<String> {
        if raw_mxid.trim().is_empty() || raw_mxid.contains('\n') {
            bail!("Matrix Goal principal is blank or unsafe");
        }
        Ok(sanitize_session_key(&format!(
            "matrix.{}_{}_{}",
            self.alias, self.room_id, raw_mxid
        )))
    }

    fn matches_binding(&self, binding: &GoalSessionBinding, expected_agent: &str) -> bool {
        binding.agent() == expected_agent
            && binding.route() == format!("{MATRIX_ROUTE_PREFIX}{}:{}", self.alias, self.room_id)
            && binding
                .principal_id()
                .and_then(|raw_mxid| self.canonical_session_id(raw_mxid).ok())
                .as_deref()
                == Some(binding.session_id())
    }
}

/// Matrix implementation of the transport-neutral Goal live-session contract.
///
/// It owns no Goal lifecycle or accounting state. The existing channel runtime
/// remains the source of the configured provider, tool policy, history, and
/// foreground lock; the generic Goal runtime owns durable control-plane work.
pub struct MatrixGoalSessionAdapter {
    pub(super) context: Arc<ChannelRuntimeContext>,
}

impl MatrixGoalSessionAdapter {
    fn route_for_binding(&self, binding: &GoalSessionBinding) -> Result<MatrixGoalRoute> {
        let route = MatrixGoalRoute::parse(binding.route())?;
        if !route.matches_binding(binding, self.context.agent_alias.as_str()) {
            bail!("Matrix Goal binding does not match the live agent or session");
        }
        let channel_key = composite_channel_key("matrix", Some(route.alias.as_str()));
        if !self.context.channels_by_name.contains_key(&channel_key) {
            bail!("Matrix Goal channel alias is no longer available");
        }
        if !route_is_owned_by_agent(
            self.context.prompt_config.as_ref(),
            &channel_key,
            self.context.agent_alias.as_str(),
        ) {
            bail!("Matrix Goal channel alias is not owned by the live agent");
        }
        Ok(route)
    }

    fn session_message(
        binding: &GoalSessionBinding,
        route: &MatrixGoalRoute,
    ) -> Result<zeroclaw_api::channel::ChannelMessage> {
        Ok(zeroclaw_api::channel::ChannelMessage {
            sender: binding
                .principal_id()
                .context("Matrix Goal binding has no raw MXID")?
                .to_owned(),
            reply_target: route.room_id.clone(),
            channel: "matrix".into(),
            channel_alias: Some(route.alias.clone()),
            ..Default::default()
        })
    }

    fn effective_route(
        &self,
        binding: &GoalSessionBinding,
        route: &MatrixGoalRoute,
        defaults: &ChannelRuntimeDefaultsSnapshot,
    ) -> Result<ChannelRouteSelection> {
        let message = Self::session_message(binding, route)?;
        Ok(get_route_selection(
            self.context.as_ref(),
            &message,
            binding.session_id(),
            defaults,
        ))
    }
}

struct MatrixGoalSessionLease {
    _foreground_guard: OwnedMutexGuard<()>,
    context: Arc<ChannelRuntimeContext>,
    binding: GoalSessionBinding,
    route: MatrixGoalRoute,
}

impl MatrixGoalSessionLease {
    fn current_route(&self) -> Result<(ChannelRuntimeDefaultsSnapshot, ChannelRouteSelection)> {
        let defaults = runtime_defaults_snapshot(self.context.as_ref());
        let route = MatrixGoalSessionAdapter::effective_route(
            &MatrixGoalSessionAdapter {
                context: Arc::clone(&self.context),
            },
            &self.binding,
            &self.route,
            &defaults,
        )?;
        Ok((defaults, route))
    }
}

#[async_trait]
impl GoalSessionAdapter for MatrixGoalSessionAdapter {
    fn owns(&self, binding: &GoalSessionBinding) -> bool {
        self.route_for_binding(binding).is_ok()
    }

    async fn preflight(
        &self,
        binding: &GoalSessionBinding,
        request: &GoalExecutionPreflight,
    ) -> std::result::Result<GoalOperationReservation, GoalHostPreflightError> {
        let route = self
            .route_for_binding(binding)
            .map_err(|_| GoalHostPreflightError::BindingUnavailable)?;
        let defaults = runtime_defaults_snapshot(self.context.as_ref());
        let selection = self
            .effective_route(binding, &route, &defaults)
            .map_err(|_| GoalHostPreflightError::BindingUnavailable)?;
        zeroclaw_runtime::agent::agent::build_session_model_provider(
            defaults.config.as_ref(),
            &defaults.config.goal.verifier.provider,
            defaults.config.goal.verifier.model.as_deref(),
        )
        .map_err(|_| GoalHostPreflightError::VerifierUnavailable)?;
        preflight_reservation(
            defaults.config.as_ref(),
            request,
            &GoalHostModelBinding {
                provider_ref: selection.model_provider,
                model: selection.model,
            },
        )
        .map_err(|_| {
            if request.cost_limit_usd.is_some() {
                GoalHostPreflightError::PricingUnavailable
            } else {
                GoalHostPreflightError::ReservationUnavailable
            }
        })
    }

    async fn model_binding(&self, binding: &GoalSessionBinding) -> Result<GoalHostModelBinding> {
        let route = self.route_for_binding(binding)?;
        let defaults = runtime_defaults_snapshot(self.context.as_ref());
        let selection = self.effective_route(binding, &route, &defaults)?;
        Ok(GoalHostModelBinding {
            provider_ref: selection.model_provider,
            model: selection.model,
        })
    }

    async fn deadline(&self, binding: &GoalSessionBinding) -> Result<Option<tokio::time::Instant>> {
        self.route_for_binding(binding)?;
        let timeout = channel_message_timeout_budget_secs_with_cap(
            self.context.message_timeout_secs,
            self.context.max_tool_iterations,
            CHANNEL_MESSAGE_TIMEOUT_SCALE_CAP,
        );
        Ok(Some(
            tokio::time::Instant::now() + std::time::Duration::from_secs(timeout),
        ))
    }

    async fn acquire(
        &self,
        scope: &GoalExecutionScope,
        binding: &GoalSessionBinding,
    ) -> Result<Box<dyn GoalSessionLease>> {
        let route = self.route_for_binding(binding)?;
        let lock = foreground_lock(&self.context.persist_locks, binding.session_id());
        let foreground_guard = wait_for_foreground_lease(lock, scope.cancellation.clone())
            .await
            .context("Matrix Goal foreground lease was cancelled")?;
        self.route_for_binding(binding)?;
        Ok(Box::new(MatrixGoalSessionLease {
            _foreground_guard: foreground_guard,
            context: Arc::clone(&self.context),
            binding: binding.clone(),
            route,
        }))
    }
}

#[async_trait]
impl GoalSessionLease for MatrixGoalSessionLease {
    fn working_history(&self) -> Result<Vec<zeroclaw_api::model_provider::ChatMessage>> {
        let history = self
            .context
            .conversation_histories
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(self.binding.session_id())
            .cloned()
            .unwrap_or_default();
        Ok(normalize_cached_channel_turns(history))
    }

    async fn run_parent_turn(
        &mut self,
        mut turn: GoalParentTurn,
        cancellation: CancellationToken,
    ) -> Result<(String, Vec<zeroclaw_api::model_provider::ChatMessage>)> {
        let (defaults, selection) = self.current_route()?;
        let provider = get_or_create_provider(
            self.context.as_ref(),
            &selection.model_provider,
            selection.api_key.as_deref(),
            &defaults,
        )
        .await?;
        let thinking_override = self
            .context
            .thinking_overrides
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(self.binding.session_id())
            .copied();
        // A Goal declaration is controller-owned, not a user prompt. Preserve
        // the session's selected thinking level without interpreting objective
        // text as a channel thinking directive.
        let thinking = resolve_channel_thinking(
            "",
            thinking_override,
            &self.context.agent_cfg.resolved.thinking,
            defaults.defaults.temperature,
        );
        let excluded_tools: &[String] =
            if self.context.autonomy_level == zeroclaw_runtime::security::AutonomyLevel::Full {
                &[]
            } else {
                self.context.non_cli_excluded_tools.as_ref()
            };
        let read_skill_available = super::read_skill_available_for_channel_turn(
            provider.as_ref(),
            self.context.agent_cfg.resolved.strict_tool_parsing,
            self.context.tools_registry.as_ref(),
            excluded_tools,
            self.context.system_prompt.as_str(),
        );
        let system_prompt = super::build_channel_system_prompt(
            &system_prompt_for_channel_turn(
                self.context.as_ref(),
                self.context.system_prompt.as_str(),
                false,
                read_skill_available,
            ),
            "matrix",
            None,
        );
        if turn
            .working_history
            .first()
            .is_some_and(|message| message.role == "system")
        {
            turn.working_history[0] =
                zeroclaw_api::model_provider::ChatMessage::system(system_prompt);
        } else {
            turn.working_history.insert(
                0,
                zeroclaw_api::model_provider::ChatMessage::system(system_prompt),
            );
        }
        turn.working_history
            .push(zeroclaw_api::model_provider::ChatMessage::user(
                turn.directive(),
            ));
        let turn_id = uuid::Uuid::new_v4().to_string();
        let loop_knobs = LoopKnobs::default();
        let tool_loop = run_tool_call_loop(ToolLoop {
            exec: ResolvedAgentExecution::resolve(
                ResolvedModelAccess {
                    model_provider: provider.as_ref(),
                    provider_name: selection.model_provider.as_str(),
                    model: selection.model.as_str(),
                    temperature: thinking.effective_temperature,
                },
                ResolvedIo {
                    tools_registry: self.context.tools_registry.as_ref(),
                    observer: self.context.observer.as_ref(),
                    silent: true,
                    approval: Some(self.context.approval_manager.as_ref()),
                    multimodal_config: &self.context.multimodal,
                    config: Some(defaults.config.as_ref()),
                    hooks: self.context.hooks.as_deref(),
                    activated_tools: self.context.activated_tools.as_ref(),
                    model_switch_callback: None,
                    receipt_generator: self.context.receipt_generator.as_ref(),
                },
                ResolvedRuntimeKnobs {
                    max_tool_iterations: self.context.max_tool_iterations,
                    excluded_tools,
                    dedup_exempt_tools: self.context.tool_call_dedup_exempt.as_ref(),
                    pacing: &self.context.pacing,
                    strict_tool_parsing: self.context.agent_cfg.resolved.strict_tool_parsing,
                    parallel_tools: self.context.agent_cfg.resolved.parallel_tools,
                    max_tool_result_chars: self.context.max_tool_result_chars,
                    context_token_budget: self.context.context_token_budget,
                    knobs: &loop_knobs,
                },
            ),
            history: &mut turn.working_history,
            channel_name: "matrix",
            channel_reply_target: Some(self.route.room_id.as_str()),
            cancellation_token: Some(cancellation.clone()),
            on_delta: None,
            shared_budget: None,
            channel: None,
            collected_receipts: None,
            event_tx: None,
            steering: None,
            new_messages_out: None,
            image_cache: None,
            memory: None,
            ingress: zeroclaw_api::ingress::IngressContext::channel(),
            agent_alias: Some(self.context.agent_alias.as_str()),
            parent_agent_alias: None,
            turn_id: &turn_id,
            sop_reassembly: Some(zeroclaw_runtime::agent::loop_::SopStepReassembly {
                config: defaults.config.as_ref(),
            }),
        });
        let tool_loop = zeroclaw_api::NATIVE_THINKING_OVERRIDE
            .scope(thinking.params.native_thinking, tool_loop);
        let response = tokio::select! {
            result = scope_session_key(Some(self.binding.session_id().to_owned()), tool_loop) => result?,
            () = cancellation.cancelled() => bail!("Matrix Goal parent turn was cancelled"),
        };
        Ok((response, turn.working_history))
    }

    async fn run_verifier(
        &mut self,
        turn: GoalVerifierTurn,
        cancellation: CancellationToken,
    ) -> Result<String> {
        let defaults = runtime_defaults_snapshot(self.context.as_ref());
        run_isolated_verifier(defaults.config.as_ref(), &turn, cancellation).await
    }

    async fn append_verified_candidate(&mut self, candidate: GoalVerifiedCandidate) -> Result<()> {
        if candidate.session_id != self.binding.session_id() {
            bail!("Matrix Goal delivery session does not match its lease");
        }
        let candidate = sanitize_verified_goal_delivery(
            &candidate.candidate,
            self.context.tools_registry.as_ref(),
            self.context.prompt_config.as_ref(),
        );
        let channel_key = composite_channel_key("matrix", Some(self.route.alias.as_str()));
        let channel = self
            .context
            .channels_by_name
            .get(&channel_key)
            .context("Matrix Goal channel alias disappeared before delivery")?;
        channel
            .send(&zeroclaw_api::channel::SendMessage::new(
                &candidate,
                &self.route.room_id,
            ))
            .await?;
        super::append_sender_turn(
            self.context.as_ref(),
            self.binding.session_id(),
            zeroclaw_api::model_provider::ChatMessage::assistant(candidate),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use zeroclaw_config::{providers::ChannelRef, schema::AliasedAgentConfig};
    use zeroclaw_runtime::goal_mode::host::GoalSessionBinding;

    use super::{MatrixGoalRoute, route_is_owned_by_agent};

    #[test]
    fn matrix_route_recomputes_the_thread_independent_session_key() {
        let route = MatrixGoalRoute::parse("matrix:primary:!room:example.org").unwrap();
        assert_eq!(route.alias, "primary");
        assert_eq!(route.room_id, "!room:example.org");
        assert_eq!(
            route.canonical_session_id("@user:example.org").unwrap(),
            "matrix_primary__room_example_org__user_example_org"
        );
    }

    #[test]
    fn verified_goal_delivery_uses_the_standard_leak_guard() {
        let candidate = "Temporary key: AKIAABCDEFGHIJKLMNOP"; // gitleaks:allow
        let delivered = super::sanitize_verified_goal_delivery(
            candidate,
            &[],
            &zeroclaw_config::schema::Config::default(),
        );
        assert!(!delivered.contains("AKIAABCDEFGHIJKLMNOP")); // gitleaks:allow
    }

    #[test]
    fn matrix_route_rejects_noncanonical_or_unsafe_bindings() {
        assert!(MatrixGoalRoute::parse("rpc").is_err());
        assert!(MatrixGoalRoute::parse("matrix::!room:example.org").is_err());
        assert!(MatrixGoalRoute::parse("matrix: primary:!room:example.org").is_err());
        assert!(MatrixGoalRoute::parse("matrix:primary:\n").is_err());
        let route = MatrixGoalRoute::parse("matrix:primary:!room:example.org").unwrap();
        assert!(route.canonical_session_id(" ").is_err());
    }

    #[test]
    fn matrix_binding_requires_the_canonical_sender_room_session() {
        let route = MatrixGoalRoute::parse("matrix:primary:!room:example.org").unwrap();
        let binding = GoalSessionBinding::new(
            "matrix_primary__room_example_org__user_example_org".into(),
            "main".into(),
            "matrix:primary:!room:example.org".into(),
            Some("@user:example.org".into()),
        )
        .unwrap();

        assert!(route.matches_binding(&binding, "main"));

        let mismatched_session = GoalSessionBinding::new(
            "matrix_primary__other_example_org__user_example_org".into(),
            "main".into(),
            "matrix:primary:!room:example.org".into(),
            Some("@user:example.org".into()),
        )
        .unwrap();
        assert!(!route.matches_binding(&mismatched_session, "main"));
    }

    #[test]
    fn matrix_route_requires_the_configured_agent_owner() {
        let mut config = zeroclaw_config::schema::Config::default();
        config.agents.insert(
            "main".into(),
            AliasedAgentConfig {
                channels: vec![ChannelRef::new("matrix.primary")],
                ..AliasedAgentConfig::default()
            },
        );

        assert!(route_is_owned_by_agent(&config, "matrix.primary", "main"));
        assert!(!route_is_owned_by_agent(&config, "matrix.primary", "other"));
    }
}
