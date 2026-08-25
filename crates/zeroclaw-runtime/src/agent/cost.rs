use crate::cost::CostTracker;
use crate::cost::types::{BudgetCheck, TokenUsage as CostTokenUsage};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use zeroclaw_config::schema::Config;
use zeroclaw_providers::pricing::ModelRates;

// ── Cost tracking via task-local ──

pub type ModelProviderPricing = HashMap<String, HashMap<String, f64>>;

/// Per-scope token/cost accumulator derived from the usage events emitted
/// during a single task-local runtime invocation.
#[derive(Default, Clone, Copy, Debug)]
pub struct TurnUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
    pub last_input_tokens: u64,
}

pub fn build_model_provider_pricing(config: &Config) -> ModelProviderPricing {
    let mut pricing: ModelProviderPricing = HashMap::new();

    for (type_k, alias_k, profile) in config.providers.models.iter_entries() {
        let mut slot = profile.pricing.clone();
        apply_rate_sheet_pricing(config, type_k, &mut slot);
        if !slot.is_empty() {
            pricing.insert(format!("{type_k}.{alias_k}"), slot);
        }
    }

    pricing
}

pub fn tool_loop_cost_tracking_context_for_agent(
    config: &Config,
    agent_alias: &str,
) -> Option<ToolLoopCostTrackingContext> {
    CostTracker::get_or_init_global(config.cost.clone(), &config.data_dir)
        .map(|tracker| tool_loop_cost_tracking_context_from_tracker(config, agent_alias, tracker))
}

pub fn tool_loop_cost_tracking_context_from_tracker(
    config: &Config,
    agent_alias: &str,
    tracker: Arc<CostTracker>,
) -> ToolLoopCostTrackingContext {
    ToolLoopCostTrackingContext::new(tracker, Arc::new(build_model_provider_pricing(config)))
        .with_agent_alias(agent_alias)
}

pub fn build_type_level_model_provider_pricing(config: &Config) -> ModelProviderPricing {
    let mut pricing: ModelProviderPricing = HashMap::new();

    for (type_k, _alias_k, profile) in config.providers.models.iter_entries() {
        if profile.pricing.is_empty() {
            continue;
        }
        let slot = pricing.entry(type_k.to_string()).or_default();
        merge_pricing(slot, &profile.pricing);
    }

    for (provider_type, _model_id, _rates) in config.cost.rates.providers.models.iter_entries() {
        let slot = pricing.entry(provider_type.to_string()).or_default();
        apply_rate_sheet_pricing(config, provider_type, slot);
    }

    pricing
}

pub fn provider_pricing<'a>(
    pricing: &'a ModelProviderPricing,
    model_provider_name: &str,
) -> Option<&'a HashMap<String, f64>> {
    if let Some(slot) = pricing.get(model_provider_name) {
        return Some(slot);
    }

    // Type-keyed maps are still used in the channels path; when the lookup
    // arrives as `<type>.<alias>`, fall back to the bare provider family.
    if let Some((provider_type, _alias)) = model_provider_name.split_once('.')
        && let Some(slot) = pricing.get(provider_type)
    {
        return Some(slot);
    }

    // Some call sites surface only the bare provider type while the
    // pricing view is keyed by `<type>.<alias>`. Fall back only when the
    // type resolves to exactly one alias entry so pricing stays deterministic.
    let prefix = format!("{model_provider_name}.");
    let mut matches = pricing
        .iter()
        .filter(|(key, _)| key.starts_with(&prefix))
        .map(|(_, slot)| slot);
    let first = matches.next()?;
    if matches.next().is_none() {
        Some(first)
    } else {
        None
    }
}

fn apply_rate_sheet_pricing(config: &Config, provider_type: &str, slot: &mut HashMap<String, f64>) {
    for (rate_provider_type, model_id, rates) in config.cost.rates.providers.models.iter_entries() {
        if rate_provider_type != provider_type {
            continue;
        }
        if let Some(input) = rates.input_per_mtok {
            slot.insert(format!("{model_id}.input"), input);
        }
        if let Some(output) = rates.output_per_mtok {
            slot.insert(format!("{model_id}.output"), output);
        }
        if let Some(cached) = rates.cached_input_per_mtok {
            slot.insert(format!("{model_id}.cached_input"), cached);
        }
    }
}

fn merge_pricing(slot: &mut HashMap<String, f64>, pricing: &HashMap<String, f64>) {
    for (key, value) in pricing {
        slot.insert(key.clone(), *value);
    }
}

/// Context for cost tracking within the tool call loop.
/// Scoped via `tokio::task_local!` at call sites (channels, gateway).
#[derive(Clone)]
pub struct ToolLoopCostTrackingContext {
    /// Shared cost tracker. `None` for usage-only contexts that accumulate
    /// per-turn token totals without persisting cost records or enforcing
    /// budgets (see [`Self::usage_only`]).
    pub tracker: Option<Arc<CostTracker>>,
    pub model_provider_pricing: Arc<ModelProviderPricing>,
    /// Per-scope usage accumulator so wrapping code can read token/cost
    /// totals after the scoped future exits (without racing concurrent
    /// traffic sharing the same tracker).
    pub turn_usage: Arc<Mutex<TurnUsage>>,
    /// Alias of the agent driving this turn. Stamped onto persisted
    /// `CostRecord`s so `/api/cost?agent=<alias>` can attribute spend.
    pub agent_alias: Option<String>,
}

impl ToolLoopCostTrackingContext {
    pub fn new(
        tracker: Arc<CostTracker>,
        model_provider_pricing: Arc<ModelProviderPricing>,
    ) -> Self {
        Self {
            tracker: Some(tracker),
            model_provider_pricing,
            turn_usage: Arc::new(Mutex::new(TurnUsage::default())),
            agent_alias: None,
        }
    }

    pub fn usage_only() -> Self {
        Self {
            tracker: None,
            model_provider_pricing: Arc::new(ModelProviderPricing::new()),
            turn_usage: Arc::new(Mutex::new(TurnUsage::default())),
            agent_alias: None,
        }
    }

    /// Attach an agent alias to this context so subsequent
    /// `record_tool_loop_cost_usage` calls stamp records with it.
    #[must_use]
    pub fn with_agent_alias(mut self, agent_alias: impl Into<String>) -> Self {
        self.agent_alias = Some(agent_alias.into());
        self
    }

    pub fn snapshot_turn_usage(&self) -> TurnUsage {
        TOOL_LOOP_TURN_USAGE
            .try_with(|turn_usage| turn_usage.as_ref().map(|u| *u.lock()).unwrap_or_default())
            .unwrap_or_else(|_| *self.turn_usage.lock())
    }
}

tokio::task_local! {
    pub static TOOL_LOOP_COST_TRACKING_CONTEXT: Option<ToolLoopCostTrackingContext>;
}

tokio::task_local! {
    pub static TOOL_LOOP_TURN_USAGE: Option<Arc<Mutex<TurnUsage>>>;
}

/// Exact request facts presented to the existing configured provider path for
/// one Goal-owned logical operation. This is transient admission input, never
/// durable Goal state or a provider-specific request factory.
#[derive(Clone)]
pub(crate) struct GoalOperationRequest {
    pub model_provider: String,
    pub model: String,
    pub temperature: Option<f64>,
    pub thinking: Option<zeroclaw_api::model_provider::NativeThinkingParams>,
    pub messages: Vec<zeroclaw_api::model_provider::ChatMessage>,
    pub tools: Vec<crate::tools::ToolSpec>,
}

/// One validated usage event from an actually reported provider attempt.
/// Provider and model stay attached until the strict Goal ledger owner has
/// durably persisted the event with its exact route attribution.
pub(crate) struct GoalUsageEvent {
    pub model_provider: String,
    pub model: String,
    pub usage: CostTokenUsage,
}

/// The complete settlement input for a single admitted logical operation.
pub(crate) struct GoalOperationSettlement {
    pub accounting_state: crate::control_plane::GoalAccountingState,
    pub pricing_complete: bool,
    pub usage_events: Vec<GoalUsageEvent>,
}

/// Private task-local bridge from the ordinary provider loop to the Goal
/// execution owner. Reliable remains the configured high-level provider path;
/// this bridge only fences and settles its one logical operation.
#[async_trait]
pub(crate) trait GoalOperationAccounting: Send + Sync {
    fn config(&self) -> &Config;

    async fn admit(&self, request: GoalOperationRequest) -> anyhow::Result<()>;
    async fn settle(&self, settlement: GoalOperationSettlement) -> anyhow::Result<()>;
}

tokio::task_local! {
    pub(crate) static GOAL_OPERATION_ACCOUNTING: Option<Arc<dyn GoalOperationAccounting>>;
}

/// Admit the next Goal logical operation when a Goal execution owner scoped
/// the ordinary tool loop. Ordinary sessions remain entirely unaffected.
pub(crate) async fn admit_goal_operation_if_scoped(
    request: GoalOperationRequest,
) -> anyhow::Result<()> {
    let GoalOperationRequest {
        model_provider,
        model,
        temperature,
        thinking,
        messages,
        tools,
    } = request;
    if model_provider.trim().is_empty() || model.trim().is_empty() {
        anyhow::bail!("goal operation is missing its provider route");
    }
    if temperature.is_some_and(|value| !value.is_finite()) {
        anyhow::bail!("goal operation has a non-finite temperature");
    }
    if let Some(accounting) = GOAL_OPERATION_ACCOUNTING
        .try_with(Clone::clone)
        .ok()
        .flatten()
    {
        accounting
            .admit(GoalOperationRequest {
                model_provider,
                model,
                temperature,
                thinking,
                messages,
                tools,
            })
            .await?;
    }
    Ok(())
}

/// Whether the current tool loop was entered by a strict Goal owner.
pub(crate) fn goal_operation_accounting_is_scoped() -> bool {
    GOAL_OPERATION_ACCOUNTING
        .try_with(|accounting| accounting.is_some())
        .unwrap_or(false)
}

fn resolve_rates_opt(pricing: &HashMap<String, f64>, model: &str) -> ModelRates {
    let try_lookup = |key: &str| -> Option<ModelRates> {
        let input = pricing.get(&format!("{key}.input")).copied();
        let output = pricing.get(&format!("{key}.output")).copied();
        let cached = pricing.get(&format!("{key}.cached_input")).copied();
        let flat = pricing.get(key).copied();
        if input.is_none() && output.is_none() && cached.is_none() && flat.is_none() {
            None
        } else {
            Some(ModelRates {
                input_per_mtok: input.or(flat),
                output_per_mtok: output.or(flat),
                cached_input_per_mtok: cached,
            })
        }
    };

    zeroclaw_providers::pricing::model_id_candidates(model)
        .find_map(try_lookup)
        .unwrap_or_default()
}

fn live_pricing_for(model_provider_name: &str, model: &str) -> Option<ModelRates> {
    let snapshot = zeroclaw_providers::pricing::current_snapshot();
    zeroclaw_providers::pricing::lookup(&snapshot, model_provider_name, model).copied()
}

/// Resolve runtime-owned pricing for one actually executed provider route.
///
/// Goal Mode must preserve an explicit zero-dollar rate as priced rather than
/// treating it as a missing rate.  Keep the `Option`-bearing representation
/// until that distinction has been recorded on the usage event.
pub(crate) fn goal_pricing_rates(
    config: &Config,
    model_provider_name: &str,
    model: &str,
) -> ModelRates {
    let configured = provider_pricing(&build_model_provider_pricing(config), model_provider_name)
        .map(|pricing| resolve_rates_opt(pricing, model))
        .unwrap_or_default();
    let live = (!configured.is_complete())
        .then(|| live_pricing_for(model_provider_name, model))
        .flatten();
    let mut rates = configured.or(live.unwrap_or_default());

    // The catalog is the existing last-resort runtime price source. It fills
    // only dimensions that neither operator configuration nor live pricing
    // supplied, so an explicit free rate remains authoritative.
    if let Some((input, output, cached)) =
        crate::agent::pricing_catalog::global_pricing_rates(model)
    {
        rates = rates.or(ModelRates {
            input_per_mtok: Some(input),
            output_per_mtok: Some(output),
            cached_input_per_mtok: Some(cached),
        });
    }

    rates
}

/// Convert one immutable provider-attempt usage event into the canonical cost
/// record shape used by Goal Mode.
///
/// Unlike ordinary best-effort tracking, Goal Mode rejects partial, zero, and
/// internally inconsistent usage. Pricing remains a separate completeness
/// dimension: unpriced but valid token usage is retained with
/// `pricing_available = false` for token-only and unlimited Goals.
pub(crate) fn goal_usage_from_provider_attempt(
    config: &Config,
    model_provider_name: &str,
    model: &str,
    usage: &zeroclaw_providers::traits::TokenUsage,
) -> anyhow::Result<CostTokenUsage> {
    let input_tokens = usage
        .input_tokens
        .ok_or_else(|| anyhow::Error::msg("goal usage is missing input tokens"))?;
    let output_tokens = usage
        .output_tokens
        .ok_or_else(|| anyhow::Error::msg("goal usage is missing output tokens"))?;
    let cached_input_tokens = usage.cached_input_tokens.unwrap_or(0);

    if cached_input_tokens > input_tokens {
        anyhow::bail!("goal usage reports cached input greater than total input");
    }
    let total_tokens = input_tokens
        .checked_add(output_tokens)
        .ok_or_else(|| anyhow::Error::msg("goal usage token total overflows u64"))?;
    if total_tokens == 0 {
        anyhow::bail!("goal usage reports zero total tokens");
    }

    let rates = goal_pricing_rates(config, model_provider_name, model);
    for rate in [
        rates.input_per_mtok,
        rates.output_per_mtok,
        rates.cached_input_per_mtok,
    ]
    .into_iter()
    .flatten()
    {
        if !rate.is_finite() || rate < 0.0 {
            anyhow::bail!("goal usage resolved an invalid pricing rate");
        }
    }

    let pricing_available = rates.input_per_mtok.is_some() && rates.output_per_mtok.is_some();
    let mut cost_usage = CostTokenUsage::new_with_cache(
        model,
        input_tokens,
        cached_input_tokens,
        output_tokens,
        rates.input_per_mtok.unwrap_or(0.0),
        rates.cached_input_per_mtok.unwrap_or(0.0),
        rates.output_per_mtok.unwrap_or(0.0),
    );
    cost_usage.pricing_available = pricing_available;
    Ok(cost_usage)
}

fn combine_goal_accounting_state(
    current: crate::control_plane::GoalAccountingState,
    next: crate::control_plane::GoalAccountingState,
) -> crate::control_plane::GoalAccountingState {
    use crate::control_plane::GoalAccountingState::{Complete, Invalid, Missing, OutcomeUnknown};

    match (current, next) {
        (OutcomeUnknown, _) | (_, OutcomeUnknown) => OutcomeUnknown,
        (Invalid, _) | (_, Invalid) => Invalid,
        (Missing, _) | (_, Missing) => Missing,
        (Complete, Complete) => Complete,
    }
}

/// Validate the immutable #10144 attempt report before the Goal owner writes
/// any canonical ledger rows. Observed lower-bound usage from an uncertain
/// outcome is retained, while the operation remains non-resumable.
fn goal_operation_settlement(
    config: &Config,
    attempts: &[zeroclaw_providers::dispatch::AccountedAttempt],
) -> GoalOperationSettlement {
    use crate::control_plane::GoalAccountingState;
    use zeroclaw_providers::dispatch::AttemptUsageOutcome;

    let mut accounting_state = if attempts.is_empty() {
        GoalAccountingState::Missing
    } else {
        GoalAccountingState::Complete
    };
    let mut pricing_complete = true;
    let mut usage_events = Vec::new();

    for attempt in attempts {
        let (usage, outcome_state) = match attempt.outcome() {
            AttemptUsageOutcome::Complete(usage) => (Some(usage), GoalAccountingState::Complete),
            AttemptUsageOutcome::Missing => (None, GoalAccountingState::Missing),
            AttemptUsageOutcome::Invalid { .. } => (None, GoalAccountingState::Invalid),
            AttemptUsageOutcome::OutcomeUnknown { observed } => {
                (observed.as_ref(), GoalAccountingState::OutcomeUnknown)
            }
        };
        accounting_state = combine_goal_accounting_state(accounting_state, outcome_state);

        let Some(usage) = usage else {
            continue;
        };
        match goal_usage_from_provider_attempt(
            config,
            attempt.provider_ref(),
            attempt.model(),
            usage,
        ) {
            Ok(usage) => {
                if !usage.pricing_available {
                    pricing_complete = false;
                }
                usage_events.push(GoalUsageEvent {
                    model_provider: attempt.provider_ref().to_string(),
                    model: attempt.model().to_string(),
                    usage,
                });
            }
            Err(_) => {
                accounting_state =
                    combine_goal_accounting_state(accounting_state, GoalAccountingState::Invalid);
            }
        }
    }

    GoalOperationSettlement {
        accounting_state,
        pricing_complete,
        usage_events,
    }
}

/// Flatten config rates merged with the live-price fallback into the
/// `(input, output, cached)` tuple used for cost math. Config wins per
/// dimension ([`ModelRates::or`]); live only fills dimensions config left
/// unset; any dimension still unset bills at `0.0`.
fn merge_config_and_live_rates(
    config_rates: ModelRates,
    live: Option<ModelRates>,
) -> (f64, f64, f64) {
    let merged = config_rates.or(live.unwrap_or_default());
    (
        merged.input_per_mtok.unwrap_or(0.0),
        merged.output_per_mtok.unwrap_or(0.0),
        merged.cached_input_per_mtok.unwrap_or(0.0),
    )
}

/// Record usage from a rejected provider attempt without replacing the accepted
/// response's context-window fill.
pub fn record_rejected_tool_loop_cost_usage(
    model_provider_name: &str,
    model: &str,
    usage: &zeroclaw_providers::traits::TokenUsage,
) -> Option<(u64, f64)> {
    record_tool_loop_cost_usage_inner(model_provider_name, model, usage, false)
}

/// Record token usage from an accepted LLM response via the task-local cost
/// tracker. Returns `(total_tokens, cost_usd)` on success, `None` when not
/// scoped or no usage.
pub fn record_tool_loop_cost_usage(
    model_provider_name: &str,
    model: &str,
    usage: &zeroclaw_providers::traits::TokenUsage,
) -> Option<(u64, f64)> {
    record_tool_loop_cost_usage_inner(model_provider_name, model, usage, true)
}

/// Settle the immutable physical-attempt report exactly once.  Only complete
/// usage (or a valid lower-bound observation preserved after interruption) is
/// billable; accepted context-window fill remains a property of the selected
/// final leaf rather than the aggregate.
pub(crate) struct BillableProviderAttempt<'a> {
    pub index: usize,
    pub attempt: &'a zeroclaw_providers::dispatch::AccountedAttempt,
    pub usage: &'a zeroclaw_providers::traits::TokenUsage,
}

/// Pure, ordered projection for current BestEffort settlement and the future
/// task-attributed persistence caller. Invalid and missing observations never
/// appear in this view.
pub(crate) fn billable_provider_attempts(
    attempts: &[zeroclaw_providers::dispatch::AccountedAttempt],
) -> impl Iterator<Item = BillableProviderAttempt<'_>> {
    use zeroclaw_providers::dispatch::AttemptUsageOutcome;

    attempts.iter().enumerate().filter_map(|(index, attempt)| {
        let usage = match attempt.outcome() {
            AttemptUsageOutcome::Complete(usage) => Some(usage),
            AttemptUsageOutcome::OutcomeUnknown {
                observed: Some(usage),
            } => Some(usage),
            AttemptUsageOutcome::Missing
            | AttemptUsageOutcome::Invalid { .. }
            | AttemptUsageOutcome::OutcomeUnknown { observed: None } => None,
        };
        usage.map(|usage| BillableProviderAttempt {
            index,
            attempt,
            usage,
        })
    })
}

pub(crate) async fn settle_provider_attempts(
    attempts: &[zeroclaw_providers::dispatch::AccountedAttempt],
    accepted_attempt: Option<usize>,
) -> anyhow::Result<()> {
    if let Some(accounting) = GOAL_OPERATION_ACCOUNTING
        .try_with(Clone::clone)
        .ok()
        .flatten()
    {
        let settlement = goal_operation_settlement(accounting.config(), attempts);
        debug_assert!(
            !settlement.usage_events.is_empty()
                || settlement.accounting_state
                    != crate::control_plane::GoalAccountingState::Complete
        );
        debug_assert!(settlement.usage_events.iter().all(|event| {
            !event.model_provider.trim().is_empty()
                && !event.model.trim().is_empty()
                && event.usage.total_tokens > 0
        }));
        return accounting.settle(settlement).await;
    }

    for event in billable_provider_attempts(attempts) {
        let updates_context_window_fill = accepted_attempt == Some(event.index);
        let _ = record_tool_loop_cost_usage_inner(
            event.attempt.provider_ref(),
            event.attempt.model(),
            event.usage,
            updates_context_window_fill,
        );
    }
    Ok(())
}

fn record_tool_loop_cost_usage_inner(
    model_provider_name: &str,
    model: &str,
    usage: &zeroclaw_providers::traits::TokenUsage,
    updates_context_window_fill: bool,
) -> Option<(u64, f64)> {
    let input_tokens = usage.input_tokens.unwrap_or(0);
    let output_tokens = usage.output_tokens.unwrap_or(0);
    let cached_input_tokens = usage.cached_input_tokens.unwrap_or(0);
    let total_tokens = input_tokens.saturating_add(output_tokens);
    if total_tokens == 0 {
        return None;
    }

    let ctx = TOOL_LOOP_COST_TRACKING_CONTEXT
        .try_with(Clone::clone)
        .ok()
        .flatten()?;
    let pricing = provider_pricing(&ctx.model_provider_pricing, model_provider_name);
    let config_rates = pricing
        .map(|map| resolve_rates_opt(map, model))
        .unwrap_or_default();

    // Live-price FALLBACK fills only the dimensions config left unset; never
    // fetches on this path (reads a cached snapshot, empty unless a provider
    // opted into `live_pricing`).
    let live = (!config_rates.is_complete())
        .then(|| live_pricing_for(model_provider_name, model))
        .flatten();
    // `mut` so the global-catalog fallback below can still fill rates config and
    // the live snapshot both left unset.
    let (mut input_rate, mut output_rate, mut cached_rate) =
        merge_config_and_live_rates(config_rates, live);

    let priced_from_catalog = if input_rate == 0.0 && output_rate == 0.0 {
        if let Some((cat_in, cat_out, cat_cached)) =
            crate::agent::pricing_catalog::global_pricing_rates(model)
        {
            input_rate = cat_in;
            output_rate = cat_out;
            if cached_rate == 0.0 {
                cached_rate = cat_cached;
            }
            true
        } else {
            false
        }
    } else {
        false
    };

    let cost_usage = CostTokenUsage::new_with_cache(
        model,
        input_tokens,
        cached_input_tokens,
        output_tokens,
        input_rate,
        cached_rate,
        output_rate,
    );

    if ctx.tracker.is_some() && !priced_from_catalog && input_rate == 0.0 && output_rate == 0.0 {
        warn_once_missing_pricing(model_provider_name, model);
    }

    // Accumulate turn usage: prefer the caller-scoped TOOL_LOOP_TURN_USAGE
    // task-local (ws.rs gateway path), fall back to the context's own
    // turn_usage field (Agent::turn_streamed path, where the task-local is
    // not set up).
    let accumulated = TOOL_LOOP_TURN_USAGE.try_with(|turn_usage| {
        if let Some(turn_usage) = turn_usage {
            let mut usage = turn_usage.lock();
            usage.input_tokens = usage.input_tokens.saturating_add(input_tokens);
            usage.output_tokens = usage.output_tokens.saturating_add(output_tokens);
            usage.cost_usd += cost_usage.cost_usd;
            if updates_context_window_fill {
                // Replace (not accumulate) last_input_tokens with the absolute
                // accepted provider-reported prompt size — this is the accurate
                // "context window fill" measure (see TurnUsage doc comment).
                usage.last_input_tokens = input_tokens;
            }
            true
        } else {
            false
        }
    });
    if !accumulated.unwrap_or(false) {
        let mut turn_usage = ctx.turn_usage.lock();
        turn_usage.input_tokens = turn_usage.input_tokens.saturating_add(input_tokens);
        turn_usage.output_tokens = turn_usage.output_tokens.saturating_add(output_tokens);
        turn_usage.cost_usd += cost_usage.cost_usd;
        if updates_context_window_fill {
            // Replace (not accumulate) last_input_tokens with the absolute
            // accepted provider-reported prompt size.
            turn_usage.last_input_tokens = input_tokens;
        }
    }

    if let Some(tracker) = &ctx.tracker
        && let Err(error) =
            tracker.record_usage_with_agent(cost_usage.clone(), ctx.agent_alias.as_deref())
    {
        ::zeroclaw_log::record!(WARN, ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note).with_category(::zeroclaw_log::EventCategory::Provider).with_outcome(::zeroclaw_log::EventOutcome::Unknown).with_attrs(::serde_json::json!({"model_provider": model_provider_name, "model": model, "error": format!("{}", error)})), "Failed to record cost tracking usage: ");
    }

    Some((cost_usage.total_tokens, cost_usage.cost_usd))
}

/// Insert `(model_provider, model)` into `seen`. Returns `true` on first sighting,
/// `false` thereafter. Split out from `warn_once_missing_pricing` so the
/// dedup contract can be unit-tested with a caller-owned set instead of the
/// process-static one.
fn missing_pricing_first_sighting(
    seen: &Mutex<HashSet<(String, String)>>,
    model_provider: &str,
    model: &str,
) -> bool {
    seen.lock()
        .insert((model_provider.to_string(), model.to_string()))
}

fn warn_once_missing_pricing(model_provider: &str, model: &str) {
    static SEEN: OnceLock<Mutex<HashSet<(String, String)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if missing_pricing_first_sighting(seen, model_provider, model) {
        ::zeroclaw_log::record!(
            WARN,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                .with_category(::zeroclaw_log::EventCategory::Provider)
                .with_outcome(::zeroclaw_log::EventOutcome::Unknown)
                .with_attrs(
                    ::serde_json::json!({"model_provider": model_provider, "model": model})
                ),
            "Cost tracking: no pricing entry found for {model_provider}/{model} — \
             token usage will be recorded with zero cost and budget enforcement \
             is inert for this model. Add a `pricing` table to the model provider \
             entry in config.toml (under `[providers.models.\"{model_provider}\"]`) \
             with `\"{model}.input\"` and `\"{model}.output\"` keys (USD per 1M tokens). \
             This warning fires once per (model_provider, model) pair per process."
        );
    } else {
        ::zeroclaw_log::record!(
            DEBUG,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                .with_category(::zeroclaw_log::EventCategory::Provider)
                .with_attrs(
                    ::serde_json::json!({"model_provider": model_provider, "model": model})
                ),
            "Cost tracking recorded token usage with zero pricing (no pricing entry found)"
        );
    }
}

/// Check budget before an LLM call. Returns `None` when no cost tracking
/// context is scoped (tests, delegate, CLI without cost config).
pub fn check_tool_loop_budget() -> Option<BudgetCheck> {
    TOOL_LOOP_COST_TRACKING_CONTEXT
        .try_with(Clone::clone)
        .ok()
        .flatten()
        .and_then(|ctx| {
            ctx.tracker
                .map(|tracker| tracker.check_budget(0.0).unwrap_or(BudgetCheck::Allowed))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeroclaw_api::attribution::{Attributable, ModelProviderKind, ProviderKind, Role};
    use zeroclaw_api::model_provider::{ChatMessage, ChatRequest, ChatResponse, ModelProvider};
    use zeroclaw_config::schema::{Config, DeepseekModelProviderConfig, ModelProviderConfig};

    #[test]
    fn goal_attempt_usage_requires_complete_nonzero_consistent_usage() {
        let config = Config::default();
        let missing_input = zeroclaw_providers::traits::TokenUsage {
            input_tokens: None,
            output_tokens: Some(1),
            cached_input_tokens: None,
        };
        assert!(
            goal_usage_from_provider_attempt(&config, "provider", "model", &missing_input).is_err()
        );

        let zero = zeroclaw_providers::traits::TokenUsage {
            input_tokens: Some(0),
            output_tokens: Some(0),
            cached_input_tokens: None,
        };
        assert!(goal_usage_from_provider_attempt(&config, "provider", "model", &zero).is_err());

        let invalid_cache = zeroclaw_providers::traits::TokenUsage {
            input_tokens: Some(1),
            output_tokens: Some(1),
            cached_input_tokens: Some(2),
        };
        assert!(
            goal_usage_from_provider_attempt(&config, "provider", "model", &invalid_cache).is_err()
        );

        let overflow = zeroclaw_providers::traits::TokenUsage {
            input_tokens: Some(u64::MAX),
            output_tokens: Some(1),
            cached_input_tokens: None,
        };
        assert!(goal_usage_from_provider_attempt(&config, "provider", "model", &overflow).is_err());
    }

    #[test]
    fn goal_attempt_usage_retains_unpriced_usage_and_explicit_zero_price() {
        let usage = zeroclaw_providers::traits::TokenUsage {
            input_tokens: Some(2),
            output_tokens: Some(3),
            cached_input_tokens: None,
        };
        let unpriced = goal_usage_from_provider_attempt(
            &Config::default(),
            "unpriced",
            "goal-mode-test-unpriced-model",
            &usage,
        )
        .unwrap();
        assert_eq!(unpriced.total_tokens, 5);
        assert!(!unpriced.pricing_available);

        use zeroclaw_config::schema::OpenAIModelProviderConfig;

        let mut config = Config::default();
        config.providers.models.openai.insert(
            "goal".to_string(),
            OpenAIModelProviderConfig {
                base: ModelProviderConfig {
                    pricing: HashMap::from([
                        ("goal-mode-test-free-model.input".to_string(), 0.0),
                        ("goal-mode-test-free-model.output".to_string(), 0.0),
                    ]),
                    ..Default::default()
                },
            },
        );
        let free = goal_usage_from_provider_attempt(
            &config,
            "openai.goal",
            "goal-mode-test-free-model",
            &usage,
        )
        .unwrap();
        assert!(free.pricing_available);
        assert_eq!(free.cost_usd, 0.0);
    }
    use zeroclaw_providers::ProviderDispatch;
    use zeroclaw_providers::dispatch::{AccountedChatScope, with_exact_dispatch_route};

    fn fresh_seen() -> Mutex<HashSet<(String, String)>> {
        Mutex::new(HashSet::new())
    }

    struct RecordingGoalOperationAccounting {
        config: Config,
        admitted: Mutex<Vec<GoalOperationRequest>>,
        settlements: Mutex<Vec<GoalOperationSettlement>>,
    }

    impl RecordingGoalOperationAccounting {
        fn new(config: Config) -> Self {
            Self {
                config,
                admitted: Mutex::new(Vec::new()),
                settlements: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl GoalOperationAccounting for RecordingGoalOperationAccounting {
        fn config(&self) -> &Config {
            &self.config
        }

        async fn admit(&self, request: GoalOperationRequest) -> anyhow::Result<()> {
            self.admitted.lock().push(request);
            Ok(())
        }

        async fn settle(&self, settlement: GoalOperationSettlement) -> anyhow::Result<()> {
            self.settlements.lock().push(settlement);
            Ok(())
        }
    }

    #[tokio::test]
    async fn scoped_goal_settlement_preserves_attempt_routes_and_unpriced_usage() {
        let accounting = Arc::new(RecordingGoalOperationAccounting::new(Config::default()));
        let provider = PricedLeaf {
            alias: "goal-wrapper",
        };
        let messages = vec![ChatMessage::user("hello")];
        let scope = AccountedChatScope::new();

        GOAL_OPERATION_ACCOUNTING
            .scope(
                Some(Arc::clone(&accounting) as Arc<dyn GoalOperationAccounting>),
                async {
                    admit_goal_operation_if_scoped(GoalOperationRequest {
                        model_provider: "goal.primary".to_string(),
                        model: "goal-model".to_string(),
                        temperature: Some(0.2),
                        thinking: None,
                        messages: messages.clone(),
                        tools: Vec::new(),
                    })
                    .await
                    .expect("Goal admission succeeds");
                    scope
                        .scope(with_exact_dispatch_route(
                            "served.fallback".to_string(),
                            "served-model".to_string(),
                            ProviderDispatch::from_ref(&provider).chat(
                                ChatRequest {
                                    messages: &messages,
                                    tools: None,
                                    thinking: None,
                                },
                                "ignored",
                                None,
                            ),
                        ))
                        .await
                        .expect("provider response");
                    settle_provider_attempts(scope.take().attempts(), Some(0))
                        .await
                        .expect("Goal settlement succeeds");
                },
            )
            .await;

        let admitted = accounting.admitted.lock();
        assert_eq!(admitted.len(), 1);
        assert_eq!(admitted[0].model_provider, "goal.primary");
        assert_eq!(admitted[0].messages.len(), 1);
        assert_eq!(admitted[0].messages[0].role, "user");
        assert_eq!(admitted[0].messages[0].content, "hello");
        drop(admitted);

        let settlements = accounting.settlements.lock();
        assert_eq!(settlements.len(), 1);
        let settlement = &settlements[0];
        assert_eq!(
            settlement.accounting_state,
            crate::control_plane::GoalAccountingState::Complete
        );
        assert!(!settlement.pricing_complete);
        assert_eq!(settlement.usage_events.len(), 1);
        assert_eq!(settlement.usage_events[0].model_provider, "served.fallback");
        assert_eq!(settlement.usage_events[0].model, "served-model");
        assert_eq!(settlement.usage_events[0].usage.total_tokens, 1_000_000);
        assert!(!settlement.usage_events[0].usage.pricing_available);
    }

    #[tokio::test]
    async fn settlement_uses_exact_attempt_routes_models_and_record_order() {
        let workspace = tempfile::TempDir::new().unwrap();
        let tracker = Arc::new(
            CostTracker::new(
                zeroclaw_config::schema::CostConfig {
                    enabled: true,
                    ..Default::default()
                },
                workspace.path(),
            )
            .unwrap(),
        );
        let pricing = Arc::new(HashMap::from([
            (
                "provider.first".to_string(),
                HashMap::from([("model-a.input".to_string(), 1.0)]),
            ),
            (
                "provider.second".to_string(),
                HashMap::from([("model-b.input".to_string(), 3.0)]),
            ),
        ]));
        let context = ToolLoopCostTrackingContext::new(Arc::clone(&tracker), pricing);
        let first = PricedLeaf {
            alias: "wrapper.first",
        };
        let second = PricedLeaf {
            alias: "wrapper.second",
        };
        let messages = vec![ChatMessage::user("hello")];
        let scope = AccountedChatScope::new();

        TOOL_LOOP_COST_TRACKING_CONTEXT
            .scope(Some(context), async {
                scope
                    .scope(async {
                        assert!(
                            with_exact_dispatch_route(
                                "provider.first".to_string(),
                                "model-a".to_string(),
                                ProviderDispatch::from_ref(&first).chat(
                                    ChatRequest {
                                        messages: &messages,
                                        tools: None,
                                        thinking: None,
                                    },
                                    "ignored",
                                    None,
                                ),
                            )
                            .await
                            .is_ok()
                        );
                        assert!(
                            with_exact_dispatch_route(
                                "provider.second".to_string(),
                                "model-b".to_string(),
                                ProviderDispatch::from_ref(&second).chat(
                                    ChatRequest {
                                        messages: &messages,
                                        tools: None,
                                        thinking: None,
                                    },
                                    "ignored",
                                    None,
                                ),
                            )
                            .await
                            .is_ok()
                        );
                    })
                    .await;
                let report = scope.take();
                settle_provider_attempts(report.attempts(), None)
                    .await
                    .expect("ordinary settlement succeeds");
            })
            .await;

        let records = std::fs::read_to_string(workspace.path().join("state").join("costs.jsonl"))
            .unwrap()
            .lines()
            .map(|line| {
                serde_json::from_str::<zeroclaw_config::cost::types::CostRecord>(line).unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].usage.model, "model-a");
        assert_eq!(records[1].usage.model, "model-b");
        assert!((records[0].usage.cost_usd - 1.0).abs() < 1e-12);
        assert!((records[1].usage.cost_usd - 3.0).abs() < 1e-12);
    }

    struct PricedLeaf {
        alias: &'static str,
    }

    impl Attributable for PricedLeaf {
        fn role(&self) -> Role {
            Role::Provider(ProviderKind::Model(ModelProviderKind::Custom))
        }

        fn alias(&self) -> &str {
            self.alias
        }
    }

    #[async_trait::async_trait]
    impl ModelProvider for PricedLeaf {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<String> {
            Ok("ok".to_string())
        }

        async fn chat(
            &self,
            _request: ChatRequest<'_>,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<ChatResponse> {
            Ok(ChatResponse {
                text: Some("ok".to_string()),
                tool_calls: Vec::new(),
                usage: Some(zeroclaw_providers::traits::TokenUsage {
                    input_tokens: Some(1_000_000),
                    output_tokens: Some(0),
                    cached_input_tokens: None,
                }),
                reasoning_content: None,
            })
        }
    }

    #[test]
    fn config_rate_wins_live_fills_only_gaps() {
        // Config priced ONLY input; live prices all three. Input must stay the
        // configured value; output/cached come from live.
        let config = ModelRates {
            input_per_mtok: Some(5.0),
            ..ModelRates::default()
        };
        let live = Some(ModelRates {
            input_per_mtok: Some(99.0),
            output_per_mtok: Some(15.0),
            cached_input_per_mtok: Some(1.5),
        });
        assert_eq!(merge_config_and_live_rates(config, live), (5.0, 15.0, 1.5));
    }

    #[test]
    fn no_live_leaves_unconfigured_dimensions_zero() {
        // Empty config + no live snapshot must reproduce today's behavior
        // exactly: all rates zero.
        assert_eq!(
            merge_config_and_live_rates(ModelRates::default(), None),
            (0.0, 0.0, 0.0)
        );
        // A configured zero (genuinely free) is preserved, not "filled".
        assert_eq!(
            merge_config_and_live_rates(
                ModelRates {
                    input_per_mtok: Some(0.0),
                    output_per_mtok: Some(0.0),
                    cached_input_per_mtok: None,
                },
                Some(ModelRates {
                    input_per_mtok: Some(3.0),
                    output_per_mtok: Some(9.0),
                    cached_input_per_mtok: Some(0.3),
                })
            ),
            (0.0, 0.0, 0.3)
        );
    }

    #[test]
    fn first_sighting_returns_true() {
        let seen = fresh_seen();
        assert!(
            missing_pricing_first_sighting(&seen, "minimax", "MiniMax-M3"),
            "first observation of a (model_provider, model) pair must report first-sighting"
        );
    }

    #[test]
    fn second_sighting_same_pair_returns_false() {
        let seen = fresh_seen();
        assert!(missing_pricing_first_sighting(
            &seen,
            "minimax",
            "MiniMax-M3"
        ));
        assert!(
            !missing_pricing_first_sighting(&seen, "minimax", "MiniMax-M3"),
            "second sighting of the same pair must NOT re-fire WARN"
        );
    }

    #[test]
    fn different_models_under_same_provider_are_independent() {
        let seen = fresh_seen();
        assert!(missing_pricing_first_sighting(
            &seen,
            "minimax",
            "MiniMax-M3"
        ));
        assert!(
            missing_pricing_first_sighting(&seen, "minimax", "MiniMax-M2.7"),
            "different model under same model_provider is a distinct pair"
        );
    }

    #[test]
    fn provider_pricing_resolves_composite_and_bare_type_keys() {
        let mut model_rates: HashMap<String, f64> = HashMap::new();
        model_rates.insert("glm-5.1.input".to_string(), 1.4);
        model_rates.insert("glm-5.1.output".to_string(), 4.4);

        // CLI / agent-loop builder keys by the composite `<type>.<alias>`.
        let mut composite_keyed: ModelProviderPricing = HashMap::new();
        composite_keyed.insert("glm.default".to_string(), model_rates.clone());
        assert!(
            provider_pricing(&composite_keyed, "glm.default").is_some(),
            "composite-keyed map must resolve via the verbatim composite lookup"
        );

        // Channel orchestrator builder keys by the bare provider `<type>`, yet
        // the lookup still arrives as the composite alias — must fall back.
        let mut type_keyed: ModelProviderPricing = HashMap::new();
        type_keyed.insert("glm".to_string(), model_rates.clone());
        assert!(
            provider_pricing(&type_keyed, "glm.default").is_some(),
            "type-keyed map must resolve the composite alias via the bare-type fallback"
        );

        // An unrelated provider must not accidentally match.
        assert!(
            provider_pricing(&type_keyed, "openai.default").is_none(),
            "fallback must not resolve a provider type absent from the map"
        );
    }

    #[test]
    fn provider_pricing_resolves_exact_alias_when_type_is_ambiguous() {
        let pricing = HashMap::from([
            (
                "openai.fast".to_string(),
                HashMap::from([
                    ("gpt-4o-mini.input".to_string(), 0.15),
                    ("gpt-4o-mini.output".to_string(), 0.60),
                ]),
            ),
            (
                "openai.smart".to_string(),
                HashMap::from([
                    ("gpt-4o.input".to_string(), 2.50),
                    ("gpt-4o.output".to_string(), 10.00),
                ]),
            ),
        ]);

        let fast = provider_pricing(&pricing, "openai.fast").unwrap();
        assert_eq!(fast.get("gpt-4o-mini.input"), Some(&0.15));
        assert!(!fast.contains_key("gpt-4o.input"));
        assert!(provider_pricing(&pricing, "openai").is_none());
    }

    #[test]
    fn build_pricing_keeps_same_type_aliases_distinct() {
        use zeroclaw_config::schema::{ModelProviderConfig, OpenAIModelProviderConfig};

        let mut config = Config::default();
        config.providers.models.openai.insert(
            "fast".to_string(),
            OpenAIModelProviderConfig {
                base: ModelProviderConfig {
                    model: Some("gpt-4o-mini".to_string()),
                    pricing: HashMap::from([
                        ("gpt-4o-mini.input".to_string(), 0.15),
                        ("gpt-4o-mini.output".to_string(), 0.60),
                    ]),
                    ..Default::default()
                },
            },
        );
        config.providers.models.openai.insert(
            "smart".to_string(),
            OpenAIModelProviderConfig {
                base: ModelProviderConfig {
                    model: Some("gpt-4o".to_string()),
                    pricing: HashMap::from([
                        ("gpt-4o.input".to_string(), 2.50),
                        ("gpt-4o.output".to_string(), 10.00),
                    ]),
                    ..Default::default()
                },
            },
        );

        let pricing = build_model_provider_pricing(&config);
        assert_eq!(
            provider_pricing(&pricing, "openai.fast")
                .unwrap()
                .get("gpt-4o-mini.input"),
            Some(&0.15)
        );
        assert_eq!(
            provider_pricing(&pricing, "openai.smart")
                .unwrap()
                .get("gpt-4o.output"),
            Some(&10.00)
        );
        assert!(provider_pricing(&pricing, "openai").is_none());
    }

    #[test]
    fn different_providers_for_same_model_are_independent() {
        // Same model name served by two different model_providers — operator may
        // configure them at different rates, so the warn must fire for each.
        let seen = fresh_seen();
        assert!(missing_pricing_first_sighting(
            &seen,
            "openrouter",
            "anthropic/claude-sonnet-4-5"
        ));
        assert!(
            missing_pricing_first_sighting(&seen, "anthropic", "anthropic/claude-sonnet-4-5"),
            "different model_provider for the same model is a distinct pair"
        );
    }

    #[test]
    fn empty_strings_dedup_independently() {
        // Defensive: empty model_provider or model shouldn't collide with each other.
        let seen = fresh_seen();
        assert!(missing_pricing_first_sighting(&seen, "", "model"));
        assert!(missing_pricing_first_sighting(&seen, "model_provider", ""));
        assert!(missing_pricing_first_sighting(&seen, "", ""));
        assert!(!missing_pricing_first_sighting(&seen, "", ""));
    }

    fn pricing_with_cache(
        model: &str,
        input: f64,
        cached_input: f64,
        output: f64,
    ) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        map.insert(format!("{model}.input"), input);
        map.insert(format!("{model}.cached_input"), cached_input);
        map.insert(format!("{model}.output"), output);
        map
    }

    #[test]
    fn build_model_provider_pricing_prefers_rate_sheet_over_legacy_alias_pricing() {
        let mut config = Config::default();
        config.providers.models.deepseek.insert(
            "default".to_string(),
            DeepseekModelProviderConfig {
                base: ModelProviderConfig {
                    model: Some("deepseek-v4-flash".into()),
                    pricing: HashMap::from([("deepseek-v4-flash.output".into(), 0.77)]),
                    ..Default::default()
                },
            },
        );
        config.cost.rates.providers.models.deepseek.insert(
            "deepseek-v4-flash".to_string(),
            zeroclaw_config::schema::ModelCostRates {
                input_per_mtok: Some(0.14),
                output_per_mtok: Some(0.28),
                cached_input_per_mtok: Some(0.0028),
            },
        );

        let alias_map = build_model_provider_pricing(&config);
        let deepseek = alias_map
            .get("deepseek.default")
            .expect("deepseek alias pricing");
        assert_eq!(deepseek.get("deepseek-v4-flash.input").copied(), Some(0.14));
        assert_eq!(
            deepseek.get("deepseek-v4-flash.cached_input").copied(),
            Some(0.0028)
        );
        assert_eq!(
            deepseek.get("deepseek-v4-flash.output").copied(),
            Some(0.28),
            "cost.rates must remain the canonical pricing source"
        );
    }

    #[test]
    fn build_model_provider_pricing_keeps_alias_legacy_pricing_isolated() {
        let mut config = Config::default();
        config.providers.models.deepseek.insert(
            "work".to_string(),
            DeepseekModelProviderConfig {
                base: ModelProviderConfig {
                    pricing: HashMap::from([("deepseek-v4-flash.output".into(), 0.77)]),
                    ..Default::default()
                },
            },
        );
        config.providers.models.deepseek.insert(
            "personal".to_string(),
            DeepseekModelProviderConfig {
                base: ModelProviderConfig {
                    pricing: HashMap::from([("deepseek-v4-flash.output".into(), 0.91)]),
                    ..Default::default()
                },
            },
        );

        let alias_map = build_model_provider_pricing(&config);
        let work = alias_map.get("deepseek.work").expect("work alias pricing");
        let personal = alias_map
            .get("deepseek.personal")
            .expect("personal alias pricing");

        assert_eq!(work.get("deepseek-v4-flash.output").copied(), Some(0.77));
        assert_eq!(
            personal.get("deepseek-v4-flash.output").copied(),
            Some(0.91)
        );
    }

    #[test]
    fn build_type_level_model_provider_pricing_merges_aliases_and_rate_sheet() {
        let mut config = Config::default();
        config.providers.models.deepseek.insert(
            "work".to_string(),
            DeepseekModelProviderConfig {
                base: ModelProviderConfig {
                    pricing: HashMap::from([
                        ("deepseek-v4-flash.input".into(), 0.33),
                        ("deepseek-v4-flash.output".into(), 0.77),
                    ]),
                    ..Default::default()
                },
            },
        );
        config.providers.models.deepseek.insert(
            "personal".to_string(),
            DeepseekModelProviderConfig {
                base: ModelProviderConfig {
                    pricing: HashMap::from([("deepseek-v4-flash.output".into(), 0.91)]),
                    ..Default::default()
                },
            },
        );
        config.cost.rates.providers.models.deepseek.insert(
            "deepseek-v4-flash".to_string(),
            zeroclaw_config::schema::ModelCostRates {
                input_per_mtok: Some(0.14),
                output_per_mtok: Some(0.28),
                cached_input_per_mtok: Some(0.0028),
            },
        );

        let by_type = build_type_level_model_provider_pricing(&config);
        let deepseek = by_type.get("deepseek").expect("deepseek type pricing");
        assert_eq!(deepseek.get("deepseek-v4-flash.input").copied(), Some(0.14));
        assert_eq!(
            deepseek.get("deepseek-v4-flash.output").copied(),
            Some(0.28)
        );
        assert_eq!(
            deepseek.get("deepseek-v4-flash.cached_input").copied(),
            Some(0.0028)
        );
    }

    #[test]
    fn build_type_level_model_provider_pricing_keeps_legacy_last_alias_wins_behavior() {
        let mut config = Config::default();
        config.providers.models.deepseek.insert(
            "work".to_string(),
            DeepseekModelProviderConfig {
                base: ModelProviderConfig {
                    pricing: HashMap::from([("deepseek-v4-flash.output".into(), 0.77)]),
                    ..Default::default()
                },
            },
        );
        config.providers.models.deepseek.insert(
            "personal".to_string(),
            DeepseekModelProviderConfig {
                base: ModelProviderConfig {
                    pricing: HashMap::from([("deepseek-v4-flash.output".into(), 0.91)]),
                    ..Default::default()
                },
            },
        );

        let mut expected = HashMap::new();
        for (type_k, _alias_k, profile) in config.providers.models.iter_entries() {
            if profile.pricing.is_empty() {
                continue;
            }
            let slot = expected
                .entry(type_k.to_string())
                .or_insert_with(HashMap::new);
            for (key, value) in &profile.pricing {
                slot.insert(key.clone(), *value);
            }
        }

        let by_type = build_type_level_model_provider_pricing(&config);
        let deepseek = by_type.get("deepseek").expect("deepseek type pricing");
        let expected_deepseek = expected.get("deepseek").expect("expected deepseek pricing");
        assert_eq!(deepseek, expected_deepseek);
    }

    #[test]
    fn record_tool_loop_cost_usage_applies_cached_input_pricing() {
        let workspace = tempfile::TempDir::new().unwrap();
        let tracker = Arc::new(
            CostTracker::new(
                zeroclaw_config::schema::CostConfig::default(),
                workspace.path(),
            )
            .unwrap(),
        );
        let ctx = ToolLoopCostTrackingContext::new(
            Arc::clone(&tracker),
            Arc::new(HashMap::from([(
                "deepseek".to_string(),
                pricing_with_cache("deepseek-chat", 0.27, 0.027, 1.10),
            )])),
        );
        let usage = zeroclaw_providers::traits::TokenUsage {
            input_tokens: Some(5_000),
            output_tokens: Some(200),
            cached_input_tokens: Some(4_000),
        };

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (total_tokens, cost_usd) = runtime
            .block_on(TOOL_LOOP_COST_TRACKING_CONTEXT.scope(Some(ctx), async {
                record_tool_loop_cost_usage("deepseek", "deepseek-chat", &usage)
            }))
            .expect("cost usage");

        let expected = (1_000.0 * 0.27 / 1_000_000.0)
            + (4_000.0 * 0.027 / 1_000_000.0)
            + (200.0 * 1.10 / 1_000_000.0);
        assert_eq!(total_tokens, 5_200);
        assert!((cost_usd - expected).abs() < 1e-12);

        let stored = std::fs::read_to_string(workspace.path().join("state").join("costs.jsonl"))
            .expect("costs.jsonl should be written");
        let record: zeroclaw_config::cost::types::CostRecord =
            serde_json::from_str(stored.lines().next().expect("one record")).unwrap();
        assert_eq!(record.usage.cached_input_tokens, 4_000);
        assert_eq!(record.usage.billable_input_tokens(), 1_000);
    }

    #[test]
    fn record_tool_loop_cost_usage_keeps_turn_usage_when_persistence_fails() {
        let workspace = tempfile::TempDir::new().unwrap();
        let tracker = Arc::new(
            CostTracker::new(
                zeroclaw_config::schema::CostConfig::default(),
                workspace.path(),
            )
            .unwrap(),
        );
        std::fs::create_dir_all(workspace.path().join("state").join("costs.jsonl"))
            .expect("make costs.jsonl path unwritable as a directory");
        let ctx = ToolLoopCostTrackingContext::new(
            Arc::clone(&tracker),
            Arc::new(HashMap::from([(
                "deepseek".to_string(),
                pricing_with_cache("deepseek-chat", 0.27, 0.027, 1.10),
            )])),
        );
        let turn_usage = Arc::new(Mutex::new(TurnUsage::default()));
        let usage = zeroclaw_providers::traits::TokenUsage {
            input_tokens: Some(5_000),
            output_tokens: Some(200),
            cached_input_tokens: Some(4_000),
        };

        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (total_tokens, cost_usd) = runtime
            .block_on(TOOL_LOOP_TURN_USAGE.scope(
                Some(Arc::clone(&turn_usage)),
                TOOL_LOOP_COST_TRACKING_CONTEXT.scope(Some(ctx), async {
                    record_tool_loop_cost_usage("deepseek", "deepseek-chat", &usage)
                }),
            ))
            .expect("cost usage");

        let expected = (1_000.0 * 0.27 / 1_000_000.0)
            + (4_000.0 * 0.027 / 1_000_000.0)
            + (200.0 * 1.10 / 1_000_000.0);
        assert_eq!(total_tokens, 5_200);
        assert!((cost_usd - expected).abs() < 1e-12);

        let recorded = *turn_usage.lock();
        assert_eq!(recorded.input_tokens, 5_000);
        assert_eq!(recorded.output_tokens, 200);
        assert!((recorded.cost_usd - expected).abs() < 1e-12);
    }

    #[test]
    fn rejected_usage_accumulates_cost_without_replacing_context_window_fill() {
        let turn_usage = Arc::new(Mutex::new(TurnUsage::default()));
        let rejected = zeroclaw_providers::traits::TokenUsage {
            input_tokens: Some(80),
            output_tokens: Some(5),
            cached_input_tokens: Some(0),
        };
        let accepted = zeroclaw_providers::traits::TokenUsage {
            input_tokens: Some(80),
            output_tokens: Some(7),
            cached_input_tokens: Some(0),
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(TOOL_LOOP_TURN_USAGE.scope(
            Some(Arc::clone(&turn_usage)),
            TOOL_LOOP_COST_TRACKING_CONTEXT.scope(
                Some(ToolLoopCostTrackingContext::usage_only()),
                async {
                    assert!(
                        record_rejected_tool_loop_cost_usage("test", "test", &rejected).is_some()
                    );
                    assert!(record_tool_loop_cost_usage("test", "test", &accepted).is_some());
                },
            ),
        ));

        let usage = *turn_usage.lock();
        assert_eq!(usage.input_tokens, 160, "both attempts remain billed");
        assert_eq!(usage.output_tokens, 12, "both attempts remain billed");
        assert_eq!(
            usage.last_input_tokens, 80,
            "only the accepted attempt controls context-window fill"
        );
    }

    #[test]
    fn tool_loop_cost_tracking_context_from_tracker_stamps_given_alias() {
        // `tool_loop_cost_tracking_context_for_agent` (the builder
        // `send_message_to_peer.rs` calls for the peer-turn cost scope)
        // delegates to this function after resolving the process-global
        // tracker. Exercise it directly with an explicit LOCAL tracker so
        // this stays flake-free (no global-tracker singleton touched) while
        // still pinning the alias-stamping contract the per-agent
        // attribution depends on.
        let workspace = tempfile::TempDir::new().unwrap();
        let tracker = Arc::new(
            CostTracker::new(
                zeroclaw_config::schema::CostConfig {
                    enabled: true,
                    track_per_agent: true,
                    ..zeroclaw_config::schema::CostConfig::default()
                },
                workspace.path(),
            )
            .unwrap(),
        );

        let ctx = tool_loop_cost_tracking_context_from_tracker(
            &Config::default(),
            "peer-recipient",
            tracker,
        );

        assert_eq!(ctx.agent_alias, Some("peer-recipient".to_string()));
    }
}
