//! RPC session state.

use crate::agent::agent::{Agent, TurnEvent};
use crate::agent::dispatcher::ToolDispatcher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;
use tokio::sync::Mutex;
use zeroclaw_api::plan::PlanEntry;
use zeroclaw_infra::session_queue::SessionActorQueue;
use zeroclaw_providers::ModelProvider;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelCause {
    /// `session/cancel` RPC arrived over the client channel. The actor is not
    /// verified here: a human interrupt and a programmatic client cancel (new
    /// prompt superseding an in-flight turn, reconnect, client-side timeout,
    /// pane teardown) all land on this path. Attribute to the channel, not a user.
    ClientRpc,
    /// Explicit `session/kill` from the dashboard or admin RPC.
    AdminKill,
    /// The session was explicitly removed/torn down while a turn was live.
    SessionRemoved,
}

impl CancelCause {
    pub fn as_str(self) -> &'static str {
        match self {
            CancelCause::ClientRpc => "client_rpc",
            CancelCause::AdminKill => "admin_kill",
            CancelCause::SessionRemoved => "session_removed",
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SessionOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

/// An entry in the per-session upload index (content-addressed by SHA-256).
#[derive(Clone, Debug)]
pub struct UploadEntry {
    pub ref_id: String,
    pub marker: String,
    pub workspace_path: String,
    pub size_bytes: u64,
}

pub struct RpcSession {
    pub agent: Arc<Mutex<Agent>>,
    /// Orders provider refreshes and configuration within this session.
    model_provider_update: Arc<Mutex<()>>,
    pub created_at: Instant,
    pub last_active: Instant,
    pub agent_alias: String,
    pub workspace_dir: String,
    pub overrides: SessionOverrides,
    pub uploads: HashMap<String, UploadEntry>,
    pub plan: Vec<PlanEntry>,
    pub chat_mode: crate::rpc::types::ChatMode,
    pub owner_tui_id: Option<String>,
    generation: u64,
}

/// Immutable metadata for an already-live RPC session.
///
/// `session/new` uses this to distinguish a same-ID reconnect from a genuine
/// replacement without exposing the mutable session record to the dispatcher.
#[derive(Clone)]
pub(crate) struct LiveRpcSession {
    pub agent_alias: String,
    pub workspace_dir: String,
    pub chat_mode: crate::rpc::types::ChatMode,
    pub owner_tui_id: Option<String>,
    generation: u64,
    agent: Arc<Mutex<Agent>>,
}

impl LiveRpcSession {
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn agent(&self) -> Arc<Mutex<Agent>> {
        Arc::clone(&self.agent)
    }
}

impl RpcSession {
    pub fn new(
        agent: Agent,
        alias: &str,
        workspace: &str,
        chat_mode: crate::rpc::types::ChatMode,
    ) -> Self {
        Self {
            agent: Arc::new(Mutex::new(agent)),
            model_provider_update: Arc::new(Mutex::new(())),
            created_at: Instant::now(),
            last_active: Instant::now(),
            agent_alias: alias.to_string(),
            workspace_dir: workspace.to_string(),
            overrides: SessionOverrides::default(),
            uploads: HashMap::new(),
            plan: Vec::new(),
            chat_mode,
            owner_tui_id: None,
            generation: 0,
        }
    }

    /// Bind this session to a TUI owner.
    pub fn with_owner(mut self, tui_id: Option<String>) -> Self {
        self.owner_tui_id = tui_id;
        self
    }
}

#[derive(Clone, Debug)]
struct ReservedRpcSession {
    id: String,
    generation: u64,
}

/// An exact set of idle session generations reserved for one `session/new`
/// replacement. The reservation also holds one incoming capacity slot, so a
/// fallible asynchronous Goal-disposal step cannot lose its eventual insert
/// to an unrelated concurrent session creation.
pub(crate) struct SessionReplacementReservation {
    targets: Vec<ReservedRpcSession>,
    incoming_id: String,
    reserved_generations: Arc<std::sync::Mutex<HashMap<String, u64>>>,
    reserved_incoming_ids: Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
}

impl Drop for SessionReplacementReservation {
    fn drop(&mut self) {
        let mut generations = self
            .reserved_generations
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for target in &self.targets {
            if generations.get(&target.id) == Some(&target.generation) {
                generations.remove(&target.id);
            }
        }
        drop(generations);
        self.reserved_incoming_ids
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.incoming_id);
    }
}

impl SessionReplacementReservation {
    pub(crate) fn target_ids(&self) -> impl Iterator<Item = &str> {
        self.targets.iter().map(|target| target.id.as_str())
    }
}

pub struct SessionStore {
    sessions: Mutex<HashMap<String, RpcSession>>,
    #[cfg(test)]
    model_provider_update_waiting: Arc<tokio::sync::Notify>,
    cancel_tokens: std::sync::Mutex<HashMap<String, (u64, tokio_util::sync::CancellationToken)>>,
    cancel_generation: std::sync::atomic::AtomicU64,
    session_generation: AtomicU64,
    reserved_generations: Arc<std::sync::Mutex<HashMap<String, u64>>>,
    reserved_incoming_ids: Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
    cancel_causes: std::sync::Mutex<HashMap<String, CancelCause>>,
    max_sessions: usize,
    pub session_queue: Arc<SessionActorQueue>,
}

impl SessionStore {
    pub fn new(max_sessions: usize, session_queue: Arc<SessionActorQueue>) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            #[cfg(test)]
            model_provider_update_waiting: Arc::new(tokio::sync::Notify::new()),
            cancel_tokens: std::sync::Mutex::new(HashMap::new()),
            cancel_generation: std::sync::atomic::AtomicU64::new(0),
            session_generation: AtomicU64::new(0),
            reserved_generations: Arc::new(std::sync::Mutex::new(HashMap::new())),
            reserved_incoming_ids: Arc::new(
                std::sync::Mutex::new(std::collections::HashSet::new()),
            ),
            cancel_causes: std::sync::Mutex::new(HashMap::new()),
            max_sessions,
            session_queue,
        }
    }

    pub async fn insert(&self, id: String, mut session: RpcSession) -> Result<(), &'static str> {
        let mut sessions = self.sessions.lock().await;
        if self
            .reserved_incoming_ids
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&id)
        {
            return Err("session replacement in progress");
        }
        if sessions.len() >= self.max_sessions {
            return Err("session limit reached");
        }
        session.generation = self.session_generation.fetch_add(1, Ordering::SeqCst) + 1;
        sessions.insert(id, session);
        Ok(())
    }

    /// Reserve all currently idle siblings for one incoming session. Reserved
    /// sessions cannot begin new turns, and only the exact incoming session
    /// identifier is reserved until this replacement commits or is released.
    pub(crate) async fn reserve_idle_same_mode_replacement(
        &self,
        tui_id: &str,
        chat_mode: &crate::rpc::types::ChatMode,
        incoming_id: &str,
    ) -> Result<SessionReplacementReservation, &'static str> {
        let sessions = self.sessions.lock().await;
        let in_flight = self.cancel_tokens.lock().unwrap_or_else(|e| e.into_inner());
        let reserved = self
            .reserved_generations
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if self
            .reserved_incoming_ids
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(incoming_id)
        {
            return Err("replacement incoming identifier is reserved");
        }
        let targets: Vec<ReservedRpcSession> = sessions
            .iter()
            .filter(|(id, session)| {
                id.as_str() != incoming_id
                    && session.owner_tui_id.as_deref() == Some(tui_id)
                    && &session.chat_mode == chat_mode
                    && !reserved.contains_key(id.as_str())
                    && !in_flight.contains_key(id.as_str())
            })
            .map(|(id, session)| ReservedRpcSession {
                id: id.clone(),
                generation: session.generation,
            })
            .collect();
        if sessions.len().saturating_sub(targets.len()) >= self.max_sessions {
            return Err("session limit reached");
        }
        for target in &targets {
            let Some(session) = sessions.get(&target.id) else {
                return Err("replacement session disappeared");
            };
            if session.generation != target.generation || reserved.contains_key(&target.id) {
                return Err("replacement session changed");
            }
        }
        drop(in_flight);
        drop(reserved);
        let mut reserved = self
            .reserved_generations
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut incoming = self
            .reserved_incoming_ids
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if !incoming.insert(incoming_id.to_string()) {
            return Err("replacement incoming identifier is reserved");
        }
        for target in &targets {
            reserved.insert(target.id.clone(), target.generation);
        }
        Ok(SessionReplacementReservation {
            targets,
            incoming_id: incoming_id.to_string(),
            reserved_generations: Arc::clone(&self.reserved_generations),
            reserved_incoming_ids: Arc::clone(&self.reserved_incoming_ids),
        })
    }

    /// Release a replacement that failed before session-map commit.
    pub(crate) async fn release_replacement_reservation(
        &self,
        reservation: &SessionReplacementReservation,
    ) {
        let _ = reservation;
    }

    /// Atomically remove the exact reserved sessions and insert their
    /// replacement. A changed generation is never removed by ID alone.
    pub(crate) async fn commit_replacement(
        &self,
        reservation: &SessionReplacementReservation,
        id: String,
        mut session: RpcSession,
    ) -> Result<Vec<(String, String)>, &'static str> {
        let mut sessions = self.sessions.lock().await;
        if sessions.contains_key(&id) {
            return Err("replacement session identifier already exists");
        }
        for target in &reservation.targets {
            let Some(current) = sessions.get(&target.id) else {
                return Err("reserved replacement session disappeared");
            };
            if current.generation != target.generation
                || self
                    .reserved_generations
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(&target.id)
                    != Some(&target.generation)
            {
                return Err("reserved replacement session changed");
            }
        }
        if sessions.len().saturating_sub(reservation.targets.len()) >= self.max_sessions {
            return Err("session limit reached");
        }
        let mut evicted = Vec::with_capacity(reservation.targets.len());
        for target in &reservation.targets {
            if let Some(removed) = sessions.remove(&target.id) {
                evicted.push((target.id.clone(), removed.agent_alias));
            }
        }
        session.generation = self.session_generation.fetch_add(1, Ordering::SeqCst) + 1;
        sessions.insert(id, session);
        Ok(evicted)
    }

    /// A reserved replacement target may not begin a new ordinary turn.
    pub(crate) async fn is_replacement_reserved(&self, id: &str) -> bool {
        self.sessions.lock().await.get(id).is_some_and(|session| {
            self.reserved_generations
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(id)
                == Some(&session.generation)
        })
    }

    /// Register an in-flight turn only if no replacement has fenced this
    /// exact live session. If a reservation wins after this check, it sees the
    /// token and refuses to select the session as an idle target.
    pub(crate) async fn try_register_cancel_token(
        &self,
        id: &str,
        token: tokio_util::sync::CancellationToken,
    ) -> Option<u64> {
        let sessions = self.sessions.lock().await;
        let session = sessions.get(id)?;
        if self
            .reserved_generations
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
            == Some(&session.generation)
        {
            return None;
        }
        let generation = self
            .cancel_generation
            .fetch_add(1, Ordering::SeqCst)
            .wrapping_add(1);
        if let Some((_, stale)) = self
            .cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id.to_string(), (generation, token))
        {
            stale.cancel();
        }
        Some(generation)
    }

    pub async fn get_agent(&self, id: &str) -> Option<Arc<Mutex<Agent>>> {
        self.sessions.lock().await.get(id).map(|s| s.agent.clone())
    }

    /// Snapshot one exact live generation for a lifecycle operation. Callers
    /// must use its generation for removal; an ID can name a later reconnect.
    pub(crate) async fn live_session(&self, id: &str) -> Option<LiveRpcSession> {
        let sessions = self.sessions.lock().await;
        let session = sessions.get(id)?;
        Some(LiveRpcSession {
            agent_alias: session.agent_alias.clone(),
            workspace_dir: session.workspace_dir.clone(),
            chat_mode: session.chat_mode.clone(),
            owner_tui_id: session.owner_tui_id.clone(),
            generation: session.generation,
            agent: Arc::clone(&session.agent),
        })
    }

    /// Snapshot reconnect identity and history while the exact live session is
    /// still present. `session/new` must not validate an unversioned identity
    /// and then turn a concurrent removal into a successful empty reconnect.
    pub(crate) async fn reconnect_snapshot(&self, id: &str) -> Option<(LiveRpcSession, usize)> {
        let (live, agent) = {
            let sessions = self.sessions.lock().await;
            let session = sessions.get(id)?;
            (
                LiveRpcSession {
                    agent_alias: session.agent_alias.clone(),
                    workspace_dir: session.workspace_dir.clone(),
                    chat_mode: session.chat_mode.clone(),
                    owner_tui_id: session.owner_tui_id.clone(),
                    generation: session.generation,
                    agent: Arc::clone(&session.agent),
                },
                Arc::clone(&session.agent),
            )
        };
        let history_len = agent.lock().await.history().len();
        self.sessions
            .lock()
            .await
            .get(id)
            .is_some_and(|session| session.generation == live.generation)
            .then_some((live, history_len))
    }

    pub(crate) async fn lock_model_provider_update(
        &self,
        id: &str,
    ) -> Option<tokio::sync::OwnedMutexGuard<()>> {
        let lock = self
            .sessions
            .lock()
            .await
            .get(id)
            .map(|session| Arc::clone(&session.model_provider_update))?;
        #[cfg(test)]
        {
            let waiting = Arc::clone(&self.model_provider_update_waiting);
            let mut lock = Box::pin(lock.lock_owned());
            let mut notified = false;
            return Some(
                std::future::poll_fn(move |cx| {
                    match std::future::Future::poll(lock.as_mut(), cx) {
                        std::task::Poll::Pending => {
                            if !notified {
                                waiting.notify_one();
                                notified = true;
                            }
                            std::task::Poll::Pending
                        }
                        std::task::Poll::Ready(guard) => std::task::Poll::Ready(guard),
                    }
                })
                .await,
            );
        }
        #[cfg(not(test))]
        Some(lock.lock_owned().await)
    }

    #[cfg(test)]
    pub(crate) fn model_provider_update_waiting(&self) -> Arc<tokio::sync::Notify> {
        Arc::clone(&self.model_provider_update_waiting)
    }

    pub async fn touch(&self, id: &str) {
        if let Some(s) = self.sessions.lock().await.get_mut(id) {
            s.last_active = Instant::now();
        }
    }

    pub async fn set_overrides(
        &self,
        id: &str,
        patch: SessionOverrides,
    ) -> Option<SessionOverrides> {
        let merged = self.preview_overrides(id, &patch).await?;
        let mut sessions = self.sessions.lock().await;
        let session = sessions.get_mut(id)?;
        session.overrides = merged.clone();
        // Apply to agent immediately.
        let overrides = session.overrides.clone();
        let agent = session.agent.clone();
        drop(sessions);
        let mut guard = agent.lock().await;
        if let Some(ref m) = overrides.model {
            guard.set_model_name(m.clone());
        }
        if overrides.temperature.is_some() {
            guard.set_temperature(overrides.temperature);
        }
        Some(overrides)
    }

    pub async fn preview_overrides(
        &self,
        id: &str,
        patch: &SessionOverrides,
    ) -> Option<SessionOverrides> {
        let sessions = self.sessions.lock().await;
        let session = sessions.get(id)?;
        let mut merged = session.overrides.clone();
        if let Some(ref m) = patch.model {
            merged.model = Some(m.clone());
        }
        if let Some(ref p) = patch.model_provider {
            merged.model_provider = Some(p.clone());
            // A provider switch without an explicit model must not carry the
            // previous provider's model forward (e.g. switching to an Ollama
            // alias while a Claude model override lingers). Clear it so the
            // dispatcher resolves the new alias's configured model.
            if patch.model.is_none() {
                merged.model = None;
            }
        }
        if let Some(t) = patch.temperature {
            merged.temperature = Some(t);
        }
        Some(merged)
    }

    /// Swap a freshly built `ModelProvider` box (and its name) onto the
    /// session's agent. Called by the dispatcher after it constructs the
    /// box from config, keeping model_provider-build logic out of the store.
    pub async fn apply_model_provider(
        &self,
        id: &str,
        model_provider: Box<dyn ModelProvider>,
        model_provider_name: String,
        model_name: String,
        tool_dispatcher: Box<dyn ToolDispatcher>,
    ) -> bool {
        let agent = {
            let sessions = self.sessions.lock().await;
            match sessions.get(id) {
                Some(s) => s.agent.clone(),
                None => return false,
            }
        };
        let mut guard = agent.lock().await;
        guard.set_model_provider(model_provider);
        guard.set_model_provider_name(model_provider_name);
        guard.set_model_name(model_name);
        guard.set_tool_dispatcher(tool_dispatcher);
        true
    }

    pub async fn get_overrides(&self, id: &str) -> Option<SessionOverrides> {
        self.sessions
            .lock()
            .await
            .get(id)
            .map(|s| s.overrides.clone())
    }

    /// Look up an existing upload by ref_id. Returns `None` if the session
    /// or entry doesn't exist.
    pub async fn get_upload(&self, session_id: &str, ref_id: &str) -> Option<UploadEntry> {
        self.sessions
            .lock()
            .await
            .get(session_id)
            .and_then(|s| s.uploads.get(ref_id).cloned())
    }

    /// Insert (or overwrite) an upload entry in the session's index.
    pub async fn insert_upload(&self, session_id: &str, entry: UploadEntry) {
        if let Some(s) = self.sessions.lock().await.get_mut(session_id) {
            s.uploads.insert(entry.ref_id.clone(), entry);
        }
    }

    /// Get the workspace directory for a session.
    pub async fn get_workspace_dir(&self, session_id: &str) -> Option<String> {
        self.sessions
            .lock()
            .await
            .get(session_id)
            .map(|s| s.workspace_dir.clone())
    }

    /// Get the agent alias bound to a session, if known. Used by the
    /// dispatcher to route uploads to the agent's own workspace dir
    /// rather than to the user's session cwd (which is often a git
    /// repo we shouldn't be writing into).
    pub async fn get_agent_alias(&self, session_id: &str) -> Option<String> {
        self.sessions
            .lock()
            .await
            .get(session_id)
            .map(|s| s.agent_alias.clone())
    }

    pub async fn seed_history(&self, id: &str, msgs: &[zeroclaw_api::model_provider::ChatMessage]) {
        let _ = self.seed_history_with_event(id, msgs).await;
    }

    pub async fn seed_history_with_event(
        &self,
        id: &str,
        msgs: &[zeroclaw_api::model_provider::ChatMessage],
    ) -> Option<TurnEvent> {
        if let Some(s) = self.sessions.lock().await.get(id) {
            s.agent.lock().await.seed_history_with_event(msgs)
        } else {
            None
        }
    }

    /// Replace the session's execution plan wholesale (TodoWrite
    /// whole-list semantics). No-op if the session is unknown.
    pub async fn set_plan(&self, id: &str, entries: Vec<PlanEntry>) {
        if let Some(s) = self.sessions.lock().await.get_mut(id) {
            s.plan = entries;
        }
    }

    /// Current stored plan for a session. `None` if the session is
    /// unknown; `Some(empty)` if the session exists with no/cleared plan.
    pub async fn get_plan(&self, id: &str) -> Option<Vec<PlanEntry>> {
        self.sessions.lock().await.get(id).map(|s| s.plan.clone())
    }

    pub async fn seed_conversation_history(
        &self,
        id: &str,
        msgs: Vec<zeroclaw_api::model_provider::ConversationMessage>,
    ) {
        let _ = self.seed_conversation_history_with_event(id, msgs).await;
    }

    pub async fn seed_conversation_history_with_event(
        &self,
        id: &str,
        msgs: Vec<zeroclaw_api::model_provider::ConversationMessage>,
    ) -> Option<TurnEvent> {
        if let Some(s) = self.sessions.lock().await.get(id) {
            s.agent
                .lock()
                .await
                .seed_conversation_history_with_event(msgs)
        } else {
            None
        }
    }

    pub async fn chat_mode(&self, id: &str) -> Option<crate::rpc::types::ChatMode> {
        self.sessions
            .lock()
            .await
            .get(id)
            .map(|s| s.chat_mode.clone())
    }

    pub async fn history_len(&self, id: &str) -> Option<usize> {
        let sessions = self.sessions.lock().await;
        let s = sessions.get(id)?;
        Some(s.agent.lock().await.history().len())
    }

    pub async fn history_slice_from(
        &self,
        id: &str,
        from: usize,
    ) -> Option<Vec<zeroclaw_api::model_provider::ConversationMessage>> {
        let sessions = self.sessions.lock().await;
        let s = sessions.get(id)?;
        let h = s.agent.lock().await;
        // Saturate: `trim_history` can shift indices past `from` between polls.
        let history = h.history();
        Some(history[from.min(history.len())..].to_vec())
    }

    pub async fn remove(&self, id: &str) -> bool {
        if let Some((_, token)) = self
            .cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(id)
        {
            self.record_cancel_cause(id, CancelCause::SessionRemoved);
            token.cancel();
        }
        self.sessions.lock().await.remove(id).is_some()
    }

    /// Remove only the captured live generation. A same-ID reconnect that
    /// wins during Goal fencing must remain reachable.
    pub(crate) async fn remove_exact(&self, id: &str, generation: u64) -> bool {
        let mut sessions = self.sessions.lock().await;
        if sessions
            .get(id)
            .is_none_or(|session| session.generation != generation)
        {
            return false;
        }
        if let Some((_, token)) = self
            .cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(id)
        {
            self.record_cancel_cause(id, CancelCause::SessionRemoved);
            token.cancel();
        }
        sessions.remove(id).is_some()
    }

    /// Read the `owner_tui_id` stamp from a session. Returns `None` if the
    /// session doesn't exist, `Some(None)` if it exists but is unowned (e.g.
    /// created by an anonymous connection), `Some(Some(id))` if owned by `id`.
    pub async fn session_owner_tui_id(&self, session_id: &str) -> Option<Option<String>> {
        let sessions = self.sessions.lock().await;
        sessions.get(session_id).map(|s| s.owner_tui_id.clone())
    }

    pub async fn list_ids(&self) -> Vec<String> {
        self.sessions.lock().await.keys().cloned().collect()
    }

    pub fn register_cancel_token(
        &self,
        id: &str,
        token: tokio_util::sync::CancellationToken,
    ) -> u64 {
        let generation = self
            .cancel_generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            .wrapping_add(1);
        if let Some((_, stale)) = self
            .cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id.to_string(), (generation, token))
        {
            stale.cancel();
        }
        generation
    }

    pub fn remove_cancel_token(&self, id: &str, generation: u64) {
        {
            let mut tokens = self.cancel_tokens.lock().unwrap_or_else(|e| e.into_inner());
            match tokens.get(id) {
                Some((g, _)) if *g == generation => {
                    tokens.remove(id);
                }
                _ => return,
            }
        }
        self.cancel_causes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(id);
    }

    pub fn cancel_session(&self, id: &str) -> bool {
        self.record_cancel_cause(id, CancelCause::ClientRpc);
        self.cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
            .map(|(_, t)| {
                t.cancel();
                true
            })
            .unwrap_or(false)
    }

    /// Returns true if a cancel token is registered — i.e. a turn is in flight.
    pub fn has_inflight_turn(&self, id: &str) -> bool {
        self.cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(id)
    }

    pub async fn kill_session(&self, id: &str) -> bool {
        if let Some((_, token)) = self
            .cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(id)
        {
            self.record_cancel_cause(id, CancelCause::AdminKill);
            token.cancel();
        }
        self.sessions.lock().await.remove(id).is_some()
    }

    /// Admin-kill only the captured live generation.
    pub(crate) async fn kill_session_exact(&self, id: &str, generation: u64) -> bool {
        let mut sessions = self.sessions.lock().await;
        if sessions
            .get(id)
            .is_none_or(|session| session.generation != generation)
        {
            return false;
        }
        if let Some((_, token)) = self
            .cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(id)
        {
            self.record_cancel_cause(id, CancelCause::AdminKill);
            token.cancel();
        }
        sessions.remove(id).is_some()
    }

    /// Record the cause for an imminent cancel-token fire. Call immediately
    /// before firing so the verdict site can attribute the cancel.
    pub fn record_cancel_cause(&self, id: &str, cause: CancelCause) {
        self.cancel_causes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id.to_string(), cause);
    }

    /// Drain the recorded cancel cause for a session. Returns `None` only
    /// when no cancel actually fired (clean completion); every firing path
    /// records before `token.cancel()`, so `Some(_)` after a fired token is
    /// the invariant the verdict audit relies on.
    pub fn take_cancel_cause(&self, id: &str) -> Option<CancelCause> {
        self.cancel_causes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(id)
    }

    pub async fn count(&self) -> usize {
        self.sessions.lock().await.len()
    }

    /// Count active sessions grouped by agent alias.
    pub async fn count_by_agent(&self) -> HashMap<String, usize> {
        let sessions = self.sessions.lock().await;
        let mut counts: HashMap<String, usize> = HashMap::new();
        for session in sessions.values() {
            *counts.entry(session.agent_alias.clone()).or_insert(0) += 1;
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store(max: usize) -> SessionStore {
        SessionStore::new(max, Arc::new(SessionActorQueue::new(4, 10, 60)))
    }

    fn make_agent() -> Agent {
        use crate::agent::dispatcher::NativeToolDispatcher;
        use crate::observability::NoopObserver;

        let mem_cfg = zeroclaw_config::schema::MemoryConfig {
            backend: "none".into(),
            ..zeroclaw_config::schema::MemoryConfig::default()
        };
        let mem = Arc::from(
            zeroclaw_memory::create_memory(&mem_cfg, &std::env::temp_dir(), None).unwrap(),
        );

        Agent::builder()
            .model_provider(Box::new(StubProvider))
            .tools(vec![])
            .memory(mem)
            .observer(Arc::new(NoopObserver {}) as Arc<dyn crate::observability::Observer>)
            .tool_dispatcher(Box::new(NativeToolDispatcher))
            .workspace_dir(std::env::temp_dir())
            .build()
            .unwrap()
    }

    /// Minimal provider that satisfies the builder. Never called in these tests.
    struct StubProvider;

    #[async_trait::async_trait]
    impl zeroclaw_providers::ModelProvider for StubProvider {
        async fn chat_with_system(
            &self,
            _: Option<&str>,
            _: &str,
            _: &str,
            _: Option<f64>,
        ) -> anyhow::Result<String> {
            Ok(String::new())
        }
        async fn chat(
            &self,
            _: zeroclaw_providers::ChatRequest<'_>,
            _: &str,
            _: Option<f64>,
        ) -> anyhow::Result<zeroclaw_providers::ChatResponse> {
            Ok(zeroclaw_providers::ChatResponse {
                text: Some("stub".into()),
                tool_calls: vec![],
                usage: None,
                reasoning_content: None,
            })
        }
    }
    impl zeroclaw_api::attribution::Attributable for StubProvider {
        fn role(&self) -> zeroclaw_api::attribution::Role {
            zeroclaw_api::attribution::Role::Provider(
                zeroclaw_api::attribution::ProviderKind::Model(
                    zeroclaw_api::attribution::ModelProviderKind::Custom,
                ),
            )
        }
        fn alias(&self) -> &str {
            "stub"
        }
    }

    #[tokio::test]
    async fn insert_and_count() {
        let store = make_store(4);
        assert_eq!(store.count().await, 0);

        store
            .insert(
                "s1".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();
        assert_eq!(store.count().await, 1);
    }

    #[tokio::test]
    async fn set_and_get_plan_round_trips() {
        use zeroclaw_api::plan::{PlanEntry, PlanPriority, PlanStatus};
        fn sample_entry(content: &str, status: PlanStatus) -> PlanEntry {
            PlanEntry {
                content: content.to_string(),
                status,
                priority: PlanPriority::Medium,
                active_form: None,
            }
        }

        let store = make_store(4);
        store
            .insert(
                "s1".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();

        // Fresh session has an empty (but present) plan.
        assert!(store.get_plan("s1").await.unwrap().is_empty());

        let plan = vec![
            sample_entry("A", PlanStatus::Completed),
            sample_entry("B", PlanStatus::InProgress),
        ];
        store.set_plan("s1", plan.clone()).await;
        assert_eq!(store.get_plan("s1").await.unwrap(), plan);

        // Whole-list replace: empty list clears.
        store.set_plan("s1", vec![]).await;
        assert!(store.get_plan("s1").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn get_plan_none_for_unknown_session() {
        let store = make_store(4);
        assert!(store.get_plan("does-not-exist").await.is_none());
    }

    #[tokio::test]
    async fn insert_rejects_over_limit() {
        let store = make_store(1);
        store
            .insert(
                "s1".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();
        let err = store
            .insert(
                "s2".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn get_agent_returns_arc() {
        let store = make_store(4);
        store
            .insert(
                "s1".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();
        assert!(store.get_agent("s1").await.is_some());
        assert!(store.get_agent("nonexistent").await.is_none());
    }

    #[tokio::test]
    async fn set_overrides_applies_model_and_temperature_live() {
        let store = make_store(4);
        store
            .insert(
                "s1".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();

        let merged = store
            .set_overrides(
                "s1",
                SessionOverrides {
                    model: Some("model-x".into()),
                    temperature: Some(0.42),
                    ..Default::default()
                },
            )
            .await
            .expect("session exists");
        assert_eq!(merged.model.as_deref(), Some("model-x"));
        assert_eq!(merged.temperature, Some(0.42));

        // The override is applied to the live agent immediately.
        let agent = store.get_agent("s1").await.unwrap();
        let (_, _, model_name) = agent.lock().await.attribution_fields();
        assert_eq!(model_name, "model-x");
    }

    #[tokio::test]
    async fn set_overrides_records_model_provider_without_rebuilding() {
        // The store records the model_provider override but does NOT rebuild the
        // provider box — that is the dispatcher's job (needs Config). Here we
        // only assert the field round-trips through the merge.
        let store = make_store(4);
        store
            .insert(
                "s1".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();

        let merged = store
            .set_overrides(
                "s1",
                SessionOverrides {
                    model_provider: Some("anthropic.default".into()),
                    ..Default::default()
                },
            )
            .await
            .expect("session exists");
        assert_eq!(merged.model_provider.as_deref(), Some("anthropic.default"));
    }

    #[tokio::test]
    async fn provider_switch_without_model_clears_prior_model() {
        // Switching provider with no explicit model must drop the prior
        // model override so the dispatcher resolves the new alias's
        // configured model (e.g. Ollama alias must not keep a Claude model).
        let store = make_store(4);
        store
            .insert(
                "s1".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();
        store
            .set_overrides(
                "s1",
                SessionOverrides {
                    model: Some("claude-opus-4-5".into()),
                    ..Default::default()
                },
            )
            .await
            .expect("session exists");
        let merged = store
            .set_overrides(
                "s1",
                SessionOverrides {
                    model_provider: Some("ollama.default".into()),
                    ..Default::default()
                },
            )
            .await
            .expect("session exists");
        assert_eq!(merged.model_provider.as_deref(), Some("ollama.default"));
        assert_eq!(
            merged.model, None,
            "a provider-only switch must clear the lingering model override"
        );
    }

    #[tokio::test]
    async fn set_overrides_missing_session_is_none() {
        let store = make_store(4);
        assert!(
            store
                .set_overrides("ghost", SessionOverrides::default())
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn remove_cleans_up() {
        let store = make_store(4);
        store
            .insert(
                "s1".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();

        let token = tokio_util::sync::CancellationToken::new();
        store.register_cancel_token("s1", token.clone());

        assert!(store.remove("s1").await);
        assert_eq!(store.count().await, 0);
        // Cancel token was also removed -- cancelling is a no-op now.
        assert!(!store.cancel_session("s1"));
    }

    #[tokio::test]
    async fn remove_nonexistent_returns_false() {
        let store = make_store(4);
        assert!(!store.remove("ghost").await);
    }

    #[tokio::test]
    async fn replacement_reservation_fences_the_exact_idle_generation_and_capacity() {
        use crate::rpc::types::ChatMode;

        let store = make_store(3);
        store
            .insert(
                "old".into(),
                RpcSession::new(make_agent(), "a", ".", ChatMode::Chat)
                    .with_owner(Some("tui".to_string())),
            )
            .await
            .expect("insert old session");

        let reservation = store
            .reserve_idle_same_mode_replacement("tui", &ChatMode::Chat, "new")
            .await
            .expect("reserve exact old generation");
        assert!(store.is_replacement_reserved("old").await);
        assert!(
            store
                .try_register_cancel_token("old", tokio_util::sync::CancellationToken::new())
                .await
                .is_none(),
            "a reserved session cannot begin a late ordinary turn"
        );
        store
            .insert(
                "unrelated".into(),
                RpcSession::new(make_agent(), "a", ".", ChatMode::Chat),
            )
            .await
            .expect("an exact replacement reservation must not block unrelated sessions");
        assert!(
            store
                .insert(
                    "new".into(),
                    RpcSession::new(make_agent(), "a", ".", ChatMode::Chat),
                )
                .await
                .is_err(),
            "only the exact incoming replacement identity is reserved"
        );

        let evicted = store
            .commit_replacement(
                &reservation,
                "new".into(),
                RpcSession::new(make_agent(), "a", ".", ChatMode::Chat)
                    .with_owner(Some("tui".to_string())),
            )
            .await
            .expect("commit exact replacement");
        assert_eq!(
            evicted.iter().map(|(id, _)| id).collect::<Vec<_>>(),
            vec!["old"]
        );
        assert!(store.get_agent("old").await.is_none());
        assert!(store.get_agent("new").await.is_some());
    }

    #[tokio::test]
    async fn dropped_replacement_reservation_releases_exact_targets_and_incoming_id() {
        use crate::rpc::types::ChatMode;

        let store = make_store(3);
        store
            .insert(
                "old".into(),
                RpcSession::new(make_agent(), "a", ".", ChatMode::Chat)
                    .with_owner(Some("tui".to_string())),
            )
            .await
            .expect("insert old session");

        let reservation = store
            .reserve_idle_same_mode_replacement("tui", &ChatMode::Chat, "incoming")
            .await
            .expect("reserve old session");
        drop(reservation);

        assert!(
            !store.is_replacement_reserved("old").await,
            "dropping a cancelled replacement must release its exact target"
        );
        let generation = store
            .try_register_cancel_token("old", tokio_util::sync::CancellationToken::new())
            .await
            .expect("the released session may accept a later ordinary turn");
        store.remove_cancel_token("old", generation);
        store
            .insert(
                "incoming".into(),
                RpcSession::new(make_agent(), "a", ".", ChatMode::Chat),
            )
            .await
            .expect("dropping a cancelled replacement must release its incoming identity");
    }

    #[tokio::test]
    async fn cancel_token_lifecycle() {
        let store = make_store(4);
        let token = tokio_util::sync::CancellationToken::new();
        let generation = store.register_cancel_token("s1", token.clone());

        assert!(!token.is_cancelled());
        assert!(store.cancel_session("s1"));
        assert!(token.is_cancelled());

        // Second cancel returns false (token was consumed by remove).
        store.remove_cancel_token("s1", generation);
        assert!(!store.cancel_session("s1"));
    }

    #[tokio::test]
    async fn reregister_force_cancels_prior_turn() {
        let store = make_store(4);
        let old = tokio_util::sync::CancellationToken::new();
        let old_gen = store.register_cancel_token("s", old.clone());

        let new = tokio_util::sync::CancellationToken::new();
        let new_gen = store.register_cancel_token("s", new.clone());

        assert!(old.is_cancelled(), "re-register must kill the prior turn");
        assert!(!new.is_cancelled());
        assert_ne!(old_gen, new_gen);

        store.remove_cancel_token("s", old_gen);
        assert!(
            store.cancel_session("s"),
            "stale-generation remove must not orphan the live turn's token"
        );
        assert!(new.is_cancelled());
    }

    #[tokio::test]
    async fn stale_remove_is_a_noop() {
        let store = make_store(4);
        let token = tokio_util::sync::CancellationToken::new();
        let generation = store.register_cancel_token("s", token.clone());
        store.remove_cancel_token("s", generation.wrapping_sub(1));
        assert!(
            store.cancel_session("s"),
            "a remove with a non-matching generation must leave the token intact"
        );
    }

    #[tokio::test]
    async fn stale_exact_lifecycle_removal_preserves_same_id_successor() {
        use crate::rpc::types::ChatMode;

        let store = make_store(4);
        store
            .insert(
                "shared".into(),
                RpcSession::new(make_agent(), "old-agent", ".", ChatMode::Chat),
            )
            .await
            .expect("insert original session");
        let original = store
            .live_session("shared")
            .await
            .expect("snapshot original generation");

        // Model the gap while a Goal disposal fence awaits quiescence: the
        // identifier is reused for a later live session before the old close
        // or kill reaches the session store.
        store
            .insert(
                "shared".into(),
                RpcSession::new(make_agent(), "successor-agent", ".", ChatMode::Chat),
            )
            .await
            .expect("insert same-id successor");

        assert!(
            !store.remove_exact("shared", original.generation()).await,
            "a stale close must not remove a successor"
        );
        assert!(
            !store
                .kill_session_exact("shared", original.generation())
                .await,
            "a stale admin kill must not remove a successor"
        );

        let successor = store
            .live_session("shared")
            .await
            .expect("same-id successor remains live");
        assert_ne!(successor.generation(), original.generation());
        assert_eq!(successor.agent_alias, "successor-agent");
    }

    #[tokio::test]
    async fn cancel_nonexistent_returns_false() {
        let store = make_store(4);
        assert!(!store.cancel_session("nope"));
    }

    #[tokio::test]
    async fn list_ids() {
        let store = make_store(4);
        store
            .insert(
                "b".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();
        store
            .insert(
                "a".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();
        let mut ids = store.list_ids().await;
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn touch_updates_last_active() {
        let store = make_store(4);
        store
            .insert(
                "s1".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();

        let before = { store.sessions.lock().await.get("s1").unwrap().last_active };
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        store.touch("s1").await;
        let after = { store.sessions.lock().await.get("s1").unwrap().last_active };
        assert!(after > before);
    }

    #[tokio::test]
    async fn session_persists_after_transport_disconnect() {
        let store = make_store(4);
        store
            .insert(
                "s1".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat)
                    .with_owner(Some("tui-x".to_string())),
            )
            .await
            .unwrap();
        // Simulate transport disconnect — store must still hold the session.
        assert_eq!(
            store.count().await,
            1,
            "session must survive transport disconnect"
        );
    }

    #[tokio::test]
    async fn kill_session_cancels_inflight_and_removes() {
        let store = make_store(4);
        store
            .insert(
                "live".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();
        let token = tokio_util::sync::CancellationToken::new();
        store.register_cancel_token("live", token.clone());

        let removed = store.kill_session("live").await;
        assert!(removed, "kill_session must return true for a real session");
        assert!(
            token.is_cancelled(),
            "kill_session must fire the cancel token"
        );
        assert_eq!(store.count().await, 0, "session must be removed");
    }

    #[tokio::test]
    async fn kill_session_idle_session_removed() {
        let store = make_store(4);
        store
            .insert(
                "cold".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();
        // No cancel token registered.
        let removed = store.kill_session("cold").await;
        assert!(removed);
        assert_eq!(store.count().await, 0);
    }

    #[tokio::test]
    async fn kill_session_missing_returns_false() {
        let store = make_store(4);
        assert!(!store.kill_session("ghost").await);
    }

    #[tokio::test]
    async fn kill_session_cause_is_admin_kill() {
        let store = make_store(4);
        store
            .insert(
                "s".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();
        let token = tokio_util::sync::CancellationToken::new();
        store.register_cancel_token("s", token.clone());

        store.kill_session("s").await;
        // The verdict site must see AdminKill, not None.
        assert_eq!(
            store.take_cancel_cause("s"),
            Some(CancelCause::AdminKill),
            "kill_session must preserve AdminKill cause for verdict-site attribution"
        );
    }
}
