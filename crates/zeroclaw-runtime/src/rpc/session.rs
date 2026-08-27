//! RPC session state.

use crate::agent::agent::{Agent, TurnEvent};
use crate::agent::dispatcher::ToolDispatcher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Instant;
use tokio::sync::{Mutex, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};
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
    /// Monotonic identity of this live incarnation. A session ID is reusable,
    /// so lifecycle operations must fence on this value rather than on the raw
    /// string key alone.
    generation: u64,
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

pub struct SessionStore {
    sessions: Mutex<HashMap<String, RpcSession>>,
    /// Process-local lifecycle coordination keyed by session ID. Weak entries
    /// avoid retaining one lock per historical session ID, while each active
    /// admission, removal, or creation holds the strong reference needed to
    /// serialize its lifetime.
    lifecycle_gates: std::sync::Mutex<HashMap<String, Weak<RwLock<()>>>>,
    /// Exact generations whose removal has crossed the durable Goal fence but
    /// has not completed. They remain unavailable to prompt/reconnect/Goal
    /// admission until a compatible cleanup retry removes the live entry.
    removing_generations: std::sync::Mutex<HashMap<String, u64>>,
    next_session_generation: AtomicU64,
    #[cfg(test)]
    model_provider_update_waiting: Arc<tokio::sync::Notify>,
    cancel_tokens:
        std::sync::Mutex<HashMap<String, (u64, u64, tokio_util::sync::CancellationToken)>>,
    cancel_generation: std::sync::atomic::AtomicU64,
    cancel_causes: std::sync::Mutex<HashMap<(String, u64, u64), CancelCause>>,
    max_sessions: usize,
    pub session_queue: Arc<SessionActorQueue>,
}

/// Exclusive ownership of one exact live session incarnation.
///
/// The guard is held across Goal quiescence and final removal. `generation`
/// prevents an earlier lifecycle request from affecting a later session that
/// reuses the same user-visible ID.
pub(crate) struct SessionRemovalReservation {
    id: String,
    generation: u64,
    sealed: bool,
    _guard: OwnedRwLockWriteGuard<()>,
}

/// Read-side admission for one exact current session. Holding this guard
/// prevents a concurrent close, kill, or replacement from beginning while a
/// Goal control command validates and reaches the durable controller.
pub(crate) struct SessionGoalAdmission {
    pub(crate) agent_alias: String,
    _guard: OwnedRwLockReadGuard<()>,
}

/// Exclusive reservation for deleting a durable session which is not currently
/// published in the live RPC map. It prevents a concurrent reconnect or
/// rehydration from creating the same ID until shared Goal supervision has
/// classified and quiesced any worker it owns.
pub(crate) struct AbsentSessionDeletionReservation {
    _guard: OwnedRwLockWriteGuard<()>,
}

/// Exclusive reservation for constructing a previously absent live session.
/// The slot is not published until `commit_creation`, so failed preparation
/// cannot expose a half-initialized agent or require teardown hooks.
pub(crate) struct SessionCreationReservation {
    id: String,
    _guard: OwnedRwLockWriteGuard<()>,
}

pub(crate) enum SessionReconnectAdmission {
    Missing,
    Incompatible,
    Admitted(SessionReconnectLease),
}

/// Read-side lease for a same-ID reconnect. The original session, agent, and
/// Goal worker remain intact; only the RPC approval backchannel is renewed.
pub(crate) struct SessionReconnectLease {
    pub(crate) agent: Arc<Mutex<Agent>>,
    pub(crate) workspace_dir: String,
    _guard: OwnedRwLockReadGuard<()>,
}

/// Exact ownership of an ordinary in-flight turn. A completion may clear only
/// this lease; a recycled session ID or newer turn is therefore unaffected.
#[derive(Clone)]
pub(crate) struct TurnLease {
    id: String,
    session_generation: u64,
    turn_generation: u64,
    token: tokio_util::sync::CancellationToken,
}

impl TurnLease {
    pub(crate) fn token(&self) -> tokio_util::sync::CancellationToken {
        self.token.clone()
    }
}

impl SessionStore {
    pub fn new(max_sessions: usize, session_queue: Arc<SessionActorQueue>) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            lifecycle_gates: std::sync::Mutex::new(HashMap::new()),
            removing_generations: std::sync::Mutex::new(HashMap::new()),
            next_session_generation: AtomicU64::new(0),
            #[cfg(test)]
            model_provider_update_waiting: Arc::new(tokio::sync::Notify::new()),
            cancel_tokens: std::sync::Mutex::new(HashMap::new()),
            cancel_generation: std::sync::atomic::AtomicU64::new(0),
            cancel_causes: std::sync::Mutex::new(HashMap::new()),
            max_sessions,
            session_queue,
        }
    }

    fn lifecycle_gate(&self, id: &str) -> Arc<RwLock<()>> {
        let mut gates = self
            .lifecycle_gates
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        gates.retain(|_, gate| gate.strong_count() > 0);
        if let Some(gate) = gates.get(id).and_then(Weak::upgrade) {
            return gate;
        }
        let gate = Arc::new(RwLock::new(()));
        gates.insert(id.to_string(), Arc::downgrade(&gate));
        gate
    }

    fn next_generation(&self) -> Result<u64, &'static str> {
        self.next_session_generation
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                current.checked_add(1)
            })
            .map(|previous| previous + 1)
            .map_err(|_| "session generation exhausted")
    }

    fn is_removing(&self, id: &str, generation: u64) -> bool {
        self.removing_generations
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(id)
            .is_some_and(|removing| *removing == generation)
    }

    fn seal_removal(&self, reservation: &mut SessionRemovalReservation, cause: CancelCause) {
        if reservation.sealed {
            return;
        }
        self.removing_generations
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(reservation.id.clone(), reservation.generation);
        // The writer reservation excludes `begin_turn`, so cancel the exact
        // live lease before any fallible Goal disposal. A sealed session is
        // quarantined; it must not retain an ordinary model turn merely
        // because subsequent durable cleanup is slow or fails.
        let mut tokens = self
            .cancel_tokens
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let turn = tokens.remove(&reservation.id).and_then(
            |(session_generation, turn_generation, token)| {
                (session_generation == reservation.generation).then_some((turn_generation, token))
            },
        );
        if let Some((turn_generation, token)) = turn {
            // Keep the token-map lock through the verdict insertion. The
            // ordinary completion path takes these locks in the same order,
            // so it cannot observe a removed lease before its cause exists.
            self.cancel_causes
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .insert(
                    (
                        reservation.id.clone(),
                        reservation.generation,
                        turn_generation,
                    ),
                    cause,
                );
            drop(tokens);
            token.cancel();
        } else {
            drop(tokens);
        }
        reservation.sealed = true;
    }

    pub async fn insert(&self, id: String, mut session: RpcSession) -> Result<(), &'static str> {
        let _gate = self.lifecycle_gate(&id).write_owned().await;
        let mut sessions = self.sessions.lock().await;
        if sessions.len() >= self.max_sessions {
            return Err("session limit reached");
        }
        if sessions.contains_key(&id) {
            return Err("session already exists");
        }
        session.generation = self.next_generation()?;
        sessions.insert(id, session);
        Ok(())
    }

    pub(crate) async fn reserve_creation(
        &self,
        id: &str,
    ) -> Result<SessionCreationReservation, &'static str> {
        let gate = self.lifecycle_gate(id);
        let guard = gate.write_owned().await;
        if self.sessions.lock().await.contains_key(id) {
            return Err("session already exists");
        }
        Ok(SessionCreationReservation {
            id: id.to_string(),
            _guard: guard,
        })
    }

    pub(crate) async fn commit_creation(
        &self,
        reservation: SessionCreationReservation,
        mut session: RpcSession,
    ) -> Result<(), &'static str> {
        let mut sessions = self.sessions.lock().await;
        if sessions.len() >= self.max_sessions {
            return Err("session limit reached");
        }
        if sessions.contains_key(&reservation.id) {
            return Err("session already exists");
        }
        session.generation = self.next_generation()?;
        sessions.insert(reservation.id, session);
        Ok(())
    }

    pub async fn get_agent(&self, id: &str) -> Option<Arc<Mutex<Agent>>> {
        let sessions = self.sessions.lock().await;
        let session = sessions.get(id)?;
        (!self.is_removing(id, session.generation)).then(|| Arc::clone(&session.agent))
    }

    /// Snapshot the agent held by an exact removal reservation. This is only
    /// for teardown work such as unregistering the RPC approval channel:
    /// regular admission must continue to use `get_agent`, which rejects a
    /// sealed generation.
    pub(crate) async fn agent_for_removal(
        &self,
        reservation: &SessionRemovalReservation,
    ) -> Option<Arc<Mutex<Agent>>> {
        self.sessions
            .lock()
            .await
            .get(&reservation.id)
            .filter(|session| session.generation == reservation.generation)
            .map(|session| Arc::clone(&session.agent))
    }

    /// Reserve exclusive lifecycle ownership of the current session. The
    /// caller must either remove that exact generation or drop the reservation
    /// without mutation. No later session with the same ID can enter while
    /// this guard is held.
    pub(crate) async fn reserve_removal(&self, id: &str) -> Option<SessionRemovalReservation> {
        let gate = self.lifecycle_gate(id);
        let guard = gate.write_owned().await;
        let generation = self.sessions.lock().await.get(id)?.generation;
        Some(SessionRemovalReservation {
            id: id.to_string(),
            generation,
            sealed: false,
            _guard: guard,
        })
    }

    pub(crate) async fn reserve_absent_deletion(
        &self,
        id: &str,
    ) -> Option<AbsentSessionDeletionReservation> {
        let gate = self.lifecycle_gate(id);
        let guard = gate.write_owned().await;
        if self.sessions.lock().await.contains_key(id) {
            return None;
        }
        Some(AbsentSessionDeletionReservation { _guard: guard })
    }

    /// Reserve an idle sibling for same-mode replacement. Idle is checked
    /// after acquiring the lifecycle writer, not only during candidate
    /// discovery, so a prompt that begins in between cannot be evicted.
    pub(crate) async fn reserve_idle_same_mode_sibling(
        &self,
        id: &str,
        tui_id: &str,
        chat_mode: &crate::rpc::types::ChatMode,
    ) -> Option<SessionRemovalReservation> {
        let gate = self.lifecycle_gate(id);
        let guard = gate.write_owned().await;
        let generation = {
            let sessions = self.sessions.lock().await;
            let session = sessions.get(id)?;
            if session.owner_tui_id.as_deref() != Some(tui_id) || &session.chat_mode != chat_mode {
                return None;
            }
            session.generation
        };
        if self
            .cancel_tokens
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .contains_key(id)
        {
            return None;
        }
        Some(SessionRemovalReservation {
            id: id.to_string(),
            generation,
            sealed: false,
            _guard: guard,
        })
    }

    pub(crate) async fn admit_goal(&self, id: &str) -> Option<SessionGoalAdmission> {
        let gate = self.lifecycle_gate(id);
        let guard = gate.read_owned().await;
        let (generation, agent_alias) = {
            let sessions = self.sessions.lock().await;
            let session = sessions.get(id)?;
            (session.generation, session.agent_alias.clone())
        };
        if self.is_removing(id, generation) {
            return None;
        }
        Some(SessionGoalAdmission {
            agent_alias,
            _guard: guard,
        })
    }

    pub(crate) async fn admit_reconnect(
        &self,
        id: &str,
        agent_alias: &str,
        chat_mode: &crate::rpc::types::ChatMode,
        owner_tui_id: Option<&str>,
    ) -> SessionReconnectAdmission {
        let gate = self.lifecycle_gate(id);
        let guard = gate.read_owned().await;
        let sessions = self.sessions.lock().await;
        let Some(session) = sessions.get(id) else {
            return SessionReconnectAdmission::Missing;
        };
        if self.is_removing(id, session.generation) {
            return SessionReconnectAdmission::Incompatible;
        }
        if session.agent_alias != agent_alias
            || &session.chat_mode != chat_mode
            || session.owner_tui_id.as_deref() != owner_tui_id
        {
            return SessionReconnectAdmission::Incompatible;
        }
        SessionReconnectAdmission::Admitted(SessionReconnectLease {
            agent: Arc::clone(&session.agent),
            workspace_dir: session.workspace_dir.clone(),
            _guard: guard,
        })
    }

    pub(crate) async fn remove_reserved(
        &self,
        reservation: SessionRemovalReservation,
        cause: CancelCause,
    ) -> bool {
        let id = reservation.id;
        let generation = reservation.generation;
        let removed = {
            let mut sessions = self.sessions.lock().await;
            if sessions.get(&id).map(|session| session.generation) != Some(generation) {
                return false;
            }
            sessions.remove(&id).is_some()
        };
        if removed {
            self.removing_generations
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .remove(&id);
            let turn = self
                .cancel_tokens
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .remove(&id)
                .and_then(|(token_generation, turn_generation, token)| {
                    (token_generation == generation).then_some((turn_generation, token))
                });
            if let Some((turn_generation, token)) = turn {
                self.record_cancel_cause(&id, generation, turn_generation, cause);
                token.cancel();
            }
        }
        removed
    }

    pub(crate) fn seal_removal_reservation(
        &self,
        reservation: &mut SessionRemovalReservation,
        cause: CancelCause,
    ) {
        self.seal_removal(reservation, cause);
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
        let Some(reservation) = self.reserve_removal(id).await else {
            return false;
        };
        self.remove_reserved(reservation, CancelCause::SessionRemoved)
            .await
    }

    /// List idle same-owner sessions eligible for replacement. The dispatcher
    /// performs Goal disposal and the exact removal under each session's
    /// lifecycle gate; this store must not drop a Goal supervisor directly.
    pub async fn same_mode_evictable_siblings(
        &self,
        tui_id: &str,
        chat_mode: &crate::rpc::types::ChatMode,
        except_id: &str,
    ) -> Vec<(String, String)> {
        let in_flight: std::collections::HashSet<String> = self
            .cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect();
        let sessions = self.sessions.lock().await;
        sessions
            .iter()
            .filter(|(key, s)| {
                key.as_str() != except_id
                    && s.owner_tui_id.as_deref() == Some(tui_id)
                    && &s.chat_mode == chat_mode
                    && !in_flight.contains(key.as_str())
            })
            .map(|(key, session)| (key.clone(), session.agent_alias.clone()))
            .collect()
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

    pub(crate) async fn begin_turn(&self, id: &str) -> Option<TurnLease> {
        // Synchronize admission with replacement/removal. The read guard only
        // spans installation of the exact turn lease; the model execution is
        // governed separately by the session actor queue.
        let _lifecycle = self.lifecycle_gate(id).read_owned().await;
        let session_generation = self.sessions.lock().await.get(id)?.generation;
        if self.is_removing(id, session_generation) {
            return None;
        }
        let token = tokio_util::sync::CancellationToken::new();
        let generation = self
            .cancel_generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            .wrapping_add(1);
        if let Some((stale_session, stale_turn, stale)) = self
            .cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(
                id.to_string(),
                (session_generation, generation, token.clone()),
            )
        {
            self.record_cancel_cause(id, stale_session, stale_turn, CancelCause::ClientRpc);
            stale.cancel();
        }
        Some(TurnLease {
            id: id.to_string(),
            session_generation,
            turn_generation: generation,
            token,
        })
    }

    pub(crate) fn finish_turn(&self, lease: &TurnLease) -> Option<CancelCause> {
        let mut tokens = self.cancel_tokens.lock().unwrap_or_else(|e| e.into_inner());
        let matches = matches!(
            tokens.get(&lease.id),
            Some((session_generation, turn_generation, _))
                if *session_generation == lease.session_generation
                    && *turn_generation == lease.turn_generation
        );
        if matches {
            tokens.remove(&lease.id);
        }
        drop(tokens);
        self.cancel_causes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&(
                lease.id.clone(),
                lease.session_generation,
                lease.turn_generation,
            ))
    }

    pub fn cancel_session(&self, id: &str) -> bool {
        self.cancel_tokens
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
            .map(|(session_generation, turn_generation, t)| {
                self.record_cancel_cause(
                    id,
                    *session_generation,
                    *turn_generation,
                    CancelCause::ClientRpc,
                );
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
        let Some(reservation) = self.reserve_removal(id).await else {
            return false;
        };
        self.remove_reserved(reservation, CancelCause::AdminKill)
            .await
    }

    /// Record the cause for an imminent cancel-token fire. Call immediately
    /// before firing so the verdict site can attribute the cancel.
    fn record_cancel_cause(
        &self,
        id: &str,
        session_generation: u64,
        turn_generation: u64,
        cause: CancelCause,
    ) {
        self.cancel_causes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((id.to_string(), session_generation, turn_generation), cause);
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

        let turn = store.begin_turn("s1").await.unwrap();
        let token = turn.token();

        assert!(store.remove("s1").await);
        assert_eq!(store.count().await, 0);
        assert!(
            token.is_cancelled(),
            "removal must cancel the exact live turn"
        );
        // Cancel token was also removed -- cancelling is a no-op now.
        assert!(!store.cancel_session("s1"));
    }

    #[tokio::test]
    async fn remove_nonexistent_returns_false() {
        let store = make_store(4);
        assert!(!store.remove("ghost").await);
    }

    #[tokio::test]
    async fn replacement_waits_for_the_old_session_removal_reservation() {
        let store = Arc::new(make_store(4));
        store
            .insert(
                "session".into(),
                RpcSession::new(make_agent(), "old", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();
        let reservation = store.reserve_removal("session").await.unwrap();
        let replacement_store = Arc::clone(&store);
        let mut replacement = tokio::spawn(async move {
            replacement_store
                .insert(
                    "session".into(),
                    RpcSession::new(make_agent(), "new", ".", crate::rpc::types::ChatMode::Chat),
                )
                .await
        });
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), &mut replacement)
                .await
                .is_err(),
            "a successor must not enter while the predecessor is still being quiesced"
        );
        assert!(
            store
                .remove_reserved(reservation, CancelCause::SessionRemoved)
                .await
        );
        replacement.await.unwrap().unwrap();
        assert_eq!(
            store.get_agent_alias("session").await.as_deref(),
            Some("new")
        );
    }

    #[tokio::test]
    async fn removal_waits_for_goal_command_admission() {
        let store = Arc::new(make_store(4));
        store
            .insert(
                "session".into(),
                RpcSession::new(
                    make_agent(),
                    "agent",
                    ".",
                    crate::rpc::types::ChatMode::Chat,
                ),
            )
            .await
            .unwrap();
        let admission = store.admit_goal("session").await.unwrap();
        let removal_store = Arc::clone(&store);
        let mut removal =
            tokio::spawn(async move { removal_store.reserve_removal("session").await });

        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), &mut removal)
                .await
                .is_err(),
            "removal must wait until the admitted Goal command leaves the controller"
        );
        drop(admission);
        let reservation = removal.await.unwrap().unwrap();
        assert!(
            store
                .remove_reserved(reservation, CancelCause::SessionRemoved)
                .await
        );
    }

    #[tokio::test]
    async fn sibling_eviction_blocks_a_late_turn_before_removal() {
        use crate::rpc::types::ChatMode;

        let store = Arc::new(make_store(4));
        store
            .insert(
                "sibling".into(),
                RpcSession::new(make_agent(), "agent", ".", ChatMode::Chat)
                    .with_owner(Some("tui".into())),
            )
            .await
            .unwrap();
        let reservation = store
            .reserve_idle_same_mode_sibling("sibling", "tui", &ChatMode::Chat)
            .await
            .expect("idle sibling must be reservable");
        let turn_store = Arc::clone(&store);
        let mut turn = tokio::spawn(async move { turn_store.begin_turn("sibling").await });

        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), &mut turn)
                .await
                .is_err(),
            "a turn cannot enter after sibling eviction has fenced the session"
        );
        assert!(
            store
                .remove_reserved(reservation, CancelCause::SessionRemoved)
                .await
        );
        assert!(turn.await.unwrap().is_none());
    }

    #[tokio::test]
    async fn sealed_removal_quarantines_the_exact_generation_until_retry() {
        let store = make_store(4);
        store
            .insert(
                "session".into(),
                RpcSession::new(
                    make_agent(),
                    "agent",
                    ".",
                    crate::rpc::types::ChatMode::Chat,
                ),
            )
            .await
            .unwrap();
        let mut failed_disposal = store.reserve_removal("session").await.unwrap();
        store.seal_removal_reservation(&mut failed_disposal, CancelCause::SessionRemoved);
        drop(failed_disposal);

        assert!(store.admit_goal("session").await.is_none());
        assert!(store.begin_turn("session").await.is_none());
        assert!(store.get_agent("session").await.is_none());

        let retry = store
            .reserve_removal("session")
            .await
            .expect("sealed removal must allow compatible cleanup retry");
        assert!(
            store
                .remove_reserved(retry, CancelCause::SessionRemoved)
                .await
        );
        assert!(store.get_agent("session").await.is_none());
    }

    #[tokio::test]
    async fn sealing_removal_cancels_the_live_turn_before_cleanup() {
        let store = make_store(4);
        store
            .insert(
                "session".into(),
                RpcSession::new(
                    make_agent(),
                    "agent",
                    ".",
                    crate::rpc::types::ChatMode::Chat,
                ),
            )
            .await
            .unwrap();
        let turn = store.begin_turn("session").await.unwrap();
        let token = turn.token();

        let mut reservation = store.reserve_removal("session").await.unwrap();
        store.seal_removal_reservation(&mut reservation, CancelCause::SessionRemoved);

        assert!(
            token.is_cancelled(),
            "the exact live turn must be cancelled before fallible Goal cleanup"
        );
        assert_eq!(
            store.finish_turn(&turn),
            Some(CancelCause::SessionRemoved),
            "the fenced turn keeps its exact cancellation verdict"
        );
        drop(reservation);
    }

    #[tokio::test]
    async fn absent_delete_blocks_recreation_until_cleanup_finishes() {
        let store = Arc::new(make_store(4));
        let deletion = store.reserve_absent_deletion("session").await.unwrap();
        let insertion_store = Arc::clone(&store);
        let mut insertion = tokio::spawn(async move {
            insertion_store
                .insert(
                    "session".into(),
                    RpcSession::new(
                        make_agent(),
                        "agent",
                        ".",
                        crate::rpc::types::ChatMode::Chat,
                    ),
                )
                .await
        });

        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), &mut insertion)
                .await
                .is_err(),
            "recreation must wait while absent-session Goal cleanup owns the ID"
        );
        drop(deletion);
        insertion.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn creation_reservation_serializes_same_id_construction() {
        let store = Arc::new(make_store(4));
        let creation = store.reserve_creation("session").await.unwrap();
        let competing_store = Arc::clone(&store);
        let mut competing =
            tokio::spawn(async move { competing_store.reserve_creation("session").await });

        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), &mut competing)
                .await
                .is_err(),
            "a second same-ID construction must wait for the first creator"
        );
        store
            .commit_creation(
                creation,
                RpcSession::new(
                    make_agent(),
                    "agent",
                    ".",
                    crate::rpc::types::ChatMode::Chat,
                ),
            )
            .await
            .unwrap();
        assert!(competing.await.unwrap().is_err());
    }

    #[tokio::test]
    async fn same_mode_evictable_siblings_selects_only_same_mode_owner() {
        use crate::rpc::types::ChatMode;
        let store = make_store(8);
        let mk = |mode: ChatMode, owner: &str| {
            RpcSession::new(make_agent(), "a", ".", mode).with_owner(Some(owner.to_string()))
        };
        store
            .insert("old_chat".into(), mk(ChatMode::Chat, "tui1"))
            .await
            .unwrap();
        store
            .insert("old_code".into(), mk(ChatMode::Acp, "tui1"))
            .await
            .unwrap();
        store
            .insert("other_chat".into(), mk(ChatMode::Chat, "tui2"))
            .await
            .unwrap();
        store
            .insert("new_chat".into(), mk(ChatMode::Chat, "tui1"))
            .await
            .unwrap();

        let evicted = store
            .same_mode_evictable_siblings("tui1", &ChatMode::Chat, "new_chat")
            .await;

        let ids: Vec<&str> = evicted.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, vec!["old_chat"]);
        assert!(
            store.get_agent("new_chat").await.is_some(),
            "new session preserved"
        );
        assert!(
            store.get_agent("old_code").await.is_some(),
            "cross-mode Code session preserved"
        );
        assert!(
            store.get_agent("other_chat").await.is_some(),
            "other TUI session preserved"
        );
        assert!(
            store.get_agent("old_chat").await.is_some(),
            "dispatcher performs the gated removal after Goal disposal"
        );
    }

    #[tokio::test]
    async fn same_mode_evictable_siblings_skips_in_flight_turn() {
        use crate::rpc::types::ChatMode;
        let store = make_store(8);
        let mk = |mode: ChatMode, owner: &str| {
            RpcSession::new(make_agent(), "a", ".", mode).with_owner(Some(owner.to_string()))
        };
        store
            .insert("busy_chat".into(), mk(ChatMode::Chat, "tui1"))
            .await
            .unwrap();
        store
            .insert("new_chat".into(), mk(ChatMode::Chat, "tui1"))
            .await
            .unwrap();
        // A registered cancel token marks a turn in flight: a spawned prompt
        // task still holds an Agent clone, so this session must NOT be force
        // evicted (that is the reaper's documented mid-turn freeze).
        let turn = store.begin_turn("busy_chat").await.unwrap();
        let token = turn.token();

        let evicted = store
            .same_mode_evictable_siblings("tui1", &ChatMode::Chat, "new_chat")
            .await;

        assert!(
            evicted.is_empty(),
            "in-flight same-mode session must be left to finish its turn"
        );
        assert!(
            store.get_agent("busy_chat").await.is_some(),
            "mid-turn session preserved"
        );
        assert!(
            !token.is_cancelled(),
            "eviction must not fire a mid-turn cancel token"
        );
    }

    #[tokio::test]
    async fn cancel_token_lifecycle() {
        let store = make_store(4);
        store
            .insert(
                "s1".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();
        let turn = store.begin_turn("s1").await.unwrap();
        let token = turn.token();

        assert!(!token.is_cancelled());
        assert!(store.cancel_session("s1"));
        assert!(token.is_cancelled());

        // Second cancel returns false once this exact turn has settled.
        store.finish_turn(&turn);
        assert!(!store.cancel_session("s1"));
    }

    #[tokio::test]
    async fn reregister_force_cancels_prior_turn() {
        let store = make_store(4);
        store
            .insert(
                "s".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();
        let old_turn = store.begin_turn("s").await.unwrap();
        let old = old_turn.token();
        let new_turn = store.begin_turn("s").await.unwrap();
        let new = new_turn.token();

        assert!(old.is_cancelled(), "re-register must kill the prior turn");
        assert!(!new.is_cancelled());
        store.finish_turn(&old_turn);
        assert!(
            store.cancel_session("s"),
            "stale-generation remove must not orphan the live turn's token"
        );
        assert!(new.is_cancelled());
    }

    #[tokio::test]
    async fn stale_remove_is_a_noop() {
        let store = make_store(4);
        store
            .insert(
                "s".into(),
                RpcSession::new(make_agent(), "a", ".", crate::rpc::types::ChatMode::Chat),
            )
            .await
            .unwrap();
        let stale = store.begin_turn("s").await.unwrap();
        let current = store.begin_turn("s").await.unwrap();
        let token = current.token();
        store.finish_turn(&stale);
        assert!(
            store.cancel_session("s"),
            "a stale completion must leave the newer turn's token intact"
        );
        assert!(token.is_cancelled());
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
        let turn = store.begin_turn("live").await.unwrap();
        let token = turn.token();

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
        let turn = store.begin_turn("s").await.unwrap();
        let token = turn.token();

        store.kill_session("s").await;
        assert!(token.is_cancelled());
        // The verdict site must see AdminKill, not None.
        assert_eq!(
            store.finish_turn(&turn),
            Some(CancelCause::AdminKill),
            "kill_session must preserve AdminKill cause for verdict-site attribution"
        );
    }
}
