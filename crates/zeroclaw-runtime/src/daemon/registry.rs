use anyhow::Result;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use tokio::sync::{broadcast, watch};
use tokio_util::sync::CancellationToken;
use zeroclaw_config::schema::{Config, MqttConfig};

use super::{GatewayReadinessReporter, SocketReadinessReporter};
use crate::goal_mode::{GoalRuntime, GoalSessionAdapter};
use crate::rpc::context::RpcContext;
use crate::rpc::tui_identity::TuiRegistry;

pub type StarterFuture = Pin<Box<dyn Future<Output = Result<()>> + Send>>;

#[derive(Clone)]
pub struct GatewayReloadControls {
    pub shutdown_tx: watch::Sender<bool>,
    pub reload_tx: watch::Sender<bool>,
}

pub type GatewayStarter = Box<
    dyn Fn(
            String,
            u16,
            Config,
            Option<broadcast::Sender<Value>>,
            Option<GatewayReloadControls>,
            Option<Arc<TuiRegistry>>,
            Option<GatewayReadinessReporter>,
        ) -> StarterFuture
        + Send
        + Sync,
>;

/// Starts the supervised channel orchestrator for one daemon run/reload iteration.
pub type ChannelsStarter =
    Box<dyn Fn(Config, CancellationToken, Option<Arc<GoalRuntime>>) -> StarterFuture + Send + Sync>;

/// Reconciles retained channel-owned Goals when no channel supervisor will
/// start for this daemon generation.
pub type ChannelGoalRevoker = Box<dyn Fn(Arc<GoalRuntime>) -> StarterFuture + Send + Sync>;

/// Starts the local IPC transport and optionally reports its secured bind.
pub type SocketStarter = Box<
    dyn Fn(
            Arc<RpcContext>,
            CancellationToken,
            Arc<AtomicUsize>,
            Option<SocketReadinessReporter>,
        ) -> StarterFuture
        + Send
        + Sync,
>;

/// Starts an RPC transport using the shared daemon RPC context.
pub type RpcStarter = Box<
    dyn Fn(Arc<RpcContext>, CancellationToken, Arc<AtomicUsize>) -> StarterFuture + Send + Sync,
>;

/// Starts the MQTT SOP listener for one configured MQTT channel alias.
pub type MqttStarter = Box<dyn Fn(MqttConfig) -> StarterFuture + Send + Sync>;

#[derive(Default)]
pub struct DaemonRegistry {
    gateway_start: Option<GatewayStarter>,
    channels_start: Option<ChannelsStarter>,
    channel_goal_revoker: Option<ChannelGoalRevoker>,
    socket_start: Option<SocketStarter>,
    wss_start: Option<RpcStarter>,
    mqtt_start: Option<MqttStarter>,
    /// Immutable transport/session owners supplied before the daemon begins
    /// listener supervision for this generation.
    goal_session_adapters: Vec<Arc<dyn GoalSessionAdapter>>,
    /// Shared SOP engine built by the daemon reload loop. Passed through to
    /// RpcContext so RPC/TUI agent sessions share the same engine.
    sop_engine: Option<Arc<std::sync::Mutex<crate::sop::SopEngine>>>,
    sop_audit: Option<Arc<crate::sop::SopAuditLogger>>,
}

impl DaemonRegistry {
    /// Create an empty registry. Missing starters are treated as unwired
    /// optional subsystems by `daemon::run`.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_gateway(&mut self, starter: GatewayStarter) -> &mut Self {
        self.gateway_start = Some(starter);
        self
    }

    #[cfg(test)]
    fn has_gateway_start(&self) -> bool {
        self.gateway_start.is_some()
    }

    pub fn register_channels(&mut self, starter: ChannelsStarter) -> &mut Self {
        self.channels_start = Some(starter);
        self
    }

    pub fn register_channel_goal_revoker(&mut self, revoker: ChannelGoalRevoker) -> &mut Self {
        self.channel_goal_revoker = Some(revoker);
        self
    }

    #[cfg(test)]
    fn has_channels_start(&self) -> bool {
        self.channels_start.is_some()
    }

    pub fn register_socket(&mut self, starter: SocketStarter) -> &mut Self {
        self.socket_start = Some(starter);
        self
    }

    pub(crate) fn has_socket_start(&self) -> bool {
        self.socket_start.is_some()
    }

    pub fn register_wss(&mut self, starter: RpcStarter) -> &mut Self {
        self.wss_start = Some(starter);
        self
    }

    pub(crate) fn has_wss_start(&self) -> bool {
        self.wss_start.is_some()
    }

    pub fn register_mqtt(&mut self, starter: MqttStarter) -> &mut Self {
        self.mqtt_start = Some(starter);
        self
    }

    /// Register one non-RPC live-session owner for the daemon-generation Goal
    /// host. The daemon owns the resulting adapter router; registrations are
    /// consumed before any listener begins accepting work.
    pub fn register_goal_session_adapter(
        &mut self,
        adapter: Arc<dyn GoalSessionAdapter>,
    ) -> &mut Self {
        self.goal_session_adapters.push(adapter);
        self
    }

    #[cfg(test)]
    fn has_mqtt_start(&self) -> bool {
        self.mqtt_start.is_some()
    }

    pub(crate) fn take_gateway_start(&mut self) -> Option<GatewayStarter> {
        self.gateway_start.take()
    }

    pub(crate) fn take_channels_start(&mut self) -> Option<ChannelsStarter> {
        self.channels_start.take()
    }

    pub(crate) fn take_channel_goal_revoker(&mut self) -> Option<ChannelGoalRevoker> {
        self.channel_goal_revoker.take()
    }

    pub(crate) fn take_socket_start(&mut self) -> Option<SocketStarter> {
        self.socket_start.take()
    }

    pub(crate) fn take_wss_start(&mut self) -> Option<RpcStarter> {
        self.wss_start.take()
    }

    pub(crate) fn take_mqtt_start(&mut self) -> Option<MqttStarter> {
        self.mqtt_start.take()
    }

    pub(crate) fn take_goal_session_adapters(&mut self) -> Vec<Arc<dyn GoalSessionAdapter>> {
        std::mem::take(&mut self.goal_session_adapters)
    }

    #[cfg(test)]
    fn goal_session_adapter_count(&self) -> usize {
        self.goal_session_adapters.len()
    }

    /// Set the shared SOP engine for this daemon iteration.
    pub fn set_sop_engine(
        &mut self,
        sop_engine: Option<Arc<std::sync::Mutex<crate::sop::SopEngine>>>,
        sop_audit: Option<Arc<crate::sop::SopAuditLogger>>,
    ) -> &mut Self {
        self.sop_engine = sop_engine;
        self.sop_audit = sop_audit;
        self
    }

    pub(crate) fn take_sop_engine(
        &mut self,
    ) -> (
        Option<Arc<std::sync::Mutex<crate::sop::SopEngine>>>,
        Option<Arc<crate::sop::SopAuditLogger>>,
    ) {
        (self.sop_engine.take(), self.sop_audit.take())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal_mode::{
        GoalExecutionScope, GoalHostModelBinding, GoalSessionAdapter, GoalSessionBinding,
        GoalSessionLease,
    };
    use anyhow::bail;
    use async_trait::async_trait;

    struct TestGoalSessionAdapter;

    #[async_trait]
    impl GoalSessionAdapter for TestGoalSessionAdapter {
        fn owns(&self, _binding: &GoalSessionBinding) -> bool {
            false
        }

        async fn model_binding(
            &self,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<GoalHostModelBinding> {
            bail!("test adapter does not resolve model bindings")
        }

        async fn acquire(
            &self,
            _scope: &GoalExecutionScope,
            _binding: &GoalSessionBinding,
        ) -> anyhow::Result<Box<dyn GoalSessionLease>> {
            bail!("test adapter does not acquire leases")
        }
    }

    fn gateway_starter() -> GatewayStarter {
        Box::new(|_, _, _, _, _, _, _| Box::pin(async { Ok(()) }))
    }

    fn channels_starter() -> ChannelsStarter {
        Box::new(|_, _, _| Box::pin(async { Ok(()) }))
    }

    fn socket_starter() -> SocketStarter {
        Box::new(|_, _, _, _| Box::pin(async { Ok(()) }))
    }

    fn rpc_starter() -> RpcStarter {
        Box::new(|_, _, _| Box::pin(async { Ok(()) }))
    }

    fn mqtt_starter() -> MqttStarter {
        Box::new(|_| Box::pin(async { Ok(()) }))
    }

    #[test]
    fn new_registry_has_no_start_hooks() {
        let registry = DaemonRegistry::new();

        assert!(!registry.has_gateway_start());
        assert!(!registry.has_channels_start());
        assert!(!registry.has_socket_start());
        assert!(!registry.has_wss_start());
        assert!(!registry.has_mqtt_start());
    }

    #[test]
    fn builder_records_typed_start_hooks() {
        let mut registry = DaemonRegistry::new();
        registry
            .register_gateway(gateway_starter())
            .register_channels(channels_starter())
            .register_socket(socket_starter())
            .register_wss(rpc_starter())
            .register_mqtt(mqtt_starter());

        assert!(registry.has_gateway_start());
        assert!(registry.has_channels_start());
        assert!(registry.has_socket_start());
        assert!(registry.has_wss_start());
        assert!(registry.has_mqtt_start());
    }

    #[test]
    fn taking_start_hooks_consumes_slots() {
        let mut registry = DaemonRegistry::new();
        registry
            .register_gateway(gateway_starter())
            .register_channels(channels_starter())
            .register_socket(socket_starter())
            .register_wss(rpc_starter())
            .register_mqtt(mqtt_starter());

        assert!(registry.take_gateway_start().is_some());
        assert!(registry.take_channels_start().is_some());
        assert!(registry.take_socket_start().is_some());
        assert!(registry.take_wss_start().is_some());
        assert!(registry.take_mqtt_start().is_some());

        assert!(!registry.has_gateway_start());
        assert!(!registry.has_channels_start());
        assert!(!registry.has_socket_start());
        assert!(!registry.has_wss_start());
        assert!(!registry.has_mqtt_start());
    }

    #[test]
    fn goal_adapter_slots_are_separate_from_component_starters() {
        let mut registry = DaemonRegistry::new();

        assert_eq!(registry.goal_session_adapter_count(), 0);
        registry.register_goal_session_adapter(Arc::new(TestGoalSessionAdapter));
        assert_eq!(registry.goal_session_adapter_count(), 1);
        assert_eq!(registry.take_goal_session_adapters().len(), 1);
        assert_eq!(registry.goal_session_adapter_count(), 0);
    }
}
