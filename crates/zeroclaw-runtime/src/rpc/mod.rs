//! Transport-agnostic JSON-RPC 2.0 dispatch for the runtime.

pub mod approval_channel;
pub mod attachments;
pub mod context;
pub mod dispatch;
pub mod fs;
pub mod git;
pub(crate) mod goal_host;
pub mod local;
pub mod locales;
pub mod session;
pub mod transport;
pub mod tui_identity;
pub mod turn;
pub mod types;
pub mod wss;
