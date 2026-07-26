//! Process-global access to the daemon's [`ControlPlaneHandle`].

use std::sync::OnceLock;

use super::boot::ControlPlaneHandle;

static CONTROL_PLANE: OnceLock<ControlPlaneHandle> = OnceLock::new();

/// Install the daemon's control-plane handle. Called ONCE at boot
/// (`daemon::run`). Subsequent calls are ignored (returns `false`), so a reload
/// iteration cannot swap the live store out from under in-flight producers.
pub fn init_control_plane(handle: ControlPlaneHandle) -> bool {
    CONTROL_PLANE.set(handle).is_ok()
}

/// The live control-plane, or `None` when not running under a booted daemon. Producers
/// MUST treat `None` as "supervision disabled" and proceed exactly as today.
pub fn control_plane() -> Option<&'static ControlPlaneHandle> {
    CONTROL_PLANE.get()
}
