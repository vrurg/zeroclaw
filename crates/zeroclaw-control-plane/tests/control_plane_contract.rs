//! Compile-time contract for the durable control-plane's stable runtime seam.

use zeroclaw_control_plane::{
    ControlPlaneHandle, GoalTaskRegistry, SqliteTaskStore, TaskRegistry, control_plane,
    init_control_plane,
};

#[test]
fn exposes_the_canonical_control_plane_contract() {
    let _ = ControlPlaneHandle::start;
    let _ = SqliteTaskStore::new_in_memory;
    let _ = control_plane;
    let _ = init_control_plane;
    let _: Option<&dyn TaskRegistry> = None;
    let _: Option<&dyn GoalTaskRegistry> = None;
}
