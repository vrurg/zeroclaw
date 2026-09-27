//! Shared, transport-neutral Goal Mode behavior and contracts.
//!
//! This crate owns Goal policy and session-bound interfaces. Concrete storage,
//! provider routing, agent execution, and transport presentation remain with
//! their existing owners.

// Share the integration fixtures with private boundary tests without exposing
// production inspection APIs or maintaining a second registry implementation.
#[cfg(test)]
extern crate self as zeroclaw_goal;
#[cfg(test)]
#[path = "../tests/support/mod.rs"]
mod test_support;

pub mod accounting;
pub mod checkpoint;
pub mod child_fence;
pub mod control;
pub mod task;
pub mod tool_pairing;

pub use checkpoint::{
    GoalCheckpointClosed, GoalDeferredControlTicket, GoalParentTurnCheckpoint,
    GoalParentTurnOperation, GoalParentTurnTicketId,
};
pub use control::{
    GoalControlAbandonReason, GoalControlAdmission, GoalControlClass, GoalController,
    GoalDeferredClosed, GoalDeferredCommit, GoalDeferredControl, GoalDeferredReady,
    GoalDiagnosticTextGuard, GoalExecutionEngine, GoalExecutionHost, GoalExecutionNotice,
    GoalExecutionOutcome, GoalExecutionRequest, GoalExecutionRestartCoordinator,
    GoalExecutionScope, GoalExecutionSubmission, GoalExecutionSupervisor, GoalExternalCancellation,
    GoalFailureNoticeDelivery, GoalFailureReason, GoalHostSettings, GoalIngressContext,
    GoalIngressPrincipal, GoalLaunchFailureReported, GoalOperationScope, GoalParentTurn,
    GoalParentTurnKind, GoalParentTurnResult, GoalPendingAction, GoalQueuedControl, GoalResponse,
    GoalRuntime, GoalRuntimeSubmission, GoalSessionBinding, GoalSessionDriver,
    GoalSessionExecutionLease, GoalSessionKey, GoalSessionLease, GoalStatusProjection,
    GoalSubmission, GoalSupervisorRetired, GoalSurface, GoalTurnBoundary, GoalVerifierAttribution,
    GoalVerifierResult, GoalVerifierTurn, classify_goal_control,
};
pub use task::{
    GoalAccountingState, GoalBlocker, GoalBlockerKind, GoalPauseReason, GoalPauseState,
    GoalTaskRecord, GoalTaskRegistry, GoalTransitionResult, TaskContinuationContext,
    TaskContinuationConversationScope, TaskGoal,
};
