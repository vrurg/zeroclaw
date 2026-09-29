//! Shared, transport-neutral Goal Mode behavior and contracts.
//!
//! This crate owns Goal policy and session-bound interfaces. Concrete storage,
//! provider routing, agent execution, and transport presentation remain with
//! their existing owners.

pub mod accounting;
pub mod checkpoint;
pub mod control;
pub mod task;

pub use checkpoint::{
    GoalCheckpointClosed, GoalDeferredControlTicket, GoalParentTurnCheckpoint,
    GoalParentTurnOperation, GoalParentTurnTicketId,
};
pub use control::{
    GoalController, GoalExecutionEngine, GoalExecutionHost, GoalExecutionNotice,
    GoalExecutionOutcome, GoalExecutionRequest, GoalExecutionRestartCoordinator,
    GoalExecutionScope, GoalExecutionSubmission, GoalExecutionSupervisor, GoalFailureReason,
    GoalHostSettings, GoalIngressContext, GoalIngressPrincipal, GoalOperationScope, GoalParentTurn,
    GoalParentTurnKind, GoalParentTurnResult, GoalPendingAction, GoalResponse, GoalRuntime,
    GoalRuntimeSubmission, GoalSessionBinding, GoalSessionDriver, GoalSessionExecutionLease,
    GoalSessionKey, GoalSessionLease, GoalStatusProjection, GoalSubmission, GoalSurface,
    GoalVerifierAttribution, GoalVerifierResult, GoalVerifierTurn,
};
pub use task::{
    GoalAccountingState, GoalBlocker, GoalBlockerKind, GoalPauseReason, GoalPauseState,
    GoalTaskRecord, GoalTaskRegistry, GoalTransitionResult, TaskContinuationContext,
    TaskContinuationConversationScope, TaskGoal,
};
