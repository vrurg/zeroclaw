//! SQLite-backed [`GoalTaskRegistry`] implementation.

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};

use crate::control_plane::goal_task::{
    GoalAccountingState, GoalBlocker, GoalPauseReason, GoalPauseState, GoalResumeSafety,
    GoalTaskRecord, GoalTaskRegistry, GoalTaskStatus, GoalTransitionResult,
    TaskContinuationContext,
};
use crate::control_plane::task_registry::{TaskKind, TaskRecord, TaskStatus};

use super::{
    SqliteTaskStore, add_column_if_missing, claim_task_owner_record, insert_task_record,
    log_unreadable_task_row, row_to_record, update_task_status_record,
};

pub(super) fn migrate_schema(conn: &mut Connection, version: i64) -> Result<()> {
    if version < 1 {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS goal_tasks (
                 task_id        TEXT PRIMARY KEY
                                REFERENCES tasks(id) ON DELETE CASCADE,
                 objective      TEXT NOT NULL
             );
             PRAGMA user_version = 1;",
        )
        .context("apply control-plane schema v1")?;
    }
    if version < 2 {
        add_column_if_missing(
            conn,
            "goal_tasks",
            "effective_token_limit",
            "ALTER TABLE goal_tasks ADD COLUMN effective_token_limit INTEGER",
        )?;
        add_column_if_missing(
            conn,
            "goal_tasks",
            "effective_cost_limit_usd",
            "ALTER TABLE goal_tasks ADD COLUMN effective_cost_limit_usd REAL",
        )?;
        conn.execute_batch("PRAGMA user_version = 2;")
            .context("mark control-plane schema v2")?;
    }
    if version < 3 {
        add_column_if_missing(
            conn,
            "goal_tasks",
            "pause_reason",
            "ALTER TABLE goal_tasks ADD COLUMN pause_reason TEXT",
        )?;
        add_column_if_missing(
            conn,
            "goal_tasks",
            "pause_description",
            "ALTER TABLE goal_tasks ADD COLUMN pause_description TEXT",
        )?;
        add_column_if_missing(
            conn,
            "goal_tasks",
            "blockers_json",
            "ALTER TABLE goal_tasks ADD COLUMN blockers_json TEXT NOT NULL DEFAULT '[]'",
        )?;
        conn.execute_batch("PRAGMA user_version = 3;")
            .context("mark control-plane schema v3")?;
    }
    if version < 4 {
        conn.execute_batch(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_active_goal_context
                ON tasks(
                    agent,
                    COALESCE(originator_route, ''),
                    COALESCE(principal_id, '')
                )
                WHERE kind = 'goal'
                  AND status NOT IN ('completed','failed','cancelled','lost','timed_out');
             PRAGMA user_version = 4;",
        )
        .context("apply control-plane schema v4")?;
    }
    if version < 5 {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS task_continuation_contexts (
                 task_id      TEXT PRIMARY KEY
                              REFERENCES tasks(id) ON DELETE CASCADE,
                 context_json TEXT NOT NULL
             );
             PRAGMA user_version = 5;",
        )
        .context("apply control-plane schema v5")?;
    }
    if version < 6 {
        conn.execute_batch(
            "CREATE TRIGGER IF NOT EXISTS trg_goal_tasks_require_goal_kind
                BEFORE INSERT ON goal_tasks
                FOR EACH ROW
                WHEN COALESCE((SELECT kind FROM tasks WHERE id = NEW.task_id), '') != 'goal'
                BEGIN
                    SELECT RAISE(ABORT, 'goal_tasks.task_id must reference TaskKind::Goal');
                END;
             PRAGMA user_version = 6;",
        )
        .context("apply control-plane schema v6")?;
    }
    if version < 7 {
        conn.execute_batch(
            "CREATE TRIGGER IF NOT EXISTS trg_goal_tasks_effective_limits_insert
                BEFORE INSERT ON goal_tasks
                FOR EACH ROW
                WHEN (NEW.effective_token_limit IS NOT NULL AND NEW.effective_token_limit < 0)
                  OR (NEW.effective_cost_limit_usd IS NOT NULL
                      AND NOT (NEW.effective_cost_limit_usd >= 0.0
                               AND NEW.effective_cost_limit_usd <= 1.7976931348623157e308))
                BEGIN
                    SELECT RAISE(ABORT, 'goal effective limits must be non-negative finite values');
                END;
             CREATE TRIGGER IF NOT EXISTS trg_goal_tasks_effective_limits_update
                BEFORE UPDATE OF effective_token_limit, effective_cost_limit_usd ON goal_tasks
                FOR EACH ROW
                WHEN (NEW.effective_token_limit IS NOT NULL AND NEW.effective_token_limit < 0)
                  OR (NEW.effective_cost_limit_usd IS NOT NULL
                      AND NOT (NEW.effective_cost_limit_usd >= 0.0
                               AND NEW.effective_cost_limit_usd <= 1.7976931348623157e308))
                BEGIN
                    SELECT RAISE(ABORT, 'goal effective limits must be non-negative finite values');
                END;
             CREATE TRIGGER IF NOT EXISTS trg_task_continuation_contexts_require_goal_insert
                BEFORE INSERT ON task_continuation_contexts
                FOR EACH ROW
                WHEN NOT EXISTS (
                    SELECT 1
                      FROM tasks
                      JOIN goal_tasks ON goal_tasks.task_id = tasks.id
                     WHERE tasks.id = NEW.task_id
                       AND tasks.kind = 'goal'
                )
                BEGIN
                    SELECT RAISE(ABORT, 'task_continuation_contexts.task_id must reference a goal task');
                END;
             CREATE TRIGGER IF NOT EXISTS trg_task_continuation_contexts_require_goal_update
                BEFORE UPDATE OF task_id ON task_continuation_contexts
                FOR EACH ROW
                WHEN NOT EXISTS (
                    SELECT 1
                      FROM tasks
                      JOIN goal_tasks ON goal_tasks.task_id = tasks.id
                     WHERE tasks.id = NEW.task_id
                       AND tasks.kind = 'goal'
                )
                BEGIN
                    SELECT RAISE(ABORT, 'task_continuation_contexts.task_id must reference a goal task');
                END;
             PRAGMA user_version = 7;",
        )
        .context("apply control-plane schema v7")?;
    }
    if version < 8 {
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .context("start control-plane schema v8")?;
        add_column_if_missing(
            &tx,
            "tasks",
            "session_id",
            "ALTER TABLE tasks ADD COLUMN session_id TEXT",
        )?;
        add_column_if_missing(
            &tx,
            "tasks",
            "execution_epoch",
            "ALTER TABLE tasks ADD COLUMN execution_epoch INTEGER NOT NULL DEFAULT 0",
        )?;
        add_column_if_missing(
            &tx,
            "goal_tasks",
            "pending_call_id",
            "ALTER TABLE goal_tasks ADD COLUMN pending_call_id TEXT",
        )?;
        add_column_if_missing(
            &tx,
            "goal_tasks",
            "pending_call_epoch",
            "ALTER TABLE goal_tasks ADD COLUMN pending_call_epoch INTEGER",
        )?;
        add_column_if_missing(
            &tx,
            "goal_tasks",
            "accounting_state",
            "ALTER TABLE goal_tasks ADD COLUMN accounting_state TEXT NOT NULL DEFAULT 'complete'",
        )?;
        tx.execute_batch(
            "DROP INDEX IF EXISTS idx_tasks_active_goal_context;
             CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_goal_session
                ON tasks(session_id)
                WHERE kind = 'goal' AND session_id IS NOT NULL;
             CREATE UNIQUE INDEX IF NOT EXISTS idx_goal_tasks_pending_call
                ON goal_tasks(pending_call_id)
                WHERE pending_call_id IS NOT NULL;
             CREATE TRIGGER IF NOT EXISTS trg_tasks_goal_session_insert
                BEFORE INSERT ON tasks FOR EACH ROW
                WHEN NEW.kind = 'goal'
                 AND (NEW.session_id IS NULL OR length(trim(NEW.session_id)) = 0)
                BEGIN SELECT RAISE(ABORT, 'goal tasks require a nonblank session_id'); END;
             CREATE TRIGGER IF NOT EXISTS trg_tasks_goal_session_update
                BEFORE UPDATE OF kind, session_id ON tasks FOR EACH ROW
                WHEN NEW.kind = 'goal'
                 AND (NEW.session_id IS NULL OR length(trim(NEW.session_id)) = 0)
                BEGIN SELECT RAISE(ABORT, 'goal tasks require a nonblank session_id'); END;
             CREATE TRIGGER IF NOT EXISTS trg_goal_pending_pair_insert
                BEFORE INSERT ON goal_tasks FOR EACH ROW
                WHEN (NEW.pending_call_id IS NULL) != (NEW.pending_call_epoch IS NULL)
                BEGIN SELECT RAISE(ABORT, 'pending call id and epoch must be paired'); END;
             CREATE TRIGGER IF NOT EXISTS trg_goal_pending_pair_update
                BEFORE UPDATE OF pending_call_id, pending_call_epoch ON goal_tasks FOR EACH ROW
                WHEN (NEW.pending_call_id IS NULL) != (NEW.pending_call_epoch IS NULL)
                BEGIN SELECT RAISE(ABORT, 'pending call id and epoch must be paired'); END;
             CREATE TRIGGER IF NOT EXISTS trg_goal_accounting_state_insert
                BEFORE INSERT ON goal_tasks FOR EACH ROW
                WHEN NEW.accounting_state NOT IN ('complete','missing','invalid','outcome_unknown')
                BEGIN SELECT RAISE(ABORT, 'invalid goal accounting state'); END;
             CREATE TRIGGER IF NOT EXISTS trg_goal_accounting_state_update
                BEFORE UPDATE OF accounting_state ON goal_tasks FOR EACH ROW
                WHEN NEW.accounting_state NOT IN ('complete','missing','invalid','outcome_unknown')
                BEGIN SELECT RAISE(ABORT, 'invalid goal accounting state'); END;
             PRAGMA user_version = 8;",
        )
        .context("apply control-plane schema v8")?;
        tx.commit().context("commit control-plane schema v8")?;
    }
    if version < 9 {
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .context("start control-plane schema v9")?;
        tx.execute_batch(
            "DROP TRIGGER IF EXISTS trg_tasks_goal_session_insert;
             DROP TRIGGER IF EXISTS trg_tasks_goal_session_update;
             DROP TRIGGER IF EXISTS trg_goal_accounting_state_insert;
             DROP TRIGGER IF EXISTS trg_goal_accounting_state_update;
             CREATE TRIGGER trg_tasks_goal_session_insert
                BEFORE INSERT ON tasks FOR EACH ROW
                WHEN NEW.kind = 'goal'
                 AND (NEW.session_id IS NULL OR length(trim(NEW.session_id)) = 0)
                BEGIN SELECT RAISE(ABORT, 'goal tasks require a nonblank session_id'); END;
             CREATE TRIGGER trg_tasks_goal_kind_rebind
                BEFORE UPDATE OF kind, session_id ON tasks FOR EACH ROW
                WHEN OLD.kind != 'goal' AND NEW.kind = 'goal'
                BEGIN SELECT RAISE(ABORT, 'task kind cannot be reclassified as goal'); END;
             CREATE TRIGGER trg_tasks_goal_session_rebind
                BEFORE UPDATE OF kind, session_id ON tasks FOR EACH ROW
                WHEN OLD.kind = 'goal' AND (
                    NEW.kind != 'goal' OR NEW.session_id IS NOT OLD.session_id
                )
                BEGIN SELECT RAISE(ABORT, 'goal session binding cannot be changed'); END;
             CREATE TRIGGER trg_goal_accounting_state_insert
                BEFORE INSERT ON goal_tasks FOR EACH ROW
                WHEN NEW.accounting_state NOT IN ('complete','missing','invalid','outcome_unknown')
                BEGIN SELECT RAISE(ABORT, 'invalid goal accounting state'); END;
             CREATE TRIGGER trg_goal_accounting_state_update
                BEFORE UPDATE OF accounting_state ON goal_tasks FOR EACH ROW
                WHEN NEW.accounting_state NOT IN ('complete','missing','invalid','outcome_unknown')
                BEGIN SELECT RAISE(ABORT, 'invalid goal accounting state'); END;
             PRAGMA user_version = 9;",
        )
        .context("apply control-plane schema v9")?;
        tx.commit().context("commit control-plane schema v9")?;
    }
    if version < 10 {
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .context("start control-plane schema v10")?;
        tx.execute_batch(
            "CREATE TRIGGER trg_tasks_goal_identity_insert
                BEFORE INSERT ON tasks FOR EACH ROW
                WHEN EXISTS (
                    SELECT 1 FROM tasks AS existing
                     WHERE existing.kind = 'goal'
                       AND (existing.id = NEW.id OR existing.rowid = NEW.rowid)
                )
                BEGIN
                    SELECT RAISE(ABORT, 'canonical goal task rows cannot be replaced');
                END;
             CREATE TRIGGER trg_tasks_goal_identity_update
                BEFORE UPDATE ON tasks FOR EACH ROW
                WHEN (OLD.kind = 'goal' AND (
                        NEW.id IS NOT OLD.id OR NEW.rowid IS NOT OLD.rowid
                    ))
                  OR EXISTS (
                    SELECT 1 FROM tasks AS existing
                     WHERE existing.kind = 'goal'
                       AND existing.rowid != OLD.rowid
                       AND (existing.id = NEW.id OR existing.rowid = NEW.rowid)
                )
                BEGIN
                    SELECT RAISE(ABORT, 'canonical goal task rows cannot be replaced');
                END;
             PRAGMA user_version = 10;",
        )
        .context("apply control-plane schema v10")?;
        tx.commit().context("commit control-plane schema v10")?;
    }
    if version < 11 {
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .context("start control-plane schema v11")?;
        tx.execute_batch(
            "DROP TRIGGER IF EXISTS trg_tasks_goal_identity_insert;
             CREATE TRIGGER trg_tasks_goal_identity_insert
                BEFORE INSERT ON tasks FOR EACH ROW
                WHEN EXISTS (
                    SELECT 1 FROM tasks AS existing
                     WHERE existing.kind = 'goal'
                       AND (
                            existing.id = NEW.id
                         OR existing.rowid = NEW.rowid
                         OR (NEW.kind = 'goal' AND existing.session_id IS NEW.session_id)
                       )
                )
                BEGIN
                    SELECT RAISE(ABORT, 'canonical goal task rows cannot be replaced');
                END;
             PRAGMA user_version = 11;",
        )
        .context("apply control-plane schema v11")?;
        tx.commit().context("commit control-plane schema v11")?;
    }
    if version < 12 {
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .context("start control-plane schema v12")?;
        add_column_if_missing(
            &tx,
            "goal_tasks",
            "success_criteria",
            "ALTER TABLE goal_tasks ADD COLUMN success_criteria TEXT NOT NULL DEFAULT ''",
        )?;
        add_column_if_missing(
            &tx,
            "goal_tasks",
            "pending_reserved_tokens",
            "ALTER TABLE goal_tasks ADD COLUMN pending_reserved_tokens INTEGER",
        )?;
        add_column_if_missing(
            &tx,
            "goal_tasks",
            "pending_reserved_cost_usd",
            "ALTER TABLE goal_tasks ADD COLUMN pending_reserved_cost_usd REAL",
        )?;
        add_column_if_missing(
            &tx,
            "goal_tasks",
            "resume_safety",
            "ALTER TABLE goal_tasks ADD COLUMN resume_safety TEXT NOT NULL DEFAULT 'clean'",
        )?;
        tx.execute_batch(
            "DROP TRIGGER IF EXISTS trg_goal_success_criteria_insert;
             DROP TRIGGER IF EXISTS trg_goal_success_criteria_update;
             DROP TRIGGER IF EXISTS trg_goal_session_objective_insert;
             DROP TRIGGER IF EXISTS trg_goal_session_objective_update;
             DROP TRIGGER IF EXISTS trg_goal_pending_reservation_insert;
             DROP TRIGGER IF EXISTS trg_goal_pending_reservation_update;
             DROP TRIGGER IF EXISTS trg_goal_resume_safety_insert;
             DROP TRIGGER IF EXISTS trg_goal_resume_safety_update;
             CREATE TRIGGER trg_goal_success_criteria_insert
                BEFORE INSERT ON goal_tasks FOR EACH ROW
                WHEN length(trim(NEW.success_criteria, char(9) || char(10) || char(11) || char(12) || char(13) || char(32) || char(133) || char(160) || char(5760) || char(8192) || char(8193) || char(8194) || char(8195) || char(8196) || char(8197) || char(8198) || char(8199) || char(8200) || char(8201) || char(8202) || char(8232) || char(8233) || char(8239) || char(8287) || char(12288))) = 0
                BEGIN SELECT RAISE(ABORT, 'goal success criteria must be nonblank'); END;
             CREATE TRIGGER trg_goal_session_objective_insert
                BEFORE INSERT ON goal_tasks FOR EACH ROW
                WHEN length(trim(NEW.objective, char(9) || char(10) || char(11) || char(12) || char(13) || char(32) || char(133) || char(160) || char(5760) || char(8192) || char(8193) || char(8194) || char(8195) || char(8196) || char(8197) || char(8198) || char(8199) || char(8200) || char(8201) || char(8202) || char(8232) || char(8233) || char(8239) || char(8287) || char(12288))) = 0
                 AND EXISTS (SELECT 1 FROM tasks WHERE id = NEW.task_id AND session_id IS NOT NULL)
                BEGIN SELECT RAISE(ABORT, 'session goal objective must be nonblank'); END;
             CREATE TRIGGER trg_goal_success_criteria_update
                BEFORE UPDATE OF success_criteria ON goal_tasks FOR EACH ROW
                WHEN NEW.success_criteria IS NOT OLD.success_criteria
                BEGIN SELECT RAISE(ABORT, 'goal success criteria cannot be changed'); END;
             CREATE TRIGGER trg_goal_session_objective_update
                BEFORE UPDATE OF objective ON goal_tasks FOR EACH ROW
                WHEN NEW.objective IS NOT OLD.objective
                 AND EXISTS (
                    SELECT 1 FROM tasks
                     WHERE id = OLD.task_id AND kind = 'goal' AND session_id IS NOT NULL
                 )
                BEGIN SELECT RAISE(ABORT, 'session goal objective cannot be changed'); END;
             CREATE TRIGGER trg_goal_pending_reservation_insert
                BEFORE INSERT ON goal_tasks FOR EACH ROW
                WHEN (NEW.pending_call_id IS NULL
                      AND (NEW.pending_reserved_tokens IS NOT NULL
                           OR NEW.pending_reserved_cost_usd IS NOT NULL))
                  OR (NEW.pending_call_id IS NOT NULL
                      AND (length(trim(NEW.pending_call_id)) = 0
                           OR NEW.pending_call_epoch < 1
                           OR (NEW.effective_token_limit IS NOT NULL
                               AND NEW.pending_reserved_tokens IS NULL)
                           OR (NEW.effective_cost_limit_usd IS NOT NULL
                               AND NEW.pending_reserved_cost_usd IS NULL)))
                  OR (NEW.pending_reserved_tokens IS NOT NULL
                      AND NEW.pending_reserved_tokens < 0)
                  OR (NEW.pending_reserved_cost_usd IS NOT NULL
                      AND NOT (NEW.pending_reserved_cost_usd >= 0.0
                               AND NEW.pending_reserved_cost_usd <= 1.7976931348623157e308))
                BEGIN SELECT RAISE(ABORT, 'invalid goal pending reservation'); END;
             CREATE TRIGGER trg_goal_pending_reservation_update
                BEFORE UPDATE OF pending_call_id, pending_call_epoch, pending_reserved_tokens,
                                 pending_reserved_cost_usd, effective_token_limit,
                                 effective_cost_limit_usd ON goal_tasks FOR EACH ROW
                WHEN (NEW.pending_call_id IS NULL
                      AND (NEW.pending_reserved_tokens IS NOT NULL
                           OR NEW.pending_reserved_cost_usd IS NOT NULL))
                  OR (OLD.pending_call_id IS NULL AND NEW.pending_call_id IS NOT NULL
                      AND (length(trim(NEW.pending_call_id)) = 0
                           OR NEW.pending_call_epoch < 1
                           OR (NEW.effective_token_limit IS NOT NULL
                               AND NEW.pending_reserved_tokens IS NULL)
                           OR (NEW.effective_cost_limit_usd IS NOT NULL
                               AND NEW.pending_reserved_cost_usd IS NULL)))
                  OR (OLD.pending_call_id IS NOT NULL AND NEW.pending_call_id IS NOT NULL
                      AND (NEW.pending_call_id IS NOT OLD.pending_call_id
                           OR NEW.pending_call_epoch IS NOT OLD.pending_call_epoch
                           OR NEW.pending_reserved_tokens IS NOT OLD.pending_reserved_tokens
                           OR NEW.pending_reserved_cost_usd IS NOT OLD.pending_reserved_cost_usd))
                  OR (NEW.pending_reserved_tokens IS NOT NULL
                      AND NEW.pending_reserved_tokens < 0)
                  OR (NEW.pending_reserved_cost_usd IS NOT NULL
                      AND NOT (NEW.pending_reserved_cost_usd >= 0.0
                               AND NEW.pending_reserved_cost_usd <= 1.7976931348623157e308))
                BEGIN SELECT RAISE(ABORT, 'invalid goal pending reservation'); END;
             CREATE TRIGGER trg_goal_resume_safety_insert
                BEFORE INSERT ON goal_tasks FOR EACH ROW
                WHEN NEW.resume_safety NOT IN ('clean','tool_executed')
                BEGIN SELECT RAISE(ABORT, 'invalid goal resume safety'); END;
             CREATE TRIGGER trg_goal_resume_safety_update
                BEFORE UPDATE OF resume_safety ON goal_tasks FOR EACH ROW
                WHEN NEW.resume_safety NOT IN ('clean','tool_executed')
                   OR (OLD.resume_safety = 'tool_executed' AND NEW.resume_safety != 'tool_executed')
                BEGIN SELECT RAISE(ABORT, 'goal resume safety cannot be relaxed'); END;
             PRAGMA user_version = 12;",
        )
        .context("apply control-plane schema v12")?;
        tx.commit().context("commit control-plane schema v12")?;
    }
    Ok(())
}

/// Recover Goal rows before generic reaping. A restart may only pause a fully
/// settled Goal; an admitted pending operation is fail-closed because its spend
/// and response cannot be replayed safely.
pub(crate) fn recover_on_boot(store: &SqliteTaskStore, boot_id: &str) -> Result<usize> {
    let mut conn = store.conn.lock();
    let tx = conn.transaction().context("start goal boot recovery")?;
    let now = chrono::Utc::now().to_rfc3339();
    let legacy = tx.execute(
        "UPDATE tasks
            SET status = 'failed', error = COALESCE(error, 'legacy_goal_missing_session'),
                finished_at = COALESCE(finished_at, ?1),
                execution_epoch = execution_epoch + 1
          WHERE kind = 'goal' AND session_id IS NULL
            AND status IN ('running','paused')",
        params![&now],
    )?;
    let legacy_missing_criteria = tx.execute(
        "UPDATE tasks
            SET status = 'failed', error = COALESCE(error, 'legacy_goal_missing_success_criteria'),
                finished_at = COALESCE(finished_at, ?1),
                execution_epoch = execution_epoch + 1
          WHERE kind = 'goal' AND session_id IS NOT NULL
            AND status IN ('running','paused')
            AND EXISTS (
                SELECT 1 FROM goal_tasks
                 WHERE task_id = tasks.id
                   AND length(trim(success_criteria, char(9) || char(10) || char(11) || char(12) || char(13) || char(32) || char(133) || char(160) || char(5760) || char(8192) || char(8193) || char(8194) || char(8195) || char(8196) || char(8197) || char(8198) || char(8199) || char(8200) || char(8201) || char(8202) || char(8232) || char(8233) || char(8239) || char(8287) || char(12288))) = 0
            )",
        params![&now],
    )?;
    tx.execute(
        "UPDATE goal_tasks
            SET accounting_state = 'outcome_unknown'
          WHERE task_id IN (
                SELECT id FROM tasks
                 WHERE kind = 'goal' AND session_id IS NOT NULL
                   AND status IN ('running','paused') AND owner_boot_id != ?1
                   AND EXISTS (
                        SELECT 1 FROM goal_tasks
                         WHERE task_id = tasks.id
                           AND (accounting_state != 'complete'
                                OR pending_call_id IS NOT NULL
                                OR pending_call_epoch IS NOT NULL)
                   )
          )",
        params![boot_id],
    )?;
    let uncertain = tx.execute(
        "UPDATE tasks
            SET status = 'failed', error = COALESCE(error, 'accounting_outcome_unknown'),
                finished_at = COALESCE(finished_at, ?2),
                execution_epoch = execution_epoch + 1
          WHERE kind = 'goal' AND session_id IS NOT NULL
            AND status IN ('running','paused') AND owner_boot_id != ?1
            AND EXISTS (
                SELECT 1 FROM goal_tasks
                 WHERE task_id = tasks.id
                   AND (accounting_state != 'complete'
                        OR pending_call_id IS NOT NULL
                        OR pending_call_epoch IS NOT NULL)
            )",
        params![boot_id, &now],
    )?;
    tx.execute(
        "UPDATE goal_tasks
            SET pause_reason = 'daemon_restarted',
                pause_description = 'daemon restart', blockers_json = '[]'
          WHERE task_id IN (
                SELECT id FROM tasks
                 WHERE kind = 'goal' AND status = 'running'
                   AND session_id IS NOT NULL AND owner_boot_id != ?1
                   AND EXISTS (
                        SELECT 1 FROM goal_tasks
                         WHERE task_id = tasks.id
                           AND accounting_state = 'complete'
                           AND pending_call_id IS NULL
                           AND pending_call_epoch IS NULL
                   )
          )",
        params![boot_id],
    )?;
    let paused = tx.execute(
        "UPDATE tasks
            SET status = 'paused', execution_epoch = execution_epoch + 1
          WHERE kind = 'goal' AND status = 'running' AND session_id IS NOT NULL
            AND owner_boot_id != ?1
            AND EXISTS (
                SELECT 1 FROM goal_tasks
                 WHERE task_id = tasks.id AND accounting_state = 'complete'
                   AND pending_call_id IS NULL AND pending_call_epoch IS NULL
            )",
        params![boot_id],
    )?;
    tx.commit().context("commit goal boot recovery")?;
    Ok(legacy + legacy_missing_criteria + uncertain + paused)
}

fn ensure_goal_task_identity(task: &TaskRecord, goal: &GoalTaskRecord) -> Result<()> {
    if task.kind != TaskKind::Goal {
        anyhow::bail!("goal task {} must use TaskKind::Goal", task.id);
    }
    if task.id != goal.task_id {
        anyhow::bail!(
            "goal task id mismatch: TaskRecord id {} does not match GoalTaskRecord task_id {}",
            task.id,
            goal.task_id
        );
    }
    Ok(())
}

fn ensure_fresh_session_goal(task: &TaskRecord, goal: &GoalTaskRecord) -> Result<()> {
    if task.status != TaskStatus::Running || task.finished_at.is_some() {
        anyhow::bail!("new session goals must start running without terminal metadata");
    }
    if goal.objective.trim().is_empty()
        || goal.success_criteria.trim().is_empty()
        || goal.pause_reason.is_some()
        || goal.pause_description.is_some()
        || !goal.blockers.is_empty()
        || goal.pending_call_id.is_some()
        || goal.pending_call_epoch.is_some()
        || goal.pending_reserved_tokens.is_some()
        || goal.pending_reserved_cost_usd.is_some()
        || goal.accounting_state != GoalAccountingState::Complete
        || goal.resume_safety != GoalResumeSafety::Clean
    {
        anyhow::bail!("new session goals must not include pause or accounting state");
    }
    Ok(())
}

fn pause_reason_to_db(reason: GoalPauseReason) -> Result<String> {
    let value = serde_json::to_value(reason).context("serialize goal pause reason")?;
    value
        .as_str()
        .map(str::to_owned)
        .context("serialized goal pause reason is not a string")
}

fn pause_reason_from_db(value: Option<String>) -> Result<Option<GoalPauseReason>> {
    value
        .map(|value| {
            serde_json::from_value(serde_json::Value::String(value.clone()))
                .with_context(|| format!("unknown goal pause reason {value:?}"))
        })
        .transpose()
}

fn blockers_to_db(blockers: &[GoalBlocker]) -> Result<String> {
    serde_json::to_string(blockers).context("serialize goal blockers")
}

fn blockers_from_db(value: String) -> rusqlite::Result<Vec<GoalBlocker>> {
    serde_json::from_str(&value).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, e.into())
    })
}

fn continuation_context_to_db(context: &TaskContinuationContext) -> Result<String> {
    serde_json::to_string(context).context("serialize task continuation context")
}

fn continuation_context_from_db(value: String) -> rusqlite::Result<TaskContinuationContext> {
    serde_json::from_str(&value).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, e.into())
    })
}

fn token_limit_to_db(value: u64) -> Result<i64> {
    i64::try_from(value)
        .with_context(|| format!("effective token limit {value} exceeds SQLite INTEGER domain"))
}

fn token_limit_from_db(value: Option<i64>) -> rusqlite::Result<Option<u64>> {
    value
        .map(|value| {
            u64::try_from(value).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Integer,
                    e.into(),
                )
            })
        })
        .transpose()
}

fn cost_limit_to_db(value: f64) -> Result<f64> {
    if !value.is_finite() || value < 0.0 {
        anyhow::bail!("effective cost limit must be a non-negative finite value");
    }
    Ok(value)
}

fn cost_limit_from_db(value: Option<f64>) -> rusqlite::Result<Option<f64>> {
    match value {
        Some(value) if !value.is_finite() || value < 0.0 => {
            Err(rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Real,
                format!("invalid effective cost limit {value}").into(),
            ))
        }
        other => Ok(other),
    }
}

fn reservation_tokens_from_db(value: Option<i64>) -> rusqlite::Result<Option<u64>> {
    token_limit_from_db(value)
}

fn reservation_cost_from_db(value: Option<f64>) -> rusqlite::Result<Option<f64>> {
    cost_limit_from_db(value)
}

fn goal_limits_to_db(
    token_limit: Option<u64>,
    cost_limit_usd: Option<f64>,
) -> Result<(Option<i64>, Option<f64>)> {
    Ok((
        token_limit.map(token_limit_to_db).transpose()?,
        cost_limit_usd.map(cost_limit_to_db).transpose()?,
    ))
}

fn row_to_goal_task(row: &rusqlite::Row<'_>) -> rusqlite::Result<GoalTaskRecord> {
    let pause_reason = pause_reason_from_db(row.get("pause_reason")?).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, e.into())
    })?;
    Ok(GoalTaskRecord {
        task_id: row.get("task_id")?,
        objective: row.get("objective")?,
        effective_token_limit: token_limit_from_db(row.get("effective_token_limit")?)?,
        effective_cost_limit_usd: cost_limit_from_db(row.get("effective_cost_limit_usd")?)?,
        pause_reason,
        pause_description: row.get("pause_description")?,
        blockers: blockers_from_db(row.get("blockers_json")?)?,
        pending_call_id: row.get("pending_call_id")?,
        pending_call_epoch: row.get("pending_call_epoch")?,
        pending_reserved_tokens: reservation_tokens_from_db(row.get("pending_reserved_tokens")?)?,
        pending_reserved_cost_usd: reservation_cost_from_db(row.get("pending_reserved_cost_usd")?)?,
        accounting_state: accounting_state_from_db(row.get("accounting_state")?)?,
        success_criteria: row.get("success_criteria")?,
        resume_safety: resume_safety_from_db(row.get("resume_safety")?)?,
    })
}

fn accounting_state_to_db(value: GoalAccountingState) -> &'static str {
    match value {
        GoalAccountingState::Complete => "complete",
        GoalAccountingState::Missing => "missing",
        GoalAccountingState::Invalid => "invalid",
        GoalAccountingState::OutcomeUnknown => "outcome_unknown",
    }
}

fn accounting_state_from_db(value: String) -> rusqlite::Result<GoalAccountingState> {
    match value.as_str() {
        "complete" => Ok(GoalAccountingState::Complete),
        "missing" => Ok(GoalAccountingState::Missing),
        "invalid" => Ok(GoalAccountingState::Invalid),
        "outcome_unknown" => Ok(GoalAccountingState::OutcomeUnknown),
        _ => Err(rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Text,
            format!("unknown goal accounting state {value:?}").into(),
        )),
    }
}

fn resume_safety_to_db(value: GoalResumeSafety) -> &'static str {
    match value {
        GoalResumeSafety::Clean => "clean",
        GoalResumeSafety::ToolExecuted => "tool_executed",
    }
}

fn resume_safety_from_db(value: String) -> rusqlite::Result<GoalResumeSafety> {
    match value.as_str() {
        "clean" => Ok(GoalResumeSafety::Clean),
        "tool_executed" => Ok(GoalResumeSafety::ToolExecuted),
        _ => Err(rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Text,
            format!("unknown goal resume safety {value:?}").into(),
        )),
    }
}

fn insert_goal_task_record(conn: &Connection, rec: GoalTaskRecord) -> Result<()> {
    let pause_reason = rec.pause_reason.map(pause_reason_to_db).transpose()?;
    let blockers_json = blockers_to_db(&rec.blockers)?;
    let (effective_token_limit, effective_cost_limit_usd) =
        goal_limits_to_db(rec.effective_token_limit, rec.effective_cost_limit_usd)?;
    conn.execute(
        "INSERT INTO goal_tasks
            (task_id, objective, success_criteria, effective_token_limit, effective_cost_limit_usd,
             pause_reason, pause_description, blockers_json, pending_call_id,
             pending_call_epoch, pending_reserved_tokens, pending_reserved_cost_usd,
             accounting_state, resume_safety)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
         ON CONFLICT(task_id) DO NOTHING",
        params![
            rec.task_id,
            rec.objective,
            rec.success_criteria,
            effective_token_limit,
            effective_cost_limit_usd,
            pause_reason,
            rec.pause_description,
            blockers_json,
            rec.pending_call_id,
            rec.pending_call_epoch,
            rec.pending_reserved_tokens
                .map(token_limit_to_db)
                .transpose()?,
            rec.pending_reserved_cost_usd
                .map(cost_limit_to_db)
                .transpose()?,
            accounting_state_to_db(rec.accounting_state),
            resume_safety_to_db(rec.resume_safety),
        ],
    )
    .context("insert goal task record")?;
    Ok(())
}

fn update_goal_pause_record(
    conn: &Connection,
    task_id: &str,
    pause: Option<GoalPauseState>,
) -> Result<usize> {
    let (reason, description, blockers_json) = match pause {
        Some(pause) => (
            Some(pause_reason_to_db(pause.reason)?),
            pause.description,
            blockers_to_db(&pause.blockers)?,
        ),
        None => (None, None, blockers_to_db(&[])?),
    };
    conn.execute(
        "UPDATE goal_tasks
            SET pause_reason = ?1,
                pause_description = ?2,
                blockers_json = ?3
          WHERE task_id = ?4",
        params![reason, description, blockers_json, task_id],
    )
    .context("update goal pause state")
}

fn ensure_goal_task_row(conn: &Connection, task_id: &str) -> Result<()> {
    let exists = conn
        .query_row(
            "SELECT EXISTS(
                 SELECT 1
                   FROM tasks
                   JOIN goal_tasks ON goal_tasks.task_id = tasks.id
                  WHERE tasks.id = ?1
                    AND tasks.kind = 'goal'
             )",
            params![task_id],
            |row| row.get::<_, i64>(0),
        )
        .context("verify goal task row")?
        != 0;
    if !exists {
        anyhow::bail!("goal task {task_id} has no canonical TaskKind::Goal row and goal extension");
    }
    Ok(())
}

fn upsert_continuation_context(
    conn: &Connection,
    task_id: &str,
    context: &TaskContinuationContext,
) -> Result<()> {
    conn.execute(
        "INSERT INTO task_continuation_contexts (task_id, context_json)
         VALUES (?1, ?2)
         ON CONFLICT(task_id) DO UPDATE
             SET context_json = excluded.context_json",
        params![task_id, continuation_context_to_db(context)?],
    )
    .context("upsert task continuation context")?;
    Ok(())
}

#[async_trait::async_trait]
impl GoalTaskRegistry for SqliteTaskStore {
    async fn create_goal(
        &self,
        mut task: TaskRecord,
        goal: GoalTaskRecord,
        continuation_context: Option<TaskContinuationContext>,
    ) -> Result<()> {
        ensure_goal_task_identity(&task, &goal)?;
        if task
            .session_id
            .as_deref()
            .is_none_or(|session_id| session_id.trim().is_empty())
        {
            anyhow::bail!("new goal requires a nonblank session_id");
        }
        task.execution_epoch = 1;
        let mut conn = self.conn.lock();
        let tx = conn
            .transaction()
            .context("start create goal transaction")?;
        let task_id = goal.task_id.clone();
        insert_task_record(&tx, task)?;
        insert_goal_task_record(&tx, goal)?;
        if let Some(context) = continuation_context {
            upsert_continuation_context(&tx, &task_id, &context)?;
        }
        tx.commit().context("commit create goal transaction")?;
        Ok(())
    }

    async fn latest_active_goal_for_agent(&self, agent: &str) -> Result<Option<TaskRecord>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare_cached(
                "SELECT * FROM tasks
              WHERE agent = ?1
                AND kind = 'goal'
                AND status IN ('running','paused')
              ORDER BY started_at DESC, rowid DESC
              LIMIT 1",
            )
            .context("prepare latest active goal by agent")?;
        let rows = stmt
            .query_map(params![agent], row_to_record)
            .context("query latest active goal by agent")?;
        for row in rows {
            match row {
                Ok(task) => return Ok(Some(task)),
                Err(error) => log_unreadable_task_row(error),
            }
        }
        Ok(None)
    }

    async fn latest_active_goal_for_context(
        &self,
        agent: &str,
        originator_route: Option<&str>,
        principal_id: Option<&str>,
    ) -> Result<Option<TaskRecord>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare_cached(
                "SELECT * FROM tasks
              WHERE agent = ?1
                AND kind = 'goal'
                AND status IN ('running','paused')
                AND (?2 IS NULL OR originator_route = ?2)
                AND (?3 IS NULL OR principal_id = ?3)
              ORDER BY started_at DESC, rowid DESC
              LIMIT 1",
            )
            .context("prepare latest active goal by context")?;
        let rows = stmt
            .query_map(
                params![agent, originator_route, principal_id],
                row_to_record,
            )
            .context("query latest active goal by context")?;
        for row in rows {
            match row {
                Ok(task) => return Ok(Some(task)),
                Err(error) => log_unreadable_task_row(error),
            }
        }
        Ok(None)
    }

    async fn latest_active_goal_id_for_context(
        &self,
        agent: &str,
        originator_route: Option<&str>,
        principal_id: Option<&str>,
    ) -> Result<Option<String>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare_cached(
                "SELECT id FROM tasks
                 WHERE agent = ?1
                   AND kind = 'goal'
                   AND status IN ('running','paused')
                   AND (?2 IS NULL OR originator_route = ?2)
                   AND (?3 IS NULL OR principal_id = ?3)
                 ORDER BY started_at DESC, rowid DESC
                 LIMIT 1",
            )
            .context("prepare latest active goal id by context")?;
        stmt.query_row(params![agent, originator_route, principal_id], |row| {
            row.get::<_, String>(0)
        })
        .optional()
        .context("query latest active goal id by context")
    }

    async fn get_goal_task(&self, task_id: &str) -> Result<Option<GoalTaskRecord>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT task_id, objective, success_criteria, effective_token_limit, effective_cost_limit_usd,
                    pause_reason, pause_description, blockers_json, pending_call_id,
                    pending_call_epoch, pending_reserved_tokens, pending_reserved_cost_usd,
                    accounting_state, resume_safety
             FROM goal_tasks WHERE task_id = ?1",
            params![task_id],
            row_to_goal_task,
        )
        .optional()
        .context("get goal task")
    }

    async fn update_goal_objective(&self, task_id: &str, objective: &str) -> Result<()> {
        let conn = self.conn.lock();
        let updated = conn
            .execute(
                "UPDATE goal_tasks
                    SET objective = ?1
                  WHERE task_id = ?2",
                params![objective, task_id],
            )
            .context("update goal objective")?;
        if updated == 0 {
            anyhow::bail!("goal task {task_id} has no goal extension row");
        }
        Ok(())
    }

    async fn update_goal_limits(
        &self,
        task_id: &str,
        token_limit: Option<u64>,
        cost_limit_usd: Option<f64>,
    ) -> Result<()> {
        let (effective_token_limit, effective_cost_limit_usd) =
            goal_limits_to_db(token_limit, cost_limit_usd)?;
        let conn = self.conn.lock();
        let updated = conn
            .execute(
                "UPDATE goal_tasks
                SET effective_token_limit = ?1,
                    effective_cost_limit_usd = ?2
              WHERE task_id = ?3",
                params![effective_token_limit, effective_cost_limit_usd, task_id],
            )
            .context("update goal effective limits")?;
        if updated == 0 {
            anyhow::bail!("goal task {task_id} has no goal extension row");
        }
        Ok(())
    }

    async fn update_goal_pause(&self, task_id: &str, pause: Option<GoalPauseState>) -> Result<()> {
        let conn = self.conn.lock();
        let updated = update_goal_pause_record(&conn, task_id, pause)?;
        if updated == 0 {
            anyhow::bail!("goal task {task_id} has no goal extension row");
        }
        Ok(())
    }

    async fn pause_goal_task(&self, task_id: &str, pause: GoalPauseState) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn
            .transaction()
            .context("start pause goal task transaction")?;
        ensure_goal_task_row(&tx, task_id)?;
        let updated = update_task_status_record(&tx, task_id, TaskStatus::Paused, None, None)?;
        if updated == 0 {
            anyhow::bail!("goal task {task_id} is terminal or missing");
        }
        let updated = update_goal_pause_record(&tx, task_id, Some(pause))?;
        if updated == 0 {
            anyhow::bail!("goal task {task_id} has no goal extension row");
        }
        tx.commit().context("commit pause goal task transaction")?;
        Ok(())
    }

    async fn resume_goal_task(
        &self,
        task_id: &str,
        owner_pid: u32,
        owner_boot_id: &str,
        continuation_context: Option<TaskContinuationContext>,
    ) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn
            .transaction()
            .context("start resume goal task transaction")?;
        ensure_goal_task_row(&tx, task_id)?;
        let resumable: bool = tx.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM goal_tasks
                 WHERE task_id = ?1
                   AND accounting_state = 'complete'
                   AND pending_call_id IS NULL
                   AND pending_call_epoch IS NULL
                   AND pending_reserved_tokens IS NULL
                   AND pending_reserved_cost_usd IS NULL
                   AND resume_safety = 'clean'
            )",
            params![task_id],
            |row| row.get(0),
        )?;
        if !resumable {
            anyhow::bail!("goal task {task_id} is not safely resumable");
        }
        if let Some(context) = continuation_context {
            upsert_continuation_context(&tx, task_id, &context)?;
        }
        let claimed = claim_task_owner_record(&tx, task_id, owner_pid, owner_boot_id)?;
        if claimed == 0 {
            anyhow::bail!("goal task {task_id} is terminal or missing");
        }
        let updated = update_task_status_record(&tx, task_id, TaskStatus::Running, None, None)?;
        if updated == 0 {
            anyhow::bail!("goal task {task_id} is terminal or missing");
        }
        let updated = update_goal_pause_record(&tx, task_id, None)?;
        if updated == 0 {
            anyhow::bail!("goal task {task_id} has no goal extension row");
        }
        tx.commit().context("commit resume goal task transaction")?;
        Ok(())
    }

    async fn set_continuation_context(
        &self,
        task_id: &str,
        context: Option<TaskContinuationContext>,
    ) -> Result<()> {
        let conn = self.conn.lock();
        ensure_goal_task_row(&conn, task_id)?;
        match context {
            Some(context) => upsert_continuation_context(&conn, task_id, &context)?,
            None => {
                conn.execute(
                    "DELETE FROM task_continuation_contexts WHERE task_id = ?1",
                    params![task_id],
                )
                .context("delete task continuation context")?;
            }
        }
        Ok(())
    }

    async fn get_continuation_context(
        &self,
        task_id: &str,
    ) -> Result<Option<TaskContinuationContext>> {
        let conn = self.conn.lock();
        ensure_goal_task_row(&conn, task_id)?;
        conn.query_row(
            "SELECT context_json
             FROM task_continuation_contexts WHERE task_id = ?1",
            params![task_id],
            |row| continuation_context_from_db(row.get("context_json")?),
        )
        .optional()
        .context("get task continuation context")
    }

    async fn current_goal_for_session(&self, session_id: &str) -> Result<Option<TaskRecord>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT * FROM tasks WHERE kind = 'goal' AND session_id = ?1",
            params![session_id],
            row_to_record,
        )
        .optional()
        .context("get current session goal")
    }

    async fn current_goal_status_for_session(
        &self,
        session_id: &str,
    ) -> Result<Option<GoalTaskStatus>> {
        let conn = self.conn.lock();
        conn.query_row(
            "SELECT tasks.*, tasks.error AS terminal_error, goal_tasks.*
               FROM tasks JOIN goal_tasks ON goal_tasks.task_id = tasks.id
              WHERE tasks.kind = 'goal' AND tasks.session_id = ?1",
            params![session_id],
            |row| {
                let terminal_error: Option<String> = row.get("terminal_error")?;
                Ok(GoalTaskStatus {
                    task: row_to_record(row)?,
                    goal: row_to_goal_task(row)?,
                    terminal_error: terminal_error
                        .map(|error| zeroclaw_providers::sanitize_api_error(&error)),
                })
            },
        )
        .optional()
        .context("get current session goal status")
    }

    async fn create_or_replace_session_goal(
        &self,
        mut task: TaskRecord,
        goal: GoalTaskRecord,
    ) -> Result<GoalTransitionResult> {
        ensure_goal_task_identity(&task, &goal)?;
        ensure_fresh_session_goal(&task, &goal)?;
        let session_id = task
            .session_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .context("new goal requires a nonblank session_id")?
            .to_owned();
        task.execution_epoch = 1;

        let mut conn = self.conn.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .context("start immediate session goal admission")?;
        let proposed_id_exists: bool = tx
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM tasks WHERE id = ?1)",
                params![&task.id],
                |row| row.get(0),
            )
            .context("check proposed goal id")?;
        if proposed_id_exists {
            return Ok(GoalTransitionResult::Stale);
        }

        let existing: Option<(String, String, String, Option<String>, Option<String>)> = tx
            .query_row(
                "SELECT id, status, agent, originator_route, principal_id
                   FROM tasks WHERE kind = 'goal' AND session_id = ?1",
                params![&session_id],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .optional()
            .context("read current session goal")?;
        if let Some((existing_id, status, agent, route, principal_id)) = existing {
            if agent != task.agent
                || route != task.originator_route
                || principal_id != task.principal_id
            {
                return Ok(GoalTransitionResult::Stale);
            }
            if !matches!(
                status.as_str(),
                "completed" | "failed" | "cancelled" | "lost" | "timed_out"
            ) {
                return Ok(GoalTransitionResult::Stale);
            }
            let pending: bool = tx
                .query_row(
                    "SELECT EXISTS(
                        SELECT 1 FROM goal_tasks
                         WHERE task_id = ?1
                           AND (pending_call_id IS NOT NULL
                                OR pending_call_epoch IS NOT NULL
                                OR pending_reserved_tokens IS NOT NULL
                                OR pending_reserved_cost_usd IS NOT NULL)
                     )",
                    params![&existing_id],
                    |row| row.get(0),
                )
                .context("check terminal session goal settlement")?;
            if pending {
                return Ok(GoalTransitionResult::Stale);
            }
            let deleted = tx
                .execute("DELETE FROM tasks WHERE id = ?1", params![existing_id])
                .context("dispose terminal session goal")?;
            if deleted != 1 {
                anyhow::bail!("terminal session goal disposal changed {deleted} rows");
            }
        }
        insert_task_record(&tx, task)?;
        insert_goal_task_record(&tx, goal)?;
        tx.commit().context("commit session goal admission")?;
        Ok(GoalTransitionResult::Applied)
    }

    async fn pause_session_goal(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        pause: GoalPauseState,
    ) -> Result<GoalTransitionResult> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().context("start session goal pause")?;
        let changed = tx.execute(
            "UPDATE tasks
                SET status = 'paused', execution_epoch = execution_epoch + 1
              WHERE id = ?1 AND kind = 'goal' AND session_id = ?2
                AND status = 'running' AND execution_epoch = ?3
                AND execution_epoch < 9223372036854775807",
            params![task_id, session_id, expected_epoch],
        )?;
        if changed == 0 {
            return goal_transition_absence(&tx, task_id);
        }
        if update_goal_pause_record(&tx, task_id, Some(pause))? != 1 {
            anyhow::bail!("goal task {task_id} has no extension row");
        }
        tx.commit().context("commit session goal pause")?;
        Ok(GoalTransitionResult::Applied)
    }

    async fn resume_session_goal(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        owner_pid: u32,
        owner_boot_id: &str,
    ) -> Result<GoalTransitionResult> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().context("start session goal resume")?;
        let changed = tx.execute(
            "UPDATE tasks
                SET status = 'running', owner_pid = ?4, owner_boot_id = ?5,
                    heartbeat_at = NULL, execution_epoch = execution_epoch + 1
              WHERE id = ?1 AND kind = 'goal' AND session_id = ?2
                AND status = 'paused' AND execution_epoch = ?3
                AND execution_epoch < 9223372036854775807
                AND EXISTS (
                    SELECT 1 FROM goal_tasks
                     WHERE task_id = tasks.id AND accounting_state = 'complete'
                       AND pending_call_id IS NULL AND pending_call_epoch IS NULL
                       AND pending_reserved_tokens IS NULL AND pending_reserved_cost_usd IS NULL
                       AND resume_safety = 'clean'
                )",
            params![
                task_id,
                session_id,
                expected_epoch,
                owner_pid as i64,
                owner_boot_id
            ],
        )?;
        if changed == 0 {
            return goal_transition_absence(&tx, task_id);
        }
        if update_goal_pause_record(&tx, task_id, None)? != 1 {
            anyhow::bail!("goal task {task_id} has no extension row");
        }
        tx.commit().context("commit session goal resume")?;
        Ok(GoalTransitionResult::Applied)
    }

    async fn finish_session_goal(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        status: TaskStatus,
        error: Option<String>,
    ) -> Result<GoalTransitionResult> {
        if !status.is_terminal() {
            anyhow::bail!("session goal finish requires a terminal status");
        }
        let conn = self.conn.lock();
        let changed = conn.execute(
            "UPDATE tasks
                SET status = ?4, error = COALESCE(?5, error),
                    finished_at = COALESCE(finished_at, ?6),
                    execution_epoch = execution_epoch + 1
              WHERE id = ?1 AND kind = 'goal' AND session_id = ?2
                AND execution_epoch = ?3 AND execution_epoch < 9223372036854775807
                AND status IN ('running','paused')
                AND (?4 != 'completed' OR status = 'running')
                AND EXISTS (SELECT 1 FROM goal_tasks WHERE task_id = tasks.id)
                AND (?4 != 'completed' OR EXISTS (
                    SELECT 1 FROM goal_tasks
                     WHERE task_id = tasks.id AND accounting_state = 'complete'
                       AND pending_call_id IS NULL AND pending_call_epoch IS NULL
                       AND pending_reserved_tokens IS NULL AND pending_reserved_cost_usd IS NULL
                ))",
            params![
                task_id,
                session_id,
                expected_epoch,
                super::status_to_db(status),
                error,
                chrono::Utc::now().to_rfc3339(),
            ],
        )?;
        if changed == 1 {
            Ok(GoalTransitionResult::Applied)
        } else {
            goal_transition_absence(&conn, task_id)
        }
    }

    async fn admit_pending_call(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        call_id: &str,
    ) -> Result<GoalTransitionResult> {
        self.admit_pending_call_with_reservation(
            task_id,
            session_id,
            expected_epoch,
            call_id,
            None,
            None,
        )
        .await
    }

    async fn admit_pending_call_with_reservation(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        call_id: &str,
        reserved_tokens: Option<u64>,
        reserved_cost_usd: Option<f64>,
    ) -> Result<GoalTransitionResult> {
        let reserved_tokens = reserved_tokens.map(token_limit_to_db).transpose()?;
        let reserved_cost_usd = reserved_cost_usd.map(cost_limit_to_db).transpose()?;
        let conn = self.conn.lock();
        let changed = conn.execute(
            "UPDATE goal_tasks
                SET pending_call_id = ?4, pending_call_epoch = ?3,
                    pending_reserved_tokens = ?5, pending_reserved_cost_usd = ?6
              WHERE task_id = ?1 AND accounting_state = 'complete'
                AND pending_call_id IS NULL AND pending_call_epoch IS NULL
                AND pending_reserved_tokens IS NULL AND pending_reserved_cost_usd IS NULL
                AND (effective_token_limit IS NULL OR ?5 IS NOT NULL)
                AND (effective_cost_limit_usd IS NULL OR ?6 IS NOT NULL)
                AND EXISTS (
                    SELECT 1 FROM tasks
                     WHERE id = ?1 AND kind = 'goal' AND session_id = ?2
                       AND status = 'running' AND execution_epoch = ?3
                )",
            params![
                task_id,
                session_id,
                expected_epoch,
                call_id,
                reserved_tokens,
                reserved_cost_usd,
            ],
        )?;
        if changed == 1 {
            Ok(GoalTransitionResult::Applied)
        } else {
            goal_transition_absence(&conn, task_id)
        }
    }

    async fn settle_pending_call(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        call_id: &str,
        accounting_state: GoalAccountingState,
    ) -> Result<GoalTransitionResult> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().context("start pending settlement")?;
        let changed = tx.execute(
            "UPDATE goal_tasks
                SET pending_call_id = CASE WHEN ?5 = 'outcome_unknown' THEN pending_call_id ELSE NULL END,
                    pending_call_epoch = CASE WHEN ?5 = 'outcome_unknown' THEN pending_call_epoch ELSE NULL END,
                    pending_reserved_tokens = CASE WHEN ?5 = 'outcome_unknown' THEN pending_reserved_tokens ELSE NULL END,
                    pending_reserved_cost_usd = CASE WHEN ?5 = 'outcome_unknown' THEN pending_reserved_cost_usd ELSE NULL END,
                    accounting_state = ?5
              WHERE task_id = ?1 AND pending_call_id = ?4
                AND pending_call_epoch = ?3
                AND EXISTS (
                    SELECT 1 FROM tasks
                     WHERE id = ?1 AND kind = 'goal' AND session_id = ?2
                       AND execution_epoch >= ?3
                )",
            params![
                task_id,
                session_id,
                expected_epoch,
                call_id,
                accounting_state_to_db(accounting_state),
            ],
        )?;
        if changed == 0 {
            return goal_transition_absence(&tx, task_id);
        }
        if accounting_state == GoalAccountingState::OutcomeUnknown {
            tx.execute(
                "UPDATE tasks
                    SET status = 'failed', error = COALESCE(error, 'accounting_outcome_unknown'),
                        finished_at = COALESCE(finished_at, ?3),
                        execution_epoch = CASE WHEN execution_epoch < 9223372036854775807
                            THEN execution_epoch + 1 ELSE execution_epoch END
                  WHERE id = ?1 AND kind = 'goal' AND session_id = ?2
                    AND status IN ('running', 'paused') AND execution_epoch >= ?4",
                params![
                    task_id,
                    session_id,
                    chrono::Utc::now().to_rfc3339(),
                    expected_epoch
                ],
            )?;
        }
        tx.commit().context("commit pending settlement")?;
        Ok(GoalTransitionResult::Applied)
    }

    async fn mark_goal_tool_executed(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
    ) -> Result<GoalTransitionResult> {
        let conn = self.conn.lock();
        let changed = conn.execute(
            "UPDATE goal_tasks
                SET resume_safety = 'tool_executed'
              WHERE task_id = ?1 AND resume_safety = 'clean'
                AND EXISTS (
                    SELECT 1 FROM tasks
                     WHERE id = ?1 AND kind = 'goal' AND session_id = ?2
                       AND status = 'running' AND execution_epoch = ?3
                )",
            params![task_id, session_id, expected_epoch],
        )?;
        if changed == 1 {
            Ok(GoalTransitionResult::Applied)
        } else {
            goal_transition_absence(&conn, task_id)
        }
    }

    async fn mark_completed_session_goal_delivered(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
    ) -> Result<GoalTransitionResult> {
        let conn = self.conn.lock();
        let changed = conn.execute(
            "UPDATE tasks
                SET delivered = 1
              WHERE id = ?1 AND kind = 'goal' AND session_id = ?2
                AND status = 'completed' AND execution_epoch = ?3 AND delivered = 0
                AND EXISTS (SELECT 1 FROM goal_tasks WHERE task_id = tasks.id)",
            params![task_id, session_id, expected_epoch],
        )?;
        if changed == 1 {
            return Ok(GoalTransitionResult::Applied);
        }
        let already_delivered = conn.query_row(
            "SELECT EXISTS(
                    SELECT 1 FROM tasks
                     WHERE id = ?1 AND kind = 'goal' AND session_id = ?2
                       AND status = 'completed' AND execution_epoch = ?3 AND delivered = 1
                       AND EXISTS (SELECT 1 FROM goal_tasks WHERE task_id = tasks.id)
                 )",
            params![task_id, session_id, expected_epoch],
            |row| row.get::<_, i64>(0),
        )? != 0;
        if already_delivered {
            Ok(GoalTransitionResult::Applied)
        } else {
            goal_transition_absence(&conn, task_id)
        }
    }

    async fn update_session_goal_limits(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
        token_limit: Option<u64>,
        cost_limit_usd: Option<f64>,
    ) -> Result<GoalTransitionResult> {
        let (tokens, cost) = goal_limits_to_db(token_limit, cost_limit_usd)?;
        let conn = self.conn.lock();
        let changed = conn.execute(
            "UPDATE goal_tasks
                SET effective_token_limit = ?4, effective_cost_limit_usd = ?5
              WHERE task_id = ?1 AND EXISTS (
                    SELECT 1 FROM tasks
                     WHERE id = ?1 AND kind = 'goal' AND session_id = ?2
                       AND execution_epoch = ?3 AND status IN ('running','paused')
              )",
            params![task_id, session_id, expected_epoch, tokens, cost],
        )?;
        if changed == 1 {
            Ok(GoalTransitionResult::Applied)
        } else {
            goal_transition_absence(&conn, task_id)
        }
    }

    async fn delete_session_goal(
        &self,
        task_id: &str,
        session_id: &str,
        expected_epoch: i64,
    ) -> Result<GoalTransitionResult> {
        let conn = self.conn.lock();
        let changed = conn.execute(
            "DELETE FROM tasks
              WHERE id = ?1 AND kind = 'goal' AND session_id = ?2
                AND execution_epoch = ?3
                AND status IN ('completed','failed','cancelled','lost','timed_out')
                AND EXISTS (
                    SELECT 1 FROM goal_tasks
                     WHERE task_id = ?1
                       AND pending_call_id IS NULL
                       AND pending_call_epoch IS NULL
                       AND pending_reserved_tokens IS NULL
                       AND pending_reserved_cost_usd IS NULL
                )",
            params![task_id, session_id, expected_epoch],
        )?;
        if changed == 1 {
            Ok(GoalTransitionResult::Applied)
        } else {
            goal_transition_absence(&conn, task_id)
        }
    }
}

fn goal_transition_absence(conn: &Connection, task_id: &str) -> Result<GoalTransitionResult> {
    let exists: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM tasks WHERE id = ?1 AND kind = 'goal')",
        params![task_id],
        |row| row.get(0),
    )?;
    Ok(if exists {
        GoalTransitionResult::Stale
    } else {
        GoalTransitionResult::Missing
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_plane::goal_task::{GoalBlockerKind, GoalTaskRegistry};
    use crate::control_plane::task_registry::{TaskKind, TaskRegistry, TaskStatus};
    use rusqlite::params;

    fn rec(id: &str, agent: &str, owner_pid: u32, boot: &str) -> TaskRecord {
        TaskRecord {
            id: id.into(),
            kind: TaskKind::Delegate,
            agent: agent.into(),
            status: TaskStatus::Running,
            owner_pid,
            owner_boot_id: boot.into(),
            heartbeat_at: None,
            depth: 0,
            parent_id: None,
            originator_route: None,
            delivered: false,
            idem_key: None,
            principal_id: None,
            session_id: Some(format!("session-{id}")),
            execution_epoch: 0,
            started_at: "2026-06-18T00:00:00Z".into(),
            finished_at: None,
        }
    }

    fn goal_record(task_id: &str, objective: &str) -> GoalTaskRecord {
        GoalTaskRecord {
            task_id: task_id.into(),
            objective: objective.into(),
            success_criteria: "complete".into(),
            effective_token_limit: None,
            effective_cost_limit_usd: None,
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
            pending_call_id: None,
            pending_call_epoch: None,
            pending_reserved_tokens: None,
            pending_reserved_cost_usd: None,
            accounting_state: GoalAccountingState::Complete,
            resume_safety: GoalResumeSafety::Clean,
        }
    }

    fn continuation_context() -> TaskContinuationContext {
        TaskContinuationContext {
            channel: "telegram".into(),
            channel_alias: Some("main".into()),
            reply_target: "chat-1".into(),
            sender: "alice".into(),
            thread_ts: None,
            interruption_scope_id: Some("scope-1".into()),
            conversation_scope:
                crate::control_plane::goal_task::TaskContinuationConversationScope::Sender,
        }
    }

    #[tokio::test]
    async fn latest_active_goal_for_agent_resolves_in_sql() {
        let s = SqliteTaskStore::new_in_memory().unwrap();

        let mut old_goal = rec("old-goal", "main", 1, "boot-1");
        old_goal.kind = TaskKind::Goal;
        old_goal.originator_route = Some("route-old".into());
        old_goal.started_at = "2026-06-18T00:00:00Z".into();
        s.create(old_goal).await.unwrap();

        let mut newer_terminal_goal = rec("done-goal", "main", 1, "boot-1");
        newer_terminal_goal.kind = TaskKind::Goal;
        newer_terminal_goal.started_at = "2026-06-20T00:00:00Z".into();
        s.create(newer_terminal_goal).await.unwrap();
        s.update_status("done-goal", TaskStatus::Completed, None, None)
            .await
            .unwrap();

        let mut newest_delegate = rec("newest-delegate", "main", 1, "boot-1");
        newest_delegate.started_at = "2026-06-21T00:00:00Z".into();
        s.create(newest_delegate).await.unwrap();

        let mut latest_active_goal = rec("latest-goal", "main", 1, "boot-1");
        latest_active_goal.kind = TaskKind::Goal;
        latest_active_goal.started_at = "2026-06-19T00:00:00Z".into();
        s.create(latest_active_goal).await.unwrap();

        let got = s
            .latest_active_goal_for_agent("main")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(got.id, "latest-goal");
        assert!(
            s.latest_active_goal_for_agent("other")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn latest_active_goal_for_context_filters_route_and_principal() {
        let s = SqliteTaskStore::new_in_memory().unwrap();

        let mut other_route_goal = rec("other-route", "main", 1, "boot-1");
        other_route_goal.kind = TaskKind::Goal;
        other_route_goal.originator_route = Some("route-b".into());
        other_route_goal.principal_id = Some("principal-a".into());
        other_route_goal.started_at = "2026-06-20T00:00:00Z".into();
        s.create(other_route_goal).await.unwrap();

        let mut other_principal_goal = rec("other-principal", "main", 1, "boot-1");
        other_principal_goal.kind = TaskKind::Goal;
        other_principal_goal.originator_route = Some("route-a".into());
        other_principal_goal.principal_id = Some("principal-b".into());
        other_principal_goal.started_at = "2026-06-19T00:00:00Z".into();
        s.create(other_principal_goal).await.unwrap();

        let mut wanted_goal = rec("wanted", "main", 1, "boot-1");
        wanted_goal.kind = TaskKind::Goal;
        wanted_goal.originator_route = Some("route-a".into());
        wanted_goal.principal_id = Some("principal-a".into());
        wanted_goal.started_at = "2026-06-18T00:00:00Z".into();
        s.create(wanted_goal).await.unwrap();

        let got = s
            .latest_active_goal_for_context("main", Some("route-a"), Some("principal-a"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(got.id, "wanted");
        let got_id = s
            .latest_active_goal_id_for_context("main", Some("route-a"), Some("principal-a"))
            .await
            .unwrap();
        assert_eq!(got_id.as_deref(), Some("wanted"));
    }

    #[tokio::test]
    async fn latest_active_goal_breaks_started_at_ties_by_insert_order() {
        let s = SqliteTaskStore::new_in_memory().unwrap();

        let mut first_goal = rec("first-goal", "main", 1, "boot-1");
        first_goal.kind = TaskKind::Goal;
        first_goal.originator_route = Some("route-a".into());
        first_goal.principal_id = Some("principal-a".into());
        first_goal.started_at = "2026-06-19T00:00:00Z".into();
        s.create(first_goal).await.unwrap();

        let mut second_goal = rec("second-goal", "main", 1, "boot-1");
        second_goal.kind = TaskKind::Goal;
        second_goal.originator_route = Some("route-b".into());
        second_goal.principal_id = Some("principal-b".into());
        second_goal.started_at = "2026-06-19T00:00:00Z".into();
        s.create(second_goal).await.unwrap();

        let got = s
            .latest_active_goal_for_agent("main")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(got.id, "second-goal");

        let got = s
            .latest_active_goal_for_context("main", None, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(got.id, "second-goal");

        let got_id = s
            .latest_active_goal_id_for_context("main", None, None)
            .await
            .unwrap();
        assert_eq!(got_id.as_deref(), Some("second-goal"));
    }

    #[tokio::test]
    async fn latest_active_goal_for_context_ignores_unknown_status_candidate() {
        let s = SqliteTaskStore::new_in_memory().unwrap();

        let mut older_valid_goal = rec("older-valid-goal", "main", 1, "boot-1");
        older_valid_goal.kind = TaskKind::Goal;
        older_valid_goal.originator_route = Some("route-a".into());
        older_valid_goal.principal_id = Some("principal-a".into());
        older_valid_goal.started_at = "2026-06-19T00:00:00Z".into();
        s.create(older_valid_goal).await.unwrap();

        let mut unreadable_newer_goal = rec("unreadable-newer-goal", "main", 1, "boot-1");
        unreadable_newer_goal.kind = TaskKind::Goal;
        unreadable_newer_goal.originator_route = Some("route-a".into());
        unreadable_newer_goal.principal_id = Some("principal-a".into());
        unreadable_newer_goal.started_at = "2026-06-20T00:00:00Z".into();
        s.create(unreadable_newer_goal).await.unwrap();
        {
            let conn = s.conn.lock();
            conn.execute(
                "UPDATE tasks SET status = 'future_paused' WHERE id = ?1",
                params!["unreadable-newer-goal"],
            )
            .unwrap();
        }

        let got = s
            .latest_active_goal_for_context("main", Some("route-a"), Some("principal-a"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(got.id, "older-valid-goal");
        let got_id = s
            .latest_active_goal_id_for_context("main", Some("route-a"), Some("principal-a"))
            .await
            .unwrap();
        assert_eq!(got_id.as_deref(), Some("older-valid-goal"));
    }

    #[tokio::test]
    async fn latest_active_goal_for_agent_ignores_unknown_status_candidate() {
        let s = SqliteTaskStore::new_in_memory().unwrap();

        let mut older_valid_goal = rec("older-valid-goal", "main", 1, "boot-1");
        older_valid_goal.kind = TaskKind::Goal;
        older_valid_goal.originator_route = Some("route-valid".into());
        older_valid_goal.started_at = "2026-06-19T00:00:00Z".into();
        s.create(older_valid_goal).await.unwrap();

        let mut unreadable_newer_goal = rec("unreadable-newer-goal", "main", 1, "boot-1");
        unreadable_newer_goal.kind = TaskKind::Goal;
        unreadable_newer_goal.started_at = "2026-06-20T00:00:00Z".into();
        s.create(unreadable_newer_goal).await.unwrap();
        {
            let conn = s.conn.lock();
            conn.execute(
                "UPDATE tasks SET status = 'future_paused' WHERE id = ?1",
                params!["unreadable-newer-goal"],
            )
            .unwrap();
        }

        let got = s
            .latest_active_goal_for_agent("main")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(got.id, "older-valid-goal");
    }

    #[tokio::test]
    async fn goal_task_extension_roundtrips_without_duplicating_lifecycle() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        let mut task = rec("goal-1", "main", 1, "boot-1");
        task.kind = TaskKind::Goal;
        task.status = TaskStatus::Paused;
        s.create_goal(
            task,
            GoalTaskRecord {
                task_id: "goal-1".into(),
                objective: "ship goal mode".into(),
                success_criteria: "complete".into(),
                effective_token_limit: Some(10_000),
                effective_cost_limit_usd: Some(1.25),
                pause_reason: Some(GoalPauseReason::NeedsUserInput),
                pause_description: Some("waiting for operator".into()),
                blockers: vec![GoalBlocker {
                    kind: GoalBlockerKind::NeedsUserInput,
                    message: "Need operator answer".into(),
                    payload: Some(serde_json::json!({"question": "continue?"})),
                }],
                pending_call_id: None,
                pending_call_epoch: None,
                pending_reserved_tokens: None,
                pending_reserved_cost_usd: None,
                accounting_state: GoalAccountingState::Complete,
                resume_safety: GoalResumeSafety::Clean,
            },
            None,
        )
        .await
        .unwrap();

        let task = s.get("goal-1").await.unwrap().unwrap();
        let goal = s.get_goal_task("goal-1").await.unwrap().unwrap();

        assert_eq!(task.kind, TaskKind::Goal);
        assert_eq!(task.status, TaskStatus::Paused);
        assert_eq!(goal.task_id, task.id);
        assert_eq!(goal.objective, "ship goal mode");
        assert_eq!(goal.effective_token_limit, Some(10_000));
        assert_eq!(goal.effective_cost_limit_usd, Some(1.25));
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::NeedsUserInput));
        assert_eq!(
            goal.pause_description.as_deref(),
            Some("waiting for operator")
        );
        assert_eq!(goal.blockers.len(), 1);

        s.update_goal_limits("goal-1", Some(20_000), None)
            .await
            .unwrap();
        let limited = s.get_goal_task("goal-1").await.unwrap().unwrap();
        assert_eq!(limited.effective_token_limit, Some(20_000));
        assert_eq!(limited.effective_cost_limit_usd, None);
        assert_eq!(
            limited.objective, "ship goal mode",
            "budget updates must not duplicate or rewrite objective state"
        );

        let err = s
            .update_goal_objective("goal-1", "ship amended goal mode")
            .await
            .expect_err("session-bound Goal objectives are immutable");
        assert!(format!("{err:#}").contains("objective cannot be changed"));
        assert_eq!(
            s.get_goal_task("goal-1").await.unwrap().unwrap().objective,
            "ship goal mode"
        );

        s.update_goal_pause("goal-1", None).await.unwrap();
        let resumed = s.get_goal_task("goal-1").await.unwrap().unwrap();
        assert!(resumed.pause_reason.is_none());
        assert!(resumed.pause_description.is_none());
        assert!(resumed.blockers.is_empty());
    }

    #[tokio::test]
    async fn goal_task_requires_canonical_goal_task_record() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        let err = {
            let conn = s.conn.lock();
            insert_goal_task_record(
                &conn,
                GoalTaskRecord {
                    task_id: "missing".into(),
                    objective: "orphan objective".into(),
                    success_criteria: "complete".into(),
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                    pending_call_id: None,
                    pending_call_epoch: None,
                    pending_reserved_tokens: None,
                    pending_reserved_cost_usd: None,
                    accounting_state: GoalAccountingState::Complete,
                    resume_safety: GoalResumeSafety::Clean,
                },
            )
        }
        .unwrap_err();

        assert!(format!("{err:#}").contains("TaskKind::Goal"));
    }

    #[tokio::test]
    async fn goal_task_extension_rejects_non_goal_task_record() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        s.create(rec("delegate-1", "main", 1, "boot-1"))
            .await
            .unwrap();

        let err = {
            let conn = s.conn.lock();
            insert_goal_task_record(
                &conn,
                GoalTaskRecord {
                    task_id: "delegate-1".into(),
                    objective: "must not attach to a delegate".into(),
                    success_criteria: "complete".into(),
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                    pending_call_id: None,
                    pending_call_epoch: None,
                    pending_reserved_tokens: None,
                    pending_reserved_cost_usd: None,
                    accounting_state: GoalAccountingState::Complete,
                    resume_safety: GoalResumeSafety::Clean,
                },
            )
        }
        .unwrap_err();

        assert!(format!("{err:#}").contains("TaskKind::Goal"));
    }

    #[tokio::test]
    async fn create_goal_rejects_non_goal_task_record() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        let task = rec("delegate-goal", "main", 1, "boot-1");

        let err = s
            .create_goal(
                task,
                GoalTaskRecord {
                    task_id: "delegate-goal".into(),
                    objective: "should fail before insert".into(),
                    success_criteria: "complete".into(),
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                    pending_call_id: None,
                    pending_call_epoch: None,
                    pending_reserved_tokens: None,
                    pending_reserved_cost_usd: None,
                    accounting_state: GoalAccountingState::Complete,
                    resume_safety: GoalResumeSafety::Clean,
                },
                None,
            )
            .await
            .expect_err("non-goal task records must not create goal extensions");

        assert!(format!("{err:#}").contains("TaskKind::Goal"));
        assert!(
            s.get("delegate-goal").await.unwrap().is_none(),
            "pre-validation failure must not create a generic task row"
        );
    }

    #[tokio::test]
    async fn create_goal_rejects_mismatched_task_and_goal_ids() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        let mut task = rec("goal-task", "main", 1, "boot-1");
        task.kind = TaskKind::Goal;

        let err = s
            .create_goal(
                task,
                GoalTaskRecord {
                    task_id: "missing-extension-parent".into(),
                    objective: "should roll back".into(),
                    success_criteria: "complete".into(),
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                    pending_call_id: None,
                    pending_call_epoch: None,
                    pending_reserved_tokens: None,
                    pending_reserved_cost_usd: None,
                    accounting_state: GoalAccountingState::Complete,
                    resume_safety: GoalResumeSafety::Clean,
                },
                None,
            )
            .await
            .expect_err("goal task id mismatch must be rejected before insert");

        assert!(format!("{err:#}").contains("id mismatch"));
        assert!(
            s.get("goal-task").await.unwrap().is_none(),
            "pre-validation failure must not create a generic task row"
        );
    }

    #[tokio::test]
    async fn update_goal_limits_rejects_invalid_effective_limits() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        let mut task = rec("goal-limit-validation", "main", 1, "boot-1");
        task.kind = TaskKind::Goal;
        s.create_goal(
            task,
            goal_record("goal-limit-validation", "validate limits"),
            None,
        )
        .await
        .unwrap();

        let err = s
            .update_goal_limits("goal-limit-validation", Some(i64::MAX as u64 + 1), None)
            .await
            .expect_err("token limit outside SQLite INTEGER domain must fail");
        assert!(format!("{err:#}").contains("SQLite INTEGER domain"));

        for invalid in [-1.0, f64::INFINITY, f64::NAN] {
            let err = s
                .update_goal_limits("goal-limit-validation", None, Some(invalid))
                .await
                .expect_err("invalid cost limit must fail before SQLite bind");
            assert!(format!("{err:#}").contains("non-negative finite"));
        }
    }

    #[tokio::test]
    async fn sqlite_rejects_invalid_effective_limit_rows() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        let mut task = rec("goal-sqlite-limit-validation", "main", 1, "boot-1");
        task.kind = TaskKind::Goal;
        s.create_goal(
            task,
            goal_record("goal-sqlite-limit-validation", "validate SQLite limits"),
            None,
        )
        .await
        .unwrap();

        let conn = s.conn.lock();
        let err = conn
            .execute(
                "UPDATE goal_tasks
                    SET effective_token_limit = -1
                  WHERE task_id = ?1",
                params!["goal-sqlite-limit-validation"],
            )
            .expect_err("SQLite must reject negative token limits");
        assert!(format!("{err:#}").contains("non-negative finite"));

        let err = conn
            .execute(
                "UPDATE goal_tasks
                    SET effective_cost_limit_usd = -0.01
                  WHERE task_id = ?1",
                params!["goal-sqlite-limit-validation"],
            )
            .expect_err("SQLite must reject negative cost limits");
        assert!(format!("{err:#}").contains("non-negative finite"));

        let err = conn
            .execute(
                "UPDATE goal_tasks
                    SET effective_cost_limit_usd = 1e999
                  WHERE task_id = ?1",
                params!["goal-sqlite-limit-validation"],
            )
            .expect_err("SQLite must reject non-finite cost limits");
        assert!(format!("{err:#}").contains("non-negative finite"));
    }

    #[tokio::test]
    async fn get_goal_task_rejects_legacy_negative_token_limit() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        let mut task = rec("goal-corrupt-limit", "main", 1, "boot-1");
        task.kind = TaskKind::Goal;
        s.create_goal(
            task,
            goal_record("goal-corrupt-limit", "reject legacy corrupt limit"),
            None,
        )
        .await
        .unwrap();

        {
            let conn = s.conn.lock();
            conn.execute("DROP TRIGGER trg_goal_tasks_effective_limits_update", [])
                .unwrap();
            conn.execute(
                "UPDATE goal_tasks
                    SET effective_token_limit = -1
                  WHERE task_id = ?1",
                params!["goal-corrupt-limit"],
            )
            .unwrap();
        }

        let err = s
            .get_goal_task("goal-corrupt-limit")
            .await
            .expect_err("legacy negative token limits must not inflate to u64");
        assert!(format!("{err:#}").contains("get goal task"));
    }

    #[tokio::test]
    async fn goal_extension_updates_reject_missing_goal_rows() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        let mut task = rec("goal-without-extension", "main", 1, "boot-1");
        task.kind = TaskKind::Goal;
        s.create(task).await.unwrap();

        let err = s
            .update_goal_limits("goal-without-extension", Some(100), None)
            .await
            .expect_err("limit updates must reject missing goal extension rows");
        assert!(format!("{err:#}").contains("goal extension"));

        let err = s
            .update_goal_pause(
                "goal-without-extension",
                Some(GoalPauseState {
                    reason: GoalPauseReason::NeedsUserInput,
                    description: None,
                    blockers: Vec::new(),
                }),
            )
            .await
            .expect_err("pause updates must reject missing goal extension rows");
        assert!(format!("{err:#}").contains("goal extension"));
    }

    #[tokio::test]
    async fn continuation_context_requires_goal_extension_row() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        s.create(rec("delegate-context", "main", 1, "boot-1"))
            .await
            .unwrap();

        let err = s
            .set_continuation_context("delegate-context", Some(continuation_context()))
            .await
            .expect_err("continuation context must reject non-goal tasks");
        assert!(format!("{err:#}").contains("goal extension"));

        let err = s
            .get_continuation_context("delegate-context")
            .await
            .expect_err("continuation context reads must reject non-goal tasks");
        assert!(format!("{err:#}").contains("goal extension"));

        let conn = s.conn.lock();
        let err = conn
            .execute(
                "INSERT INTO task_continuation_contexts (task_id, context_json)
                    VALUES (?1, ?2)",
                params![
                    "delegate-context",
                    continuation_context_to_db(&continuation_context()).unwrap()
                ],
            )
            .expect_err("SQLite must reject continuation contexts for non-goal tasks");
        assert!(format!("{err:#}").contains("goal task"));
    }

    #[tokio::test]
    async fn pause_goal_task_updates_status_and_pause_atomically() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        let mut task = rec("goal-paused", "main", 1, "boot-1");
        task.kind = TaskKind::Goal;
        s.create_goal(
            task,
            GoalTaskRecord {
                task_id: "goal-paused".into(),
                objective: "pause me".into(),
                success_criteria: "complete".into(),
                effective_token_limit: None,
                effective_cost_limit_usd: None,
                pause_reason: None,
                pause_description: None,
                blockers: Vec::new(),
                pending_call_id: None,
                pending_call_epoch: None,
                pending_reserved_tokens: None,
                pending_reserved_cost_usd: None,
                accounting_state: GoalAccountingState::Complete,
                resume_safety: GoalResumeSafety::Clean,
            },
            None,
        )
        .await
        .unwrap();

        s.pause_goal_task(
            "goal-paused",
            GoalPauseState {
                reason: GoalPauseReason::NeedsUserInput,
                description: Some("waiting".into()),
                blockers: vec![GoalBlocker {
                    kind: GoalBlockerKind::NeedsUserInput,
                    message: "Need operator answer".into(),
                    payload: None,
                }],
            },
        )
        .await
        .unwrap();

        let task = s.get("goal-paused").await.unwrap().unwrap();
        let goal = s.get_goal_task("goal-paused").await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Paused);
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::NeedsUserInput));
        assert_eq!(goal.pause_description.as_deref(), Some("waiting"));
        assert_eq!(goal.blockers.len(), 1);
    }

    #[tokio::test]
    async fn pause_goal_task_roundtrips_operator_pause() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        let mut task = rec("goal-operator-paused", "main", 1, "boot-1");
        task.kind = TaskKind::Goal;
        s.create_goal(
            task,
            GoalTaskRecord {
                task_id: "goal-operator-paused".into(),
                objective: "pause from operator command".into(),
                success_criteria: "complete".into(),
                effective_token_limit: None,
                effective_cost_limit_usd: None,
                pause_reason: None,
                pause_description: None,
                blockers: Vec::new(),
                pending_call_id: None,
                pending_call_epoch: None,
                pending_reserved_tokens: None,
                pending_reserved_cost_usd: None,
                accounting_state: GoalAccountingState::Complete,
                resume_safety: GoalResumeSafety::Clean,
            },
            None,
        )
        .await
        .unwrap();

        s.pause_goal_task(
            "goal-operator-paused",
            GoalPauseState {
                reason: GoalPauseReason::OperatorPaused,
                description: Some("maintenance window".into()),
                blockers: vec![GoalBlocker {
                    kind: GoalBlockerKind::OperatorPause,
                    message: "maintenance window".into(),
                    payload: None,
                }],
            },
        )
        .await
        .unwrap();

        let task = s.get("goal-operator-paused").await.unwrap().unwrap();
        let goal = s
            .get_goal_task("goal-operator-paused")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(task.status, TaskStatus::Paused);
        assert_eq!(goal.pause_reason, Some(GoalPauseReason::OperatorPaused));
        assert_eq!(
            goal.pause_description.as_deref(),
            Some("maintenance window")
        );
        assert_eq!(goal.blockers[0].kind, GoalBlockerKind::OperatorPause);
        assert_eq!(goal.blockers[0].message, "maintenance window");
    }

    #[tokio::test]
    async fn pause_goal_task_rejects_goal_without_extension_before_status_change() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        let mut task = rec("goal-missing-extension", "main", 1, "boot-1");
        task.kind = TaskKind::Goal;
        s.create(task).await.unwrap();

        let err = s
            .pause_goal_task(
                "goal-missing-extension",
                GoalPauseState {
                    reason: GoalPauseReason::NeedsUserInput,
                    description: None,
                    blockers: Vec::new(),
                },
            )
            .await
            .expect_err("goal pause must require the goal extension row");

        assert!(format!("{err:#}").contains("goal extension"));
        let task = s.get("goal-missing-extension").await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Running);
    }

    #[tokio::test]
    async fn resume_goal_task_clears_pause_claims_owner_and_sets_running() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        let mut task = rec("goal-resume", "main", 1, "boot-1");
        task.kind = TaskKind::Goal;
        task.status = TaskStatus::Paused;
        s.create_goal(
            task,
            GoalTaskRecord {
                task_id: "goal-resume".into(),
                objective: "resume me".into(),
                success_criteria: "complete".into(),
                effective_token_limit: None,
                effective_cost_limit_usd: None,
                pause_reason: Some(GoalPauseReason::NeedsUserInput),
                pause_description: Some("waiting".into()),
                blockers: vec![GoalBlocker {
                    kind: GoalBlockerKind::NeedsUserInput,
                    message: "Need operator answer".into(),
                    payload: None,
                }],
                pending_call_id: None,
                pending_call_epoch: None,
                pending_reserved_tokens: None,
                pending_reserved_cost_usd: None,
                accounting_state: GoalAccountingState::Complete,
                resume_safety: GoalResumeSafety::Clean,
            },
            None,
        )
        .await
        .unwrap();

        s.resume_goal_task("goal-resume", 42, "boot-2", None)
            .await
            .unwrap();

        let task = s.get("goal-resume").await.unwrap().unwrap();
        let goal = s.get_goal_task("goal-resume").await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Running);
        assert_eq!(task.owner_pid, 42);
        assert_eq!(task.owner_boot_id, "boot-2");
        assert!(goal.pause_reason.is_none());
        assert!(goal.pause_description.is_none());
        assert!(goal.blockers.is_empty());
    }

    #[tokio::test]
    async fn resume_goal_task_rejects_tool_executed_goal() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        let mut task = rec("goal-tool-executed", "main", 1, "boot-1");
        task.kind = TaskKind::Goal;
        task.status = TaskStatus::Paused;
        s.create_goal(
            task,
            GoalTaskRecord {
                task_id: "goal-tool-executed".into(),
                objective: "resume me".into(),
                success_criteria: "complete".into(),
                effective_token_limit: None,
                effective_cost_limit_usd: None,
                pause_reason: Some(GoalPauseReason::OperatorPaused),
                pause_description: None,
                blockers: Vec::new(),
                pending_call_id: None,
                pending_call_epoch: None,
                pending_reserved_tokens: None,
                pending_reserved_cost_usd: None,
                accounting_state: GoalAccountingState::Complete,
                resume_safety: GoalResumeSafety::ToolExecuted,
            },
            None,
        )
        .await
        .unwrap();

        let err = s
            .resume_goal_task("goal-tool-executed", 42, "boot-2", None)
            .await
            .expect_err("tool-executed goal must not resume");

        assert!(format!("{err:#}").contains("not safely resumable"));
        assert_eq!(
            s.get("goal-tool-executed").await.unwrap().unwrap().status,
            TaskStatus::Paused
        );
    }
}
