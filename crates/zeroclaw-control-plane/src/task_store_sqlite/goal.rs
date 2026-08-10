//! SQLite-backed [`GoalTaskRegistry`] implementation.

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};

#[cfg(test)]
use std::cell::Cell;

use crate::goal_task::{
    GoalAccountingState, GoalBlocker, GoalPauseReason, GoalPauseState, GoalTaskRecord,
    GoalTaskRegistry, GoalTransitionResult, TaskContinuationContext,
};
use crate::task_registry::{TaskKind, TaskRecord, TaskStatus};

use super::{
    SqliteTaskStore, add_column_if_missing, insert_task_record, insert_task_record_strict,
    log_unreadable_task_row, row_to_record, status_to_db,
};

#[cfg(test)]
thread_local! {
    static FAIL_IDENTITY_TRIGGER_REFRESH: Cell<bool> = const { Cell::new(false) };
}

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
        add_column_if_missing(
            conn,
            "tasks",
            "session_id",
            "ALTER TABLE tasks ADD COLUMN session_id TEXT",
        )?;
        add_column_if_missing(
            conn,
            "tasks",
            "execution_epoch",
            "ALTER TABLE tasks ADD COLUMN execution_epoch INTEGER NOT NULL DEFAULT 0",
        )?;
        add_column_if_missing(
            conn,
            "goal_tasks",
            "pending_call_id",
            "ALTER TABLE goal_tasks ADD COLUMN pending_call_id TEXT",
        )?;
        add_column_if_missing(
            conn,
            "goal_tasks",
            "pending_call_epoch",
            "ALTER TABLE goal_tasks ADD COLUMN pending_call_epoch INTEGER",
        )?;
        add_column_if_missing(
            conn,
            "goal_tasks",
            "accounting_state",
            "ALTER TABLE goal_tasks ADD COLUMN accounting_state TEXT NOT NULL DEFAULT 'complete'",
        )?;
        conn.execute_batch(
            "DROP INDEX IF EXISTS idx_tasks_active_goal_context;
             CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_goal_session
                ON tasks(session_id) WHERE kind = 'goal' AND session_id IS NOT NULL;
             CREATE UNIQUE INDEX IF NOT EXISTS idx_goal_tasks_pending_call
                ON goal_tasks(pending_call_id) WHERE pending_call_id IS NOT NULL;
             CREATE TRIGGER IF NOT EXISTS trg_tasks_goal_session_insert
                BEFORE INSERT ON tasks FOR EACH ROW
                WHEN NEW.kind = 'goal' AND (NEW.session_id IS NULL OR length(trim(NEW.session_id)) = 0)
                BEGIN SELECT RAISE(ABORT, 'goal tasks require a nonblank session_id'); END;
             CREATE TRIGGER IF NOT EXISTS trg_tasks_goal_session_rebind
                BEFORE UPDATE OF kind, session_id ON tasks FOR EACH ROW
                WHEN NEW.kind = 'goal' AND (NEW.session_id IS NULL OR length(trim(NEW.session_id)) = 0)
                     AND OLD.session_id IS NOT NULL
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
        ).context("apply control-plane schema v8")?;
    }
    if version < 9 {
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .context("start control-plane schema v9 transaction")?;
        tx.execute_batch(
            "DROP TRIGGER IF EXISTS trg_task_continuation_contexts_require_goal_insert;
             DROP TRIGGER IF EXISTS trg_task_continuation_contexts_require_goal_update;
             CREATE TABLE goal_tasks_v9 (
                 task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE RESTRICT,
                 objective TEXT NOT NULL,
                 effective_token_limit INTEGER,
                 effective_cost_limit_usd REAL,
                 pause_reason TEXT,
                 pause_description TEXT,
                 blockers_json TEXT NOT NULL DEFAULT '[]',
                 pending_call_id TEXT,
                 pending_call_epoch INTEGER,
                 accounting_state TEXT NOT NULL DEFAULT 'complete'
             );
             INSERT INTO goal_tasks_v9
                 SELECT task_id, objective, effective_token_limit, effective_cost_limit_usd,
                        pause_reason, pause_description, blockers_json, pending_call_id,
                        pending_call_epoch, accounting_state
                   FROM goal_tasks;
             DROP TABLE goal_tasks;
             ALTER TABLE goal_tasks_v9 RENAME TO goal_tasks;
             CREATE UNIQUE INDEX idx_goal_tasks_pending_call
                ON goal_tasks(pending_call_id) WHERE pending_call_id IS NOT NULL;
             CREATE TRIGGER trg_goal_tasks_require_goal_kind
                BEFORE INSERT ON goal_tasks FOR EACH ROW
                WHEN COALESCE((SELECT kind FROM tasks WHERE id = NEW.task_id), '') != 'goal'
                BEGIN SELECT RAISE(ABORT, 'goal_tasks.task_id must reference TaskKind::Goal'); END;
             CREATE TRIGGER trg_goal_tasks_effective_limits_insert BEFORE INSERT ON goal_tasks FOR EACH ROW
                WHEN (NEW.effective_token_limit IS NOT NULL AND NEW.effective_token_limit < 0)
                  OR (NEW.effective_cost_limit_usd IS NOT NULL AND NOT (NEW.effective_cost_limit_usd >= 0.0 AND NEW.effective_cost_limit_usd <= 1.7976931348623157e308))
                BEGIN SELECT RAISE(ABORT, 'goal effective limits must be non-negative finite values'); END;
             CREATE TRIGGER trg_goal_tasks_effective_limits_update BEFORE UPDATE OF effective_token_limit, effective_cost_limit_usd ON goal_tasks FOR EACH ROW
                WHEN (NEW.effective_token_limit IS NOT NULL AND NEW.effective_token_limit < 0)
                  OR (NEW.effective_cost_limit_usd IS NOT NULL AND NOT (NEW.effective_cost_limit_usd >= 0.0 AND NEW.effective_cost_limit_usd <= 1.7976931348623157e308))
                BEGIN SELECT RAISE(ABORT, 'goal effective limits must be non-negative finite values'); END;
             CREATE TRIGGER trg_goal_pending_pair_insert BEFORE INSERT ON goal_tasks FOR EACH ROW
                WHEN (NEW.pending_call_id IS NULL) != (NEW.pending_call_epoch IS NULL)
                BEGIN SELECT RAISE(ABORT, 'pending call id and epoch must be paired'); END;
             CREATE TRIGGER trg_goal_pending_pair_update BEFORE UPDATE OF pending_call_id, pending_call_epoch ON goal_tasks FOR EACH ROW
                WHEN (NEW.pending_call_id IS NULL) != (NEW.pending_call_epoch IS NULL)
                BEGIN SELECT RAISE(ABORT, 'pending call id and epoch must be paired'); END;
             CREATE TRIGGER trg_goal_accounting_state_insert BEFORE INSERT ON goal_tasks FOR EACH ROW
                WHEN NEW.accounting_state NOT IN ('complete','missing','invalid','outcome_unknown')
                BEGIN SELECT RAISE(ABORT, 'invalid goal accounting state'); END;
             CREATE TRIGGER trg_goal_accounting_state_update BEFORE UPDATE OF accounting_state ON goal_tasks FOR EACH ROW
                WHEN NEW.accounting_state NOT IN ('complete','missing','invalid','outcome_unknown')
                BEGIN SELECT RAISE(ABORT, 'invalid goal accounting state'); END;",
        ).context("apply control-plane schema v9")?;
        let has_continuation_table: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'task_continuation_contexts')", [], |row| row.get(0)).context("inspect continuation context table")?;
        if has_continuation_table {
            tx.execute_batch("CREATE TRIGGER trg_task_continuation_contexts_require_goal_insert BEFORE INSERT ON task_continuation_contexts FOR EACH ROW WHEN NOT EXISTS (SELECT 1 FROM tasks JOIN goal_tasks ON goal_tasks.task_id = tasks.id WHERE tasks.id = NEW.task_id AND tasks.kind = 'goal') BEGIN SELECT RAISE(ABORT, 'task_continuation_contexts.task_id must reference a goal task'); END; CREATE TRIGGER trg_task_continuation_contexts_require_goal_update BEFORE UPDATE OF task_id ON task_continuation_contexts FOR EACH ROW WHEN NOT EXISTS (SELECT 1 FROM tasks JOIN goal_tasks ON goal_tasks.task_id = tasks.id WHERE tasks.id = NEW.task_id AND tasks.kind = 'goal') BEGIN SELECT RAISE(ABORT, 'task_continuation_contexts.task_id must reference a goal task'); END;").context("restore continuation context triggers")?;
        }
        refresh_goal_identity_triggers(&tx)?;
        tx.execute_batch("PRAGMA user_version = 9;")
            .context("mark control-plane schema v9")?;
        tx.commit().context("commit control-plane schema v9")?;
    } else if version == 9 {
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .context("start control-plane schema v9 trigger refresh transaction")?;
        refresh_goal_identity_triggers(&tx)?;
        tx.commit()
            .context("commit control-plane schema v9 trigger refresh")?;
    }
    if version < 10 {
        conn.execute_batch(
            "CREATE TRIGGER IF NOT EXISTS trg_tasks_goal_delete_requires_extension_disposal
                BEFORE DELETE ON tasks FOR EACH ROW
                WHEN OLD.kind = 'goal'
                 AND EXISTS (SELECT 1 FROM goal_tasks WHERE task_id = OLD.id)
                BEGIN
                    SELECT RAISE(ABORT, 'goal task deletion requires extension disposal');
                END;
             CREATE TRIGGER IF NOT EXISTS trg_goal_tasks_delete_requires_authorization
                BEFORE DELETE ON goal_tasks FOR EACH ROW
                WHEN control_plane_goal_extension_delete_authorized() != 1
                BEGIN
                    SELECT RAISE(ABORT, 'goal extension deletion requires control-plane authorization');
                END;
             CREATE TRIGGER IF NOT EXISTS trg_tasks_goal_session_replace_requires_extension_disposal
                BEFORE INSERT ON tasks FOR EACH ROW
                WHEN NEW.kind = 'goal'
                 AND NEW.session_id IS NOT NULL
                 AND EXISTS (
                    SELECT 1
                      FROM tasks
                      JOIN goal_tasks ON goal_tasks.task_id = tasks.id
                     WHERE tasks.kind = 'goal'
                       AND tasks.session_id = NEW.session_id
                 )
                BEGIN
                    SELECT RAISE(ABORT, 'session goal replacement requires extension disposal');
                END;
             PRAGMA user_version = 10;",
        )
        .context("apply control-plane schema v10")?;
    }
    if version < 11 {
        conn.execute_batch(
            "CREATE TRIGGER IF NOT EXISTS trg_tasks_id_immutable
                BEFORE UPDATE OF id ON tasks FOR EACH ROW
                WHEN NEW.id IS NOT OLD.id
                BEGIN
                    SELECT RAISE(ABORT, 'task id is immutable');
                END;
             PRAGMA user_version = 11;",
        )
        .context("apply control-plane schema v11")?;
    }
    if version < 12 {
        conn.execute_batch(
            "DROP TRIGGER IF EXISTS trg_tasks_id_immutable;
             CREATE TRIGGER trg_tasks_id_immutable
                BEFORE UPDATE OF id ON tasks FOR EACH ROW
                WHEN NEW.id IS NOT OLD.id
                BEGIN
                    SELECT RAISE(ABORT, 'task id is immutable');
                END;
             PRAGMA user_version = 12;",
        )
        .context("apply control-plane schema v12")?;
    }
    if version < 13 {
        conn.execute_batch(
            "DROP TRIGGER IF EXISTS trg_goal_tasks_task_id_immutable;
             CREATE TRIGGER trg_goal_tasks_task_id_immutable
                BEFORE UPDATE OF task_id ON goal_tasks FOR EACH ROW
                WHEN NEW.task_id IS NOT OLD.task_id
                BEGIN
                    SELECT RAISE(ABORT, 'goal task_id is immutable');
                END;
             PRAGMA user_version = 13;",
        )
        .context("apply control-plane schema v13")?;
    }
    if version < 14 {
        conn.execute_batch(
            "CREATE TRIGGER IF NOT EXISTS trg_tasks_id_required
                BEFORE INSERT ON tasks FOR EACH ROW
                WHEN NEW.id IS NULL
                BEGIN
                    SELECT RAISE(ABORT, 'task id must not be NULL');
                END;
             CREATE TRIGGER IF NOT EXISTS trg_goal_tasks_task_id_required
                BEFORE INSERT ON goal_tasks FOR EACH ROW
                WHEN NEW.task_id IS NULL
                BEGIN
                    SELECT RAISE(ABORT, 'goal task_id must not be NULL');
                END;
             PRAGMA user_version = 14;",
        )
        .context("apply control-plane schema v14")?;
    }
    if version < 15 {
        conn.execute_batch(
            "CREATE TRIGGER IF NOT EXISTS trg_goal_tasks_prevent_conflict_replace
                BEFORE INSERT ON goal_tasks FOR EACH ROW
                WHEN EXISTS (SELECT 1 FROM goal_tasks WHERE task_id = NEW.task_id)
                  OR (NEW.pending_call_id IS NOT NULL AND EXISTS (
                      SELECT 1 FROM goal_tasks WHERE pending_call_id = NEW.pending_call_id
                  ))
                BEGIN
                    SELECT RAISE(ABORT, 'goal extension replacement requires control-plane disposal');
                END;
             PRAGMA user_version = 15;",
        )
        .context("apply control-plane schema v15")?;
    }
    Ok(())
}

fn refresh_goal_identity_triggers(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "DROP TRIGGER IF EXISTS trg_tasks_goal_session_insert;
         DROP TRIGGER IF EXISTS trg_tasks_goal_session_rebind;
         DROP TRIGGER IF EXISTS trg_tasks_goal_non_goal_to_goal;
         DROP TRIGGER IF EXISTS trg_tasks_legacy_goal_identity;
         DROP TRIGGER IF EXISTS trg_tasks_session_goal_identity;
         DROP TRIGGER IF EXISTS trg_tasks_goal_active_session_insert;
         DROP TRIGGER IF EXISTS trg_tasks_existing_goal_id_insert;
         DROP TRIGGER IF EXISTS trg_tasks_nonterminal_goal_delete;
         DROP TRIGGER IF EXISTS trg_goal_tasks_task_id_immutable;",
    )
    .context("drop control-plane goal identity triggers")?;

    #[cfg(test)]
    if FAIL_IDENTITY_TRIGGER_REFRESH.with(Cell::get) {
        anyhow::bail!("test failpoint after identity trigger drops");
    }

    conn.execute_batch(
        "CREATE TRIGGER trg_tasks_goal_session_insert
            BEFORE INSERT ON tasks FOR EACH ROW
            WHEN NEW.kind = 'goal'
             AND (NEW.session_id IS NULL OR length(trim(NEW.session_id)) = 0)
            BEGIN
                SELECT RAISE(ABORT, 'goal tasks require a nonblank session_id');
            END;
         CREATE TRIGGER trg_tasks_goal_non_goal_to_goal
            BEFORE UPDATE OF kind, session_id ON tasks FOR EACH ROW
            WHEN OLD.kind != 'goal' AND NEW.kind = 'goal'
            BEGIN
                SELECT RAISE(ABORT, 'task kind cannot be reclassified as goal');
            END;
         CREATE TRIGGER trg_tasks_legacy_goal_identity
            BEFORE UPDATE OF kind, session_id ON tasks FOR EACH ROW
            WHEN OLD.kind = 'goal' AND OLD.session_id IS NULL
             AND (NEW.kind != 'goal' OR NEW.session_id IS NOT NULL)
            BEGIN
                SELECT RAISE(ABORT, 'legacy goal identity cannot be rebound');
            END;
         CREATE TRIGGER trg_tasks_session_goal_identity
            BEFORE UPDATE OF kind, session_id ON tasks FOR EACH ROW
            WHEN OLD.kind = 'goal' AND OLD.session_id IS NOT NULL
             AND (NEW.kind != 'goal'
                  OR NEW.session_id IS NULL
                  OR length(trim(NEW.session_id)) = 0
                  OR NEW.session_id != OLD.session_id)
            BEGIN
                SELECT RAISE(ABORT, 'session goal identity cannot be changed');
            END;
         CREATE TRIGGER trg_tasks_goal_active_session_insert
            BEFORE INSERT ON tasks FOR EACH ROW
            WHEN NEW.kind = 'goal' AND NEW.session_id IS NOT NULL
             AND EXISTS (
                SELECT 1 FROM tasks
                 WHERE kind = 'goal' AND session_id = NEW.session_id
                   AND status NOT IN ('completed','failed','cancelled','lost','timed_out')
             )
            BEGIN
                SELECT RAISE(ABORT, 'session already has a nonterminal goal');
            END;
         CREATE TRIGGER trg_tasks_existing_goal_id_insert
            BEFORE INSERT ON tasks FOR EACH ROW
            WHEN EXISTS (SELECT 1 FROM tasks WHERE id = NEW.id AND kind = 'goal')
            BEGIN
                SELECT RAISE(ABORT, 'existing goal task id cannot be replaced');
            END;
         CREATE TRIGGER trg_tasks_nonterminal_goal_delete
            BEFORE DELETE ON tasks FOR EACH ROW
            WHEN OLD.kind = 'goal'
             AND OLD.status NOT IN ('completed','failed','cancelled','lost','timed_out')
            BEGIN SELECT RAISE(ABORT, 'nonterminal goal cannot be deleted'); END;
         CREATE TRIGGER trg_goal_tasks_task_id_immutable
            BEFORE UPDATE OF task_id ON goal_tasks FOR EACH ROW
            WHEN NEW.task_id IS NOT OLD.task_id
            BEGIN
                SELECT RAISE(ABORT, 'goal task_id is immutable');
            END;",
    )
    .context("refresh control-plane goal identity triggers")?;
    Ok(())
}

/// Apply boot-only goal recovery without reconstructing an executor or any
/// continuation context. Legacy null-session goals are inspection-only.
pub(crate) fn recover_on_boot(store: &SqliteTaskStore, boot_id: &str) -> Result<usize> {
    let mut conn = store.conn.lock();
    let tx = conn.transaction().context("start goal boot recovery")?;
    let legacy = tx.execute(
        "UPDATE tasks SET status = 'failed', error = COALESCE(error, 'legacy_goal_missing_session'),
             finished_at = COALESCE(finished_at, ?1), execution_epoch = execution_epoch + 1
         WHERE kind = 'goal' AND session_id IS NULL
           AND status IN ('running','paused')",
        params![chrono::Utc::now().to_rfc3339()],
    ).context("fail legacy sessionless goals")?;
    let unsafe_nonterminal = tx.execute(
        "UPDATE tasks SET status = 'failed', error = COALESCE(error, 'accounting_outcome_unknown'),
             finished_at = COALESCE(finished_at, ?2), execution_epoch = execution_epoch + 1
         WHERE kind = 'goal' AND status IN ('running','paused') AND owner_boot_id != ?1 AND session_id IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM goal_tasks WHERE task_id = tasks.id
               AND accounting_state = 'complete' AND pending_call_id IS NULL AND pending_call_epoch IS NULL)",
        params![boot_id, chrono::Utc::now().to_rfc3339()],
    ).context("fail unsafe prior-boot nonterminal goals")?;
    tx.execute(
        "UPDATE goal_tasks SET accounting_state = 'outcome_unknown'
         WHERE task_id IN (SELECT id FROM tasks WHERE kind = 'goal' AND status = 'failed'
             AND error = 'accounting_outcome_unknown')
           AND pending_call_id IS NOT NULL AND pending_call_epoch IS NOT NULL",
        [],
    )
    .context("record ambiguous accounting outcome")?;
    tx.execute(
        "UPDATE goal_tasks SET pause_reason = 'daemon_restarted',
             pause_description = 'daemon restart', blockers_json = '[]'
         WHERE task_id IN (SELECT id FROM tasks WHERE kind = 'goal' AND status = 'running'
             AND owner_boot_id != ?1 AND session_id IS NOT NULL
             AND EXISTS (SELECT 1 FROM goal_tasks WHERE task_id = tasks.id
                 AND accounting_state = 'complete' AND pending_call_id IS NULL AND pending_call_epoch IS NULL))",
        params![boot_id],
    ).context("record exact daemon restart pause reason")?;
    let paused = tx.execute(
        "UPDATE tasks SET status = 'paused', execution_epoch = execution_epoch + 1
         WHERE kind = 'goal' AND status = 'running' AND owner_boot_id != ?1
           AND session_id IS NOT NULL
           AND EXISTS (SELECT 1 FROM goal_tasks WHERE task_id = tasks.id AND accounting_state = 'complete' AND pending_call_id IS NULL AND pending_call_epoch IS NULL)",
        params![boot_id],
    ).context("pause prior-boot running goals")?;
    tx.commit().context("commit goal boot recovery")?;
    Ok(legacy + unsafe_nonterminal + paused)
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
        accounting_state: match row.get::<_, String>("accounting_state")?.as_str() {
            "complete" => GoalAccountingState::Complete,
            "missing" => GoalAccountingState::Missing,
            "invalid" => GoalAccountingState::Invalid,
            "outcome_unknown" => GoalAccountingState::OutcomeUnknown,
            value => {
                return Err(rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    format!("unknown goal accounting state {value:?}").into(),
                ));
            }
        },
    })
}

fn insert_goal_task_record(conn: &Connection, rec: GoalTaskRecord) -> Result<()> {
    let pause_reason = rec.pause_reason.map(pause_reason_to_db).transpose()?;
    let blockers_json = blockers_to_db(&rec.blockers)?;
    let (effective_token_limit, effective_cost_limit_usd) =
        goal_limits_to_db(rec.effective_token_limit, rec.effective_cost_limit_usd)?;
    conn.execute(
        "INSERT INTO goal_tasks
            (task_id, objective, effective_token_limit, effective_cost_limit_usd,
             pause_reason, pause_description, blockers_json, pending_call_id, pending_call_epoch, accounting_state)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
         ON CONFLICT(task_id) DO NOTHING",
        params![
            rec.task_id,
            rec.objective,
            effective_token_limit,
            effective_cost_limit_usd,
            pause_reason,
            rec.pause_description,
            blockers_json,
            rec.pending_call_id,
            rec.pending_call_epoch,
            match rec.accounting_state { GoalAccountingState::Complete => "complete", GoalAccountingState::Missing => "missing", GoalAccountingState::Invalid => "invalid", GoalAccountingState::OutcomeUnknown => "outcome_unknown" },
        ],
    )
    .context("insert goal task record")?;
    Ok(())
}

fn insert_goal_task_record_strict(conn: &Connection, rec: GoalTaskRecord) -> Result<()> {
    let pause_reason = rec.pause_reason.map(pause_reason_to_db).transpose()?;
    let blockers_json = blockers_to_db(&rec.blockers)?;
    let (effective_token_limit, effective_cost_limit_usd) =
        goal_limits_to_db(rec.effective_token_limit, rec.effective_cost_limit_usd)?;
    let changed = conn
        .execute(
            "INSERT INTO goal_tasks
                (task_id, objective, effective_token_limit, effective_cost_limit_usd,
                 pause_reason, pause_description, blockers_json, pending_call_id, pending_call_epoch, accounting_state)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                rec.task_id,
                rec.objective,
                effective_token_limit,
                effective_cost_limit_usd,
                pause_reason,
                rec.pause_description,
                blockers_json,
                rec.pending_call_id,
                rec.pending_call_epoch,
                match rec.accounting_state {
                    GoalAccountingState::Complete => "complete",
                    GoalAccountingState::Missing => "missing",
                    GoalAccountingState::Invalid => "invalid",
                    GoalAccountingState::OutcomeUnknown => "outcome_unknown",
                },
            ],
        )
        .context("strictly insert goal task record")?;
    if changed != 1 {
        anyhow::bail!("strict goal task insertion changed {changed} rows");
    }
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

/// Compatibility-only transition for pre-V1 goal callers. New session-bound
/// goal paths must use the fenced APIs below.
fn update_legacy_goal_status(
    conn: &Connection,
    task_id: &str,
    status: TaskStatus,
) -> Result<usize> {
    conn.execute(
        "UPDATE tasks SET status = ?1, finished_at = COALESCE(finished_at, ?2)
         WHERE id = ?3 AND kind = 'goal'
           AND status NOT IN ('completed','failed','cancelled','lost','timed_out')",
        params![
            status_to_db(status),
            status
                .is_terminal()
                .then(|| chrono::Utc::now().to_rfc3339()),
            task_id
        ],
    )
    .context("update legacy goal status")
}

fn claim_legacy_goal_owner(
    conn: &Connection,
    task_id: &str,
    owner_pid: u32,
    owner_boot_id: &str,
) -> Result<usize> {
    conn.execute(
        "UPDATE tasks SET owner_pid = ?1, owner_boot_id = ?2, heartbeat_at = NULL
         WHERE id = ?3 AND kind = 'goal' AND session_id IS NULL AND status = 'paused'",
        params![owner_pid as i64, owner_boot_id, task_id],
    )
    .context("claim legacy goal owner")
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

fn ensure_legacy_goal_mutable(conn: &Connection, task_id: &str) -> Result<()> {
    let session_id: Option<Option<String>> = conn
        .query_row(
            "SELECT session_id FROM tasks WHERE id = ?1 AND kind = 'goal'",
            params![task_id],
            |row| row.get(0),
        )
        .optional()
        .context("classify legacy goal mutator")?;
    if matches!(session_id, Some(Some(_))) {
        anyhow::bail!("session-bound goal {task_id} requires a fenced transition");
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
        task: TaskRecord,
        goal: GoalTaskRecord,
        continuation_context: Option<TaskContinuationContext>,
    ) -> Result<()> {
        ensure_goal_task_identity(&task, &goal)?;
        if task.session_id.is_some() {
            anyhow::bail!("legacy create_goal cannot create session-bound goals");
        }
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
            "SELECT task_id, objective, effective_token_limit, effective_cost_limit_usd,
                    pause_reason, pause_description, blockers_json, pending_call_id,
                    pending_call_epoch, accounting_state
             FROM goal_tasks WHERE task_id = ?1",
            params![task_id],
            row_to_goal_task,
        )
        .optional()
        .context("get goal task")
    }

    async fn update_goal_objective(&self, task_id: &str, objective: &str) -> Result<()> {
        let conn = self.conn.lock();
        ensure_legacy_goal_mutable(&conn, task_id)?;
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
        ensure_legacy_goal_mutable(&conn, task_id)?;
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
        ensure_legacy_goal_mutable(&conn, task_id)?;
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
        ensure_legacy_goal_mutable(&tx, task_id)?;
        ensure_goal_task_row(&tx, task_id)?;
        let updated = update_legacy_goal_status(&tx, task_id, TaskStatus::Paused)?;
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
        ensure_legacy_goal_mutable(&tx, task_id)?;
        ensure_goal_task_row(&tx, task_id)?;
        if let Some(context) = continuation_context {
            upsert_continuation_context(&tx, task_id, &context)?;
        }
        let claimed = claim_legacy_goal_owner(&tx, task_id, owner_pid, owner_boot_id)?;
        if claimed == 0 {
            anyhow::bail!("goal task {task_id} is terminal or missing");
        }
        let updated = update_legacy_goal_status(&tx, task_id, TaskStatus::Running)?;
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
        ensure_legacy_goal_mutable(&conn, task_id)?;
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

    async fn create_or_replace_session_goal(
        &self,
        mut task: TaskRecord,
        goal: GoalTaskRecord,
    ) -> Result<GoalTransitionResult> {
        ensure_goal_task_identity(&task, &goal)?;
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
        let id_exists: bool = tx
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM tasks WHERE id = ?1)",
                params![&task.id],
                |r| r.get(0),
            )
            .context("check proposed session goal id")?;
        if id_exists {
            return Ok(GoalTransitionResult::Stale);
        }
        let existing: Option<(String, String)> = tx
            .query_row(
                "SELECT id, status FROM tasks WHERE kind = 'goal' AND session_id = ?1",
                params![&session_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .context("read current session goal")?;
        if let Some((id, status)) = existing {
            if !matches!(
                status.as_str(),
                "completed" | "failed" | "cancelled" | "lost" | "timed_out"
            ) {
                return Ok(GoalTransitionResult::Stale);
            }
            let _authorization = self.authorize_goal_extension_delete();
            let extension_deleted = tx
                .execute("DELETE FROM goal_tasks WHERE task_id = ?1", params![&id])
                .context("dispose terminal session goal extension")?;
            if extension_deleted != 1 {
                anyhow::bail!("terminal session goal {id} has no goal extension row");
            }
            let task_deleted = tx
                .execute("DELETE FROM tasks WHERE id = ?1", params![id])
                .context("dispose terminal session goal")?;
            if task_deleted != 1 {
                anyhow::bail!("terminal session goal disposal changed {task_deleted} task rows");
            }
        }
        insert_task_record_strict(&tx, task)?;
        insert_goal_task_record_strict(&tx, goal)?;
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
        let tx = conn.transaction().context("start goal pause")?;
        let changed = tx.execute("UPDATE tasks SET status = 'paused', execution_epoch = execution_epoch + 1 WHERE id = ?1 AND kind = 'goal' AND session_id = ?2 AND status = 'running' AND execution_epoch = ?3", params![task_id, session_id, expected_epoch]).context("pause session goal")?;
        if changed == 0 {
            return goal_transition_absence(&tx, task_id);
        }
        let extension_changed = update_goal_pause_record(&tx, task_id, Some(pause))?;
        if extension_changed != 1 {
            anyhow::bail!("goal task {task_id} has no goal extension row");
        }
        tx.commit().context("commit goal pause")?;
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
        let tx = conn.transaction().context("start goal resume")?;
        let changed = tx.execute("UPDATE tasks SET status = 'running', owner_pid = ?4, owner_boot_id = ?5, heartbeat_at = NULL, execution_epoch = execution_epoch + 1 WHERE id = ?1 AND kind = 'goal' AND session_id = ?2 AND status = 'paused' AND execution_epoch = ?3 AND EXISTS (SELECT 1 FROM goal_tasks WHERE task_id = tasks.id AND accounting_state = 'complete' AND pending_call_id IS NULL AND pending_call_epoch IS NULL)", params![task_id, session_id, expected_epoch, owner_pid as i64, owner_boot_id]).context("resume session goal")?;
        if changed == 0 {
            return goal_transition_absence(&tx, task_id);
        }
        let extension_changed = update_goal_pause_record(&tx, task_id, None)?;
        if extension_changed != 1 {
            anyhow::bail!("goal task {task_id} has no goal extension row");
        }
        tx.commit().context("commit goal resume")?;
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
            anyhow::bail!("goal finish requires a terminal task status");
        }
        let conn = self.conn.lock();
        let changed = conn.execute("UPDATE tasks SET status = ?4, error = COALESCE(?5, error), finished_at = COALESCE(finished_at, ?6), execution_epoch = execution_epoch + 1 WHERE id = ?1 AND kind = 'goal' AND session_id = ?2 AND execution_epoch = ?3 AND status IN ('running','paused') AND EXISTS (SELECT 1 FROM goal_tasks WHERE task_id = tasks.id) AND (?4 != 'completed' OR EXISTS (SELECT 1 FROM goal_tasks WHERE task_id = tasks.id AND accounting_state = 'complete' AND pending_call_id IS NULL AND pending_call_epoch IS NULL))", params![task_id, session_id, expected_epoch, super::status_to_db(status), error, chrono::Utc::now().to_rfc3339()]).context("finish session goal")?;
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
        let conn = self.conn.lock();
        let changed = conn.execute("UPDATE goal_tasks SET pending_call_id = ?4, pending_call_epoch = ?3 WHERE task_id = ?1 AND accounting_state = 'complete' AND pending_call_id IS NULL AND pending_call_epoch IS NULL AND EXISTS (SELECT 1 FROM tasks WHERE id = ?1 AND kind = 'goal' AND session_id = ?2 AND status = 'running' AND execution_epoch = ?3)", params![task_id, session_id, expected_epoch, call_id]).context("admit pending goal call")?;
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
        let conn = self.conn.lock();
        let accounting_state = match accounting_state {
            GoalAccountingState::Complete => "complete",
            GoalAccountingState::Missing => "missing",
            GoalAccountingState::Invalid => "invalid",
            GoalAccountingState::OutcomeUnknown => "outcome_unknown",
        };
        let changed = conn.execute("UPDATE goal_tasks SET pending_call_id = NULL, pending_call_epoch = NULL, accounting_state = ?5 WHERE task_id = ?1 AND pending_call_id = ?4 AND pending_call_epoch = ?3 AND EXISTS (SELECT 1 FROM tasks WHERE id = ?1 AND kind = 'goal' AND session_id = ?2 AND status IN ('running','paused'))", params![task_id, session_id, expected_epoch, call_id, accounting_state]).context("settle pending goal call")?;
        if changed == 1 {
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
        let changed = conn
            .execute(
                "UPDATE goal_tasks SET effective_token_limit = ?4, effective_cost_limit_usd = ?5
             WHERE task_id = ?1 AND EXISTS (
               SELECT 1 FROM tasks WHERE id = ?1 AND kind = 'goal' AND session_id = ?2
                 AND execution_epoch = ?3 AND status IN ('running','paused'))",
                params![task_id, session_id, expected_epoch, tokens, cost],
            )
            .context("update session goal limits")?;
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
        let mut conn = self.conn.lock();
        let tx = conn
            .transaction()
            .context("start terminal session goal deletion")?;
        let _authorization = self.authorize_goal_extension_delete();
        let extension_deleted = tx.execute(
            "DELETE FROM goal_tasks WHERE task_id = ?1 AND EXISTS (SELECT 1 FROM tasks WHERE id = ?1 AND kind = 'goal' AND session_id = ?2 AND execution_epoch = ?3 AND status IN ('completed','failed','cancelled','lost','timed_out'))",
            params![task_id, session_id, expected_epoch],
        ).context("delete terminal session goal extension")?;
        if extension_deleted == 0 {
            return goal_transition_absence(&tx, task_id);
        }
        let changed = tx.execute(
            "DELETE FROM tasks WHERE id = ?1 AND kind = 'goal' AND session_id = ?2 AND execution_epoch = ?3
             AND status IN ('completed','failed','cancelled','lost','timed_out')",
            params![task_id, session_id, expected_epoch],
        ).context("delete terminal session goal")?;
        if changed == 1 {
            tx.commit()
                .context("commit terminal session goal deletion")?;
            Ok(GoalTransitionResult::Applied)
        } else {
            goal_transition_absence(&tx, task_id)
        }
    }
}

fn goal_transition_absence(conn: &Connection, task_id: &str) -> Result<GoalTransitionResult> {
    let exists: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM tasks WHERE id = ?1 AND kind = 'goal')",
            params![task_id],
            |r| r.get(0),
        )
        .context("classify goal transition miss")?;
    Ok(if exists {
        GoalTransitionResult::Stale
    } else {
        GoalTransitionResult::Missing
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal_task::{GoalBlockerKind, GoalTaskRegistry};
    use crate::task_registry::{TaskKind, TaskRegistry, TaskStatus};
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
            execution_epoch: 1,
            started_at: "2026-06-18T00:00:00Z".into(),
            finished_at: None,
        }
    }

    fn goal_record(task_id: &str, objective: &str) -> GoalTaskRecord {
        GoalTaskRecord {
            task_id: task_id.into(),
            objective: objective.into(),
            effective_token_limit: None,
            effective_cost_limit_usd: None,
            pause_reason: None,
            pause_description: None,
            blockers: Vec::new(),
            pending_call_id: None,
            pending_call_epoch: None,
            accounting_state: GoalAccountingState::Complete,
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
            conversation_scope: crate::goal_task::TaskContinuationConversationScope::Sender,
        }
    }

    struct LegacyGoalRow<'a> {
        id: &'a str,
        status: &'a str,
        route: Option<&'a str>,
        principal: Option<&'a str>,
        started_at: &'a str,
    }

    fn legacy_goal_store(id: &str, status: &str) -> (tempfile::TempDir, SqliteTaskStore) {
        legacy_goal_store_rows(&[LegacyGoalRow {
            id,
            status,
            route: None,
            principal: None,
            started_at: "now",
        }])
    }

    fn legacy_goal_store_with_token_limit(
        id: &str,
        status: &str,
        effective_token_limit: i64,
    ) -> (tempfile::TempDir, SqliteTaskStore) {
        let dir = tempfile::tempdir().unwrap();
        let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
        db.execute_batch("CREATE TABLE tasks (id TEXT PRIMARY KEY, kind TEXT NOT NULL, agent TEXT NOT NULL, status TEXT NOT NULL, owner_pid INTEGER NOT NULL DEFAULT 0, owner_boot_id TEXT NOT NULL DEFAULT '', heartbeat_at TEXT, depth INTEGER NOT NULL DEFAULT 0, parent_id TEXT, originator_route TEXT, delivered INTEGER NOT NULL DEFAULT 0, idem_key TEXT, principal_id TEXT, started_at TEXT NOT NULL, finished_at TEXT, output TEXT, error TEXT); CREATE TABLE goal_tasks (task_id TEXT PRIMARY KEY, objective TEXT NOT NULL, effective_token_limit INTEGER, effective_cost_limit_usd REAL, pause_reason TEXT, pause_description TEXT, blockers_json TEXT NOT NULL DEFAULT '[]'); PRAGMA user_version=7;").unwrap();
        db.execute(
            "INSERT INTO tasks (id, kind, agent, status, started_at)
             VALUES (?1, 'goal', 'main', ?2, 'now')",
            params![id, status],
        )
        .unwrap();
        db.execute(
            "INSERT INTO goal_tasks (task_id, objective, effective_token_limit)
             VALUES (?1, 'legacy', ?2)",
            params![id, effective_token_limit],
        )
        .unwrap();
        drop(db);
        let store = SqliteTaskStore::new(dir.path()).unwrap();
        (dir, store)
    }

    fn legacy_goal_store_rows(rows: &[LegacyGoalRow<'_>]) -> (tempfile::TempDir, SqliteTaskStore) {
        let dir = tempfile::tempdir().unwrap();
        let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
        db.execute_batch("CREATE TABLE tasks (id TEXT PRIMARY KEY, kind TEXT NOT NULL, agent TEXT NOT NULL, status TEXT NOT NULL, owner_pid INTEGER NOT NULL DEFAULT 0, owner_boot_id TEXT NOT NULL DEFAULT '', heartbeat_at TEXT, depth INTEGER NOT NULL DEFAULT 0, parent_id TEXT, originator_route TEXT, delivered INTEGER NOT NULL DEFAULT 0, idem_key TEXT, principal_id TEXT, started_at TEXT NOT NULL, finished_at TEXT, output TEXT, error TEXT); CREATE TABLE goal_tasks (task_id TEXT PRIMARY KEY, objective TEXT NOT NULL, effective_token_limit INTEGER, effective_cost_limit_usd REAL, pause_reason TEXT, pause_description TEXT, blockers_json TEXT NOT NULL DEFAULT '[]'); PRAGMA user_version=7;").unwrap();
        for row in rows {
            db.execute("INSERT INTO tasks (id,kind,agent,status,originator_route,principal_id,started_at) VALUES (?1,'goal','main',?2,?3,?4,?5)", params![row.id, row.status, row.route, row.principal, row.started_at]).unwrap();
            db.execute(
                "INSERT INTO goal_tasks (task_id,objective) VALUES (?1,'legacy')",
                params![row.id],
            )
            .unwrap();
        }
        drop(db);
        let store = SqliteTaskStore::new(dir.path()).unwrap();
        (dir, store)
    }

    fn v8_goal_store_with_orphan_extension() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
        db.execute_batch(
            "PRAGMA foreign_keys = OFF;
             CREATE TABLE tasks (
                 id TEXT PRIMARY KEY,
                 kind TEXT NOT NULL,
                 agent TEXT NOT NULL,
                 status TEXT NOT NULL,
                 owner_pid INTEGER NOT NULL DEFAULT 0,
                 owner_boot_id TEXT NOT NULL DEFAULT '',
                 heartbeat_at TEXT,
                 depth INTEGER NOT NULL DEFAULT 0,
                 parent_id TEXT,
                 originator_route TEXT,
                 delivered INTEGER NOT NULL DEFAULT 0,
                 idem_key TEXT,
                 principal_id TEXT,
                 session_id TEXT,
                 execution_epoch INTEGER NOT NULL DEFAULT 0,
                 started_at TEXT NOT NULL,
                 finished_at TEXT,
                 output TEXT,
                 error TEXT
             );
             CREATE TABLE goal_tasks (
                 task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,
                 objective TEXT NOT NULL,
                 effective_token_limit INTEGER,
                 effective_cost_limit_usd REAL,
                 pause_reason TEXT,
                 pause_description TEXT,
                 blockers_json TEXT NOT NULL DEFAULT '[]',
                 pending_call_id TEXT,
                 pending_call_epoch INTEGER,
                 accounting_state TEXT NOT NULL DEFAULT 'complete'
             );
             CREATE TABLE task_continuation_contexts (
                 task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,
                 context_json TEXT NOT NULL
             );
             CREATE TRIGGER trg_task_continuation_contexts_require_goal_insert
                BEFORE INSERT ON task_continuation_contexts FOR EACH ROW
                WHEN NOT EXISTS (
                    SELECT 1 FROM tasks
                    JOIN goal_tasks ON goal_tasks.task_id = tasks.id
                    WHERE tasks.id = NEW.task_id AND tasks.kind = 'goal'
                )
                BEGIN SELECT RAISE(ABORT, 'continuation context requires goal'); END;
             CREATE TRIGGER trg_task_continuation_contexts_require_goal_update
                BEFORE UPDATE OF task_id ON task_continuation_contexts FOR EACH ROW
                WHEN NOT EXISTS (
                    SELECT 1 FROM tasks
                    JOIN goal_tasks ON goal_tasks.task_id = tasks.id
                    WHERE tasks.id = NEW.task_id AND tasks.kind = 'goal'
                )
                BEGIN SELECT RAISE(ABORT, 'continuation context requires goal'); END;
             INSERT INTO goal_tasks (task_id, objective) VALUES ('orphan-goal', 'orphan');
             PRAGMA user_version = 8;",
        )
        .unwrap();
        dir
    }

    fn schema_objects(conn: &Connection) -> Vec<(String, String, String)> {
        let mut statement = conn
            .prepare(
                "SELECT type, name, sql FROM sqlite_master
                 WHERE name NOT LIKE 'sqlite_%'
                 ORDER BY type, name",
            )
            .unwrap();
        statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    }

    #[test]
    fn v9_migration_rolls_back_every_schema_change_when_copy_rejects_orphan() {
        let dir = v8_goal_store_with_orphan_extension();

        let error = match SqliteTaskStore::new(dir.path()) {
            Ok(_) => panic!("V9 copy must reject an orphan goal extension"),
            Err(error) => error,
        };
        assert!(format!("{error:#}").contains("FOREIGN KEY constraint failed"));

        let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
        let version: i64 = db
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 8, "a failed V9 migration must remain recoverable");
        assert_eq!(
            db.query_row(
                "SELECT COUNT(*) FROM goal_tasks WHERE task_id = 'orphan-goal'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
            1,
            "the original V8 table must survive the failed rebuild"
        );
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'goal_tasks_v9'", [], |row| row.get::<_, i64>(0)).unwrap(),
            0,
            "the transient V9 table must not leak after rollback"
        );
        assert_eq!(
            db.query_row("SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' AND name IN ('trg_task_continuation_contexts_require_goal_insert', 'trg_task_continuation_contexts_require_goal_update')", [], |row| row.get::<_, i64>(0)).unwrap(),
            2,
            "continuation-context trigger drops must roll back too"
        );
        db.execute_batch(
            "PRAGMA foreign_keys = OFF; DELETE FROM goal_tasks WHERE task_id = 'orphan-goal';",
        )
        .unwrap();
        drop(db);

        SqliteTaskStore::new(dir.path())
            .expect("removing the orphan must allow a V8 database to migrate on reopen");
        let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
        let version: i64 = db
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 15);
    }

    #[test]
    fn v9_migration_rolls_back_when_identity_trigger_refresh_fails() {
        let dir = v8_goal_store_with_orphan_extension();
        let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
        db.execute_batch(
            "PRAGMA foreign_keys = OFF;
             DELETE FROM goal_tasks WHERE task_id = 'orphan-goal';
             CREATE INDEX idx_tasks_status ON tasks(status);
             CREATE INDEX idx_tasks_agent ON tasks(agent);
             CREATE INDEX idx_tasks_agent_kind_started ON tasks(agent, kind, started_at DESC);
             INSERT INTO tasks (id, kind, agent, status, session_id, started_at)
                VALUES ('goal-1', 'goal', 'main', 'running', 'session-1', 'now');
             INSERT INTO goal_tasks (task_id, objective) VALUES ('goal-1', 'goal');",
        )
        .unwrap();
        let before = schema_objects(&db);
        drop(db);

        FAIL_IDENTITY_TRIGGER_REFRESH.with(|enabled| enabled.set(true));
        let error = match SqliteTaskStore::new(dir.path()) {
            Ok(_) => panic!("refresh failpoint must abort V9"),
            Err(error) => error,
        };
        FAIL_IDENTITY_TRIGGER_REFRESH.with(|enabled| enabled.set(false));
        assert!(format!("{error:#}").contains("test failpoint after identity trigger drops"));

        let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
        let version: i64 = db
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 8, "the failed V9 migration must remain at V8");
        assert_eq!(
            schema_objects(&db),
            before,
            "the complete V8 schema must roll back"
        );
        assert_eq!(
            db.query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' AND name = 'trg_task_continuation_contexts_require_goal_insert'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
            1,
            "the V8 goal guard must survive the failed refresh"
        );
        drop(db);

        SqliteTaskStore::new(dir.path()).expect("reopening must repair the V8 database");
        let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
        let version: i64 = db
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 15);
        db.execute(
            "INSERT INTO tasks (id, kind, agent, status, session_id, started_at)
             VALUES ('goal-2', 'goal', 'main', 'running', 'session-2', 'now')",
            [],
        )
        .unwrap();
        let error = db
            .execute(
                "UPDATE goal_tasks SET task_id = 'goal-2' WHERE task_id = 'goal-1'",
                [],
            )
            .expect_err("the repaired V9 schema must make goal task IDs immutable");
        assert!(format!("{error}").contains("goal task_id is immutable"));
    }

    #[test]
    fn v14_migration_adds_non_null_task_and_goal_extension_key_guards() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("control_plane.db");
        let db = Connection::open(&path).unwrap();
        db.execute_batch(
            "CREATE TABLE tasks (
                 id TEXT PRIMARY KEY,
                 kind TEXT NOT NULL,
                 agent TEXT NOT NULL,
                 status TEXT NOT NULL,
                 owner_pid INTEGER NOT NULL DEFAULT 0,
                 owner_boot_id TEXT NOT NULL DEFAULT '',
                 heartbeat_at TEXT,
                 depth INTEGER NOT NULL DEFAULT 0,
                 parent_id TEXT,
                 originator_route TEXT,
                 delivered INTEGER NOT NULL DEFAULT 0,
                 idem_key TEXT,
                 principal_id TEXT,
                 session_id TEXT,
                 execution_epoch INTEGER NOT NULL DEFAULT 0,
                 started_at TEXT NOT NULL,
                 finished_at TEXT,
                 output TEXT,
                 error TEXT
             );
             CREATE TABLE goal_tasks (
                 task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE RESTRICT,
                 objective TEXT NOT NULL,
                 effective_token_limit INTEGER,
                 effective_cost_limit_usd REAL,
                 pause_reason TEXT,
                 pause_description TEXT,
                 blockers_json TEXT NOT NULL DEFAULT '[]',
                 pending_call_id TEXT,
                 pending_call_epoch INTEGER,
                 accounting_state TEXT NOT NULL DEFAULT 'complete'
             );
             CREATE TRIGGER trg_tasks_id_immutable
                BEFORE UPDATE OF id ON tasks FOR EACH ROW
                WHEN NEW.id IS NOT OLD.id
                BEGIN
                    SELECT RAISE(ABORT, 'task id is immutable');
                END;
             CREATE TRIGGER trg_goal_tasks_task_id_immutable
                BEFORE UPDATE OF task_id ON goal_tasks FOR EACH ROW
                WHEN NEW.task_id != OLD.task_id
                BEGIN
                    SELECT RAISE(ABORT, 'goal task_id is immutable');
                END;
             INSERT INTO tasks (id, kind, agent, status, session_id, started_at)
                VALUES ('goal-parent', 'goal', 'main', 'running', 'session-1', 'now');
             INSERT INTO goal_tasks (task_id, objective) VALUES ('goal-parent', 'goal');
             PRAGMA user_version = 12;",
        )
        .unwrap();
        let parent_guard_before: String = db
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = 'trg_tasks_id_immutable'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        db.execute_batch("PRAGMA foreign_keys = OFF; PRAGMA recursive_triggers = OFF;")
            .unwrap();
        assert_eq!(
            db.execute(
                "UPDATE goal_tasks SET task_id = NULL WHERE task_id = 'goal-parent'",
                [],
            )
            .unwrap(),
            1,
            "the V12 trigger must reproduce the NULL bypass",
        );
        db.execute(
            "UPDATE goal_tasks SET task_id = 'goal-parent' WHERE objective = 'goal'",
            [],
        )
        .unwrap();
        drop(db);

        let store = SqliteTaskStore::new(dir.path()).expect("V12 database must migrate to V14");
        drop(store);

        let db = Connection::open(&path).unwrap();
        let version: i64 = db
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 15);
        let parent_guard_after: String = db
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = 'trg_tasks_id_immutable'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(parent_guard_after, parent_guard_before);

        db.execute_batch("PRAGMA foreign_keys = OFF; PRAGMA recursive_triggers = OFF;")
            .unwrap();
        let error = db
            .execute(
                "UPDATE goal_tasks SET task_id = NULL WHERE task_id = 'goal-parent'",
                [],
            )
            .expect_err("V13 must reject NULL goal extension task IDs");
        assert!(format!("{error}").contains("goal task_id is immutable"));
        let task_id: String = db
            .query_row(
                "SELECT task_id FROM goal_tasks WHERE objective = 'goal'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            task_id, "goal-parent",
            "the extension must remain bound to its parent"
        );
        let task_error = db
            .execute(
                "INSERT INTO tasks (id, kind, agent, status, session_id, started_at)
                 VALUES (NULL, 'goal', 'main', 'running', 'new-session', 'now')",
                [],
            )
            .expect_err("V14 must reject NULL task IDs on raw goal insertion");
        assert!(format!("{task_error}").contains("task id must not be NULL"));
        let extension_error = db
            .execute(
                "INSERT INTO goal_tasks (task_id, objective) VALUES (NULL, 'orphan')",
                [],
            )
            .expect_err("V14 must reject NULL goal extension task IDs on insertion");
        assert!(format!("{extension_error}").contains("goal task_id must not be NULL"));
    }

    #[test]
    fn future_schema_version_does_not_refresh_or_replace_existing_triggers() {
        let dir = tempfile::tempdir().unwrap();
        let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
        db.execute_batch(
            "PRAGMA journal_mode = DELETE;
             CREATE TABLE tasks (
                 id TEXT PRIMARY KEY, kind TEXT NOT NULL, agent TEXT NOT NULL, status TEXT NOT NULL,
                 owner_pid INTEGER NOT NULL DEFAULT 0, owner_boot_id TEXT NOT NULL DEFAULT '', heartbeat_at TEXT,
                 depth INTEGER NOT NULL DEFAULT 0, parent_id TEXT, originator_route TEXT, delivered INTEGER NOT NULL DEFAULT 0,
                 idem_key TEXT, principal_id TEXT, session_id TEXT, execution_epoch INTEGER NOT NULL DEFAULT 0,
                 started_at TEXT NOT NULL, finished_at TEXT, output TEXT, error TEXT
             );
             CREATE TABLE goal_tasks (
                 task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE RESTRICT,
                 objective TEXT NOT NULL, effective_token_limit INTEGER, effective_cost_limit_usd REAL,
                 pause_reason TEXT, pause_description TEXT, blockers_json TEXT NOT NULL DEFAULT '[]',
                 pending_call_id TEXT, pending_call_epoch INTEGER,
                 accounting_state TEXT NOT NULL DEFAULT 'complete'
             );
             CREATE TRIGGER trg_tasks_goal_session_insert
                BEFORE INSERT ON tasks FOR EACH ROW
                BEGIN SELECT RAISE(ABORT, 'future schema sentinel'); END;
             PRAGMA user_version = 16;",
        )
        .unwrap();
        let journal_mode: String = db
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        assert_eq!(journal_mode, "delete");
        let before = schema_objects(&db);
        drop(db);

        let store = SqliteTaskStore::new(dir.path())
            .expect("a newer database must open warning-only without schema mutation");
        drop(store);
        let store = SqliteTaskStore::new(dir.path())
            .expect("reopening a newer database must remain warning-only");
        drop(store);

        let db = Connection::open(dir.path().join("control_plane.db")).unwrap();
        assert_eq!(schema_objects(&db), before);
        let journal_mode: String = db
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        assert_eq!(
            journal_mode, "delete",
            "opening a newer database must not change its persistent journal mode"
        );
        let version: i64 = db
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 16);
    }

    #[tokio::test]
    async fn latest_active_goal_for_agent_resolves_in_sql() {
        let (_dir, s) = legacy_goal_store_rows(&[
            LegacyGoalRow {
                id: "old-goal",
                status: "running",
                route: Some("route-old"),
                principal: None,
                started_at: "2026-06-18T00:00:00Z",
            },
            LegacyGoalRow {
                id: "done-goal",
                status: "running",
                route: None,
                principal: None,
                started_at: "2026-06-20T00:00:00Z",
            },
            LegacyGoalRow {
                id: "latest-goal",
                status: "running",
                route: None,
                principal: None,
                started_at: "2026-06-19T00:00:00Z",
            },
        ]);

        let mut newest_delegate = rec("newest-delegate", "main", 1, "boot-1");
        newest_delegate.started_at = "2026-06-21T00:00:00Z".into();
        s.create(newest_delegate).await.unwrap();

        let got = s
            .latest_active_goal_for_agent("main")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(got.id, "done-goal");
        assert!(
            s.latest_active_goal_for_agent("other")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn latest_active_goal_for_context_filters_route_and_principal() {
        let (_dir, s) = legacy_goal_store_rows(&[
            LegacyGoalRow {
                id: "other-route",
                status: "running",
                route: Some("route-b"),
                principal: Some("principal-a"),
                started_at: "2026-06-20T00:00:00Z",
            },
            LegacyGoalRow {
                id: "other-principal",
                status: "running",
                route: Some("route-a"),
                principal: Some("principal-b"),
                started_at: "2026-06-19T00:00:00Z",
            },
            LegacyGoalRow {
                id: "wanted",
                status: "running",
                route: Some("route-a"),
                principal: Some("principal-a"),
                started_at: "2026-06-18T00:00:00Z",
            },
        ]);

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
        let (_dir, s) = legacy_goal_store_rows(&[
            LegacyGoalRow {
                id: "first-goal",
                status: "running",
                route: Some("route-a"),
                principal: Some("principal-a"),
                started_at: "2026-06-19T00:00:00Z",
            },
            LegacyGoalRow {
                id: "second-goal",
                status: "running",
                route: Some("route-b"),
                principal: Some("principal-b"),
                started_at: "2026-06-19T00:00:00Z",
            },
        ]);

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
        let (_dir, s) = legacy_goal_store_rows(&[
            LegacyGoalRow {
                id: "older-valid-goal",
                status: "running",
                route: Some("route-a"),
                principal: Some("principal-a"),
                started_at: "2026-06-19T00:00:00Z",
            },
            LegacyGoalRow {
                id: "unreadable-newer-goal",
                status: "future_paused",
                route: Some("route-a"),
                principal: Some("principal-a"),
                started_at: "2026-06-20T00:00:00Z",
            },
        ]);

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
        let (_dir, s) = legacy_goal_store_rows(&[
            LegacyGoalRow {
                id: "older-valid-goal",
                status: "running",
                route: Some("route-valid"),
                principal: None,
                started_at: "2026-06-19T00:00:00Z",
            },
            LegacyGoalRow {
                id: "unreadable-newer-goal",
                status: "future_paused",
                route: None,
                principal: None,
                started_at: "2026-06-20T00:00:00Z",
            },
        ]);

        let got = s
            .latest_active_goal_for_agent("main")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(got.id, "older-valid-goal");
    }

    #[tokio::test]
    async fn goal_task_extension_roundtrips_without_duplicating_lifecycle() {
        let (_dir, s) = legacy_goal_store("goal-1", "paused");
        s.update_goal_objective("goal-1", "ship goal mode")
            .await
            .unwrap();
        s.update_goal_limits("goal-1", Some(10_000), Some(1.25))
            .await
            .unwrap();
        s.update_goal_pause(
            "goal-1",
            Some(GoalPauseState {
                reason: GoalPauseReason::NeedsUserInput,
                description: Some("waiting for operator".into()),
                blockers: vec![GoalBlocker {
                    kind: GoalBlockerKind::NeedsUserInput,
                    message: "Need operator answer".into(),
                    payload: Some(serde_json::json!({"question": "continue?"})),
                }],
            }),
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

        s.update_goal_objective("goal-1", "ship amended goal mode")
            .await
            .unwrap();
        let amended = s.get_goal_task("goal-1").await.unwrap().unwrap();
        assert_eq!(amended.objective, "ship amended goal mode");

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
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                    pending_call_id: None,
                    pending_call_epoch: None,
                    accounting_state: GoalAccountingState::Complete,
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
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                    pending_call_id: None,
                    pending_call_epoch: None,
                    accounting_state: GoalAccountingState::Complete,
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
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                    pending_call_id: None,
                    pending_call_epoch: None,
                    accounting_state: GoalAccountingState::Complete,
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
                    effective_token_limit: None,
                    effective_cost_limit_usd: None,
                    pause_reason: None,
                    pause_description: None,
                    blockers: Vec::new(),
                    pending_call_id: None,
                    pending_call_epoch: None,
                    accounting_state: GoalAccountingState::Complete,
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
        let (_dir, s) = legacy_goal_store("goal-limit-validation", "running");

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
        let admitted = s
            .create_or_replace_session_goal(
                task,
                goal_record("goal-sqlite-limit-validation", "validate SQLite limits"),
            )
            .await
            .unwrap();
        assert_eq!(admitted, GoalTransitionResult::Applied);

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
        let (_dir, s) = legacy_goal_store_with_token_limit("goal-corrupt-limit", "running", -1);

        let err = s
            .get_goal_task("goal-corrupt-limit")
            .await
            .expect_err("legacy negative token limits must not inflate to u64");
        assert!(format!("{err:#}").contains("get goal task"));
    }

    #[tokio::test]
    async fn goal_extension_updates_reject_missing_goal_rows() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        {
            let conn = s.conn.lock();
            conn.execute(
                "INSERT INTO tasks (id, kind, agent, status, session_id, execution_epoch, started_at)
                 VALUES (?1, 'goal', 'main', 'running', ?2, 1, 'now')",
                params!["goal-without-extension", "session-goal-without-extension"],
            )
            .unwrap();
        }

        let err = s
            .update_goal_limits("goal-without-extension", Some(100), None)
            .await
            .expect_err("limit updates must reject missing goal extension rows");
        assert!(format!("{err:#}").contains("requires a fenced transition"));

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
        assert!(format!("{err:#}").contains("requires a fenced transition"));
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
        let (_dir, s) = legacy_goal_store("goal-paused", "running");

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
        let (_dir, s) = legacy_goal_store("goal-operator-paused", "running");

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
        {
            let conn = s.conn.lock();
            conn.execute(
                "INSERT INTO tasks (id, kind, agent, status, session_id, execution_epoch, started_at)
                 VALUES (?1, 'goal', 'main', 'running', ?2, 1, 'now')",
                params!["goal-missing-extension", "session-goal-missing-extension"],
            )
            .unwrap();
        }

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

        assert!(format!("{err:#}").contains("requires a fenced transition"));
        let task = s.get("goal-missing-extension").await.unwrap().unwrap();
        assert_eq!(task.status, TaskStatus::Running);
    }

    #[tokio::test]
    async fn resume_goal_task_clears_pause_claims_owner_and_sets_running() {
        let (_dir, s) = legacy_goal_store("goal-resume", "paused");
        s.update_goal_pause(
            "goal-resume",
            Some(GoalPauseState {
                reason: GoalPauseReason::NeedsUserInput,
                description: Some("waiting".into()),
                blockers: vec![GoalBlocker {
                    kind: GoalBlockerKind::NeedsUserInput,
                    message: "Need operator answer".into(),
                    payload: None,
                }],
            }),
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
    async fn terminal_replacement_rolls_back_when_extension_insert_fails() {
        let s = SqliteTaskStore::new_in_memory().unwrap();
        let mut predecessor = rec("predecessor", "main", 1, "boot-1");
        predecessor.kind = TaskKind::Goal;
        assert_eq!(
            s.create_or_replace_session_goal(predecessor, goal_record("predecessor", "x"))
                .await
                .unwrap(),
            GoalTransitionResult::Applied
        );
        assert_eq!(
            s.finish_session_goal(
                "predecessor",
                "session-predecessor",
                1,
                TaskStatus::Completed,
                None
            )
            .await
            .unwrap(),
            GoalTransitionResult::Applied
        );
        {
            let conn = s.conn.lock();
            conn.execute_batch("CREATE TRIGGER test_fail_goal_extension_insert BEFORE INSERT ON goal_tasks WHEN NEW.task_id = 'replacement' BEGIN SELECT RAISE(ABORT, 'test failpoint'); END;").unwrap();
        }
        let mut replacement = rec("replacement", "main", 1, "boot-1");
        replacement.kind = TaskKind::Goal;
        replacement.session_id = Some("session-predecessor".into());
        assert!(
            s.create_or_replace_session_goal(replacement, goal_record("replacement", "x"))
                .await
                .is_err()
        );
        assert_eq!(
            s.current_goal_for_session("session-predecessor")
                .await
                .unwrap()
                .unwrap()
                .id,
            "predecessor"
        );
        assert!(s.get_goal_task("predecessor").await.unwrap().is_some());
    }
}
