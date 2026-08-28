use async_trait::async_trait;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use zeroclaw_api::tool::{Tool, ToolOutput, ToolResult};
use zeroclaw_config::autonomy::AutonomyLevel;
use zeroclaw_config::policy::SecurityPolicy;

/// Git operations tool for structured repository management.
/// Provides safe, parsed git operations with JSON output.
pub struct GitOperationsTool {
    security: Arc<SecurityPolicy>,
    workspace_dir: std::path::PathBuf,
}

impl GitOperationsTool {
    pub fn new(security: Arc<SecurityPolicy>, workspace_dir: std::path::PathBuf) -> Self {
        Self {
            security,
            workspace_dir,
        }
    }

    /// Sanitize git arguments to prevent injection attacks
    fn sanitize_git_args(&self, args: &str) -> anyhow::Result<Vec<String>> {
        let mut result = Vec::new();
        for arg in args.split_whitespace() {
            // Block dangerous git options that could lead to command injection
            let arg_lower = arg.to_lowercase();
            if arg_lower.starts_with("--exec=")
                || arg_lower.starts_with("--upload-pack=")
                || arg_lower.starts_with("--receive-pack=")
                || arg_lower.starts_with("--pager=")
                || arg_lower.starts_with("--editor=")
                || arg_lower == "--no-verify"
                || arg_lower.contains("$(")
                || arg_lower.contains('`')
                || arg.contains('|')
                || arg.contains(';')
                || arg.contains('>')
            {
                anyhow::bail!("Blocked potentially dangerous git argument: {arg}");
            }
            // Block `-c` config injection (exact match or `-c=...` prefix).
            // This must not false-positive on `--cached` or `-cached`.
            if arg_lower == "-c" || arg_lower.starts_with("-c=") {
                anyhow::bail!("Blocked potentially dangerous git argument: {arg}");
            }
            result.push(arg.to_string());
        }
        Ok(result)
    }

    /// Check if an operation requires write access.
    fn requires_write_access(&self, operation: &str, args: &serde_json::Value) -> bool {
        matches!(
            operation,
            "commit" | "add" | "checkout" | "reset" | "revert"
        ) || (operation == "stash"
            && !matches!(
                args.get("action").and_then(|value| value.as_str()),
                Some("list")
            ))
    }

    /// Return whether a repository's physical `.git` directory is within any
    /// authorized root.
    ///
    /// Git normally discovers repositories by walking to parent directories. That
    /// would let an approved child path operate on an unapproved parent repository,
    /// so discovery must stop after it leaves every root that authorized the
    /// requested path.
    /// `.git` files used by linked worktrees are intentionally rejected: their
    /// indirection cannot be validated against this boundary without widening
    /// the authorization contract.
    fn has_repository_within_authorized_roots(
        &self,
        working_dir: &Path,
        authorized_roots: &[PathBuf],
    ) -> bool {
        let mut current_dir = working_dir;
        loop {
            if std::fs::symlink_metadata(current_dir.join(".git"))
                .is_ok_and(|metadata| metadata.file_type().is_dir())
                && (authorized_roots.is_empty()
                    || authorized_roots
                        .iter()
                        .any(|root| current_dir.starts_with(root)))
            {
                return true;
            }
            let Some(parent) = current_dir.parent() else {
                return false;
            };
            if !authorized_roots.is_empty()
                && !authorized_roots.iter().any(|root| parent.starts_with(root))
            {
                return false;
            }
            current_dir = parent;
        }
    }

    /// Resolve a user-provided path authorized for the requested Git operation.
    /// Returns the workspace_dir if no path is provided.
    /// Rejects paths that escape the workspace via traversal.
    fn resolve_working_dir(
        &self,
        path: Option<&str>,
        requires_write_access: bool,
    ) -> anyhow::Result<std::path::PathBuf> {
        let base = match path {
            Some(p) if !p.is_empty() => {
                let candidate = if std::path::Path::new(p).is_absolute() {
                    std::path::PathBuf::from(p)
                } else {
                    self.workspace_dir.join(p)
                };
                let resolved = candidate.canonicalize().map_err(|e| {
                    ::zeroclaw_log::record!(
                        WARN,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                            .with_attrs(::serde_json::json!({
                                "path": p,
                                "error": format!("{}", e),
                            })),
                        "git_operations: cannot resolve path"
                    );
                    anyhow::Error::msg(format!("Cannot resolve path '{}': {}", p, e))
                })?;
                if !self.security.is_resolved_path_readable(&resolved)
                    || (requires_write_access && !self.security.is_resolved_path_allowed(&resolved))
                {
                    anyhow::bail!("Path '{}' is not authorized for this Git operation", p);
                }
                resolved
            }
            _ => self.workspace_dir.clone(),
        };
        Ok(base)
    }

    async fn run_git_command(
        &self,
        args: &[&str],
        working_dir: &std::path::Path,
    ) -> anyhow::Result<String> {
        let output = tokio::process::Command::new("git")
            .args(args)
            .current_dir(working_dir)
            .env("GIT_TERMINAL_PROMPT", "0")
            .stdin(std::process::Stdio::null())
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Git command failed: {stderr}");
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Run a read-classified Git command without repository-configured command
    /// hooks. `status` can invoke `core.fsmonitor`; `diff` also disables its
    /// external-diff and text-conversion paths at the call site below.
    async fn run_git_read_command(
        &self,
        args: &[&str],
        working_dir: &std::path::Path,
    ) -> anyhow::Result<String> {
        let output = tokio::process::Command::new("git")
            .args(["-c", "core.fsmonitor=false", "-c", "core.pager=cat"])
            .args(args)
            .current_dir(working_dir)
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_PAGER", "cat")
            .stdin(std::process::Stdio::null())
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Git command failed: {stderr}");
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    async fn git_status(
        &self,
        _args: serde_json::Value,
        working_dir: &std::path::Path,
    ) -> anyhow::Result<ToolResult> {
        let output = self
            .run_git_read_command(
                &["--no-optional-locks", "status", "--porcelain=2", "--branch"],
                working_dir,
            )
            .await?;

        // Parse git status output into structured format
        let mut result = serde_json::Map::new();
        let mut branch = String::new();
        let mut staged = Vec::new();
        let mut unstaged = Vec::new();
        let mut untracked = Vec::new();

        for line in output.lines() {
            if line.starts_with("# branch.head ") {
                branch = line.trim_start_matches("# branch.head ").to_string();
            } else if let Some(rest) = line.strip_prefix("1 ") {
                // Ordinary changed entry
                let mut parts = rest.splitn(3, ' ');
                if let (Some(staging), Some(path)) = (parts.next(), parts.next())
                    && !staging.is_empty()
                {
                    let status_char = staging.chars().next().unwrap_or(' ');
                    if status_char != '.' && status_char != ' ' {
                        staged.push(json!({"path": path, "status": status_char}));
                    }
                    let status_char = staging.chars().nth(1).unwrap_or(' ');
                    if status_char != '.' && status_char != ' ' {
                        unstaged.push(json!({"path": path, "status": status_char}));
                    }
                }
            } else if let Some(rest) = line.strip_prefix("? ") {
                untracked.push(rest.to_string());
            }
        }

        result.insert("branch".to_string(), json!(branch));
        result.insert("staged".to_string(), json!(staged));
        result.insert("unstaged".to_string(), json!(unstaged));
        result.insert("untracked".to_string(), json!(untracked));
        result.insert(
            "clean".to_string(),
            json!(staged.is_empty() && unstaged.is_empty() && untracked.is_empty()),
        );

        Ok(ToolResult {
            success: true,
            output: serde_json::to_string_pretty(&result)
                .unwrap_or_default()
                .into(),
            error: None,
        })
    }

    async fn git_diff(
        &self,
        args: serde_json::Value,
        working_dir: &std::path::Path,
    ) -> anyhow::Result<ToolResult> {
        let files = args.get("files").and_then(|v| v.as_str()).unwrap_or(".");
        let cached = args
            .get("cached")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Validate files argument against injection patterns
        self.sanitize_git_args(files)?;

        let mut git_args = vec![
            "--no-optional-locks",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--unified=3",
        ];
        if cached {
            git_args.push("--cached");
        }
        git_args.push("--");
        git_args.push(files);

        let output = self.run_git_read_command(&git_args, working_dir).await?;

        // Parse diff into structured hunks
        let mut result = serde_json::Map::new();
        let mut hunks = Vec::new();
        let mut current_file = String::new();
        let mut current_hunk = serde_json::Map::new();
        let mut lines = Vec::new();

        for line in output.lines() {
            if line.starts_with("diff --git ") {
                if !lines.is_empty() {
                    current_hunk.insert("lines".to_string(), json!(lines));
                    if !current_hunk.is_empty() {
                        hunks.push(serde_json::Value::Object(current_hunk.clone()));
                    }
                    lines = Vec::new();
                    current_hunk = serde_json::Map::new();
                }
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    current_file = parts[3].trim_start_matches("b/").to_string();
                    current_hunk.insert("file".to_string(), json!(current_file));
                }
            } else if line.starts_with("@@ ") {
                if !lines.is_empty() {
                    current_hunk.insert("lines".to_string(), json!(lines));
                    if !current_hunk.is_empty() {
                        hunks.push(serde_json::Value::Object(current_hunk.clone()));
                    }
                    lines = Vec::new();
                    current_hunk = serde_json::Map::new();
                    current_hunk.insert("file".to_string(), json!(current_file));
                }
                current_hunk.insert("header".to_string(), json!(line));
            } else if !line.is_empty() {
                lines.push(json!({
                    "text": line,
                    "type": if line.starts_with('+') { "add" }
                           else if line.starts_with('-') { "delete" }
                           else { "context" }
                }));
            }
        }

        if !lines.is_empty() {
            current_hunk.insert("lines".to_string(), json!(lines));
            if !current_hunk.is_empty() {
                hunks.push(serde_json::Value::Object(current_hunk));
            }
        }

        result.insert("hunks".to_string(), json!(hunks));
        result.insert("file_count".to_string(), json!(hunks.len()));

        Ok(ToolResult {
            success: true,
            output: serde_json::to_string_pretty(&result)
                .unwrap_or_default()
                .into(),
            error: None,
        })
    }

    async fn git_log(
        &self,
        args: serde_json::Value,
        working_dir: &std::path::Path,
    ) -> anyhow::Result<ToolResult> {
        let limit_raw = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10);
        let limit = usize::try_from(limit_raw).unwrap_or(usize::MAX).min(1000);
        let limit_str = limit.to_string();

        let output = self
            .run_git_read_command(
                &[
                    "--no-optional-locks",
                    "log",
                    &format!("-{limit_str}"),
                    "--pretty=format:%H|%an|%ae|%ad|%s",
                    "--date=iso",
                ],
                working_dir,
            )
            .await?;

        let mut commits = Vec::new();

        for line in output.lines() {
            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() >= 5 {
                commits.push(json!({
                    "hash": parts[0],
                    "author": parts[1],
                    "email": parts[2],
                    "date": parts[3],
                    "message": parts[4]
                }));
            }
        }

        Ok(ToolResult {
            success: true,
            output: serde_json::to_string_pretty(&json!({ "commits": commits }))
                .unwrap_or_default()
                .into(),
            error: None,
        })
    }

    async fn git_branch(
        &self,
        _args: serde_json::Value,
        working_dir: &std::path::Path,
    ) -> anyhow::Result<ToolResult> {
        let output = self
            .run_git_read_command(
                &[
                    "--no-optional-locks",
                    "branch",
                    "--format=%(refname:short)|%(HEAD)",
                ],
                working_dir,
            )
            .await?;

        let mut branches = Vec::new();
        let mut current = String::new();

        for line in output.lines() {
            if let Some((name, head)) = line.split_once('|') {
                let is_current = head == "*";
                if is_current {
                    current = name.to_string();
                }
                branches.push(json!({
                    "name": name,
                    "current": is_current
                }));
            }
        }

        Ok(ToolResult {
            success: true,
            output: serde_json::to_string_pretty(&json!({
                "current": current,
                "branches": branches
            }))
            .unwrap_or_default()
            .into(),
            error: None,
        })
    }

    fn truncate_commit_message(message: &str) -> String {
        if message.chars().count() > 2000 {
            format!("{}...", message.chars().take(1997).collect::<String>())
        } else {
            message.to_string()
        }
    }

    async fn git_commit(
        &self,
        args: serde_json::Value,
        working_dir: &std::path::Path,
    ) -> anyhow::Result<ToolResult> {
        let message = args
            .get("message")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                        .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                        .with_attrs(::serde_json::json!({"param": "message"})),
                    "git_operations: missing message parameter"
                );
                anyhow::Error::msg("Missing 'message' parameter")
            })?;

        let trimmed_lines: Vec<&str> = message.lines().map(|l| l.trim_end()).collect();
        // Drop leading blank lines.
        let trimmed_lines = trimmed_lines
            .iter()
            .copied()
            .skip_while(|l| l.is_empty())
            .collect::<Vec<_>>();
        // Collapse runs of more than 2 consecutive blank lines to 2.
        let mut sanitized_lines: Vec<&str> = Vec::with_capacity(trimmed_lines.len());
        let mut consecutive_blanks = 0usize;
        for line in &trimmed_lines {
            if line.is_empty() {
                consecutive_blanks += 1;
                if consecutive_blanks <= 2 {
                    sanitized_lines.push(line);
                }
            } else {
                consecutive_blanks = 0;
                sanitized_lines.push(line);
            }
        }
        // Drop trailing blank lines.
        while sanitized_lines.last().is_some_and(|l: &&str| l.is_empty()) {
            sanitized_lines.pop();
        }
        let sanitized = sanitized_lines.join("\n");

        if sanitized.is_empty() {
            anyhow::bail!("Commit message cannot be empty");
        }

        // Limit message length
        let message = Self::truncate_commit_message(&sanitized);

        let output = self
            .run_git_command(&["commit", "-m", &message], working_dir)
            .await;

        match output {
            Ok(_) => Ok(ToolResult {
                success: true,
                output: format!("Committed: {message}").into(),
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: ToolOutput::default(),
                error: Some(format!("Commit failed: {e}")),
            }),
        }
    }

    async fn git_add(
        &self,
        args: serde_json::Value,
        working_dir: &std::path::Path,
    ) -> anyhow::Result<ToolResult> {
        let paths = args.get("paths").and_then(|v| v.as_str()).ok_or_else(|| {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(::serde_json::json!({"param": "paths"})),
                "git_operations: missing paths parameter"
            );
            anyhow::Error::msg("Missing 'paths' parameter")
        })?;

        // Validate paths against injection patterns. Returns each
        // whitespace-separated pathspec as its own argument so the join is
        // not handed to git as a single literal path.
        let sanitized = self.sanitize_git_args(paths)?;
        if sanitized.is_empty() {
            anyhow::bail!("No paths to stage");
        }

        let mut git_args: Vec<&str> = vec!["add", "--"];
        git_args.extend(sanitized.iter().map(String::as_str));

        let output = self.run_git_command(&git_args, working_dir).await;

        match output {
            Ok(_) => Ok(ToolResult {
                success: true,
                output: format!("Staged: {}", sanitized.join(" ")).into(),
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: ToolOutput::default(),
                error: Some(format!("Add failed: {e}")),
            }),
        }
    }

    async fn git_checkout(
        &self,
        args: serde_json::Value,
        working_dir: &std::path::Path,
    ) -> anyhow::Result<ToolResult> {
        let branch = args.get("branch").and_then(|v| v.as_str()).ok_or_else(|| {
            ::zeroclaw_log::record!(
                WARN,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                    .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                    .with_attrs(::serde_json::json!({"param": "branch"})),
                "git_operations: missing branch parameter"
            );
            anyhow::Error::msg("Missing 'branch' parameter")
        })?;

        // Sanitize branch name
        let sanitized = self.sanitize_git_args(branch)?;

        if sanitized.is_empty() || sanitized.len() > 1 {
            anyhow::bail!("Invalid branch specification");
        }

        let branch_name = &sanitized[0];

        // Block dangerous branch names
        if branch_name.contains('@') || branch_name.contains('^') || branch_name.contains('~') {
            anyhow::bail!("Branch name contains invalid characters");
        }

        let output = self
            .run_git_command(&["checkout", branch_name], working_dir)
            .await;

        match output {
            Ok(_) => Ok(ToolResult {
                success: true,
                output: format!("Switched to branch: {branch_name}").into(),
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: ToolOutput::default(),
                error: Some(format!("Checkout failed: {e}")),
            }),
        }
    }

    async fn git_stash(
        &self,
        args: serde_json::Value,
        working_dir: &std::path::Path,
    ) -> anyhow::Result<ToolResult> {
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("push");

        let output = match action {
            "push" | "save" => {
                let message = args
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("auto-stash")
                    .to_string();
                let keep_index = args
                    .get("keep_index")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let include_untracked = args
                    .get("include_untracked")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let paths_raw = args
                    .get("paths")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let mut cmd: Vec<String> =
                    vec!["stash".into(), "push".into(), "-m".into(), message];
                if keep_index {
                    cmd.push("-k".into());
                }
                if include_untracked {
                    cmd.push("-u".into());
                }
                if !paths_raw.is_empty() {
                    cmd.push("--".into());
                    for p in paths_raw.split_whitespace() {
                        cmd.push(p.to_string());
                    }
                }
                let cmd_refs: Vec<&str> = cmd.iter().map(String::as_str).collect();
                self.run_git_command(&cmd_refs, working_dir).await
            }
            "pop" => self.run_git_command(&["stash", "pop"], working_dir).await,
            "list" => {
                self.run_git_read_command(&["--no-optional-locks", "stash", "list"], working_dir)
                    .await
            }
            "drop" => {
                let index_raw = args.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                let index = i32::try_from(index_raw).map_err(|_| {
                    ::zeroclaw_log::record!(
                        WARN,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Reject)
                            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                            .with_attrs(::serde_json::json!({"index": index_raw})),
                        "git_operations: stash index too large"
                    );
                    anyhow::Error::msg(format!("stash index too large: {index_raw}"))
                })?;
                self.run_git_command(
                    &["stash", "drop", &format!("stash@{{{index}}}")],
                    working_dir,
                )
                .await
            }
            _ => anyhow::bail!("Unknown stash action: {action}. Use: push, pop, list, drop"),
        };

        match output {
            Ok(out) => Ok(ToolResult {
                success: true,
                output: out.into(),
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: ToolOutput::default(),
                error: Some(format!("Stash {action} failed: {e}")),
            }),
        }
    }
}

#[async_trait]
impl Tool for GitOperationsTool {
    fn name(&self) -> &str {
        "git_operations"
    }

    fn description(&self) -> &str {
        "Perform structured Git operations (status, diff, log, branch, commit, add, checkout, stash). Provides parsed JSON output and integrates with security policy for autonomy controls."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["status", "diff", "log", "branch", "commit", "add", "checkout", "stash"],
                    "description": "Git operation to perform"
                },
                "message": {
                    "type": "string",
                    "description": "Commit message (for 'commit' operation); stash message (for 'stash push', defaults to 'auto-stash')"
                },
                "paths": {
                    "type": "string",
                    "description": "Space-separated file paths. For 'add', files to stage. For 'stash push', pathspecs to scope the stash to — without this, the entire working tree is stashed."
                },
                "branch": {
                    "type": "string",
                    "description": "Branch name for the 'checkout' operation"
                },
                "files": {
                    "type": "string",
                    "description": "File or path to diff (for 'diff' operation, default: '.')"
                },
                "cached": {
                    "type": "boolean",
                    "description": "Show staged changes (for 'diff' operation)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of log entries (for 'log' operation, default: 10)"
                },
                "action": {
                    "type": "string",
                    "enum": ["push", "pop", "list", "drop"],
                    "description": "Stash action (for 'stash' operation)"
                },
                "index": {
                    "type": "integer",
                    "description": "Stash index (for 'stash' with 'drop' action)"
                },
                "keep_index": {
                    "type": "boolean",
                    "description": "For 'stash push': preserve staged changes in the working tree after stashing — only unstaged changes go into the stash."
                },
                "include_untracked": {
                    "type": "boolean",
                    "description": "For 'stash push': also stash untracked files (-u). Without this, `git stash push` only touches tracked files."
                },
                "path": {
                    "type": "string",
                    "description": "Optional repository path authorized by the agent's workspace policy. Defaults to workspace root."
                }
            },
            "required": ["operation"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let operation = match args.get("operation").and_then(|v| v.as_str()) {
            Some(op) => op,
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: ToolOutput::default(),
                    error: Some("Missing 'operation' parameter".into()),
                });
            }
        };

        let requires_write_access = self.requires_write_access(operation, &args);
        let path = args.get("path").and_then(|v| v.as_str());
        let working_dir = match self.resolve_working_dir(path, requires_write_access) {
            Ok(d) => d,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: ToolOutput::default(),
                    error: Some(format!("Invalid path: {e}")),
                });
            }
        };

        // Repository discovery must not escape the root that authorized this path.
        let authorized_roots = if requires_write_access {
            self.security.approved_write_roots(&working_dir)
        } else {
            self.security.approved_read_roots(&working_dir)
        };
        if !self.has_repository_within_authorized_roots(&working_dir, &authorized_roots) {
            let path_display = working_dir.display().to_string();
            let error_msg = crate::i18n::get_required_tool_string_with_args(
                "tool-git-operations-error-not-in-repo",
                &[("path", &path_display)],
            );
            return Ok(ToolResult {
                success: false,
                output: ToolOutput::default(),
                error: Some(error_msg),
            });
        }

        // Check autonomy level for write operations
        if requires_write_access {
            if !self.security.can_act() {
                return Ok(ToolResult {
                    success: false,
                    output: ToolOutput::default(),
                    error: Some(
                        "Action blocked: git write operations require higher autonomy level".into(),
                    ),
                });
            }

            match self.security.autonomy {
                AutonomyLevel::ReadOnly => {
                    return Ok(ToolResult {
                        success: false,
                        output: ToolOutput::default(),
                        error: Some("Action blocked: read-only mode".into()),
                    });
                }
                AutonomyLevel::Supervised | AutonomyLevel::Full => {}
            }
        }

        // Record action for rate limiting
        if !self.security.record_action() {
            return Ok(ToolResult {
                success: false,
                output: ToolOutput::default(),
                error: Some("Action blocked: rate limit exceeded".into()),
            });
        }

        // Execute the requested operation
        match operation {
            "status" => self.git_status(args, &working_dir).await,
            "diff" => self.git_diff(args, &working_dir).await,
            "log" => self.git_log(args, &working_dir).await,
            "branch" => self.git_branch(args, &working_dir).await,
            "commit" => self.git_commit(args, &working_dir).await,
            "add" => self.git_add(args, &working_dir).await,
            "checkout" => self.git_checkout(args, &working_dir).await,
            "stash" => self.git_stash(args, &working_dir).await,
            _ => Ok(ToolResult {
                success: false,
                output: ToolOutput::default(),
                error: Some(format!("Unknown operation: {operation}")),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use zeroclaw_config::policy::SecurityPolicy;

    fn test_tool(dir: &std::path::Path) -> GitOperationsTool {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            workspace_dir: dir.to_path_buf(),
            ..SecurityPolicy::default()
        });
        GitOperationsTool::new(security, dir.to_path_buf())
    }

    /// Initialise a git repo for tests with commit/tag signing disabled and a
    /// fixed identity. Tests run real `git commit`; without this they inherit
    /// the developer's global `commit.gpgsign`, blocking the suite on a
    /// hardware-key tap.
    fn git_init_no_sign(dir: &std::path::Path, extra_init: &[&str]) {
        let mut init = vec!["init"];
        init.extend_from_slice(extra_init);
        for args in [
            init.as_slice(),
            &["config", "user.email", "test@test.com"],
            &["config", "user.name", "Test"],
            &["config", "commit.gpgsign", "false"],
            &["config", "tag.gpgsign", "false"],
        ] {
            std::process::Command::new("git")
                .args(args)
                .current_dir(dir)
                .output()
                .unwrap();
        }
    }

    fn test_tool_with_allowed_root(
        dir: &std::path::Path,
        allowed_root: std::path::PathBuf,
    ) -> GitOperationsTool {
        test_tool_with_allowed_roots(dir, vec![allowed_root])
    }

    fn test_tool_with_allowed_roots(
        dir: &std::path::Path,
        allowed_roots: Vec<std::path::PathBuf>,
    ) -> GitOperationsTool {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            workspace_dir: dir.to_path_buf(),
            allowed_roots,
            ..SecurityPolicy::default()
        });
        GitOperationsTool::new(security, dir.to_path_buf())
    }

    fn test_tool_with_read_only_root(
        dir: &std::path::Path,
        read_only_root: std::path::PathBuf,
    ) -> GitOperationsTool {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            workspace_dir: dir.to_path_buf(),
            allowed_roots_read_only: vec![read_only_root],
            ..SecurityPolicy::default()
        });
        GitOperationsTool::new(security, dir.to_path_buf())
    }

    fn test_tool_with_allowed_and_read_only_roots(
        dir: &std::path::Path,
        allowed_root: std::path::PathBuf,
        read_only_root: std::path::PathBuf,
    ) -> GitOperationsTool {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            workspace_dir: dir.to_path_buf(),
            allowed_roots: vec![allowed_root],
            allowed_roots_read_only: vec![read_only_root],
            ..SecurityPolicy::default()
        });
        GitOperationsTool::new(security, dir.to_path_buf())
    }

    fn test_tool_with_write_only_root(
        dir: &std::path::Path,
        write_only_root: std::path::PathBuf,
    ) -> GitOperationsTool {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            workspace_dir: dir.to_path_buf(),
            allowed_roots_write_only: vec![write_only_root],
            ..SecurityPolicy::default()
        });
        GitOperationsTool::new(security, dir.to_path_buf())
    }

    #[test]
    fn sanitize_git_blocks_injection() {
        let tmp = TempDir::new().unwrap();
        let tool = test_tool(tmp.path());

        // Should block dangerous arguments
        assert!(tool.sanitize_git_args("--exec=rm -rf /").is_err());
        assert!(tool.sanitize_git_args("$(echo pwned)").is_err());
        assert!(tool.sanitize_git_args("`malicious`").is_err());
        assert!(tool.sanitize_git_args("arg | cat").is_err());
        assert!(tool.sanitize_git_args("arg; rm file").is_err());
    }

    #[test]
    fn sanitize_git_blocks_pager_editor_injection() {
        let tmp = TempDir::new().unwrap();
        let tool = test_tool(tmp.path());

        assert!(tool.sanitize_git_args("--pager=less").is_err());
        assert!(tool.sanitize_git_args("--editor=vim").is_err());
    }

    #[test]
    fn sanitize_git_blocks_config_injection() {
        let tmp = TempDir::new().unwrap();
        let tool = test_tool(tmp.path());

        // Exact `-c` flag (config injection)
        assert!(tool.sanitize_git_args("-c core.sshCommand=evil").is_err());
        assert!(tool.sanitize_git_args("-c=core.pager=less").is_err());
    }

    #[test]
    fn sanitize_git_blocks_no_verify() {
        let tmp = TempDir::new().unwrap();
        let tool = test_tool(tmp.path());

        assert!(tool.sanitize_git_args("--no-verify").is_err());
    }

    #[test]
    fn sanitize_git_blocks_redirect_in_args() {
        let tmp = TempDir::new().unwrap();
        let tool = test_tool(tmp.path());

        assert!(tool.sanitize_git_args("file.txt > /tmp/out").is_err());
    }

    #[test]
    fn sanitize_git_cached_not_blocked() {
        let tmp = TempDir::new().unwrap();
        let tool = test_tool(tmp.path());

        // --cached must NOT be blocked by the `-c` check
        assert!(tool.sanitize_git_args("--cached").is_ok());
        // Other safe flags starting with -c prefix
        assert!(tool.sanitize_git_args("-cached").is_ok());
    }

    #[test]
    fn sanitize_git_allows_safe() {
        let tmp = TempDir::new().unwrap();
        let tool = test_tool(tmp.path());

        // Should allow safe arguments
        assert!(tool.sanitize_git_args("main").is_ok());
        assert!(tool.sanitize_git_args("feature/test-branch").is_ok());
        assert!(tool.sanitize_git_args("--cached").is_ok());
        assert!(tool.sanitize_git_args("src/main.rs").is_ok());
        assert!(tool.sanitize_git_args(".").is_ok());
    }

    #[test]
    fn requires_write_detection() {
        let tmp = TempDir::new().unwrap();
        let tool = test_tool(tmp.path());

        assert!(tool.requires_write_access("commit", &json!({})));
        assert!(tool.requires_write_access("add", &json!({})));
        assert!(tool.requires_write_access("checkout", &json!({})));
        assert!(tool.requires_write_access("stash", &json!({})));
        assert!(tool.requires_write_access("stash", &json!({"action": "push"})));

        assert!(!tool.requires_write_access("status", &json!({})));
        assert!(!tool.requires_write_access("diff", &json!({})));
        assert!(!tool.requires_write_access("log", &json!({})));
        assert!(!tool.requires_write_access("branch", &json!({})));
        assert!(!tool.requires_write_access("stash", &json!({"action": "list"})));
        assert!(!tool.requires_write_access("status", &json!({"subcommand": "add"})));
    }

    #[tokio::test]
    async fn git_operations_do_not_expose_linked_worktree_lifecycle() {
        let tmp = TempDir::new().unwrap();
        git_init_no_sign(tmp.path(), &[]);
        let tool = test_tool(tmp.path());

        let schema = tool.parameters_schema();
        let operations = schema["properties"]["operation"]["enum"]
            .as_array()
            .unwrap();
        assert!(
            !operations.iter().any(|operation| operation == "worktree"),
            "the public Git operation schema must not advertise worktree"
        );

        let result = tool
            .execute(json!({"operation": "worktree"}))
            .await
            .unwrap();
        assert!(!result.success, "worktree must not be dispatched");
        assert!(
            result
                .error
                .as_deref()
                .is_some_and(|error| error.contains("Unknown operation")),
            "unexpected worktree rejection: {:?}",
            result.error
        );
    }

    #[tokio::test]
    async fn git_credential_op_fails_fast_without_terminal_prompt() {
        let tmp = TempDir::new().unwrap();
        git_init_no_sign(tmp.path(), &[]);
        let tool = test_tool(tmp.path());

        let fetch = tool.run_git_command(
            &["fetch", "https://127.0.0.1:1/private/repo.git"],
            tmp.path(),
        );
        let res = tokio::time::timeout(std::time::Duration::from_secs(10), fetch).await;

        assert!(
            res.is_ok(),
            "git fetch hung — it likely prompted for credentials on the terminal"
        );
        assert!(
            res.unwrap().is_err(),
            "fetch to an unreachable private remote should fail, not succeed"
        );
    }

    #[tokio::test]
    async fn blocks_readonly_mode_for_write_ops() {
        let tmp = TempDir::new().unwrap();
        git_init_no_sign(tmp.path(), &[]);

        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::ReadOnly,
            ..SecurityPolicy::default()
        });
        let tool = GitOperationsTool::new(security, tmp.path().to_path_buf());

        let result = tool
            .execute(json!({"operation": "commit", "message": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        // can_act() returns false for ReadOnly, so we get the "higher autonomy level" message
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or("")
                .contains("higher autonomy")
        );
    }

    #[tokio::test]
    async fn allows_branch_listing_in_readonly_mode() {
        let tmp = TempDir::new().unwrap();
        git_init_no_sign(tmp.path(), &[]);

        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::ReadOnly,
            ..SecurityPolicy::default()
        });
        let tool = GitOperationsTool::new(security, tmp.path().to_path_buf());

        let result = tool.execute(json!({"operation": "branch"})).await.unwrap();
        // Branch listing must not be blocked by read-only autonomy
        let error_msg = result.error.as_deref().unwrap_or("");
        assert!(
            !error_msg.contains("read-only") && !error_msg.contains("higher autonomy"),
            "branch listing should not be blocked in read-only mode, got: {error_msg}"
        );
    }

    #[tokio::test]
    async fn allows_readonly_ops_in_readonly_mode() {
        let tmp = TempDir::new().unwrap();
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::ReadOnly,
            ..SecurityPolicy::default()
        });
        let tool = GitOperationsTool::new(security, tmp.path().to_path_buf());

        // This will fail because there's no git repo, but it shouldn't be blocked by autonomy
        let result = tool.execute(json!({"operation": "status"})).await.unwrap();
        // The error should be about git (not about autonomy/read-only mode)
        assert!(!result.success, "Expected failure due to missing git repo");
        let error_msg = result.error.as_deref().unwrap_or("");
        assert!(
            !error_msg.contains("read-only") && !error_msg.contains("autonomy"),
            "Error should be about git, not about autonomy restrictions: {error_msg}"
        );
    }

    #[tokio::test]
    async fn rejects_missing_operation() {
        let tmp = TempDir::new().unwrap();
        let tool = test_tool(tmp.path());

        let result = tool.execute(json!({})).await.unwrap();
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or("")
                .contains("Missing 'operation'")
        );
    }

    #[tokio::test]
    async fn rejects_unknown_operation() {
        let tmp = TempDir::new().unwrap();
        git_init_no_sign(tmp.path(), &[]);

        let tool = test_tool(tmp.path());

        let result = tool.execute(json!({"operation": "push"})).await.unwrap();
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or("")
                .contains("Unknown operation")
        );
    }

    #[tokio::test]
    async fn commit_message_preserves_blank_line_between_subject_and_body() {
        let tmp = TempDir::new().unwrap();
        git_init_no_sign(tmp.path(), &[]);
        // Create an initial commit so HEAD exists.
        std::fs::write(tmp.path().join("README.md"), "hello").unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(tmp.path())
            .output()
            .unwrap();

        let tool = test_tool(tmp.path());

        let msg = "fix(foo): subject line\n\nThis is the body paragraph.\n\nSecond paragraph.";
        let result = tool
            .execute(json!({"operation": "commit", "message": msg}))
            .await
            .unwrap();
        assert!(result.success, "commit failed: {:?}", result.error);

        // Read back the raw commit message via git log.
        let log_out = std::process::Command::new("git")
            .args(["log", "-1", "--format=%B"])
            .current_dir(tmp.path())
            .output()
            .unwrap();
        let log_msg = String::from_utf8_lossy(&log_out.stdout);

        // Subject line must be on its own line.
        assert!(
            log_msg.starts_with("fix(foo): subject line\n"),
            "subject line missing or not first: {log_msg:?}"
        );
        // A blank line must follow the subject.
        assert!(
            log_msg.contains("fix(foo): subject line\n\n"),
            "blank line between subject and body missing: {log_msg:?}"
        );
        // Body text must be present.
        assert!(
            log_msg.contains("This is the body paragraph."),
            "body paragraph missing: {log_msg:?}"
        );
    }

    #[test]
    fn truncates_multibyte_commit_message_without_panicking() {
        let long = "🦀".repeat(2500);
        let truncated = GitOperationsTool::truncate_commit_message(&long);

        assert_eq!(truncated.chars().count(), 2000);
    }

    #[test]
    fn resolve_working_dir_none_returns_workspace() {
        let tmp = TempDir::new().unwrap();
        let tool = test_tool(tmp.path());

        let result = tool.resolve_working_dir(None, false).unwrap();
        assert_eq!(result, tmp.path().to_path_buf());
    }

    #[test]
    fn resolve_working_dir_empty_returns_workspace() {
        let tmp = TempDir::new().unwrap();
        let tool = test_tool(tmp.path());

        let result = tool.resolve_working_dir(Some(""), false).unwrap();
        assert_eq!(result, tmp.path().to_path_buf());
    }

    #[test]
    fn resolve_working_dir_valid_subdir() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir(tmp.path().join("subproject")).unwrap();
        let tool = test_tool(tmp.path());

        let result = tool.resolve_working_dir(Some("subproject"), false).unwrap();
        let expected = tmp.path().join("subproject").canonicalize().unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn resolve_working_dir_rejects_traversal() {
        let tmp = TempDir::new().unwrap();
        let tool = test_tool(tmp.path());

        let result = tool.resolve_working_dir(Some(".."), false);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not authorized for this Git operation"),
            "Expected traversal rejection, got: {err_msg}"
        );
    }

    #[tokio::test]
    async fn git_operations_work_in_subdirectory() {
        let tmp = TempDir::new().unwrap();
        let sub = tmp.path().join("nested");
        std::fs::create_dir(&sub).unwrap();
        git_init_no_sign(&sub, &[]);

        let tool = test_tool(tmp.path());

        let result = tool
            .execute(json!({"operation": "status", "path": "nested"}))
            .await
            .unwrap();
        assert!(
            result.success,
            "Expected success, got error: {:?}",
            result.error
        );
        assert!(result.output.contains("branch"));
    }

    #[tokio::test]
    async fn git_operations_work_in_configured_allowed_root() {
        let workspace = TempDir::new().unwrap();
        let allowed_root = TempDir::new().unwrap();
        git_init_no_sign(allowed_root.path(), &[]);
        let tool = test_tool_with_allowed_root(workspace.path(), allowed_root.path().to_path_buf());

        let result = tool
            .execute(json!({"operation": "status", "path": allowed_root.path()}))
            .await
            .unwrap();

        assert!(
            result.success,
            "Expected success, got error: {:?}",
            result.error
        );
        assert!(result.output.contains("branch"));
    }

    #[tokio::test]
    async fn git_operations_find_parent_repository_through_allowed_parent_of_workspace() {
        let repository = TempDir::new().unwrap();
        bootstrap_repo(repository.path(), &[]).await;
        let workspace = repository.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        let tool = test_tool_with_allowed_root(&workspace, repository.path().to_path_buf());

        let result = tool.execute(json!({"operation": "status"})).await.unwrap();

        assert!(
            result.success,
            "an allowed parent repository must remain reachable: {:?}",
            result.error
        );
    }

    #[tokio::test]
    async fn git_operations_find_parent_repository_regardless_of_overlapping_grant_order() {
        let workspace = TempDir::new().unwrap();
        let repository = TempDir::new().unwrap();
        bootstrap_repo(repository.path(), &[]).await;
        let child = repository.path().join("authorized-child");
        std::fs::create_dir(&child).unwrap();

        for allowed_roots in [
            vec![child.clone(), repository.path().to_path_buf()],
            vec![repository.path().to_path_buf(), child.clone()],
        ] {
            let tool = test_tool_with_allowed_roots(workspace.path(), allowed_roots);
            let result = tool
                .execute(json!({"operation": "status", "path": &child}))
                .await
                .unwrap();
            assert!(
                result.success,
                "overlapping grants must reach the authorized parent repository: {:?}",
                result.error
            );
        }
    }

    #[tokio::test]
    async fn git_operations_do_not_mutate_parent_repository_via_read_only_grant() {
        let workspace = TempDir::new().unwrap();
        let repository = TempDir::new().unwrap();
        bootstrap_repo(repository.path(), &[]).await;
        let writable_child = repository.path().join("writable-child");
        std::fs::create_dir(&writable_child).unwrap();
        std::fs::write(writable_child.join("new-file"), "content").unwrap();
        let index_path = repository.path().join(".git/index");
        let index_before = std::fs::read(&index_path).unwrap();
        let tool = test_tool_with_allowed_and_read_only_roots(
            workspace.path(),
            writable_child.clone(),
            repository.path().to_path_buf(),
        );

        let read = tool
            .execute(json!({"operation": "status", "path": &writable_child}))
            .await
            .unwrap();
        assert!(
            read.success,
            "the read-only parent grant should permit status: {:?}",
            read.error
        );

        let write = tool
            .execute(json!({
                "operation": "add",
                "path": &writable_child,
                "paths": "new-file"
            }))
            .await
            .unwrap();
        assert!(
            !write.success,
            "a read-only parent grant must not authorize mutation: {write:?}"
        );
        assert_eq!(
            std::fs::read(&index_path).unwrap(),
            index_before,
            "rejected mutation must not change the parent repository index"
        );
    }

    #[tokio::test]
    async fn git_operations_reject_reads_from_write_only_root() {
        let workspace = TempDir::new().unwrap();
        let write_only_root = TempDir::new().unwrap();
        git_init_no_sign(write_only_root.path(), &[]);
        let tool =
            test_tool_with_write_only_root(workspace.path(), write_only_root.path().to_path_buf());

        for operation in ["status", "diff", "log", "branch"] {
            let result = tool
                .execute(json!({"operation": operation, "path": write_only_root.path()}))
                .await
                .unwrap();
            assert!(
                !result.success,
                "{operation} must not read from a write-only root"
            );
            assert!(
                result
                    .error
                    .as_deref()
                    .is_some_and(|error| error.contains("Invalid path")),
                "{operation} must fail during authorization: {:?}",
                result.error
            );
        }
    }

    #[tokio::test]
    async fn git_operations_adds_in_configured_allowed_root() {
        let workspace = TempDir::new().unwrap();
        let allowed_root = TempDir::new().unwrap();
        git_init_no_sign(allowed_root.path(), &[]);
        std::fs::write(allowed_root.path().join("tracked.txt"), "content").unwrap();
        let tool = test_tool_with_allowed_root(workspace.path(), allowed_root.path().to_path_buf());

        let result = tool
            .execute(json!({
                "operation": "add",
                "path": allowed_root.path(),
                "paths": "tracked.txt"
            }))
            .await
            .unwrap();

        assert!(
            result.success,
            "Expected success, got error: {:?}",
            result.error
        );
    }

    #[tokio::test]
    async fn git_operations_reject_writes_from_read_only_root() {
        let workspace = TempDir::new().unwrap();
        let read_only_root = TempDir::new().unwrap();
        git_init_no_sign(read_only_root.path(), &[]);
        std::fs::write(read_only_root.path().join("tracked.txt"), "content").unwrap();
        let tool =
            test_tool_with_read_only_root(workspace.path(), read_only_root.path().to_path_buf());

        let result = tool
            .execute(json!({
                "operation": "add",
                "path": read_only_root.path(),
                "paths": "tracked.txt"
            }))
            .await
            .unwrap();

        assert!(!result.success, "add must not write to a read-only root");
        assert!(
            result
                .error
                .as_deref()
                .is_some_and(|error| error.contains("Invalid path")),
            "add must fail during authorization: {:?}",
            result.error
        );
    }

    #[tokio::test]
    async fn git_operations_rejects_parent_repository_above_allowed_root() {
        let workspace = TempDir::new().unwrap();
        let parent_repository = TempDir::new().unwrap();
        git_init_no_sign(parent_repository.path(), &[]);
        let allowed_child = parent_repository.path().join("allowed-child");
        std::fs::create_dir(&allowed_child).unwrap();
        std::fs::write(allowed_child.join("tracked.txt"), "content").unwrap();
        let tool = test_tool_with_allowed_root(workspace.path(), allowed_child.clone());

        for args in [
            json!({"operation": "status", "path": &allowed_child}),
            json!({
                "operation": "add",
                "path": &allowed_child,
                "paths": "tracked.txt"
            }),
        ] {
            let result = tool.execute(args).await.unwrap();
            assert!(
                !result.success,
                "parent repository must not be usable through an allowed child: {result:?}"
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn git_operations_rejects_git_symlink_to_parent_repository() {
        let workspace = TempDir::new().unwrap();
        let parent_repository = TempDir::new().unwrap();
        git_init_no_sign(parent_repository.path(), &[]);
        let allowed_child = parent_repository.path().join("allowed-child");
        std::fs::create_dir(&allowed_child).unwrap();
        std::fs::write(allowed_child.join("tracked.txt"), "content").unwrap();
        std::os::unix::fs::symlink(
            parent_repository.path().join(".git"),
            allowed_child.join(".git"),
        )
        .unwrap();
        let tool = test_tool_with_allowed_root(workspace.path(), allowed_child.clone());

        for args in [
            json!({"operation": "status", "path": &allowed_child}),
            json!({
                "operation": "add",
                "path": &allowed_child,
                "paths": "tracked.txt"
            }),
        ] {
            let result = tool.execute(args).await.unwrap();
            assert!(
                !result.success,
                "Git metadata outside the allowed root must be denied: {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn git_operations_rejects_authorized_linked_worktree() {
        let workspace = TempDir::new().unwrap();
        let parent_repository = TempDir::new().unwrap();
        let linked_worktree_parent = TempDir::new().unwrap();
        bootstrap_repo(parent_repository.path(), &[]).await;
        let linked_worktree = linked_worktree_parent.path().join("linked-worktree");

        let status = std::process::Command::new("git")
            .args(["worktree", "add", "--detach"])
            .arg(&linked_worktree)
            .current_dir(parent_repository.path())
            .status()
            .unwrap();
        assert!(status.success(), "linked worktree setup must succeed");
        assert!(linked_worktree.join(".git").is_file());

        let tool = test_tool_with_allowed_root(workspace.path(), linked_worktree.clone());
        let result = tool
            .execute(json!({"operation": "status", "path": &linked_worktree}))
            .await
            .unwrap();

        assert!(
            !result.success,
            "linked worktree metadata indirection must fail closed: {result:?}"
        );
        assert!(
            result
                .error
                .as_deref()
                .is_some_and(|error| error.contains("Not in a Git repository")),
            "linked worktree must fail as not-in-repository: {:?}",
            result.error
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn git_status_in_read_only_root_does_not_run_repository_fsmonitor() {
        let workspace = TempDir::new().unwrap();
        let read_only_root = TempDir::new().unwrap();
        bootstrap_repo(read_only_root.path(), &[]).await;
        let marker = workspace.path().join("fsmonitor-ran");
        let fsmonitor = format!("sh -c 'touch {}'", marker.display());
        let config = std::process::Command::new("git")
            .args(["config", "core.fsmonitor", &fsmonitor])
            .current_dir(read_only_root.path())
            .status()
            .unwrap();
        assert!(
            config.success(),
            "test repository configuration must succeed"
        );
        let tool =
            test_tool_with_read_only_root(workspace.path(), read_only_root.path().to_path_buf());

        let result = tool
            .execute(json!({"operation": "status", "path": read_only_root.path()}))
            .await
            .unwrap();

        assert!(result.success, "status failed: {:?}", result.error);
        assert!(
            !marker.exists(),
            "read-only status must not execute repository core.fsmonitor"
        );
    }

    #[tokio::test]
    async fn git_status_does_not_refresh_index_in_read_only_root() {
        let workspace = TempDir::new().unwrap();
        let read_only_root = TempDir::new().unwrap();
        bootstrap_repo(read_only_root.path(), &["tracked.txt"]).await;
        std::thread::sleep(std::time::Duration::from_secs(1));
        std::fs::write(read_only_root.path().join("tracked.txt"), "initial").unwrap();
        let index_path = read_only_root.path().join(".git/index");
        let index_before = std::fs::read(&index_path).unwrap();
        let tool =
            test_tool_with_read_only_root(workspace.path(), read_only_root.path().to_path_buf());

        let result = tool
            .execute(json!({"operation": "status", "path": read_only_root.path()}))
            .await
            .unwrap();

        assert!(
            result.success,
            "Expected success, got error: {:?}",
            result.error
        );
        assert_eq!(
            std::fs::read(&index_path).unwrap(),
            index_before,
            "status must not refresh the index in a read-only root"
        );
    }

    #[tokio::test]
    async fn git_stash_list_reads_from_configured_read_only_root() {
        let workspace = TempDir::new().unwrap();
        let read_only_root = TempDir::new().unwrap();
        bootstrap_repo(read_only_root.path(), &[]).await;
        let tool =
            test_tool_with_read_only_root(workspace.path(), read_only_root.path().to_path_buf());

        let result = tool
            .execute(json!({
                "operation": "stash",
                "action": "list",
                "path": read_only_root.path()
            }))
            .await
            .unwrap();

        assert!(
            result.success,
            "Expected success, got error: {:?}",
            result.error
        );
    }

    async fn bootstrap_repo(dir: &std::path::Path, tracked_files: &[&str]) {
        git_init_no_sign(dir, &["-b", "master"]);
        std::fs::write(dir.join("README.md"), "hello").unwrap();
        for f in tracked_files {
            std::fs::write(dir.join(f), "initial").unwrap();
        }
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "initial"])
            .current_dir(dir)
            .output()
            .unwrap();
    }

    #[tokio::test]
    async fn stash_push_default_stashes_staged_and_unstaged() {
        let tmp = TempDir::new().unwrap();
        bootstrap_repo(tmp.path(), &["staged.txt", "unstaged.txt"]).await;

        std::fs::write(tmp.path().join("staged.txt"), "s-modified").unwrap();
        std::fs::write(tmp.path().join("unstaged.txt"), "u-modified").unwrap();
        std::process::Command::new("git")
            .args(["add", "staged.txt"])
            .current_dir(tmp.path())
            .output()
            .unwrap();

        let tool = test_tool(tmp.path());
        let result = tool
            .execute(json!({"operation": "stash", "action": "push"}))
            .await
            .unwrap();
        assert!(result.success, "stash push failed: {:?}", result.error);

        let status = std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(tmp.path())
            .output()
            .unwrap();
        let status_out = String::from_utf8_lossy(&status.stdout);
        assert!(
            status_out.trim().is_empty(),
            "expected clean working tree after default stash, got: {status_out:?}"
        );
    }

    #[tokio::test]
    async fn stash_push_with_keep_index_preserves_staged() {
        let tmp = TempDir::new().unwrap();
        bootstrap_repo(tmp.path(), &["staged.txt", "unstaged.txt"]).await;

        std::fs::write(tmp.path().join("staged.txt"), "s-modified").unwrap();
        std::fs::write(tmp.path().join("unstaged.txt"), "u-modified").unwrap();
        std::process::Command::new("git")
            .args(["add", "staged.txt"])
            .current_dir(tmp.path())
            .output()
            .unwrap();

        let tool = test_tool(tmp.path());
        let result = tool
            .execute(json!({
                "operation": "stash",
                "action": "push",
                "keep_index": true,
            }))
            .await
            .unwrap();
        assert!(result.success, "stash push -k failed: {:?}", result.error);

        let status = std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(tmp.path())
            .output()
            .unwrap();
        let status_out = String::from_utf8_lossy(&status.stdout).to_string();
        // `staged.txt` modification still present and staged (`M ` prefix);
        // `unstaged.txt` modification was stashed away — file matches HEAD.
        assert!(
            status_out.contains("M  staged.txt"),
            "staged modification should remain staged, status: {status_out:?}"
        );
        assert!(
            !status_out.contains("unstaged.txt"),
            "unstaged modification should have been stashed, status: {status_out:?}"
        );
    }

    #[tokio::test]
    async fn stash_push_with_paths_scopes_to_pathspec() {
        let tmp = TempDir::new().unwrap();
        bootstrap_repo(tmp.path(), &["a.txt", "b.txt"]).await;

        std::fs::write(tmp.path().join("a.txt"), "a-modified").unwrap();
        std::fs::write(tmp.path().join("b.txt"), "b-modified").unwrap();

        let tool = test_tool(tmp.path());
        let result = tool
            .execute(json!({
                "operation": "stash",
                "action": "push",
                "paths": "a.txt",
            }))
            .await
            .unwrap();
        assert!(
            result.success,
            "stash push -- a.txt failed: {:?}",
            result.error
        );

        let status = std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(tmp.path())
            .output()
            .unwrap();
        let status_out = String::from_utf8_lossy(&status.stdout).to_string();
        assert!(
            !status_out.contains("a.txt"),
            "a.txt should have been stashed, status: {status_out:?}"
        );
        assert!(
            status_out.contains("b.txt"),
            "b.txt should remain modified, status: {status_out:?}"
        );
    }

    #[tokio::test]
    async fn stash_push_with_custom_message() {
        let tmp = TempDir::new().unwrap();
        bootstrap_repo(tmp.path(), &["a.txt"]).await;
        std::fs::write(tmp.path().join("a.txt"), "a-modified").unwrap();

        let tool = test_tool(tmp.path());
        let result = tool
            .execute(json!({
                "operation": "stash",
                "action": "push",
                "message": "scoped-fix-wip",
            }))
            .await
            .unwrap();
        assert!(result.success, "stash push -m failed: {:?}", result.error);

        let list = std::process::Command::new("git")
            .args(["stash", "list"])
            .current_dir(tmp.path())
            .output()
            .unwrap();
        let list_out = String::from_utf8_lossy(&list.stdout).to_string();
        assert!(
            list_out.contains("scoped-fix-wip"),
            "custom stash message missing from list, got: {list_out:?}"
        );
    }

    #[tokio::test]
    async fn stash_push_with_include_untracked_captures_new_files() {
        let tmp = TempDir::new().unwrap();
        bootstrap_repo(tmp.path(), &[]).await;
        std::fs::write(tmp.path().join("new.txt"), "untracked").unwrap();

        let tool = test_tool(tmp.path());
        let result = tool
            .execute(json!({
                "operation": "stash",
                "action": "push",
                "include_untracked": true,
            }))
            .await
            .unwrap();
        assert!(result.success, "stash push -u failed: {:?}", result.error);

        let status = std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(tmp.path())
            .output()
            .unwrap();
        let status_out = String::from_utf8_lossy(&status.stdout);
        assert!(
            status_out.trim().is_empty(),
            "expected clean tree after -u stash, got: {status_out:?}"
        );
    }

    #[tokio::test]
    async fn add_stages_multiple_space_separated_paths() {
        let tmp = TempDir::new().unwrap();
        git_init_no_sign(tmp.path(), &[]);
        std::fs::write(tmp.path().join("a.txt"), "a").unwrap();
        std::fs::write(tmp.path().join("b.txt"), "b").unwrap();

        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Full,
            workspace_dir: tmp.path().to_path_buf(),
            ..SecurityPolicy::default()
        });
        let tool = GitOperationsTool::new(security, tmp.path().to_path_buf());

        let result = tool
            .execute(json!({"operation": "add", "paths": "a.txt b.txt"}))
            .await
            .unwrap();
        assert!(result.success, "add failed: {:?}", result.error);

        let status = std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(tmp.path())
            .output()
            .unwrap();
        let out = String::from_utf8_lossy(&status.stdout);
        assert!(out.contains("A  a.txt"), "a.txt not staged: {out:?}");
        assert!(out.contains("A  b.txt"), "b.txt not staged: {out:?}");
    }

    #[tokio::test]
    async fn non_repository_error_includes_path_context_and_recovery_hint() {
        let tmp = TempDir::new().unwrap();
        // Do NOT git-init the temp dir — we want a non-repository path.
        let tool = test_tool(tmp.path());

        let result = tool.execute(json!({"operation": "status"})).await.unwrap();

        assert!(
            !result.success,
            "git_operations should fail when not in a repository"
        );

        let error = result.error.as_deref().unwrap_or("");
        let path_display = tmp.path().display().to_string();

        // The error message must include the resolved working directory
        // path so the user can see where the tool was looking.
        assert!(
            error.contains(&path_display),
            "error should contain the working directory path '{path_display}', got: {error}"
        );

        // The error message must include recovery guidance keywords
        // that tell the user how to resolve the issue.
        assert!(
            error.contains("worktree") || error.contains("work tree") || error.contains("path"),
            "error should contain a recovery keyword (worktree/work tree/path), got: {error}"
        );
        assert!(
            error.contains("initialize") || error.contains("init"),
            "error should mention initializing a repository, got: {error}"
        );
    }
}
