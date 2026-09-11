# Goal Mode

Goal Mode is an experimental, opt-in control plane for pursuing a declared
success criterion across foreground agent turns. It is disabled by default and
is available only through Matrix and ZeroCode's existing local IPC or WSS
session control. It is not available in Web, Quickstart, or other channels.

## Configure it deliberately

Goal Mode requires declared defaults and a verifier model route when enabled.
An explicit zero default means unlimited for that dimension; omission is an
error while Goal Mode is enabled, so no hidden hard-coded budget is applied.

```toml
[goal]
enabled = true
default_token_limit = 0       # 0 means unlimited
default_cost_limit_usd = 0.0  # 0 means unlimited

[goal.verifier]
model_provider = "openai.default"
# model = "optional-verifier-model-override"
```

The verifier provider must resolve through the ordinary configured provider
profiles. The owning agent and an authorized foreground child keep their normal
provider selection. A cost-limited goal requires complete configured pricing
for eligible routes; token-only and unlimited goals still require valid token
usage. A provider/model priced only by an ordinary reporting catalog is not
enough to admit a cost-limited Goal operation.

## Commands

Only these commands are accepted:

```text
/goal start [--tokens N] [--cost-usd D] -- SUCCESS CRITERION
/goal start --unlimited -- SUCCESS CRITERION
/goal status
/goal budget
/goal budget set [--tokens N] [--cost-usd D]
/goal budget set --unlimited
/goal pause
/goal resume
/goal cancel
/goal help
```

The `--` delimiter and a nonblank success criterion are required for `start`.
The criterion is injected into the agent prompt as user-declared, untrusted
data and is supplied unchanged to the verifier; it is not runtime authority.
Finite token limits are positive integers and finite cost limits are positive,
finite values. Supplying a finite command limit replaces both defaults; an
omitted dimension becomes unlimited. `--unlimited` cannot be combined with
finite flags.

## Lifecycle and accounting

Each session has at most one current goal. A terminal goal remains visible to
that session until it is replaced or the session is disposed. Pause fences new
work but lets an already-admitted model operation settle its usage. Restart
requires an explicit resume; it does not reconstruct a previous working
transcript, so resumed work can repeat external side effects.

`/goal cancel` retains the terminal Goal audit while its session exists. A
session close, delete, kill, or genuine replacement instead fences and drains
the Goal, then removes its Goal control state; already-recorded ledger usage
remains queryable. A live policy revocation (for example, disabling Goal Mode
or removing its Matrix surface) cancels affected Goals and they cannot be
resumed if the policy is later restored. A same-session reconnect is not a
replacement and does not change the Goal.

Budgets are admission limits, not strict spend ceilings: one admitted logical
operation can exceed a finite limit, but no subsequent operation is admitted.
Usage and remaining budget are derived from the canonical ledger. Missing,
invalid, unattributed, or uncertain usage fails the goal closed and prevents
further calls; provider operations are never replayed to guess their outcome.
If interruption leaves an admitted operation unsettled, the Goal is likewise
failed as outcome-unknown rather than being resumed from an incomplete record.

Goal Mode keeps the configured provider path intact. A single serialized Goal
operation may use the ordinary Reliable retry, route, fallback, or stream
recovery behavior; all usage surfaced by that operation is attributed to the
Goal before it settles. This does not make budget limits strict spend ceilings.

A verifier is mandatory. It receives only the success criterion and exact
candidate, then returns `complete`, `continue`, or `blocked`. Only a verified
complete candidate is added to session history and delivered. `blocked` pauses
the goal with actionable blockers; malformed verifier output and provider or
accounting failures fail the goal rather than becoming blockers.

## Design context

The verifier-loop pattern has prior art. [Ralph Loop Agent][ralph-loop] repeats
work until its completion verifier accepts the result and feeds an unsuccessful
verifier's reason into a later iteration. [OpenCode Goal Plugin][opencode-goal]
similarly lets an operator declare success criteria and requires completion
evidence before accepting a completion claim.

Goal Mode deliberately does not inherit either project's control plane. Its
runtime controller, rather than prompt text or a model claim, owns the durable
task transition; its verifier is required and goal-attributed; and its budget,
session-disposal, and trusted-ingress rules remain the contracts described
above.

Goal calls do not add progress chatter for ordinary execution. A verified
completion is delivered, and a verifier-caused pause reports that resolution
is required. Explicit control commands return their own typed acknowledgement.

## Foreground children and trust boundaries

Foreground `delegate` and `spawn_subagent` calls retain their ordinary target
authorization and inherit the goal identity and usage attribution. They run
serially with parent and verifier work. Goal-owned background, detached,
parallel, and recursive work is rejected before work is created.

Matrix uses its immutable authenticated sender identity captured before hooks;
hooks may cancel processing but cannot rewrite Goal authority, session binding,
or the parsed command. ZeroCode uses the existing session-control permission
semantics and connection-registered `tui_id`; it does not add a separate owner
rule or claim that `tui_id` is a remote-user identity. Expose WSS only where
its existing ability to create and control sessions is appropriate.

## Rollout evidence

Before enabling Goal Mode for regular use, compare a disabled configuration
with the intended enabled configuration and record stripped binary size,
startup-to-ready time, and steady-state idle RSS. This evidence determines
whether runtime opt-in remains sufficient; Goal Mode does not add a
compile-time feature switch merely to reduce footprint.

See [ADR-008](../architecture/decisions/ADR-008-goal-mode-control-plane-and-usage-accounting.md)
for the durable source-of-truth decision.

[ralph-loop]: https://github.com/vercel-labs/ralph-loop-agent#verifycompletion
[opencode-goal]: https://github.com/willytop8/OpenCode-goal-plugin#how-it-works
