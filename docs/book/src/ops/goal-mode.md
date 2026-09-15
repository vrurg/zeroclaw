# Goal Mode

Goal Mode is an experimental, opt-in way to let an agent pursue a declared
success criterion across more than one foreground turn. It is not a general
background-job system: a Goal belongs to one existing session, runs in that
session's foreground slot, and remains under the session's normal authority.

V1 accepts explicit commands only from Matrix and [zerocode](../zerocode/overview.md).
It has no Web controls, no natural-language admission, and no model-visible
Goal-management tools.

## Enable it deliberately

Goal Mode is disabled unless the complete `[goal]` section is configured. An
enabled configuration must declare both defaults and a verifier provider. Zero
is an explicit unlimited value; omitting either default is an error, not a
hidden product default.

> **Warning:** A reload is a policy cutover, not a way to temporarily park
> active Goals. Reloading a configuration that disables Goal Mode, or removes
> or disables a Goal's owning agent or bound Matrix channel, durably cancels
> that nonterminal Goal. Re-enabling the setting later cannot resume it.

```toml
[goal]
enabled = true
# A positive value is a finite default. Zero means unlimited for that dimension.
default_token_limit = 120000
default_cost_limit_usd = 5.00

[goal.verifier]
# An existing provider-profile reference, resolved like other providers.
model_provider = "openai.default"
# Optional model override for the verifier.
# model = "gpt-5"
```

The verifier is mandatory. It is a normal configured provider call and has its
own Goal-attributed usage. The owning agent and any authorized foreground child
retain their ordinary provider selection.

For an explicitly unlimited default, declare both values as zero:

```toml
[goal]
enabled = true
default_token_limit = 0
default_cost_limit_usd = 0.0

[goal.verifier]
model_provider = "openai.default"
```

When a Goal has a finite cost limit, the requested provider/model route and all
prior recorded Goal usage must be priced before the next operation is admitted.
An unpriced route reached later by ordinary provider failover is recorded, then
fails the Goal after settlement; no following operation is admitted. Token-only
and unlimited Goals can run without pricing, but still require usable token
usage.

## Commands

Use these commands in an existing Matrix conversation or a zerocode Chat
session. Goal Mode is unavailable for zerocode ACP sessions:

```text
/goal start [--tokens N] [--cost-usd D] -- SUCCESS CRITERION
/goal start --unlimited -- SUCCESS CRITERION
/goal status
/goal budget
/goal budget set --tokens N [--cost-usd D]
/goal budget set --cost-usd D [--tokens N]
/goal budget set --unlimited
/goal pause
/goal resume
/goal cancel
/goal help
```

The `--` delimiter is required. The text after it is the declared success
criterion, not an additional authority source or a task identifier. ZeroClaw
injects that exact criterion into the Goal parent prompt and presents it to the
verifier with the exact candidate response. It is not editable after start and
is limited to 4096 characters.

With no `start` flags, Goal Mode copies both configured defaults. Supplying a
finite flag replaces both defaults: an omitted dimension becomes unlimited.
`budget set` follows the same replacement rule. `--unlimited` cannot be mixed
with finite flags. Token limits must be positive integers and cost limits must
be finite positive values. Unlike `start`, `budget set` has no defaults-copying
form: it requires a finite selector or `--unlimited`.
Flags use a separate value (`--tokens 1000`), not an equals form such as
`--tokens=1000`.

`/goal help` is local grammar help and remains available while Goal Mode is
disabled. Other commands report that the feature is disabled until the complete
configuration is enabled.

## Lifecycle and completion

Only one current Goal may exist for a session. A terminal Goal remains visible
to `/goal status` until another Goal replaces it or its owning session is
disposed. Starting a Goal while a running or paused one exists is rejected.

A Goal is completed only after its configured verifier receives the exact
success criterion and candidate response and returns `Complete`. A verifier
`Continue` keeps work eligible; `Blocked` pauses the Goal with actionable
blockers. Provider, protocol, attribution, malformed-output, and verifier
failures fail the Goal rather than becoming a semantic blocker.

`/goal pause` first durably fences the Goal as paused, then waits for an
already-admitted operation to settle before returning. `/goal resume` starts a
fresh executor after the persisted checks pass. `/goal cancel` retains the
terminal audit record while the session still exists. Closing, deleting,
killing, or truly replacing a session fences and disposes its Goal control
state; the canonical usage ledger remains intact.

If the admitted operation cannot settle while pausing, Goal Mode fails it
closed as outcome-unknown instead of leaving a resumable paused Goal.

On daemon restart, settled running Goals pause and require an explicit resume.
A reload first re-evaluates Goal policy. A reload that revokes a Goal, for
example by disabling Goal Mode or removing or disabling its owning agent or
bound Matrix channel, durably cancels that Goal; later re-enablement cannot
resume it. A reload that retains authorization pauses settled running Goals in
the same way as restart. Goal Mode does not transfer an in-progress executor.
If an operation was still unsettled, the Goal fails closed as outcome-unknown
and is never replayed. Resuming can therefore repeat externally visible side
effects; use it only when that is acceptable.

## Budget and accounting

Budgets are admission limits, not hard spend ceilings. One already-admitted
logical operation can cross a finite limit; no later operation is admitted once
known usage reaches it. Status and budget output report effective limits,
accounting state, resumability, and pause or blocker detail. The controller
derives usage from the canonical JSONL cost ledger for admission decisions;
use [Cost tracking](./cost-tracking.md) to inspect recorded spend rather than
expecting consumed or remaining values in a Goal response.

Each Goal-owned parent, verifier, or foreground-child operation is serialized.
The normal provider path may retry, route, fail over, or recover a stream, but
the whole logical operation has one Goal admission. Every surfaced usage event
is attributed to the Goal task and its actual provider/model route before that
operation settles.

Missing, invalid, uncertain, or insufficiently attributed usage is not treated
as zero. The Goal fails and admits no further Goal-owned model call. A finite
cost limit also fails if pricing cannot be established. This conservative rule
is intentional: recorded known usage is only a lower bound after an accounting
failure.

Goal admission also requires the canonical cost ledger to be structurally
readable. A malformed or empty ledger row has no trustworthy Goal attribution,
so Goal Mode fails closed rather than guessing that the row belongs to another
task. Repair the ledger through the normal operator recovery procedure before
starting or resuming budgeted Goal work.

## Children, tools, and surfaces

Goal Mode preserves normal authorization and target policy for foreground
`delegate` and `spawn_subagent` calls. Eligible children inherit the Goal's
identity, cancellation, deadline, and usage attribution. Their model calls
share the Goal's serialized admission path.

V1 rejects background, detached, parallel, and recursive Goal-owned children.
It does not remove ordinary child tools: tool execution remains local to each
agentic loop, while model-operation admission is serialized at the Goal.

For Matrix, the adapter uses the authenticated raw Matrix identity captured
before hooks can alter message content. For zerocode, both local IPC and WSS
reuse the daemon's existing session-control permission semantics. A WSS
operator must expose that listener only where its existing ability to create
and control sessions is appropriate; Goal Mode adds no independent remote-user
identity or ownership rule.

## Rollout checklist

Before enabling Goal Mode for real work:

1. Confirm the verifier profile resolves and pricing is present for every route
   that a cost-limited Goal may use.
2. Verify Matrix sender/room policy or zerocode session-control access as
   appropriate for the transport.
3. Exercise `/goal help`, `/goal start`, status, pause, resume, and cancel in
   a non-sensitive session.
4. Record comparable disabled/enabled binary-size, startup-time, and idle-memory
   evidence before adopting it broadly.

See [Cost tracking](./cost-tracking.md) for the canonical ledger and pricing
model, [Background work lifecycle](../architecture/background-work-lifecycle.md)
for the durable lifecycle, and [ADR-008](../architecture/decisions/ADR-008-goal-mode-control-plane-and-usage-accounting.md)
for the architectural decision.
