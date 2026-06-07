# Resumed Session: Refreshing Stale Context Without Restarting

## Problem

You resume a named investigation session ("refund-errors") from yesterday.
Claude already read `refund_policy.md`, the `process_refund` tool schema, and
`escalation_rules.md`, and most of its in-context analysis is still load-bearing.

Since the session ended:
- A teammate changed the **refund threshold rules** in `refund_policy.md`.
- The `process_refund` tool **parameter names** were updated.

Continuing naively means Claude designs the fix against yesterday's snapshot —
correct-looking code that calls a tool signature that no longer exists.

## Root Cause

A resumed session restores the model's prior context verbatim. That context is
*memory of what files said*, not a live view. Anything that changed on disk or
in the MCP server is silently stale until Claude re-reads it.

The two failure modes:

| Stale artifact | Symptom if not refreshed |
|---|---|
| `refund_policy.md` thresholds | Logic gates fire at the wrong dollar amount |
| `process_refund` parameter names | Tool call raises `InputValidationError` at runtime, or silently passes wrong fields |

## Fix: Resume, Then Selectively Invalidate

Keep the session (the analysis is valuable). Open with a refresh instruction
that names *exactly* what changed and forces a re-read before Claude proceeds.

```python
# WRONG — resume and continue, trusting prior context
async with ClaudeSDKClient(
    options=ClaudeAgentOptions(resume="refund-errors")
) as client:
    await client.query("Continue designing the fix.")

# CORRECT — resume, but pin the stale surfaces and force a refresh
async with ClaudeSDKClient(
    options=ClaudeAgentOptions(resume="refund-errors")
) as client:
    await client.query(
        "Before continuing, two things changed since we paused:\n"
        "1. `refund_policy.md` — refund threshold rules were updated. "
        "Re-read it and diff against your prior understanding.\n"
        "2. `process_refund` — parameter names changed. "
        "Re-inspect the tool schema before any call.\n\n"
        "List what changed, flag any prior conclusions that no longer hold, "
        "then continue the fix design."
    )
```

## Why This Works

- **Resume preserves the expensive part** — the investigation, the mental model
  of the bug, the ruled-out hypotheses. You don't pay to rebuild it.
- **Naming the stale surfaces is precise** — Claude refreshes only what
  actually changed, instead of re-reading the whole project or trusting all
  prior context.
- **Asking for a diff against prior understanding** — surfaces invalidated
  conclusions explicitly, instead of letting them quietly propagate into the
  fix.

## Alternative: When to Start Fresh Instead

Resume + refresh is the right call when *most* prior context is still valid.
Start a new session if:

- The schema/policy changes invalidate the **core hypothesis** (not just
  details).
- You can't enumerate what changed — broad uncertainty is cheaper to re-derive
  than to audit.
- The prior session is long and most of it is now irrelevant; the resumed
  tokens cost more than they save.

## Rule

> Resuming a session restores Claude's *memory* of external state, not the
> state itself. When resuming, explicitly name what changed on disk or in tool
> schemas and require a re-read before continuing — otherwise prior conclusions
> silently outvote current reality.
