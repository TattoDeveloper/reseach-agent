# Subagent Handoffs: Losing Attribution Through Free-Form Summaries

## Problem

A coordinator delegates codebase exploration to several subagents, then asks
an implementation subagent to generate migration scaffolding. In review,
engineers find the final proposal:

- Mixes findings from different packages as if they came from one place.
- Cites helper functions with no file path or line number.
- Cannot point to which search result or source file supports a recommended
  change.

The individual exploration subagents did find correct facts. The information
was lost on the way out.

## Root Cause

Each subagent returns a **prose summary** of what it found. Prose is lossy:

```
"In the auth package, the token validator uses a 30-minute TTL and
the legacy session helper appears to bypass the new middleware."
```

Useful — but no `file:line`, no distinction between the two findings'
provenance, and no way for the implementation subagent to verify or
re-inspect. By the time the implementation subagent stitches multiple prose
blobs together, attribution is gone and there is nothing to ground
recommendations in. The model fills the gap with plausible-sounding
references it cannot actually cite.

## Fix: Require Structured, Citation-Bound Handoffs

Make every exploration subagent return a typed list of findings where each
finding carries its own evidence. The implementation subagent then operates
*only* on this structured input — and is forbidden from making claims that
don't trace back to one of these entries.

```python
# WRONG — free-form prose handoff
exploration_result = await run_subagent("explore-auth", task)
# returns: "Found that the token validator uses a 30-min TTL..."

# CORRECT — structured findings with required attribution
class Finding(TypedDict):
    claim: str            # what was observed
    file: str             # absolute path
    line: int             # exact line
    snippet: str          # the relevant code excerpt
    subagent: str         # which subagent produced this

exploration_result: list[Finding] = await run_subagent(
    "explore-auth",
    task,
    output_schema=Finding,
)

# Implementation subagent receives the structured list,
# not prose summaries
implementation = await run_subagent(
    "implement-migration",
    inputs={"findings": exploration_result},
    system="Every recommendation must reference a finding by index. "
           "Do not introduce claims that lack a corresponding finding. "
           "If evidence is missing, request another exploration pass.",
)
```

## Why This Preserves Reliability and Attribution

| Change | Effect |
|---|---|
| Typed `Finding` schema | Each fact carries `file`, `line`, `snippet` — provenance survives the handoff |
| `subagent` field | Findings from different packages stay distinguishable instead of blurring into one narrative |
| Implementation subagent constrained to cited findings | Removes the surface area for plausible-sounding fabrication |
| "Request another exploration pass" escape hatch | Missing evidence becomes a re-query, not a hallucination |

## Coordinator-Level Discipline

Equally important: the coordinator must pass the **structured list itself**
downstream — not its own prose summary of it. Re-summarizing at the
coordinator layer reintroduces the same loss the schema was meant to
prevent.

```python
# WRONG — coordinator collapses findings before handoff
summary = summarize(all_findings)
await run_subagent("implement-migration", inputs={"context": summary})

# CORRECT — pass the structured findings through untouched
await run_subagent("implement-migration", inputs={"findings": all_findings})
```

## Rule

> Subagent handoffs should be structured records with embedded provenance
> (`file`, `line`, `snippet`), not prose. Downstream agents must be
> constrained to cite only what the structured input contains — and the
> coordinator must pass that structure through, not re-summarize it.
