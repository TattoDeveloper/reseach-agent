# Comparative Research: Decompose on the Comparison Axis, Not at Write Time

## Problem

A multi-agent research system (coordinator → web search, document analysis,
synthesis, report generation) gets a **comparative** request:

> Compare how proposed AI copyright rules affect **music licensing**, **model
> training data**, and **independent film production** across the **same
> jurisdictions and dates**.

Logs show the coordinator routes the entire request to **one document analysis
pass**, then asks the report agent to write separate sections. The result:

- **Music licensing** is covered deeply.
- **Model training data** is barely addressed.
- **Recommendations conflict across sectors.**

## Root Cause

The request is comparative and multi-dimensional — three sectors × shared
jurisdictions × shared dates — but the work was decomposed on the **wrong axis**:
a single flat analysis pass, split only at the *writing* stage. That ordering is
backwards and produces both symptoms:

| Symptom | Cause |
|---|---|
| Uneven depth (licensing deep, training data thin) | One undifferentiated pass can't allocate equal attention; it drifts to whatever dominates the sources. Sectioning *after* analysis can't recover coverage that was never produced. |
| Conflicting recommendations | The report agent writes siloed sections with no shared comparison frame — nothing aligns them on the same jurisdictions/dates, nothing reconciles their verdicts. |

The split happened at report-writing time. By then it's too late to fix either
coverage or coherence.

## Solution: Fan Out on the Comparison Axis + Shared Schema + Reconcile

Decompose the analysis along the axis the user is actually comparing — **one
scoped pass per sector** — against a **pinned shared schema**, then **reconcile
across the matrix** before the report agent writes.

```
request ──► coordinator builds shared schema (sectors × jurisdictions × dates)
                 │
        ┌────────┼────────┐          parallel, scoped, equal depth
   ┌────┴────┐ ┌─┴─────┐ ┌─┴─────┐
   │licensing│ │training│ │ film  │   each fills the SAME matrix cells
   └────┬────┘ └─┬─────┘ └─┬─────┘
        └────────┼────────┘
        cross-sector reconciliation (resolve conflicts on the shared axis)
                 │
            report generator (renders the reconciled matrix)
```

**1. Decompose by sector into parallel scoped passes.**
Music licensing, training data, and film each get their *own* analysis pass.
This structurally guarantees training data gets dedicated attention instead of
being crowded out — depth no longer depends on which sector the sources favor.

**2. Pin the shared dimensions so outputs are comparable.**
Every sector pass answers the **same question set** for the **same
jurisdictions** and the **same date ranges**, filling a structured matrix:

```
                    │ Jurisdiction A   │ Jurisdiction B   │ ...
                    │ (date range)     │ (date range)     │
────────────────────┼──────────────────┼──────────────────┤
Music licensing     │  effect + cite   │  effect + cite   │
Model training data │  effect + cite   │  effect + cite   │
Indie film          │  effect + cite   │  effect + cite   │
```

A shared schema is what turns three parallel passes into a genuine *comparison*
instead of three unrelated mini-reports.

**3. Reconcile across the matrix before report generation.**
A synthesis pass works *across* the matrix (down the columns, not just within a
sector): it surfaces where sector recommendations conflict for the same
jurisdiction and resolves or explicitly scopes them. The report agent then
*renders* a reconciled comparison rather than inventing three independent
verdicts.

## Implementation

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AgentDefinition

SECTORS = ["music_licensing", "model_training_data", "independent_film"]

options = ClaudeAgentOptions(
    agents={
        "sector_analyst": AgentDefinition(
            description="Analyzes one sector against the shared comparison schema.",
            prompt=(
                "You analyze EXACTLY ONE sector. For every jurisdiction and date "
                "range in the shared schema you are given, answer the same fixed "
                "question set and cite each claim. Do not skip cells — if a "
                "jurisdiction has no rule, say so explicitly. Stay in your sector."
            ),
            tools=["search_web", "analyze_document"],
        ),
    },
    system_prompt=(
        "You are the research coordinator for a COMPARATIVE request.\n"
        "1. SCHEMA: extract the comparison axes — sectors, the shared set of "
        "   jurisdictions, and the shared date ranges. Build a matrix spec.\n"
        "2. FAN OUT: launch one sector_analyst per sector IN PARALLEL, each "
        "   pinned to the SAME jurisdictions/dates/question set. Equal depth.\n"
        "3. RECONCILE: synthesize ACROSS the matrix (down each jurisdiction "
        "   column). Flag and resolve conflicting recommendations before "
        "   writing; note unavoidable trade-offs explicitly.\n"
        "4. Only then hand the reconciled matrix to the report generator.\n"
        "Never collapse a multi-sector comparison into one analysis pass, and "
        "never defer the split to the writing stage."
    ),
)

async with ClaudeSDKClient(options=options) as client:
    await client.query(
        "Compare how proposed AI copyright rules affect music licensing, model "
        "training data, and independent film across the same jurisdictions and "
        "dates. Produce a cited, reconciled report."
    )
    async for msg in client.receive_response():
        ...
```

## Why Other Approaches Fail

| Approach | Problem |
|---|---|
| Tell the single pass to "cover all three equally" | One pass still can't balance depth; no shared frame → conflicts remain |
| Split only at the report-writing stage (current) | Writing can't fix missing analysis; siloed sections still collide |
| Add an evaluator loop to catch the gaps | Helps, but treats the symptom — the real fix is decomposing on the comparison axis with a shared schema |
| **Per-sector parallel passes + shared matrix + reconcile** | Equal depth per sector, aligned dimensions, conflicts resolved before writing |

## SDK Components Involved

| Component | Role |
|---|---|
| `ClaudeAgentOptions.system_prompt` | Encodes schema → fan-out → reconcile on the coordinator |
| `AgentDefinition` (`sector_analyst`) | One reusable scoped analyst, instantiated per sector |
| `AgentDefinition.prompt` | Pins each pass to the shared question set / jurisdictions / dates |
| `AgentDefinition.tools` | Scopes each sector pass to search + document analysis |
| Coordinator reconciliation step | Resolves cross-sector conflicts before report generation |

## Rule

> Decompose along the axis the user is actually comparing — not at the
> report-writing stage. Give each branch a shared schema so the results line up,
> run them in parallel for equal depth, and reconcile across them before
> anything is written. Splitting at write time is too late to fix coverage or
> coherence.
