# Multi-Agent Research: Reports Generated Before Synthesis Completes

## Problem

A coordinator agent delegates to four specialized subagents: web-search,
document-analysis, synthesis, and report-generation. In staging, final reports
occasionally contain polished prose with missing or unverifiable citations.

Traces show the coordinator invoking the report-generation subagent after
web-search and document-analysis return — while synthesis is still pending or
has returned an error. The report subagent then fabricates connective tissue
(including citations) to fill the gap.

## Root Cause

The coordinator treats subagent invocations as a loose plan rather than a
dependency graph. Report-generation has an implicit prerequisite — verified,
cross-referenced findings from synthesis — but nothing enforces it:

```
web-search ─┐
            ├─▶ synthesis ─▶ report-generation
doc-analysis ┘
```

When synthesis stalls or errors, the coordinator sees "enough" upstream output
and proceeds. The report subagent receives raw search/analysis blobs without
the citation-bound claims synthesis was supposed to produce, and the model
fills the void by hallucinating.

## Fix: Enforce Synthesis as a Hard Prerequisite

Make the dependency explicit and machine-checked rather than relying on the
coordinator LLM to sequence correctly.

```python
# WRONG — coordinator decides order from prose instructions
coordinator_prompt = """
Use the web-search, document-analysis, synthesis, and report-generation
subagents to produce a cited report on {topic}.
"""

# CORRECT — gate report-generation on a verified synthesis result
search_result = await run_subagent("web-search", topic)
docs_result = await run_subagent("document-analysis", topic)

synthesis = await run_subagent(
    "synthesis",
    inputs=[search_result, docs_result],
)

if synthesis.status != "ok" or not synthesis.citations:
    raise SynthesisIncompleteError(
        "report-generation blocked: synthesis did not return verified citations"
    )

report = await run_subagent(
    "report-generation",
    inputs=[synthesis],   # only synthesis output — never raw search blobs
)
```

## Why This Works

| Change | Effect |
|---|---|
| Programmatic sequencing | Removes the LLM's ability to skip synthesis under time/output pressure |
| Status + citations check | Distinguishes "synthesis ran" from "synthesis succeeded with grounded claims" |
| Report subagent receives only synthesis output | No raw blobs to "summarize," so it cannot invent citations from unverified material |
| Explicit error on failure | Surfaces the broken run instead of producing a confidently-wrong report |

## Alternative: Keep LLM Coordination, Add a Contract

If the coordinator must remain LLM-driven, require synthesis to emit a
structured handoff token that report-generation validates before running:

```json
{
  "synthesis_complete": true,
  "claims": [
    {"text": "...", "source_ids": ["s1", "s3"]}
  ]
}
```

Report-generation refuses to run if `synthesis_complete` is false or any claim
lacks `source_ids`. The contract — not the coordinator — enforces the
dependency.

## Rule

> When a subagent's output is a hard prerequisite for another, enforce the
> dependency in code or via a validated contract — not in the coordinator's
> prompt. LLM coordinators will skip prerequisites under pressure, and
> downstream agents will fabricate to fill the gap.
