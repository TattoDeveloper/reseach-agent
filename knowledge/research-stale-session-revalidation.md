# Multi-Agent Research: A Saved Session Is Frozen Evidence, Not Live Evidence

## Problem

A multi-agent research system (web-search, document-analysis, synthesis,
report-generation) has a **saved investigation session** holding earlier web
results, document analyses, and synthesis notes. Over the weekend:

- several source documents were **revised**,
- the search index was **refreshed**,
- and two previously collected subagent outputs now **cite superseded
  versions**.

The coordinator wants to continue toward the final report. The trap: the saved
session's evidence is no longer true, but it's sitting in the transcript looking
exactly as authoritative as it did Friday.

## Root Cause

A saved session captures evidence *as of the moment it was collected*. The
tool-result blocks in that transcript are a **point-in-time snapshot**, not a
live view of the sources — and nothing in them flags that the world moved on.

```
saved session (Friday)                 sources (Monday)
  web results        ───── stale? ─────  index refreshed
  doc analyses (×2)  ───── stale  ─────  documents revised
  synthesis notes    ── derived from ──  the two stale analyses ← also contaminated
```

The danger isn't just the two stale outputs — it's that **synthesis notes built
on them are contaminated too**, transitively. And if you simply `resume` the
saved session, those stale tool-results re-enter the model's context as trusted
evidence. The model cannot distinguish a stale `tool_result` from a fresh one;
they're identical blocks in the transcript. So "continue from the saved session"
silently launders outdated evidence into the final report.

The two wrong moves:

- **Resume the session as-is** → stale snapshots become authoritative context;
  contamination is guaranteed.
- **Manually edit the stale blocks out** → you can delete text, but the
  synthesis notes derived from them remain, and you're trusting a hand-scrub to
  be complete.

## Fix: Re-Validate Freshness, Re-Collect the Stale Branch, Rebuild Downstream

Treat the saved session as *reusable where current, invalid where superseded* —
and never let derived findings outlive their inputs.

```python
# 1. CHECK freshness per collected source against the system of record.
#    (Requires provenance on each finding — version / timestamp / hash.)
stale = [
    f for f in session.findings
    if source_of_record(f.source.doc_id).version != f.source.version
]

# 2. RE-COLLECT only the superseded branch against refreshed sources.
refreshed = []
for f in stale:
    refreshed.append(await run_subagent("document-analysis", f.source.doc_id))
# (re-run web-search too, since the index was refreshed)

# 3. INVALIDATE everything derived from stale inputs — synthesis must be redone,
#    not reused, because its notes encode superseded evidence.
fresh_findings = [f for f in session.findings if f not in stale] + refreshed

synthesis = await run_subagent("synthesis", inputs=fresh_findings)   # rebuilt

# 4. CONTINUE from a clean checkpoint containing only current evidence —
#    do not resume the contaminated transcript wholesale.
report = await run_subagent("report-generation", inputs=[synthesis])
```

## Why This Works

| Step | Effect |
|---|---|
| Freshness check via provenance | Detects staleness by data, not by hoping nothing changed |
| Re-collect only the stale branch | Preserves the still-current work; doesn't redo everything |
| Invalidate derived synthesis | Stops contamination from surviving in downstream notes |
| Clean checkpoint, not raw resume | Stale `tool_result` blocks never re-enter authoritative context |

## What This Depends On

This only works if each finding carries **provenance** — a source version,
timestamp, or hash you can diff against the system of record. Without it you
can't tell which outputs went stale, and the whole saved session becomes
suspect (forcing a full re-run). Capturing provenance per finding is what makes
*selective* re-validation possible. (See the companion failure: prose handoffs
that drop provenance entirely.)

## Why Not Just Re-Run Everything From Scratch?

You can, and it's safe — but it discards hours of still-valid web results and
analyses that *weren't* superseded. Selective re-validation is the reliable
*and* economical path: keep what provenance proves is current, re-collect what's
stale, and rebuild only what was derived from stale inputs.

## Rule

> A saved session is frozen evidence, not live evidence — resuming it re-injects
> point-in-time snapshots as if they were current. When sources change
> underneath a saved investigation, validate each finding's provenance against
> the system of record, re-collect the superseded branch, invalidate every
> derived finding (synthesis included), and continue from a clean checkpoint.
> Never resume a transcript whose tool-results you can no longer vouch for.
