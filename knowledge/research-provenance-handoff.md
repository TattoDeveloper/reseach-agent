# Multi-Agent Research: Claims Survive the Handoff, Provenance Doesn't

## Problem

A coordinator delegates to four subagents — web-search, document-analysis,
synthesis, report-generation — to produce cited reports. On fast-changing
market topics, reviewers find the final reports make *accurate individual
claims* but misstate the **evidence's standing**: whether it's current,
peer-reviewed, or based on a small regional sample.

When reviewers ask the report generator to add citations, it can sometimes name
the correct document — but **cannot identify the page, publication date, or
study basis**. That metadata isn't recoverable because it was never in the
handoff.

```
source ──▶ search/doc subagent ──▶ "Market X grew 40% in 2024."
 │                                   ✗ page    ✗ pub date
 ├─ page 12                          ✗ peer-reviewed?  ✗ sample basis
 ├─ published 2021 (stale!)
 ├─ n=40, one region
 └─ vendor white paper (not peer-reviewed)
```

## Root Cause

This is *not* the synthesis-ordering bug (where report-generation runs before
synthesis finishes). Here synthesis runs fine — but it's reasoning over
**lossy prose handoffs**. The upstream subagents flatten each source into a
sentence and drop everything that qualifies the claim:

- page / location in document
- publication or last-updated date (decisive on fast-changing topics)
- peer-reviewed vs. vendor/blog
- sample size and regional scope

A claim's *truth* and its *standing* are different facts. Prose summaries
preserve the first and discard the second. Synthesis can only assess currency,
rigor, and representativeness if those attributes ride along with the claim —
and once they're gone at the search/document stage, no downstream agent can
reconstruct them. The report generator then either omits the caveats or
hallucinates them.

## Fix: Carry a Structured Claim+Provenance Record Through Every Handoff

Make the upstream subagents emit, per claim, a structured record that binds the
assertion to its evidentiary metadata — and forbid downstream agents from
passing prose that strips it.

```python
# WRONG — handoff is a sentence; provenance is gone
"Market X grew 40% in 2024."

# CORRECT — claim bound to its evidentiary standing, carried end to end
{
    "claim": "Market X grew 40% year-over-year",
    "source": {
        "doc_id": "rpt-2024-114",
        "title": "APAC SaaS Outlook",
        "page": 12,
        "published": "2021-03-01",      # synthesis can now flag as stale
        "peer_reviewed": False,         # vendor white paper
        "sample": {"n": 40, "scope": "single region: SE Asia"},
    },
    "quote": "...grew 40% YoY across surveyed firms...",
}
```

Synthesis then operates on standing, not just text:

```python
for c in claims:
    if stale(c.source.published, topic_half_life):
        c.flags.append("possibly-outdated")
    if not c.source.peer_reviewed:
        c.flags.append("non-peer-reviewed")
    if c.source.sample and c.source.sample["n"] < MIN_N:
        c.flags.append("small-sample")
```

Report-generation receives the flags and the bound metadata, so a citation
*includes* page and date, and qualifiers ("based on a 2021 single-region vendor
survey, n=40") are grounded — not invented.

## Why This Works

| Change | Effect |
|---|---|
| Provenance captured at the source stage | The only stage that *has* page/date/sample — capture it before it's lost |
| Metadata bound to each claim, not the batch | Synthesis judges each claim's standing individually |
| Structured record, not prose | Downstream agents can't silently drop qualifiers a sentence would omit |
| Flags computed in synthesis | "Current / peer-reviewed / representative?" becomes a check, not a vibe |
| Page+date travel to the report | Citations are complete; no back-fill, no hallucinated caveats |

## Why Not Just Tell the Report Generator to Add Citations?

The report generator is the *last* stage — it can only cite what reached it. By
then page, date, and sample basis are gone. Asking it to add them forces a
guess. Provenance has to be captured where it exists (the search/document
stage) and preserved through every handoff; you cannot reconstruct at the end
what was discarded at the start.

## Rule

> A claim's truth and its evidentiary standing (currency, rigor,
> representativeness) are separate facts, and prose handoffs preserve the first
> while silently dropping the second. Capture provenance — page, date,
> peer-review status, sample — at the source stage and carry it as structured
> data bound to each claim through every handoff. Downstream agents can only
> assess and cite what survives the handoff.
