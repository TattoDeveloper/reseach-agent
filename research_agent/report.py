"""Report generation — renders verified Claims only (PLAN §6).

From `multi-agent-synthesis-dependency-fix.md` + `research-provenance-handoff.md`:

- Report-gen receives **only verified Claims** — never raw `Finding` blobs — so
  it has nothing to "summarize" into invented citations. `render` rejects a
  `Finding` passed where a `Claim` is expected, failing loud.
- Every citation includes **page (location) + date (published)**, because that
  metadata rode along the whole pipeline on `Source`. A cited source whose
  metadata is missing is an integrity error, not a silent gap.
- Standing flags become **explicit caveats** grounded in provenance
  ("not peer-reviewed; small sample n=40, SE Asia") rather than being dropped or
  hallucinated.

This stage is pure rendering — no LLM call. Everything it prints is already
verified and provenance-bound; its only job is to format, not to reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from research_agent.types import Claim, Finding, Flag, Source


class ReportInputError(TypeError):
    """Raised when render receives something other than verified Claims."""


def sources_from_findings(findings: Sequence[Finding]) -> dict[str, Source]:
    """Build the doc_id -> Source index render needs, from collected findings."""
    return {f["source"]["doc_id"]: f["source"] for f in findings}


def render(
    claims: Sequence[Claim],
    sources_by_doc: Mapping[str, Source],
    *,
    title: str = "Research Report",
) -> str:
    """Render verified claims to markdown with page+date citations and caveats.

    Raises:
        ReportInputError: if an item is not a Claim (e.g. a raw Finding), or a
            claim cites a source_id with no metadata in ``sources_by_doc``.
    """
    lines = [f"# {title}", ""]
    cited_order: list[str] = []

    for index, claim in enumerate(claims, start=1):
        _assert_is_claim(claim)
        for doc_id in claim["source_ids"]:
            if doc_id not in sources_by_doc:
                raise ReportInputError(
                    f"claim {index} cites unknown source_id {doc_id!r} "
                    "(no page/date metadata available)"
                )
            if doc_id not in cited_order:
                cited_order.append(doc_id)

        markers = ", ".join(claim["source_ids"])
        lines.append(f"{index}. {claim['text']} [{markers}]")
        caveat = _caveat(claim["flags"], claim["source_ids"], sources_by_doc)
        if caveat:
            lines.append(f"   _Caveat: {caveat}._")

    lines.extend(["", "## Sources", ""])
    for doc_id in cited_order:
        source = sources_by_doc[doc_id]
        # page (location) + date (published) — required on every citation.
        lines.append(
            f"- [{doc_id}] {source['title']} — {source['location']}, {source['published']}"
        )

    return "\n".join(lines) + "\n"


# --- internals --------------------------------------------------------------


def _assert_is_claim(obj: object) -> None:
    if not isinstance(obj, Mapping):
        raise ReportInputError(f"expected a Claim mapping, got {type(obj).__name__}")
    if "quote" in obj or "subagent" in obj:
        raise ReportInputError(
            "received a raw Finding where a verified Claim was expected"
        )
    if "text" not in obj or "source_ids" not in obj:
        raise ReportInputError(
            "expected a Claim with 'text' and 'source_ids'; "
            f"got keys {sorted(obj.keys())}"
        )


def _caveat(
    flags: Sequence[Flag],
    source_ids: Sequence[str],
    sources_by_doc: Mapping[str, Source],
) -> str | None:
    """Turn standing flags into a human caveat grounded in source provenance."""
    if not flags:
        return None

    sources = [sources_by_doc[sid] for sid in source_ids if sid in sources_by_doc]
    parts: list[str] = []

    if "possibly-outdated" in flags:
        dates = sorted(s["published"] for s in sources)
        oldest = dates[0] if dates else "unknown date"
        parts.append(f"may be outdated (published {oldest})")

    if "non-peer-reviewed" in flags:
        parts.append("not peer-reviewed")

    if "small-sample" in flags:
        samples = [s["sample"] for s in sources if s["sample"] is not None]
        if samples:
            smallest = min(samples, key=lambda sample: sample["n"])
            parts.append(f"small sample (n={smallest['n']}, {smallest['scope']})")
        else:
            parts.append("small sample")

    return "; ".join(parts)
