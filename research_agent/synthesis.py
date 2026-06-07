"""Synthesis: Findings → cited Claims + standing flags, and the gate (PLAN §4).

Two `knowledge/` invariants are enforced here:

- **Standing is a check, not a vibe** (`research-provenance-handoff.md`): the LLM
  step only *groups findings into claims*; the standing flags
  (`possibly-outdated` / `non-peer-reviewed` / `small-sample`) are computed in
  **code** from each cited source's provenance. The model never decides them.
- **Synthesis is a hard prerequisite** (`multi-agent-synthesis-dependency-fix.md`):
  `assert_synthesis_complete` is the machine-checked gate. Report generation
  cannot start unless synthesis returned ``ok`` AND every claim carries
  `source_ids`. The dependency lives in code, not in a coordinator prompt.

The worker operates on the structured findings only (no tools, no web access), so
it cannot introduce a source that wasn't collected — it can only cite what it was
given. Flags are then computed against those same findings.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal

from claude_agent_sdk import ClaudeAgentOptions, query

from research_agent.collectors import QueryFn
from research_agent.parsing import ParseError, extract_json_object, extract_text
from research_agent.types import Claim, Finding, Flag, Source

# Tunable thresholds for standing flags (PLAN §4a). A topic with a shorter
# half-life should pass a smaller `half_life_days`.
DEFAULT_HALF_LIFE_DAYS = 365
DEFAULT_MIN_N = 30

SynthesisStatus = Literal["ok", "error"]


@dataclass
class SynthesisResult:
    """Output of the synthesis stage — consumed by the gate before report-gen."""

    status: SynthesisStatus
    claims: list[Claim] = field(default_factory=list)
    error: str | None = None


class SynthesisIncompleteError(RuntimeError):
    """Raised by the gate when report generation must be blocked (PLAN §4c)."""


def default_synthesis_options() -> ClaudeAgentOptions:
    """Options for the synthesis worker: NO tools — findings in, claims out.

    Denying tools is what guarantees the worker cannot fetch a new source and
    must cite only the findings it was handed.
    """
    return ClaudeAgentOptions(
        allowed_tools=[],
        system_prompt=(
            "You are a research synthesizer. You are given a JSON list of "
            "structured findings, each with a `source.doc_id`. Produce claims "
            "SOLELY from these findings — never add outside knowledge.\n"
            "Reconcile across the shared dimensions: when findings for the same "
            "dimension disagree, surface and resolve the conflict in the claim "
            "text rather than emitting contradictory claims.\n"
            'Return exactly: {"claims": [{"text": str, "source_ids": '
            '[doc_id, ...]}]}. Every claim MUST cite at least one doc_id drawn '
            "from the findings. Do not compute caveats; just cite accurately."
        ),
    )


def build_synthesis_prompt(findings: Sequence[Finding]) -> str:
    return (
        "Synthesize claims from these findings. Cite source_ids by doc_id.\n"
        + json.dumps(list(findings))
    )


async def synthesize(
    findings: Sequence[Finding],
    *,
    as_of: date | None = None,
    half_life_days: int = DEFAULT_HALF_LIFE_DAYS,
    min_n: int = DEFAULT_MIN_N,
    options: ClaudeAgentOptions | None = None,
    query_fn: QueryFn = query,
) -> SynthesisResult:
    """Turn findings into cited claims, computing standing flags in code."""
    if not findings:
        return SynthesisResult(status="error", error="no findings to synthesize")

    prompt = build_synthesis_prompt(findings)
    opts = options if options is not None else default_synthesis_options()
    messages: list[Any] = []
    async for message in query_fn(prompt=prompt, options=opts):
        messages.append(message)

    text = extract_text(messages)
    if not text.strip():
        return SynthesisResult(status="error", error="synthesis worker returned no content")
    try:
        payload = extract_json_object(text)
    except ParseError as exc:
        return SynthesisResult(status="error", error=str(exc))

    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        return SynthesisResult(status="error", error="synthesis output had no 'claims' list")

    sources_by_doc = {f["source"]["doc_id"]: f["source"] for f in findings}
    as_of = as_of or datetime.now(UTC).date()

    claims: list[Claim] = []
    for raw in raw_claims:
        if not isinstance(raw, dict) or "text" not in raw:
            return SynthesisResult(status="error", error="malformed claim in synthesis output")
        source_ids = [str(s) for s in raw.get("source_ids", []) if s]
        claims.append(
            Claim(
                text=str(raw["text"]),
                source_ids=source_ids,
                flags=compute_standing_flags(
                    source_ids,
                    sources_by_doc,
                    as_of=as_of,
                    half_life_days=half_life_days,
                    min_n=min_n,
                ),
            )
        )
    return SynthesisResult(status="ok", claims=claims)


def compute_standing_flags(
    source_ids: Sequence[str],
    sources_by_doc: dict[str, Source],
    *,
    as_of: date,
    half_life_days: int,
    min_n: int,
) -> list[Flag]:
    """Derive standing caveats from the provenance of a claim's cited sources.

    Rule: a claim is flagged when **no cited source clears the bar** — a single
    strong source (peer-reviewed / current / robust sample) lifts the caveat.
    Computed entirely in code so currency/rigor/representativeness is a check,
    never the model's opinion.
    """
    sources = [sources_by_doc[sid] for sid in source_ids if sid in sources_by_doc]
    if not sources:
        return []

    flags: list[Flag] = []
    if all(_is_stale(s["published"], as_of, half_life_days) for s in sources):
        flags.append("possibly-outdated")
    if not any(s["peer_reviewed"] for s in sources):
        flags.append("non-peer-reviewed")
    if not any(_clears_sample_bar(s, min_n) for s in sources):
        flags.append("small-sample")
    return flags


def assert_synthesis_complete(result: SynthesisResult) -> SynthesisResult:
    """Gate report generation on a verified synthesis result (PLAN §4c).

    Raises:
        SynthesisIncompleteError: if synthesis errored, produced no claims, or
            any claim lacks `source_ids`. This is the machine-checked dependency
            that stops report-gen from running on incomplete/uncited material.
    """
    if result.status != "ok":
        raise SynthesisIncompleteError(
            f"report blocked: synthesis status={result.status} ({result.error})"
        )
    if not result.claims:
        raise SynthesisIncompleteError("report blocked: synthesis produced no claims")
    uncited = [i for i, c in enumerate(result.claims) if not c["source_ids"]]
    if uncited:
        raise SynthesisIncompleteError(
            f"report blocked: claims at indices {uncited} have no source_ids"
        )
    return result


# --- internals --------------------------------------------------------------


def _is_stale(published: str, as_of: date, half_life_days: int) -> bool:
    parsed = _parse_date(published)
    if parsed is None:
        return True  # unparseable date → treat as stale rather than vouch for it
    return (as_of - parsed).days > half_life_days


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        pass
    # tolerate year-only ("2024") or year-month ("2024-03")
    parts = value.strip().split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return date(year, month, day)
    except (ValueError, IndexError):
        return None


def _clears_sample_bar(source: Source, min_n: int) -> bool:
    sample = source["sample"]
    return sample is None or sample["n"] >= min_n
