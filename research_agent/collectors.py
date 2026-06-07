"""Parallel scoped collectors → Finding[] (PLAN §3, T2.2).

Each collector is a **stateless ``query()`` worker** (PLAN §0.5): one isolated
subprocess per sub-question/axis-value, fanned out behind a semaphore so a broad
request can't exhaust file descriptors or rate limits.

Three `knowledge/` invariants are enforced here in code, not by trusting the
model:

- **Structured, provenance-bound output** (`research-provenance-handoff.md`,
  `subagent-handoff-attribution-fix.md`): workers must emit JSON findings; each
  is run through `validate_finding`, and the producing `subagent` label is
  stamped in code so attribution can't be lost even if the model omits it.
- **No skipped cells** (`comparative-research-axis-decomposition.md` §3c): a
  collector must either return findings or *explicitly* mark `no_rule_found`. A
  cell that is neither filled nor marked raises `CollectionError` — silence is
  never accepted as an answer.

Note on the SDK: the `knowledge/` notes use an illustrative `run_subagent(...,
output_schema=...)` helper that does not exist. The real path is a plain
`query()` call plus JSON parsing/validation at this boundary.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import anyio
from claude_agent_sdk import ClaudeAgentOptions, query

from research_agent.types import Finding, SchemaSpec, validate_finding

# A `query`-compatible callable. Defaulting to the real SDK `query` while
# allowing tests to inject a scripted async generator (see tests/conftest.py).
QueryFn = Callable[..., AsyncIterator[Any]]

DEFAULT_MAX_CONCURRENCY = 8
COLLECTOR_TOOLS = ["WebSearch", "WebFetch", "Read"]


class CollectionError(RuntimeError):
    """Raised when a collector cell is neither filled nor marked no_rule_found."""


@dataclass(frozen=True)
class SubQuestion:
    """One scoped unit of collection work (a sub-question or a matrix cell)."""

    id: str
    prompt: str  # the question this collector must answer
    subagent: str  # producer label, stamped onto every finding (provenance)
    axis_value: str | None = None  # which comparison cell, if any (§2b)


@dataclass
class CellResult:
    """The outcome of one collector — every cell is accounted for (§3c)."""

    subquestion_id: str
    findings: list[Finding] = field(default_factory=list)
    no_rule_found: bool = False


def default_collector_options() -> ClaudeAgentOptions:
    """Stateless options for a collector worker.

    No `resume`/`session_id`: collectors are isolated subprocesses, so the
    orchestrator (not a transcript) owns sequencing and gating.
    """
    return ClaudeAgentOptions(
        allowed_tools=COLLECTOR_TOOLS,
        system_prompt=(
            "You are a scoped research collector. Investigate ONLY the question "
            "you are given. Return a JSON object and nothing else.\n"
            'If you find supporting evidence: {"findings": [{"claim": str, '
            '"source": {"doc_id": str, "title": str, "location": "url|file:line|page", '
            '"published": "ISO-date", "version": str|null, "peer_reviewed": bool, '
            '"sample": {"n": int, "scope": str}|null}, "quote": "exact excerpt"}]}.\n'
            'If no rule/evidence applies, return exactly {"no_rule_found": true}. '
            "Never invent a source; every claim needs a real location and date."
        ),
    )


def build_collector_prompt(subq: SubQuestion, schema: SchemaSpec | None = None) -> str:
    """Wrap the sub-question with the shared schema so cells stay comparable."""
    parts = [subq.prompt]
    if schema is not None:
        parts.append(
            "\nPin your answer to this shared comparison schema "
            "(answer the same question set for the same dimensions):\n"
            + json.dumps(schema)
        )
    if subq.axis_value is not None:
        parts.append(f"\nYou cover exactly this axis value: {subq.axis_value}.")
    return "".join(parts)


async def collect_one(
    subq: SubQuestion,
    *,
    schema: SchemaSpec | None = None,
    options: ClaudeAgentOptions | None = None,
    query_fn: QueryFn = query,
) -> CellResult:
    """Run one stateless collector and parse its structured result."""
    prompt = build_collector_prompt(subq, schema)
    opts = options if options is not None else default_collector_options()
    messages: list[Any] = []
    async for message in query_fn(prompt=prompt, options=opts):
        messages.append(message)
    return _parse_cell(subq, messages)


async def collect_all(
    subqs: Sequence[SubQuestion],
    *,
    schema: SchemaSpec | None = None,
    options: ClaudeAgentOptions | None = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    query_fn: QueryFn = query,
) -> list[CellResult]:
    """Fan out collectors in parallel, bounded by a semaphore (PLAN §0.5).

    Results preserve input order, and every sub-question yields exactly one
    `CellResult` — no cell is silently dropped.
    """
    limiter = anyio.Semaphore(max_concurrency)
    results: list[CellResult | None] = [None] * len(subqs)

    async def _run(index: int, sub: SubQuestion) -> None:
        async with limiter:
            results[index] = await collect_one(
                sub, schema=schema, options=options, query_fn=query_fn
            )

    async with anyio.create_task_group() as task_group:
        for index, sub in enumerate(subqs):
            task_group.start_soon(_run, index, sub)

    return [result for result in results if result is not None]


# --- parsing ----------------------------------------------------------------


def _parse_cell(subq: SubQuestion, messages: list[Any]) -> CellResult:
    text = _extract_text(messages)
    if not text.strip():
        raise CollectionError(f"collector {subq.id!r} returned no content")

    payload = _extract_json_object(text)
    if payload.get("no_rule_found") is True:
        return CellResult(subq.id, [], no_rule_found=True)

    raw_findings = payload.get("findings")
    if not isinstance(raw_findings, list) or not raw_findings:
        raise CollectionError(
            f"cell {subq.id!r} neither filled nor marked 'no_rule_found' (§3c)"
        )

    findings: list[Finding] = []
    for raw in raw_findings:
        if isinstance(raw, dict):
            # Stamp provenance in code so attribution survives a forgetful model.
            raw.setdefault("subagent", subq.subagent)
        findings.append(validate_finding(raw))
    return CellResult(subq.id, findings, no_rule_found=False)


def _extract_text(messages: list[Any]) -> str:
    """Concatenate every text block across messages (real or mocked)."""
    parts: list[str] = []
    for message in messages:
        content = getattr(message, "content", None)
        if not content:
            continue
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse the JSON object a worker returned, tolerating surrounding prose."""
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            raise CollectionError("collector did not return a JSON object") from None
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise CollectionError("collector JSON was not an object")
    return parsed
