"""Intake & classification (PLAN §1).

Classify the request *before* doing anything, because the decomposition strategy
depends on it:

- **simple/factual** → linear collect → synthesize → verify → report.
- **comparative / multi-dimensional** → axis decomposition (§2b).
- **open-ended exploratory** → discovery pass first (§2a).

Comparison axes (sectors × jurisdictions × dates) are detected here so the
planner can pin a shared schema. Classification is an LLM call with no tools
(it reasons over the request text only); the structured contract + validation
live in code so a malformed classification fails loud rather than silently
mis-routing the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, query

from research_agent.collectors import QueryFn
from research_agent.parsing import ParseError, extract_json_object, extract_text
from research_agent.types import RequestType


class IntakeError(ValueError):
    """Raised when a request cannot be classified into a usable IntakeResult."""


@dataclass
class IntakeResult:
    """Structured classification of a request — drives the rest of the pipeline."""

    request_type: RequestType
    axis: str | None = None  # e.g. "sector" (comparative only)
    axis_values: list[str] = field(default_factory=list)
    dimensions: dict[str, Any] = field(default_factory=dict)  # jurisdictions, dates


def default_intake_options() -> ClaudeAgentOptions:
    """Options for the classifier: no tools — reason over the request only."""
    return ClaudeAgentOptions(
        allowed_tools=[],
        system_prompt=(
            "You classify a research request before work begins. Return JSON:\n"
            '{"request_type": "simple|comparative|exploratory", '
            '"axis": str|null, "axis_values": [str], '
            '"dimensions": {"jurisdictions": [str], "date_range": str}}.\n'
            "- 'comparative' when the user compares things across shared "
            "dimensions; set `axis` (what is compared) and `axis_values` (the "
            "things being compared), plus shared dimensions.\n"
            "- 'exploratory' for open-ended 'how does X work / landscape of X'.\n"
            "- 'simple' for a single factual question. Leave axis null otherwise."
        ),
    )


def build_intake_prompt(request: str) -> str:
    return f"Classify this research request:\n{request}"


async def classify(
    request: str,
    *,
    options: ClaudeAgentOptions | None = None,
    query_fn: QueryFn = query,
) -> IntakeResult:
    """Classify a request and detect comparison axes (PLAN §1)."""
    opts = options if options is not None else default_intake_options()
    messages: list[Any] = []
    async for message in query_fn(prompt=build_intake_prompt(request), options=opts):
        messages.append(message)

    text = extract_text(messages)
    if not text.strip():
        raise IntakeError("classifier returned no content")
    try:
        payload = extract_json_object(text)
    except ParseError as exc:
        raise IntakeError(f"classifier output not parseable: {exc}") from exc

    raw_type = str(payload.get("request_type", "")).strip().lower()
    try:
        request_type = RequestType(raw_type)
    except ValueError as exc:
        raise IntakeError(f"unknown request_type {raw_type!r}") from exc

    axis = payload.get("axis")
    axis = str(axis) if axis else None
    axis_values = [str(v) for v in payload.get("axis_values", []) if v]
    dimensions = dict(payload.get("dimensions", {}))

    if request_type is RequestType.COMPARATIVE and (axis is None or not axis_values):
        raise IntakeError(
            "comparative request must have a detected axis and axis_values"
        )

    return IntakeResult(
        request_type=request_type,
        axis=axis,
        axis_values=axis_values,
        dimensions=dimensions,
    )
