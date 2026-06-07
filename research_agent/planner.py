"""Plan / discovery — enumerate before you decompose (PLAN §2).

Two `knowledge/` invariants:

- **Discovery before decomposition** (`broad-codebase-question-coverage.md` per
  the PLAN): `discover` runs a read-only pass that *enumerates* the sub-questions
  a topic has — it does not answer them. This stops the decomposition from baking
  in the answer's shape so collectors can only confirm an assumption.
- **Shared schema for comparison** (`comparative-research-axis-decomposition.md`):
  `build_schema` produces the matrix contract (axis-values × dimensions × fixed
  question set), and `schema_to_subquestions` expands it into one scoped
  collector per cell — so the N parallel passes line up into a real comparison
  with equal depth and no skipped cells.

The plan is a structured object (`Plan`), not prose — it becomes the contract the
rest of the pipeline executes against.
"""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, query

from research_agent.collectors import QueryFn, SubQuestion
from research_agent.intake import IntakeResult
from research_agent.parsing import ParseError, extract_json_object, extract_text
from research_agent.types import Plan, RequestType, SchemaSpec


class PlannerError(ValueError):
    """Raised when discovery or schema construction cannot produce a usable plan."""


def default_discovery_options() -> ClaudeAgentOptions:
    """Read-only discovery: enumerate sub-questions, do NOT answer them."""
    return ClaudeAgentOptions(
        allowed_tools=[],
        system_prompt=(
            "You run a READ-ONLY discovery pass. Your ONLY job is to enumerate "
            "the distinct sub-questions / entry points a topic actually has — "
            "NOT to answer any of them. Coverage must track the real surface of "
            'the topic, not an assumption. Return JSON: {"sub_questions": [str]}.'
        ),
    )


async def discover(
    topic: str,
    *,
    options: ClaudeAgentOptions | None = None,
    query_fn: QueryFn = query,
) -> list[str]:
    """Enumerate the sub-questions a topic has, without answering them (§2a)."""
    opts = options if options is not None else default_discovery_options()
    messages: list[Any] = []
    async for message in query_fn(
        prompt=f"Enumerate the sub-questions for this topic:\n{topic}", options=opts
    ):
        messages.append(message)

    text = extract_text(messages)
    try:
        payload = extract_json_object(text)
    except ParseError as exc:
        raise PlannerError(f"discovery output not parseable: {exc}") from exc

    sub_questions = [str(q) for q in payload.get("sub_questions", []) if q]
    if not sub_questions:
        raise PlannerError("discovery enumerated no sub-questions")
    return sub_questions


def build_schema(
    axis: str,
    values: list[str],
    dimensions: dict[str, Any],
    question_set: list[str],
) -> SchemaSpec:
    """Build the shared comparison matrix; pin every collector to it (§2b)."""
    if not axis:
        raise PlannerError("schema requires a non-empty axis")
    if not values:
        raise PlannerError("schema requires at least one axis value")
    if not question_set:
        raise PlannerError("schema requires a non-empty question set")
    return SchemaSpec(
        axis=axis,
        values=values,
        dimensions=dimensions,
        question_set=question_set,
    )


def schema_to_subquestions(schema: SchemaSpec) -> list[SubQuestion]:
    """Expand the matrix into one scoped collector per axis value (§3).

    Every cell is covered: one SubQuestion per axis value, each pinned to the
    same question set and dimensions so depth can't drift between cells.
    """
    subquestions: list[SubQuestion] = []
    for value in schema["values"]:
        prompt = (
            f"Cover {schema['axis']} = {value}. Answer EVERY question in the "
            f"question set for EACH dimension; mark a cell 'no rule found' if it "
            f"has no answer.\nQuestion set: {json.dumps(schema['question_set'])}\n"
            f"Dimensions: {json.dumps(schema['dimensions'])}"
        )
        subquestions.append(
            SubQuestion(
                id=f"{schema['axis']}:{value}",
                prompt=prompt,
                subagent=f"{schema['axis']}:{value}",
                axis_value=value,
            )
        )
    return subquestions


def build_plan(
    intake: IntakeResult,
    sub_questions: list[str],
    *,
    question_set: list[str] | None = None,
) -> Plan:
    """Assemble the structured Plan; attach a schema for comparative requests."""
    plan: Plan = {
        "request_type": intake.request_type,
        "sub_questions": sub_questions,
    }
    if intake.request_type is RequestType.COMPARATIVE:
        if intake.axis is None:
            raise PlannerError("comparative plan requires an axis from intake")
        plan["schema"] = build_schema(
            intake.axis,
            intake.axis_values,
            intake.dimensions,
            question_set or sub_questions,
        )
    return plan
