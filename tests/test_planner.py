"""Tests for the planner: discovery + shared schema (T6.2).

Acceptance criteria from IMPLEMENTATION-PLAN.md:
- discovery returns enumerated sub-questions without answers;
- build_schema emits a SchemaSpec;
- collectors pinned to it cover every cell.
"""

from __future__ import annotations

import json

import pytest

from research_agent.intake import IntakeResult
from research_agent.planner import (
    PlannerError,
    build_plan,
    build_schema,
    default_discovery_options,
    discover,
    schema_to_subquestions,
)
from research_agent.types import RequestType
from tests.conftest import FakeMessage, scripted_query


@pytest.mark.anyio
async def test_discover_enumerates_sub_questions() -> None:
    msg = FakeMessage.text(json.dumps({"sub_questions": ["q1", "q2", "q3"]}))
    result = await discover("the landscape of X", query_fn=scripted_query([msg]))
    assert result == ["q1", "q2", "q3"]


@pytest.mark.anyio
async def test_discover_raises_on_empty() -> None:
    msg = FakeMessage.text(json.dumps({"sub_questions": []}))
    with pytest.raises(PlannerError, match="no sub-questions"):
        await discover("x", query_fn=scripted_query([msg]))


def test_discovery_options_are_read_only() -> None:
    options = default_discovery_options()
    assert options.allowed_tools == []
    assert "NOT to answer" in (options.system_prompt or "")


def test_build_schema_emits_spec() -> None:
    schema = build_schema(
        axis="sector",
        values=["music", "training"],
        dimensions={"jurisdictions": ["US", "EU"]},
        question_set=["what rule applies", "effective date"],
    )
    assert schema["axis"] == "sector"
    assert schema["values"] == ["music", "training"]
    assert schema["question_set"] == ["what rule applies", "effective date"]


@pytest.mark.parametrize(
    ("axis", "values", "question_set"),
    [
        ("", ["m"], ["q"]),  # empty axis
        ("sector", [], ["q"]),  # no values
        ("sector", ["m"], []),  # no question set
    ],
)
def test_build_schema_validates_inputs(
    axis: str, values: list[str], question_set: list[str]
) -> None:
    with pytest.raises(PlannerError):
        build_schema(axis, values, {}, question_set)


def test_schema_to_subquestions_covers_every_cell() -> None:
    schema = build_schema(
        axis="sector",
        values=["music", "training", "film"],
        dimensions={"jurisdictions": ["US"]},
        question_set=["what rule applies"],
    )
    subqs = schema_to_subquestions(schema)

    assert len(subqs) == 3  # one collector per axis value
    assert {s.axis_value for s in subqs} == {"music", "training", "film"}
    # each cell is pinned to the same question set
    assert all("what rule applies" in s.prompt for s in subqs)


def test_build_plan_comparative_attaches_schema() -> None:
    intake = IntakeResult(
        request_type=RequestType.COMPARATIVE,
        axis="sector",
        axis_values=["music", "training"],
        dimensions={"jurisdictions": ["US"]},
    )
    plan = build_plan(intake, ["what rule applies", "effective date"])

    assert plan["request_type"] is RequestType.COMPARATIVE
    assert "schema" in plan
    assert plan["schema"]["values"] == ["music", "training"]


def test_build_plan_simple_has_no_schema() -> None:
    intake = IntakeResult(request_type=RequestType.SIMPLE)
    plan = build_plan(intake, ["one question"])

    assert plan["request_type"] is RequestType.SIMPLE
    assert "schema" not in plan
