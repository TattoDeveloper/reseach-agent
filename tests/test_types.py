"""Tests for the structured records (T1.1).

Acceptance criteria from IMPLEMENTATION-PLAN.md:
- construct one of each record;
- `validate_finding` rejects a Finding missing `source.location` or `published`.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from research_agent.types import (
    Claim,
    Finding,
    InvalidFindingError,
    Plan,
    RequestType,
    SchemaSpec,
    Source,
    Verdict,
    validate_finding,
)


def _good_source() -> Source:
    return Source(
        doc_id="rpt-2024-114",
        title="APAC SaaS Outlook",
        location="page 12",
        published="2021-03-01",
        version="v3",
        peer_reviewed=False,
        sample={"n": 40, "scope": "single region: SE Asia"},
    )


def _good_finding() -> Finding:
    return Finding(
        claim="Market X grew 40% year-over-year",
        source=_good_source(),
        quote="...grew 40% YoY across surveyed firms...",
        subagent="sector_analyst:music_licensing",
    )


# --- construction -----------------------------------------------------------


def test_construct_each_record() -> None:
    source = _good_source()
    finding = _good_finding()
    claim = Claim(text="Market X grew 40% YoY", source_ids=["rpt-2024-114"], flags=[])
    schema = SchemaSpec(
        axis="sector",
        values=["music_licensing", "model_training_data"],
        dimensions={"jurisdictions": ["US", "EU"], "date_range": "2023-2025"},
        question_set=["what rule applies", "effective date"],
    )
    plan = Plan(
        request_type=RequestType.COMPARATIVE,
        sub_questions=["q1", "q2"],
        schema=schema,
    )

    assert source["sample"] == {"n": 40, "scope": "single region: SE Asia"}
    assert finding["source"]["doc_id"] == "rpt-2024-114"
    assert claim["source_ids"] == ["rpt-2024-114"]
    assert schema["question_set"][0] == "what rule applies"
    assert plan["request_type"] is RequestType.COMPARATIVE
    assert "schema" in plan


def test_plan_schema_is_optional_for_simple_requests() -> None:
    plan = Plan(request_type=RequestType.SIMPLE, sub_questions=["one fact"])
    assert "schema" not in plan


def test_enums_have_expected_members() -> None:
    assert {r.value for r in RequestType} == {"simple", "comparative", "exploratory"}
    assert {v.value for v in Verdict} == {"supported", "unsupported", "overstated"}


# --- validate_finding -------------------------------------------------------


def test_validate_finding_accepts_a_complete_finding() -> None:
    finding = _good_finding()
    assert validate_finding(finding) is finding


@pytest.mark.parametrize("missing_field", ["location", "published"])
def test_validate_finding_rejects_missing_source_provenance(missing_field: str) -> None:
    raw: dict[str, Any] = copy.deepcopy(dict(_good_finding()))
    del raw["source"][missing_field]

    with pytest.raises(InvalidFindingError, match=missing_field):
        validate_finding(raw)


@pytest.mark.parametrize("empty_field", ["location", "published"])
def test_validate_finding_rejects_empty_source_provenance(empty_field: str) -> None:
    raw: dict[str, Any] = copy.deepcopy(dict(_good_finding()))
    raw["source"][empty_field] = ""

    with pytest.raises(InvalidFindingError, match=empty_field):
        validate_finding(raw)


def test_validate_finding_rejects_missing_source_entirely() -> None:
    raw: dict[str, Any] = {"claim": "x", "quote": "y", "subagent": "z"}
    with pytest.raises(InvalidFindingError, match="source"):
        validate_finding(raw)


@pytest.mark.parametrize("field", ["claim", "quote", "subagent"])
def test_validate_finding_rejects_missing_finding_fields(field: str) -> None:
    raw: dict[str, Any] = copy.deepcopy(dict(_good_finding()))
    del raw[field]

    with pytest.raises(InvalidFindingError, match=field):
        validate_finding(raw)
