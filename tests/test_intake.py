"""Tests for intake classification (T6.1).

Acceptance criteria from IMPLEMENTATION-PLAN.md:
- fixtures route simple/comparative/exploratory correctly;
- a comparative request yields detected axes.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from research_agent.intake import IntakeError, classify
from research_agent.types import RequestType
from tests.conftest import FakeMessage, scripted_query


def _intake_message(payload: dict[str, Any]) -> FakeMessage:
    return FakeMessage.text(json.dumps(payload))


@pytest.mark.anyio
async def test_classifies_simple() -> None:
    msg = _intake_message({"request_type": "simple", "axis": None, "axis_values": []})
    result = await classify("What is the capital of France?", query_fn=scripted_query([msg]))
    assert result.request_type is RequestType.SIMPLE
    assert result.axis is None


@pytest.mark.anyio
async def test_classifies_exploratory() -> None:
    msg = _intake_message({"request_type": "exploratory", "axis": None, "axis_values": []})
    result = await classify("How does AI copyright law work?", query_fn=scripted_query([msg]))
    assert result.request_type is RequestType.EXPLORATORY


@pytest.mark.anyio
async def test_classifies_comparative_with_axes() -> None:
    msg = _intake_message(
        {
            "request_type": "comparative",
            "axis": "sector",
            "axis_values": ["music_licensing", "model_training_data"],
            "dimensions": {"jurisdictions": ["US", "EU"], "date_range": "2023-2025"},
        }
    )
    result = await classify("Compare AI copyright across sectors", query_fn=scripted_query([msg]))

    assert result.request_type is RequestType.COMPARATIVE
    assert result.axis == "sector"
    assert result.axis_values == ["music_licensing", "model_training_data"]
    assert result.dimensions["jurisdictions"] == ["US", "EU"]


@pytest.mark.anyio
async def test_comparative_without_axes_raises() -> None:
    msg = _intake_message({"request_type": "comparative", "axis": None, "axis_values": []})
    with pytest.raises(IntakeError, match="axis"):
        await classify("compare things", query_fn=scripted_query([msg]))


@pytest.mark.anyio
async def test_unknown_request_type_raises() -> None:
    msg = _intake_message({"request_type": "nonsense"})
    with pytest.raises(IntakeError, match="unknown request_type"):
        await classify("x", query_fn=scripted_query([msg]))


@pytest.mark.anyio
async def test_unparseable_output_raises() -> None:
    with pytest.raises(IntakeError, match="parseable"):
        await classify("x", query_fn=scripted_query([FakeMessage.text("not json")]))
