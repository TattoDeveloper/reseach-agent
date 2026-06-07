"""Tests for the parallel collectors (T2.2).

Acceptance criteria from IMPLEMENTATION-PLAN.md:
- collect_all returns one CellResult per sub-question;
- every returned finding has source.location + published;
- concurrency is capped (max in-flight <= semaphore limit);
- a missing-answer case produces an explicit "no rule found" marker (§3c),
  and a neither-filled-nor-marked case raises rather than dropping the cell.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anyio
import pytest

from research_agent.collectors import (
    CellResult,
    CollectionError,
    SubQuestion,
    collect_all,
    collect_one,
)
from tests.conftest import FakeMessage, scripted_query


def _finding_dict(with_subagent: bool = True) -> dict[str, Any]:
    finding: dict[str, Any] = {
        "claim": "Market X grew 40% YoY",
        "source": {
            "doc_id": "rpt-1",
            "title": "APAC SaaS Outlook",
            "location": "page 12",
            "published": "2021-03-01",
            "version": "v3",
            "peer_reviewed": False,
            "sample": {"n": 40, "scope": "SE Asia"},
        },
        "quote": "...grew 40% YoY...",
    }
    if with_subagent:
        finding["subagent"] = "collector:a"
    return finding


def _subq(qid: str = "q1") -> SubQuestion:
    return SubQuestion(id=qid, prompt="What is the growth rate?", subagent=f"collector:{qid}")


def _json_message(payload: dict[str, Any]) -> FakeMessage:
    return FakeMessage.text(json.dumps(payload))


@pytest.mark.anyio
async def test_collect_one_parses_findings_with_provenance() -> None:
    msg = _json_message({"findings": [_finding_dict()]})
    result = await collect_one(_subq(), query_fn=scripted_query([msg]))

    assert result.no_rule_found is False
    assert len(result.findings) == 1
    source = result.findings[0]["source"]
    assert source["location"] and source["published"]


@pytest.mark.anyio
async def test_collect_one_stamps_subagent_when_model_omits_it() -> None:
    msg = _json_message({"findings": [_finding_dict(with_subagent=False)]})
    subq = _subq("q7")
    result = await collect_one(subq, query_fn=scripted_query([msg]))

    assert result.findings[0]["subagent"] == subq.subagent


@pytest.mark.anyio
async def test_collect_one_tolerates_prose_around_json() -> None:
    text = "Here is what I found:\n```json\n" + json.dumps({"findings": [_finding_dict()]})
    text += "\n```\nThanks!"
    result = await collect_one(_subq(), query_fn=scripted_query([FakeMessage.text(text)]))
    assert len(result.findings) == 1


@pytest.mark.anyio
async def test_collect_one_marks_no_rule_found() -> None:
    msg = _json_message({"no_rule_found": True})
    result = await collect_one(_subq(), query_fn=scripted_query([msg]))

    assert result.no_rule_found is True
    assert result.findings == []


@pytest.mark.anyio
async def test_collect_one_raises_on_dropped_cell() -> None:
    # findings empty AND not marked no_rule_found -> §3c violation, must raise.
    msg = _json_message({"findings": []})
    with pytest.raises(CollectionError, match="§3c"):
        await collect_one(_subq(), query_fn=scripted_query([msg]))


@pytest.mark.anyio
async def test_collect_one_raises_on_empty_output() -> None:
    with pytest.raises(CollectionError):
        await collect_one(_subq(), query_fn=scripted_query([FakeMessage.text("   ")]))


@pytest.mark.anyio
async def test_collect_one_rejects_finding_without_provenance() -> None:
    bad = _finding_dict()
    del bad["source"]["published"]
    from research_agent.types import InvalidFindingError

    with pytest.raises(InvalidFindingError, match="published"):
        await collect_one(_subq(), query_fn=scripted_query([_json_message({"findings": [bad]})]))


@pytest.mark.anyio
async def test_collect_all_returns_one_result_per_subquestion() -> None:
    subqs = [_subq(f"q{i}") for i in range(5)]
    msg = _json_message({"no_rule_found": True})
    results = await collect_all(subqs, query_fn=scripted_query([msg]))

    assert len(results) == len(subqs)
    assert all(isinstance(r, CellResult) for r in results)


@pytest.mark.anyio
async def test_collect_all_isolates_a_failing_collector() -> None:
    # One collector raises; the fan-out must NOT abort — the cell records the error.
    calls = 0

    def flaky_query(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        nonlocal calls
        calls += 1
        nth = calls

        async def _gen() -> AsyncIterator[Any]:
            if nth == 2:
                raise RuntimeError("boom")
            yield _json_message({"no_rule_found": True})

        return _gen()

    subqs = [_subq("q0"), _subq("q1"), _subq("q2")]
    results = await collect_all(subqs, query_fn=flaky_query)

    assert len(results) == 3  # no cell dropped, no ExceptionGroup raised
    errored = [r for r in results if r.error]
    assert len(errored) == 1
    assert "boom" in (errored[0].error or "")


@pytest.mark.anyio
async def test_collect_all_reports_each_cell_via_on_cell() -> None:
    seen: list[str] = []
    subqs = [_subq("q0"), _subq("q1")]
    msg = _json_message({"no_rule_found": True})

    await collect_all(
        subqs, query_fn=scripted_query([msg]), on_cell=lambda c: seen.append(c.subquestion_id)
    )

    assert len(seen) == 2


@pytest.mark.anyio
async def test_collect_all_caps_concurrency() -> None:
    active = 0
    peak = 0

    def tracking_query(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await anyio.sleep(0.01)  # hold the slot so overlap is observable
            yield _json_message({"no_rule_found": True})
            active -= 1

        return _gen()

    subqs = [_subq(f"q{i}") for i in range(8)]
    results = await collect_all(subqs, max_concurrency=2, query_fn=tracking_query)

    assert len(results) == 8
    assert peak <= 2
    assert active == 0
