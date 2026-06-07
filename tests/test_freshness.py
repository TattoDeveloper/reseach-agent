"""Tests for freshness re-validation on resume (T7.1).

Acceptance criteria from IMPLEMENTATION-PLAN.md:
- a stale-version fixture is detected and refreshed;
- synthesis is rebuilt (not reused) after refresh.
"""

from __future__ import annotations

import pytest

from research_agent.store import find_stale, partition_fresh_stale, revalidate
from research_agent.types import Finding, Source


def _finding(doc_id: str, version: str, claim: str = "c") -> Finding:
    return Finding(
        claim=claim,
        source=Source(
            doc_id=doc_id,
            title="Doc",
            location="page 1",
            published="2024-01-01",
            version=version,
            peer_reviewed=True,
            sample=None,
        ),
        quote="q",
        subagent="collector:a",
    )


def test_find_stale_detects_version_mismatch() -> None:
    findings = [_finding("a", "v1"), _finding("b", "v1")]
    record = {"a": "v1", "b": "v2"}  # b was revised

    stale = find_stale(findings, lambda doc_id: record.get(doc_id))

    assert [f["source"]["doc_id"] for f in stale] == ["b"]


def test_partition_keeps_current_work() -> None:
    findings = [_finding("a", "v1"), _finding("b", "v1")]
    record = {"a": "v1", "b": "v2"}

    fresh, stale = partition_fresh_stale(findings, lambda d: record.get(d))

    assert [f["source"]["doc_id"] for f in fresh] == ["a"]
    assert [f["source"]["doc_id"] for f in stale] == ["b"]


def test_unknown_doc_is_not_treated_as_stale() -> None:
    findings = [_finding("a", "v1")]
    stale = find_stale(findings, lambda _doc: None)  # record can't resolve it
    assert stale == []


@pytest.mark.anyio
async def test_revalidate_recollects_stale_and_rebuilds_synthesis() -> None:
    findings = [_finding("a", "v1"), _finding("b", "v1")]
    record = {"a": "v1", "b": "v2"}  # b is stale

    recollected: list[Finding] = []
    resynth_inputs: list[list[Finding]] = []

    async def recollect(finding: Finding) -> list[Finding]:
        # return the refreshed version of the superseded document
        fresh = _finding(finding["source"]["doc_id"], "v2", claim="refreshed")
        recollected.append(fresh)
        return [fresh]

    async def resynthesize(fresh: list[Finding]) -> str:
        resynth_inputs.append(fresh)
        return "rebuilt-synthesis"

    result = await revalidate(
        findings,
        lambda d: record.get(d),
        recollect=recollect,
        resynthesize=resynthesize,
    )

    # stale branch re-collected
    assert [f["source"]["doc_id"] for f in result.stale_findings] == ["b"]
    assert len(recollected) == 1
    # fresh set = kept 'a' + refreshed 'b'
    assert [f["claim"] for f in result.fresh_findings] == ["c", "refreshed"]
    # synthesis REBUILT from the refreshed set, not reused
    assert result.synthesis == "rebuilt-synthesis"
    assert resynth_inputs == [result.fresh_findings]
