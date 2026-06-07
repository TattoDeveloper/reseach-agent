"""Tests for synthesis + the dependency gate (T3.1, T3.2).

Acceptance criteria from IMPLEMENTATION-PLAN.md:
- flag-computation unit tests (stale date -> flag; small n -> flag);
- a comparative fixture surfaces a cross-axis conflict;
- the gate RAISES when status != ok or any claim lacks source_ids.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

from research_agent.synthesis import (
    SynthesisIncompleteError,
    SynthesisResult,
    assert_synthesis_complete,
    compute_standing_flags,
    synthesize,
)
from research_agent.types import Claim, Finding, Source
from tests.conftest import FakeMessage, scripted_query

AS_OF = date(2025, 1, 1)


def _source(
    doc_id: str,
    *,
    published: str = "2024-12-01",
    peer_reviewed: bool = True,
    n: int | None = 100,
) -> Source:
    return Source(
        doc_id=doc_id,
        title=f"Doc {doc_id}",
        location="page 1",
        published=published,
        version="v1",
        peer_reviewed=peer_reviewed,
        sample=None if n is None else {"n": n, "scope": "global"},
    )


def _finding(source: Source, claim: str = "fact") -> Finding:
    return Finding(claim=claim, source=source, quote="q", subagent="collector:a")


def _claims_message(claims: list[dict[str, Any]]) -> FakeMessage:
    return FakeMessage.text(json.dumps({"claims": claims}))


# --- flag computation (deterministic) ---------------------------------------


def test_flag_stale_source_is_possibly_outdated() -> None:
    sources = {"d": _source("d", published="2021-01-01")}
    flags = compute_standing_flags(
        ["d"], sources, as_of=AS_OF, half_life_days=365, min_n=30
    )
    assert "possibly-outdated" in flags


def test_flag_non_peer_reviewed() -> None:
    sources = {"d": _source("d", peer_reviewed=False)}
    flags = compute_standing_flags(
        ["d"], sources, as_of=AS_OF, half_life_days=365, min_n=30
    )
    assert "non-peer-reviewed" in flags


def test_flag_small_sample() -> None:
    sources = {"d": _source("d", n=5)}
    flags = compute_standing_flags(
        ["d"], sources, as_of=AS_OF, half_life_days=365, min_n=30
    )
    assert "small-sample" in flags


def test_strong_source_has_no_flags() -> None:
    sources = {"d": _source("d", published="2024-12-01", peer_reviewed=True, n=100)}
    flags = compute_standing_flags(
        ["d"], sources, as_of=AS_OF, half_life_days=365, min_n=30
    )
    assert flags == []


def test_one_strong_source_lifts_the_caveat() -> None:
    # weak + strong cited together: the strong source clears every bar.
    sources = {
        "weak": _source("weak", published="2019-01-01", peer_reviewed=False, n=5),
        "strong": _source("strong", published="2024-12-01", peer_reviewed=True, n=500),
    }
    flags = compute_standing_flags(
        ["weak", "strong"], sources, as_of=AS_OF, half_life_days=365, min_n=30
    )
    assert flags == []


def test_unknown_source_ids_yield_no_flags() -> None:
    assert compute_standing_flags(
        ["missing"], {}, as_of=AS_OF, half_life_days=365, min_n=30
    ) == []


# --- synthesize -------------------------------------------------------------


@pytest.mark.anyio
async def test_synthesize_binds_claims_to_sources_and_flags() -> None:
    findings = [_finding(_source("d1", peer_reviewed=False, n=5))]
    msg = _claims_message([{"text": "X grew", "source_ids": ["d1"]}])

    result = await synthesize(
        findings, as_of=AS_OF, query_fn=scripted_query([msg])
    )

    assert result.status == "ok"
    assert result.claims[0]["source_ids"] == ["d1"]
    assert set(result.claims[0]["flags"]) == {"non-peer-reviewed", "small-sample"}


@pytest.mark.anyio
async def test_synthesize_errors_on_empty_findings() -> None:
    result = await synthesize([], query_fn=scripted_query([]))
    assert result.status == "error"


@pytest.mark.anyio
async def test_synthesize_errors_on_unparseable_output() -> None:
    findings = [_finding(_source("d1"))]
    result = await synthesize(
        findings, query_fn=scripted_query([FakeMessage.text("not json at all")])
    )
    assert result.status == "error"


@pytest.mark.anyio
async def test_synthesize_reconciles_across_the_axis() -> None:
    # Two axis values (music vs training), same dimension (US): conflicting
    # conclusions. Synthesis reconciles into one claim citing BOTH columns.
    findings = [
        _finding(_source("music-US"), claim="music: rule permits use"),
        _finding(_source("training-US"), claim="training: rule restricts use"),
    ]
    reconciled = _claims_message(
        [{"text": "In the US the rule permits music use but restricts training use",
          "source_ids": ["music-US", "training-US"]}]
    )

    result = await synthesize(findings, as_of=AS_OF, query_fn=scripted_query([reconciled]))

    assert result.status == "ok"
    assert result.claims[0]["source_ids"] == ["music-US", "training-US"]


# --- the gate (T3.2) --------------------------------------------------------


def test_gate_raises_when_status_not_ok() -> None:
    result = SynthesisResult(status="error", error="synthesis stalled")
    with pytest.raises(SynthesisIncompleteError, match="status=error"):
        assert_synthesis_complete(result)


def test_gate_raises_when_a_claim_lacks_source_ids() -> None:
    result = SynthesisResult(
        status="ok",
        claims=[
            Claim(text="cited", source_ids=["d1"], flags=[]),
            Claim(text="uncited", source_ids=[], flags=[]),  # the offender
        ],
    )
    with pytest.raises(SynthesisIncompleteError, match="source_ids"):
        assert_synthesis_complete(result)


def test_gate_raises_when_no_claims() -> None:
    result = SynthesisResult(status="ok", claims=[])
    with pytest.raises(SynthesisIncompleteError, match="no claims"):
        assert_synthesis_complete(result)


def test_gate_passes_for_fully_cited_synthesis() -> None:
    result = SynthesisResult(
        status="ok", claims=[Claim(text="cited", source_ids=["d1"], flags=[])]
    )
    assert assert_synthesis_complete(result) is result
