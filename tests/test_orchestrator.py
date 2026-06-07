"""Tests for the orchestrator + gates under fault injection (T8.1).

Acceptance criteria from IMPLEMENTATION-PLAN.md:
- happy path (injected stages) produces a cited report;
- gates fire: (a) stale source -> revalidation triggered;
  (b) failed/incomplete synthesis -> report blocked;
  (c) unsupported claim -> excluded from report.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from research_agent.collectors import CellResult, SubQuestion
from research_agent.intake import IntakeResult
from research_agent.orchestrator import (
    Pipeline,
    ReportBlockedError,
    resume_research,
    run_research,
)
from research_agent.store import ProvenanceStore
from research_agent.synthesis import SynthesisIncompleteError, SynthesisResult
from research_agent.types import Claim, Finding, RequestType, Source, Verdict
from research_agent.verifier import VerificationResult


def _finding(doc_id: str = "d1", version: str = "v1") -> Finding:
    return Finding(
        claim="X grew 40%",
        source=Source(
            doc_id=doc_id,
            title=f"Doc {doc_id}",
            location="page 1",
            published="2024-01-01",
            version=version,
            peer_reviewed=True,
            sample=None,
        ),
        quote="...grew 40%...",
        subagent="collector:a",
    )


def _store(tmp_path: Path) -> ProvenanceStore:
    return ProvenanceStore("run", base_dir=tmp_path)


def make_pipeline(**overrides: object) -> Pipeline:
    """Default = happy path; override one stage to fault-inject."""

    async def classify(_request: str) -> IntakeResult:
        return IntakeResult(RequestType.SIMPLE)

    async def discover(_topic: str) -> list[str]:
        return ["q1"]

    async def collect(
        subqs: Sequence[SubQuestion], on_cell: object = None
    ) -> list[CellResult]:
        cells = [CellResult("q0", [_finding("d1")])]
        if callable(on_cell):
            for cell in cells:
                on_cell(cell)
        return cells

    async def synthesize(_findings: Sequence[Finding]) -> SynthesisResult:
        return SynthesisResult("ok", [Claim(text="X grew 40%", source_ids=["d1"], flags=[])])

    async def verify(
        claims: Sequence[Claim], _findings: Sequence[Finding]
    ) -> list[VerificationResult]:
        return [VerificationResult(c, Verdict.SUPPORTED) for c in claims]

    return Pipeline(
        classify=overrides.get("classify", classify),  # type: ignore[arg-type]
        discover=overrides.get("discover", discover),  # type: ignore[arg-type]
        collect=overrides.get("collect", collect),  # type: ignore[arg-type]
        synthesize=overrides.get("synthesize", synthesize),  # type: ignore[arg-type]
        verify=overrides.get("verify", verify),  # type: ignore[arg-type]
    )


@pytest.mark.anyio
async def test_happy_path_produces_cited_report(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = await run_research("how big is X?", store=store, pipeline=make_pipeline())

    assert "X grew 40%" in result.report
    assert "## Sources" in result.report
    assert "[d1]" in result.report and "page 1" in result.report  # page + date citation
    # provenance persisted
    assert len(store.load_findings()) == 1
    assert len(store.load_claims()) == 1


@pytest.mark.anyio
async def test_progress_callback_reports_each_stage(tmp_path: Path) -> None:
    messages: list[str] = []
    await run_research(
        "how big is X?",
        store=_store(tmp_path),
        pipeline=make_pipeline(),
        progress=messages.append,
    )

    joined = " | ".join(messages)
    assert "Classifying" in joined
    assert "Collecting" in joined
    assert "Synthesizing" in joined
    assert "Verifying" in joined
    assert "Rendering" in joined
    # per-task progress line for the single collector
    assert any("[1/1]" in m for m in messages)


@pytest.mark.anyio
async def test_failed_synthesis_blocks_report(tmp_path: Path) -> None:
    async def failed_synthesis(_findings: Sequence[Finding]) -> SynthesisResult:
        return SynthesisResult(status="error", error="synthesis stalled")

    with pytest.raises(SynthesisIncompleteError):
        await run_research(
            "x", store=_store(tmp_path), pipeline=make_pipeline(synthesize=failed_synthesis)
        )


@pytest.mark.anyio
async def test_uncited_claim_blocks_report(tmp_path: Path) -> None:
    async def uncited(_findings: Sequence[Finding]) -> SynthesisResult:
        return SynthesisResult("ok", [Claim(text="floating claim", source_ids=[], flags=[])])

    with pytest.raises(SynthesisIncompleteError, match="source_ids"):
        await run_research("x", store=_store(tmp_path), pipeline=make_pipeline(synthesize=uncited))


@pytest.mark.anyio
async def test_unsupported_claim_excluded_from_report(tmp_path: Path) -> None:
    store = _store(tmp_path)

    async def two_findings(
        _subqs: Sequence[SubQuestion], _on_cell: object = None
    ) -> list[CellResult]:
        return [CellResult("q0", [_finding("d1"), _finding("d2")])]

    async def two_claims(_findings: Sequence[Finding]) -> SynthesisResult:
        return SynthesisResult(
            "ok",
            [
                Claim(text="solid claim", source_ids=["d1"], flags=[]),
                Claim(text="overstated claim", source_ids=["d2"], flags=[]),
            ],
        )

    async def split_verdicts(
        claims: Sequence[Claim], _findings: Sequence[Finding]
    ) -> list[VerificationResult]:
        return [
            VerificationResult(
                c,
                Verdict.SUPPORTED if c["text"] == "solid claim" else Verdict.OVERSTATED,
            )
            for c in claims
        ]

    result = await run_research(
        "x",
        store=store,
        pipeline=make_pipeline(collect=two_findings, synthesize=two_claims, verify=split_verdicts),
    )

    assert "solid claim" in result.report
    assert "overstated claim" not in result.report  # excluded by verification
    assert [c["text"] for c in store.load_claims()] == ["solid claim"]


@pytest.mark.anyio
async def test_all_unsupported_blocks_report(tmp_path: Path) -> None:
    async def all_overstated(
        claims: Sequence[Claim], _findings: Sequence[Finding]
    ) -> list[VerificationResult]:
        return [VerificationResult(c, Verdict.OVERSTATED) for c in claims]

    with pytest.raises(ReportBlockedError, match="no claims survived"):
        await run_research(
            "x", store=_store(tmp_path), pipeline=make_pipeline(verify=all_overstated)
        )


@pytest.mark.anyio
async def test_comparative_request_fans_out_per_axis_value(tmp_path: Path) -> None:
    captured: dict[str, list[SubQuestion]] = {}

    async def comparative_classify(_request: str) -> IntakeResult:
        return IntakeResult(
            request_type=RequestType.COMPARATIVE,
            axis="sector",
            axis_values=["music", "training"],
            dimensions={"jurisdictions": ["US"]},
        )

    async def capturing_collect(
        subqs: Sequence[SubQuestion], _on_cell: object = None
    ) -> list[CellResult]:
        captured["subqs"] = list(subqs)
        return [CellResult(s.id, [_finding("d1")]) for s in subqs]

    await run_research(
        "compare sectors",
        store=_store(tmp_path),
        pipeline=make_pipeline(classify=comparative_classify, collect=capturing_collect),
    )

    assert {s.axis_value for s in captured["subqs"]} == {"music", "training"}


@pytest.mark.anyio
async def test_resume_revalidates_stale_source_before_continuing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save_findings([_finding("d1", "v1"), _finding("d2", "v1")])

    record = {"d1": "v1", "d2": "v2"}  # d2 was revised → stale
    recollected: list[str] = []

    async def recollect_collect(
        subqs: Sequence[SubQuestion], _on_cell: object = None
    ) -> list[CellResult]:
        # the resume path re-collects exactly the stale doc
        results = []
        for s in subqs:
            recollected.append(s.id)
            results.append(CellResult(s.id, [_finding("d2", "v2")]))
        return results

    async def synth(_findings: Sequence[Finding]) -> SynthesisResult:
        return SynthesisResult("ok", [Claim(text="reconciled", source_ids=["d1"], flags=[])])

    result = await resume_research(
        store,
        lambda doc: record.get(doc),
        pipeline=make_pipeline(collect=recollect_collect, synthesize=synth),
    )

    assert result.revalidation is not None
    assert [f["source"]["doc_id"] for f in result.revalidation.stale_findings] == ["d2"]
    assert recollected == ["d2"]  # only the stale branch re-collected
    assert "reconciled" in result.report
