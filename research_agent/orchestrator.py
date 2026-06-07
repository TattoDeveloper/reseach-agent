"""Orchestrator — wires the stages and enforces every gate IN CODE (PLAN §9).

The coordinator LLM plans and reasons within each stage; it never gets to *skip a
prerequisite*. The dependencies are enforced here, in Python:

- **synthesis-before-report** — `assert_synthesis_complete` raises before any
  report is rendered (`multi-agent-synthesis-dependency-fix.md`).
- **verify-before-report** — only `keep_supported` claims reach the report; if
  none survive, the report is blocked (`independent-review-architecture` per §5).
- **freshness-before-resume** — `resume_research` re-validates a saved run and
  rebuilds synthesis from fresh evidence before continuing
  (`research-stale-session-revalidation.md`); it never resumes a transcript.

Stages are injected as a `Pipeline` of callables (defaults bound to the real
stage functions). Tests swap in fakes to fault-inject — a stale source, a failed
synthesis, an unsupported claim — and assert each gate fires.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from claude_agent_sdk import query

from research_agent.collectors import CellResult, QueryFn, SubQuestion, collect_all
from research_agent.intake import IntakeResult, classify
from research_agent.planner import build_plan, discover, schema_to_subquestions
from research_agent.report import render, sources_from_findings
from research_agent.store import (
    ProvenanceStore,
    RevalidationResult,
    SourceOfRecord,
    revalidate,
)
from research_agent.synthesis import (
    SynthesisResult,
    assert_synthesis_complete,
    synthesize,
)
from research_agent.types import Claim, Finding, Plan
from research_agent.verifier import (
    VerificationResult,
    keep_supported,
    verify_all,
)

ClassifyFn = Callable[[str], Awaitable[IntakeResult]]
DiscoverFn = Callable[[str], Awaitable[list[str]]]
CollectFn = Callable[[Sequence[SubQuestion]], Awaitable[list[CellResult]]]
SynthesizeFn = Callable[[Sequence[Finding]], Awaitable[SynthesisResult]]
VerifyFn = Callable[[Sequence[Claim], Sequence[Finding]], Awaitable[list[VerificationResult]]]


class ReportBlockedError(RuntimeError):
    """Raised when no claim survives verification — the report is blocked (§5)."""


@dataclass
class Pipeline:
    """The five LLM-backed stages, injectable for testing/fault-injection."""

    classify: ClassifyFn
    discover: DiscoverFn
    collect: CollectFn
    synthesize: SynthesizeFn
    verify: VerifyFn


@dataclass
class ResearchResult:
    """Everything a run produced — the report plus its provenance trail."""

    report: str
    findings: list[Finding]
    claims: list[Claim]
    verifications: list[VerificationResult]
    plan: Plan | None = None  # None for a resumed run (no fresh intake)
    revalidation: RevalidationResult | None = None


def default_pipeline(query_fn: QueryFn = query) -> Pipeline:
    """Bind the real stage functions to a shared ``query`` implementation."""
    return Pipeline(
        classify=lambda request: classify(request, query_fn=query_fn),
        discover=lambda topic: discover(topic, query_fn=query_fn),
        collect=lambda subqs: collect_all(subqs, query_fn=query_fn),
        synthesize=lambda findings: synthesize(findings, query_fn=query_fn),
        verify=lambda claims, findings: verify_all(claims, findings, query_fn=query_fn),
    )


def _plan_to_subquestions(plan: Plan) -> list[SubQuestion]:
    """Comparative → one collector per matrix cell; otherwise one per question."""
    if "schema" in plan:
        return schema_to_subquestions(plan["schema"])
    return [
        SubQuestion(id=f"q{i}", prompt=question, subagent=f"collector:q{i}")
        for i, question in enumerate(plan["sub_questions"])
    ]


async def run_research(
    request: str,
    *,
    store: ProvenanceStore,
    pipeline: Pipeline | None = None,
    title: str = "Research Report",
) -> ResearchResult:
    """Run the full pipeline, enforcing every gate in code (PLAN §9)."""
    pipeline = pipeline or default_pipeline()

    # [1] intake → [2] plan/discovery
    intake_result = await pipeline.classify(request)
    sub_questions = await pipeline.discover(request)
    plan = build_plan(intake_result, sub_questions)

    # [3] collect (parallel, scoped) → structured findings on disk
    cells = await pipeline.collect(_plan_to_subquestions(plan))
    findings = [finding for cell in cells for finding in cell.findings]
    store.save_findings(findings)

    # [4] synthesize → cited claims
    synthesis_result = await pipeline.synthesize(findings)

    # GATE: synthesis-before-report (raises SynthesisIncompleteError)
    assert_synthesis_complete(synthesis_result)

    # [5] independent verification → keep only supported claims
    verifications = await pipeline.verify(synthesis_result.claims, findings)
    verified = keep_supported(verifications)

    # GATE: verify-before-report
    if not verified:
        raise ReportBlockedError("report blocked: no claims survived verification")
    store.save_claims(verified)

    # [6] report renders verified claims only
    report_md = render(verified, sources_from_findings(findings), title=title)
    store.checkpoint()

    return ResearchResult(
        report=report_md,
        findings=findings,
        claims=verified,
        verifications=verifications,
        plan=plan,
    )


async def resume_research(
    store: ProvenanceStore,
    source_of_record: SourceOfRecord,
    *,
    pipeline: Pipeline | None = None,
    title: str = "Research Report",
) -> ResearchResult:
    """Continue a saved run, re-validating freshness FIRST (PLAN §7).

    Never resumes the transcript: it diffs saved findings against the source of
    record, re-collects the stale branch, rebuilds synthesis, and only then
    proceeds through the same verify/report gates.
    """
    pipeline = pipeline or default_pipeline()
    saved = store.load_findings()

    async def recollect(stale: Finding) -> list[Finding]:
        doc_id = stale["source"]["doc_id"]
        subq = SubQuestion(
            id=doc_id,
            prompt=f"Re-collect the current evidence for document {doc_id}.",
            subagent=f"recollect:{doc_id}",
        )
        cells = await pipeline.collect([subq])
        return [finding for cell in cells for finding in cell.findings]

    async def resynthesize(fresh: Sequence[Finding]) -> SynthesisResult:
        return await pipeline.synthesize(fresh)

    # freshness-before-resume: rebuild from current evidence (never the transcript)
    reval = await revalidate(
        saved, source_of_record, recollect=recollect, resynthesize=resynthesize
    )
    store.save_findings(reval.fresh_findings)

    synthesis_result = reval.synthesis
    assert_synthesis_complete(synthesis_result)  # GATE

    verifications = await pipeline.verify(synthesis_result.claims, reval.fresh_findings)
    verified = keep_supported(verifications)
    if not verified:
        raise ReportBlockedError("report blocked: no claims survived verification")
    store.save_claims(verified)

    report_md = render(verified, sources_from_findings(reval.fresh_findings), title=title)
    store.checkpoint()

    return ResearchResult(
        report=report_md,
        findings=reval.fresh_findings,
        claims=verified,
        verifications=verifications,
        revalidation=reval,
    )
