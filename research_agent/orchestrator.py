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

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import query

from research_agent.collectors import (
    CellProgressFn,
    CellResult,
    QueryFn,
    SubQuestion,
    collect_all,
)
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
from research_agent.tracing import Span, Tracer, traced_query
from research_agent.types import Claim, Finding, Plan
from research_agent.verifier import (
    VerificationResult,
    keep_supported,
    verify_all,
)

# Called at each stage boundary with a human-readable status line. Optional so
# the autonomous pipeline stays silent unless a caller (e.g. the CLI) opts in.
ProgressFn = Callable[[str], None]

ClassifyFn = Callable[[str], Awaitable[IntakeResult]]
DiscoverFn = Callable[[str], Awaitable[list[str]]]
CollectFn = Callable[
    [Sequence[SubQuestion], "CellProgressFn | None"], Awaitable[list[CellResult]]
]
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
    report_path: Path | None = None  # where the report.md was written
    plan: Plan | None = None  # None for a resumed run (no fresh intake)
    revalidation: RevalidationResult | None = None


def error_aware_query(query_fn: QueryFn) -> QueryFn:
    """Surface the real CLI error text instead of the SDK's terse wrapper.

    The SDK raises ``Claude Code returned an error result: <subtype>`` (e.g.
    "success") when the CLI flags ``is_error`` — hiding the actual reason
    ("Credit balance is too low", rate limits, etc.), which rides on the
    `ResultMessage.result`. This captures that text and re-raises it clearly.
    """

    async def _wrapped(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        last_error: str | None = None
        try:
            async for message in query_fn(*args, **kwargs):
                if type(message).__name__ == "ResultMessage" and getattr(
                    message, "is_error", False
                ):
                    last_error = getattr(message, "result", None) or str(
                        getattr(message, "subtype", "error")
                    )
                yield message
        except Exception as exc:
            if last_error:
                raise RuntimeError(f"model call failed: {last_error}") from exc
            raise

    return _wrapped


def default_pipeline(query_fn: QueryFn = query, *, tracer: Tracer | None = None) -> Pipeline:
    """Bind the real stage functions to a shared ``query`` implementation.

    When a tracer is supplied, every model call is wrapped so prompts, LLM
    responses, and tool calls are recorded under the active stage span.
    """
    query_fn = error_aware_query(query_fn)
    if tracer is not None:
        query_fn = traced_query(tracer, query_fn)

    async def collect(
        subqs: Sequence[SubQuestion], on_cell: CellProgressFn | None = None
    ) -> list[CellResult]:
        return await collect_all(subqs, query_fn=query_fn, on_cell=on_cell)

    return Pipeline(
        classify=lambda request: classify(request, query_fn=query_fn),
        discover=lambda topic: discover(topic, query_fn=query_fn),
        collect=collect,
        synthesize=lambda findings: synthesize(findings, query_fn=query_fn),
        verify=lambda claims, findings: verify_all(claims, findings, query_fn=query_fn),
    )


def _emit(progress: ProgressFn | None, message: str) -> None:
    if progress is not None:
        progress(message)


@asynccontextmanager
async def _maybe_span(
    tracer: Tracer | None, name: str, kind: str, *, inputs: object = None
) -> AsyncIterator[Span | None]:
    """Open a trace span if tracing is on; otherwise a no-op context."""
    if tracer is None:
        yield None
    else:
        async with tracer.span(name, kind, inputs=inputs) as span:
            yield span


def _make_cell_progress(
    progress: ProgressFn | None, total: int
) -> CellProgressFn | None:
    """A per-collector callback that emits 'N/total' lines as each finishes."""
    if progress is None:
        return None
    done = 0

    def on_cell(cell: CellResult) -> None:
        nonlocal done
        done += 1
        if cell.error:
            status = f"failed ({cell.error})"
        else:
            status = f"{len(cell.findings)} finding(s)"
        _emit(progress, f"  [{done}/{total}] {cell.subquestion_id}: {status}")

    return on_cell


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
    progress: ProgressFn | None = None,
    tracer: Tracer | None = None,
) -> ResearchResult:
    """Run the full pipeline, enforcing every gate in code (PLAN §9)."""
    pipeline = pipeline or default_pipeline(tracer=tracer)

    async with _maybe_span(tracer, "run_research", "run", inputs={"request": request}):
        # [1] intake → [2] plan/discovery
        _emit(progress, "Classifying request…")
        async with _maybe_span(tracer, "intake", "stage", inputs={"request": request}):
            intake_result = await pipeline.classify(request)
        _emit(progress, f"Request type: {intake_result.request_type}")

        _emit(progress, "Discovering sub-questions…")
        async with _maybe_span(tracer, "discover", "stage", inputs={"request": request}):
            sub_questions = await pipeline.discover(request)
        plan = build_plan(intake_result, sub_questions)
        subqs = _plan_to_subquestions(plan)

        # [3] collect (parallel, scoped) → structured findings on disk
        _emit(progress, f"Collecting evidence ({len(subqs)} parallel task(s))…")
        async with _maybe_span(tracer, "collect", "stage", inputs={"tasks": len(subqs)}):
            cells = await pipeline.collect(subqs, _make_cell_progress(progress, len(subqs)))
        findings = [finding for cell in cells for finding in cell.findings]
        failures = [cell for cell in cells if cell.error]
        store.save_findings(findings)
        _emit(
            progress,
            f"Collected {len(findings)} finding(s) from {len(cells) - len(failures)} "
            f"task(s); {len(failures)} failed",
        )

        # [4] synthesize → cited claims
        _emit(progress, "Synthesizing claims…")
        async with _maybe_span(tracer, "synthesize", "stage", inputs={"findings": len(findings)}):
            synthesis_result = await pipeline.synthesize(findings)

        # GATE: synthesis-before-report (raises SynthesisIncompleteError)
        assert_synthesis_complete(synthesis_result)
        _emit(progress, f"Synthesized {len(synthesis_result.claims)} claim(s)")

        # [5] independent verification → keep only supported claims
        _emit(progress, "Verifying claims independently…")
        claim_count = len(synthesis_result.claims)
        async with _maybe_span(tracer, "verify", "stage", inputs={"claims": claim_count}):
            verifications = await pipeline.verify(synthesis_result.claims, findings)
        verified = keep_supported(verifications)
        _emit(progress, f"Verified {len(verified)}/{len(verifications)} claim(s) supported")

        # GATE: verify-before-report
        if not verified:
            raise ReportBlockedError("report blocked: no claims survived verification")
        store.save_claims(verified)

        # [6] report renders verified claims only
        _emit(progress, "Rendering report…")
        report_md = render(verified, sources_from_findings(findings), title=title)
        report_path = store.save_report(report_md)
        store.checkpoint()
        _emit(progress, f"Report written to {report_path}")

    return ResearchResult(
        report=report_md,
        findings=findings,
        claims=verified,
        verifications=verifications,
        report_path=report_path,
        plan=plan,
    )


async def resume_research(
    store: ProvenanceStore,
    source_of_record: SourceOfRecord,
    *,
    pipeline: Pipeline | None = None,
    title: str = "Research Report",
    progress: ProgressFn | None = None,
) -> ResearchResult:
    """Continue a saved run, re-validating freshness FIRST (PLAN §7).

    Never resumes the transcript: it diffs saved findings against the source of
    record, re-collects the stale branch, rebuilds synthesis, and only then
    proceeds through the same verify/report gates.
    """
    pipeline = pipeline or default_pipeline()
    _emit(progress, "Loading saved run and re-validating freshness…")
    saved = store.load_findings()

    async def recollect(stale: Finding) -> list[Finding]:
        doc_id = stale["source"]["doc_id"]
        subq = SubQuestion(
            id=doc_id,
            prompt=f"Re-collect the current evidence for document {doc_id}.",
            subagent=f"recollect:{doc_id}",
        )
        cells = await pipeline.collect([subq], None)
        return [finding for cell in cells for finding in cell.findings]

    async def resynthesize(fresh: Sequence[Finding]) -> SynthesisResult:
        return await pipeline.synthesize(fresh)

    # freshness-before-resume: rebuild from current evidence (never the transcript)
    reval = await revalidate(
        saved, source_of_record, recollect=recollect, resynthesize=resynthesize
    )
    store.save_findings(reval.fresh_findings)
    _emit(
        progress,
        f"Re-collected {len(reval.stale_findings)} stale source(s); synthesis rebuilt",
    )

    synthesis_result = reval.synthesis
    assert_synthesis_complete(synthesis_result)  # GATE

    _emit(progress, "Verifying claims independently…")
    verifications = await pipeline.verify(synthesis_result.claims, reval.fresh_findings)
    verified = keep_supported(verifications)
    _emit(progress, f"Verified {len(verified)}/{len(verifications)} claim(s) supported")
    if not verified:
        raise ReportBlockedError("report blocked: no claims survived verification")
    store.save_claims(verified)

    report_md = render(verified, sources_from_findings(reval.fresh_findings), title=title)
    report_path = store.save_report(report_md)
    store.checkpoint()

    return ResearchResult(
        report=report_md,
        findings=reval.fresh_findings,
        claims=verified,
        verifications=verifications,
        report_path=report_path,
        revalidation=reval,
    )
