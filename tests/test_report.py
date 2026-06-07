"""Tests for report generation (T5.1).

Acceptance criteria from IMPLEMENTATION-PLAN.md:
- citations carry page + date;
- a flagged claim renders its caveat;
- passing a raw Finding (not a Claim) is rejected.
"""

from __future__ import annotations

import pytest

from research_agent.report import ReportInputError, render, sources_from_findings
from research_agent.types import Claim, Finding, Source


def _source(
    doc_id: str = "rpt-1",
    *,
    location: str = "page 12",
    published: str = "2021-03-01",
    n: int | None = 40,
) -> Source:
    return Source(
        doc_id=doc_id,
        title="APAC SaaS Outlook",
        location=location,
        published=published,
        version="v1",
        peer_reviewed=False,
        sample=None if n is None else {"n": n, "scope": "SE Asia"},
    )


def _claim(
    text: str = "Market X grew 40% YoY",
    source_ids: list[str] | None = None,
    flags: list[str] | None = None,
) -> Claim:
    return Claim(
        text=text,
        source_ids=source_ids or ["rpt-1"],
        flags=flags or [],  # type: ignore[arg-type]
    )


def test_citations_carry_page_and_date() -> None:
    sources = {"rpt-1": _source(location="page 12", published="2021-03-01")}
    out = render([_claim()], sources)

    assert "## Sources" in out
    assert "[rpt-1]" in out
    assert "page 12" in out  # page (location)
    assert "2021-03-01" in out  # date (published)


def test_flagged_claim_renders_grounded_caveat() -> None:
    sources = {"rpt-1": _source(published="2021-03-01", n=40)}
    claim = _claim(flags=["possibly-outdated", "non-peer-reviewed", "small-sample"])

    out = render([claim], sources)

    assert "Caveat:" in out
    assert "may be outdated (published 2021-03-01)" in out
    assert "not peer-reviewed" in out
    assert "small sample (n=40, SE Asia)" in out


def test_unflagged_claim_has_no_caveat() -> None:
    sources = {"rpt-1": _source()}
    out = render([_claim(flags=[])], sources)
    assert "Caveat" not in out


def test_render_rejects_raw_finding() -> None:
    finding = Finding(
        claim="X grew 40%",
        source=_source(),
        quote="...grew 40%...",
        subagent="collector:a",
    )
    with pytest.raises(ReportInputError, match="Finding"):
        render([finding], {"rpt-1": _source()})  # type: ignore[list-item]


def test_render_rejects_unknown_source_id() -> None:
    with pytest.raises(ReportInputError, match="unknown source_id"):
        render([_claim(source_ids=["missing"])], {"rpt-1": _source()})


def test_render_numbers_multiple_claims_and_dedups_sources() -> None:
    sources = {"rpt-1": _source("rpt-1"), "rpt-2": _source("rpt-2")}
    claims = [
        _claim("First claim", source_ids=["rpt-1"]),
        _claim("Second claim", source_ids=["rpt-1", "rpt-2"]),
    ]

    out = render(claims, sources)

    assert "1. First claim [rpt-1]" in out
    assert "2. Second claim [rpt-1, rpt-2]" in out
    # rpt-1 cited twice but listed once in Sources.
    assert out.count("- [rpt-1]") == 1


def test_sources_from_findings_indexes_by_doc_id() -> None:
    findings = [
        Finding(claim="a", source=_source("rpt-1"), quote="q", subagent="s"),
        Finding(claim="b", source=_source("rpt-2"), quote="q", subagent="s"),
    ]
    index = sources_from_findings(findings)
    assert set(index) == {"rpt-1", "rpt-2"}
