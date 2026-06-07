"""Tests for the provenance store (T2.1).

Acceptance criteria from IMPLEMENTATION-PLAN.md:
- round-trip: write -> read -> equal;
- checkpoint writes a file that survives a process restart (a fresh store
  instance reads identical data).
"""

from __future__ import annotations

import json
from pathlib import Path

from research_agent.store import ProvenanceStore
from research_agent.types import Claim, Finding, Source


def _finding(doc_id: str = "rpt-1", claim: str = "X grew 40%") -> Finding:
    return Finding(
        claim=claim,
        source=Source(
            doc_id=doc_id,
            title="APAC SaaS Outlook",
            location="page 12",
            published="2021-03-01",
            version="v3",
            peer_reviewed=False,
            sample={"n": 40, "scope": "SE Asia"},
        ),
        quote="...grew 40% YoY...",
        subagent="collector:a",
    )


def test_findings_round_trip(tmp_path: Path) -> None:
    store = ProvenanceStore("run-1", base_dir=tmp_path)
    findings = [_finding("rpt-1"), _finding("rpt-2", claim="Y fell 10%")]

    store.save_findings(findings)

    assert store.load_findings() == findings


def test_claims_round_trip(tmp_path: Path) -> None:
    store = ProvenanceStore("run-1", base_dir=tmp_path)
    claims = [
        Claim(text="X grew", source_ids=["rpt-1"], flags=["small-sample"]),
        Claim(text="Y fell", source_ids=["rpt-2"], flags=[]),
    ]

    store.save_claims(claims)

    assert store.load_claims() == claims


def test_append_findings_accumulates(tmp_path: Path) -> None:
    store = ProvenanceStore("run-1", base_dir=tmp_path)
    store.save_findings([_finding("rpt-1")])
    store.append_findings([_finding("rpt-2")])

    docs = [f["source"]["doc_id"] for f in store.load_findings()]
    assert docs == ["rpt-1", "rpt-2"]


def test_findings_by_doc_filters(tmp_path: Path) -> None:
    store = ProvenanceStore("run-1", base_dir=tmp_path)
    store.save_findings([_finding("rpt-1"), _finding("rpt-2"), _finding("rpt-1")])

    assert len(store.findings_by_doc("rpt-1")) == 2
    assert len(store.findings_by_doc("rpt-2")) == 1


def test_checkpoint_survives_process_restart(tmp_path: Path) -> None:
    # First "process": write + checkpoint, then drop the store reference.
    store = ProvenanceStore("run-1", base_dir=tmp_path)
    store.save_findings([_finding("rpt-1"), _finding("rpt-2")])
    store.save_claims([Claim(text="X grew", source_ids=["rpt-1"], flags=[])])
    manifest_path = store.checkpoint()
    del store

    # Second "process": a brand-new store at the same location reads it all back.
    reopened = ProvenanceStore("run-1", base_dir=tmp_path)
    assert len(reopened.load_findings()) == 2
    assert len(reopened.load_claims()) == 1

    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == "run-1"
    assert manifest["findings"] == 2
    assert manifest["claims"] == 1
    assert "checkpointed_at" in manifest


def test_load_from_empty_store_returns_empty(tmp_path: Path) -> None:
    store = ProvenanceStore("fresh", base_dir=tmp_path)
    assert store.load_findings() == []
    assert store.load_claims() == []


def test_save_report_writes_markdown(tmp_path: Path) -> None:
    store = ProvenanceStore("run-1", base_dir=tmp_path)
    path = store.save_report("# Report\nbody")

    assert path == store.report_path
    assert path.name == "report.md"
    assert path.read_text() == "# Report\nbody"
