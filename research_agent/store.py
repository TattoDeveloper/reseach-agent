"""Provenance store + checkpointing (PLAN §7/§8, T2.1).

The structured Finding/Claim records — not the SDK session JSONL — are the
source of truth. From `stale-session-context-fix.md`: transcripts are
disposable; checkpoint the structured store to disk so a run survives a process
restart and can be re-validated for freshness (§7) without trusting a resumed
transcript.

Layout (one directory per run)::

    <base_dir>/<run_id>/
        findings.jsonl     # one Finding per line
        claims.jsonl       # one Claim per line
        checkpoint.json    # manifest: run_id, counts, timestamp

Findings are re-validated through `validate_finding` on load, so a corrupted or
provenance-stripped record fails loudly at the read boundary rather than
silently entering synthesis.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_agent.types import Claim, Finding, validate_finding


class ProvenanceStore:
    """Disk-backed store for one research run's Findings and Claims."""

    def __init__(self, run_id: str, base_dir: Path | str = "runs") -> None:
        self.run_id = run_id
        self.run_dir = Path(base_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def findings_path(self) -> Path:
        return self.run_dir / "findings.jsonl"

    @property
    def claims_path(self) -> Path:
        return self.run_dir / "claims.jsonl"

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / "checkpoint.json"

    # --- findings -----------------------------------------------------------

    def save_findings(self, findings: list[Finding]) -> None:
        """Persist the full finding set, replacing any prior contents."""
        self._write_jsonl(self.findings_path, findings)

    def append_findings(self, findings: list[Finding]) -> None:
        """Append findings (e.g. a re-collected stale branch, §7)."""
        with self.findings_path.open("a", encoding="utf-8") as fh:
            for finding in findings:
                fh.write(json.dumps(finding) + "\n")

    def load_findings(self) -> list[Finding]:
        """Read findings back, re-validating provenance on the way in."""
        return [
            validate_finding(record) for record in self._read_jsonl(self.findings_path)
        ]

    def findings_by_doc(self, doc_id: str) -> list[Finding]:
        """All findings whose source is the given document."""
        return [f for f in self.load_findings() if f["source"]["doc_id"] == doc_id]

    # --- claims -------------------------------------------------------------

    def save_claims(self, claims: list[Claim]) -> None:
        self._write_jsonl(self.claims_path, claims)

    def load_claims(self) -> list[Claim]:
        records = self._read_jsonl(self.claims_path)
        # Claims are produced internally (not from an untrusted worker boundary),
        # so a structural cast is sufficient here.
        return [
            Claim(
                text=str(r["text"]),
                source_ids=list(r["source_ids"]),
                flags=list(r.get("flags", [])),
            )
            for r in records
        ]

    # --- checkpoint ---------------------------------------------------------

    def checkpoint(self) -> Path:
        """Write a durable manifest of the current store state.

        Returns the manifest path. After this, a fresh `ProvenanceStore` for the
        same run_id/base_dir reads back identical data — the run survives a
        process restart.
        """
        manifest = {
            "run_id": self.run_id,
            "findings": self._count_lines(self.findings_path),
            "claims": self._count_lines(self.claims_path),
            "checkpointed_at": datetime.now(UTC).isoformat(),
        }
        self.checkpoint_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return self.checkpoint_path

    # --- internals ----------------------------------------------------------

    @staticmethod
    def _write_jsonl(path: Path, records: list[Finding] | list[Claim]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    @staticmethod
    def _count_lines(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
