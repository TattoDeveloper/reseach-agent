"""Tests for the CLI entrypoint (no live model calls)."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_agent import cli
from research_agent.orchestrator import ReportBlockedError, ResearchResult
from tests.test_orchestrator import make_pipeline


def test_parser_reads_request_and_options() -> None:
    args = cli.build_parser().parse_args(
        ["What is X?", "--run-id", "r1", "--runs-dir", "out", "--title", "T"]
    )
    assert args.request == "What is X?"
    assert args.run_id == "r1"
    assert args.runs_dir == "out"
    assert args.title == "T"


@pytest.mark.anyio
async def test_research_core_returns_result_with_report_file(tmp_path: Path) -> None:
    result = await cli.research(
        "how big is X?",
        run_id="demo",
        runs_dir=str(tmp_path),
        pipeline=make_pipeline(),
    )
    assert "X grew 40%" in result.report
    assert result.report_path is not None
    assert result.report_path.exists()
    assert "## Sources" in result.report_path.read_text()


def test_main_writes_report_file_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_run_research(request: str, **_kwargs: object) -> ResearchResult:
        return ResearchResult(report="# Done\nbody", findings=[], claims=[], verifications=[])

    monkeypatch.setattr(cli, "run_research", fake_run_research)
    out = tmp_path / "report.md"

    code = cli.main(["a question", "--runs-dir", str(tmp_path), "--output", str(out)])

    assert code == 0
    assert out.read_text() == "# Done\nbody"
    assert "Report written to" in capsys.readouterr().err


def test_main_reports_failure_and_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_run_research(request: str, **_kwargs: object) -> ResearchResult:
        raise ReportBlockedError("no claims survived verification")

    monkeypatch.setattr(cli, "run_research", failing_run_research)

    code = cli.main(["a question", "--runs-dir", str(tmp_path)])

    assert code == 1
    assert "research failed" in capsys.readouterr().err
