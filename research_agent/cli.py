"""Command-line entrypoint: ``research-agent "your question"`` (T-extra).

Thin wrapper over `orchestrator.run_research` so the system is runnable without
writing a script. The pipeline defaults to the real stages, so this makes live
model calls — it loads `.env` (for `ANTHROPIC_API_KEY`) and drives the installed
`claude` CLI via the SDK.

The core (`research`) is separated from `main` (argparse + dotenv + error
handling) so it can be tested with an injected pipeline and no live calls.
"""

from __future__ import annotations

import argparse
import functools
import sys

import anyio
from dotenv import load_dotenv

from research_agent.orchestrator import Pipeline, run_research
from research_agent.store import ProvenanceStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="Run a multi-agent research query and print a cited report.",
    )
    parser.add_argument("request", help="the research question to investigate")
    parser.add_argument(
        "--run-id",
        default="default",
        help="run identifier; provenance is stored under <runs-dir>/<run-id>/",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="directory for the provenance store (default: runs)",
    )
    parser.add_argument(
        "--title",
        default="Research Report",
        help="title for the rendered report",
    )
    return parser


async def research(
    request: str,
    *,
    run_id: str = "default",
    runs_dir: str = "runs",
    title: str = "Research Report",
    pipeline: Pipeline | None = None,
) -> str:
    """Run the pipeline and return the rendered report (testable core)."""
    store = ProvenanceStore(run_id, base_dir=runs_dir)
    result = await run_research(request, store=store, pipeline=pipeline, title=title)
    return result.report


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    load_dotenv()  # pick up ANTHROPIC_API_KEY from .env if present
    args = build_parser().parse_args(argv)

    try:
        report = anyio.run(
            functools.partial(
                research,
                args.request,
                run_id=args.run_id,
                runs_dir=args.runs_dir,
                title=args.title,
            )
        )
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure cleanly
        print(f"research failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(report)
    return 0
