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
from pathlib import Path

import anyio
from dotenv import load_dotenv

from research_agent.orchestrator import (
    Pipeline,
    ProgressFn,
    ResearchResult,
    run_research,
)
from research_agent.store import ProvenanceStore
from research_agent.tracing import Tracer


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
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="path to write the markdown report "
        "(default: <runs-dir>/<run-id>/report.md)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="record an execution trace (run/stage/llm/tool tree) and print it; "
        "also saved as <runs-dir>/<run-id>/trace.json",
    )
    parser.add_argument(
        "--trace-output",
        default=None,
        help="path to write the trace JSON (default: <runs-dir>/<run-id>/trace.json); "
        "implies --trace",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress stage progress output on stderr",
    )
    return parser


async def research(
    request: str,
    *,
    run_id: str = "default",
    runs_dir: str = "runs",
    title: str = "Research Report",
    pipeline: Pipeline | None = None,
    progress: ProgressFn | None = None,
    tracer: Tracer | None = None,
) -> ResearchResult:
    """Run the pipeline and return the full result (testable core)."""
    store = ProvenanceStore(run_id, base_dir=runs_dir)
    return await run_research(
        request,
        store=store,
        pipeline=pipeline,
        title=title,
        progress=progress,
        tracer=tracer,
    )


def _stderr_progress(message: str) -> None:
    print(f"» {message}", file=sys.stderr, flush=True)


def _format_error(exc: BaseException) -> str:
    """Flatten an ExceptionGroup into readable leaf errors (Python 3.11+)."""
    if isinstance(exc, BaseExceptionGroup):
        inner = "; ".join(_format_error(e) for e in exc.exceptions)
        return f"{type(exc).__name__}({inner})"
    return f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    load_dotenv()  # pick up ANTHROPIC_API_KEY from .env if present
    args = build_parser().parse_args(argv)
    progress = None if args.quiet else _stderr_progress
    tracer = Tracer() if (args.trace or args.trace_output) else None

    try:
        result = anyio.run(
            functools.partial(
                research,
                args.request,
                run_id=args.run_id,
                runs_dir=args.runs_dir,
                title=args.title,
                progress=progress,
                tracer=tracer,
            )
        )
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure cleanly
        print(f"research failed: {_format_error(exc)}", file=sys.stderr)
        if tracer is not None:
            _write_trace(tracer, args)  # persist the partial trace for debugging
        return 1

    output = (
        Path(args.output)
        if args.output
        else (result.report_path or Path(args.runs_dir) / args.run_id / "report.md")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.report, encoding="utf-8")
    print(f"Report written to {output}", file=sys.stderr)

    if tracer is not None:
        trace_path = _write_trace(tracer, args)
        print("\n--- trace ---", file=sys.stderr)
        print(tracer.render(), file=sys.stderr)
        print(f"Trace written to {trace_path}", file=sys.stderr)
    return 0


def _write_trace(tracer: Tracer, args: argparse.Namespace) -> Path:
    path = (
        Path(args.trace_output)
        if args.trace_output
        else Path(args.runs_dir) / args.run_id / "trace.json"
    )
    return tracer.save(path)
