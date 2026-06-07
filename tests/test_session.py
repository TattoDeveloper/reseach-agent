"""Tests for session-hygiene helpers (T7.2)."""

from __future__ import annotations

from typing import Any

import pytest

from research_agent.session import (
    ToolRoundTripError,
    assert_complete_tool_roundtrip,
    build_refresh_instruction,
    fork_for_parallel_analysis,
    fresh_session_options,
    should_start_fresh,
)


def test_refresh_instruction_names_each_surface_and_asks_for_diff() -> None:
    out = build_refresh_instruction(["refund_policy.md thresholds", "process_refund schema"])
    assert "refund_policy.md thresholds" in out
    assert "process_refund schema" in out
    assert "diff" in out


def test_refresh_instruction_requires_a_surface() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_refresh_instruction([])


def test_should_start_fresh_when_core_hypothesis_invalidated() -> None:
    assert should_start_fresh(
        invalidates_core_hypothesis=True,
        changes_enumerable=True,
        mostly_irrelevant_now=False,
    )


def test_should_start_fresh_when_changes_not_enumerable() -> None:
    assert should_start_fresh(
        invalidates_core_hypothesis=False,
        changes_enumerable=False,
        mostly_irrelevant_now=False,
    )


def test_resume_and_refresh_when_changes_are_small_and_known() -> None:
    assert not should_start_fresh(
        invalidates_core_hypothesis=False,
        changes_enumerable=True,
        mostly_irrelevant_now=False,
    )


def test_fresh_session_options_carries_summary_and_no_resume() -> None:
    options = fresh_session_options("vetted summary of the investigation")
    assert options.system_prompt == "vetted summary of the investigation"
    assert options.resume is None
    assert options.session_id is None


def test_fresh_session_options_requires_summary() -> None:
    with pytest.raises(ValueError, match="vetted summary"):
        fresh_session_options("   ")


def test_fork_for_parallel_analysis_creates_independent_branches() -> None:
    calls: list[dict[str, Any]] = []

    class _Result:
        def __init__(self, session_id: str) -> None:
            self.session_id = session_id

    def fake_fork(session_id: str, *, title: str) -> _Result:
        calls.append({"session_id": session_id, "title": title})
        return _Result(f"{session_id}-{title}")

    forks = fork_for_parallel_analysis("base", ["bull", "bear"], fork_fn=fake_fork)

    assert forks == {"bull": "base-bull", "bear": "base-bear"}
    assert len(calls) == 2  # one independent fork per analysis


def test_fork_requires_labels() -> None:
    with pytest.raises(ValueError, match="at least one label"):
        fork_for_parallel_analysis("base", [], fork_fn=lambda *a, **k: None)


def test_tool_roundtrip_passes_when_complete() -> None:
    history = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1"}]},
    ]
    assert_complete_tool_roundtrip(history)  # no raise


def test_tool_roundtrip_raises_when_result_dropped() -> None:
    # assistant called a tool, but a text-only summary replaced the tool_result.
    history = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1"}]},
        {"role": "user", "content": [{"type": "text", "text": "summary of the result"}]},
    ]
    with pytest.raises(ToolRoundTripError, match="t1"):
        assert_complete_tool_roundtrip(history)
