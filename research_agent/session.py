"""Session hygiene & history integrity (PLAN §8, T7.2).

A resumed session restores Claude's *memory* of external state, not the state
itself. These helpers encode the §8 rules:

- **Resume + selectively invalidate** (`resumed-session-stale-context-refresh.md`):
  name the exact surfaces that changed and force a re-read/diff before
  continuing — `build_refresh_instruction`.
- **Know when to start fresh instead** (same note): `should_start_fresh`.
- **New session after a structural change**: start fresh with a vetted summary as
  the `system_prompt` — `fresh_session_options`.
- **Fork at a checkpoint for parallel independent analyses**
  (PLAN §8; fork against the real SDK `fork_session`): `fork_for_parallel_analysis`
  — don't continue-in-one (contaminating) or summarize-restart (lossy).
- **Preserve the full tool round-trip** (PLAN §8): if a loop is hand-rolled,
  `assert_complete_tool_roundtrip` rejects a history where a `tool_use` has no
  matching `tool_result` — i.e. a text-only summary replaced the round-trip.

NOTE: the fork / new-session / tool-round-trip notes are not present in
`knowledge/`; these are implemented from the PLAN summary against the real SDK
and should be reviewed if exact behavior matters.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, fork_session


class ToolRoundTripError(RuntimeError):
    """Raised when a hand-rolled history drops a tool_use/tool_result round-trip."""


def build_refresh_instruction(changed_surfaces: Sequence[str]) -> str:
    """Resume prompt that names changed surfaces and forces a re-read/diff.

    Precise naming lets Claude refresh only what actually changed instead of
    trusting all prior context or re-reading everything.
    """
    if not changed_surfaces:
        raise ValueError("name at least one changed surface to refresh")
    lines = ["Before continuing, these changed since we paused:"]
    for i, surface in enumerate(changed_surfaces, start=1):
        lines.append(f"{i}. {surface} — re-read it and diff against your prior understanding.")
    lines.append(
        "List what changed, flag any prior conclusions that no longer hold, "
        "then continue."
    )
    return "\n".join(lines)


def should_start_fresh(
    *,
    invalidates_core_hypothesis: bool,
    changes_enumerable: bool,
    mostly_irrelevant_now: bool,
) -> bool:
    """Decide resume+refresh vs. a fresh session (`resumed-session...` note).

    Start fresh when the framing (not just details) shifted, when you can't
    enumerate what changed, or when most of the prior session is now irrelevant.
    """
    return (
        invalidates_core_hypothesis
        or not changes_enumerable
        or mostly_irrelevant_now
    )


def fresh_session_options(vetted_summary: str, **overrides: Any) -> ClaudeAgentOptions:
    """New-session options carrying a vetted summary as system_prompt.

    Used after a structural change: rather than resuming a contaminated
    transcript, start clean with a curated summary and NO resume/session_id.
    """
    if not vetted_summary.strip():
        raise ValueError("a vetted summary is required to seed a fresh session")
    return ClaudeAgentOptions(system_prompt=vetted_summary, **overrides)


def fork_for_parallel_analysis(
    session_id: str,
    labels: Sequence[str],
    *,
    fork_fn: Callable[..., Any] = fork_session,
) -> dict[str, str]:
    """Fork an accumulated session into independent branches, one per label.

    Forking gives each analysis the same checkpoint without cross-contamination
    (unlike continue-in-one) and without losing context (unlike summarize-restart).
    Returns {label: forked_session_id}.
    """
    if not labels:
        raise ValueError("provide at least one label to fork for")
    forks: dict[str, str] = {}
    for label in labels:
        result = fork_fn(session_id, title=label)
        forks[label] = result.session_id
    return forks


def assert_complete_tool_roundtrip(history: Sequence[dict[str, Any]]) -> None:
    """Reject a hand-rolled history that dropped a tool_use/tool_result pair.

    For every `tool_use` block in an assistant turn there must be a matching
    `tool_result` (by id) later in the history. A text-only summary standing in
    for the round-trip is exactly the bug `agentic-loop-tool-result-fix` warns
    about. Prefer the SDK's loop, which handles this; use this guard if you must
    hand-roll one.
    """
    tool_use_ids: list[str] = []
    result_ids: set[str] = set()
    for message in history:
        for block in message.get("content", []):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_use" and "id" in block:
                tool_use_ids.append(str(block["id"]))
            elif block_type == "tool_result" and "tool_use_id" in block:
                result_ids.add(str(block["tool_use_id"]))

    missing = [tid for tid in tool_use_ids if tid not in result_ids]
    if missing:
        raise ToolRoundTripError(
            f"tool_use blocks without a matching tool_result: {missing}"
        )
