"""Shared test harness.

T0.2 — mock the SDK at the ``query()`` boundary so every stage (collectors,
synthesis, verifier, report) is unit-testable without a live ``ANTHROPIC_API_KEY``.

The real ``claude_agent_sdk.query`` is an async generator:

    async for message in query(prompt=..., options=...):
        ...

``scripted_query`` below builds a drop-in replacement that yields a fixed list
of messages. Downstream tests inject it with, e.g.::

    monkeypatch.setattr("research_agent.collectors.query", scripted_query(msgs))
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class FakeTextBlock:
    """Minimal stand-in for an SDK text content block."""

    text: str


@dataclass
class FakeMessage:
    """Minimal stand-in for an SDK assistant message.

    Carries a ``content`` list of blocks, mirroring the real message shape
    closely enough for parsers that read ``msg.content[*].text``.
    """

    content: list[FakeTextBlock] = field(default_factory=list)

    @classmethod
    def text(cls, text: str) -> FakeMessage:
        return cls(content=[FakeTextBlock(text=text)])


def scripted_query(messages: list[Any]) -> Callable[..., AsyncIterator[Any]]:
    """Return a fake ``query`` that ignores its args and yields ``messages``.

    Matches the real signature ``query(prompt=..., options=...)`` loosely by
    accepting and discarding ``*args, **kwargs``.
    """

    async def _fake_query(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        for message in messages:
            yield message

    return _fake_query


@pytest.fixture
def make_query() -> Callable[[list[Any]], Callable[..., AsyncIterator[Any]]]:
    """Fixture exposing ``scripted_query`` as a factory."""

    return scripted_query
