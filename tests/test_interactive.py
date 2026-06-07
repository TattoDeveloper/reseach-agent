"""Tests for the interactive session wrapper (T8.2).

Uses a fake client so no CLI subprocess is spawned.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from research_agent.interactive import InteractiveResearchSession
from tests.conftest import FakeMessage


class _FakeClient:
    def __init__(self, options: Any) -> None:
        self.options = options
        self.events: list[str] = []
        self.prompts: list[str] = []

    async def connect(self) -> None:
        self.events.append("connect")

    async def disconnect(self) -> None:
        self.events.append("disconnect")

    async def query(self, prompt: str) -> None:
        self.prompts.append(prompt)

    async def receive_response(self) -> AsyncIterator[Any]:
        yield FakeMessage.text("answer")

    async def interrupt(self) -> None:
        self.events.append("interrupt")


@pytest.mark.anyio
async def test_session_connects_asks_and_disconnects() -> None:
    created: list[_FakeClient] = []

    def factory(options: Any) -> _FakeClient:
        client = _FakeClient(options)
        created.append(client)
        return client

    async with InteractiveResearchSession(client_factory=factory) as session:
        messages = await session.ask("now dig into the EU")

    client = created[0]
    assert client.events[0] == "connect"
    assert client.prompts == ["now dig into the EU"]
    assert [m.content[0].text for m in messages] == ["answer"]
    assert client.events[-1] == "disconnect"  # cleaned up on exit


@pytest.mark.anyio
async def test_session_can_interrupt() -> None:
    created: list[_FakeClient] = []

    def factory(options: Any) -> _FakeClient:
        client = _FakeClient(options)
        created.append(client)
        return client

    async with InteractiveResearchSession(client_factory=factory) as session:
        await session.interrupt()

    assert "interrupt" in created[0].events


@pytest.mark.anyio
async def test_ask_before_connect_raises() -> None:
    session = InteractiveResearchSession(client_factory=_FakeClient)
    with pytest.raises(RuntimeError, match="not started"):
        await session.ask("hello")


def test_default_system_prompt_is_set() -> None:
    session = InteractiveResearchSession()
    assert session._options.system_prompt is not None
