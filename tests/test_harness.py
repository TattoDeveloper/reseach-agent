"""Smoke test for the SDK mock harness (T0.2)."""

from __future__ import annotations

import pytest

from tests.conftest import FakeMessage, scripted_query


@pytest.mark.anyio
async def test_scripted_query_yields_canned_messages() -> None:
    messages = [FakeMessage.text("hello"), FakeMessage.text("world")]
    fake_query = scripted_query(messages)

    collected = [msg async for msg in fake_query(prompt="ignored", options=None)]

    assert [m.content[0].text for m in collected] == ["hello", "world"]


@pytest.mark.anyio
async def test_scripted_query_ignores_call_args() -> None:
    fake_query = scripted_query([FakeMessage.text("ok")])

    # Any args/kwargs are accepted and discarded.
    collected = [msg async for msg in fake_query("positional", foo="bar")]

    assert len(collected) == 1
