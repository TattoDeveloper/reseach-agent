"""Tests for the execution tracer (run/stage/llm/tool tree)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import pytest

from research_agent.tracing import Tracer, traced_query
from tests.conftest import FakeMessage, FakeTextBlock, scripted_query


@dataclass
class FakeToolUse:
    name: str
    input: dict[str, Any]
    id: str


@dataclass
class FakeToolResult:
    tool_use_id: str
    content: str


@dataclass
class FakeBlockMessage:
    content: list[Any]


@pytest.mark.anyio
async def test_span_nesting_via_context() -> None:
    tracer = Tracer()
    async with tracer.span("run", "run"):
        async with tracer.span("stage", "stage"):
            pass

    spans = {s.name: s for s in tracer.spans}
    assert spans["stage"].parent_id == spans["run"].id
    assert spans["run"].parent_id is None
    assert spans["stage"].duration_ms is not None


@pytest.mark.anyio
async def test_span_records_error() -> None:
    tracer = Tracer()
    with pytest.raises(ValueError):
        async with tracer.span("boom", "stage"):
            raise ValueError("kaboom")

    assert "kaboom" in (tracer.spans[0].error or "")


@pytest.mark.anyio
async def test_traced_query_captures_text_tool_and_result() -> None:
    tracer = Tracer()
    messages = [
        FakeBlockMessage(content=[FakeTextBlock("searching now")]),
        FakeBlockMessage(content=[FakeToolUse("WebSearch", {"q": "rag"}, "t1")]),
        FakeBlockMessage(content=[FakeToolResult("t1", "3 results")]),
        FakeBlockMessage(content=[FakeTextBlock("done")]),
    ]
    wrapped = traced_query(tracer, scripted_query(messages))

    collected = [m async for m in wrapped(prompt="find RAG sources", options=None)]
    assert len(collected) == 4

    llm = next(s for s in tracer.spans if s.kind == "llm")
    assert llm.inputs["prompt"] == "find RAG sources"
    assert llm.outputs == {"text": "searching nowdone"}
    # text events recorded
    assert [e["type"] for e in llm.events] == ["text", "text"]

    tool = next(s for s in tracer.spans if s.kind == "tool")
    assert tool.name == "WebSearch"
    assert tool.inputs == {"q": "rag"}
    assert tool.outputs == "3 results"  # matched by tool_use_id
    assert tool.parent_id == llm.id


@pytest.mark.anyio
async def test_traced_query_yields_unchanged_messages() -> None:
    tracer = Tracer()
    wrapped = traced_query(tracer, scripted_query([FakeMessage.text("hello")]))
    out = [m async for m in wrapped(prompt="x", options=None)]
    assert out[0].content[0].text == "hello"


def test_render_and_to_dict_produce_tree() -> None:
    tracer = Tracer()

    async def build() -> None:
        async with tracer.span("run", "run", inputs={"request": "q"}):
            async with tracer.span("collect", "stage"):
                pass

    anyio.run(build)

    tree = tracer.to_dict()
    assert len(tree) == 1
    assert tree[0]["name"] == "run"
    assert tree[0]["children"][0]["name"] == "collect"

    rendered = tracer.render()
    assert "run: run" in rendered
    assert "stage: collect" in rendered


def test_save_writes_json(tmp_path: Path) -> None:
    tracer = Tracer()

    async def build() -> None:
        async with tracer.span("run", "run"):
            pass

    anyio.run(build)
    path = tracer.save(tmp_path / "trace.json")

    data = json.loads(path.read_text())
    assert data[0]["name"] == "run"
