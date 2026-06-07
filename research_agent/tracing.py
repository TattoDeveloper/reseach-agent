"""Execution tracing — a LangGraph/LangSmith-style run tree.

Captures everything from the first input down to individual tool calls and LLM
responses, as a tree of **spans**:

    run
    └─ stage: collect
       └─ llm                         (one query() call = one collector agent)
          ├─ • text: "searching for…"
          ├─ tool: WebSearch          (inputs = query, outputs = results)
          └─ • text: "found 3 sources"

Two seams make this possible without touching stage logic:

- **`traced_query`** wraps the injectable `QueryFn` (the single seam every stage
  uses to reach the model). It records each `query()` call as an ``llm`` span and
  walks the streamed messages — `TextBlock`, `ThinkingBlock`, `ToolUseBlock`,
  `ToolResultBlock` — into child tool spans and text events.
- **`Tracer.span`** is an async context manager that nests via a `ContextVar`, so
  spans opened inside a stage (even across concurrent collector tasks) parent
  correctly.

Export the tree with `render()` (human-readable) or `to_dict()` / `save()`
(JSON, for an external UI).
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

QueryFn = Callable[..., AsyncIterator[Any]]

# The span currently open on this task — child spans read it as their parent.
# anyio/asyncio copy the context when a task is spawned, so concurrent collectors
# each inherit the correct parent stage.
_current_span: ContextVar[str | None] = ContextVar("current_span", default=None)

_PREVIEW_LEN = 160


@dataclass
class Span:
    """One node in the trace tree."""

    id: str
    name: str
    kind: str  # run | stage | llm | tool
    parent_id: str | None
    start: float
    inputs: Any = None
    outputs: Any = None
    error: str | None = None
    end: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        return None if self.end is None else (self.end - self.start) * 1000


class Tracer:
    """Collects spans and renders/serializes the run tree."""

    def __init__(self) -> None:
        self._spans: list[Span] = []

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    @asynccontextmanager
    async def span(
        self,
        name: str,
        kind: str,
        *,
        inputs: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[Span]:
        span = Span(
            id=uuid4().hex,
            name=name,
            kind=kind,
            parent_id=_current_span.get(),
            start=time.time(),
            inputs=inputs,
            metadata=dict(metadata or {}),
        )
        self._spans.append(span)
        token = _current_span.set(span.id)
        try:
            yield span
        except BaseException as exc:
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end = time.time()
            _current_span.reset(token)

    def add_child(self, parent: Span, name: str, kind: str, *, inputs: Any = None) -> Span:
        """Add a child span finalized manually (used for tool_use/tool_result)."""
        span = Span(
            id=uuid4().hex,
            name=name,
            kind=kind,
            parent_id=parent.id,
            start=time.time(),
            inputs=inputs,
        )
        self._spans.append(span)
        return span

    # --- export -------------------------------------------------------------

    def to_dict(self) -> list[dict[str, Any]]:
        """Nested tree of plain dicts (JSON-safe via str fallback at save time)."""
        nodes = {s.id: _span_to_node(s) for s in self._spans}
        roots: list[dict[str, Any]] = []
        for span in self._spans:
            node = nodes[span.id]
            parent = nodes.get(span.parent_id) if span.parent_id else None
            if parent is not None:
                parent["children"].append(node)
            else:
                roots.append(node)
        return roots

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return target

    def render(self) -> str:
        """Human-readable indented tree (LangGraph-console style)."""
        lines: list[str] = []
        for root in self.to_dict():
            _render_node(root, 0, lines)
        return "\n".join(lines)


def traced_query(tracer: Tracer, query_fn: QueryFn, *, name: str = "llm") -> QueryFn:
    """Wrap a QueryFn so each call records an llm span with tool/text children."""

    async def _wrapped(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        inputs: dict[str, Any] = {"prompt": kwargs.get("prompt")}
        options = kwargs.get("options")
        system_prompt = getattr(options, "system_prompt", None)
        if system_prompt:
            inputs["system_prompt"] = system_prompt

        async with tracer.span(name, "llm", inputs=inputs) as span:
            tool_spans: dict[str, Span] = {}
            texts: list[str] = []
            async for message in query_fn(*args, **kwargs):
                _record_message(tracer, span, message, tool_spans, texts)
                yield message
            if texts:
                span.outputs = {"text": "".join(texts)}

    return _wrapped


# --- internals --------------------------------------------------------------


def _record_message(
    tracer: Tracer,
    span: Span,
    message: Any,
    tool_spans: dict[str, Span],
    texts: list[str],
) -> None:
    for block in getattr(message, "content", None) or []:
        kind, data = _classify_block(block)
        if kind == "text":
            texts.append(data)
            span.events.append({"type": "text", "text": data})
        elif kind == "thinking":
            span.events.append({"type": "thinking", "text": data})
        elif kind == "tool_use":
            child = tracer.add_child(span, data["name"], "tool", inputs=data["input"])
            child.metadata["tool_use_id"] = data["id"]
            if data["id"] is not None:
                tool_spans[data["id"]] = child
        elif kind == "tool_result":
            tool_span = tool_spans.get(data["id"])
            if tool_span is not None:
                tool_span.outputs = data["content"]
                tool_span.end = time.time()

    usage = getattr(message, "usage", None)
    if usage is not None:
        span.metadata["usage"] = str(usage)


def _classify_block(block: Any) -> tuple[str, Any]:
    text = getattr(block, "text", None)
    if isinstance(text, str):
        return "text", text
    thinking = getattr(block, "thinking", None)
    if isinstance(thinking, str):
        return "thinking", thinking
    if hasattr(block, "name") and hasattr(block, "input"):
        return "tool_use", {
            "name": getattr(block, "name", "tool"),
            "input": getattr(block, "input", None),
            "id": getattr(block, "id", None),
        }
    tool_use_id = getattr(block, "tool_use_id", None)
    if tool_use_id is not None:
        return "tool_result", {"id": tool_use_id, "content": getattr(block, "content", None)}
    return "unknown", None


def _span_to_node(span: Span) -> dict[str, Any]:
    return {
        "name": span.name,
        "kind": span.kind,
        "duration_ms": span.duration_ms,
        "inputs": span.inputs,
        "outputs": span.outputs,
        "events": span.events,
        "error": span.error,
        "metadata": span.metadata,
        "children": [],
    }


_ICONS = {"run": "◆", "stage": "▸", "llm": "✦", "tool": "⚙"}


def _render_node(node: dict[str, Any], depth: int, lines: list[str]) -> None:
    indent = "  " * depth
    icon = _ICONS.get(node["kind"], "•")
    dur = "" if node["duration_ms"] is None else f" ({node['duration_ms']:.0f}ms)"
    lines.append(f"{indent}{icon} {node['kind']}: {node['name']}{dur}")
    if node["inputs"] is not None:
        lines.append(f"{indent}    in:  {_preview(node['inputs'])}")
    if node["outputs"] is not None:
        lines.append(f"{indent}    out: {_preview(node['outputs'])}")
    for event in node["events"]:
        lines.append(f"{indent}    • {event['type']}: {_preview(event.get('text', event))}")
    if node["error"]:
        lines.append(f"{indent}    ✗ {node['error']}")
    for child in node["children"]:
        _render_node(child, depth + 1, lines)


def _preview(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= _PREVIEW_LEN else text[: _PREVIEW_LEN - 1] + "…"
