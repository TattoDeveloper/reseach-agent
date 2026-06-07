"""Shared helpers for reading structured output back from a ``query()`` worker.

The SDK yields a stream of messages; an `AssistantMessage` carries a list of
content blocks, and a `TextBlock` holds text. Workers in this system are
instructed to return JSON, so these helpers concatenate the text and parse it,
tolerating prose or code-fences a model may wrap around the JSON.

Kept provider-agnostic and dependency-free so both `collectors` and `synthesis`
parse worker output the same way.
"""

from __future__ import annotations

import json
from typing import Any


class ParseError(ValueError):
    """Raised when a worker's output cannot be parsed as the expected JSON."""


def extract_text(messages: list[Any]) -> str:
    """Concatenate every text block across messages (real SDK or mocked)."""
    parts: list[str] = []
    for message in messages:
        content = getattr(message, "content", None)
        if not content:
            continue
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the JSON object a worker returned, tolerating surrounding prose."""
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            raise ParseError("worker did not return a JSON object") from None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ParseError("worker output was not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ParseError("worker JSON was not an object")
    return parsed
