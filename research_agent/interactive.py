"""Human-facing research session (PLAN §0.5 / T8.2, optional).

This is the ONE place a stateful `ClaudeSDKClient` appears. The autonomous
pipeline (`orchestrator.run_research`) is built from stateless `query()` workers
so the orchestrator can enforce gates in code; but a *human-facing* session needs
bidirectional turns ("now dig into the EU"), conversation state, and the ability
to interrupt a long investigation — exactly what the stateful client provides and
`query()` does not.

The underlying client is injected (`client_factory`) so this wrapper is testable
without spawning a CLI subprocess.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from types import TracebackType
from typing import Any, Protocol

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

DEFAULT_SYSTEM_PROMPT = (
    "You are an interactive research assistant. Answer follow-up questions, dig "
    "deeper on request, and keep the conversation's context across turns."
)


class _Client(Protocol):
    """The slice of ClaudeSDKClient this wrapper uses (keeps it injectable)."""

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def query(self, prompt: str) -> None: ...
    def receive_response(self) -> AsyncIterator[Any]: ...
    async def interrupt(self) -> None: ...


ClientFactory = Callable[[ClaudeAgentOptions], _Client]


class InteractiveResearchSession:
    """A stateful, interruptible research conversation."""

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        options: ClaudeAgentOptions | None = None,
        client_factory: ClientFactory = ClaudeSDKClient,
    ) -> None:
        self._options = options or ClaudeAgentOptions(
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT
        )
        self._client_factory = client_factory
        self._client: _Client | None = None

    async def __aenter__(self) -> InteractiveResearchSession:
        self._client = self._client_factory(self._options)
        await self._client.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.disconnect()
            self._client = None

    async def ask(self, prompt: str) -> list[Any]:
        """Send a follow-up turn and collect the full response."""
        client = self._require_client()
        await client.query(prompt)
        return [message async for message in client.receive_response()]

    async def interrupt(self) -> None:
        """Interrupt a running investigation (only the stateful client can)."""
        await self._require_client().interrupt()

    def _require_client(self) -> _Client:
        if self._client is None:
            raise RuntimeError("session not started; use 'async with' to connect")
        return self._client
