"""Tests for the independent verifier (T4.1).

Acceptance criteria from IMPLEMENTATION-PLAN.md:
- catches a deliberately overstated claim (verdict == overstated);
- the verifier call carries no session/resume args;
- no synthesis content (e.g. standing flags) leaks into the prompt;
- fail-closed on unparseable output; keep_supported filters correctly.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from research_agent.types import Claim, Finding, Source, Verdict
from research_agent.verifier import (
    VerifierIndependenceError,
    build_verifier_prompt,
    default_verifier_options,
    keep_supported,
    verify_all,
    verify_claim,
)
from tests.conftest import FakeMessage, scripted_query


def _finding(doc_id: str = "d1", quote: str = "grew 5% in one region") -> Finding:
    return Finding(
        claim="X grew 5%",
        source=Source(
            doc_id=doc_id,
            title="Doc",
            location="page 3",
            published="2024-01-01",
            version="v1",
            peer_reviewed=True,
            sample={"n": 100, "scope": "global"},
        ),
        quote=quote,
        subagent="collector:a",
    )


def _claim(text: str = "X grew 5%", source_ids: list[str] | None = None) -> Claim:
    return Claim(text=text, source_ids=source_ids or ["d1"], flags=[])


def _verdict_message(verdict: str, reason: str = "") -> FakeMessage:
    return FakeMessage.text(json.dumps({"verdict": verdict, "reason": reason}))


@pytest.mark.anyio
async def test_catches_overstated_claim() -> None:
    # Source supports "grew 5% in one region"; claim says "grew 40% globally".
    findings = [_finding(quote="grew 5% in one region")]
    claim = _claim(text="X grew 40% globally")
    msg = _verdict_message("overstated", "source says 5% in one region, not 40% globally")

    result = await verify_claim(claim, findings, query_fn=scripted_query([msg]))

    assert result.verdict is Verdict.OVERSTATED


@pytest.mark.anyio
async def test_supported_claim_passes() -> None:
    result = await verify_claim(
        _claim(), [_finding()], query_fn=scripted_query([_verdict_message("supported")])
    )
    assert result.verdict is Verdict.SUPPORTED


@pytest.mark.anyio
async def test_verifier_call_has_no_resume_or_session_id() -> None:
    captured: dict[str, Any] = {}

    def recording_query(*_args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        captured["options"] = kwargs.get("options")

        async def _gen() -> AsyncIterator[Any]:
            yield _verdict_message("supported")

        return _gen()

    await verify_claim(_claim(), [_finding()], query_fn=recording_query)

    options = captured["options"]
    assert options.resume is None
    assert options.session_id is None


def test_default_options_are_fresh_context() -> None:
    options = default_verifier_options()
    assert options.resume is None
    assert options.session_id is None


@pytest.mark.anyio
async def test_rejects_options_with_resume() -> None:
    bad = ClaudeAgentOptions(resume="prior-session")
    with pytest.raises(VerifierIndependenceError, match="fresh context"):
        await verify_claim(_claim(), [_finding()], options=bad)


def test_prompt_contains_evidence_but_not_synthesis_flags() -> None:
    # A claim carrying a synthesis-computed flag must not leak that flag to the
    # independent verifier — it sees only the raw excerpt + claim text.
    claim = Claim(text="X grew 5%", source_ids=["d1"], flags=["non-peer-reviewed"])
    prompt = build_verifier_prompt(claim, [_finding(quote="grew 5% in one region")])

    assert "X grew 5%" in prompt
    assert "grew 5% in one region" in prompt
    assert "non-peer-reviewed" not in prompt


@pytest.mark.anyio
async def test_unsupported_when_no_cited_excerpts_without_calling_model() -> None:
    def exploding_query(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        raise AssertionError("verifier must not call the model with no evidence")

    claim = _claim(source_ids=["missing"])
    result = await verify_claim(claim, [_finding("d1")], query_fn=exploding_query)

    assert result.verdict is Verdict.UNSUPPORTED


@pytest.mark.anyio
async def test_fails_closed_on_unparseable_output() -> None:
    result = await verify_claim(
        _claim(), [_finding()], query_fn=scripted_query([FakeMessage.text("garbage")])
    )
    assert result.verdict is Verdict.UNSUPPORTED


@pytest.mark.anyio
async def test_fails_closed_on_unknown_verdict() -> None:
    result = await verify_claim(
        _claim(), [_finding()], query_fn=scripted_query([_verdict_message("maybe")])
    )
    assert result.verdict is Verdict.UNSUPPORTED


@pytest.mark.anyio
async def test_verify_all_and_keep_supported() -> None:
    claims = [_claim(text="a"), _claim(text="b"), _claim(text="c")]

    def per_claim_query(*_args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        prompt = kwargs.get("prompt", "")
        verdict = "supported" if "Claim: a" in prompt else "overstated"

        async def _gen() -> AsyncIterator[Any]:
            yield _verdict_message(verdict)

        return _gen()

    results = await verify_all(claims, [_finding()], query_fn=per_claim_query)

    assert len(results) == 3
    supported = keep_supported(results)
    assert [c["text"] for c in supported] == ["a"]
