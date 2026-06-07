"""Independent verification — fresh reasoner, raw sources (PLAN §5).

A single reasoner can't verify its own work; it carries the priors that shaped
the claim. So each claim is re-checked by a **structurally independent**
fact-checker (`independent-review-architecture.md` per the PLAN):

- a separate ``query()`` call — **no `resume`, no `session_id`** → genuinely
  fresh context. Statelessness *is* the independence; sharing a client would
  re-introduce the anchoring bias §5 exists to remove.
- the verifier sees **only the raw cited excerpts + the claim text** — never
  synthesis's reasoning or its standing flags. `build_verifier_prompt` is built
  solely from `Finding.quote`/`Source.location` so nothing from synthesis leaks.
- verdict per claim: ``supported | unsupported | overstated``.

Unsupported/overstated claims are dropped (or routed back for re-collection) by
the orchestrator — `keep_supported` is the filter; it never lets them reach the
report. The verifier **fails closed**: any unparseable/missing verdict is treated
as `unsupported` so a broken check can't launder a claim into the report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import anyio
from claude_agent_sdk import ClaudeAgentOptions, query

from research_agent.collectors import QueryFn
from research_agent.parsing import ParseError, extract_json_object, extract_text
from research_agent.types import Claim, Finding, Verdict

DEFAULT_MAX_CONCURRENCY = 8


class VerifierIndependenceError(RuntimeError):
    """Raised when verifier options would compromise fresh-context independence."""


@dataclass
class VerificationResult:
    """A claim paired with its independent verdict."""

    claim: Claim
    verdict: Verdict
    reason: str | None = None


def default_verifier_options() -> ClaudeAgentOptions:
    """Fresh-context options for the fact-checker (no resume/session_id)."""
    return ClaudeAgentOptions(
        allowed_tools=["Read", "WebFetch"],
        system_prompt=(
            "You are an independent fact-checker. Using ONLY the cited source "
            "excerpts you are given, judge whether they support the claim exactly:\n"
            "- 'supported': the excerpts state the claim.\n"
            "- 'overstated': the excerpts support a weaker version than the claim.\n"
            "- 'unsupported': the excerpts do not support the claim.\n"
            "Ignore any outside knowledge or prior reasoning. Reply with JSON: "
            '{"verdict": "supported|unsupported|overstated", "reason": str}.'
        ),
    )


def build_verifier_prompt(claim: Claim, supporting: Sequence[Finding]) -> str:
    """Build the prompt from raw excerpts + claim ONLY (no synthesis content)."""
    lines = [
        f"Claim: {claim['text']}",
        "",
        "Cited source excerpts (the ONLY evidence you may use):",
    ]
    for i, finding in enumerate(supporting, start=1):
        location = finding["source"]["location"]
        lines.append(f'[{i}] {location}: "{finding["quote"]}"')
    return "\n".join(lines)


def _assert_independent(options: ClaudeAgentOptions) -> None:
    """Enforce fresh context in code, even if a caller passes custom options."""
    if options.resume is not None or options.session_id is not None:
        raise VerifierIndependenceError(
            "verifier must run in fresh context: resume/session_id must be unset"
        )


async def verify_claim(
    claim: Claim,
    findings: Sequence[Finding],
    *,
    options: ClaudeAgentOptions | None = None,
    query_fn: QueryFn = query,
) -> VerificationResult:
    """Independently fact-check one claim against its cited raw excerpts."""
    opts = options if options is not None else default_verifier_options()
    _assert_independent(opts)

    cited = set(claim["source_ids"])
    supporting = [f for f in findings if f["source"]["doc_id"] in cited]
    if not supporting:
        # No evidence reached the verifier → it cannot be confirmed. Fail closed.
        return VerificationResult(
            claim, Verdict.UNSUPPORTED, "no cited source excerpts available"
        )

    prompt = build_verifier_prompt(claim, supporting)
    messages: list[object] = []
    async for message in query_fn(prompt=prompt, options=opts):
        messages.append(message)

    verdict, reason = _parse_verdict(extract_text(messages))
    return VerificationResult(claim, verdict, reason)


async def verify_all(
    claims: Sequence[Claim],
    findings: Sequence[Finding],
    *,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    options: ClaudeAgentOptions | None = None,
    query_fn: QueryFn = query,
) -> list[VerificationResult]:
    """Verify every claim, each in its own independent fresh ``query()`` call."""
    limiter = anyio.Semaphore(max_concurrency)
    results: list[VerificationResult | None] = [None] * len(claims)

    async def _run(index: int, claim: Claim) -> None:
        async with limiter:
            results[index] = await verify_claim(
                claim, findings, options=options, query_fn=query_fn
            )

    async with anyio.create_task_group() as task_group:
        for index, claim in enumerate(claims):
            task_group.start_soon(_run, index, claim)

    return [result for result in results if result is not None]


def keep_supported(results: Sequence[VerificationResult]) -> list[Claim]:
    """The report-bound filter: only ``supported`` claims survive (PLAN §5)."""
    return [r.claim for r in results if r.verdict is Verdict.SUPPORTED]


# --- internals --------------------------------------------------------------


def _parse_verdict(text: str) -> tuple[Verdict, str | None]:
    if not text.strip():
        return Verdict.UNSUPPORTED, "verifier returned no content"
    try:
        payload = extract_json_object(text)
    except ParseError:
        return Verdict.UNSUPPORTED, "verifier output was not parseable"

    raw = str(payload.get("verdict", "")).strip().lower()
    try:
        verdict = Verdict(raw)
    except ValueError:
        return Verdict.UNSUPPORTED, f"verifier returned unknown verdict {raw!r}"

    reason = payload.get("reason")
    return verdict, str(reason) if reason is not None else None
