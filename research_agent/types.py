"""Structured records — the floor everything else depends on (PLAN §3a/§4).

These TypedDicts are the *contracts* between pipeline stages. Two invariants
from `knowledge/` are encoded here and enforced downstream:

- **Provenance per finding** (`research-provenance-handoff.md`): a claim's truth
  and its *standing* (currency, rigor, representativeness) are separate facts.
  `Source` carries `published`/`version`/`peer_reviewed`/`sample` so synthesis can
  compute standing flags instead of guessing.
- **Citation-bound handoffs** (`subagent-handoff-attribution-fix.md`): every
  `Finding` records which `subagent` produced it and the exact `quote`, so
  provenance survives the handoff and stays distinguishable.

Records are plain TypedDicts (dicts at runtime). Data crossing the ``query()``
boundary is untrusted JSON, so it is validated into these types via
``validate_finding`` rather than blindly cast.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal, NotRequired, TypedDict, cast


class RequestType(StrEnum):
    """How a request is decomposed (PLAN §1)."""

    SIMPLE = "simple"  # linear collect → synthesize → verify → report
    COMPARATIVE = "comparative"  # axis decomposition + shared schema (§2b)
    EXPLORATORY = "exploratory"  # discovery pass first (§2a)


class Verdict(StrEnum):
    """Independent fact-checker's per-claim verdict (PLAN §5)."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    OVERSTATED = "overstated"


# Standing flags computed by synthesis from provenance (PLAN §4a). A closed set:
# adding a flag means adding the rule that raises it.
Flag = Literal["possibly-outdated", "non-peer-reviewed", "small-sample"]


class Sample(TypedDict):
    """Sample basis behind a source — decisive for representativeness."""

    n: int
    scope: str  # e.g. "single region: SE Asia"


class Source(TypedDict):
    """WHERE a finding came from, with everything needed to judge its standing.

    `published` and `version` are what make freshness re-validation (§7) and the
    `possibly-outdated` flag possible; capture them at the collect stage — the
    only stage that has them — because they cannot be reconstructed later.
    """

    doc_id: str
    title: str
    location: str  # url | file:line | page
    published: str  # ISO date — decisive on fast-changing topics
    version: str | None  # or content hash — enables freshness checks (§7)
    peer_reviewed: bool
    sample: Sample | None


class Finding(TypedDict):
    """A single observed fact bound to its evidence (PLAN §3a).

    This is the *only* shape collectors emit — never prose. The orchestrator
    forwards lists of these untouched (§3b); it must not re-summarize them.
    """

    claim: str  # what was observed
    source: Source  # WHERE — provenance rides along
    quote: str  # the exact supporting excerpt
    subagent: str  # who produced it (keeps provenance distinct)


class Claim(TypedDict):
    """A synthesized assertion bound to the findings that support it (PLAN §4a).

    `source_ids` is the citation contract: the synthesis gate (§4c) refuses to
    let report-gen run if any claim has an empty `source_ids`.
    """

    text: str
    source_ids: list[str]  # doc_ids of the supporting sources
    flags: list[Flag]  # standing caveats computed in synthesis


class SchemaSpec(TypedDict):
    """The shared comparison matrix for comparative requests (PLAN §2b).

    Pins every collector to the same axis-values × dimensions × question set so
    N parallel passes line up into one comparison instead of N mini-reports.
    """

    axis: str  # e.g. "sector"
    values: list[str]  # e.g. ["music_licensing", "model_training_data"]
    dimensions: dict[str, Any]  # e.g. {"jurisdictions": [...], "date_range": "..."}
    question_set: list[str]  # the fixed questions every cell must answer


class Plan(TypedDict):
    """The structured output of the plan/discovery stage (PLAN §2).

    The plan is an object, not prose — it becomes the contract the rest of the
    pipeline executes against. `schema` is present only for comparative requests.
    """

    request_type: RequestType
    sub_questions: list[str]
    schema: NotRequired[SchemaSpec]


class InvalidFindingError(ValueError):
    """Raised when untrusted data cannot be validated into a `Finding`."""


# Source fields without which a finding has no usable provenance (PLAN §3a).
REQUIRED_SOURCE_FIELDS: tuple[str, ...] = ("location", "published")
REQUIRED_FINDING_FIELDS: tuple[str, ...] = ("claim", "quote", "subagent")


def validate_finding(raw: Mapping[str, Any]) -> Finding:
    """Validate untrusted data (e.g. parsed JSON from a worker) into a `Finding`.

    This is the boundary that enforces the provenance invariant: a finding with
    no `source.location` or `source.published` is rejected here, not silently
    accepted and discovered missing at report time.

    Raises:
        InvalidFindingError: if any required field is missing or empty.
    """
    source = raw.get("source")
    if not isinstance(source, Mapping):
        raise InvalidFindingError("finding is missing a structured 'source'")
    for field in REQUIRED_SOURCE_FIELDS:
        if not source.get(field):
            raise InvalidFindingError(f"source is missing required field {field!r}")
    for field in REQUIRED_FINDING_FIELDS:
        if not raw.get(field):
            raise InvalidFindingError(f"finding is missing required field {field!r}")
    return cast(Finding, raw)
