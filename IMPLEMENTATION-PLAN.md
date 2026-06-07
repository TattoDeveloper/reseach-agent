# Implementation Plan: General-Purpose Multi-Agent Research System

Derived from `PLAN-general-research-agent.md`. This breaks the architecture into
discrete, testable tasks with explicit dependencies and acceptance criteria.

**Guiding constraint (from the plan):** structured records carry signal;
gates are enforced in code, never in coordinator prose. Every task below is
sequenced so that provenance and gates exist *before* the code that depends on
them.

---

## ⚠️ Knowledge-base invariants — read `knowledge/*.md` before implementing any stage

Each architecture decision traces to a documented failure + fix in `knowledge/`.
Treat the **"Fix" / "Rule"** section of each note as the binding spec; the prose
elsewhere is motivation. Below: the invariant, the test that proves it, and — for
the SDK-level traps — the **real-API translation** (the notes use illustrative
pseudo-code that will *not* run as written).

### Pseudo-API → real `claude-agent-sdk` 0.2.93 (verified by introspection)

The notes use helpers that **do not exist in the SDK**. Do not copy them literally.

| In the notes (illustrative) | Reality | What to actually do |
|---|---|---|
| `run_subagent("synthesis", inputs=...)` | ❌ no such function | Each stage is **our own async function** that calls `query(prompt=..., options=...)`. The orchestrator sequences them in Python. |
| `output_schema=Finding` on a subagent | ❌ no such param | Instruct the worker to emit **JSON**, then parse + validate it into the TypedDict at the `query()` boundary. The SDK does not enforce schemas. |
| `client.query(...)` coordinator that "fans out then reconciles" via `system_prompt` | ✅ `ClaudeSDKClient` is real, but… | …letting the **model** sequence sub-steps is exactly the `multi-agent-synthesis-dependency-fix` bug. Use the note's `system_prompt` only as the *per-worker* instruction; do the fan-out / gate / reconcile in **orchestrator code**. |
| `fork_session()` | ✅ real: `fork_session(session_id, directory=None, up_to_message_id=None, title=None)` | Usable directly for §8 parallel-analysis forking. |

Verified real surface: `query` (keyword-only `prompt`, yields
`AssistantMessage | UserMessage | SystemMessage | ResultMessage | …`),
`ClaudeSDKClient`, `ClaudeAgentOptions`, `AgentDefinition`, `fork_session`,
`get_session_messages`, `list_sessions`, `tool`, `create_sdk_mcp_server`.
`AssistantMessage.content` is a list of blocks; `TextBlock.text` holds text —
which is why the T0.2 mock mirrors that shape.

### Invariant table (note → code rule → enforcing test)

| Note | Binding invariant | Enforced in | Test |
|---|---|---|---|
| `subagent-handoff-attribution-fix` | **No prose handoffs.** Every inter-stage payload is a typed list of records with embedded provenance + a `subagent` field; the orchestrator forwards the **structure untouched**, never a re-summary. | T2.2, T8.1 | assert collectors return `Finding[]` (not str); assert orchestrator passes the list object through, not a summarized string |
| `research-provenance-handoff` | Provenance (`location/page`, `published`, `peer_reviewed`, `sample`) is **captured at the collect stage** (the only stage that has it) and bound per-claim. Standing **flags computed in synthesis**, not vibes. | T1.1, T2.2, T3.1 | reject a `Finding` missing `source.location`/`published`; stale date → `possibly-outdated`; `n<MIN_N` → `small-sample` |
| `multi-agent-synthesis-dependency-fix` | `collect → synthesize → report` is a **machine-checked gate**. Report-gen receives **only synthesis output**, never raw findings, so it can't invent citations. | T3.2, T8.1 | gate **raises** when status≠ok or any claim lacks `source_ids`; report-gen rejects raw `Finding` input |
| `comparative-research-axis-decomposition` | Decompose on the **comparison axis up front** (one scoped pass per axis-value, pinned to a shared matrix schema, equal depth, no skipped cells); **reconcile down the columns before writing** — never split at write time. | T2.2, T3.1, T6.2 | unfilled cell → explicit "no rule found", never dropped; cross-axis conflict surfaced in synthesis |
| `independent-review-architecture` *(plan §5; note file absent — rely on plan summary)* | Verifier is a **separate `query()` with no `resume`/`session_id`**, fed **raw cited source + claim only**, never synthesis reasoning. | T4.1 | assert verifier call carries no session/resume args and no synthesis text; catches a deliberately overstated claim |
| `research-stale-session-revalidation` | A saved session is **frozen evidence**. On resume: diff each finding's `version` vs source-of-record, re-collect only the stale branch, **rebuild** (never reuse) derived synthesis, continue from a clean checkpoint — never wholesale-`resume`. | T7.1 | stale version detected + refreshed; synthesis rebuilt, not reused |
| `resumed-session-stale-context-refresh` | If resuming a live client session, **name the exact changed surfaces** and force a re-read/diff before continuing. | T7.2 | invalidation list names changed surfaces; guard against blind "continue" |

### Notes referenced by the PLAN but **not present** in `knowledge/`

These back PLAN sections whose detailed fix-note is missing — implement from the
PLAN summary and flag for review; don't assume extra detail exists:
`broad-codebase-question-coverage` (§2a discovery), `independent-review-architecture`
(§5 verifier), `stale-session-context-fix` (§8 checkpoint/new-session),
`fork-session-parallel-analysis` (§8 fork), `agentic-loop-tool-result-fix`
(§8 full tool round-trip). **Ask the user for these files before building §2a, §5,
and the §8 hygiene helpers** if exact behavior matters.

---

## Stack assumptions

- Language: **Python 3.11+** (TypedDict, `anyio`).
- Env & packaging: **`uv`** — a `uv`-managed virtual env (`.venv`) and
  `uv`-resolved lockfile drive all install/run/test commands.
- SDK: **`claude-agent-sdk`** (`query`, `ClaudeSDKClient`, `ClaudeAgentOptions`).
- Concurrency: **`anyio`** (`Semaphore` for fan-out bounding).
- Default worker model: **`claude-sonnet-4-6`**; coordinator/synthesis reasoning:
  **`claude-opus-4-8`**. (Adjust per cost/latency budget.)
- Tests: **`pytest`** + **`pytest-anyio`**; LLM calls mocked at the `query()`
  boundary for deterministic unit tests.

### uv workflow (used throughout)

```bash
uv init --package research-agent     # or: uv init  (scaffolds pyproject.toml)
uv venv                              # create .venv (Python 3.11+)
uv add claude-agent-sdk anyio        # runtime deps → pyproject + uv.lock
uv add --dev pytest pytest-anyio ruff mypy
uv sync                              # reproduce env from uv.lock
uv run pytest                        # run anything inside the env
uv run ruff check . && uv run mypy research_agent
```

> Convention: every command in this plan runs via `uv run ...`; never invoke a
> global `python`/`pip`. The `.venv` and `uv.lock` are the reproducible env.

---

## Phase 0 — Project scaffolding

### T0.1 — Toolchain & package skeleton
- **Deliverables:** `uv`-managed project: `pyproject.toml` + `uv.lock` (deps via
  `uv add`: `claude-agent-sdk`, `anyio`, `typing-extensions`; `uv add --dev`:
  `pytest`, `pytest-anyio`, `ruff`, `mypy`); `.venv/` created by `uv venv`;
  `research_agent/` package with empty modules matching §9 layout; `tests/` dir;
  `.python-version` pinning 3.11+; `.env.example` for `ANTHROPIC_API_KEY`;
  `.gitignore` covering `.venv/` and `.env`.
- **Acceptance:** `uv sync` reproduces the env from `uv.lock`; `uv run pytest`
  runs (0 tests); `uv run ruff check .` and `uv run mypy research_agent` pass on
  the empty skeleton.
- **Depends on:** —

### T0.2 — Test harness for SDK mocking
- **Deliverables:** `tests/conftest.py` with a fixture that patches `query()` to
  yield scripted messages, so every downstream stage is unit-testable without a
  live API key.
- **Acceptance:** a smoke test asserts the mock yields canned messages.
- **Depends on:** T0.1

---

## Phase 1 — Structured records (the floor)

### T1.1 — `types.py`: Source, Finding, Claim, Plan
- **Deliverables:** TypedDicts from §3a/§4 — `Source` (doc_id, title, location,
  published, version, peer_reviewed, sample), `Finding` (claim, source, quote,
  subagent), `Claim` (text, source_ids, flags), `Plan`/`SchemaSpec` (axis,
  values, dimensions, question_set). Add enums: `RequestType`
  (simple/comparative/exploratory), `Verdict` (supported/unsupported/overstated).
- **Acceptance:** `mypy` clean; a construction test builds one of each record;
  a `validate_finding()` helper rejects a Finding missing `source.location` or
  `source.published`.
- **Depends on:** T0.1
- **Plan ref:** §3a, §4, §10.1

---

## Phase 2 — Provenance store + collection

### T2.1 — `store.py`: provenance store + checkpointing
- **Deliverables:** disk-backed store (JSON/JSONL under a run dir) that persists
  Findings and Claims keyed by `doc_id`/run; `save_findings`, `load_findings`,
  `checkpoint(run_id)`. Treat the SDK session JSONL as disposable; the store is
  the source of truth (§8).
- **Acceptance:** round-trip test (write → read → equal); checkpoint writes a
  file that survives process restart in test.
- **Depends on:** T1.1
- **Plan ref:** §7, §8, §10.2

### T2.2 — `collectors.py`: scoped subagent → Finding[]
- **Deliverables:** `collect_one(subq, opts)` runs a single stateless `query()`
  call scoped to one sub-question/axis-value and parses results into `Finding`
  records (with full `Source` provenance). `collect_all(subqs)` fans out behind
  an `anyio.Semaphore(8)` (§0.5 caveat). Each cell must be filled or explicitly
  marked `"no rule found"` (§3c).
- **Acceptance:** with mocked `query()`, `collect_all` returns one Finding per
  sub-question; every Finding has `source.location` + `published`; concurrency
  is capped (assert max in-flight ≤ semaphore limit); a missing-answer case
  produces an explicit "no rule found" marker, not a dropped cell.
- **Depends on:** T1.1, T2.1
- **Plan ref:** §3, §3a, §3c, §0.5, §10.2

---

## Phase 3 — Synthesis with the hard gate

### T3.1 — `synthesis.py`: Findings → cited Claims + standing flags
- **Deliverables:** `synthesize(findings) -> SynthesisResult{status, claims}`.
  Consumes the Finding list **untouched** (no coordinator re-summarization, §3b),
  emits `Claim`s bound to `source_ids`, and computes standing flags:
  `possibly-outdated` (stale vs topic half-life), `non-peer-reviewed`,
  `small-sample` (n < MIN_N). For comparative runs, reconcile down the columns
  per shared dimension (§4b).
- **Acceptance:** flag-computation unit tests (stale date → flag; small n →
  flag); a comparative fixture surfaces a cross-axis conflict.
- **Depends on:** T1.1, T2.2
- **Plan ref:** §4a, §4b, §3b, §10.3

### T3.2 — Dependency gate (machine-checked)
- **Deliverables:** `assert_synthesis_complete(result)` raising
  `SynthesisIncompleteError` unless `status == "ok"` AND every claim has
  non-empty `source_ids`. Wire so report-gen cannot be called otherwise.
- **Acceptance:** **test that it raises** when a claim lacks `source_ids` and
  when status != ok; passes when both hold.
- **Depends on:** T3.1
- **Plan ref:** §4c, §10.3

---

## Phase 4 — Independent verification

### T4.1 — `verifier.py`: fresh-context fact-checker
- **Deliverables:** `verify_claim(claim, sources) -> Verdict` via a **separate
  `query()` call with no `resume`/`session_id`** (structural independence). It
  receives only the raw cited source excerpt + the claim text — never synthesis
  reasoning. `verify_all(claims, store)` returns verdicts; unsupported/overstated
  claims are dropped or routed back for re-collection.
- **Acceptance:** **test that a deliberately overstated claim is caught**
  (verdict `overstated`); assert the verifier call carries no session/resume
  args and no synthesis text in its prompt.
- **Depends on:** T1.1, T2.1, T3.1
- **Plan ref:** §5, §10.4

---

## Phase 5 — Report

### T5.1 — `report.py`: render verified Claims only
- **Deliverables:** `render(verified_claims) -> str/markdown`. Input is verified
  Claims only — never raw Finding blobs. Every citation includes page + date;
  standing flags become explicit caveats ("based on a 2021 single-region vendor
  survey, n=40").
- **Acceptance:** **test citations carry page + date**; a flagged claim renders
  its caveat; passing a raw Finding (not a Claim) is rejected/typed out.
- **Depends on:** T3.2, T4.1
- **Plan ref:** §6, §10.5

---

## Phase 6 — Intake & planning

### T6.1 — `intake.py`: classify + axis detection
- **Deliverables:** `classify(request) -> RequestType` and axis detection
  (sectors × jurisdictions × dates) feeding the schema builder.
- **Acceptance:** fixtures route simple/comparative/exploratory correctly; a
  comparative request yields detected axes.
- **Depends on:** T1.1
- **Plan ref:** §1, §10.6

### T6.2 — `planner.py`: discovery pass + shared-schema builder
- **Deliverables:** `discover(topic) -> sub_questions[]` (read-only enumeration,
  no answers — §2a) and `build_schema(axes, dims) -> SchemaSpec` (§2b) producing
  the structured matrix contract every collector is pinned to.
- **Acceptance:** discovery returns enumerated sub-questions without answers;
  `build_schema` emits a SchemaSpec; collectors pinned to it cover every cell.
- **Depends on:** T1.1, T6.1
- **Plan ref:** §2a, §2b, §10.6

---

## Phase 7 — Freshness & session hygiene

### T7.1 — `store.py`: freshness diff on resume
- **Deliverables:** `find_stale(session) -> Finding[]` comparing each finding's
  `source.version` against the source-of-record; `revalidate(session)` re-runs
  collection for stale findings and **rebuilds** synthesis from fresh inputs
  (derived findings can't outlive their inputs). Never wholesale-`resume` a
  contaminated transcript.
- **Acceptance:** a stale-version fixture is detected and refreshed; synthesis is
  rebuilt (not reused) after refresh.
- **Depends on:** T2.1, T3.1
- **Plan ref:** §7, §10.7

### T7.2 — Session-hygiene helpers
- **Deliverables:** helpers for: resume + selective invalidation (name changed
  surfaces, force re-read); new-session-after-structural-change (vetted summary
  as `system_prompt`); `fork_session()` for parallel independent analyses;
  full-tool-round-trip preservation if any loop is hand-rolled (prefer SDK loop).
- **Acceptance:** unit tests for the invalidation list and fork-vs-continue
  selection logic; a guard test that hand-rolled history appends the full
  assistant `content` + matching `tool_result`, never a text-only summary.
- **Depends on:** T2.1
- **Plan ref:** §8, §10.7

---

## Phase 8 — Orchestration

### T8.1 — `orchestrator.py`: wire stages, enforce gates in code
- **Deliverables:** end-to-end flow intake → plan/discover → collect → synthesize
  → verify → report, with all gates enforced programmatically (synthesis-before-
  report, verify-before-report, freshness-before-resume). The LLM plans/reasons
  but cannot skip a prerequisite.
- **Acceptance:** **fault-injection tests** assert each gate fires:
  (a) stale source → revalidation triggered;
  (b) failed/incomplete synthesis → report blocked;
  (c) unsupported claim → excluded from report.
  Plus one happy-path integration run (mocked SDK) producing a cited report.
- **Depends on:** all prior phases
- **Plan ref:** §9, §4c, §10.8

### T8.2 — Human-facing session entrypoint (optional, post-MVP)
- **Deliverables:** a `ClaudeSDKClient`-based interactive entrypoint for
  follow-ups / interrupts / resume — the *only* place the stateful client
  appears (§0.5). Workers stay stateless `query()` calls.
- **Acceptance:** a follow-up turn ("now dig into EU") reuses session state; an
  interrupt cancels an in-flight investigation.
- **Depends on:** T8.1
- **Plan ref:** §0.5, §8

---

## Critical path & parallelization

```
T0.1 → T0.2
T0.1 → T1.1 ─┬─ T2.1 ─┬─ T2.2 ─ T3.1 ─ T3.2 ─┐
             │        └─ T7.1                 ├─ T5.1 ─ T8.1 → T8.2
             ├─ T4.1 ──────────────────────────┘
             └─ T6.1 ─ T6.2
                  T7.2 (parallel, needs T2.1)
```

- **Critical path:** T1.1 → T2.1 → T2.2 → T3.1 → T3.2 → T5.1 → T8.1.
- **Parallelizable once T1.1 lands:** intake/planner (T6.x), verifier (T4.1),
  session hygiene (T7.2).

## Build-order note vs. the plan

This follows §10's order with two adjustments:
1. **Phase 0 scaffolding** is added before the floor — toolchain + SDK mock
   harness make every later task testable without live API calls.
2. **Intake/planner (T6.x)** is sequenced after the verify/report spine (matching
   §10.6) but can begin in parallel as soon as `types.py` exists.

## Definition of done (whole system)

- Every Finding carries `source.location` + `published`; no prose handoffs.
- Synthesis gate and verify gate are code-enforced and covered by raising tests.
- Report renders only verified Claims with page + date and standing caveats.
- Fault-injection suite (stale source, failed synthesis, unsupported claim) is
  green.
- `uv run ruff`, `uv run mypy`, and `uv run pytest` all pass in CI (env
  reproduced via `uv sync --frozen`).
