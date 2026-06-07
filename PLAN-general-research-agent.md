# Build Plan: A General-Purpose Multi-Agent Research System

A single design that folds in every fix from the research `.md` notes. The
guiding principle across all of them is the same:

> **Capture critical signal in structured form, enforce critical constraints in
> code — never trust the coordinator's prose or a resumed transcript to do it.**

Each numbered practice below cites the note it comes from.

---

## 0. Architecture at a glance

```
request
  │
  ▼
[1] INTAKE & CLASSIFY ──────────── request type + comparison axes (if any)
  │
  ▼
[2] PLAN / DISCOVERY ───────────── enumerate sub-questions  →  shared schema
  │                                (read-only; no answers yet)
  ▼
[3] COLLECT (parallel fan-out) ─── one scoped subagent per sub-question/axis,
  │                                each returns STRUCTURED Findings (+provenance)
  ▼
[4] SYNTHESIZE ─────────────────── operate on Findings only; emit cited Claims;
  │                                compute freshness/quality flags
  │   ◀── gate: synthesis must succeed + every claim has source_ids
  ▼
[5] VERIFY (independent) ───────── fresh-context fact-checker; raw sources only;
  │                                no exposure to synthesis reasoning
  │   ◀── gate: report blocked unless claims verified
  ▼
[6] REPORT ─────────────────────── renders verified Claims only; cites page+date
```

Cross-cutting: a **provenance store**, **freshness re-validation** on resume,
and **session hygiene** wrap the whole pipeline.

---

## 0.5 `query()` vs `ClaudeSDKClient` — pick the right primitive per stage

They are **not competitors** — they are two primitives with different
semantics (confirmed in the SDK's own `query.py` docstring). Production
guidance: **keep durable state in your provenance store, make subagents
stateless `query()` workers, and reach for `ClaudeSDKClient` only at the
human-facing or fork/resume boundary.**

| | `query()` | `ClaudeSDKClient` |
|---|---|---|
| State | Stateless, each call independent | Stateful, persistent connection |
| Direction | Unidirectional (send-all → receive-all) | Bidirectional (send based on responses) |
| Lifecycle | Fire-and-forget, no connection mgmt | Connect / converse / disconnect |
| Interrupts & follow-ups | No | Yes |
| Concurrency | Each call = isolated subprocess | One conversation per client |

### Where each applies better in THIS pipeline

| Stage | Primitive | Why |
|---|---|---|
| [3] Collect (parallel fan-out) | **`query()`** | Each call is an isolated subprocess → true parallelism, zero cross-contamination, equal-depth passes |
| [4] Synthesize | **`query()`** | Pure-ish `findings → claims` worker; statelessness is what lets the orchestrator enforce the gate in code |
| [5] Verify (independent) | **`query()`** (no `resume`/`session_id`) | Fresh isolated context is *required* for independence — the whole point of `independent-review-architecture.md` |
| [6] Report | **`query()`** | One-shot render of verified claims; no conversation needed |
| Autonomous coordinator (request in → report out) | **`query()` calls driven by your orchestrator** | State lives in *your* store, not a transcript — avoids stale-context hazards entirely |
| Human-facing research session (follow-ups, "now dig into EU") | **`ClaudeSDKClient`** | Needs bidirectional turns + conversation state |
| Interruptible long investigation | **`ClaudeSDKClient`** | Only the client can interrupt a running query |
| Resume / fork an accumulated session | **`ClaudeSDKClient`** | `resume`/`fork_session()` operate on a live client session (§7, §8) |

### Why `query()` is the right default for workers

- **Independence** — a stateless call gives the verifier a genuinely fresh
  context; doing it inside a shared client would re-introduce the anchoring
  bias §5 exists to remove. *(`independent-review-architecture.md`)*
- **Parallelism** — subprocess-per-call fans out cleanly; a client is one
  conversation and fights concurrency.
- **Code-enforced gates** — stateless workers behave like functions, so the
  orchestrator sequences them (§4c) instead of trusting a long conversation.
- **No stale-context leak** — workers can't accumulate old `tool_result`
  blocks or drop a tool round-trip; the SDK loop owns that per call.
  *(`stale-session-context-fix.md`, `agentic-loop-tool-result-fix.md`)*

### Production caveat: cap concurrency on the fan-out

`query()` spawns **one subprocess per call**, so a broad request fanning out to
many collectors can exhaust file descriptors or hit rate limits. Wrap the
fan-out in a concurrency limiter:

```python
import anyio

limiter = anyio.Semaphore(8)            # cap concurrent CLI subprocesses

async def collect_one(subq):
    async with limiter:                 # bound the fan-out
        results = []
        async for msg in query(prompt=subq.prompt, options=collector_opts):
            results.append(msg)
        return to_findings(results)
```

> Rule: subagents = stateless `query()` workers behind a semaphore; the
> stateful `ClaudeSDKClient` appears only where interactivity, interrupts, or
> session continuity are genuinely required.

---

## 1. Intake & classification

Classify the request before doing anything — the decomposition strategy depends
on it.

- **Simple/factual** → linear collect → synthesize → verify → report.
- **Comparative / multi-dimensional** → axis decomposition (§2b).
  *(from `comparative-research-axis-decomposition.md`)*
- **Open-ended exploratory** → discovery pass first (§2a).

Detect comparison axes here (sectors × jurisdictions × dates, etc.) so the next
stage can pin a shared schema.

---

## 2. Plan / discovery — *enumerate before you decompose*

### 2a. Discovery pass (read-only)
*(from `broad-codebase-question-coverage.md`)*

Before partitioning work, run a discovery subagent whose only job is to
**enumerate the sub-questions / entry points** the topic actually has — not to
answer them. This prevents the classic failure where the decomposition bakes in
the answer's shape and the collectors can only confirm it.

> Rule: for any broad "how does X work / what's the landscape of X" question,
> enumerate the surface first; coverage must track reality, not an assumption.

### 2b. Build the shared schema (comparative requests)
*(from `comparative-research-axis-decomposition.md`)*

For comparative work, the coordinator builds a **matrix spec** (axis-values ×
shared dimensions × fixed question set) and pins every collector to it, so the
N parallel passes line up into a genuine comparison instead of N mini-reports.

```python
SCHEMA = {
    "axis": "sector",
    "values": ["music_licensing", "model_training_data", "independent_film"],
    "dimensions": {"jurisdictions": ["US", "EU"], "date_range": "2023-2025"},
    "question_set": ["what rule applies", "effective date", "penalty regime"],
}
```

The plan is itself a **structured object**, not prose — it becomes the contract
the rest of the pipeline executes against.

---

## 3. Collection — parallel, scoped, structured handoffs

One subagent **per sub-question / axis-value**, run in parallel for equal depth.
Each is scoped to its slice and pinned to the shared question set.

### 3a. Every finding is a structured, provenance-bound record
*(from `subagent-handoff-attribution-fix.md` + `research-provenance-handoff.md`)*

No prose handoffs. A claim's *truth* and its *standing* are separate facts;
prose preserves the first and silently drops the second.

```python
class Finding(TypedDict):
    claim: str                  # what was observed
    source: "Source"            # WHERE — see below
    quote: str                  # the exact supporting excerpt
    subagent: str               # who produced it (keeps provenance distinct)

class Source(TypedDict):
    doc_id: str
    title: str
    location: str               # url | file:line | page
    published: str              # ISO date — decisive on fast-changing topics
    version: str | None         # or hash — enables freshness checks (§7)
    peer_reviewed: bool
    sample: dict | None         # {"n": 40, "scope": "single region"}
```

### 3b. Pass structure through untouched
*(from `subagent-handoff-attribution-fix.md`)*

The coordinator forwards the **list of Findings**, never its own prose summary
of them. Re-summarizing at the coordinator reintroduces exactly the loss the
schema prevents.

### 3c. Equal depth, no skipped cells
*(from `comparative-research-axis-decomposition.md`)*

Each collector must fill every assigned cell or explicitly mark it "no rule
found" — depth can't drift to whatever the sources happen to favor.

---

## 4. Synthesis — *gated, citation-producing*

### 4a. Operate only on structured Findings, emit cited Claims

Synthesis consumes the Finding list and produces `Claim` objects that bind each
assertion to `source_ids`. It also computes **standing flags** from provenance:

```python
for c in claims:
    if stale(c.source.published, topic_half_life): c.flags.append("possibly-outdated")
    if not c.source.peer_reviewed:                 c.flags.append("non-peer-reviewed")
    if c.source.sample and c.source.sample["n"] < MIN_N: c.flags.append("small-sample")
```
*(from `research-provenance-handoff.md`)*

### 4b. Reconcile across the matrix (comparative)
*(from `comparative-research-axis-decomposition.md`)*

Synthesis works *down the columns*, not just within an axis-value, surfacing and
resolving conflicting conclusions for the same dimension before anything is
written.

### 4c. Enforce synthesis as a hard prerequisite — in code
*(from `multi-agent-synthesis-dependency-fix.md`)*

The dependency `collect → synthesize → report` is a **machine-checked gate**,
not a coordinator instruction. Report generation cannot start until synthesis
returns `ok` AND every claim carries `source_ids`.

```python
synthesis = await run_subagent("synthesis", inputs=findings)
if synthesis.status != "ok" or any(not c.source_ids for c in synthesis.claims):
    raise SynthesisIncompleteError("report blocked: unverified/missing citations")
```

---

## 5. Independent verification — *fresh reasoner, raw sources*

*(from `independent-review-architecture.md`)*

A single reasoner can't verify its own work — it carries the priors that shaped
the claim. Add a **structurally independent** fact-checker:

- separate `query()` call — **no `resume`, no `session_id`** → fresh context;
- receives the **raw cited sources + the claim**, never synthesis's reasoning;
- verdict per claim: `supported | unsupported | overstated`.

```python
async def verify_claim(claim, sources):           # fresh subprocess each call
    async for msg in query(
        f"Does this source support the claim, exactly? Claim: {claim.text}\n"
        f"Source excerpt: {sources[claim.source_ids[0]].quote}",
        ClaudeAgentOptions(
            system_prompt="You are a fact-checker. Confirm support strictly; "
                          "flag any overstatement. Ignore everything else.",
            tools=["Read", "WebFetch"],
        ),
    ): ...
```

Unsupported/overstated claims are dropped or routed back for re-collection —
not passed to the report.

---

## 6. Report generation — *renders verified claims only*

*(from `multi-agent-synthesis-dependency-fix.md` + `research-provenance-handoff.md`)*

Report-gen receives **only verified Claims** (never raw Finding blobs), so it
has nothing to "summarize" into invented citations. Each citation includes page
+ date because that metadata rode along the whole way. Standing flags become
explicit caveats ("based on a 2021 single-region vendor survey, n=40").

---

## 7. Cross-cutting: freshness re-validation on resume

*(from `research-stale-session-revalidation.md`)*

A saved session is **frozen evidence, not live evidence**. Before continuing a
saved investigation:

```python
stale = [f for f in session.findings
         if source_of_record(f.source.doc_id).version != f.source.version]

refreshed = [await run_subagent("collect", f.source.doc_id) for f in stale]
fresh = [f for f in session.findings if f not in stale] + refreshed

synthesis = await run_subagent("synthesis", inputs=fresh)   # REBUILD — derived
                                                            # findings can't outlive inputs
```

Never `resume` the contaminated transcript wholesale — stale `tool_result`
blocks re-enter context as trusted evidence. This is **only possible because §3a
captured `version`/`published` per finding.**

---

## 8. Cross-cutting: session hygiene & history integrity

- **Disposable transcripts; checkpoint to files.** Persist the structured
  Finding/Claim store to disk; treat the session JSONL as throwaway.
  *(from `stale-session-context-fix.md`)*
- **Resume + selectively invalidate.** When resuming, name the exact surfaces
  that changed and force a re-read/diff before continuing.
  *(from `resumed-session-stale-context-refresh.md`)*
- **New session after structural change.** If the topic's framing (not just
  details) shifted, start fresh with a vetted summary as `system_prompt`.
  *(from `stale-session-context-fix.md`)*
- **Fork at a checkpoint for parallel independent analyses.** When two analyses
  must start from the same accumulated context without cross-contamination,
  `fork_session()` — don't continue-in-one (contaminating) or summarize-restart
  (lossy). *(from `fork-session-parallel-analysis.md`)*
- **Preserve the full tool round-trip.** If you hand-roll any loop, append the
  complete assistant `content` array + the matching `tool_result` user message —
  never a text-only summary. Prefer the SDK's loop, which handles this.
  *(from `agentic-loop-tool-result-fix.md`)*

---

## 9. Suggested module layout

```
research_agent/
  types.py            # Finding, Source, Claim, Plan, schemas (TypedDict)
  intake.py           # §1 classify + axis detection
  planner.py          # §2 discovery pass + shared-schema builder
  collectors.py       # §3 parallel scoped subagents → Finding[]
  synthesis.py        # §4 Findings → Claims + flags + dependency gate
  verifier.py         # §5 independent fresh-context fact-check
  report.py           # §6 render verified Claims
  store.py            # §7/§8 provenance store, freshness diff, checkpointing
  orchestrator.py     # wires stages; enforces gates IN CODE, not prompts
```

The **orchestrator enforces every gate programmatically** (synthesis-before-
report, verify-before-report, freshness-before-resume). The coordinator LLM
plans and reasons; it never gets to *skip a prerequisite*.

---

## 10. Build order (incremental, each step testable)

1. `types.py` — the structured records. Everything else depends on provenance
   existing, so this is the floor.
2. `collectors.py` + `store.py` — get structured Findings with provenance
   landing on disk. Test: every finding has `source.location` + `published`.
3. `synthesis.py` with the **dependency gate** — test it *raises* when a claim
   lacks `source_ids`.
4. `verifier.py` — independent `query()`, no shared session. Test: it catches a
   deliberately overstated claim.
5. `report.py` — renders only verified claims; test citations carry page+date.
6. `planner.py` + `intake.py` — discovery pass and comparative axis schema.
7. `store.py` freshness diff + session-hygiene helpers (resume/fork/refresh).
8. `orchestrator.py` — wire it together; assert gates fire under fault injection
   (stale source, failed synthesis, unsupported claim).

---

## Practice-to-note traceability

| # | Practice | Note |
|---|---|---|
| 2a | Discovery before decomposition | `broad-codebase-question-coverage.md` |
| 2b/4b | Axis decomposition + shared schema + reconcile | `comparative-research-axis-decomposition.md` |
| 3a | Provenance per finding | `research-provenance-handoff.md` |
| 3a/3b | Structured citation-bound handoffs; no re-summarize | `subagent-handoff-attribution-fix.md` |
| 4c | Dependency gate in code | `multi-agent-synthesis-dependency-fix.md` |
| 5 | Independent verifier, fresh context | `independent-review-architecture.md` |
| 7 | Freshness re-validation on resume | `research-stale-session-revalidation.md` |
| 8 | New session after refactor; checkpoint | `stale-session-context-fix.md` |
| 8 | Resume + selective invalidation | `resumed-session-stale-context-refresh.md` |
| 8 | Fork for parallel independent analysis | `fork-session-parallel-analysis.md` |
| 8 | Full tool round-trip in history | `agentic-loop-tool-result-fix.md` |
