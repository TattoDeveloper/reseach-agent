# research-agent

A general-purpose multi-agent research system. Architecture in
[`PLAN-general-research-agent.md`](PLAN-general-research-agent.md); task
breakdown in [`IMPLEMENTATION-PLAN.md`](IMPLEMENTATION-PLAN.md).

## Development

This project is managed with [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev                       # reproduce env from uv.lock
uv run pytest                             # run tests
uv run ruff check . && uv run mypy        # lint + type-check
```

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` to run against the
live API. Unit tests mock the SDK and need no key.

## Running it

The system drives the installed `claude` CLI via the SDK, using its auth (your
`ANTHROPIC_API_KEY` from `.env`/env, or your existing `claude` login):

```bash
uv run research-agent "What are the main approaches to retrieval-augmented generation?"
# or:  uv run python -m research_agent "..."
```

The cited markdown report is written to a file — by default
`runs/<run-id>/report.md`, alongside the structured `findings.jsonl` /
`claims.jsonl` / `checkpoint.json`. Stage progress streams to stderr as it runs.

Options:
- `-o` / `--output PATH` — write the report to a specific path.
- `--run-id` / `--runs-dir` — where the provenance store + report are written.
- `--title` — report title. `-q` / `--quiet` — suppress progress output.

## CI & pre-commit

The same gate (`ruff` + `mypy --strict` + `pytest`) runs in two places:

- **GitHub Actions** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) on
  every push and pull request, with the env synced `--frozen` to `uv.lock`.
- **Pre-commit** ([`.pre-commit-config.yaml`](.pre-commit-config.yaml)) before
  every local commit. Enable it once:

  ```bash
  uv run pre-commit install
  ```
