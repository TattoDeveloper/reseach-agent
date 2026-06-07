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

## CI & pre-commit

The same gate (`ruff` + `mypy --strict` + `pytest`) runs in two places:

- **GitHub Actions** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) on
  every push and pull request, with the env synced `--frozen` to `uv.lock`.
- **Pre-commit** ([`.pre-commit-config.yaml`](.pre-commit-config.yaml)) before
  every local commit. Enable it once:

  ```bash
  uv run pre-commit install
  ```
