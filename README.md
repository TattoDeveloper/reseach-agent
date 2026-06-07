# research-agent

A general-purpose multi-agent research system. Architecture in
[`PLAN-general-research-agent.md`](PLAN-general-research-agent.md); task
breakdown in [`IMPLEMENTATION-PLAN.md`](IMPLEMENTATION-PLAN.md).

## Development

This project is managed with [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                                   # reproduce env from uv.lock
uv run pytest                             # run tests
uv run ruff check . && uv run mypy        # lint + type-check
```

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` to run against the
live API. Unit tests mock the SDK and need no key.
