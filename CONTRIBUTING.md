# Contributing

## Prerequisites

- Python 3.14+
- `uv` (<https://docs.astral.sh/uv/getting-started/installation/>)

## Setup

From repo root:

```bash
uv sync
uv run pre-commit install
```

This is the same on Linux, macOS, and Windows.

## Linters, Formatters, Type-Checkers

This project uses several tools that ensure good code quality and consistency.

Run all hooks:

```bash
uv run pre-commit run --all-files
```

Or run individual tools directly:

```bash
uv run ruff check .
uv run ruff format .
uv run basedpyright
uvx ty check
```

It is recommended that you use `pre-commit` though, which you should also install as a git hook that will make it run before every commit you try to make:

```bash
uv run pre-commit install
```

## Notes

- Pre-commit is the source of truth for CI-style checks.
- The hook config intentionally uses `uv run` and `uvx` entries.
