# Development

## Install

Use an editable install for local work:

```bash
pip install -e ".[test,dev]"
```

The `doc` extra is intentionally empty in this fork because documentation is Markdown source under `docs/`.

## Test

Run the full test suite:

```bash
python -m backtesting.test
```

Run docs checks:

```bash
python -m unittest backtesting.test._test.TestDocs
```

Run lint and type checks:

```bash
flake8 backtesting setup.py
mypy --no-warn-unused-ignores backtesting
```

## Documentation

Docs are maintained by editing Markdown directly.

There is no documentation build step. Do not add a pdoc, Sphinx, MkDocs, Jupytext, or notebook execution requirement unless the docs policy is intentionally changed.

When behavior changes:

- API change: update `docs/api/`.
- User workflow change: update `docs/guides/`.
- Architecture change: update `docs/architecture.md`.
- Maintenance rule change: update `docs/agent-guide.md` or this file.

## Release Notes

Keep `CHANGELOG.md` focused on released user-facing changes. Prefer stable local links into `docs/` for new changelog entries.
