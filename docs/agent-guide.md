# Agent Guide

This file is optimized for coding agents. Read it before making broad changes.

## Primary Rules

- Treat `docs/` as the documentation source of truth.
- Do not resurrect the old `doc/` generated documentation workflow.
- Prefer local source and tests over assumptions from upstream.
- Keep public API documentation in sync with `backtesting/backtesting.py`, `backtesting/lib.py`, `backtesting/_stats.py`, and `backtesting/_plotting.py`.
- Keep docs concrete: include accepted inputs, returned objects, important edge cases, and paths to source/tests.

## High-Value Files

- `backtesting/backtesting.py`: core engine, `Strategy`, `Backtest`, `PortfolioBacktest`, `Order`, `Trade`, `Position`.
- `backtesting/lib.py`: reusable helpers, strategy base classes, `FractionalBacktest`, `MultiBacktest`.
- `backtesting/_stats.py`: stats series and equity/trade calculations.
- `backtesting/_plotting.py`: Bokeh output and plotting.
- `backtesting/_util.py`: array/data wrappers, multi-asset data access, shared memory helpers.
- `backtesting/test/_test.py`: main behavior test suite.
- `docs/`: Markdown documentation.

## Public API Surface

The normal import path is:

```python
from backtesting import Backtest, PortfolioBacktest, Strategy
from backtesting.lib import crossover, resample_apply
```

Treat the following as public unless a change explicitly says otherwise:

- `Backtest`
- `PortfolioBacktest`
- `Strategy`
- `Order`
- `Trade`
- `Position`
- helpers exported from `backtesting.lib`
- result fields returned by `Backtest.run()` and `PortfolioBacktest.run()`

Internals usually start with `_` and should not be documented as user-facing API unless they explain architecture.

## Common Change Checklist

For a new public parameter:

- Update the implementation.
- Add or update tests in `backtesting/test/_test.py`.
- Update the relevant `docs/api/*.md` page.
- Add guide coverage if the parameter changes a normal user workflow.

For portfolio behavior:

- Check `PortfolioBacktest` docs.
- Check `Strategy` docs for multi-symbol data and position access.
- Check `Statistics` docs if result fields or trade tables change.
- Prefer tests that cover multiple symbols and misaligned indexes.

For stats behavior:

- Update `docs/api/stats.md`.
- Keep the README stats example accurate if result keys change.
- Run the docs tests because they assert stats keys are documented.

## Commands

Install locally:

```bash
pip install -e ".[test,dev]"
```

Run tests:

```bash
python -m backtesting.test
```

Run docs checks only:

```bash
python -m unittest backtesting.test._test.TestDocs
```

Run lint/type checks:

```bash
flake8 backtesting setup.py
mypy --no-warn-unused-ignores backtesting
```

## Documentation Style

Write docs for maintenance, not marketing.

- Name the source file that owns the behavior.
- Name the test area that proves the behavior.
- Include small code examples.
- Document warnings and edge cases.
- Prefer stable concepts over generated method listings.
- Link within `docs/` using relative Markdown links.

Avoid:

- Generated HTML.
- Hosted upstream docs links.
- Notebook synchronization requirements.
- Long narrative examples that obscure the API contract.
