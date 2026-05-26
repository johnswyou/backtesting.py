# Backtesting.py Fork Documentation

This directory is the documentation source of truth for this fork.

The documentation policy is intentionally simple:

- Durable documentation is written as Markdown under `docs/`.
- The docs are committed source files, not generated HTML.
- Notebooks and scripts can exist as runnable examples, but they are not the canonical documentation.
- Public behavior changes should update the matching page in `docs/api/` or `docs/guides/`.
- Architecture changes should update `docs/architecture.md` or add a decision note under `docs/decisions/`.

## Start Here

- [Agent Guide](agent-guide.md): how to navigate and modify this repo efficiently.
- [Project Map](project-map.md): what each important source file owns.
- [Architecture](architecture.md): runtime model, strategy lifecycle, orders, stats, and portfolio flow.
- [Development](development.md): setup, tests, linting, and docs maintenance.

## API Reference

- [API Overview](api/README.md)
- [Backtest](api/backtest.md)
- [PortfolioBacktest](api/portfolio-backtest.md)
- [Strategy](api/strategy.md)
- [Orders, Trades, and Positions](api/orders-trades.md)
- [Statistics](api/stats.md)
- [Plotting](api/plotting.md)
- [Library Helpers](api/lib.md)

## Guides

- [Quick Start](guides/quick-start.md)
- [Optimization](guides/optimization.md)
- [Multi-Asset Portfolio](guides/multi-asset-portfolio.md)
- [Leveraged Pairs Trading](guides/leveraged-pairs-trading.md)
- [Multiple Time Frames](guides/multiple-timeframes.md)
- [Strategy Library](guides/strategy-library.md)
- [Machine Learning](guides/machine-learning.md)

## Reference Notes

- [Examples](examples/README.md)
- [Alternatives](alternatives.md)
- [Decisions](decisions/README.md)

## Documentation Workflow

Edit Markdown directly. There is no pdoc, Jupytext, notebook execution, or GitHub Pages build step required for documentation changes.

Before submitting a documentation change, run:

```bash
python -m unittest backtesting.test._test.TestDocs
```

For behavior changes, also run the normal test command:

```bash
python -m backtesting.test
```
