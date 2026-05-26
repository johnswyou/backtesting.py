# Project Map

This map describes the repository as maintained in this fork.

## Root Files

- `README.md`: package overview, installation, minimal usage, and links into `docs/`.
- `CHANGELOG.md`: release history inherited from upstream and fork changes.
- `CONTRIBUTING.md`: development workflow and documentation policy.
- `setup.py`: package metadata, dependencies, and extras.
- `pyproject.toml`: ruff configuration.
- `setup.cfg`: flake8, mypy, and coverage configuration.

## Runtime Package

- `backtesting/__init__.py`: top-level imports and package-level help text.
- `backtesting/backtesting.py`: core simulation engine.
- `backtesting/lib.py`: helper functions and reusable strategy classes.
- `backtesting/_stats.py`: performance statistics and stats series construction.
- `backtesting/_plotting.py`: Bokeh plotting helpers.
- `backtesting/_util.py`: data containers, indicators, shared memory, and utility helpers.
- `backtesting/autoscale_cb.js`: JavaScript callback used by Bokeh plots.

## Tests

- `backtesting/test/_test.py`: main unit and regression tests.
- `backtesting/test/__main__.py`: `python -m backtesting.test` entry point.
- `backtesting/test/__init__.py`: bundled sample data and test helpers.
- `backtesting/test/GOOG.csv`: bundled daily stock OHLCV data.
- `backtesting/test/EURUSD.csv`: bundled forex OHLCV data.
- `backtesting/test/BTCUSD.csv`: bundled crypto OHLCV data.

## Documentation

- `docs/README.md`: documentation index and policy.
- `docs/agent-guide.md`: repo navigation and maintenance rules for agents.
- `docs/architecture.md`: conceptual model and execution flow.
- `docs/api/`: handwritten public API reference.
- `docs/guides/`: task-oriented user guides.
- `docs/decisions/`: documentation and architecture decisions.

## Ownership Hints

- Strategy lifecycle changes usually live in `backtesting/backtesting.py`.
- Data access changes usually touch `_Data`, `_MultiData`, or `_Array` in `backtesting/_util.py`.
- Multi-asset behavior usually touches `PortfolioBacktest`, `_Broker`, `_MultiData`, stats, and plotting.
- Result metric changes usually touch `backtesting/_stats.py` and docs that list result fields.
- Plot behavior usually touches `backtesting/_plotting.py` and tests under `TestPlot`.
