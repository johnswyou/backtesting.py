# Examples

Examples should be plain, runnable Python snippets or short scripts.

The documentation source of truth is Markdown under `docs/`. Example notebooks are not required for docs and should not be used as canonical docs sources.

## Bundled Data

The package includes sample data under `backtesting/test/`:

- `GOOG`
- `EURUSD`
- `BTCUSD`

Use them in examples:

```python
from backtesting.test import GOOG
```

## Example Guidelines

- Keep examples deterministic.
- Prefer bundled data unless the example is explicitly about external data loading.
- Include imports.
- Avoid network access.
- Keep runtime short enough for tests if the example is executed in CI.
- Link back to the relevant guide or API page.
