# 0001: Markdown Documentation

## Status

Accepted.

## Context

The upstream project used generated pdoc HTML, Jupytext-synchronized notebooks, executed notebook exports, pdoc templates, and GitHub Pages deployment.

This fork prioritizes local discoverability and agent comprehension. The generated-docs workflow made documentation harder to inspect in the repository because the actual user-facing site was produced by a build step.

## Decision

Use Markdown files under `docs/` as the documentation source of truth.

The repository does not require:

- pdoc,
- Jupytext,
- notebook execution,
- generated HTML,
- a docs deployment workflow.

## Consequences

Benefits:

- Documentation is directly searchable with `rg`.
- Agents can inspect the docs without executing build steps.
- API and architecture docs can link directly to owning files and tests.
- CI docs validation can stay lightweight.

Tradeoffs:

- API docs are not generated automatically from docstrings.
- Maintainers must update Markdown when public behavior changes.
- Examples must be kept concise enough to maintain manually.
