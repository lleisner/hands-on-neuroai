# Contributing Guidelines

These guidelines define how to work inside projects created from this template.

## Branching

Use short, descriptive branch names:

- feature/<description>
- bugfix/<description>
- refactor/<description>
- docs/<description>

## Commits

Use clear, consistent commit messages:

type: short description

Examples:
- feature: add dataloader
- bugfix: fix shape mismatch
- refactor: simplify utils
- docs: update setup guide

Allowed types: feature, bugfix, refactor, docs, style, test, chore.

## Code Style

- Use ruff for linting and formatting.
- Prefer explicit imports.
- Add type hints everywhere.
- Keep modules and functions small and focused.
- Avoid unnecessary abstraction or clever hacks.

## Testing

Tests live in the tests/ folder.

Run tests using: pytest

Add tests for:
- utility functions
- core logic
- anything critical that should not silently break

## Adding Dependencies

1. Edit environment.yml (add/remove packages).
2. Update the environment: conda env update -f environment.yml

Keep dependencies minimal and project-specific.
Deep learning frameworks (PyTorch, TensorFlow, etc.) should be added in the individual project repos, not in this template.
