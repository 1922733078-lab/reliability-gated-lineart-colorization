# Contributing

Thanks for considering a contribution to this project.

## Before You Open a PR

- Open an issue first for large changes, especially training pipeline changes or metric changes.
- Keep paths portable. Do not introduce machine-specific absolute paths.
- Do not commit datasets, generated outputs, checkpoints, caches, or local experiment logs.
- If a change affects reported results, describe the expected impact and how to reproduce it.

## Development Guidelines

- Prefer relative paths or environment variables over hard-coded local directories.
- Keep configuration changes in example files under `configs/`.
- If you add a new dependency, update `requirements.txt` and explain why in the PR.
- Preserve backward compatibility for existing run directories whenever practical.

## Pull Request Checklist

- Code runs with a fresh clone plus documented setup steps.
- Documentation is updated when behavior or CLI flags change.
- New scripts include a short module docstring and CLI help text.
- Large binary files are not included in the PR.
