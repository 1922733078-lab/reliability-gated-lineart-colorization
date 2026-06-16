# Release Checklist

Use this checklist before publishing the repository to GitHub.

## Must Check

- Confirm the chosen open-source license matches your intent.
- Make sure no dataset files, generated images, checkpoints, or local logs are staged.
- Verify `config.json` is not committed. Publish only example configs from `configs/`.
- Check that paths in docs and scripts are relative or environment-driven.
- Scan for local absolute paths, tokens, private keys, and internal package traces.
- Confirm the public repository URL in `docs/CODE_AVAILABILITY_STATEMENT.md`.

## Recommended

- Publish trained checkpoints separately only if redistribution rights are clear.
- Publish datasets separately only if licensing and redistribution rights are clear.
- Add a project homepage, paper link, and citation information after the manuscript is public.
- Run one clean-room setup test on another machine or a fresh virtual environment.

## Suggested Release Contents

- Source code under `sci_color_lab/`
- CLI entry scripts
- `README.md`
- `LICENSE`
- `CONTRIBUTING.md`
- `requirements.txt`
- Example configs under `configs/`

## Suggested Non-Release Contents

- Raw datasets
- Validation images
- Training outputs
- Archived inference artifacts
- Third-party helper tool copies with unclear redistribution status
- Personal notes, local session exports, and unrelated workspace folders
