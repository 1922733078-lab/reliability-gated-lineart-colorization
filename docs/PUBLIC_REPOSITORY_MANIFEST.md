# Public Repository Manifest

This manifest documents what is included in the public repository and what is intentionally excluded. The goal is to make the method inspectable while avoiding redistribution of restricted data, trained weights, generated outputs, local logs, or workspace-specific artifacts.

## Included

### Core research code

- `sci_color_lab/`
- `sci_color_lab/lineart_features.py`
- `sci_color_lab/fuzzy_selector.py`
- `sci_color_lab/reliability_gate.py`
- `sci_color_lab/selector_utility.py`
- `sci.py`
- `train_only.py`
- `inference_batch.py`
- `analyze_results.py`

### Utility scripts

- `tools/plot_seed_comparison.py`
- `tools/summarize_run.py`
- `tools/evaluate_fuzzy_selector.py`
- `tools/extract_lineart_features.py`
- `tools/run_minimal_public_validation.py`

## Manuscript Traceability Labels

The manuscript uses natural-language roles for readability and leaves raw implementation labels to the supplement and repository. This repository preserves those labels only for audit traceability:

- `E5` corresponds to the fixed ARF+AN conservative baseline discussed as `Fixed E5`.
- `E_FULL` is the complete-module negative control, not a success label.
- `V1` is the aggressive selector variant used as a contrast case for unsafe average-gain selection.
- `V2` is the reliability-aware selector variant.
- `v2_strict_only` denotes the strict-only reporting route for surviving local exceptions.
- `current46` denotes the repository alias for the 46-image local calibration set.
- `top20` / `shared_validation_selection_top20.json` denote a local 20-image diagnostic subset and its non-release manifest, not an independent public benchmark.
- `edges2shoes_ready` is a public-refusal/deployment-blocking flag in `tools/run_minimal_public_validation.py`; `false` means public deployment remains blocked.

These labels support audit-table and decision-trace reproducibility. They are not claims of public benchmark success, broad cross-dataset transferability, public deployment readiness, SOTA behavior, or human-preference validation.

### Configuration templates

- `configs/experiment_config.example.json`
- `configs/experiment_config.training_only.json`
- `configs/experiment_config.efull_12epoch.json`
- `configs/fuzzy_rules_local.yaml`
- `configs/fuzzy_rules_reliability_aware.yaml`

### Release documentation

- `README.md`
- `LICENSE`
- `CONTRIBUTING.md`
- `.gitignore`
- `.gitattributes`
- `docs/RELEASE_CHECKLIST.md`
- `docs/CODE_AVAILABILITY_STATEMENT.md`
- `docs/DATA_AVAILABILITY_STATEMENT.md`
- `docs/PUBLIC_REPOSITORY_MANIFEST.md`

### 5. 依赖声明

- `requirements.txt`

## Excluded

### Data

- raw training images
- raw validation images
- any real dataset copy under `data/`

Reason:

- the underlying images may have copyright, licensing, privacy, or redistribution restrictions

### Generated outputs and checkpoints

- `outputs_*/`
- `outputs/`
- `artifacts/`
- generated qualitative images
- trained checkpoints and model weights

Reason:

- these files are large and may expose local runtime details or non-redistributable generated material

### Local logs and workspace files

- `*.log`
- local training logs
- local configuration files such as `config.json`
- notebook caches, editor metadata, and session files

Reason:

- these files are not needed for method inspection and often include local paths or timestamps

### Third-party helper-tool copies

- helper-tool copies whose redistribution license has not been separately verified

Reason:

- users should install third-party tools from their original sources unless their redistribution terms are clear

## Renamed Files

The release copy uses publication-friendly file names:

- `configs/config.example.json` became `configs/experiment_config.example.json`
- `configs/config_training_only.json` became `configs/experiment_config.training_only.json`
- `configs/config_training_efull_adamw8bit_12epoch.json` became `configs/experiment_config.efull_12epoch.json`
- `docs/OPEN_SOURCE_RELEASE_CHECKLIST.md` became `docs/RELEASE_CHECKLIST.md`
- `plot_seed_comparison.py` became `tools/plot_seed_comparison.py`
- `summarize_run.py` became `tools/summarize_run.py`

## Additional Non-Release Materials

- `queue_remaining_groups_after_wait.sh`
- `shared_validation_selection_top20.json`
- root-level historical `*.log` files
- `docs/示例图/`
- `docs/配置分析/`
- all `outputs_*` directories
- all data directories

These files are retained outside this repository because they are local workflow materials, private dataset references, large outputs, or historical analysis artifacts rather than public method code.

## Minimal Reuse Set

For readers who only want to inspect the selector and reliability gate, the core files are:

- `README.md`
- `requirements.txt`
- `LICENSE`
- `sci.py`
- `train_only.py`
- `inference_batch.py`
- `analyze_results.py`
- `configs/`
- `configs/fuzzy_rules_local.yaml`
- `configs/fuzzy_rules_reliability_aware.yaml`
- `sci_color_lab/`
- `sci_color_lab/lineart_features.py`
- `sci_color_lab/fuzzy_selector.py`
- `sci_color_lab/reliability_gate.py`
- `sci_color_lab/selector_utility.py`
