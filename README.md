# Reliability-Gated Line-Art Colorization

Code release for a reliability-gated line-art colorization ablation study built on `SDXL + ControlNet + LoRA`.

This repository contains the public method code for training, centralized inference, post-hoc analysis, line-art feature extraction, fuzzy proposal scoring, selector-utility computation, and reliability gating. It intentionally excludes raw datasets, trained checkpoints, generated outputs, private logs, and local workspace artifacts.

## Scope

Included:

- Core research code under `sci_color_lab/`
- Training, inference, and analysis entry scripts
- Reliability-aware selector support code, including line-art feature extraction, fuzzy proposal scoring, selector-utility computation, and reliability gating
- Public reproducibility config templates
- Public release documentation

Excluded:

- Raw training and validation datasets
- Trained weights and checkpoints
- Generated qualitative results and archived inference folders
- Local experiment logs and personal workspace files
- Third-party helper-tool copies with uncertain redistribution scope

## Repository Structure

```text
.
├── sci.py
├── train_only.py
├── inference_batch.py
├── analyze_results.py
├── configs/
│   ├── experiment_config.example.json
│   ├── experiment_config.training_only.json
│   ├── experiment_config.efull_12epoch.json
│   ├── fuzzy_rules_local.yaml
│   └── fuzzy_rules_reliability_aware.yaml
├── docs/
│   ├── PUBLIC_REPOSITORY_MANIFEST.md
│   ├── CODE_AVAILABILITY_STATEMENT.md
│   ├── DATA_AVAILABILITY_STATEMENT.md
│   └── RELEASE_CHECKLIST.md
├── tools/
│   ├── plot_seed_comparison.py
│   └── summarize_run.py
└── sci_color_lab/
```

## Method Overview

The project studies four switchable adapter modules:

- `DW`: dynamic weighting
- `AT`: adaptive threshold
- `ARF`: adaptive receptive field
- `AN`: adaptive normalization

The ablation definitions are implemented in `sci_color_lab/ablation.py`.

Group summary:

- `E0`: baseline, all adaptive modules off
- `E_FULL`: full model, all adaptive modules on
- `E1-E4`: single-module ablations
- `E5-E10`: two-module ablations

## Environment

Recommended:

- Python `3.8+`
- Linux with CUDA for training
- 24 GB class GPU for the `768 x 1024` setting

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Notes:

- `bitsandbytes` is only required for `optimizer_name=adamw8bit`
- If `adamw8bit` is unavailable on your system, use `adamw`

## Models

Model paths or Hugging Face repo IDs can be specified through config:

- `base_model`
- `controlnet_model`

Optional environment variables:

- `SCI_BASE_MODEL`
- `SCI_CONTROLNET_SCRIBBLE_MODEL`
- `SCI_CONTROLNET_CANNY_MODEL`

If config fields are empty, the code tries:

1. explicit environment variables
2. repo-local `models/` directories
3. the standard Hugging Face cache
4. downloading the default public repo if needed

## Dataset Layout

Expected layout:

```text
data/
├── train/
│   ├── color/
│   └── lineart/
└── validation/
    ├── color/
    └── lineart/
```

Pairing is filename-based.

## Configuration

Do not publish a machine-specific `config.json`.

Use one of the public templates instead:

- `configs/experiment_config.example.json`
- `configs/experiment_config.training_only.json`
- `configs/experiment_config.efull_12epoch.json`

Example:

```bash
cp configs/experiment_config.example.json config.local.json
```

Then edit paths and model fields for your environment.

## Quick Start

Full workflow:

```bash
python3 sci.py full \
  --group-id E_FULL \
  --config-json config.local.json \
  --epochs 12 \
  --seeds 42 123 456
```

Step-by-step:

```bash
python3 train_only.py \
  --group-id E_FULL \
  --config-json config.local.json \
  --epochs 12 \
  --seeds 42 123 456

python3 inference_batch.py \
  --run-dirs outputs/E_FULL/seed_42 outputs/E_FULL/seed_123 outputs/E_FULL/seed_456 \
  --epochs all

python3 analyze_results.py \
  --output-root outputs \
  --groups E_FULL
```

Utilities:

```bash
python3 tools/summarize_run.py --run-dir outputs/E_FULL/seed_42
python3 tools/plot_seed_comparison.py --group E_FULL --output-root outputs
```

## Output Layout

Default output directory:

```text
outputs/
└── E_FULL/
    └── seed_42/
        ├── checkpoints/
        ├── evaluations/
        ├── logs/
        ├── plots/
        ├── run_metadata.json
        └── run_summary.json
```

Archived inference artifacts default to:

```text
artifacts/inference_archive/
```

## Reproducibility Scope

This repository supports method inspection and audit-level reproducibility for the selector and reliability-gate logic. It is not a complete public regeneration package because raw training/validation images, trained model weights, generated outputs, private logs, and local machine caches are not redistributed.

The release is intended to make the following parts inspectable:

- `sci_color_lab/lineart_features.py`
- `sci_color_lab/fuzzy_selector.py`
- `sci_color_lab/reliability_gate.py`
- `sci_color_lab/selector_utility.py`
- `configs/fuzzy_rules_local.yaml`
- `configs/fuzzy_rules_reliability_aware.yaml`

See:

- `docs/PUBLIC_REPOSITORY_MANIFEST.md`
- `docs/CODE_AVAILABILITY_STATEMENT.md`
- `docs/DATA_AVAILABILITY_STATEMENT.md`
- `docs/RELEASE_CHECKLIST.md`

## License

This release uses the MIT License.
