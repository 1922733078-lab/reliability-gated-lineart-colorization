# Code Availability Statement

The code used to inspect the reliability-gated line-art colorization workflow is publicly available at:

https://github.com/1922733078-lab/reliability-gated-lineart-colorization

The repository includes training, inference, analysis, line-art feature extraction, fuzzy selector, selector-utility, and reliability-gate code, together with configuration templates and fuzzy-rule YAML files. Raw datasets, trained model weights, generated outputs, private logs, and local workspace artifacts are not included.

Suggested manuscript wording:

> The training, inference, analysis, line-art feature extraction, fuzzy selector, selector-utility, and reliability-gate code are publicly available at https://github.com/1922733078-lab/reliability-gated-lineart-colorization. The release supports method inspection and audit-level reproducibility of the selector and gate decisions. Raw images, trained weights, generated outputs, private logs, and local machine caches are excluded because of licensing, redistribution, and storage constraints.

Key selector materials in this release include:

- `sci_color_lab/lineart_features.py`
- `sci_color_lab/fuzzy_selector.py`
- `sci_color_lab/reliability_gate.py`
- `sci_color_lab/selector_utility.py`
- `configs/fuzzy_rules_local.yaml`
- `configs/fuzzy_rules_reliability_aware.yaml`
