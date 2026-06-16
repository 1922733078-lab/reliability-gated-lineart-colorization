#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sci_color_lab.fuzzy_selector import load_fuzzy_config, score_fuzzy_rules
from sci_color_lab.selector_utility import utility_column_for_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the fuzzy selector against fixed/random/oracle selectors.")
    parser.add_argument("--features-csv", type=str, default="analysis/selector/lineart_features_local_validation.csv", help="Lineart feature CSV.")
    parser.add_argument("--oracle-utilities-csv", type=str, default="analysis/selector/oracle_candidate_utilities_local_validation.csv", help="Oracle utility CSV.")
    parser.add_argument("--oracle-labels-csv", type=str, default="analysis/selector/oracle_labels_local_validation.csv", help="Oracle label CSV.")
    parser.add_argument("--rules-config", type=str, default="configs/fuzzy_rules_local.yaml", help="YAML fuzzy rule configuration.")
    parser.add_argument("--output-dir", type=str, default="analysis/selector", help="Output directory for selector analysis.")
    return parser.parse_args()


def load_selector_inputs(
    *,
    features_csv: str,
    oracle_utilities_csv: str,
    oracle_labels_csv: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(features_csv), pd.read_csv(oracle_utilities_csv), pd.read_csv(oracle_labels_csv)

def normalize_candidate_label(candidate_label: str) -> str:
    return str(candidate_label).lower().replace("-", "_")


def _flatten_memberships(payload: dict[str, dict[str, float]]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for variable_name, memberships in payload.items():
        for membership_name, score in memberships.items():
            flat[f"{variable_name}__{membership_name}"] = float(score)
    return flat


def _flatten_expected_gains(payload: dict[str, float]) -> dict[str, float]:
    return {f"expected_gain__{label}": float(score) for label, score in payload.items()}


def _feature_columns_for_policy(features_frame: pd.DataFrame, config_variables: list[str]) -> list[str]:
    columns = [column for column in config_variables if column in features_frame.columns]
    if columns:
        return columns
    fallback_columns = [
        "line_density",
        "edge_entropy",
        "blank_ratio",
        "complexity_score",
        "component_density",
    ]
    return [column for column in fallback_columns if column in features_frame.columns]


def collect_prediction_rows(
    *,
    features_frame: pd.DataFrame,
    rules_config_path: str,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    config = load_fuzzy_config(rules_config_path)
    candidate_labels = list(config.labels)
    policy_feature_columns = _feature_columns_for_policy(features_frame, list(config.variables.keys()))

    rows: list[dict[str, object]] = []
    for row in features_frame.to_dict(orient="records"):
        decision = score_fuzzy_rules(config, row)
        prediction_row = dict(row)
        prediction_row["predicted_label"] = decision["predicted_label"]
        prediction_row["raw_best_label"] = decision["raw_best_label"]
        prediction_row["default_label"] = decision["default_label"]
        prediction_row["raw_best_score"] = decision["raw_best_score"]
        prediction_row["default_score"] = decision["default_score"]
        prediction_row["score_margin"] = decision["score_margin"]
        prediction_row["fallback_applied"] = int(bool(decision["fallback_applied"]))
        prediction_row["fallback_reason"] = decision["fallback_reason"]
        prediction_row["confidence_margin"] = decision["confidence_margin"]
        prediction_row["decision_score"] = decision["top_score"]
        prediction_row["expected_gain_raw_best_label"] = decision["expected_gain_raw_best_label"]
        prediction_row["expected_gain_default_label"] = decision["expected_gain_default_label"]
        prediction_row.update({f"score__{label}": float(score) for label, score in decision["label_scores"].items()})
        prediction_row.update({f"rule__{name}": float(score) for name, score in decision["rule_scores"].items()})
        prediction_row.update(_flatten_memberships(decision["memberships"]))
        prediction_row.update(_flatten_expected_gains(decision["expected_gain_by_label"]))
        rows.append(prediction_row)

    metadata = {
        "candidate_labels": candidate_labels,
        "selection_policy": {
            "mode": config.selection_policy.mode,
            "default_label": config.selection_policy.default_label,
            "min_margin": config.selection_policy.min_margin,
            "min_expected_gain": config.selection_policy.min_expected_gain,
            "fallback_to_default": config.selection_policy.fallback_to_default,
            "cost_sensitive_mode": config.selection_policy.cost_sensitive_mode,
            "override_requirements": config.selection_policy.override_requirements or {},
        },
        "policy_feature_columns": policy_feature_columns,
        "policy_neighbors": 0,
    }
    return pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True), candidate_labels, metadata


def _method_utility(row: pd.Series, label: str) -> float:
    return float(row[utility_column_for_label(label)])


def _method_cost(row: pd.Series, label: str, suffix: str) -> float:
    return float(row.get(f"{label}__{suffix}", np.nan))


def _fixed_method_display_name(label: str) -> str:
    return f"Fixed {label}"


def build_selector_tables(
    *,
    merged: pd.DataFrame,
    candidate_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods: list[dict[str, object]] = []
    per_image_rows: list[dict[str, object]] = []

    utility_columns = [utility_column_for_label(label) for label in candidate_labels]
    random_accuracy = 1.0 / float(len(candidate_labels))

    for row in merged.to_dict(orient="records"):
        row_series = pd.Series(row)
        predicted_label = str(row["predicted_label"])
        fuzzy_utility = _method_utility(row_series, predicted_label)
        random_utility = float(np.mean([row[column] for column in utility_columns]))
        fuzzy_flops = _method_cost(row_series, predicted_label, "flops_g_prior")
        fuzzy_params = _method_cost(row_series, predicted_label, "params_m_prior")

        per_image_row: dict[str, object] = {
            "image_id": row["image_id"],
            "oracle_label": row["oracle_label"],
            "predicted_label": predicted_label,
            "raw_best_label": row.get("raw_best_label"),
            "default_label": row.get("default_label"),
            "score_margin": float(row.get("score_margin", 0.0) or 0.0),
            "confidence_margin": float(row.get("confidence_margin", 0.0) or 0.0),
            "fallback_applied": int(row.get("fallback_applied", 0) or 0),
            "fallback_reason": row.get("fallback_reason", ""),
            "complexity_score": row.get("complexity_score"),
            "fuzzy_utility": fuzzy_utility,
            "random_selector_utility": random_utility,
            "oracle_utility": float(row["oracle_utility"]),
            "fuzzy_flops_g": fuzzy_flops,
            "fuzzy_params_m": fuzzy_params,
            "correct": int(predicted_label == str(row["oracle_label"])),
        }
        for candidate_label in candidate_labels:
            normalized = normalize_candidate_label(candidate_label)
            per_image_row[f"fixed_{normalized}_utility"] = _method_utility(row_series, candidate_label)
            per_image_row[f"fixed_{normalized}_flops_g"] = _method_cost(row_series, candidate_label, "flops_g_prior")
            per_image_row[f"fixed_{normalized}_params_m"] = _method_cost(row_series, candidate_label, "params_m_prior")
            per_image_row[f"expected_gain__{candidate_label}"] = row.get(f"expected_gain__{candidate_label}")
        per_image_rows.append(per_image_row)

    per_image_frame = pd.DataFrame(per_image_rows).sort_values("image_id").reset_index(drop=True)
    oracle_mean = float(per_image_frame["oracle_utility"].mean())
    random_mean = float(per_image_frame["random_selector_utility"].mean())
    fixed_e5_column = "fixed_e5_utility" if "fixed_e5_utility" in per_image_frame.columns else None
    fixed_e5_mean = float(per_image_frame[fixed_e5_column].mean()) if fixed_e5_column else float("nan")

    selector_specs: list[tuple[str, str, pd.Series, float, pd.Series, pd.Series]] = []
    for candidate_label in candidate_labels:
        normalized = normalize_candidate_label(candidate_label)
        selector_specs.append(
            (
                _fixed_method_display_name(candidate_label),
                candidate_label,
                per_image_frame[f"fixed_{normalized}_utility"],
                float((merged["oracle_label"] == candidate_label).mean()),
                per_image_frame[f"fixed_{normalized}_flops_g"],
                per_image_frame[f"fixed_{normalized}_params_m"],
            )
        )
    selector_specs.extend(
        [
            ("Random Selector", "RANDOM", per_image_frame["random_selector_utility"], random_accuracy, pd.Series(np.nan, index=per_image_frame.index), pd.Series(np.nan, index=per_image_frame.index)),
            ("Oracle Selector", "ORACLE", per_image_frame["oracle_utility"], 1.0, pd.Series(np.nan, index=per_image_frame.index), pd.Series(np.nan, index=per_image_frame.index)),
            ("Fuzzy Selector", "FUZZY", per_image_frame["fuzzy_utility"], float(per_image_frame["correct"].mean()), per_image_frame["fuzzy_flops_g"], per_image_frame["fuzzy_params_m"]),
        ]
    )

    for method_name, chosen_label, utility_series, accuracy, flops_series, params_series in selector_specs:
        average_utility = float(pd.to_numeric(utility_series, errors="coerce").mean())
        methods.append(
            {
                "method": method_name,
                "selector_label": chosen_label,
                "selection_accuracy_vs_oracle": float(accuracy),
                "average_utility": average_utility,
                "utility_gain_over_random": float(average_utility - random_mean),
                "utility_gain_over_fixed_e5": float(average_utility - fixed_e5_mean) if fixed_e5_column else np.nan,
                "oracle_gap": float(oracle_mean - average_utility),
                "average_flops_g": float(pd.to_numeric(flops_series, errors="coerce").mean()) if not flops_series.empty else np.nan,
                "average_params_m": float(pd.to_numeric(params_series, errors="coerce").mean()) if not params_series.empty else np.nan,
            }
        )

    comparison_frame = pd.DataFrame(methods).sort_values("average_utility", ascending=False).reset_index(drop=True)
    confusion = (
        merged.assign(count=1)
        .pivot_table(index="oracle_label", columns="predicted_label", values="count", aggfunc="sum", fill_value=0)
        .sort_index(axis=0)
        .sort_index(axis=1)
        .reset_index()
    )
    return comparison_frame, confusion, per_image_frame


def _save_bar_plot(comparison_frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    ordered = comparison_frame.sort_values("average_utility", ascending=False)
    positions = np.arange(len(ordered))
    ax.bar(positions, ordered["average_utility"], color=["#264653" if name == "Fuzzy Selector" else "#6c757d" for name in ordered["method"]])
    ax.set_ylabel("Average Utility")
    ax.set_title("Selector Utility Comparison")
    ax.set_xticks(positions)
    ax.set_xticklabels(ordered["method"], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_label_distribution(predictions: pd.DataFrame, candidate_labels: list[str], output_path: Path) -> None:
    counts = predictions["predicted_label"].value_counts().reindex(candidate_labels, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=180)
    ax.bar(counts.index.tolist(), counts.values.tolist(), color="#457b9d")
    ax.set_ylabel("Count")
    ax.set_title("Fuzzy Selector Decision Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _complexity_subgroups(per_image_frame: pd.DataFrame, fixed_e5_column: str | None) -> list[dict[str, object]]:
    if per_image_frame.empty or "complexity_score" not in per_image_frame.columns:
        return []
    frame = per_image_frame.copy()
    frame["complexity_bin"] = pd.qcut(frame["complexity_score"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    rows: list[dict[str, object]] = []
    for bin_name, subgroup in frame.groupby("complexity_bin", dropna=True):
        rows.append(
            {
                "complexity_bin": str(bin_name),
                "count": int(len(subgroup)),
                "fuzzy_utility_mean": float(subgroup["fuzzy_utility"].mean()),
                "fixed_e5_utility_mean": float(subgroup[fixed_e5_column].mean()) if fixed_e5_column else np.nan,
                "oracle_utility_mean": float(subgroup["oracle_utility"].mean()),
                "fuzzy_accuracy": float(subgroup["correct"].mean()),
            }
        )
    return rows


def _confidence_proxy(per_image_frame: pd.DataFrame) -> list[dict[str, object]]:
    if per_image_frame.empty:
        return []
    frame = per_image_frame.copy()
    frame["confidence_bin"] = pd.qcut(frame["confidence_margin"], q=min(4, len(frame)), duplicates="drop")
    rows: list[dict[str, object]] = []
    for bin_name, subgroup in frame.groupby("confidence_bin", dropna=True):
        rows.append(
            {
                "confidence_bin": str(bin_name),
                "count": int(len(subgroup)),
                "mean_confidence_margin": float(subgroup["confidence_margin"].mean()),
                "accuracy": float(subgroup["correct"].mean()),
            }
        )
    return rows


def run_selector_evaluation(
    *,
    features_csv: str,
    oracle_utilities_csv: str,
    oracle_labels_csv: str,
    rules_config: str,
    output_dir: str,
) -> dict[str, Any]:
    features_frame, oracle_utility_frame, oracle_label_frame = load_selector_inputs(
        features_csv=features_csv,
        oracle_utilities_csv=oracle_utilities_csv,
        oracle_labels_csv=oracle_labels_csv,
    )
    predictions, candidate_labels, metadata = collect_prediction_rows(
        features_frame=features_frame,
        rules_config_path=rules_config,
    )

    merged = predictions.merge(oracle_utility_frame, on="image_id", how="inner").merge(
        oracle_label_frame[["image_id", "oracle_label", "oracle_utility", "random_baseline_expected_utility"]],
        on=["image_id", "oracle_label", "oracle_utility", "random_baseline_expected_utility"],
        how="inner",
    )
    comparison_frame, confusion_frame, per_image_frame = build_selector_tables(merged=merged, candidate_labels=candidate_labels)

    output_dir_path = Path(output_dir)
    plots_dir = output_dir_path / "plots"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fuzzy_predictions_path = output_dir_path / "fuzzy_predictions_local_validation.csv"
    comparison_csv_path = output_dir_path / "selector_comparison_local.csv"
    comparison_json_path = output_dir_path / "selector_comparison_local.json"
    confusion_path = output_dir_path / "selector_confusion_matrix_local.csv"
    summary_path = output_dir_path / "fuzzy_selector_local_summary.json"
    per_image_path = output_dir_path / "selector_method_utilities_local.csv"
    confidence_path = output_dir_path / "selector_confidence_proxy_local.csv"
    subgroup_path = output_dir_path / "selector_complexity_subgroups_local.csv"
    utility_plot_path = plots_dir / "selector_utility_comparison.png"
    distribution_plot_path = plots_dir / "selector_decision_distribution.png"

    predictions.to_csv(fuzzy_predictions_path, index=False, encoding="utf-8")
    comparison_frame.to_csv(comparison_csv_path, index=False, encoding="utf-8")
    comparison_json_path.write_text(comparison_frame.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    confusion_frame.to_csv(confusion_path, index=False, encoding="utf-8")
    per_image_frame.to_csv(per_image_path, index=False, encoding="utf-8")

    fixed_e5_column = "fixed_e5_utility" if "fixed_e5_utility" in per_image_frame.columns else None
    confidence_rows = _confidence_proxy(per_image_frame)
    subgroup_rows = _complexity_subgroups(per_image_frame, fixed_e5_column)
    pd.DataFrame(confidence_rows).to_csv(confidence_path, index=False, encoding="utf-8")
    pd.DataFrame(subgroup_rows).to_csv(subgroup_path, index=False, encoding="utf-8")

    _save_bar_plot(comparison_frame, utility_plot_path)
    _save_label_distribution(predictions, candidate_labels, distribution_plot_path)

    fuzzy_row = comparison_frame[comparison_frame["method"] == "Fuzzy Selector"]
    fixed_e5_row = comparison_frame[comparison_frame["method"] == "Fixed E5"]
    summary_payload = {
        "rules_config": str(Path(rules_config).resolve()),
        "num_samples": int(len(merged)),
        "candidate_labels": candidate_labels,
        "selection_policy": metadata["selection_policy"],
        "policy_feature_columns": metadata["policy_feature_columns"],
        "policy_neighbors": metadata["policy_neighbors"],
        "methods": comparison_frame.to_dict(orient="records"),
        "complexity_subgroups": subgroup_rows,
        "confidence_proxy": confidence_rows,
        "predicted_label_distribution": predictions["predicted_label"].value_counts().to_dict(),
        "raw_best_label_distribution": predictions["raw_best_label"].value_counts().to_dict(),
        "average_confidence_margin": float(predictions["confidence_margin"].mean()) if not predictions.empty else 0.0,
        "average_score_margin": float(predictions["score_margin"].mean()) if not predictions.empty else 0.0,
        "fallback_rate": float(predictions["fallback_applied"].mean()) if "fallback_applied" in predictions.columns else 0.0,
        "utility_delta_vs_fixed_e5": float(fuzzy_row["utility_gain_over_fixed_e5"].iloc[0]) if not fuzzy_row.empty else np.nan,
        "average_flops_delta_vs_fixed_e5": (
            float(fuzzy_row["average_flops_g"].iloc[0] - fixed_e5_row["average_flops_g"].iloc[0])
            if not fuzzy_row.empty and not fixed_e5_row.empty
            else np.nan
        ),
        "artifacts": {
            "fuzzy_predictions_csv": str(fuzzy_predictions_path.resolve()),
            "selector_comparison_csv": str(comparison_csv_path.resolve()),
            "selector_confusion_matrix_csv": str(confusion_path.resolve()),
            "selector_method_utilities_csv": str(per_image_path.resolve()),
            "selector_utility_plot": str(utility_plot_path.resolve()),
            "selector_distribution_plot": str(distribution_plot_path.resolve()),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    return {
        "predictions": predictions,
        "comparison": comparison_frame,
        "confusion": confusion_frame,
        "per_image": per_image_frame,
        "summary": summary_payload,
        "candidate_labels": candidate_labels,
    }

def main() -> None:
    args = parse_args()
    result = run_selector_evaluation(
        features_csv=args.features_csv,
        oracle_utilities_csv=args.oracle_utilities_csv,
        oracle_labels_csv=args.oracle_labels_csv,
        rules_config=args.rules_config,
        output_dir=args.output_dir,
    )
    print(f"[evaluate_fuzzy_selector] rows={len(result['per_image'])} comparison={(Path(args.output_dir) / 'selector_comparison_local.csv').resolve()}")


if __name__ == "__main__":
    main()
