from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CANDIDATE_LABELS = ("E0", "E5", "E14", "E_FULL")


ALL_LOCAL_CANDIDATE_SPECS = {
    "E0": {"output_root": "runs/group_outputs/outputs_e0_adamw8bit_12epoch", "group_id": "E0"},
    "E2": {"output_root": "runs/group_outputs/outputs_e2_adamw8bit_12epoch", "group_id": "E2"},
    "E5": {"output_root": "runs/group_outputs/outputs_e5_adamw8bit_12epoch", "group_id": "E5"},
    "E14": {"output_root": "runs/group_outputs/outputs_e14_adamw8bit_12epoch", "group_id": "E14"},
    "E_FULL": {"output_root": "runs/group_outputs/outputs_efull_adamw8bit_12epoch", "group_id": "E_FULL"},
}


def resolve_local_candidate_specs(candidate_labels: list[str] | tuple[str, ...]) -> dict[str, dict[str, str]]:
    resolved: dict[str, dict[str, str]] = {}
    for candidate_label in candidate_labels:
        normalized = str(candidate_label).strip()
        if normalized not in ALL_LOCAL_CANDIDATE_SPECS:
            raise KeyError(f"Unsupported local candidate label: {normalized}")
        resolved[normalized] = dict(ALL_LOCAL_CANDIDATE_SPECS[normalized])
    return resolved


@dataclass(frozen=True)
class CandidateRunArtifact:
    candidate_label: str
    seed: int
    per_sample_csv: str
    metrics_json: str


def _safe_json_load(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_seed(seed_dir: Path) -> int:
    try:
        return int(seed_dir.name.split("_", 1)[1])
    except Exception:
        return 0


def discover_candidate_run_artifacts(
    *,
    output_root: str | Path,
    group_id: str,
    eval_root_name: str,
    epoch: int,
    seeds: list[int] | tuple[int, ...] | None = None,
) -> list[CandidateRunArtifact]:
    root = Path(output_root)
    artifacts: list[CandidateRunArtifact] = []
    allowed_seeds = {int(seed) for seed in seeds} if seeds else None
    for seed_dir in sorted((root / group_id).glob("seed_*")):
        seed_value = _parse_seed(seed_dir)
        if allowed_seeds is not None and seed_value not in allowed_seeds:
            continue
        eval_dir = seed_dir / "evaluations" / eval_root_name / f"epoch_{int(epoch):03d}"
        per_sample_csv = eval_dir / "per_sample_metrics.csv"
        metrics_json = eval_dir / "metrics.json"
        if not per_sample_csv.exists():
            continue
        artifacts.append(
            CandidateRunArtifact(
                candidate_label=group_id,
                seed=seed_value,
                per_sample_csv=str(per_sample_csv.resolve()),
                metrics_json=str(metrics_json.resolve()),
            )
        )
    return artifacts


def load_candidate_runs(
    *,
    candidate_specs: dict[str, dict[str, str]],
    eval_root_name: str,
    epoch: int,
    seeds: list[int] | tuple[int, ...] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], list[CandidateRunArtifact]]:
    frames_by_candidate: dict[str, pd.DataFrame] = {}
    priors_by_candidate: dict[str, dict[str, Any]] = {}
    artifacts: list[CandidateRunArtifact] = []

    for candidate_label, spec in candidate_specs.items():
        runs = discover_candidate_run_artifacts(
            output_root=spec["output_root"],
            group_id=spec.get("group_id", candidate_label),
            eval_root_name=eval_root_name,
            epoch=epoch,
            seeds=seeds,
        )
        if not runs:
            raise RuntimeError(
                f"No per-sample metrics found for candidate {candidate_label} under {spec['output_root']} "
                f"for {eval_root_name}/epoch_{int(epoch):03d}"
            )
        artifacts.extend(runs)

        run_frames: list[pd.DataFrame] = []
        prior_rows: list[dict[str, Any]] = []
        for run in runs:
            frame = pd.read_csv(run.per_sample_csv)
            frame = frame.copy()
            frame["candidate_label"] = candidate_label
            frame["seed"] = run.seed
            run_frames.append(frame)
            metrics_payload = _safe_json_load(run.metrics_json)
            if metrics_payload:
                prior_rows.append(metrics_payload)
        candidate_frame = pd.concat(run_frames, ignore_index=True)
        frames_by_candidate[candidate_label] = candidate_frame
        priors_by_candidate[candidate_label] = {
            "seed_count": int(candidate_frame["seed"].nunique()) if "seed" in candidate_frame.columns else len(runs),
            "fid": _mean_or_none(prior_rows, "fid"),
            "kid_mean": _mean_or_none(prior_rows, "kid_mean"),
            "flops_g": _mean_or_none(prior_rows, "flops_g"),
            "params_m": _mean_or_none(prior_rows, "params_m"),
            "inference_time_ms_mean": _mean_or_none(prior_rows, "inference_time_ms_mean"),
            "group_name": str(prior_rows[0].get("group_id", candidate_label)) if prior_rows else candidate_label,
        }
    return frames_by_candidate, priors_by_candidate, artifacts


def _mean_or_none(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def build_candidate_aggregate_table(frames_by_candidate: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged_frames: list[pd.DataFrame] = []
    for candidate_label, frame in frames_by_candidate.items():
        group = (
            frame.groupby("image_id", as_index=False)
            .agg(
                lineart_path=("lineart_path", "first"),
                target_path=("target_path", "first"),
                generated_path=("generated_path", "first"),
                seed_count=("seed", "nunique"),
                lpips=("lpips", "mean"),
                ssim=("ssim", "mean"),
                edge_consistency_f1=("edge_consistency_f1", "mean"),
                color_bleeding_rate=("color_bleeding_rate", "mean"),
                histogram_correlation=("histogram_correlation", "mean"),
                inference_time_ms=("inference_time_ms", "mean"),
            )
            .copy()
        )
        group["candidate_label"] = candidate_label
        merged_frames.append(group)
    merged = pd.concat(merged_frames, ignore_index=True)
    merged = merged.sort_values(["image_id", "candidate_label"]).reset_index(drop=True)
    return merged


def _rowwise_minmax_score(
    frame: pd.DataFrame,
    *,
    labels: list[str],
    metric_name: str,
    higher_is_better: bool,
    missing_fill: float = 0.5,
) -> dict[str, pd.Series]:
    data = pd.DataFrame({label: pd.to_numeric(frame[f"{label}__{metric_name}"], errors="coerce") for label in labels})
    row_min = data.min(axis=1, skipna=True)
    row_max = data.max(axis=1, skipna=True)
    denom = (row_max - row_min).replace(0.0, np.nan)
    if higher_is_better:
        score = data.sub(row_min, axis=0).div(denom, axis=0)
    else:
        score = row_max.sub(data, axis=0).div(denom, axis=0)
    score = score.fillna(missing_fill)
    return {label: score[label].astype(float) for label in labels}


def _normalize_scalar_map(values: dict[str, float | None], *, higher_is_better: bool, missing_fill: float = 0.5) -> dict[str, float]:
    numeric = {key: value for key, value in values.items() if value is not None}
    if not numeric:
        return {key: missing_fill for key in values}
    min_value = min(numeric.values())
    max_value = max(numeric.values())
    if max_value == min_value:
        return {key: missing_fill for key in values}
    result: dict[str, float] = {}
    for key, value in values.items():
        if value is None:
            result[key] = missing_fill
        elif higher_is_better:
            result[key] = float((value - min_value) / (max_value - min_value))
        else:
            result[key] = float((max_value - value) / (max_value - min_value))
    return result


def build_oracle_tables(
    *,
    aggregate_frame: pd.DataFrame,
    priors_by_candidate: dict[str, dict[str, Any]],
    candidate_labels: list[str],
    quality_weight: float = 0.40,
    structure_weight: float = 0.30,
    robustness_weight: float = 0.20,
    cost_weight: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    wide_parts: list[pd.DataFrame] = []
    for candidate_label in candidate_labels:
        candidate_frame = aggregate_frame[aggregate_frame["candidate_label"] == candidate_label].copy()
        candidate_frame = candidate_frame.drop(columns=["candidate_label"])
        renamed = candidate_frame.rename(columns={column: f"{candidate_label}__{column}" for column in candidate_frame.columns if column != "image_id"})
        wide_parts.append(renamed)

    wide = wide_parts[0]
    for part in wide_parts[1:]:
        wide = wide.merge(part, on="image_id", how="inner")

    lpips_scores = _rowwise_minmax_score(wide, labels=candidate_labels, metric_name="lpips", higher_is_better=False)
    ssim_scores = _rowwise_minmax_score(wide, labels=candidate_labels, metric_name="ssim", higher_is_better=True)
    edge_scores = _rowwise_minmax_score(wide, labels=candidate_labels, metric_name="edge_consistency_f1", higher_is_better=True)
    hist_scores = _rowwise_minmax_score(wide, labels=candidate_labels, metric_name="histogram_correlation", higher_is_better=True)
    bleeding_scores = _rowwise_minmax_score(wide, labels=candidate_labels, metric_name="color_bleeding_rate", higher_is_better=False)

    fid_good = _normalize_scalar_map({label: priors_by_candidate.get(label, {}).get("fid") for label in candidate_labels}, higher_is_better=False)
    params_bad = _normalize_scalar_map({label: priors_by_candidate.get(label, {}).get("params_m") for label in candidate_labels}, higher_is_better=True)
    flops_bad = _normalize_scalar_map({label: priors_by_candidate.get(label, {}).get("flops_g") for label in candidate_labels}, higher_is_better=True)

    utility_columns: list[str] = []
    for candidate_label in candidate_labels:
        quality = (lpips_scores[candidate_label] + fid_good[candidate_label]) / 2.0
        structure = (ssim_scores[candidate_label] + edge_scores[candidate_label]) / 2.0
        robustness = (hist_scores[candidate_label] + bleeding_scores[candidate_label]) / 2.0
        cost = (params_bad[candidate_label] + flops_bad[candidate_label]) / 2.0
        utility = (
            quality_weight * quality
            + structure_weight * structure
            + robustness_weight * robustness
            - cost_weight * cost
        )

        wide[f"{candidate_label}__quality_score"] = quality.astype(float)
        wide[f"{candidate_label}__structure_score"] = structure.astype(float)
        wide[f"{candidate_label}__robustness_score"] = robustness.astype(float)
        wide[f"{candidate_label}__cost_score"] = float(cost)
        wide[f"{candidate_label}__fid_prior_score"] = float(fid_good[candidate_label])
        wide[f"{candidate_label}__utility"] = utility.astype(float)
        wide[f"{candidate_label}__params_m_prior"] = priors_by_candidate.get(candidate_label, {}).get("params_m")
        wide[f"{candidate_label}__flops_g_prior"] = priors_by_candidate.get(candidate_label, {}).get("flops_g")
        wide[f"{candidate_label}__fid_prior"] = priors_by_candidate.get(candidate_label, {}).get("fid")
        utility_columns.append(f"{candidate_label}__utility")

    utility_frame = wide[["image_id", *utility_columns]].copy()
    utility_matrix = utility_frame[[column for column in utility_columns]].to_numpy(dtype=np.float64)
    best_indices = utility_matrix.argmax(axis=1)
    sorted_utilities = np.sort(utility_matrix, axis=1)
    top_gap = sorted_utilities[:, -1] - sorted_utilities[:, -2] if utility_matrix.shape[1] >= 2 else np.zeros(len(utility_frame))

    label_lookup = np.array(candidate_labels, dtype=object)
    wide["oracle_label"] = label_lookup[best_indices]
    wide["oracle_utility"] = utility_matrix[np.arange(len(utility_matrix)), best_indices]
    wide["oracle_utility_gap"] = top_gap
    wide["random_baseline_expected_utility"] = utility_matrix.mean(axis=1)

    label_frame = wide[
        [
            "image_id",
            "oracle_label",
            "oracle_utility",
            "oracle_utility_gap",
            "random_baseline_expected_utility",
            f"{candidate_labels[0]}__lineart_path",
            f"{candidate_labels[0]}__target_path",
        ]
    ].copy()
    label_frame = label_frame.rename(
        columns={
            f"{candidate_labels[0]}__lineart_path": "lineart_path",
            f"{candidate_labels[0]}__target_path": "target_path",
        }
    )

    metadata = {
        "candidate_labels": candidate_labels,
        "weights": {
            "quality_weight": quality_weight,
            "structure_weight": structure_weight,
            "robustness_weight": robustness_weight,
            "cost_weight": cost_weight,
        },
        "candidate_priors": priors_by_candidate,
    }
    return wide, label_frame, metadata


def utility_column_for_label(candidate_label: str) -> str:
    return f"{candidate_label}__utility"
