#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sci_color_lab.selector_utility import build_oracle_tables
from sci_color_lab.trainer import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble the minimal public validation utility and decision package.")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--eval-root-name", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=12)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--records-json", type=str, required=True)
    parser.add_argument("--manifest-csv", type=str, required=True)
    parser.add_argument("--selector-decisions-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def metric_payload_ready(payload: dict[str, Any]) -> bool:
    return bool(
        payload
        and payload.get("generated_samples_count") is not None
        and (payload.get("ssim_mean") is not None or payload.get("ssim") is not None)
        and (payload.get("lpips_mean") is not None or payload.get("lpips") is not None)
        and payload.get("kid_mean") is not None
        and payload.get("kid_std") is not None
    )


def mean_or_none(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not clean:
        return None
    return float(np.mean(clean))


def safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def run_dir_for(method: str, seed: int) -> Path:
    if method == "E0":
        return PROJECT_ROOT / f"runs/group_outputs/outputs_e0_adamw8bit_12epoch/E0/seed_{seed}"
    if method == "E5":
        return PROJECT_ROOT / f"runs/group_outputs/outputs_e5_adamw8bit_12epoch/E5/seed_{seed}"
    raise KeyError(method)


def eval_dir_for(method: str, seed: int, eval_root_name: str, epoch: int) -> Path:
    return run_dir_for(method, seed) / "evaluations" / eval_root_name / f"epoch_{epoch:03d}"


def load_summary_rows(
    *,
    manifest_frame: pd.DataFrame,
    records_payload: dict[str, Any],
    selector_frame: pd.DataFrame,
    eval_root_name: str,
    epoch: int,
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    expected_image_ids = [str(item) for item in records_payload.get("image_ids", [])]
    completion_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    all_candidate_rows: list[pd.DataFrame] = []
    blocked_reasons: list[str] = []

    for method in ("E0", "E5"):
        for seed in seeds:
            eval_dir = eval_dir_for(method, seed, eval_root_name, epoch)
            metrics_path = eval_dir / "metrics.json"
            per_sample_path = eval_dir / "per_sample_metrics.csv"
            generation_records_path = eval_dir / "generation_records.json"
            generated_dir = eval_dir / "generated"
            target_dir = eval_dir / "target"
            lineart_dir = eval_dir / "lineart"
            metrics_payload = load_json(metrics_path)
            generation_payload = load_json(generation_records_path)

            generated_count = len(list(generated_dir.glob("*.png"))) if generated_dir.exists() else 0
            target_count = len(list(target_dir.glob("*.png"))) if target_dir.exists() else 0
            lineart_count = len(list(lineart_dir.glob("*.png"))) if lineart_dir.exists() else 0
            complete = (
                eval_dir.exists()
                and per_sample_path.exists()
                and metric_payload_ready(metrics_payload)
                and generated_count == len(expected_image_ids)
                and target_count == len(expected_image_ids)
                and lineart_count == len(expected_image_ids)
            )
            image_ids = [str(item) for item in generation_payload.get("selection_image_ids", [])]
            same_image_ids = image_ids == expected_image_ids
            completion_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "eval_dir": str(eval_dir.resolve()),
                    "exists": eval_dir.exists(),
                    "metrics_ready": metric_payload_ready(metrics_payload),
                    "per_sample_exists": per_sample_path.exists(),
                    "generated_count": generated_count,
                    "target_count": target_count,
                    "lineart_count": lineart_count,
                    "expected_count": len(expected_image_ids),
                    "same_image_ids": same_image_ids,
                    "complete": complete and same_image_ids,
                }
            )
            if not (complete and same_image_ids):
                blocked_reasons.append(f"incomplete_or_mismatched:{method}:seed_{seed}")
                continue

            sample_frame = pd.read_csv(per_sample_path).copy()
            sample_frame["candidate_label"] = method
            sample_frame["seed"] = seed
            sample_frame = sample_frame.merge(
                manifest_frame[["image_id", "dataset"]],
                on="image_id",
                how="left",
            )
            all_candidate_rows.append(sample_frame)
            method_rows.extend(
                [
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "sample_count": int(len(group)),
                        "fid": metrics_payload.get("fid"),
                        "kid_mean": metrics_payload.get("kid_mean"),
                        "kid_std": metrics_payload.get("kid_std"),
                        "lpips_mean": float(pd.to_numeric(group["lpips"], errors="coerce").mean()),
                        "ssim_mean": float(pd.to_numeric(group["ssim"], errors="coerce").mean()),
                        "edge_f1_mean": float(pd.to_numeric(group["edge_consistency_f1"], errors="coerce").mean()),
                        "color_bleeding_rate_mean": float(pd.to_numeric(group["color_bleeding_rate"], errors="coerce").mean()),
                        "histogram_correlation_mean": float(pd.to_numeric(group["histogram_correlation"], errors="coerce").mean()),
                        "flops_g": metrics_payload.get("flops_g"),
                        "params_m": metrics_payload.get("params_m"),
                        "inference_time_ms_mean": float(
                            pd.to_numeric(group["inference_time_ms"], errors="coerce").mean()
                        ),
                    }
                    for dataset, group in sample_frame.groupby("dataset")
                ]
            )

    completion_frame = pd.DataFrame(completion_rows)
    if all_candidate_rows:
        candidate_frame = pd.concat(all_candidate_rows, ignore_index=True)
    else:
        candidate_frame = pd.DataFrame()

    if not candidate_frame.empty:
        pooled_rows: list[dict[str, Any]] = []
        for (method, seed), group in candidate_frame.groupby(["candidate_label", "seed"]):
            metrics_payload = load_json(eval_dir_for(method, seed, eval_root_name, epoch) / "metrics.json")
            pooled_rows.append(
                {
                    "dataset": "pooled",
                    "method": method,
                    "seed": seed,
                    "sample_count": int(len(group)),
                    "fid": metrics_payload.get("fid"),
                    "kid_mean": metrics_payload.get("kid_mean"),
                    "kid_std": metrics_payload.get("kid_std"),
                    "lpips_mean": float(pd.to_numeric(group["lpips"], errors="coerce").mean()),
                    "ssim_mean": float(pd.to_numeric(group["ssim"], errors="coerce").mean()),
                    "edge_f1_mean": float(pd.to_numeric(group["edge_consistency_f1"], errors="coerce").mean()),
                    "color_bleeding_rate_mean": float(pd.to_numeric(group["color_bleeding_rate"], errors="coerce").mean()),
                    "histogram_correlation_mean": float(pd.to_numeric(group["histogram_correlation"], errors="coerce").mean()),
                    "flops_g": metrics_payload.get("flops_g"),
                    "params_m": metrics_payload.get("params_m"),
                    "inference_time_ms_mean": float(pd.to_numeric(group["inference_time_ms"], errors="coerce").mean()),
                }
            )
        method_rows.extend(pooled_rows)

    metadata = {"blocked_reasons": blocked_reasons}
    return completion_frame, pd.DataFrame(method_rows), metadata


def build_priors(method_summary_frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    priors: dict[str, dict[str, Any]] = {}
    for method in ("E0", "E5"):
        frame = method_summary_frame[(method_summary_frame["method"] == method) & (method_summary_frame["dataset"] == "pooled")]
        priors[method] = {
            "fid": mean_or_none([safe_float(item) for item in frame["fid"].tolist()]) if not frame.empty else None,
            "kid_mean": mean_or_none([safe_float(item) for item in frame["kid_mean"].tolist()]) if not frame.empty else None,
            "flops_g": mean_or_none([safe_float(item) for item in frame["flops_g"].tolist()]) if not frame.empty else None,
            "params_m": mean_or_none([safe_float(item) for item in frame["params_m"].tolist()]) if not frame.empty else None,
            "inference_time_ms_mean": mean_or_none([safe_float(item) for item in frame["inference_time_ms_mean"].tolist()])
            if not frame.empty
            else None,
            "group_name": method,
        }
    return priors


def make_decision(
    dataset_summary_frame: pd.DataFrame,
    paired_summary_frame: pd.DataFrame,
    completion_ok: bool,
) -> tuple[str, bool, bool, str, list[str]]:
    reasons: list[str] = []
    if not completion_ok:
        reasons.append("completion_or_metrics_incomplete")
    public_accept_count = int(paired_summary_frame["reliability_aware_gate_passed"].astype(bool).sum()) if not paired_summary_frame.empty else 0
    strict_accept_count = int(paired_summary_frame["strict_public_safe_e0"].astype(bool).sum()) if not paired_summary_frame.empty else 0
    accepted_frame = paired_summary_frame[
        paired_summary_frame["reliability_aware_gate_passed"].astype(bool) | paired_summary_frame["strict_only_gate_passed"].astype(bool)
    ].copy() if not paired_summary_frame.empty else pd.DataFrame()
    if dataset_summary_frame.empty:
        reasons.append("dataset_summary_missing")
    else:
        real_datasets = dataset_summary_frame[dataset_summary_frame["dataset"] != "pooled"].copy()
        if real_datasets.empty:
            reasons.append("real_dataset_rows_missing")
        else:
            for row in real_datasets.to_dict(orient="records"):
                if safe_float(row.get("mean_e0_delta_vs_e5")) is None or float(row["mean_e0_delta_vs_e5"]) <= 0.0:
                    reasons.append(f"nonpositive_mean_delta:{row['dataset']}")
                if int(row.get("negative_seed_count", 0) or 0) > 0:
                    reasons.append(f"negative_seed_count_nonzero:{row['dataset']}")
                if int(row.get("per_image_loss_count", 0) or 0) > 0:
                    reasons.append(f"per_image_loss_count_nonzero:{row['dataset']}")
                if int(row.get("strict_public_safe_e0_count", 0) or 0) <= 0:
                    reasons.append(f"strict_public_safe_sparse:{row['dataset']}")
                if int(row.get("reliability_aware_public_accept_count", 0) or 0) <= 0:
                    reasons.append(f"selector_public_accept_zero:{row['dataset']}")

    if public_accept_count == 0:
        reasons.append("selector_public_accepts_zero")
    if strict_accept_count == 0:
        reasons.append("strict_public_safe_e0_count_zero")
    if not accepted_frame.empty and int((accepted_frame["e0_negative_seed_count"] > 0).sum()) > 0:
        reasons.append("accepted_images_have_negative_seeds")
    if not accepted_frame.empty and int((accepted_frame["e0_min_delta_vs_e5"] <= 0.0).sum()) > 0:
        reasons.append("accepted_images_have_per_image_losses")

    if reasons:
        decision = "public_no_go_supported"
        shoes_ready = False
        handbags_ready = False
        boundary = "public external validation supports conservative refusal/no-go; no public deployment or transferability claim"
    else:
        decision = "public_selector_candidate_only"
        shoes_ready = False
        handbags_ready = False
        boundary = "The public validation subset identifies only a small candidate region and does not support broad public-dataset deployment."
    return decision, shoes_ready, handbags_ready, boundary, sorted(set(reasons))


def build_summary_markdown(
    *,
    decision_payload: dict[str, Any],
    dataset_summary_frame: pd.DataFrame,
    completion_payload: dict[str, Any],
    method_summary_path: Path,
    per_image_path: Path,
    paired_path: Path,
    dataset_path: Path,
) -> str:
    lines = [
        "# PUBLIC_VALIDATION_SUMMARY",
        "",
        "## Status and Completion",
        "",
        f"- completion: {str(bool(decision_payload.get('completion', False))).lower()}",
        f"- public_validation_decision: {decision_payload.get('public_validation_decision', '')}",
        f"- blocked: {str(bool(completion_payload.get('blocked', False))).lower()}",
        "",
        "## Dataset and Sample Design",
        "",
        "- datasets: edges2shoes val, edges2handbags val",
        "- samples_per_dataset: 64",
        "- total_public_samples: 128",
        "- methods: E0, Fixed E5",
        "- seeds: 42, 123, 456",
        "- total_generated_images: 768",
        "",
        "## Generation Count",
        "",
        f"- completed_eval_dirs: {completion_payload.get('completed_eval_dirs', 0)} / 6",
        "",
        "## Metric Availability",
        "",
        f"- metrics_ready_eval_dirs: {completion_payload.get('metrics_ready_eval_dirs', 0)} / 6",
        f"- per_sample_metric_rows_expected: {completion_payload.get('expected_image_count', 0)} per eval dir",
        "",
        "## E0 vs Fixed E5 Public Utility Results",
        "",
        "| dataset | samples | mean E0-FixedE5 utility delta | negative seeds | per-image losses | strict public-safe E0 | decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    if not dataset_summary_frame.empty:
        for row in dataset_summary_frame.to_dict(orient="records"):
            decision = "no_go" if row["dataset"] != "pooled" and (float(row["mean_e0_delta_vs_e5"]) <= 0.0 or int(row["per_image_loss_count"]) > 0) else "pooled_audit"
            lines.append(
                "| {dataset} | {sample_count} | {delta:.6f} | {negative_seed_count} | {per_image_loss_count} | {strict_public_safe_e0_count} | {decision} |".format(
                    dataset=row["dataset"],
                    sample_count=int(row["sample_count"]),
                    delta=float(row["mean_e0_delta_vs_e5"]),
                    negative_seed_count=int(row["negative_seed_count"]),
                    per_image_loss_count=int(row["per_image_loss_count"]),
                    strict_public_safe_e0_count=int(row["strict_public_safe_e0_count"]),
                    decision=decision,
                )
            )
    lines.extend(
        [
            "",
            "## Public Selector and Gate Acceptance Counts",
            "",
            f"- strict_only_public_accept_count: {completion_payload.get('strict_only_public_accept_count', 0)}",
            f"- reliability_aware_public_accept_count: {completion_payload.get('reliability_aware_public_accept_count', 0)}",
            "",
            "## Final Public Validation Decision",
            "",
            f"- {decision_payload.get('claim_boundary', '')}",
            "",
            "## Allowed Manuscript Wording",
            "",
            f"- {decision_payload.get('allowed_manuscript_wording', '')}",
            "",
            "## Blocked Manuscript Wording",
            "",
            f"- {decision_payload.get('blocked_manuscript_wording', '')}",
            "",
            "## Claim Boundary",
            "",
            "Allowed:",
            "- small public validation subset",
            "- external validation of the no-go/refusal boundary, if the gate fails",
            "- public E0/E5 metric audit on edges2shoes/edges2handbags validation subsets",
            "",
            "Blocked:",
            "- SOTA or benchmark-leading claim",
            "- broad cross-dataset transferability",
            "- public deployment readiness",
            "- human preference validation",
            "- public validation of methods not run in this phase",
            "",
            "## Artifact List",
            "",
            f"- method_summary_csv: `{method_summary_path}`",
            f"- public_per_image_seed_utilities_csv: `{per_image_path}`",
            f"- public_paired_delta_summary_csv: `{paired_path}`",
            f"- public_dataset_delta_summary_csv: `{dataset_path}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    project_root = resolve_path(Path.cwd(), args.project_root)
    output_dir = resolve_path(project_root, args.output_dir)
    metrics_dir = output_dir / "metrics"
    gate_dir = output_dir / "gate"
    quality_dir = output_dir / "quality_checks"
    paper_notes_dir = output_dir / "paper_notes"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    gate_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)
    paper_notes_dir.mkdir(parents=True, exist_ok=True)

    records_payload = json.loads(resolve_path(project_root, args.records_json).read_text(encoding="utf-8"))
    manifest_frame = pd.read_csv(resolve_path(project_root, args.manifest_csv))
    selector_frame = pd.read_csv(resolve_path(project_root, args.selector_decisions_csv))

    completion_frame, method_summary_frame, completion_meta = load_summary_rows(
        manifest_frame=manifest_frame,
        records_payload=records_payload,
        selector_frame=selector_frame,
        eval_root_name=args.eval_root_name,
        epoch=int(args.epoch),
        seeds=[int(seed) for seed in args.seeds],
    )
    completion_ok = bool(not completion_frame.empty and completion_frame["complete"].astype(bool).all())
    completion_payload = {
        "blocked": not completion_ok,
        "blocked_reasons": completion_meta.get("blocked_reasons", []),
        "completed_eval_dirs": int(completion_frame["complete"].astype(bool).sum()) if not completion_frame.empty else 0,
        "metrics_ready_eval_dirs": int(completion_frame["metrics_ready"].astype(bool).sum()) if not completion_frame.empty else 0,
        "expected_image_count": int(len(records_payload.get("image_ids", []))),
    }
    write_json(quality_dir / "public_validation_completion.json", completion_payload)

    if not completion_ok:
        decision_payload = {
            "completion": False,
            "public_validation_decision": "public_no_go_supported",
            "edges2shoes_ready": False,
            "edges2handbags_ready": False,
            "claim_boundary": "public external validation supports conservative refusal/no-go; no public deployment or transferability claim",
            "allowed_manuscript_wording": "A small public validation subset is reported only as an external no-go/refusal audit.",
            "blocked_manuscript_wording": "SOTA, deployment readiness, broad transferability, or human preference validation.",
            "reasons": sorted(set(completion_meta.get("blocked_reasons", []))),
        }
        write_json(gate_dir / "public_validation_decision.json", decision_payload)
        method_summary_frame.to_csv(metrics_dir / "public_method_summary.csv", index=False, encoding="utf-8")
        pd.DataFrame().to_csv(metrics_dir / "public_per_image_seed_utilities.csv", index=False, encoding="utf-8")
        pd.DataFrame().to_csv(metrics_dir / "public_paired_delta_summary.csv", index=False, encoding="utf-8")
        pd.DataFrame().to_csv(metrics_dir / "public_dataset_delta_summary.csv", index=False, encoding="utf-8")
        summary_text = build_summary_markdown(
            decision_payload=decision_payload,
            dataset_summary_frame=pd.DataFrame(),
            completion_payload=completion_payload,
            method_summary_path=metrics_dir / "public_method_summary.csv",
            per_image_path=metrics_dir / "public_per_image_seed_utilities.csv",
            paired_path=metrics_dir / "public_paired_delta_summary.csv",
            dataset_path=metrics_dir / "public_dataset_delta_summary.csv",
        )
        (output_dir / "PUBLIC_VALIDATION_SUMMARY.md").write_text(summary_text, encoding="utf-8")
        (paper_notes_dir / "ASC_PUBLIC_VALIDATION_NOTES.md").write_text(
            "Public validation assembly was blocked by incomplete generation or metric outputs.\n",
            encoding="utf-8",
        )
        return 0

    priors_by_candidate = build_priors(method_summary_frame)
    per_image_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for seed in [int(seed) for seed in args.seeds]:
        candidate_frames: list[pd.DataFrame] = []
        for method in ("E0", "E5"):
            frame = pd.read_csv(eval_dir_for(method, seed, args.eval_root_name, int(args.epoch)) / "per_sample_metrics.csv").copy()
            frame["candidate_label"] = method
            frame["seed"] = seed
            frame = frame.merge(manifest_frame[["image_id", "dataset"]], on="image_id", how="left")
            candidate_frames.append(frame)
        seed_frame = pd.concat(candidate_frames, ignore_index=True)
        aggregate_frame = (
            seed_frame.groupby(["image_id", "dataset", "candidate_label"], as_index=False)
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
        utility_wide, _, _ = build_oracle_tables(
            aggregate_frame=aggregate_frame.drop(columns=["dataset"]),
            priors_by_candidate=priors_by_candidate,
            candidate_labels=["E0", "E5"],
            quality_weight=0.40,
            structure_weight=0.30,
            robustness_weight=0.20,
            cost_weight=0.10,
        )
        utility_wide = utility_wide.merge(manifest_frame[["image_id", "dataset"]], on="image_id", how="left")
        utility_wide = utility_wide.merge(
            selector_frame[
                [
                    "image_id",
                    "strict_only_gate_passed",
                    "reliability_aware_gate_passed",
                    "raw_best_label",
                ]
            ],
            on="image_id",
            how="left",
        )
        for row in utility_wide.to_dict(orient="records"):
            per_image_rows.append(
                {
                    "dataset": row["dataset"],
                    "image_id": row["image_id"],
                    "seed": seed,
                    "fixed_e0_utility": float(row["E0__utility"]),
                    "fixed_e5_utility": float(row["E5__utility"]),
                    "e0_delta_vs_e5": float(row["E0__utility"] - row["E5__utility"]),
                    "fixed_e0_flops_g": priors_by_candidate["E0"].get("flops_g"),
                    "fixed_e5_flops_g": priors_by_candidate["E5"].get("flops_g"),
                    "fixed_e0_params_m": priors_by_candidate["E0"].get("params_m"),
                    "fixed_e5_params_m": priors_by_candidate["E5"].get("params_m"),
                    "strict_only_gate_passed": bool(row.get("strict_only_gate_passed", False)),
                    "reliability_aware_gate_passed": bool(row.get("reliability_aware_gate_passed", False)),
                    "raw_best_label": row.get("raw_best_label", ""),
                }
            )

    per_image_frame = pd.DataFrame(per_image_rows)
    for (dataset, image_id), group in per_image_frame.groupby(["dataset", "image_id"]):
        deltas = pd.to_numeric(group["e0_delta_vs_e5"], errors="coerce").astype(float).tolist()
        strict_gate_passed = bool(group["strict_only_gate_passed"].astype(bool).all())
        reliability_gate_passed = bool(group["reliability_aware_gate_passed"].astype(bool).all())
        raw_best_labels = group["raw_best_label"].astype(str).tolist()
        positive_seed_count = int(sum(delta > 0.0 for delta in deltas))
        negative_seed_count = int(sum(delta < 0.0 for delta in deltas))
        paired_rows.append(
            {
                "dataset": dataset,
                "image_id": image_id,
                "seed_count": int(len(deltas)),
                "e0_mean_delta_vs_e5": float(np.mean(deltas)),
                "e0_std_delta_vs_e5": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "e0_min_delta_vs_e5": float(min(deltas)),
                "e0_max_delta_vs_e5": float(max(deltas)),
                "e0_positive_seed_count": positive_seed_count,
                "e0_negative_seed_count": negative_seed_count,
                "strict_public_safe_e0": bool(strict_gate_passed and negative_seed_count == 0 and min(deltas) > 0.0),
                "strict_only_gate_passed": strict_gate_passed,
                "reliability_aware_gate_passed": reliability_gate_passed,
                "raw_best_label": statistics.mode(raw_best_labels) if raw_best_labels else "",
            }
        )

    paired_frame = pd.DataFrame(paired_rows)
    dataset_rows: list[dict[str, Any]] = []
    for dataset_name, group in list(paired_frame.groupby("dataset")) + [("pooled", paired_frame)]:
        if group.empty:
            continue
        subset_seed_rows = per_image_frame[per_image_frame["dataset"].eq(dataset_name)] if dataset_name != "pooled" else per_image_frame
        dataset_rows.append(
            {
                "dataset": dataset_name,
                "sample_count": int(group["image_id"].nunique()),
                "seed_count": int(len(subset_seed_rows)),
                "mean_e0_delta_vs_e5": float(pd.to_numeric(subset_seed_rows["e0_delta_vs_e5"], errors="coerce").mean()),
                "median_e0_delta_vs_e5": float(pd.to_numeric(subset_seed_rows["e0_delta_vs_e5"], errors="coerce").median()),
                "min_e0_delta_vs_e5": float(pd.to_numeric(subset_seed_rows["e0_delta_vs_e5"], errors="coerce").min()),
                "max_e0_delta_vs_e5": float(pd.to_numeric(subset_seed_rows["e0_delta_vs_e5"], errors="coerce").max()),
                "negative_seed_count": int((pd.to_numeric(subset_seed_rows["e0_delta_vs_e5"], errors="coerce") < 0.0).sum()),
                "per_image_loss_count": int((group["e0_min_delta_vs_e5"] < 0.0).sum()),
                "strict_public_safe_e0_count": int(group["strict_public_safe_e0"].astype(bool).sum()),
                "strict_only_public_accept_count": int(group["strict_only_gate_passed"].astype(bool).sum()),
                "reliability_aware_public_accept_count": int(group["reliability_aware_gate_passed"].astype(bool).sum()),
                "mean_flops_reduction_e0_vs_e5_g": float(
                    (safe_float(priors_by_candidate["E5"].get("flops_g")) or 0.0) - (safe_float(priors_by_candidate["E0"].get("flops_g")) or 0.0)
                ),
            }
        )
    dataset_summary_frame = pd.DataFrame(dataset_rows)

    decision, shoes_ready, handbags_ready, boundary, reasons = make_decision(
        dataset_summary_frame=dataset_summary_frame,
        paired_summary_frame=paired_frame,
        completion_ok=completion_ok,
    )
    decision_payload = {
        "completion": True,
        "public_validation_decision": decision,
        "edges2shoes_ready": shoes_ready,
        "edges2handbags_ready": handbags_ready,
        "claim_boundary": boundary,
        "allowed_manuscript_wording": (
            "A small public validation subset is reported only as an external no-go/refusal audit."
            if decision == "public_no_go_supported"
            else "The public validation subset identifies only a small candidate region and does not support broad public-dataset deployment."
        ),
        "blocked_manuscript_wording": "SOTA, public deployment readiness, broad cross-dataset transferability, or human preference validation.",
        "reasons": reasons,
    }
    write_json(gate_dir / "public_validation_decision.json", decision_payload)

    completion_payload["strict_only_public_accept_count"] = (
        int(paired_frame["strict_only_gate_passed"].astype(bool).sum()) if not paired_frame.empty else 0
    )
    completion_payload["reliability_aware_public_accept_count"] = (
        int(paired_frame["reliability_aware_gate_passed"].astype(bool).sum()) if not paired_frame.empty else 0
    )
    write_json(quality_dir / "public_validation_completion.json", completion_payload)

    method_summary_path = metrics_dir / "public_method_summary.csv"
    per_image_path = metrics_dir / "public_per_image_seed_utilities.csv"
    paired_path = metrics_dir / "public_paired_delta_summary.csv"
    dataset_path = metrics_dir / "public_dataset_delta_summary.csv"
    method_summary_frame.to_csv(method_summary_path, index=False, encoding="utf-8")
    per_image_frame.to_csv(per_image_path, index=False, encoding="utf-8")
    paired_frame.to_csv(paired_path, index=False, encoding="utf-8")
    dataset_summary_frame.to_csv(dataset_path, index=False, encoding="utf-8")

    summary_text = build_summary_markdown(
        decision_payload=decision_payload,
        dataset_summary_frame=dataset_summary_frame,
        completion_payload=completion_payload,
        method_summary_path=method_summary_path,
        per_image_path=per_image_path,
        paired_path=paired_path,
        dataset_path=dataset_path,
    )
    (output_dir / "PUBLIC_VALIDATION_SUMMARY.md").write_text(summary_text, encoding="utf-8")

    notes_lines = [
        "# ASC_PUBLIC_VALIDATION_NOTES",
        "",
        f"- completion: {str(decision_payload['completion']).lower()}",
        f"- public_validation_decision: {decision_payload['public_validation_decision']}",
        f"- claim_boundary: {decision_payload['claim_boundary']}",
        "- interpretation: The minimal public validation improves submission credibility by replacing a pure absence of public validation with an external no-go/refusal audit.",
        "- blocked_claims: public benchmark leadership, broad transferability, deployment readiness, and human preference validation remain blocked.",
        "",
    ]
    (paper_notes_dir / "ASC_PUBLIC_VALIDATION_NOTES.md").write_text("\n".join(notes_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
