from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from .adapter import AdapterConfig, SXDLConditionAdapter
from .data import EvaluationRecord, PairRecord, SplitBundle, create_or_load_split, normalize_lineart_image, select_validation_records
from .helper_metrics import compute_helper_tool_metrics
from .inference_archive import archive_evaluation_outputs
from .localized_outputs import export_localized_csv_artifact, export_localized_json_artifact, sync_evaluation_localized_outputs
from .memory import PeakMemoryMonitor
from .pipeline import InferenceEngine

_LPIPS_MODEL_CACHE: dict[str, Any] = {}


def _release_torch_memory(device: str | torch.device | None = None) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        normalized = torch.device(device) if device is not None else torch.device("cuda")
    except Exception:
        normalized = torch.device("cuda")
    if normalized.type != "cuda":
        return
    try:
        torch.cuda.synchronize(normalized)
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def resolve_metric_device(device: str | torch.device | None) -> str:
    if device is None:
        return "cpu"
    try:
        normalized = torch.device(device)
    except Exception:
        return str(device)
    if normalized.type == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return str(normalized)


@dataclass
class EvaluationResult:
    split: str
    num_samples: int
    fid: float | None
    precision: float | None
    recall: float | None
    f_score: float | None
    pr_curve_auc: float | None
    lpips_mean: float | None
    lpips_std: float | None
    ssim_mean: float
    ssim_std: float
    edge_consistency_mean: float
    edge_consistency_std: float
    color_bleeding_rate_mean: float
    color_bleeding_rate_std: float
    histogram_correlation_mean: float
    histogram_correlation_std: float
    inference_time_ms_mean: float
    inference_time_ms_std: float
    params_m: float
    flops_g: float | None
    gpu_memory_peak_gb: float | None
    gpu_memory_reserved_peak_gb: float | None
    cpu_memory_peak_gb: float | None
    eval_dir: str
    generated_dir: str
    target_dir: str
    lineart_dir: str
    checkpoint_dir: str
    archive_dir: str
    pr_curve_csv: str = ""
    pr_curve_plot: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpochMetricResult:
    fid: float | None
    precision: float | None
    recall: float | None
    f_score: float | None
    pr_curve_auc: float | None
    lpips: float | None
    ssim: float
    edge_consistency: float
    color_bleeding_rate: float
    histogram_correlation: float
    inference_time_ms: float
    subgroup_metrics: dict[str, Any]
    per_sample_rows: list[dict[str, Any]]
    pr_curve_points: list[dict[str, Any]]
    pr_metrics_error: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def should_compute_fid(epoch: int, total_epochs: int) -> bool:
    if total_epochs <= 10:
        return True
    if epoch > max(total_epochs - 10, 0):
        return True
    if epoch > max(total_epochs - 50, 0):
        return epoch % 2 == 0
    return epoch % 5 == 0


def _to_unit_tensor(image_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _load_json_payload(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_line_edges_from_lineart(lineart_np: np.ndarray) -> np.ndarray:
    if lineart_np.ndim == 3:
        gray = cv2.cvtColor(lineart_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = lineart_np
    normalized = normalize_lineart_image(gray.astype(np.uint8))
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(255 - binary, 50, 150)
    return (edges > 0).astype(np.uint8)


def _edge_consistency_f1(lineart_np: np.ndarray, generated_np: np.ndarray) -> float:
    target_edges = _extract_line_edges_from_lineart(lineart_np)
    generated_gray = cv2.cvtColor(generated_np, cv2.COLOR_RGB2GRAY)
    generated_edges = (cv2.Canny(generated_gray, 100, 200) > 0).astype(np.uint8)
    tp = int(np.logical_and(target_edges == 1, generated_edges == 1).sum())
    fp = int(np.logical_and(target_edges == 0, generated_edges == 1).sum())
    fn = int(np.logical_and(target_edges == 1, generated_edges == 0).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _color_bleeding_rate(lineart_np: np.ndarray, generated_np: np.ndarray) -> float:
    if lineart_np.ndim == 3:
        lineart_gray = cv2.cvtColor(lineart_np, cv2.COLOR_RGB2GRAY)
    else:
        lineart_gray = lineart_np
    normalized = normalize_lineart_image(lineart_gray.astype(np.uint8))
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    line_mask = binary < 128
    dilated = cv2.dilate(line_mask.astype(np.uint8) * 255, np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
    hsv = cv2.cvtColor(generated_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    if dilated.sum() == 0:
        return 0.0
    return float(saturation[dilated].mean())


def _histogram_correlation(target_np: np.ndarray, generated_np: np.ndarray) -> float:
    scores = []
    for channel in range(3):
        target_hist = cv2.calcHist([target_np], [channel], None, [256], [0, 256]).flatten()
        generated_hist = cv2.calcHist([generated_np], [channel], None, [256], [0, 256]).flatten()
        target_hist = target_hist / max(float(target_hist.sum()), 1.0)
        generated_hist = generated_hist / max(float(generated_hist.sum()), 1.0)
        corr = np.corrcoef(target_hist, generated_hist)[0, 1]
        if not np.isnan(corr):
            scores.append(float(corr))
    return float(sum(scores) / max(len(scores), 1))


def _line_density(lineart_np: np.ndarray) -> float:
    if lineart_np.ndim == 3:
        gray = cv2.cvtColor(lineart_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = lineart_np
    normalized = normalize_lineart_image(gray.astype(np.uint8))
    return float(np.mean(normalized < 180))


def _color_complexity(target_np: np.ndarray) -> float:
    hsv = cv2.cvtColor(target_np, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256]).flatten()
    hist = hist / max(float(hist.sum()), 1.0)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist))
    return float(entropy)


def _region_scale(lineart_np: np.ndarray) -> float:
    if lineart_np.ndim == 3:
        gray = cv2.cvtColor(lineart_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = lineart_np
    normalized = normalize_lineart_image(gray.astype(np.uint8))
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_regions = (binary > 128).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(white_regions, connectivity=8)
    if num_labels <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return float(np.mean(areas))


def count_trainable_params_m(run_dir: str | Path) -> float:
    metadata_path = Path(run_dir) / "run_metadata.json"
    if not metadata_path.exists():
        return 0.0
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    adapter_cfg = AdapterConfig(**payload["adapter_config"])
    from .ablation import ModuleFlags

    flags = ModuleFlags(**payload["flags"])
    model = SXDLConditionAdapter(adapter_cfg, flags)
    total = sum(parameter.numel() for parameter in model.parameters())

    lora_dir = Path(run_dir) / "lora"
    if lora_dir.exists():
        adapter_model = lora_dir / "adapter_model.safetensors"
        if adapter_model.exists():
            from safetensors.torch import load_file

            weights = load_file(str(adapter_model))
            total += sum(value.numel() for value in weights.values())
    return float(total / 1_000_000.0)


def estimate_adapter_flops_g(run_dir: str | Path, width: int, height: int) -> float | None:
    try:
        from ptflops import get_model_complexity_info
    except Exception:
        return None

    metadata_path = Path(run_dir) / "run_metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    adapter_cfg = AdapterConfig(**payload["adapter_config"])
    from .ablation import ModuleFlags

    flags = ModuleFlags(**payload["flags"])
    model = SXDLConditionAdapter(adapter_cfg, flags)
    macs, _ = get_model_complexity_info(model, (3, int(height), int(width)), as_strings=False, print_per_layer_stat=False)
    return float((macs * 2.0) / 1_000_000_000.0)


def _prepare_metric_models(device: str):
    fid_metric = None
    lpips_metric = None
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    except Exception:
        fid_metric = None
    try:
        import lpips

        lpips_metric = _LPIPS_MODEL_CACHE.get(device)
        if lpips_metric is None:
            lpips_metric = lpips.LPIPS(net="alex").to(device)
            lpips_metric.eval()
            _LPIPS_MODEL_CACHE[device] = lpips_metric
    except Exception:
        lpips_metric = None
    return fid_metric, lpips_metric


def _compute_pr_curve_auc(points: list[dict[str, Any]]) -> float | None:
    if len(points) < 2:
        return None
    ordered = sorted(points, key=lambda item: float(item.get("recall", 0.0)))
    recall = [float(item.get("recall", 0.0)) for item in ordered]
    precision = [float(item.get("precision", 0.0)) for item in ordered]
    return float(np.trapz(precision, recall))


def save_pr_curve_outputs(eval_dir: Path, points: list[dict[str, Any]]) -> tuple[str, str]:
    if not points:
        return "", ""
    csv_path = eval_dir / "pr_curve_points.csv"
    plot_path = eval_dir / "pr_curve.png"
    frame = pd.DataFrame(points)
    frame.to_csv(csv_path, index=False)
    export_localized_csv_artifact(csv_path, frame)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = frame.sort_values("recall")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.plot(ordered["recall"], ordered["precision"], color="#1d3557", linewidth=1.8, marker="o", markersize=3)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return str(csv_path.resolve()), str(plot_path.resolve())


def _compute_pr_metrics_from_dirs(
    *,
    generated_dir: str | Path,
    target_dir: str | Path,
    device: str,
    compute_curve: bool,
    neighborhoods: tuple[int, ...] = (1, 3, 5, 7, 11),
) -> dict[str, Any]:
    try:
        from torch_fidelity import calculate_metrics
    except Exception as exc:
        return {
            "precision": None,
            "recall": None,
            "f_score": None,
            "pr_curve_auc": None,
            "pr_curve_points": [],
            "pr_metrics_error": str(exc),
        }

    unique_neighborhoods = tuple(sorted({max(1, int(value)) for value in neighborhoods}))
    scalar_neighborhood = 3 if 3 in unique_neighborhoods else unique_neighborhoods[0]
    metric_kwargs = {
        "input1": str(Path(generated_dir).resolve()),
        "input2": str(Path(target_dir).resolve()),
        "cuda": device.startswith("cuda") and torch.cuda.is_available(),
        "batch_size": 8 if device.startswith("cuda") else 1,
        "fid": False,
        "kid": False,
        "isc": False,
        "prc": True,
        "verbose": False,
        "save_cpu_ram": True,
    }

    try:
        scalar_metrics = calculate_metrics(prc_neighborhood=scalar_neighborhood, **metric_kwargs)
        precision = float(scalar_metrics.get("precision")) if scalar_metrics.get("precision") is not None else None
        recall = float(scalar_metrics.get("recall")) if scalar_metrics.get("recall") is not None else None
        f_score = float(scalar_metrics.get("f_score")) if scalar_metrics.get("f_score") is not None else None
        curve_points: list[dict[str, Any]] = []
        if compute_curve:
            for neighborhood in unique_neighborhoods:
                current = scalar_metrics if neighborhood == scalar_neighborhood else calculate_metrics(
                    prc_neighborhood=neighborhood,
                    **metric_kwargs,
                )
                curve_points.append(
                    {
                        "neighborhood": int(neighborhood),
                        "precision": float(current.get("precision", 0.0)),
                        "recall": float(current.get("recall", 0.0)),
                        "f_score": float(current.get("f_score", 0.0)),
                    }
                )
        return {
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            "pr_curve_auc": _compute_pr_curve_auc(curve_points),
            "pr_curve_points": curve_points,
            "pr_metrics_error": "",
        }
    except Exception as exc:
        return {
            "precision": None,
            "recall": None,
            "f_score": None,
            "pr_curve_auc": None,
            "pr_curve_points": [],
            "pr_metrics_error": str(exc),
        }


def compute_generated_metrics(
    *,
    device: str,
    generated_rows: list[dict[str, Any]],
    compute_fid: bool,
    generated_dir: str | Path | None = None,
    target_dir: str | Path | None = None,
    compute_pr_curve: bool = False,
) -> EpochMetricResult:
    metric_device = resolve_metric_device(device)
    fid_metric, lpips_metric = _prepare_metric_models(metric_device)
    if not compute_fid:
        fid_metric = None

    lpips_scores: list[float] = []
    ssim_scores: list[float] = []
    edge_scores: list[float] = []
    bleeding_scores: list[float] = []
    hist_scores: list[float] = []
    times_ms: list[float] = []

    try:
        for row in generated_rows:
            target_np = row["target_np"]
            generated_np = row["generated_np"]
            lineart_np = row["lineart_np"]

            if fid_metric is not None:
                fid_metric.update(_to_unit_tensor(target_np).to(metric_device), real=True)
                fid_metric.update(_to_unit_tensor(generated_np).to(metric_device), real=False)

            if lpips_metric is not None:
                with torch.no_grad():
                    target_tensor = _to_unit_tensor(target_np).to(metric_device) * 2.0 - 1.0
                    generated_tensor = _to_unit_tensor(generated_np).to(metric_device) * 2.0 - 1.0
                    lpips_scores.append(float(lpips_metric(target_tensor, generated_tensor).mean().item()))

            ssim_scores.append(float(structural_similarity(target_np, generated_np, channel_axis=2, data_range=255)))
            edge_scores.append(_edge_consistency_f1(lineart_np, generated_np))
            bleeding_scores.append(_color_bleeding_rate(lineart_np, generated_np))
            hist_scores.append(_histogram_correlation(target_np, generated_np))
            times_ms.append(float(row["inference_time_ms"]))

            row["ssim"] = ssim_scores[-1]
            row["edge_consistency_f1"] = edge_scores[-1]
            row["color_bleeding_rate"] = bleeding_scores[-1]
            row["histogram_correlation"] = hist_scores[-1]
            if lpips_scores:
                row["lpips"] = lpips_scores[-1]

        subgroup_metrics = build_subgroup_metrics(generated_rows)
        pr_metrics = {
            "precision": None,
            "recall": None,
            "f_score": None,
            "pr_curve_auc": None,
            "pr_curve_points": [],
            "pr_metrics_error": "",
        }
        if generated_dir is not None and target_dir is not None:
            pr_metrics = _compute_pr_metrics_from_dirs(
                generated_dir=generated_dir,
                target_dir=target_dir,
                device=metric_device,
                compute_curve=compute_pr_curve,
            )
        return EpochMetricResult(
            fid=float(fid_metric.compute().item()) if fid_metric is not None else None,
            precision=pr_metrics["precision"],
            recall=pr_metrics["recall"],
            f_score=pr_metrics["f_score"],
            pr_curve_auc=pr_metrics["pr_curve_auc"],
            lpips=float(sum(lpips_scores) / max(len(lpips_scores), 1)) if lpips_scores else None,
            ssim=float(sum(ssim_scores) / max(len(ssim_scores), 1)),
            edge_consistency=float(sum(edge_scores) / max(len(edge_scores), 1)),
            color_bleeding_rate=float(sum(bleeding_scores) / max(len(bleeding_scores), 1)),
            histogram_correlation=float(sum(hist_scores) / max(len(hist_scores), 1)),
            inference_time_ms=float(sum(times_ms) / max(len(times_ms), 1)),
            subgroup_metrics=subgroup_metrics,
            per_sample_rows=generated_rows,
            pr_curve_points=pr_metrics["pr_curve_points"],
            pr_metrics_error=pr_metrics["pr_metrics_error"],
        )
    finally:
        if fid_metric is not None:
            del fid_metric
        if lpips_metric is not None:
            del lpips_metric
        _release_torch_memory(metric_device)


def count_trainable_params_from_models(unet, adapter) -> float:
    total = 0
    for module in (unet, adapter):
        for parameter in module.parameters():
            if parameter.requires_grad:
                total += int(parameter.numel())
    return float(total / 1_000_000.0)


def apply_helper_metric_overrides(
    *,
    metrics_payload: dict[str, Any],
    helper_tools_root: str | Path,
    reference_dir: str | Path,
    generated_dir: str | Path,
    eval_dir: str | Path,
    device: str,
) -> dict[str, Any]:
    helper_result = compute_helper_tool_metrics(
        helper_tools_root=helper_tools_root,
        reference_dir=reference_dir,
        generated_dir=generated_dir,
        eval_dir=eval_dir,
        device=device,
    )

    metrics_payload["helper_metrics_available"] = helper_result.get("available", False)
    metrics_payload["helper_metric_reports"] = helper_result.get("reports", {})
    metrics_payload["helper_metric_errors"] = helper_result.get("errors", {})

    if helper_result.get("fid") is not None:
        metrics_payload["fid_internal"] = metrics_payload.get("fid")
        metrics_payload["fid"] = helper_result["fid"]
        metrics_payload["fid_source"] = "helper_tool"
    else:
        metrics_payload["fid_source"] = "internal"

    if helper_result.get("lpips_mean") is not None:
        original_lpips = metrics_payload.get("lpips_mean", metrics_payload.get("lpips"))
        metrics_payload["lpips_internal"] = original_lpips
        metrics_payload["lpips"] = helper_result["lpips_mean"]
        metrics_payload["lpips_mean"] = helper_result["lpips_mean"]
        metrics_payload["lpips_source"] = "helper_tool"
        if helper_result.get("lpips_std") is not None:
            metrics_payload["lpips_std"] = helper_result["lpips_std"]
    else:
        metrics_payload["lpips_source"] = "internal"

    if helper_result.get("ssim_mean") is not None:
        original_ssim = metrics_payload.get("ssim_mean", metrics_payload.get("ssim"))
        metrics_payload["ssim_internal"] = original_ssim
        metrics_payload["ssim"] = helper_result["ssim_mean"]
        metrics_payload["ssim_mean"] = helper_result["ssim_mean"]
        metrics_payload["ssim_source"] = "helper_tool"
        if helper_result.get("ssim_std") is not None:
            metrics_payload["ssim_std"] = helper_result["ssim_std"]
    else:
        metrics_payload["ssim_source"] = "internal"

    return metrics_payload


def build_subgroup_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    frame = pd.DataFrame(
        [
            {
                "image_id": row["image_id"],
                "line_density": _line_density(row["lineart_np"]),
                "color_complexity": _color_complexity(row["target_np"]),
                "region_scale": _region_scale(row["lineart_np"]),
                "ssim": row.get("ssim"),
                "edge_consistency_f1": row.get("edge_consistency_f1"),
                "color_bleeding_rate": row.get("color_bleeding_rate"),
                "histogram_correlation": row.get("histogram_correlation"),
                "lpips": row.get("lpips"),
            }
            for row in rows
        ]
    )
    if frame.empty:
        return {}

    for feature_name, low_label, high_label in [
        ("line_density", "sparse_lineart", "dense_lineart"),
        ("color_complexity", "simple_color", "complex_color"),
        ("region_scale", "small_region", "large_region"),
    ]:
        threshold = float(frame[feature_name].median())
        frame[f"{feature_name}_group"] = np.where(frame[feature_name] <= threshold, low_label, high_label)

    metrics = {}
    for feature_name in ["line_density", "color_complexity", "region_scale"]:
        group_col = f"{feature_name}_group"
        grouped = (
            frame.groupby(group_col)[["ssim", "edge_consistency_f1", "color_bleeding_rate", "histogram_correlation", "lpips"]]
            .mean(numeric_only=True)
            .reset_index()
        )
        metrics[feature_name] = grouped.to_dict(orient="records")
    return metrics


def save_per_sample_outputs(eval_dir: Path, rows: list[dict[str, Any]], subgroup_metrics: dict[str, Any]) -> None:
    serializable_rows = []
    for row in rows:
        serializable_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"target_np", "generated_np", "lineart_np"}
            }
        )
    per_sample_frame = pd.DataFrame(serializable_rows)
    per_sample_path = eval_dir / "per_sample_metrics.csv"
    subgroup_path = eval_dir / "subgroup_metrics.json"
    per_sample_frame.to_csv(per_sample_path, index=False)
    with subgroup_path.open("w", encoding="utf-8") as handle:
        json.dump(subgroup_metrics, handle, ensure_ascii=False, indent=2)
    export_localized_csv_artifact(per_sample_path, per_sample_frame)
    export_localized_json_artifact(subgroup_path, subgroup_metrics)


def load_generated_rows_from_eval_dir(eval_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_root = Path(eval_dir)
    generated_dir = eval_root / "generated"
    target_dir = eval_root / "target"
    lineart_dir = eval_root / "lineart"
    manifest = _load_json_payload(eval_root / "generation_records.json")

    manifest_rows = manifest.get("rows", [])
    manifest_by_name: dict[str, dict[str, Any]] = {}
    manifest_by_image_id: dict[str, dict[str, Any]] = {}
    if isinstance(manifest_rows, list):
        for row in manifest_rows:
            if not isinstance(row, dict):
                continue
            file_name = str(row.get("file_name", "")).strip()
            image_id = str(row.get("image_id", "")).strip()
            if file_name:
                manifest_by_name[file_name] = row
            if image_id:
                manifest_by_image_id[image_id] = row

    rows: list[dict[str, Any]] = []
    for generated_path in sorted(generated_dir.glob("*")):
        if not generated_path.is_file():
            continue
        target_path = target_dir / generated_path.name
        lineart_path = lineart_dir / generated_path.name
        if not target_path.exists() or not lineart_path.exists():
            continue

        image_id = generated_path.stem
        if "_" in image_id:
            image_id = image_id.split("_", 1)[1]
        manifest_row = manifest_by_name.get(generated_path.name) or manifest_by_image_id.get(image_id) or {}
        rows.append(
            {
                "image_id": str(manifest_row.get("image_id", image_id)),
                "generated_path": str(generated_path.resolve()),
                "target_path": str(target_path.resolve()),
                "lineart_path": str(lineart_path.resolve()),
                "inference_time_ms": float(manifest_row.get("inference_time_ms", 0.0) or 0.0),
                "target_np": np.array(Image.open(target_path).convert("RGB")),
                "generated_np": np.array(Image.open(generated_path).convert("RGB")),
                "lineart_np": np.array(Image.open(lineart_path).convert("RGB")),
            }
        )
    return rows, manifest


def compute_saved_epoch_metrics(
    *,
    eval_dir: str | Path,
    device: str | torch.device,
    compute_fid: bool,
    compute_pr_curve: bool,
    helper_tools_root: str | Path = "",
    archive_root: str | Path | None = None,
    group_id: str = "",
    seed: int = 0,
    epoch: int | None = None,
    params_m: float | None = None,
    flops_g: float | None = None,
    validation_source: str = "",
    validation_note: str = "",
) -> dict[str, Any]:
    eval_root = Path(eval_dir)
    generated_dir = eval_root / "generated"
    target_dir = eval_root / "target"
    lineart_dir = eval_root / "lineart"
    rows, manifest = load_generated_rows_from_eval_dir(eval_root)
    existing_metrics = _load_json_payload(eval_root / "metrics.json")
    if not rows:
        raise ValueError(f"No generated evaluation samples found under {eval_root}.")

    memory_monitor = PeakMemoryMonitor(device=device).start()
    metric_device = resolve_metric_device(device)
    try:
        metric_result = compute_generated_metrics(
            device=metric_device,
            generated_rows=rows,
            compute_fid=compute_fid,
            generated_dir=generated_dir,
            target_dir=target_dir,
            compute_pr_curve=compute_pr_curve,
        )
        save_per_sample_outputs(eval_root, metric_result.per_sample_rows, metric_result.subgroup_metrics)
        metrics_payload = metric_result.to_dict()
        metrics_payload["per_sample_rows"] = [
            {
                key: value
                for key, value in row.items()
                if key not in {"target_np", "generated_np", "lineart_np"}
            }
            for row in metric_result.per_sample_rows
        ]
        metrics_payload["eval_dir"] = str(eval_root.resolve())
        metrics_payload["generated_dir"] = str(generated_dir.resolve())
        metrics_payload["target_dir"] = str(target_dir.resolve())
        metrics_payload["lineart_dir"] = str(lineart_dir.resolve())
        metrics_payload["split"] = str(existing_metrics.get("split") or manifest.get("split") or "validation_epoch")
        metrics_payload["group_id"] = str(group_id or manifest.get("group_id", ""))
        metrics_payload["seed"] = int(seed or manifest.get("seed", 0) or 0)
        metrics_payload["epoch"] = int(epoch or manifest.get("epoch", 0) or 0)
        metrics_payload["checkpoint_dir"] = str(
            existing_metrics.get("checkpoint_dir")
            or manifest.get("checkpoint_dir", "")
        )
        metrics_payload["fid_computed"] = bool(compute_fid)
        metrics_payload["validation_source"] = validation_source or str(manifest.get("validation_source", ""))
        metrics_payload["validation_note"] = validation_note or str(manifest.get("validation_note", ""))
        metrics_payload["params_m"] = params_m if params_m is not None else manifest.get("params_m")
        metrics_payload["flops_g"] = flops_g if flops_g is not None else manifest.get("flops_g")
        metrics_payload["generated_samples_count"] = len(rows)
        metrics_payload["generation_records_path"] = str((eval_root / "generation_records.json").resolve())
        peak_memory = memory_monitor.stop()
        metrics_payload["gpu_memory_peak_gb"] = peak_memory.get("gpu_memory_peak_gb")
        metrics_payload["gpu_memory_reserved_peak_gb"] = peak_memory.get("gpu_memory_reserved_peak_gb")
        metrics_payload["cpu_memory_peak_gb"] = peak_memory.get("cpu_memory_peak_gb")
        metrics_payload["memory_unit"] = peak_memory.get("memory_unit", "GB")
        metrics_payload["gpu_memory_peak"] = peak_memory.get("gpu_memory_peak_gb")
        metrics_payload["cpu_memory_peak"] = peak_memory.get("cpu_memory_peak_gb")
        pr_curve_csv, pr_curve_plot = ("", "")
        if metric_result.pr_curve_points:
            pr_curve_csv, pr_curve_plot = save_pr_curve_outputs(eval_root, metric_result.pr_curve_points)
        metrics_payload["pr_curve_csv"] = pr_curve_csv
        metrics_payload["pr_curve_plot"] = pr_curve_plot
        if helper_tools_root:
            metrics_payload = apply_helper_metric_overrides(
                metrics_payload=metrics_payload,
                helper_tools_root=helper_tools_root,
                reference_dir=target_dir,
                generated_dir=generated_dir,
                eval_dir=eval_root,
                device=metric_device,
            )
        if archive_root:
            metrics_payload["archive_dir"] = archive_evaluation_outputs(
                source_eval_dir=eval_root,
                archive_root=archive_root,
                archive_kind="training_validation",
                group_id=group_id or str(manifest.get("group_id", "")),
                seed=int(seed or manifest.get("seed", 0)),
                epoch=int(epoch or manifest.get("epoch", 0)),
            )
        with (eval_root / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)
        sync_evaluation_localized_outputs(eval_root)
        return metrics_payload
    finally:
        memory_monitor.stop()


def evaluate_run(
    *,
    run_dir: str | Path,
    split_bundle: SplitBundle,
    split_name: str,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_scale: float,
    seed: int,
    width: int,
    height: int,
    max_samples: int,
    output_dir: str | Path,
    device: str = "cuda",
    dtype: str = "fp16",
    checkpoint_dir: str | Path | None = None,
) -> EvaluationResult:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "run_summary.json"
    run_summary: dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            run_summary = json.load(handle)

    metadata_path = run_dir / "run_metadata.json"
    trainer_config: dict[str, Any] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            trainer_config = json.load(handle).get("trainer_config", {})

    group_id = str(
        run_summary.get("group_id")
        or trainer_config.get("group_id")
        or run_dir.parent.name
        or "unknown"
    )
    seed_value = int(run_summary.get("seed", seed))
    archive_root = trainer_config.get("inference_archive_root", "")

    validation_source = ""
    validation_note = ""

    if split_name == "test":
        if split_bundle.test:
            records = [
                EvaluationRecord(
                    image_id=record.image_id,
                    lineart_path=record.lineart_path,
                    color_path=record.color_path,
                    has_reference=True,
                    source="split_test",
                )
                for record in split_bundle.test
            ]
            validation_source = "split_test"
            validation_note = "Using the internal test split for standalone evaluation."
        else:
            selection = select_validation_records(
                split_val_records=list(split_bundle.val),
                validation_dataset_root=trainer_config.get("validation_dataset_root", ""),
                validation_color_dir_name=trainer_config.get("validation_color_dir_name", "彩色数据"),
                validation_lineart_dir_name=trainer_config.get("validation_lineart_dir_name", "黑白线稿"),
                prefer_external_validation_dataset=bool(trainer_config.get("prefer_external_validation_dataset", True)),
            )
            if selection.source == "external_paired_validation" and selection.metric_records:
                records = list(selection.metric_records)
                validation_source = selection.source
                validation_note = (
                    "Requested test evaluation, but the internal test split is empty. "
                    "Falling back to the paired external validation dataset."
                )
            else:
                raise ValueError(
                    "The internal test split is empty, and no paired external validation dataset is available for evaluation."
                )
    elif split_name == "val":
        selection = select_validation_records(
            split_val_records=list(split_bundle.val),
            validation_dataset_root=trainer_config.get("validation_dataset_root", ""),
            validation_color_dir_name=trainer_config.get("validation_color_dir_name", "彩色数据"),
            validation_lineart_dir_name=trainer_config.get("validation_lineart_dir_name", "黑白线稿"),
            prefer_external_validation_dataset=bool(trainer_config.get("prefer_external_validation_dataset", True)),
        )
        records = list(selection.metric_records)
        validation_source = selection.source
        validation_note = selection.note
    else:
        records = [
            EvaluationRecord(
                image_id=record.image_id,
                lineart_path=record.lineart_path,
                color_path=record.color_path,
                has_reference=True,
                source="split_train",
            )
            for record in split_bundle.train
        ]
        validation_source = "split_train"
        validation_note = "Using the training split for standalone evaluation."
    if not records:
        raise ValueError(f"No evaluation records available for split '{split_name}'.")
    records = records[: max(1, int(max_samples))]

    memory_monitor = PeakMemoryMonitor(device=device).start()
    metric_device = resolve_metric_device(device)
    engine = InferenceEngine(run_dir=run_dir, checkpoint_dir=checkpoint_dir, device=device, dtype=dtype)
    checkpoint_label = Path(checkpoint_dir).name if checkpoint_dir else "final"
    eval_dir = output_dir / f"{split_name}_{len(records)}samples_{checkpoint_label}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = eval_dir / "generated"
    target_dir = eval_dir / "target"
    lineart_dir = eval_dir / "lineart"
    generated_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    lineart_dir.mkdir(parents=True, exist_ok=True)

    generated_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        lineart_np = np.array(Image.open(record.lineart_path).convert("RGB"))
        target_np = np.array(Image.open(record.color_path).convert("RGB").resize((width, height), Image.LANCZOS))
        started = time.perf_counter()
        generated_np = engine.colorize(
            lineart_np,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_scale=controlnet_scale,
            seed=seed,
            width=width,
            height=height,
        )
        inference_time_ms = (time.perf_counter() - started) * 1000.0

        file_name = f"{index:03d}_{record.image_id}.png"
        generated_path = generated_dir / file_name
        target_path = target_dir / file_name
        lineart_path = lineart_dir / file_name
        Image.fromarray(generated_np).save(generated_path)
        Image.fromarray(target_np).save(target_path)
        Image.fromarray(np.array(Image.open(record.lineart_path).convert("RGB").resize((width, height), Image.NEAREST))).save(lineart_path)

        generated_rows.append(
            {
                "image_id": record.image_id,
                "generated_path": str(generated_path.resolve()),
                "target_path": str(target_path.resolve()),
                "lineart_path": str(lineart_path.resolve()),
                "inference_time_ms": inference_time_ms,
                "target_np": target_np,
                "generated_np": generated_np,
                "lineart_np": np.array(Image.open(record.lineart_path).convert("RGB").resize((width, height), Image.NEAREST)),
            }
        )

    del engine
    _release_torch_memory(device)
    metric_result = compute_generated_metrics(
        device=metric_device,
        generated_rows=generated_rows,
        compute_fid=True,
        generated_dir=generated_dir,
        target_dir=target_dir,
        compute_pr_curve=True,
    )
    save_per_sample_outputs(eval_dir, metric_result.per_sample_rows, metric_result.subgroup_metrics)
    lpips_values = [row["lpips"] for row in metric_result.per_sample_rows if "lpips" in row]
    ssim_values = [row["ssim"] for row in metric_result.per_sample_rows]
    edge_values = [row["edge_consistency_f1"] for row in metric_result.per_sample_rows]
    bleeding_values = [row["color_bleeding_rate"] for row in metric_result.per_sample_rows]
    hist_values = [row["histogram_correlation"] for row in metric_result.per_sample_rows]
    time_values = [row["inference_time_ms"] for row in metric_result.per_sample_rows]
    peak_memory = memory_monitor.stop()
    metrics_payload = {
        "split": split_name,
        "num_samples": len(records),
        "fid": metric_result.fid,
        "precision": metric_result.precision,
        "recall": metric_result.recall,
        "f_score": metric_result.f_score,
        "pr_curve_auc": metric_result.pr_curve_auc,
        "lpips_mean": float(sum(lpips_values) / max(len(lpips_values), 1)) if lpips_values else None,
        "lpips_std": _safe_std(lpips_values) if lpips_values else None,
        "ssim_mean": float(sum(ssim_values) / max(len(ssim_values), 1)),
        "ssim_std": _safe_std(ssim_values),
        "edge_consistency_mean": float(sum(edge_values) / max(len(edge_values), 1)),
        "edge_consistency_std": _safe_std(edge_values),
        "color_bleeding_rate_mean": float(sum(bleeding_values) / max(len(bleeding_values), 1)),
        "color_bleeding_rate_std": _safe_std(bleeding_values),
        "histogram_correlation_mean": float(sum(hist_values) / max(len(hist_values), 1)),
        "histogram_correlation_std": _safe_std(hist_values),
        "inference_time_ms_mean": float(sum(time_values) / max(len(time_values), 1)),
        "inference_time_ms_std": _safe_std(time_values),
        "params_m": count_trainable_params_m(run_dir),
        "flops_g": estimate_adapter_flops_g(run_dir, width=width, height=height),
        "gpu_memory_peak_gb": peak_memory.get("gpu_memory_peak_gb"),
        "gpu_memory_reserved_peak_gb": peak_memory.get("gpu_memory_reserved_peak_gb"),
        "cpu_memory_peak_gb": peak_memory.get("cpu_memory_peak_gb"),
        "memory_unit": peak_memory.get("memory_unit", "GB"),
        "gpu_memory_peak": peak_memory.get("gpu_memory_peak_gb"),
        "cpu_memory_peak": peak_memory.get("cpu_memory_peak_gb"),
        "eval_dir": str(eval_dir.resolve()),
        "generated_dir": str(generated_dir.resolve()),
        "target_dir": str(target_dir.resolve()),
        "lineart_dir": str(lineart_dir.resolve()),
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()) if checkpoint_dir else "",
    }
    metrics_payload["pr_metrics_error"] = metric_result.pr_metrics_error
    pr_curve_csv, pr_curve_plot = save_pr_curve_outputs(eval_dir, metric_result.pr_curve_points)
    metrics_payload["pr_curve_csv"] = pr_curve_csv
    metrics_payload["pr_curve_plot"] = pr_curve_plot
    metrics_payload["validation_source"] = validation_source
    metrics_payload["validation_note"] = validation_note
    helper_tools_root = trainer_config.get("helper_tools_root", "")
    if helper_tools_root:
        metrics_payload = apply_helper_metric_overrides(
            metrics_payload=metrics_payload,
            helper_tools_root=helper_tools_root,
            reference_dir=target_dir,
            generated_dir=generated_dir,
            eval_dir=eval_dir,
            device=metric_device,
        )
    metrics_payload["archive_dir"] = archive_evaluation_outputs(
        source_eval_dir=eval_dir,
        archive_root=archive_root,
        archive_kind="standalone_evaluation",
        group_id=group_id,
        seed=seed_value,
        split_name=split_name,
        checkpoint_label=checkpoint_label,
    )
    with (eval_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)
    sync_evaluation_localized_outputs(eval_dir)
    return EvaluationResult(
        split=metrics_payload["split"],
        num_samples=metrics_payload["num_samples"],
        fid=metrics_payload.get("fid"),
        precision=metrics_payload.get("precision"),
        recall=metrics_payload.get("recall"),
        f_score=metrics_payload.get("f_score"),
        pr_curve_auc=metrics_payload.get("pr_curve_auc"),
        lpips_mean=metrics_payload.get("lpips_mean", metrics_payload.get("lpips")),
        lpips_std=metrics_payload.get("lpips_std"),
        ssim_mean=metrics_payload.get("ssim_mean", metrics_payload.get("ssim")),
        ssim_std=metrics_payload.get("ssim_std", 0.0),
        edge_consistency_mean=metrics_payload["edge_consistency_mean"],
        edge_consistency_std=metrics_payload["edge_consistency_std"],
        color_bleeding_rate_mean=metrics_payload["color_bleeding_rate_mean"],
        color_bleeding_rate_std=metrics_payload["color_bleeding_rate_std"],
        histogram_correlation_mean=metrics_payload["histogram_correlation_mean"],
        histogram_correlation_std=metrics_payload["histogram_correlation_std"],
        inference_time_ms_mean=metrics_payload["inference_time_ms_mean"],
        inference_time_ms_std=metrics_payload["inference_time_ms_std"],
        params_m=metrics_payload["params_m"],
        flops_g=metrics_payload["flops_g"],
        gpu_memory_peak_gb=metrics_payload.get("gpu_memory_peak_gb"),
        gpu_memory_reserved_peak_gb=metrics_payload.get("gpu_memory_reserved_peak_gb"),
        cpu_memory_peak_gb=metrics_payload.get("cpu_memory_peak_gb"),
        eval_dir=metrics_payload["eval_dir"],
        generated_dir=metrics_payload["generated_dir"],
        target_dir=metrics_payload["target_dir"],
        lineart_dir=metrics_payload["lineart_dir"],
        checkpoint_dir=metrics_payload["checkpoint_dir"],
        archive_dir=metrics_payload["archive_dir"],
        pr_curve_csv=metrics_payload.get("pr_curve_csv", ""),
        pr_curve_plot=metrics_payload.get("pr_curve_plot", ""),
    )


def load_or_create_split_from_metadata(run_dir: str | Path) -> SplitBundle:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    trainer_config = payload["trainer_config"]
    return create_or_load_split(
        dataset_root=trainer_config["dataset_root"],
        color_dir_name=trainer_config["color_dir_name"],
        lineart_dir_name=trainer_config["lineart_dir_name"],
        train_ratio=float(trainer_config["train_ratio"]),
        val_ratio=float(trainer_config["val_ratio"]),
        test_ratio=float(trainer_config["test_ratio"]),
        split_seed=int(trainer_config["split_seed"]),
        output_path=run_dir / "dataset_split.json",
        use_all_training_pairs_for_training=bool(trainer_config.get("use_all_training_pairs_for_training", False)),
    )
