from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pandas as pd


def _extract_float(text: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)}=([^\s]+)", text)
    if not match:
        return None
    raw = match.group(1).strip()
    if raw.lower() in {"n/a", "nan", "none"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _extract_path(text: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}=([^\s]+)", text)
    if not match:
        return ""
    return match.group(1).strip()


def _safe_std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(pd.Series(values, dtype="float64").std(ddof=1))


def _load_batch_metric_values(csv_path: str, metric_column: str) -> list[float]:
    if not csv_path:
        return []
    path = Path(csv_path)
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if metric_column not in frame.columns:
        return []
    numeric = pd.to_numeric(frame[metric_column], errors="coerce").dropna()
    return [float(value) for value in numeric.tolist()]


def _run_command(command: list[str], env: dict[str, str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            env=env,
        )
    except Exception as exc:
        return False, str(exc)
    output = (completed.stdout or "").strip()
    error_output = (completed.stderr or "").strip()
    if completed.returncode != 0:
        return False, output or error_output or f"exit_code={completed.returncode}"
    return True, output or error_output


def compute_helper_tool_metrics(
    *,
    helper_tools_root: str | Path,
    reference_dir: str | Path,
    generated_dir: str | Path,
    eval_dir: str | Path,
    device: str,
) -> dict[str, Any]:
    helper_root = Path(helper_tools_root)
    if not helper_root.exists():
        return {
            "available": False,
            "errors": {"helper_tools_root": f"Missing helper tools root: {helper_root}"},
        }

    runtime_root = Path(eval_dir) / "helper_tools_runtime"
    report_root = runtime_root / "reports"
    shared_cache_root = helper_root / "_runtime_cache"
    shared_torch_home = shared_cache_root / "torch-cache"
    env = os.environ.copy()
    env["ABLATION_RUNTIME_ROOT"] = str(runtime_root.resolve())
    env["TORCH_HOME"] = str(shared_torch_home.resolve())
    env["PYTHONUNBUFFERED"] = "1"
    runtime_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    shared_torch_home.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "available": True,
        "fid": None,
        "lpips_mean": None,
        "lpips_std": None,
        "ssim_mean": None,
        "ssim_std": None,
        "reports": {},
        "errors": {},
    }

    normalized_device = "cuda" if "cuda" in device.lower() else "cpu"

    tool_specs = [
        (
            "lpips",
            helper_root / "LPIPS" / "LPIPS.py",
            [
                sys.executable,
                str(helper_root / "LPIPS" / "LPIPS.py"),
                "--batch",
                str(reference_dir),
                str(generated_dir),
                "--device",
                normalized_device,
                "--resize",
                "image1",
                "--quiet",
            ],
        ),
        (
            "ssim",
            helper_root / "SSIM" / "SSIM-tool.py",
            [
                sys.executable,
                str(helper_root / "SSIM" / "SSIM-tool.py"),
                "--batch",
                str(reference_dir),
                str(generated_dir),
                "--mode",
                "rgb",
                "--resize",
                "image1",
                "--quiet",
            ],
        ),
        (
            "fid",
            helper_root / "FID" / "FID-tool.py",
            [
                sys.executable,
                str(helper_root / "FID" / "FID-tool.py"),
                str(reference_dir),
                str(generated_dir),
                "--device",
                normalized_device,
                "--quiet",
            ],
        ),
    ]

    raw_outputs: dict[str, str] = {}
    for metric_name, script_path, command in tool_specs:
        if not script_path.exists():
            results["errors"][metric_name] = f"Missing helper script: {script_path}"
            continue
        if metric_name == "fid" and find_spec("pytorch_fid") is None:
            results["errors"][metric_name] = "Missing Python dependency: pytorch_fid"
            continue
        ok, output = _run_command(command, env)
        raw_outputs[metric_name] = output
        if not ok:
            results["errors"][metric_name] = output
            continue

        if metric_name == "lpips":
            csv_path = _extract_path(output, "csv")
            values = _load_batch_metric_values(csv_path, "lpips")
            results["lpips_mean"] = _extract_float(output, "average_lpips")
            results["lpips_std"] = _safe_std(values)
            if csv_path:
                results["reports"]["lpips_csv"] = str(Path(csv_path).resolve())
        elif metric_name == "ssim":
            csv_path = _extract_path(output, "csv")
            values = _load_batch_metric_values(csv_path, "ssim")
            results["ssim_mean"] = _extract_float(output, "average_ssim")
            results["ssim_std"] = _safe_std(values)
            if csv_path:
                results["reports"]["ssim_csv"] = str(Path(csv_path).resolve())
        else:
            try:
                results["fid"] = float(output.strip())
            except ValueError:
                results["errors"][metric_name] = output

    summary_path = Path(eval_dir) / "helper_tool_metrics.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **results,
                "raw_outputs": raw_outputs,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    results["reports"]["summary_json"] = str(summary_path.resolve())
    return results
