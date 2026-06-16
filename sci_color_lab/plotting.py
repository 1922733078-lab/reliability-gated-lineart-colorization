from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .localized_outputs import export_localized_json_artifact


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_rows(log_path: Path, *, event_name: str | None = None) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if event_name is not None and str(payload.get("event", "")) != event_name:
                continue
            rows.append(payload)
    return rows


def _extract_series(rows: list[dict[str, Any]], *, x_key: str, y_key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        try:
            x_value = int(row.get(x_key, 0))
            y_value = float(row.get(y_key))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(y_value):
            continue
        xs.append(x_value)
        ys.append(y_value)
    return xs, ys


def _plot_single_curve(
    steps: list[int],
    values: list[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
    save_path: Path,
) -> str:
    if not steps or not values:
        return ""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.plot(steps, values, color=color, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return str(save_path.resolve())


def _plot_curve_grid(
    specs: list[dict[str, Any]],
    *,
    save_path: Path,
    title: str,
) -> str:
    valid_specs = [spec for spec in specs if spec.get("x") and spec.get("y")]
    if not valid_specs:
        return ""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ncols = 2
    nrows = math.ceil(len(valid_specs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), dpi=160)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for axis in axes_list:
        axis.set_visible(False)
    for axis, spec in zip(axes_list, valid_specs):
        axis.set_visible(True)
        axis.plot(spec["x"], spec["y"], color=spec["color"], linewidth=1.8)
        axis.set_title(spec["title"])
        axis.set_xlabel(spec["xlabel"])
        axis.set_ylabel(spec["ylabel"])
        axis.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return str(save_path.resolve())


def export_training_curves(run_dir: str | Path) -> dict[str, str]:
    run_dir = Path(run_dir)
    train_rows = _load_rows(run_dir / "logs" / "train.jsonl", event_name="train")
    epoch_rows = _load_rows(run_dir / "logs" / "metrics.jsonl", event_name="epoch_end")
    if not train_rows and not epoch_rows:
        return {}

    plots_dir = run_dir / "plots"
    train_steps, loss_values = _extract_series(train_rows, x_key="global_step", y_key="loss")
    _, lr_values = _extract_series(train_rows, x_key="global_step", y_key="lr")

    loss_curve_path = _plot_single_curve(
        train_steps,
        loss_values,
        title="Training Loss vs Step",
        xlabel="Step",
        ylabel="Loss",
        color="#2d6a4f",
        save_path=plots_dir / "loss_curve.png",
    )
    lr_curve_path = _plot_single_curve(
        train_steps,
        lr_values,
        title="Learning Rate vs Step",
        xlabel="Step",
        ylabel="Learning Rate",
        color="#c44536",
        save_path=plots_dir / "lr_curve.png",
    )

    step_dashboard_path = _plot_curve_grid(
        [
            {
                "x": _extract_series(train_rows, x_key="global_step", y_key="adapter_bottleneck_reconstruction_loss")[0],
                "y": _extract_series(train_rows, x_key="global_step", y_key="adapter_bottleneck_reconstruction_loss")[1],
                "title": "Adapter Bottleneck Recon",
                "xlabel": "Step",
                "ylabel": "Loss",
                "color": "#355070",
            },
            {
                "x": _extract_series(train_rows, x_key="global_step", y_key="beta_vae_kl_loss")[0],
                "y": _extract_series(train_rows, x_key="global_step", y_key="beta_vae_kl_loss")[1],
                "title": "KL Loss",
                "xlabel": "Step",
                "ylabel": "KL",
                "color": "#6d597a",
            },
            {
                "x": _extract_series(train_rows, x_key="global_step", y_key="beta_vae_beta")[0],
                "y": _extract_series(train_rows, x_key="global_step", y_key="beta_vae_beta")[1],
                "title": "KL Anneal Beta",
                "xlabel": "Step",
                "ylabel": "Beta",
                "color": "#b56576",
            },
            {
                "x": _extract_series(train_rows, x_key="global_step", y_key="kl_per_dim_mean")[0],
                "y": _extract_series(train_rows, x_key="global_step", y_key="kl_per_dim_mean")[1],
                "title": "KL Per Dim Mean",
                "xlabel": "Step",
                "ylabel": "KL/Dim",
                "color": "#e56b6f",
            },
            {
                "x": _extract_series(train_rows, x_key="global_step", y_key="z_norm_l2_mean")[0],
                "y": _extract_series(train_rows, x_key="global_step", y_key="z_norm_l2_mean")[1],
                "title": "Z Norm L2 Mean",
                "xlabel": "Step",
                "ylabel": "L2",
                "color": "#eaac8b",
            },
            {
                "x": _extract_series(train_rows, x_key="global_step", y_key="posterior_mu_abs_mean")[0],
                "y": _extract_series(train_rows, x_key="global_step", y_key="posterior_mu_abs_mean")[1],
                "title": "Posterior Mu Abs Mean",
                "xlabel": "Step",
                "ylabel": "|mu|",
                "color": "#1b4965",
            },
        ],
        save_path=plots_dir / "latent_step_dashboard.png",
        title="Realtime Latent Dashboard (Step)",
    )
    epoch_dashboard_path = _plot_curve_grid(
        [
            {
                "x": _extract_series(epoch_rows, x_key="epoch", y_key="adapter_bottleneck_reconstruction_loss")[0],
                "y": _extract_series(epoch_rows, x_key="epoch", y_key="adapter_bottleneck_reconstruction_loss")[1],
                "title": "Adapter Bottleneck Recon",
                "xlabel": "Epoch",
                "ylabel": "Loss",
                "color": "#355070",
            },
            {
                "x": _extract_series(epoch_rows, x_key="epoch", y_key="beta_vae_kl_loss")[0],
                "y": _extract_series(epoch_rows, x_key="epoch", y_key="beta_vae_kl_loss")[1],
                "title": "KL Loss",
                "xlabel": "Epoch",
                "ylabel": "KL",
                "color": "#6d597a",
            },
            {
                "x": _extract_series(epoch_rows, x_key="epoch", y_key="kl_per_dim_mean")[0],
                "y": _extract_series(epoch_rows, x_key="epoch", y_key="kl_per_dim_mean")[1],
                "title": "KL Per Dim Mean",
                "xlabel": "Epoch",
                "ylabel": "KL/Dim",
                "color": "#e56b6f",
            },
            {
                "x": _extract_series(epoch_rows, x_key="epoch", y_key="z_norm_l2_mean")[0],
                "y": _extract_series(epoch_rows, x_key="epoch", y_key="z_norm_l2_mean")[1],
                "title": "Z Norm L2 Mean",
                "xlabel": "Epoch",
                "ylabel": "L2",
                "color": "#eaac8b",
            },
            {
                "x": _extract_series(epoch_rows, x_key="epoch", y_key="posterior_logvar_mean")[0],
                "y": _extract_series(epoch_rows, x_key="epoch", y_key="posterior_logvar_mean")[1],
                "title": "Posterior LogVar Mean",
                "xlabel": "Epoch",
                "ylabel": "logvar",
                "color": "#4d908e",
            },
            {
                "x": _extract_series(epoch_rows, x_key="epoch", y_key="free_bits_active_fraction")[0],
                "y": _extract_series(epoch_rows, x_key="epoch", y_key="free_bits_active_fraction")[1],
                "title": "Free Bits Active Fraction",
                "xlabel": "Epoch",
                "ylabel": "Fraction",
                "color": "#577590",
            },
        ],
        save_path=plots_dir / "latent_epoch_dashboard.png",
        title="Realtime Latent Dashboard (Epoch)",
    )

    latest_step = train_rows[-1] if train_rows else {}
    latest_epoch = epoch_rows[-1] if epoch_rows else {}
    summary_payload = {
        "updated_at": _now_iso(),
        "run_dir": str(run_dir.resolve()),
        "plots": {
            "loss_curve_path": loss_curve_path,
            "lr_curve_path": lr_curve_path,
            "latent_step_dashboard_path": step_dashboard_path,
            "latent_epoch_dashboard_path": epoch_dashboard_path,
        },
        "latest_step": {
            key: latest_step.get(key)
            for key in [
                "global_step",
                "loss",
                "lr",
                "adapter_bottleneck_reconstruction_loss",
                "beta_vae_kl_loss",
                "beta_vae_beta",
                "kl_per_dim_mean",
                "z_norm_l2_mean",
                "posterior_mu_abs_mean",
            ]
        },
        "latest_epoch": {
            key: latest_epoch.get(key)
            for key in [
                "epoch",
                "train_loss",
                "adapter_bottleneck_reconstruction_loss",
                "beta_vae_kl_loss",
                "beta_vae_beta",
                "kl_per_dim_mean",
                "kl_per_dim_max",
                "free_bits_active_fraction",
                "z_norm_l2_mean",
                "z_std_mean",
                "posterior_mu_abs_mean",
                "posterior_logvar_mean",
                "latent_snapshot_dir",
                "latent_snapshot_manifest_path",
            ]
        },
    }
    dashboard_summary_path = plots_dir / "dashboard_summary.json"
    with dashboard_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)
    export_localized_json_artifact(dashboard_summary_path, summary_payload)

    return {
        "loss_curve_path": loss_curve_path,
        "lr_curve_path": lr_curve_path,
        "latent_step_dashboard_path": str(Path(step_dashboard_path).resolve()) if step_dashboard_path else "",
        "latent_epoch_dashboard_path": str(Path(epoch_dashboard_path).resolve()) if epoch_dashboard_path else "",
        "dashboard_summary_path": str(dashboard_summary_path.resolve()),
    }
