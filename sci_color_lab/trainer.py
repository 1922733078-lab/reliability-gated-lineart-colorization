from __future__ import annotations

import gc
import json
import math
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from PIL import Image, ImageDraw, ImageOps
from tqdm.auto import tqdm

from .ablation import AblationGroup
from .adapter import AdapterConfig, SXDLConditionAdapter
from .config import InferenceConfig, TrainerConfig
from .critic import PatchImageCritic, compute_flat_color_penalty, compute_gradient_penalty, resize_for_critic
from .data import LineartColorizationDataset, PairRecord, build_control_image, create_or_load_split, discover_pairs, select_validation_records
from .environment import ensure_environment_lock
from .inference_archive import DEFAULT_INFERENCE_ARCHIVE_ROOT
from .localized_outputs import export_localized_json_artifact, sync_evaluation_localized_outputs, sync_run_localized_outputs
from .memory import PeakMemoryMonitor
from .metrics import compute_saved_epoch_metrics, count_trainable_params_from_models, estimate_adapter_flops_g, should_compute_fid
from .pipeline import ModelManager, load_scheduler, maybe_enable_xformers, resolve_dtype
from .plotting import export_training_curves


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _max_optional(current: float | None, candidate: float | None) -> float | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return max(current, candidate)


def _sanitize_tensor(tensor: torch.Tensor, *, nan: float = 0.0, posinf: float = 10.0, neginf: float = -10.0) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _metric_higher_is_better(metric_name: str) -> bool:
    normalized = str(metric_name).strip().lower()
    return normalized in {
        "precision",
        "recall",
        "f_score",
        "pr_curve_auc",
        "ssim",
        "edge_consistency",
        "histogram_correlation",
    }


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _predict_original_latents(
    noise_scheduler: DDPMScheduler,
    noisy_latents: torch.Tensor,
    model_pred: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    beta_prod_t = 1.0 - alpha_prod_t
    prediction_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "sample":
        pred_original = model_pred
    elif prediction_type == "v_prediction":
        pred_original = alpha_prod_t.sqrt() * noisy_latents - beta_prod_t.sqrt() * model_pred
    else:
        pred_original = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt().clamp(min=1e-6)
    return _sanitize_tensor(pred_original, nan=0.0, posinf=4.0, neginf=-4.0)


def _decode_latents_to_images(vae, latents: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    decoder_input = latents.to(dtype) / vae.config.scaling_factor
    if torch.is_grad_enabled() and decoder_input.requires_grad:
        def _decode_fn(value: torch.Tensor) -> torch.Tensor:
            return vae.decode(value).sample

        try:
            decoded = torch.utils.checkpoint.checkpoint(_decode_fn, decoder_input, use_reentrant=False)
        except TypeError:
            decoded = torch.utils.checkpoint.checkpoint(_decode_fn, decoder_input)
    else:
        decoded = vae.decode(decoder_input).sample
    decoded = _sanitize_tensor(decoded.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    return decoded.clamp(-1.0, 1.0)


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_make_json_safe(payload), ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    return rows


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in rows:
            handle.write(json.dumps(_make_json_safe(payload), ensure_ascii=False) + "\n")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_make_json_safe(payload), handle, ensure_ascii=False, indent=2)
    export_localized_json_artifact(path, _make_json_safe(payload))


def create_run_dir(output_root: str, group_id: str, seed: int, run_name: str = "") -> Path:
    if run_name.strip():
        run_dir = Path(output_root) / run_name.strip()
    else:
        run_dir = Path(output_root) / group_id / f"seed_{seed}"
    for subdir in ("checkpoints", "logs", "previews", "evaluations", "plots", "latents"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def build_optimizer(name: str, params, lr: float, weight_decay: float):
    normalized = name.strip().lower().replace("-", "").replace("_", "")
    if normalized == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if normalized == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if normalized in {"adamw8bit", "bnbadamw8bit"}:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "optimizer_name=adamw8bit requires bitsandbytes. Install it with `python -m pip install bitsandbytes`."
            ) from exc
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)


def build_critic_optimizer(params, lr: float, beta1: float, beta2: float):
    return torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))


def _release_torch_memory(device: torch.device | str | None = None) -> None:
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


def _gpu_total_memory_gb(device: torch.device | str | None = None) -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        normalized = torch.device(device) if device is not None else torch.device("cuda")
    except Exception:
        normalized = torch.device("cuda")
    if normalized.type != "cuda":
        return None
    try:
        return round(torch.cuda.get_device_properties(normalized).total_memory / (1024.0**3), 4)
    except Exception:
        return None


def _gpu_memory_snapshot_gb(device: torch.device | str | None = None) -> dict[str, float | None]:
    snapshot: dict[str, float | None] = {
        "allocated_gb": None,
        "reserved_gb": None,
    }
    if not torch.cuda.is_available():
        return snapshot
    try:
        normalized = torch.device(device) if device is not None else torch.device("cuda")
    except Exception:
        normalized = torch.device("cuda")
    if normalized.type != "cuda":
        return snapshot
    try:
        snapshot["allocated_gb"] = round(torch.cuda.memory_allocated(normalized) / (1024.0**3), 4)
    except Exception:
        pass
    try:
        snapshot["reserved_gb"] = round(torch.cuda.memory_reserved(normalized) / (1024.0**3), 4)
    except Exception:
        pass
    return snapshot


def _is_cuda_oom_error(exc: BaseException) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _resolve_effective_mixed_precision(config: TrainerConfig) -> str:
    requested = str(config.mixed_precision).strip().lower()
    if requested != "fp16" or not config.enable_wgan_gp or not torch.cuda.is_available():
        return requested
    try:
        if torch.cuda.is_bf16_supported():
            return "bf16"
    except Exception:
        pass
    return requested


class ColorizationTrainer:
    def __init__(self, config: TrainerConfig, group: AblationGroup, seed: int) -> None:
        self.config = config
        self.group = group
        self.seed = int(seed)
        self.run_dir = create_run_dir(config.output_root, group.group_id, self.seed, config.run_name)
        self.log_path = self.run_dir / "logs" / "train.jsonl"
        self.metric_log_path = self.run_dir / "logs" / "metrics.jsonl"
        self.status_path = self.run_dir / "train_status.json"
        self.summary_path = self.run_dir / "run_summary.json"
        self.split_path = self.run_dir / "dataset_split.json"
        self.preview_dir = self.run_dir / "previews"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.environment_lock_path = Path(config.output_root) / "experiment_config.lock.yaml"
        self.validation_selection_path = self.run_dir / "validation_selection.json"
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.effective_mixed_precision = _resolve_effective_mixed_precision(config)
        accelerator_precision = "no" if self.effective_mixed_precision == "fp32" else self.effective_mixed_precision
        self.accelerator = Accelerator(
            mixed_precision=accelerator_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs],
            project_dir=str(self.run_dir),
        )
        self.run_created_at_iso = now_iso()
        set_seed(self.seed)
        self.best_train_loss = float("inf")
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.total_optimizer_steps = 0
        self.latest_preview_path = ""
        self.best_fid = float("inf")
        self.best_fid_epoch = 0
        self.best_fid_checkpoint_path = ""
        self.best_fid_preview_path = ""
        self.best_fid_eval_dir = ""
        self.best_fid_archive_dir = ""
        self.best_fid_learning_rate: float | None = None
        self.previous_quality_metric_value: float | None = None
        self.consecutive_quality_decline_epochs = 0
        self.best_fid_lr_recovery_count = 0
        self.best_fid_lr_recovery_active = False
        self.latest_quality_recovery_triggered = False
        self.adversarial_cooldown_until_epoch = 0
        self.collapse_guard_last_reason = ""
        self.learning_rate_cap: float | None = None
        self.previous_epoch_sanity_min_ratio: float | None = None
        self.epoch_sanity_recovery_until_epoch = 0
        self.epoch_sanity_recovery_min_ratio: float | None = None
        self.epoch_sanity_recovery_below_threshold_count = 0
        self.epoch_sanity_recovery_warning_has_rapid_drop = False
        self.best_val_loss_checkpoint_path = ""
        self.epoch_history: list[dict[str, Any]] = []
        self.validation_source = "split_val"
        self.preview_source = "split_val"
        self.validation_note = ""
        self.metric_eval_records: list[Any] = []
        self.preview_records: list[Any] = []
        self.params_m: float | None = None
        self.flops_g: float | None = None
        self.loss_curve_path = ""
        self.lr_curve_path = ""
        self.dashboard_summary_path = ""
        self.latent_step_dashboard_path = ""
        self.latent_epoch_dashboard_path = ""
        self.latest_latent_snapshot_dir = ""
        self.train_gpu_memory_peak_gb: float | None = None
        self.train_gpu_memory_reserved_peak_gb: float | None = None
        self.train_cpu_memory_peak_gb: float | None = None
        self.latest_eval_gpu_memory_peak_gb: float | None = None
        self.latest_eval_gpu_memory_reserved_peak_gb: float | None = None
        self.latest_eval_cpu_memory_peak_gb: float | None = None
        self.latest_eval_archive_dir = ""
        self.wgan_disabled_reason = ""
        self.wgan_disabled_epoch = 0
        self.recovery_image_aux_disabled_due_to_oom = False
        self.recovery_image_aux_disabled_due_to_skips = False
        self.recovery_aux_disabled_due_to_skips = False
        self.recovery_skipped_step_streak = 0
        self.latest_epoch_time_seconds: float | None = None
        self.latest_epoch_time_hms = ""
        self.seed_started_at_iso = ""
        self.seed_elapsed_seconds: float | None = None
        self.seed_elapsed_hms = ""
        self.training_xformers_requested = bool(config.enable_xformers)
        self.training_xformers_enabled = False
        self.training_xformers_status = "pending" if self.training_xformers_requested else "disabled_by_config"
        self.training_xformers_enabled_modules: list[str] = []
        self.training_xformers_failed_modules: dict[str, str] = {}
        self.training_xformers_skipped_modules: list[str] = []
        self.adapter_config = AdapterConfig(
            in_channels=3,
            hidden_channels=int(config.adapter_channels),
            num_blocks=int(config.adapter_blocks),
            fixed_threshold_value=float(config.fixed_threshold_value),
            enable_variational_bottleneck=bool(getattr(config, "enable_adapter_variational_bottleneck", True)),
            latent_channels=int(getattr(config, "adapter_variational_latent_channels", 12)),
            bottleneck_channels=int(getattr(config, "adapter_variational_bottleneck_channels", max(8, int(config.adapter_channels)))),
            decoder_channels=int(getattr(config, "adapter_variational_decoder_channels", max(4, int(config.adapter_channels) // 2))),
            bottleneck_dropout=float(getattr(config, "adapter_variational_dropout", 0.05)),
            logvar_min=float(getattr(config, "adapter_variational_logvar_min", -6.0)),
            logvar_max=float(getattr(config, "adapter_variational_logvar_max", 2.0)),
        )
        self.inference_defaults = InferenceConfig(
            prompt=config.prompt_template,
            negative_prompt=config.negative_prompt,
            num_inference_steps=35,
            guidance_scale=7.5,
            controlnet_scale=config.controlnet_conditioning_scale,
            seed=self.seed,
            width=config.image_width,
            height=config.image_height,
            scheduler="unipc",
            device=config.device,
            dtype=self.effective_mixed_precision,
            cpu_offload=config.cpu_offload,
            enable_xformers=config.enable_xformers,
        )

    def _training_validation_archive_seed_dir(self) -> Path:
        archive_root_text = str(self.config.inference_archive_root).strip()
        archive_root = Path(archive_root_text) if archive_root_text else DEFAULT_INFERENCE_ARCHIVE_ROOT
        return archive_root / "训练过程验证" / f"group_{self.group.group_id}" / f"seed_{self.seed}"

    def _training_xformers_payload(self) -> dict[str, Any]:
        return {
            "requested": self.training_xformers_requested,
            "enabled": self.training_xformers_enabled,
            "status": self.training_xformers_status,
            "enabled_modules": list(self.training_xformers_enabled_modules),
            "failed_modules": dict(self.training_xformers_failed_modules),
            "skipped_modules": list(self.training_xformers_skipped_modules),
        }

    def _update_training_xformers_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            return
        failed_modules = state.get("failed_modules", {})
        self.training_xformers_requested = bool(state.get("requested", self.training_xformers_requested))
        self.training_xformers_enabled = bool(state.get("enabled", False))
        self.training_xformers_status = str(
            state.get(
                "status",
                "pending" if self.training_xformers_requested else "disabled_by_config",
            )
        )
        self.training_xformers_enabled_modules = [str(item) for item in state.get("enabled_modules", [])]
        self.training_xformers_skipped_modules = [str(item) for item in state.get("skipped_modules", [])]
        if isinstance(failed_modules, dict):
            self.training_xformers_failed_modules = {
                str(key): str(value) for key, value in failed_modules.items()
            }
        else:
            self.training_xformers_failed_modules = {}

    def _record_training_component_runtime_state(self, components: dict[str, Any]) -> None:
        self._update_training_xformers_state(components.get("xformers_state"))
        self.write_metadata()
        append_jsonl(
            self.metric_log_path,
            {
                "timestamp": now_iso(),
                "event": "training_xformers",
                "group_id": self.group.group_id,
                "seed": self.seed,
                "training_xformers": self._training_xformers_payload(),
            },
        )

    def _build_training_memory_warning(self) -> dict[str, Any]:
        total_memory_gb = _gpu_total_memory_gb(self.accelerator.device)
        snapshot = _gpu_memory_snapshot_gb(self.accelerator.device)
        allocated_gb = snapshot.get("allocated_gb")
        reserved_gb = snapshot.get("reserved_gb")
        reasons: list[str] = []
        score = 0

        pixel_count = int(self.config.image_width) * int(self.config.image_height)
        if _is_finite_number(total_memory_gb):
            total_memory_gb = float(total_memory_gb)
            if total_memory_gb <= 16.0:
                score += 4
                reasons.append("16GB_or_less_gpu")
            elif total_memory_gb <= 24.5:
                score += 2
                reasons.append("24GB_class_gpu")
        else:
            reasons.append("unknown_gpu_memory")

        if pixel_count >= 1024 * 1024:
            score += 3
            reasons.append("very_high_resolution")
        elif pixel_count >= 768 * 1024:
            score += 2
            reasons.append("high_resolution")
        elif pixel_count >= 512 * 768:
            score += 1
            reasons.append("mid_high_resolution")

        if int(self.config.batch_size) >= 2:
            score += 2 + max(0, int(self.config.batch_size) - 2)
            reasons.append(f"batch_size_{int(self.config.batch_size)}")

        precision_name = str(self.effective_mixed_precision).strip().lower()
        if precision_name == "fp32":
            score += 4
            reasons.append("fp32_training")

        if not bool(self.config.enable_gradient_checkpointing):
            score += 2
            reasons.append("gradient_checkpointing_off")

        if not self.training_xformers_enabled:
            score += 2
            reasons.append("xformers_not_enabled")

        if bool(self.config.enable_wgan_gp):
            score += 2
            reasons.append(f"wgan_from_epoch_{max(1, int(self.config.wgan_start_epoch))}")

        reserved_ratio: float | None = None
        if _is_finite_number(reserved_gb) and _is_finite_number(total_memory_gb) and float(total_memory_gb) > 0.0:
            reserved_ratio = float(reserved_gb) / float(total_memory_gb)
            if reserved_ratio >= 0.70:
                score += 2
                reasons.append("high_startup_reserved_ratio")
            elif reserved_ratio >= 0.55:
                score += 1
                reasons.append("moderate_startup_reserved_ratio")

        if score >= 8:
            level = "high"
            message = "高显存风险，尤其是启用 WGAN 后的训练阶段。"
        elif score >= 5:
            level = "medium-high"
            message = "启动阶段大概率能跑，但显存会比较紧，后续 WGAN 阶段需要重点关注。"
        elif score >= 3:
            level = "medium"
            message = "当前配置通常可运行，但建议持续观察每轮峰值显存。"
        else:
            level = "low"
            message = "当前配置的启动显存压力较低。"

        if self.training_xformers_enabled and level in {"medium-high", "high"}:
            message += " 已启用 xformers，但它不能完全抵消高分辨率和 WGAN 带来的显存压力。"
        elif self.training_xformers_enabled and level == "medium":
            message += " xformers 已生效，这会降低一部分注意力显存开销。"

        return {
            "level": level,
            "message": message,
            "gpu_total_memory_gb": total_memory_gb,
            "startup_allocated_gb": allocated_gb,
            "startup_reserved_gb": reserved_gb,
            "startup_reserved_ratio": round(reserved_ratio, 4) if reserved_ratio is not None else None,
            "image_width": int(self.config.image_width),
            "image_height": int(self.config.image_height),
            "batch_size": int(self.config.batch_size),
            "gradient_accumulation_steps": int(self.config.gradient_accumulation_steps),
            "effective_precision": precision_name,
            "gradient_checkpointing": bool(self.config.enable_gradient_checkpointing),
            "wgan_enabled": bool(self.config.enable_wgan_gp),
            "wgan_start_epoch": max(1, int(self.config.wgan_start_epoch)),
            "reasons": reasons,
        }

    def _emit_training_startup_report(self, mode: str) -> None:
        if not self.accelerator.is_local_main_process:
            return
        xformers_payload = self._training_xformers_payload()
        memory_warning = self._build_training_memory_warning()
        enabled_modules = ", ".join(xformers_payload.get("enabled_modules", [])) or "-"
        skipped_modules = ", ".join(xformers_payload.get("skipped_modules", [])) or "-"
        failed_modules = xformers_payload.get("failed_modules", {})
        failed_text = "; ".join(f"{key}: {value}" for key, value in failed_modules.items()) if failed_modules else "-"
        gpu_total_text = (
            f"{float(memory_warning['gpu_total_memory_gb']):.2f}GB"
            if _is_finite_number(memory_warning.get("gpu_total_memory_gb"))
            else "unknown"
        )
        startup_allocated_text = (
            f"{float(memory_warning['startup_allocated_gb']):.2f}GB"
            if _is_finite_number(memory_warning.get("startup_allocated_gb"))
            else "unknown"
        )
        startup_reserved_text = (
            f"{float(memory_warning['startup_reserved_gb']):.2f}GB"
            if _is_finite_number(memory_warning.get("startup_reserved_gb"))
            else "unknown"
        )
        print(
            "[启动检查] "
            f"mode={mode} | xformers={xformers_payload.get('status')} | "
            f"enabled={enabled_modules} | skipped={skipped_modules} | failed={failed_text}"
        )
        print(
            "[启动检查] "
            f"显存预警={memory_warning['level']} | gpu_total={gpu_total_text} | "
            f"startup_allocated={startup_allocated_text} | startup_reserved={startup_reserved_text} | "
            f"config={int(self.config.image_width)}x{int(self.config.image_height)} "
            f"batch={int(self.config.batch_size)} accum={int(self.config.gradient_accumulation_steps)} "
            f"precision={memory_warning['effective_precision']} grad_ckpt={'on' if self.config.enable_gradient_checkpointing else 'off'} "
            f"wgan={'on' if self.config.enable_wgan_gp else 'off'}(epoch>={max(1, int(self.config.wgan_start_epoch))})"
        )
        print(f"[启动检查] {memory_warning['message']}")
        append_jsonl(
            self.metric_log_path,
            {
                "timestamp": now_iso(),
                "event": "training_startup_check",
                "group_id": self.group.group_id,
                "seed": self.seed,
                "mode": mode,
                "training_xformers": xformers_payload,
                "memory_warning": memory_warning,
            },
        )

    def reset_managed_run_outputs(self) -> None:
        managed_dirs = [
            self.checkpoint_dir,
            self.run_dir / "logs",
            self.preview_dir,
            self.run_dir / "evaluations",
            self.run_dir / "plots",
            self.run_dir / "latents",
            self.run_dir / "lora",
        ]
        managed_files = [
            self.run_dir / "adapter.pt",
            self.run_dir / "dataset_split.json",
            self.run_dir / "run_metadata.json",
            self.run_dir / "run_summary.json",
            self.run_dir / "train_status.json",
            self.run_dir / "validation_selection.json",
        ]

        for path in managed_dirs:
            _remove_path(path)
        for path in managed_files:
            _remove_path(path)

        _remove_path(self._training_validation_archive_seed_dir())

        for subdir in ("checkpoints", "logs", "previews", "evaluations", "plots", "latents"):
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _ensure_managed_run_output_dirs(self) -> None:
        for subdir in ("checkpoints", "logs", "previews", "evaluations", "plots", "latents"):
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _latest_completed_epoch(self) -> int:
        latest_epoch = 0
        summary = load_json(self.summary_path)
        try:
            latest_epoch = max(latest_epoch, int(summary.get("epoch", 0) or 0))
        except Exception:
            latest_epoch = max(latest_epoch, 0)
        for row in load_jsonl(self.metric_log_path):
            if str(row.get("event", "")) != "epoch_end":
                continue
            try:
                latest_epoch = max(latest_epoch, int(row.get("epoch", 0) or 0))
            except Exception:
                continue
        return latest_epoch

    def _discover_resume_checkpoint(self) -> dict[str, Any] | None:
        completed_epoch = self._latest_completed_epoch()
        if completed_epoch <= 0:
            return None
        checkpoint_dir = self._latest_checkpoint_dir_for_epoch(completed_epoch)
        checkpoint_epoch = int(completed_epoch)
        if checkpoint_dir is None or not checkpoint_dir.exists():
            fallback = self._latest_checkpoint_dir_up_to_epoch(completed_epoch)
            if fallback is None:
                return None
            checkpoint_epoch, checkpoint_dir = fallback
        metadata = load_json(checkpoint_dir / "checkpoint.json")
        try:
            checkpoint_step = int(metadata.get("step", 0) or 0)
        except Exception:
            checkpoint_step = 0
        if checkpoint_step <= 0:
            return None
        return {
            "epoch": int(checkpoint_epoch),
            "step": int(checkpoint_step),
            "checkpoint_dir": checkpoint_dir.resolve(),
            "lora_dir": (checkpoint_dir / "lora").resolve(),
        }

    def _truncate_incomplete_resume_logs(self, completed_epoch: int) -> None:
        train_rows = load_jsonl(self.log_path)
        if train_rows:
            filtered_train_rows: list[dict[str, Any]] = []
            for row in train_rows:
                try:
                    row_epoch = int(row.get("epoch", 0) or 0)
                except Exception:
                    row_epoch = 0
                if row_epoch <= int(completed_epoch):
                    filtered_train_rows.append(row)
            if len(filtered_train_rows) != len(train_rows):
                save_jsonl(self.log_path, filtered_train_rows)

        metric_rows = load_jsonl(self.metric_log_path)
        if metric_rows:
            filtered_metric_rows: list[dict[str, Any]] = []
            for row in metric_rows:
                epoch_value = row.get("epoch", None)
                if epoch_value is None:
                    filtered_metric_rows.append(row)
                    continue
                try:
                    row_epoch = int(epoch_value)
                except Exception:
                    filtered_metric_rows.append(row)
                    continue
                if row_epoch <= int(completed_epoch):
                    filtered_metric_rows.append(row)
            if len(filtered_metric_rows) != len(metric_rows):
                save_jsonl(self.metric_log_path, filtered_metric_rows)

    def _restore_resume_runtime_state(self, resume_info: dict[str, Any]) -> None:
        completed_epoch = int(resume_info.get("epoch", 0) or 0)
        self._ensure_managed_run_output_dirs()
        self._truncate_incomplete_resume_logs(completed_epoch)

        epoch_history_payload = load_json(self.run_dir / "logs" / "epoch_history.json")
        epoch_rows = epoch_history_payload.get("epochs", [])
        if isinstance(epoch_rows, list):
            restored_epoch_rows: list[dict[str, Any]] = []
            for row in epoch_rows:
                if not isinstance(row, dict):
                    continue
                try:
                    row_epoch = int(row.get("epoch", 0) or 0)
                except Exception:
                    row_epoch = 0
                if row_epoch <= completed_epoch:
                    restored_epoch_rows.append(row)
            self.epoch_history = restored_epoch_rows
            save_json(self.run_dir / "logs" / "epoch_history.json", {"epochs": self.epoch_history})

        summary = load_json(self.summary_path)
        if _is_finite_number(summary.get("best_train_loss")):
            self.best_train_loss = float(summary["best_train_loss"])
        if _is_finite_number(summary.get("best_val_loss")):
            self.best_val_loss = float(summary["best_val_loss"])
        if _is_finite_number(summary.get("best_fid")):
            self.best_fid = float(summary["best_fid"])
        try:
            self.best_fid_epoch = int(summary.get("best_fid_epoch", self.best_fid_epoch) or self.best_fid_epoch)
        except Exception:
            pass
        self.best_fid_checkpoint_path = str(summary.get("best_fid_checkpoint_path", self.best_fid_checkpoint_path) or self.best_fid_checkpoint_path)
        self.best_fid_preview_path = str(summary.get("best_fid_preview_path", self.best_fid_preview_path) or self.best_fid_preview_path)
        self.best_fid_eval_dir = str(summary.get("best_fid_eval_dir", self.best_fid_eval_dir) or self.best_fid_eval_dir)
        self.best_fid_archive_dir = str(summary.get("best_fid_archive_dir", self.best_fid_archive_dir) or self.best_fid_archive_dir)
        if _is_finite_number(summary.get("best_fid_learning_rate")):
            self.best_fid_learning_rate = float(summary["best_fid_learning_rate"])
        try:
            self.consecutive_quality_decline_epochs = int(
                summary.get("consecutive_quality_decline_epochs", self.consecutive_quality_decline_epochs)
                or self.consecutive_quality_decline_epochs
            )
        except Exception:
            pass
        try:
            self.best_fid_lr_recovery_count = int(
                summary.get("best_fid_lr_recovery_count", self.best_fid_lr_recovery_count)
                or self.best_fid_lr_recovery_count
            )
        except Exception:
            pass
        self.best_fid_lr_recovery_active = bool(summary.get("best_fid_lr_recovery_active", self.best_fid_lr_recovery_active))
        try:
            self.adversarial_cooldown_until_epoch = int(
                summary.get("adversarial_cooldown_until_epoch", self.adversarial_cooldown_until_epoch)
                or self.adversarial_cooldown_until_epoch
            )
        except Exception:
            pass
        self.collapse_guard_last_reason = str(summary.get("collapse_guard_reason", self.collapse_guard_last_reason) or self.collapse_guard_last_reason)
        self.wgan_disabled_reason = str(summary.get("wgan_disabled_reason", self.wgan_disabled_reason) or self.wgan_disabled_reason)
        try:
            self.wgan_disabled_epoch = int(summary.get("wgan_disabled_epoch", self.wgan_disabled_epoch) or self.wgan_disabled_epoch)
        except Exception:
            pass
        if _is_finite_number(summary.get("train_gpu_memory_peak_gb")):
            self.train_gpu_memory_peak_gb = float(summary["train_gpu_memory_peak_gb"])
        if _is_finite_number(summary.get("train_gpu_memory_reserved_peak_gb")):
            self.train_gpu_memory_reserved_peak_gb = float(summary["train_gpu_memory_reserved_peak_gb"])
        if _is_finite_number(summary.get("train_cpu_memory_peak_gb")):
            self.train_cpu_memory_peak_gb = float(summary["train_cpu_memory_peak_gb"])
        if _is_finite_number(summary.get("latest_eval_gpu_memory_peak_gb")):
            self.latest_eval_gpu_memory_peak_gb = float(summary["latest_eval_gpu_memory_peak_gb"])
        if _is_finite_number(summary.get("latest_eval_gpu_memory_reserved_peak_gb")):
            self.latest_eval_gpu_memory_reserved_peak_gb = float(summary["latest_eval_gpu_memory_reserved_peak_gb"])
        if _is_finite_number(summary.get("latest_eval_cpu_memory_peak_gb")):
            self.latest_eval_cpu_memory_peak_gb = float(summary["latest_eval_cpu_memory_peak_gb"])
        if _is_finite_number(summary.get("latest_epoch_time_seconds")):
            self.latest_epoch_time_seconds = float(summary["latest_epoch_time_seconds"])
        self.latest_epoch_time_hms = str(summary.get("latest_epoch_time_hms", self.latest_epoch_time_hms) or self.latest_epoch_time_hms)
        self.loss_curve_path = str(summary.get("loss_curve_path", self.loss_curve_path) or self.loss_curve_path)
        self.lr_curve_path = str(summary.get("lr_curve_path", self.lr_curve_path) or self.lr_curve_path)
        self.dashboard_summary_path = str(summary.get("dashboard_summary_path", self.dashboard_summary_path) or self.dashboard_summary_path)
        self.latent_step_dashboard_path = str(summary.get("latent_step_dashboard_path", self.latent_step_dashboard_path) or self.latent_step_dashboard_path)
        self.latent_epoch_dashboard_path = str(summary.get("latent_epoch_dashboard_path", self.latent_epoch_dashboard_path) or self.latent_epoch_dashboard_path)
        self.latest_latent_snapshot_dir = str(summary.get("latest_latent_snapshot_dir", self.latest_latent_snapshot_dir) or self.latest_latent_snapshot_dir)
        self.seed_started_at_iso = str(summary.get("seed_started_at", self.seed_started_at_iso) or self.seed_started_at_iso)
        if _is_finite_number(summary.get("seed_elapsed_seconds")):
            self.seed_elapsed_seconds = float(summary["seed_elapsed_seconds"])
            self.seed_elapsed_hms = _format_duration(self.seed_elapsed_seconds)
        self.global_step = int(resume_info.get("step", 0) or 0)

        metric_rows = load_jsonl(self.metric_log_path)
        last_warning_row: dict[str, Any] | None = None
        last_epoch_end_row: dict[str, Any] | None = None
        for row in metric_rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("event", "")) == "epoch_sanity_warning":
                last_warning_row = row
            if str(row.get("event", "")) == "epoch_end":
                last_epoch_end_row = row

        if last_warning_row is not None:
            if _is_finite_number(last_warning_row.get("learning_rate_cap")):
                restored_cap = float(last_warning_row["learning_rate_cap"])
                # Avoid carrying a near-zero cap from a finished cosine schedule into a new refinement resume.
                resume_cap_floor = max(0.0, float(getattr(self.config, "epoch_sanity_recovery_min_lr", 0.0)))
                configured_base_lr = max(0.0, float(getattr(self.config, "learning_rate", 0.0)))
                if resume_cap_floor > 0.0:
                    restored_cap = max(restored_cap, resume_cap_floor)
                if configured_base_lr > 0.0:
                    restored_cap = min(restored_cap, configured_base_lr)
                self.learning_rate_cap = restored_cap
            try:
                self.epoch_sanity_recovery_until_epoch = int(
                    last_warning_row.get("recovery_until_epoch", self.epoch_sanity_recovery_until_epoch)
                    or self.epoch_sanity_recovery_until_epoch
                )
            except Exception:
                pass
            if _is_finite_number(last_warning_row.get("min_ratio")):
                self.epoch_sanity_recovery_min_ratio = float(last_warning_row["min_ratio"])
            try:
                self.epoch_sanity_recovery_below_threshold_count = int(
                    last_warning_row.get(
                        "below_threshold_count",
                        self.epoch_sanity_recovery_below_threshold_count,
                    )
                    or self.epoch_sanity_recovery_below_threshold_count
                )
            except Exception:
                pass
            self.epoch_sanity_recovery_warning_has_rapid_drop = bool(
                last_warning_row.get("warning_has_rapid_drop", self.epoch_sanity_recovery_warning_has_rapid_drop)
            )

        if last_epoch_end_row is not None:
            sanity_payload = last_epoch_end_row.get("epoch_sanity_check", {})
            if isinstance(sanity_payload, dict):
                ratios: list[float] = []
                ratio_fields = [
                    ("generated_std", "std_threshold"),
                    ("generated_dynamic_range", "dynamic_range_threshold"),
                    ("generated_gradient_mean", "gradient_threshold"),
                ]
                for value_key, threshold_key in ratio_fields:
                    value = sanity_payload.get(value_key)
                    threshold = sanity_payload.get(threshold_key)
                    if not _is_finite_number(value) or not _is_finite_number(threshold):
                        continue
                    if float(threshold) <= 0.0:
                        continue
                    ratios.append(float(value) / float(threshold))
                if ratios:
                    self.previous_epoch_sanity_min_ratio = min(ratios)

    def _load_training_modules_from_checkpoint(
        self,
        checkpoint_dir: Path,
        adapter: SXDLConditionAdapter,
        critic: PatchImageCritic | None = None,
    ) -> None:
        adapter_path = checkpoint_dir / "adapter.pt"
        if adapter_path.exists():
            adapter_state = torch.load(adapter_path, map_location="cpu")
            if isinstance(adapter_state, dict):
                adapter.load_state_dict(adapter_state, strict=True)
        critic_path = checkpoint_dir / "critic.pt"
        if critic is not None and critic_path.exists():
            critic_state = torch.load(critic_path, map_location="cpu")
            if isinstance(critic_state, dict):
                critic.load_state_dict(critic_state, strict=True)

    def _current_learning_rate(self, optimizer=None, lr_scheduler=None) -> float:
        optimizer_candidates = [
            optimizer,
            getattr(optimizer, "optimizer", None) if optimizer is not None else None,
            getattr(lr_scheduler, "optimizer", None) if lr_scheduler is not None else None,
        ]
        for candidate in optimizer_candidates:
            if candidate is None:
                continue
            param_groups = getattr(candidate, "param_groups", None)
            if not param_groups:
                continue
            try:
                return float(param_groups[0]["lr"])
            except Exception:
                continue

        if lr_scheduler is not None and hasattr(lr_scheduler, "get_last_lr"):
            try:
                last_lrs = [float(value) for value in lr_scheduler.get_last_lr() if value is not None]
                if last_lrs:
                    return last_lrs[0]
            except Exception:
                pass
        return 0.0

    def _evaluation_metric_device(self) -> str:
        if self.accelerator.device.type == "cuda":
            return "cpu"
        return str(self.accelerator.device)

    def _apply_best_fid_lr_if_needed(self, optimizer) -> None:
        if not self.best_fid_lr_recovery_active or self.best_fid_learning_rate is None:
            self._enforce_learning_rate_cap(optimizer)
            return
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(self.best_fid_learning_rate)
        self._enforce_learning_rate_cap(optimizer)

    def _enforce_learning_rate_cap(self, optimizer) -> None:
        if optimizer is None or self.learning_rate_cap is None:
            return
        try:
            cap = float(self.learning_rate_cap)
        except Exception:
            return
        if not math.isfinite(cap) or cap <= 0.0:
            return
        param_groups = getattr(optimizer, "param_groups", None) or []
        for param_group in param_groups:
            try:
                current_lr = float(param_group.get("lr", cap))
            except Exception:
                current_lr = cap
            param_group["lr"] = min(current_lr, cap)

    def _epoch_sanity_health_ratios(self, sanity_check_payload: dict[str, Any]) -> dict[str, float]:
        ratios: dict[str, float] = {}
        for name, generated_key, threshold_key in [
            ("std_ratio", "generated_std", "std_threshold"),
            ("dynamic_range_ratio", "generated_dynamic_range", "dynamic_range_threshold"),
            ("gradient_ratio", "generated_gradient_mean", "gradient_threshold"),
        ]:
            generated = sanity_check_payload.get(generated_key)
            threshold = sanity_check_payload.get(threshold_key)
            if not (_is_finite_number(generated) and _is_finite_number(threshold)):
                continue
            threshold_value = float(threshold)
            if threshold_value <= 0.0:
                continue
            ratios[name] = float(generated) / threshold_value
        return ratios

    def _detect_epoch_sanity_warning(self, sanity_check_payload: dict[str, Any]) -> tuple[bool, str]:
        ratios = self._epoch_sanity_health_ratios(sanity_check_payload)
        min_ratio = min(ratios.values()) if ratios else None
        previous_min_ratio = self.previous_epoch_sanity_min_ratio
        if min_ratio is not None:
            self.previous_epoch_sanity_min_ratio = float(min_ratio)

        if (
            not bool(getattr(self.config, "epoch_sanity_warning_enabled", True))
            or not sanity_check_payload
            or bool(sanity_check_payload.get("collapsed"))
            or min_ratio is None
        ):
            return False, ""

        warning_min_ratio = max(1.0, float(getattr(self.config, "epoch_sanity_warning_min_ratio", 1.08)))
        drop_ratio = float(getattr(self.config, "epoch_sanity_warning_drop_ratio", 0.15))
        drop_ratio = max(0.0, min(drop_ratio, 0.95))

        below_threshold_reasons: list[str] = []
        for label, generated_key, threshold_key in [
            ("std", "generated_std", "std_threshold"),
            ("dynamic_range", "generated_dynamic_range", "dynamic_range_threshold"),
            ("gradient", "generated_gradient_mean", "gradient_threshold"),
        ]:
            generated = sanity_check_payload.get(generated_key)
            threshold = sanity_check_payload.get(threshold_key)
            if not (_is_finite_number(generated) and _is_finite_number(threshold)):
                continue
            if float(generated) < float(threshold):
                below_threshold_reasons.append(f"{label}={float(generated):.2f}<{float(threshold):.2f}")

        near_threshold = float(min_ratio) < warning_min_ratio
        rapid_drop = (
            previous_min_ratio is not None
            and _is_finite_number(previous_min_ratio)
            and float(min_ratio) < float(previous_min_ratio) * (1.0 - drop_ratio)
        )
        if not (below_threshold_reasons or near_threshold or rapid_drop):
            return False, ""

        reasons: list[str] = []
        if below_threshold_reasons:
            reasons.append("below_threshold:" + ",".join(below_threshold_reasons))
        if near_threshold:
            reasons.append(f"min_ratio={float(min_ratio):.3f}<{warning_min_ratio:.3f}")
        if rapid_drop and previous_min_ratio is not None:
            reasons.append(f"ratio_drop={float(previous_min_ratio):.3f}->{float(min_ratio):.3f}")
        return True, "epoch_sanity_warning: " + "; ".join(reasons)

    def _apply_epoch_sanity_recovery(
        self,
        *,
        epoch: int,
        reason: str,
        sanity_check_payload: dict[str, Any],
        optimizer,
    ) -> None:
        current_lr = self._current_learning_rate(optimizer)
        if not _is_finite_number(current_lr) or float(current_lr) <= 0.0:
            return

        below_threshold_count = 0
        for generated_key, threshold_key in [
            ("generated_std", "std_threshold"),
            ("generated_dynamic_range", "dynamic_range_threshold"),
            ("generated_gradient_mean", "gradient_threshold"),
        ]:
            generated = sanity_check_payload.get(generated_key)
            threshold = sanity_check_payload.get(threshold_key)
            if not (_is_finite_number(generated) and _is_finite_number(threshold)):
                continue
            if float(generated) < float(threshold):
                below_threshold_count += 1

        min_ratio = None
        ratios = self._epoch_sanity_health_ratios(sanity_check_payload)
        if ratios:
            min_ratio = min(ratios.values())

        warning_has_below_threshold = "below_threshold:" in reason
        warning_has_rapid_drop = "ratio_drop=" in reason
        lr_scale = float(getattr(self.config, "epoch_sanity_recovery_lr_scale", 0.55))
        severe_lr_scale = float(getattr(self.config, "epoch_sanity_recovery_severe_lr_scale", 0.45))
        if (
            below_threshold_count >= 2
            or warning_has_below_threshold
            or warning_has_rapid_drop
            or (_is_finite_number(min_ratio) and float(min_ratio) < 0.9)
        ):
            lr_scale = severe_lr_scale
        lr_scale = max(0.05, min(lr_scale, 1.0))
        min_lr = float(getattr(self.config, "epoch_sanity_recovery_min_lr", 8e-6))
        target_cap = float(current_lr) * lr_scale
        if _is_finite_number(min_lr) and float(min_lr) > 0.0:
            target_cap = max(target_cap, float(min_lr))
        target_cap = min(float(current_lr), target_cap)
        if not math.isfinite(target_cap) or target_cap <= 0.0:
            return

        recovery_epochs = max(1, int(getattr(self.config, "epoch_sanity_recovery_epochs", 1)))
        if warning_has_below_threshold or warning_has_rapid_drop:
            recovery_epochs = max(recovery_epochs, int(getattr(self.config, "epoch_sanity_recovery_epochs", 1)) + 1)
        self.epoch_sanity_recovery_until_epoch = max(
            int(self.epoch_sanity_recovery_until_epoch),
            int(epoch) + recovery_epochs,
        )
        self.epoch_sanity_recovery_min_ratio = float(min_ratio) if _is_finite_number(min_ratio) else None
        self.epoch_sanity_recovery_below_threshold_count = int(below_threshold_count)
        self.epoch_sanity_recovery_warning_has_rapid_drop = bool(warning_has_rapid_drop)
        if self.learning_rate_cap is None:
            self.learning_rate_cap = float(target_cap)
        else:
            self.learning_rate_cap = min(float(self.learning_rate_cap), float(target_cap))
        self._enforce_learning_rate_cap(optimizer)
        recovery_profile = self._epoch_sanity_recovery_profile()
        append_jsonl(
            self.metric_log_path,
            {
                "timestamp": now_iso(),
                "event": "epoch_sanity_warning",
                "group_id": self.group.group_id,
                "seed": self.seed,
                "epoch": int(epoch),
                "reason": reason,
                "learning_rate_before": float(current_lr),
                "learning_rate_scale": float(lr_scale),
                "learning_rate_cap": float(self.learning_rate_cap),
                "recovery_until_epoch": int(self.epoch_sanity_recovery_until_epoch),
                "below_threshold_count": int(below_threshold_count),
                "min_ratio": float(min_ratio) if _is_finite_number(min_ratio) else None,
                "warning_has_below_threshold": bool(warning_has_below_threshold),
                "warning_has_rapid_drop": bool(warning_has_rapid_drop),
                "recovery_weight_multiplier": float(recovery_profile["weight_multiplier"]),
                "recovery_severity_signals": int(recovery_profile["severity_signals"]),
                "image_recovery_guidance_active": bool(recovery_profile["image_guidance_active"]),
            },
        )

    def _epoch_sanity_recovery_active(self, epoch: int) -> bool:
        return int(epoch) <= int(self.epoch_sanity_recovery_until_epoch)

    def _wgan_is_active(self, epoch: int) -> bool:
        if not self.config.enable_wgan_gp:
            return False
        if self.wgan_disabled_reason:
            return False
        return int(epoch) >= max(1, int(self.config.wgan_start_epoch))

    def _maybe_disable_wgan_for_memory(self, epoch: int) -> tuple[bool, str]:
        if not self.config.enable_wgan_gp:
            return False, ""
        if not bool(getattr(self.config, "enable_wgan_memory_guard", True)):
            return False, ""
        if self.wgan_disabled_reason:
            return False, self.wgan_disabled_reason
        if int(epoch) < max(1, int(self.config.wgan_start_epoch)):
            return False, ""

        total_memory_gb = _gpu_total_memory_gb(self.accelerator.device)
        if not _is_finite_number(total_memory_gb) or float(total_memory_gb) <= 0.0:
            return False, ""

        observed_peak_gb = max(
            float(self.train_gpu_memory_peak_gb or 0.0),
            float(self.train_gpu_memory_reserved_peak_gb or 0.0),
            float(self.latest_eval_gpu_memory_peak_gb or 0.0),
            float(self.latest_eval_gpu_memory_reserved_peak_gb or 0.0),
        )
        if observed_peak_gb <= 0.0:
            return False, ""

        guard_ratio = max(0.0, float(getattr(self.config, "wgan_memory_guard_ratio", 0.9)))
        observed_ratio = observed_peak_gb / float(total_memory_gb)
        if observed_ratio < guard_ratio:
            return False, ""

        self.wgan_disabled_epoch = int(epoch)
        self.wgan_disabled_reason = (
            f"memory_guard: observed_peak={observed_peak_gb:.2f}GB, "
            f"total_gpu={float(total_memory_gb):.2f}GB, "
            f"ratio={observed_ratio:.3f} >= threshold={guard_ratio:.3f}"
        )
        return True, self.wgan_disabled_reason

    def _adversarial_warmup_factor(self, epoch: int) -> float:
        if not self._wgan_is_active(epoch):
            return 0.0
        start_epoch = max(1, int(self.config.wgan_start_epoch))
        warmup_epochs = max(1, int(self.config.wgan_warmup_epochs))
        progress = int(epoch) - start_epoch + 1
        return max(0.0, min(1.0, float(progress) / float(warmup_epochs)))

    def _collapse_guard_active(self, epoch: int) -> bool:
        if not self.config.collapse_guard_enabled:
            return False
        return int(epoch) <= int(self.adversarial_cooldown_until_epoch)

    def _decoded_image_spatial_std(self, images: torch.Tensor) -> float:
        spatial_std = images.detach().float().flatten(start_dim=2).std(dim=2, unbiased=False).mean()
        return float(spatial_std.item())

    def _latent_auxiliary_active(self) -> bool:
        return (
            float(getattr(self.config, "latent_reconstruction_weight", 0.0)) > 0.0
            or float(getattr(self.config, "latent_distribution_weight", 0.0)) > 0.0
            or float(getattr(self.config, "latent_local_variance_weight", 0.0)) > 0.0
        )

    def _compute_latent_consistency_losses(
        self,
        predicted_original_latents: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        predicted = _sanitize_tensor(predicted_original_latents.float(), nan=0.0, posinf=4.0, neginf=-4.0)
        target = _sanitize_tensor(target_latents.float(), nan=0.0, posinf=4.0, neginf=-4.0)
        reconstruction_loss = F.l1_loss(predicted, target)

        predicted_flat = predicted.flatten(start_dim=2)
        target_flat = target.flatten(start_dim=2)
        predicted_mean = predicted_flat.mean(dim=2)
        target_mean = target_flat.mean(dim=2)
        predicted_std = predicted_flat.std(dim=2, unbiased=False)
        target_std = target_flat.std(dim=2, unbiased=False)
        distribution_loss = F.l1_loss(predicted_mean, target_mean) + F.l1_loss(predicted_std, target_std)

        latent_spatial_std = float(predicted_std.mean().item())
        return reconstruction_loss, distribution_loss, latent_spatial_std

    def _compute_latent_local_variance_loss(
        self,
        predicted_original_latents: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> torch.Tensor:
        kernel_size = max(3, int(getattr(self.config, "latent_local_variance_kernel_size", 3)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        target_ratio = max(0.0, float(getattr(self.config, "latent_local_variance_target_ratio", 0.75)))

        predicted = _sanitize_tensor(predicted_original_latents.float(), nan=0.0, posinf=4.0, neginf=-4.0)
        target = _sanitize_tensor(target_latents.float(), nan=0.0, posinf=4.0, neginf=-4.0)

        predicted_mean = F.avg_pool2d(predicted, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        predicted_mean_square = F.avg_pool2d(predicted * predicted, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        predicted_std = (predicted_mean_square - predicted_mean.square()).clamp(min=0.0).sqrt()

        target_mean = F.avg_pool2d(target, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        target_mean_square = F.avg_pool2d(target * target, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        target_std = (target_mean_square - target_mean.square()).clamp(min=0.0).sqrt()

        target_floor = target_std * target_ratio
        return F.relu(target_floor - predicted_std).mean()

    def _compute_latent_gradient_consistency_loss(
        self,
        predicted_original_latents: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> torch.Tensor:
        predicted = _sanitize_tensor(predicted_original_latents.float(), nan=0.0, posinf=4.0, neginf=-4.0)
        target = _sanitize_tensor(target_latents.float(), nan=0.0, posinf=4.0, neginf=-4.0)
        predicted_dx = torch.abs(predicted[:, :, :, 1:] - predicted[:, :, :, :-1])
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        predicted_dy = torch.abs(predicted[:, :, 1:, :] - predicted[:, :, :-1, :])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        return F.l1_loss(predicted_dx, target_dx) + F.l1_loss(predicted_dy, target_dy)

    def _compute_latent_global_flatness_loss(
        self,
        predicted_original_latents: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> torch.Tensor:
        predicted = _sanitize_tensor(predicted_original_latents.float(), nan=0.0, posinf=4.0, neginf=-4.0)
        target = _sanitize_tensor(target_latents.float(), nan=0.0, posinf=4.0, neginf=-4.0)
        predicted_flat = predicted.flatten(start_dim=2)
        target_flat = target.flatten(start_dim=2)
        predicted_std = predicted_flat.std(dim=2, unbiased=False)
        target_std = target_flat.std(dim=2, unbiased=False)
        predicted_range = predicted_flat.amax(dim=2) - predicted_flat.amin(dim=2)
        target_range = target_flat.amax(dim=2) - target_flat.amin(dim=2)
        std_ratio = max(0.0, float(getattr(self.config, "latent_global_flatness_target_std_ratio", 0.85)))
        range_ratio = max(0.0, float(getattr(self.config, "latent_global_flatness_target_range_ratio", 0.75)))
        std_floor = target_std * std_ratio
        range_floor = target_range * range_ratio
        std_loss = F.relu(std_floor - predicted_std).mean()
        range_loss = F.relu(range_floor - predicted_range).mean()
        return std_loss + range_loss

    def _compute_image_gradient_consistency_loss(
        self,
        predicted_images: torch.Tensor,
        target_images: torch.Tensor,
    ) -> torch.Tensor:
        predicted = _sanitize_tensor(predicted_images.float(), nan=0.0, posinf=1.0, neginf=-1.0)
        target = _sanitize_tensor(target_images.float(), nan=0.0, posinf=1.0, neginf=-1.0)
        predicted_gray = predicted.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        predicted_dx = torch.abs(predicted_gray[:, :, :, 1:] - predicted_gray[:, :, :, :-1])
        target_dx = torch.abs(target_gray[:, :, :, 1:] - target_gray[:, :, :, :-1])
        predicted_dy = torch.abs(predicted_gray[:, :, 1:, :] - predicted_gray[:, :, :-1, :])
        target_dy = torch.abs(target_gray[:, :, 1:, :] - target_gray[:, :, :-1, :])
        return F.l1_loss(predicted_dx, target_dx) + F.l1_loss(predicted_dy, target_dy)

    def _adapter_variational_enabled(self) -> bool:
        return bool(getattr(self.adapter_config, "enable_variational_bottleneck", False))

    def _current_adapter_vae_beta(self, optimizer_step_hint: int | None = None) -> float:
        if not self._adapter_variational_enabled():
            return 0.0
        beta_start = max(0.0, float(getattr(self.config, "adapter_variational_beta_start", 0.0)))
        beta_end = max(beta_start, float(getattr(self.config, "adapter_variational_beta_end", beta_start)))
        anneal_epochs = max(1, int(getattr(self.config, "adapter_variational_anneal_epochs", 1)))
        total_epochs = max(1, int(getattr(self.config, "epochs", 1)))
        steps_per_epoch = max(1, int(math.ceil(float(self.total_optimizer_steps) / float(total_epochs))))
        anneal_steps = max(1, steps_per_epoch * anneal_epochs)
        current_step = max(0, int(optimizer_step_hint if optimizer_step_hint is not None else self.global_step))
        progress = min(1.0, float(current_step) / float(anneal_steps))
        return float(beta_start + (beta_end - beta_start) * progress)

    def _compute_adapter_variational_metrics(
        self,
        adapter_stats: dict[str, Any] | None,
        *,
        optimizer_step_hint: int | None = None,
    ) -> dict[str, Any]:
        if not self._adapter_variational_enabled() or not isinstance(adapter_stats, dict):
            return {}
        mu = adapter_stats.get("posterior_mu")
        logvar = adapter_stats.get("posterior_logvar")
        z = adapter_stats.get("latent_sample")
        reconstruction_loss = adapter_stats.get("bottleneck_reconstruction_loss")
        if not (torch.is_tensor(mu) and torch.is_tensor(logvar) and torch.is_tensor(z) and torch.is_tensor(reconstruction_loss)):
            return {}

        mu = _sanitize_tensor(mu.float(), nan=0.0, posinf=4.0, neginf=-4.0)
        logvar = _sanitize_tensor(logvar.float(), nan=0.0, posinf=4.0, neginf=-6.0)
        z = _sanitize_tensor(z.float(), nan=0.0, posinf=4.0, neginf=-4.0)
        kl_map = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        reduce_dims = (0,) + tuple(range(2, kl_map.ndim))
        kl_per_dim = kl_map.mean(dim=reduce_dims)
        free_bits = max(0.0, float(getattr(self.config, "adapter_variational_free_bits", 0.0)))
        kl_loss = torch.clamp(kl_per_dim, min=free_bits).mean()
        raw_kl_loss = kl_per_dim.mean()
        beta = self._current_adapter_vae_beta(optimizer_step_hint=optimizer_step_hint)
        reconstruction_weight = max(
            0.0,
            float(getattr(self.config, "adapter_variational_reconstruction_weight", 0.0)),
        )

        z_flat = z.flatten(start_dim=1)
        return {
            "reconstruction_loss_tensor": reconstruction_loss,
            "kl_loss_tensor": kl_loss,
            "raw_kl_loss_tensor": raw_kl_loss,
            "total_loss_tensor": reconstruction_loss * reconstruction_weight + kl_loss * beta,
            "reconstruction_loss": float(reconstruction_loss.detach().item()),
            "kl_loss": float(kl_loss.detach().item()),
            "raw_kl_loss": float(raw_kl_loss.detach().item()),
            "beta": float(beta),
            "reconstruction_weight": float(reconstruction_weight),
            "kl_per_dim_mean": float(kl_per_dim.mean().detach().item()),
            "kl_per_dim_max": float(kl_per_dim.max().detach().item()),
            "free_bits_active_fraction": float((kl_per_dim > max(free_bits, 1e-8)).float().mean().detach().item()),
            "z_norm_l2_mean": float(z_flat.norm(dim=1).mean().detach().item()),
            "z_std_mean": float(z_flat.std(dim=1, unbiased=False).mean().detach().item()),
            "posterior_mu_abs_mean": float(mu.abs().mean().detach().item()),
            "posterior_logvar_mean": float(logvar.mean().detach().item()),
        }

    def _map_to_rgb_image(self, array_like: Any) -> Image.Image:
        if torch.is_tensor(array_like):
            data = array_like.detach().float().cpu().numpy()
        else:
            data = np.asarray(array_like, dtype=np.float32)
        if data.ndim == 4:
            data = data[0]
        if data.ndim == 3:
            data = data.mean(axis=0)
        data = np.asarray(data, dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data_min = float(data.min()) if data.size else 0.0
        data_max = float(data.max()) if data.size else 0.0
        if data_max - data_min < 1e-8:
            normalized = np.zeros_like(data, dtype=np.uint8)
        else:
            normalized = np.clip((data - data_min) / (data_max - data_min), 0.0, 1.0)
            normalized = (normalized * 255.0).astype(np.uint8)
        return Image.fromarray(normalized).convert("RGB")

    def _make_snapshot_panel(self, image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
        resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
        panel = ImageOps.fit(image.convert("RGB"), size, method=resample)
        canvas = Image.new("RGB", (size[0], size[1] + 24), color=(255, 255, 255))
        canvas.paste(panel, (0, 24))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, size[0], 24), fill=(24, 24, 24))
        draw.text((6, 5), label, fill=(255, 255, 255))
        return canvas

    def _write_latent_snapshot_image(
        self,
        *,
        save_path: Path,
        control_image: Image.Image,
        adapted_image: Image.Image,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        kl_map: torch.Tensor,
    ) -> None:
        panel_size = (192, 256)
        panels = [
            self._make_snapshot_panel(control_image, "control", panel_size),
            self._make_snapshot_panel(adapted_image, "adapted", panel_size),
            self._make_snapshot_panel(self._map_to_rgb_image(mu.abs()), "mu_abs", panel_size),
            self._make_snapshot_panel(self._map_to_rgb_image(logvar), "logvar", panel_size),
            self._make_snapshot_panel(self._map_to_rgb_image(z), "z_mean", panel_size),
            self._make_snapshot_panel(self._map_to_rgb_image(kl_map), "kl_mean", panel_size),
        ]
        canvas = Image.new("RGB", (panel_size[0] * 3, (panel_size[1] + 24) * 2), color=(245, 245, 245))
        for index, panel in enumerate(panels):
            row = index // 3
            col = index % 3
            canvas.paste(panel, (col * panel.width, row * panel.height))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(save_path)

    @torch.no_grad()
    def _save_epoch_latent_artifacts(self, *, epoch: int, adapter) -> dict[str, Any]:
        if not self.accelerator.is_local_main_process or not self._adapter_variational_enabled():
            return {}
        records = list(self.preview_records or self.metric_eval_records)
        max_samples = max(1, int(getattr(self.config, "latent_snapshot_samples", 3)))
        if not records:
            return {}

        epoch_dir = self.run_dir / "latents" / f"epoch_{int(epoch):03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = epoch_dir / "latent_snapshot_manifest.json"
        unwrapped_adapter = self.accelerator.unwrap_model(adapter)
        previous_training = bool(unwrapped_adapter.training)
        unwrapped_adapter.eval()
        resample = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
        rows: list[dict[str, Any]] = []
        aggregate_fields: dict[str, list[float]] = {
            "kl_per_dim_mean": [],
            "kl_per_dim_max": [],
            "z_norm_l2_mean": [],
            "z_std_mean": [],
            "posterior_mu_abs_mean": [],
            "posterior_logvar_mean": [],
        }
        try:
            for index, record in enumerate(records[:max_samples], start=1):
                lineart_gray = np.array(Image.open(record.lineart_path).convert("L"), dtype=np.uint8)
                control_gray = build_control_image(lineart_gray, self.config.controlnet_model or "scribble")
                control_image = Image.fromarray(control_gray).convert("RGB").resize(
                    (self.config.image_width, self.config.image_height),
                    resample,
                )
                control_np = np.array(control_image, dtype=np.uint8)
                control_tensor = torch.from_numpy(control_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                control_tensor = control_tensor.to(self.accelerator.device, dtype=torch.float32)
                adapted, adapter_stats = unwrapped_adapter(control_tensor, return_stats=True)
                mu = adapter_stats.get("posterior_mu")
                logvar = adapter_stats.get("posterior_logvar")
                z = adapter_stats.get("latent_sample")
                if not (torch.is_tensor(mu) and torch.is_tensor(logvar) and torch.is_tensor(z)):
                    continue
                mu = mu.float()
                logvar = logvar.float()
                z = z.float()
                kl_map = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
                kl_per_dim = kl_map.mean(dim=(0, 2, 3))
                z_flat = z.flatten(start_dim=1)
                adapted_image = Image.fromarray(
                    (adapted[0].detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                )
                sample_prefix = f"{index:03d}_{record.image_id}"
                npz_path = epoch_dir / f"{sample_prefix}_latent.npz"
                snapshot_path = epoch_dir / f"{sample_prefix}_snapshot.png"
                np.savez_compressed(
                    npz_path,
                    mu=mu[0].detach().cpu().numpy(),
                    logvar=logvar[0].detach().cpu().numpy(),
                    z=z[0].detach().cpu().numpy(),
                    kl=kl_map[0].detach().cpu().numpy(),
                )
                self._write_latent_snapshot_image(
                    save_path=snapshot_path,
                    control_image=control_image,
                    adapted_image=adapted_image,
                    mu=mu,
                    logvar=logvar,
                    z=z,
                    kl_map=kl_map,
                )
                row = {
                    "image_id": record.image_id,
                    "lineart_path": record.lineart_path,
                    "latent_npz_path": str(npz_path.resolve()),
                    "snapshot_path": str(snapshot_path.resolve()),
                    "kl_per_dim_mean": float(kl_per_dim.mean().detach().item()),
                    "kl_per_dim_max": float(kl_per_dim.max().detach().item()),
                    "z_norm_l2_mean": float(z_flat.norm(dim=1).mean().detach().item()),
                    "z_std_mean": float(z_flat.std(dim=1, unbiased=False).mean().detach().item()),
                    "posterior_mu_abs_mean": float(mu.abs().mean().detach().item()),
                    "posterior_logvar_mean": float(logvar.mean().detach().item()),
                }
                rows.append(row)
                for key in aggregate_fields:
                    aggregate_fields[key].append(float(row[key]))
        finally:
            if previous_training:
                unwrapped_adapter.train()

        aggregate = {
            key: (float(sum(values) / len(values)) if values else None)
            for key, values in aggregate_fields.items()
        }
        payload = {
            "epoch": int(epoch),
            "saved_at": now_iso(),
            "epoch_dir": str(epoch_dir.resolve()),
            "samples_saved": len(rows),
            "aggregate": aggregate,
            "rows": rows,
        }
        save_json(manifest_path, payload)
        self.latest_latent_snapshot_dir = str(epoch_dir.resolve())
        return {
            "latent_snapshot_dir": str(epoch_dir.resolve()),
            "latent_snapshot_manifest_path": str(manifest_path.resolve()),
            "latent_snapshot_samples": len(rows),
            **aggregate,
        }

    def _epoch_sanity_recovery_profile(self) -> dict[str, Any]:
        below_threshold_count = max(0, int(getattr(self, "epoch_sanity_recovery_below_threshold_count", 0) or 0))
        min_ratio = getattr(self, "epoch_sanity_recovery_min_ratio", None)
        warning_has_rapid_drop = bool(getattr(self, "epoch_sanity_recovery_warning_has_rapid_drop", False))
        multiplier_step = max(
            0.0,
            float(getattr(self.config, "epoch_sanity_recovery_weight_multiplier_step", 0.35)),
        )
        max_multiplier = max(
            1.0,
            float(getattr(self.config, "epoch_sanity_recovery_max_weight_multiplier", 2.5)),
        )

        severity_signals = 0
        if below_threshold_count >= 1:
            severity_signals += 1
        if below_threshold_count >= 2:
            severity_signals += 1
        if _is_finite_number(min_ratio) and float(min_ratio) < 0.95:
            severity_signals += 1
        if _is_finite_number(min_ratio) and float(min_ratio) < 0.9:
            severity_signals += 1
        if warning_has_rapid_drop:
            severity_signals += 1

        weight_multiplier = min(
            max_multiplier,
            1.0 + multiplier_step * float(severity_signals),
        )
        image_guidance_active = bool(
            below_threshold_count > 0
            or warning_has_rapid_drop
            or (_is_finite_number(min_ratio) and float(min_ratio) < 0.95)
        )
        return {
            "below_threshold_count": below_threshold_count,
            "min_ratio": float(min_ratio) if _is_finite_number(min_ratio) else None,
            "warning_has_rapid_drop": warning_has_rapid_drop,
            "severity_signals": severity_signals,
            "weight_multiplier": float(weight_multiplier),
            "image_guidance_active": image_guidance_active,
        }

    def _epoch_sanity_recovery_weights(self, epoch: int) -> dict[str, float]:
        if not self._epoch_sanity_recovery_active(epoch):
            return {
                "latent_reconstruction": 0.0,
                "latent_distribution": 0.0,
                "latent_local_variance": 0.0,
                "latent_gradient": 0.0,
                "latent_global_flatness": 0.0,
                "image_reconstruction": 0.0,
                "image_flat_color": 0.0,
                "image_gradient": 0.0,
                "weight_multiplier": 1.0,
            }
        if self.recovery_aux_disabled_due_to_skips:
            return {
                "latent_reconstruction": 0.0,
                "latent_distribution": 0.0,
                "latent_local_variance": 0.0,
                "latent_gradient": 0.0,
                "latent_global_flatness": 0.0,
                "image_reconstruction": 0.0,
                "image_flat_color": 0.0,
                "image_gradient": 0.0,
                "weight_multiplier": 1.0,
            }
        recovery_profile = self._epoch_sanity_recovery_profile()
        weight_multiplier = float(recovery_profile["weight_multiplier"])
        image_reconstruction_weight = 0.0
        image_flat_color_weight = 0.0
        image_gradient_weight = 0.0
        if (
            bool(recovery_profile["image_guidance_active"])
            and not self.recovery_image_aux_disabled_due_to_oom
            and not self.recovery_image_aux_disabled_due_to_skips
        ):
            image_reconstruction_weight = (
                float(getattr(self.config, "epoch_sanity_recovery_image_reconstruction_weight", 0.06))
                * weight_multiplier
            )
            image_flat_color_weight = (
                float(getattr(self.config, "epoch_sanity_recovery_image_flat_color_weight", 0.12))
                * weight_multiplier
            )
            image_gradient_weight = (
                float(getattr(self.config, "epoch_sanity_recovery_image_gradient_weight", 0.05))
                * weight_multiplier
            )
        return {
            "latent_reconstruction": (
                float(getattr(self.config, "epoch_sanity_recovery_latent_reconstruction_weight", 0.03))
                * weight_multiplier
            ),
            "latent_distribution": (
                float(getattr(self.config, "epoch_sanity_recovery_latent_distribution_weight", 0.04))
                * weight_multiplier
            ),
            "latent_local_variance": (
                float(getattr(self.config, "epoch_sanity_recovery_latent_local_variance_weight", 0.03))
                * weight_multiplier
            ),
            "latent_gradient": (
                float(getattr(self.config, "epoch_sanity_recovery_latent_gradient_weight", 0.08))
                * weight_multiplier
            ),
            "latent_global_flatness": (
                float(getattr(self.config, "epoch_sanity_recovery_latent_global_flatness_weight", 0.08))
                * weight_multiplier
            ),
            "image_reconstruction": image_reconstruction_weight,
            "image_flat_color": image_flat_color_weight,
            "image_gradient": image_gradient_weight,
            "weight_multiplier": weight_multiplier,
        }

    def _compute_effective_adversarial_weights(
        self,
        *,
        epoch: int,
        critic_real_score: float | None,
        critic_fake_score: float | None,
        decoded_image_spatial_std: float | None,
    ) -> tuple[float, float, float]:
        adv_weight = float(self.config.wgan_generator_weight)
        recon_weight = float(self.config.wgan_reconstruction_weight)
        flat_weight = float(self.config.anti_flat_color_weight)

        if not self._wgan_is_active(epoch):
            return 0.0, recon_weight, flat_weight

        adv_weight *= self._adversarial_warmup_factor(epoch)

        if (
            _is_finite_number(critic_real_score)
            and _is_finite_number(critic_fake_score)
            and float(critic_fake_score) >= float(critic_real_score)
        ):
            adv_weight *= float(self.config.wgan_balance_penalty_factor)
            recon_weight *= 1.25
            flat_weight *= 1.25

        std_guard_threshold = float(self.config.anti_flat_color_min_std) * 1.15
        if _is_finite_number(decoded_image_spatial_std) and float(decoded_image_spatial_std) < std_guard_threshold:
            adv_weight *= 0.25
            recon_weight *= 1.5
            flat_weight *= 2.0

        if self._collapse_guard_active(epoch):
            adv_weight *= float(self.config.collapse_guard_adversarial_scale)
            recon_weight *= float(self.config.collapse_guard_reconstruction_boost)
            flat_weight *= float(self.config.collapse_guard_flat_boost)

        return adv_weight, recon_weight, flat_weight

    def _detect_collapse_from_generation_metrics(self, generation_metrics: dict[str, Any]) -> tuple[bool, str]:
        if not self.config.collapse_guard_enabled:
            return False, ""
        if not generation_metrics:
            return False, ""

        color_bleeding = generation_metrics.get("color_bleeding_rate")
        ssim = generation_metrics.get("ssim")
        fid = generation_metrics.get("fid")
        precision = generation_metrics.get("precision")
        recall = generation_metrics.get("recall")
        best_fid = None if self.best_fid == float("inf") else self.best_fid

        bleeding_flag = _is_finite_number(color_bleeding) and float(color_bleeding) >= float(self.config.collapse_guard_color_bleeding_threshold)
        ssim_flag = _is_finite_number(ssim) and float(ssim) <= float(self.config.collapse_guard_ssim_threshold)
        precision_flag = _is_finite_number(precision) and float(precision) <= float(self.config.collapse_guard_precision_threshold)
        recall_flag = _is_finite_number(recall) and float(recall) <= float(self.config.collapse_guard_precision_threshold)
        fid_flag = (
            _is_finite_number(fid)
            and _is_finite_number(best_fid)
            and float(fid) >= float(best_fid) * float(self.config.collapse_guard_fid_ratio_threshold)
        )

        hard_collapse = (bleeding_flag and ssim_flag) or (fid_flag and precision_flag and recall_flag)
        if not hard_collapse:
            return False, ""

        reasons: list[str] = []
        if bleeding_flag:
            reasons.append(f"color_bleeding_rate={float(color_bleeding):.4f}")
        if ssim_flag:
            reasons.append(f"ssim={float(ssim):.4f}")
        if fid_flag:
            reasons.append(f"fid={float(fid):.4f}")
        if precision_flag:
            reasons.append(f"precision={float(precision):.4f}")
        if recall_flag:
            reasons.append(f"recall={float(recall):.4f}")
        return True, "; ".join(reasons)

    def _activate_collapse_guard(self, *, epoch: int, reason: str, optimizer) -> None:
        self.collapse_guard_last_reason = reason
        cooldown_epochs = max(1, int(self.config.collapse_guard_cooldown_epochs))
        self.adversarial_cooldown_until_epoch = max(self.adversarial_cooldown_until_epoch, int(epoch) + cooldown_epochs)
        if self.best_fid_learning_rate is None:
            return
        if not self.latest_quality_recovery_triggered:
            self.best_fid_lr_recovery_count += 1
        self.best_fid_lr_recovery_active = True
        self.latest_quality_recovery_triggered = True
        self._apply_best_fid_lr_if_needed(optimizer)

    def _update_quality_decline_state(self, current_metric_value: float | None) -> bool:
        self.latest_quality_recovery_triggered = False
        if not _is_finite_number(current_metric_value):
            return False

        metric_value = float(current_metric_value)
        if self.previous_quality_metric_value is not None:
            if _metric_higher_is_better(self.config.quality_monitor_metric):
                declined = metric_value < self.previous_quality_metric_value
            else:
                declined = metric_value > self.previous_quality_metric_value
            self.consecutive_quality_decline_epochs = self.consecutive_quality_decline_epochs + 1 if declined else 0
        else:
            self.consecutive_quality_decline_epochs = 0

        self.previous_quality_metric_value = metric_value
        patience = max(1, int(self.config.quality_decline_patience_epochs))
        if not self.config.enable_best_fid_lr_recovery:
            return False
        if self.consecutive_quality_decline_epochs < patience:
            return False
        if self.best_fid_learning_rate is None:
            return False

        self.latest_quality_recovery_triggered = True
        self.best_fid_lr_recovery_active = True
        self.best_fid_lr_recovery_count += 1
        self.consecutive_quality_decline_epochs = 0
        return True

    def _publish_best_fid_display_artifacts(self, *, generation_metrics: dict[str, Any], preview_path: str) -> None:
        eval_dir_text = str(generation_metrics.get("eval_dir", "")).strip()
        archive_dir_text = str(generation_metrics.get("archive_dir", "")).strip()

        if eval_dir_text:
            source_eval_dir = Path(eval_dir_text)
            target_eval_dir = self.run_dir / "evaluations" / "best_fid_epoch"
            _remove_path(target_eval_dir)
            if source_eval_dir.exists():
                shutil.copytree(source_eval_dir, target_eval_dir)
                self.best_fid_eval_dir = str(target_eval_dir.resolve())

                preview_name = Path(preview_path).name if preview_path else ""
                target_preview_path = target_eval_dir / "generated" / preview_name if preview_name else None
                if target_preview_path is not None and target_preview_path.exists():
                    self.best_fid_preview_path = str(target_preview_path.resolve())
                elif preview_path:
                    self.best_fid_preview_path = preview_path

        if archive_dir_text:
            source_archive_dir = Path(archive_dir_text)
            target_archive_dir = self._training_validation_archive_seed_dir() / "best_fid"
            _remove_path(target_archive_dir)
            if source_archive_dir.exists():
                shutil.copytree(source_archive_dir, target_archive_dir)
                self.best_fid_archive_dir = str(target_archive_dir.resolve())

        if self.best_fid_preview_path:
            self.latest_preview_path = self.best_fid_preview_path
        elif preview_path:
            self.latest_preview_path = preview_path

        if self.best_fid_archive_dir:
            self.latest_eval_archive_dir = self.best_fid_archive_dir
        elif archive_dir_text:
            self.latest_eval_archive_dir = archive_dir_text

    def write_status(self, **payload: Any) -> None:
        base = {
            "state": payload.pop("state", "running"),
            "group_id": self.group.group_id,
            "group_name": self.group.display_name,
            "seed": self.seed,
            "updated_at": now_iso(),
            "run_dir": str(self.run_dir.resolve()),
            "latest_preview_path": self.latest_preview_path,
            "best_fid_preview_path": self.best_fid_preview_path,
            "best_fid_eval_dir": self.best_fid_eval_dir,
            "best_fid_archive_dir": self.best_fid_archive_dir,
            "best_fid_learning_rate": self.best_fid_learning_rate,
            "quality_monitor_metric": self.config.quality_monitor_metric,
            "consecutive_quality_decline_epochs": self.consecutive_quality_decline_epochs,
            "best_fid_lr_recovery_count": self.best_fid_lr_recovery_count,
            "best_fid_lr_recovery_active": self.best_fid_lr_recovery_active,
            "latest_quality_recovery_triggered": self.latest_quality_recovery_triggered,
            "adversarial_cooldown_until_epoch": self.adversarial_cooldown_until_epoch,
            "collapse_guard_reason": self.collapse_guard_last_reason,
            "wgan_disabled_reason": self.wgan_disabled_reason,
            "wgan_disabled_epoch": self.wgan_disabled_epoch,
            "latest_eval_archive_dir": self.latest_eval_archive_dir,
            "training_xformers": self._training_xformers_payload(),
        }
        base.update(payload)
        save_json(self.status_path, base)

    def write_metadata(self) -> None:
        payload = {
            "created_at": self.run_created_at_iso,
            "group": {
                "group_id": self.group.group_id,
                "display_name": self.group.display_name,
                "description": self.group.description,
            },
            "flags": asdict(self.group.flags),
            "trainer_config": asdict(self.config),
            "effective_training_precision": self.effective_mixed_precision,
            "adversarial_cooldown_until_epoch": self.adversarial_cooldown_until_epoch,
            "adapter_config": asdict(self.adapter_config),
            "inference_defaults": asdict(self.inference_defaults),
            "training_xformers": self._training_xformers_payload(),
        }
        save_json(self.run_dir / "run_metadata.json", payload)

    def build_dataloaders(self):
        split_bundle = create_or_load_split(
            dataset_root=self.config.dataset_root,
            color_dir_name=self.config.color_dir_name,
            lineart_dir_name=self.config.lineart_dir_name,
            train_ratio=float(self.config.train_ratio),
            val_ratio=float(self.config.val_ratio),
            test_ratio=float(self.config.test_ratio),
            split_seed=int(self.config.split_seed),
            output_path=self.split_path,
            use_all_training_pairs_for_training=bool(self.config.use_all_training_pairs_for_training),
        )
        val_records_for_loss: list[PairRecord] = list(split_bundle.val)
        if self.config.prefer_external_validation_dataset:
            try:
                val_records_for_loss = discover_pairs(
                    self.config.validation_dataset_root,
                    self.config.validation_color_dir_name,
                    self.config.validation_lineart_dir_name,
                )
            except Exception:
                val_records_for_loss = list(split_bundle.val)
        validation_selection = select_validation_records(
            split_val_records=list(split_bundle.val),
            validation_dataset_root=self.config.validation_dataset_root,
            validation_color_dir_name=self.config.validation_color_dir_name,
            validation_lineart_dir_name=self.config.validation_lineart_dir_name,
            prefer_external_validation_dataset=self.config.prefer_external_validation_dataset,
        )
        self.validation_source = validation_selection.source
        self.preview_source = validation_selection.preview_source
        self.validation_note = validation_selection.note
        self.metric_eval_records = list(validation_selection.metric_records)
        self.preview_records = list(validation_selection.preview_records)
        save_json(self.validation_selection_path, validation_selection.to_dict())
        train_dataset = LineartColorizationDataset(
            split_bundle.train,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
            controlnet_model=self.config.controlnet_model or "scribble",
            enable_horizontal_flip=self.config.horizontal_flip,
        )
        val_dataset = LineartColorizationDataset(
            val_records_for_loss,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
            controlnet_model=self.config.controlnet_model or "scribble",
            enable_horizontal_flip=False,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=max(1, min(self.config.num_workers, 2)),
            pin_memory=True,
        )
        return split_bundle, train_dataset, val_dataset, train_loader, val_loader

    def encode_prompts(self, tokenizer, tokenizer_2, text_encoder, text_encoder_2, batch_size: int, device: torch.device, dtype: torch.dtype):
        prompt = [self.config.prompt_template] * batch_size
        tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        tokens_2 = tokenizer_2(
            prompt,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            enc1 = text_encoder(tokens.input_ids, output_hidden_states=True)
            enc2 = text_encoder_2(tokens_2.input_ids, output_hidden_states=True)
            encoder_hidden_states = torch.cat([enc1.hidden_states[-2], enc2.hidden_states[-2]], dim=-1)
            text_embeds = enc2.text_embeds
            add_time_ids = torch.tensor(
                [[self.config.image_height, self.config.image_width, 0, 0, self.config.image_height, self.config.image_width]],
                device=device,
                dtype=dtype,
            ).repeat(batch_size, 1)
        return encoder_hidden_states, {"text_embeds": text_embeds, "time_ids": add_time_ids}

    def save_checkpoint(self, unet, adapter, step: int, epoch: int, critic=None) -> None:
        if not self.accelerator.is_local_main_process:
            return
        checkpoint_path = self.checkpoint_dir / f"step_{step:06d}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        unwrapped_unet = self.accelerator.unwrap_model(unet)
        unwrapped_adapter = self.accelerator.unwrap_model(adapter)
        unwrapped_unet.save_pretrained(checkpoint_path / "lora")
        torch.save(unwrapped_adapter.state_dict(), checkpoint_path / "adapter.pt")
        if critic is not None:
            torch.save(self.accelerator.unwrap_model(critic).state_dict(), checkpoint_path / "critic.pt")
        save_json(
            checkpoint_path / "checkpoint.json",
            {
                "step": step,
                "epoch": epoch,
                "saved_at": now_iso(),
            },
        )

    def save_named_checkpoint(self, unet, adapter, name: str, epoch: int, critic=None) -> str:
        if not self.accelerator.is_local_main_process:
            return ""
        checkpoint_path = self.checkpoint_dir / name
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        unwrapped_unet = self.accelerator.unwrap_model(unet)
        unwrapped_adapter = self.accelerator.unwrap_model(adapter)
        unwrapped_unet.save_pretrained(checkpoint_path / "lora")
        torch.save(unwrapped_adapter.state_dict(), checkpoint_path / "adapter.pt")
        if critic is not None:
            torch.save(self.accelerator.unwrap_model(critic).state_dict(), checkpoint_path / "critic.pt")
        save_json(
            checkpoint_path / "checkpoint.json",
            {
                "name": name,
                "epoch": epoch,
                "step": self.global_step,
                "saved_at": now_iso(),
            },
        )
        return str(checkpoint_path.resolve())

    def save_named_checkpoint_from_existing(self, source_checkpoint_dir: str | Path, name: str, epoch: int) -> str:
        if not self.accelerator.is_local_main_process:
            return ""
        source_path = Path(source_checkpoint_dir)
        if not source_path.exists():
            return ""
        checkpoint_path = self.checkpoint_dir / name
        _remove_path(checkpoint_path)
        shutil.copytree(source_path, checkpoint_path)
        source_metadata = load_json(source_path / "checkpoint.json")
        save_json(
            checkpoint_path / "checkpoint.json",
            {
                "name": name,
                "epoch": int(epoch),
                "step": source_metadata.get("step"),
                "saved_at": now_iso(),
                "source_checkpoint_dir": str(source_path.resolve()),
            },
        )
        return str(checkpoint_path.resolve())

    def _latest_checkpoint_dir_for_epoch(self, epoch: int) -> Path | None:
        candidates: list[tuple[int, Path]] = []
        for checkpoint_path in sorted(self.checkpoint_dir.glob("step_*")):
            metadata = load_json(checkpoint_path / "checkpoint.json")
            try:
                checkpoint_epoch = int(metadata.get("epoch", -1))
            except Exception:
                checkpoint_epoch = -1
            if checkpoint_epoch != int(epoch):
                continue
            try:
                checkpoint_step = int(metadata.get("step", 0))
            except Exception:
                checkpoint_step = 0
            candidates.append((checkpoint_step, checkpoint_path))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], str(item[1])))
        return candidates[-1][1]

    def _latest_checkpoint_dir_up_to_epoch(self, epoch: int) -> tuple[int, Path] | None:
        candidates: list[tuple[int, int, Path]] = []
        for checkpoint_path in sorted(self.checkpoint_dir.glob("step_*")):
            metadata = load_json(checkpoint_path / "checkpoint.json")
            try:
                checkpoint_epoch = int(metadata.get("epoch", -1))
            except Exception:
                checkpoint_epoch = -1
            if checkpoint_epoch <= 0 or checkpoint_epoch > int(epoch):
                continue
            try:
                checkpoint_step = int(metadata.get("step", 0))
            except Exception:
                checkpoint_step = 0
            if checkpoint_step <= 0:
                continue
            candidates.append((checkpoint_epoch, checkpoint_step, checkpoint_path))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
        checkpoint_epoch, _, checkpoint_path = candidates[-1]
        return checkpoint_epoch, checkpoint_path

    def _refresh_epoch_metric_logs(self) -> None:
        preserved_rows = [row for row in load_jsonl(self.metric_log_path) if str(row.get("event", "")) != "epoch_end"]
        save_jsonl(self.metric_log_path, preserved_rows + self.epoch_history)
        save_json(self.run_dir / "logs" / "epoch_history.json", {"epochs": self.epoch_history})

    def _merge_posthoc_generation_metrics(self, epoch_record: dict[str, Any], generation_metrics: dict[str, Any]) -> None:
        direct_keys = [
            "fid",
            "precision",
            "recall",
            "f_score",
            "pr_curve_auc",
            "lpips",
            "lpips_mean",
            "lpips_std",
            "lpips_source",
            "lpips_internal",
            "ssim",
            "ssim_mean",
            "ssim_std",
            "ssim_source",
            "ssim_internal",
            "edge_consistency",
            "color_bleeding_rate",
            "histogram_correlation",
            "inference_time_ms",
            "subgroup_metrics",
            "eval_dir",
            "generated_dir",
            "target_dir",
            "lineart_dir",
            "archive_dir",
            "helper_metrics_available",
            "helper_metric_reports",
            "helper_metric_errors",
            "pr_curve_csv",
            "pr_curve_plot",
            "pr_metrics_error",
            "fid_computed",
            "fid_source",
            "fid_internal",
            "params_m",
            "flops_g",
            "validation_source",
            "validation_note",
            "generation_records_path",
            "generated_samples_count",
        ]
        for key in direct_keys:
            epoch_record[key] = generation_metrics.get(key)
        epoch_record["eval_gpu_memory_peak_gb"] = generation_metrics.get("gpu_memory_peak_gb")
        epoch_record["eval_gpu_memory_reserved_peak_gb"] = generation_metrics.get("gpu_memory_reserved_peak_gb")
        epoch_record["eval_cpu_memory_peak_gb"] = generation_metrics.get("cpu_memory_peak_gb")
        epoch_record["generation_metrics_deferred"] = False
        epoch_record["posthoc_metrics_completed"] = True
        epoch_record["metrics_computed_at"] = now_iso()

    def _release_training_models_to_cpu(self, vae, text_encoder, text_encoder_2, controlnet, unet, adapter, critic) -> None:
        for name, model in [
            ("vae", vae),
            ("text_encoder", text_encoder),
            ("text_encoder_2", text_encoder_2),
            ("controlnet", controlnet),
            ("unet", unet),
            ("adapter", adapter),
            ("critic", critic),
        ]:
            if model is None:
                continue
            try:
                model.cpu()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()

    def _finalize_deferred_generation_metrics(self, *, vae, text_encoder, text_encoder_2, controlnet, unet, adapter, critic) -> dict[str, Any]:
        if not bool(getattr(self.config, "defer_generation_metrics_until_seed_end", False)):
            return {}
        if not self.accelerator.is_local_main_process:
            return {}

        validation_epochs_root = self.run_dir / "evaluations" / "validation_epochs"
        if not validation_epochs_root.exists():
            return {}

        best_payload: dict[str, Any] = {}
        best_preview_path = ""
        latest_payload: dict[str, Any] = {}
        successful_epochs = 0
        epoch_records = sorted(self.epoch_history, key=lambda item: int(item.get("epoch", 0)))
        total_epochs = len([item for item in epoch_records if int(item.get("epoch", 0)) > 0])

        self.best_fid = float("inf")
        self.best_fid_epoch = 0
        self.best_fid_checkpoint_path = ""
        self.best_fid_preview_path = ""
        self.best_fid_eval_dir = ""
        self.best_fid_archive_dir = ""
        self.best_fid_learning_rate = None

        for index, epoch_record in enumerate(epoch_records, start=1):
            epoch = int(epoch_record.get("epoch", 0) or 0)
            if epoch <= 0:
                continue
            eval_dir = Path(epoch_record.get("eval_dir") or validation_epochs_root / f"epoch_{epoch:03d}")
            if not eval_dir.exists():
                epoch_record["generation_metrics_deferred"] = False
                epoch_record["posthoc_metrics_completed"] = False
                epoch_record["generation_metric_error"] = f"Missing evaluation directory: {eval_dir}"
                continue

            self.write_status(
                state="running",
                message=f"Computing deferred generation metrics ({index}/{max(total_epochs, 1)})",
                epoch=epoch,
                global_step=self.global_step,
                total_optimizer_steps=self.total_optimizer_steps,
                seed_started_at=self.seed_started_at_iso,
                seed_elapsed_seconds=self.seed_elapsed_seconds,
                seed_elapsed_hms=self.seed_elapsed_hms,
            )

            if index == 1:
                self._release_training_models_to_cpu(vae, text_encoder, text_encoder_2, controlnet, unet, adapter, critic)

            try:
                generation_metrics = compute_saved_epoch_metrics(
                    eval_dir=eval_dir,
                    device=self.accelerator.device,
                    compute_fid=should_compute_fid(epoch, int(self.config.epochs)),
                    compute_pr_curve=epoch == int(self.config.epochs),
                    helper_tools_root=self.config.helper_tools_root,
                    archive_root=self.config.inference_archive_root,
                    group_id=self.group.group_id,
                    seed=self.seed,
                    epoch=epoch,
                    params_m=self.params_m,
                    flops_g=self.flops_g,
                    validation_source=self.validation_source,
                    validation_note=self.validation_note,
                )
            except Exception as exc:
                epoch_record["generation_metrics_deferred"] = False
                epoch_record["posthoc_metrics_completed"] = False
                epoch_record["generation_metric_error"] = str(exc)
                continue

            self._merge_posthoc_generation_metrics(epoch_record, generation_metrics)
            epoch_record.pop("generation_metric_error", None)
            latest_payload = generation_metrics
            successful_epochs += 1
            self.latest_eval_gpu_memory_peak_gb = generation_metrics.get("gpu_memory_peak_gb")
            self.latest_eval_gpu_memory_reserved_peak_gb = generation_metrics.get("gpu_memory_reserved_peak_gb")
            self.latest_eval_cpu_memory_peak_gb = generation_metrics.get("cpu_memory_peak_gb")
            if generation_metrics.get("archive_dir"):
                self.latest_eval_archive_dir = generation_metrics["archive_dir"]

            fid_value = generation_metrics.get("fid")
            if _is_finite_number(fid_value) and float(fid_value) < self.best_fid:
                self.best_fid = float(fid_value)
                self.best_fid_epoch = epoch
                self.best_fid_learning_rate = epoch_record.get("learning_rate")
                checkpoint_source = self._latest_checkpoint_dir_for_epoch(epoch)
                self.best_fid_checkpoint_path = (
                    self.save_named_checkpoint_from_existing(checkpoint_source, "best_fid", epoch)
                    if checkpoint_source is not None
                    else ""
                )
                preview_candidates = sorted((Path(generation_metrics.get("generated_dir", ""))).glob("*.png"))
                best_preview_path = str(preview_candidates[0].resolve()) if preview_candidates else ""
                best_payload = dict(generation_metrics)

        if best_payload:
            self._publish_best_fid_display_artifacts(
                generation_metrics=best_payload,
                preview_path=best_preview_path,
            )
        elif latest_payload.get("archive_dir"):
            self.latest_eval_archive_dir = latest_payload["archive_dir"]

        for epoch_record in self.epoch_history:
            epoch_record["best_fid"] = None if self.best_fid == float("inf") else self.best_fid
            epoch_record["best_fid_epoch"] = self.best_fid_epoch
            epoch_record["best_fid_checkpoint_path"] = self.best_fid_checkpoint_path
            epoch_record["best_fid_preview_path"] = self.best_fid_preview_path
            epoch_record["best_fid_eval_dir"] = self.best_fid_eval_dir
            epoch_record["best_fid_archive_dir"] = self.best_fid_archive_dir
            epoch_record["best_fid_learning_rate"] = self.best_fid_learning_rate

        self._refresh_epoch_metric_logs()
        sync_run_localized_outputs(self.run_dir)
        return {
            "successful_epochs": successful_epochs,
            "latest_metrics": latest_payload,
            "best_metrics": best_payload,
        }

    def save_final_artifacts(self, unet, adapter, critic=None) -> None:
        if not self.accelerator.is_local_main_process:
            return
        final_lora_dir = self.run_dir / "lora"
        final_lora_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_unet = self.accelerator.unwrap_model(unet)
        unwrapped_adapter = self.accelerator.unwrap_model(adapter)
        unwrapped_unet.save_pretrained(final_lora_dir)
        torch.save(unwrapped_adapter.state_dict(), self.run_dir / "adapter.pt")
        if critic is not None:
            torch.save(self.accelerator.unwrap_model(critic).state_dict(), self.run_dir / "critic.pt")

    def _export_final_artifacts_from_checkpoint(self, source_checkpoint_dir: str | Path) -> None:
        source_path = Path(source_checkpoint_dir)
        if not source_path.exists():
            return
        source_lora_dir = source_path / "lora"
        source_adapter_path = source_path / "adapter.pt"
        source_critic_path = source_path / "critic.pt"

        final_lora_dir = self.run_dir / "lora"
        _remove_path(final_lora_dir)
        if source_lora_dir.exists():
            shutil.copytree(source_lora_dir, final_lora_dir)
        if source_adapter_path.exists():
            shutil.copy2(source_adapter_path, self.run_dir / "adapter.pt")
        if source_critic_path.exists():
            shutil.copy2(source_critic_path, self.run_dir / "critic.pt")

    def _should_use_cpu_offload_for_preview(self) -> bool:
        return bool(self.config.cpu_offload)

    def _compute_image_statistics(self, image_path: str | Path) -> dict[str, float]:
        image = Image.open(image_path).convert("RGB")
        array = np.asarray(image, dtype=np.float32)
        luminance = array.mean(axis=2)
        grad_y, grad_x = np.gradient(luminance)
        gradient_magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))
        p05 = float(np.percentile(luminance, 5))
        p95 = float(np.percentile(luminance, 95))
        return {
            "mean": float(luminance.mean()),
            "std": float(luminance.std()),
            "dynamic_range": float(max(0.0, p95 - p05)),
            "gradient_mean": float(gradient_magnitude.mean()),
            "min": float(luminance.min()),
            "max": float(luminance.max()),
        }

    def _run_epoch_sanity_check(
        self,
        *,
        epoch: int,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        vae,
        controlnet,
        unet,
        adapter,
        base_path: str,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        if not self.accelerator.is_local_main_process or not bool(getattr(self.config, "enable_epoch_sanity_check", False)):
            return {}

        records = list(self.preview_records or self.metric_eval_records)
        records = records[: max(1, int(getattr(self.config, "epoch_sanity_check_samples", 1)))]
        if not records:
            return {}

        preview_paths: list[str] = []
        generated_stats: list[dict[str, float]] = []
        target_stats: list[dict[str, float]] = []
        for index, record in enumerate(records, start=1):
            preview_path = self.save_preview(
                [record],
                text_encoder,
                text_encoder_2,
                tokenizer,
                tokenizer_2,
                vae,
                controlnet,
                unet,
                adapter,
                base_path,
                epoch,
                dtype,
                name_prefix=f"sanity_{index}",
                num_inference_steps=int(getattr(self.config, "epoch_sanity_check_inference_steps", 10)),
                guidance_scale=float(getattr(self.config, "epoch_sanity_check_guidance_scale", 5.5)),
            )
            if not preview_path:
                continue
            preview_paths.append(preview_path)
            generated_stats.append(self._compute_image_statistics(preview_path))
            if getattr(record, "color_path", ""):
                target_stats.append(self._compute_image_statistics(record.color_path))

        if not generated_stats:
            return {}

        generated_std = float(sum(item["std"] for item in generated_stats) / len(generated_stats))
        generated_dynamic_range = float(sum(item["dynamic_range"] for item in generated_stats) / len(generated_stats))
        generated_gradient_mean = float(sum(item["gradient_mean"] for item in generated_stats) / len(generated_stats))
        target_std = float(sum(item["std"] for item in target_stats) / len(target_stats)) if target_stats else 0.0
        target_dynamic_range = (
            float(sum(item["dynamic_range"] for item in target_stats) / len(target_stats)) if target_stats else 0.0
        )
        target_gradient_mean = (
            float(sum(item["gradient_mean"] for item in target_stats) / len(target_stats)) if target_stats else 0.0
        )
        std_threshold = max(
            float(getattr(self.config, "epoch_sanity_min_std", 18.0)),
            target_std * float(getattr(self.config, "epoch_sanity_target_std_ratio", 0.35)),
        )
        dynamic_range_threshold = max(
            float(getattr(self.config, "epoch_sanity_min_dynamic_range", 72.0)),
            target_dynamic_range * float(getattr(self.config, "epoch_sanity_target_dynamic_range_ratio", 0.4)),
        )
        gradient_threshold = max(
            float(getattr(self.config, "epoch_sanity_min_gradient_mean", 1.25)),
            target_gradient_mean * float(getattr(self.config, "epoch_sanity_target_gradient_ratio", 0.35)),
        )
        low_std = generated_std < std_threshold
        low_dynamic_range = generated_dynamic_range < dynamic_range_threshold
        low_gradient = generated_gradient_mean < gradient_threshold
        collapsed = low_gradient and (low_std or low_dynamic_range)
        reason = ""
        if collapsed:
            failing_checks: list[str] = []
            if low_std:
                failing_checks.append(f"generated_std={generated_std:.2f} < {std_threshold:.2f}")
            if low_dynamic_range:
                failing_checks.append(f"generated_dynamic_range={generated_dynamic_range:.2f} < {dynamic_range_threshold:.2f}")
            if low_gradient:
                failing_checks.append(f"generated_gradient_mean={generated_gradient_mean:.2f} < {gradient_threshold:.2f}")
            reason = (
                "epoch_sanity_check_failed: " + " and ".join(failing_checks)
            )
        return {
            "collapsed": collapsed,
            "reason": reason,
            "preview_paths": preview_paths,
            "generated_std": generated_std,
            "generated_dynamic_range": generated_dynamic_range,
            "generated_gradient_mean": generated_gradient_mean,
            "target_std": target_std,
            "target_dynamic_range": target_dynamic_range,
            "target_gradient_mean": target_gradient_mean,
            "std_threshold": std_threshold,
            "dynamic_range_threshold": dynamic_range_threshold,
            "gradient_threshold": gradient_threshold,
        }

    @torch.no_grad()
    def evaluate_val_loss(self, val_loader, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, controlnet, unet, adapter, noise_scheduler, dtype: torch.dtype) -> float | None:
        unet.eval()
        adapter.eval()
        losses: list[float] = []
        for batch_index, batch in enumerate(val_loader):
            if batch_index >= max(1, self.config.max_eval_samples // max(self.config.batch_size, 1)):
                break
            color_images = _sanitize_tensor(batch["color"].to(self.accelerator.device, dtype=dtype), nan=0.0, posinf=1.0, neginf=-1.0)
            lineart_images = batch["lineart"].to(self.accelerator.device, dtype=torch.float32)
            lineart_images = _sanitize_tensor(adapter(lineart_images), nan=0.0, posinf=1.0, neginf=0.0).to(dtype)
            with torch.no_grad():
                latents = vae.encode(color_images).latent_dist.sample()
                latents = _sanitize_tensor(latents * vae.config.scaling_factor)
                noise = _sanitize_tensor(torch.randn_like(latents), nan=0.0, posinf=1.0, neginf=-1.0)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = _sanitize_tensor(noise_scheduler.add_noise(latents, noise, timesteps))
                encoder_hidden_states, added_cond_kwargs = self.encode_prompts(
                    tokenizer,
                    tokenizer_2,
                    text_encoder,
                    text_encoder_2,
                    batch_size=bsz,
                    device=self.accelerator.device,
                    dtype=dtype,
                )
                encoder_hidden_states = _sanitize_tensor(encoder_hidden_states)
                added_cond_kwargs = {
                    key: _sanitize_tensor(value, nan=0.0, posinf=10.0, neginf=-10.0)
                    for key, value in added_cond_kwargs.items()
                }
                down_block_residuals, mid_block_residual = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=lineart_images,
                    conditioning_scale=float(self.config.controlnet_conditioning_scale),
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )
                down_block_residuals = tuple(_sanitize_tensor(item) for item in down_block_residuals)
                mid_block_residual = _sanitize_tensor(mid_block_residual)
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_residuals,
                    mid_block_additional_residual=mid_block_residual,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                model_pred = _sanitize_tensor(model_pred)
                loss_tensor = F.mse_loss(model_pred.float(), noise.float())
                if torch.isfinite(loss_tensor):
                    losses.append(float(loss_tensor.item()))
        unet.train()
        adapter.train()
        if not losses:
            return None
        return float(sum(losses) / len(losses))

    @torch.no_grad()
    def generate_validation_images_for_epoch(
        self,
        *,
        epoch: int,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        vae,
        controlnet,
        unet,
        adapter,
        base_path: str,
        weight_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Generate validation images after training completes. Used for deferred FID computation."""
        if not self.accelerator.is_local_main_process:
            return {}
        eval_dir = self.run_dir / "evaluations" / "validation_epochs" / f"epoch_{epoch:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        generated_dir = eval_dir / "generated"
        target_dir = eval_dir / "target"
        lineart_dir = eval_dir / "lineart"
        generated_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        lineart_dir.mkdir(parents=True, exist_ok=True)
        memory_monitor = PeakMemoryMonitor(device=self.accelerator.device).start()
        preview_pipeline = None
        rows: list[dict[str, Any]] = []
        records = self.metric_eval_records[: max(1, int(self.config.max_eval_samples))]
        try:
            from diffusers import StableDiffusionXLControlNetPipeline

            scheduler = load_scheduler(base_path, "unipc")
            preview_pipeline = StableDiffusionXLControlNetPipeline(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=self.accelerator.unwrap_model(unet),
                controlnet=controlnet,
                scheduler=scheduler,
                add_watermarker=False,
            )
            preview_pipeline.enable_vae_slicing()
            preview_pipeline.enable_attention_slicing()
            maybe_enable_xformers(preview_pipeline, self.config.enable_xformers)
            if self._should_use_cpu_offload_for_preview():
                preview_pipeline.enable_model_cpu_offload()
            else:
                preview_pipeline = preview_pipeline.to(self.accelerator.device)

            # 推理前释放训练模型显存（Unet/Adapter不再需要）
            _release_torch_memory(self.accelerator.device)

            for index, record in enumerate(records, start=1):
                lineart_image = Image.open(record.lineart_path).convert("RGB").resize(
                    (self.config.image_width, self.config.image_height), Image.NEAREST
                )
                target_image = Image.open(record.color_path).convert("RGB").resize(
                    (self.config.image_width, self.config.image_height), Image.LANCZOS
                )
                lineart_np = np.array(lineart_image)
                lineart_tensor = torch.from_numpy(lineart_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                lineart_tensor = lineart_tensor.to(self.accelerator.device)
                adapted = self.accelerator.unwrap_model(adapter)(lineart_tensor).clamp(0.0, 1.0)
                control_image = Image.fromarray(
                    (adapted.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                )
                started = time.perf_counter()
                generated = preview_pipeline(
                    prompt=self.config.prompt_template,
                    negative_prompt=self.config.negative_prompt,
                    image=control_image,
                    num_inference_steps=20,
                    guidance_scale=7.0,
                    controlnet_conditioning_scale=float(self.config.controlnet_conditioning_scale),
                    width=self.config.image_width,
                    height=self.config.image_height,
                    generator=torch.Generator(device=self.accelerator.device).manual_seed(self.seed + index),
                ).images[0]
                inference_time_ms = (time.perf_counter() - started) * 1000.0
                file_name = f"{index:03d}_{record.image_id}.png"
                generated_path = generated_dir / file_name
                target_path = target_dir / file_name
                lineart_path = lineart_dir / file_name
                generated.save(generated_path)
                target_image.save(target_path)
                lineart_image.save(lineart_path)
                # 每张图推理后释放中间tensor显存
                del lineart_tensor, adapted, control_image, generated
                _release_torch_memory(self.accelerator.device)
                rows.append(
                    {
                        "image_id": record.image_id,
                        "file_name": file_name,
                        "generated_path": str(generated_path.resolve()),
                        "target_path": str(target_path.resolve()),
                        "lineart_path": str(lineart_path.resolve()),
                        "inference_time_ms": inference_time_ms,
                    }
                )

            if preview_pipeline is not None:
                del preview_pipeline
                preview_pipeline = None
            _release_torch_memory(self.accelerator.device)

            generation_records_payload = {
                "group_id": self.group.group_id,
                "seed": self.seed,
                "epoch": int(epoch),
                "validation_source": self.validation_source,
                "validation_note": self.validation_note,
                "params_m": self.params_m,
                "flops_g": self.flops_g,
                "rows": rows,
            }
            save_json(eval_dir / "generation_records.json", generation_records_payload)

            peak_memory = memory_monitor.stop()
            deferred_payload = {
                "eval_dir": str(eval_dir.resolve()),
                "generated_dir": str(generated_dir.resolve()),
                "target_dir": str(target_dir.resolve()),
                "lineart_dir": str(lineart_dir.resolve()),
                "generation_records_path": str((eval_dir / "generation_records.json").resolve()),
                "generated_samples_count": len(rows),
                "validation_source": self.validation_source,
                "validation_note": self.validation_note,
                "params_m": self.params_m,
                "flops_g": self.flops_g,
                "generation_metrics_deferred": True,
                "posthoc_metrics_completed": False,
                "fid_computed": False,
                "gpu_memory_peak_gb": peak_memory.get("gpu_memory_peak_gb"),
                "gpu_memory_reserved_peak_gb": peak_memory.get("gpu_memory_reserved_peak_gb"),
                "cpu_memory_peak_gb": peak_memory.get("cpu_memory_peak_gb"),
                "memory_unit": peak_memory.get("memory_unit", "GB"),
            }
            save_json(eval_dir / "metrics.json", deferred_payload)
            sync_evaluation_localized_outputs(eval_dir)
            return deferred_payload
        finally:
            memory_monitor.stop()
            if preview_pipeline is not None:
                del preview_pipeline
            _release_torch_memory(self.accelerator.device)

    @torch.no_grad()
    def save_preview(
        self,
        records: list[Any],
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        vae,
        controlnet,
        unet,
        adapter,
        base_path: str,
        epoch: int,
        dtype: torch.dtype,
        name_prefix: str = "epoch",
        *,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.0,
    ) -> str:
        if not self.accelerator.is_local_main_process or not records:
            return ""
        first_record = records[0]
        image = Image.open(first_record.lineart_path).convert("RGB").resize((self.config.image_width, self.config.image_height), Image.NEAREST)
        lineart_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        lineart_tensor = lineart_tensor.to(self.accelerator.device)
        adapted = self.accelerator.unwrap_model(adapter)(lineart_tensor).clamp(0.0, 1.0)
        control_image = Image.fromarray((adapted.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

        preview_pipeline = None
        try:
            from diffusers import StableDiffusionXLControlNetPipeline

            scheduler = load_scheduler(base_path, "unipc")
            preview_pipeline = StableDiffusionXLControlNetPipeline(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=self.accelerator.unwrap_model(unet),
                controlnet=controlnet,
                scheduler=scheduler,
                add_watermarker=False,
            )
            preview_pipeline.enable_vae_slicing()
            preview_pipeline.enable_attention_slicing()
            maybe_enable_xformers(preview_pipeline, self.config.enable_xformers)
            if self._should_use_cpu_offload_for_preview():
                preview_pipeline.enable_model_cpu_offload()
            else:
                preview_pipeline = preview_pipeline.to(self.accelerator.device)

            result = preview_pipeline(
                prompt=self.config.prompt_template,
                negative_prompt=self.config.negative_prompt,
                image=control_image,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                controlnet_conditioning_scale=float(self.config.controlnet_conditioning_scale),
                width=self.config.image_width,
                height=self.config.image_height,
                generator=torch.Generator(device=self.accelerator.device).manual_seed(self.seed),
            ).images[0]
            preview_path = self.preview_dir / f"{name_prefix}_{epoch:03d}.png"
            result.save(preview_path)
            return str(preview_path.resolve())
        finally:
            if preview_pipeline is not None:
                del preview_pipeline
            _release_torch_memory(self.accelerator.device)

    @torch.no_grad()
    def evaluate_generation_metrics(
        self,
        *,
        records: list[Any],
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        vae,
        controlnet,
        unet,
        adapter,
        base_path: str,
        epoch: int,
    ) -> dict[str, Any]:
        eval_dir = self.run_dir / "evaluations" / "validation_epochs" / f"epoch_{epoch:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        generated_dir = eval_dir / "generated"
        target_dir = eval_dir / "target"
        lineart_dir = eval_dir / "lineart"
        generated_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        lineart_dir.mkdir(parents=True, exist_ok=True)
        memory_monitor = PeakMemoryMonitor(device=self.accelerator.device).start()
        preview_pipeline = None
        rows: list[dict[str, Any]] = []
        try:
            from diffusers import StableDiffusionXLControlNetPipeline

            scheduler = load_scheduler(base_path, "unipc")
            preview_pipeline = StableDiffusionXLControlNetPipeline(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=self.accelerator.unwrap_model(unet),
                controlnet=controlnet,
                scheduler=scheduler,
                add_watermarker=False,
            )
            preview_pipeline.enable_vae_slicing()
            preview_pipeline.enable_attention_slicing()
            maybe_enable_xformers(preview_pipeline, self.config.enable_xformers)
            if self._should_use_cpu_offload_for_preview():
                preview_pipeline.enable_model_cpu_offload()
            else:
                preview_pipeline = preview_pipeline.to(self.accelerator.device)

            for index, record in enumerate(records, start=1):
                lineart_image = Image.open(record.lineart_path).convert("RGB").resize(
                    (self.config.image_width, self.config.image_height), Image.NEAREST
                )
                target_image = Image.open(record.color_path).convert("RGB").resize(
                    (self.config.image_width, self.config.image_height), Image.LANCZOS
                )
                lineart_np = np.array(lineart_image)
                lineart_tensor = torch.from_numpy(lineart_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                lineart_tensor = lineart_tensor.to(self.accelerator.device)
                adapted = self.accelerator.unwrap_model(adapter)(lineart_tensor).clamp(0.0, 1.0)
                control_image = Image.fromarray(
                    (adapted.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                )
                started = time.perf_counter()
                generated = preview_pipeline(
                    prompt=self.config.prompt_template,
                    negative_prompt=self.config.negative_prompt,
                    image=control_image,
                    num_inference_steps=20,
                    guidance_scale=7.0,
                    controlnet_conditioning_scale=float(self.config.controlnet_conditioning_scale),
                    width=self.config.image_width,
                    height=self.config.image_height,
                    generator=torch.Generator(device=self.accelerator.device).manual_seed(self.seed + index),
                ).images[0]
                inference_time_ms = (time.perf_counter() - started) * 1000.0
                file_name = f"{index:03d}_{record.image_id}.png"
                generated_path = generated_dir / file_name
                target_path = target_dir / file_name
                lineart_path = lineart_dir / file_name
                generated.save(generated_path)
                target_image.save(target_path)
                lineart_image.save(lineart_path)
                # 每张图推理后释放中间tensor显存
                del lineart_tensor, adapted, control_image, generated
                _release_torch_memory(self.accelerator.device)
                rows.append(
                    {
                        "image_id": record.image_id,
                        "file_name": file_name,
                        "generated_path": str(generated_path.resolve()),
                        "target_path": str(target_path.resolve()),
                        "lineart_path": str(lineart_path.resolve()),
                        "inference_time_ms": inference_time_ms,
                    }
                )

            if preview_pipeline is not None:
                del preview_pipeline
                preview_pipeline = None
            _release_torch_memory(self.accelerator.device)

            generation_records_payload = {
                "group_id": self.group.group_id,
                "seed": self.seed,
                "epoch": int(epoch),
                "validation_source": self.validation_source,
                "validation_note": self.validation_note,
                "params_m": self.params_m,
                "flops_g": self.flops_g,
                "rows": rows,
            }
            save_json(eval_dir / "generation_records.json", generation_records_payload)

            if bool(getattr(self.config, "defer_generation_metrics_until_seed_end", False)):
                peak_memory = memory_monitor.stop()
                deferred_payload = {
                    "eval_dir": str(eval_dir.resolve()),
                    "generated_dir": str(generated_dir.resolve()),
                    "target_dir": str(target_dir.resolve()),
                    "lineart_dir": str(lineart_dir.resolve()),
                    "generation_records_path": str((eval_dir / "generation_records.json").resolve()),
                    "generated_samples_count": len(rows),
                    "validation_source": self.validation_source,
                    "validation_note": self.validation_note,
                    "params_m": self.params_m,
                    "flops_g": self.flops_g,
                    "generation_metrics_deferred": True,
                    "posthoc_metrics_completed": False,
                    "fid_computed": False,
                    "gpu_memory_peak_gb": peak_memory.get("gpu_memory_peak_gb"),
                    "gpu_memory_reserved_peak_gb": peak_memory.get("gpu_memory_reserved_peak_gb"),
                    "cpu_memory_peak_gb": peak_memory.get("cpu_memory_peak_gb"),
                    "memory_unit": peak_memory.get("memory_unit", "GB"),
                    "gpu_memory_peak": peak_memory.get("gpu_memory_peak_gb"),
                    "cpu_memory_peak": peak_memory.get("cpu_memory_peak_gb"),
                }
                save_json(eval_dir / "metrics.json", deferred_payload)
                sync_evaluation_localized_outputs(eval_dir)
                return deferred_payload

            metrics_payload = compute_saved_epoch_metrics(
                eval_dir=eval_dir,
                device=self.accelerator.device,
                compute_fid=should_compute_fid(epoch, int(self.config.epochs)),
                compute_pr_curve=epoch == int(self.config.epochs),
                helper_tools_root=self.config.helper_tools_root,
                archive_root=self.config.inference_archive_root,
                group_id=self.group.group_id,
                seed=self.seed,
                epoch=epoch,
                params_m=self.params_m,
                flops_g=self.flops_g,
                validation_source=self.validation_source,
                validation_note=self.validation_note,
            )
            return metrics_payload
        finally:
            memory_monitor.stop()
            if preview_pipeline is not None:
                del preview_pipeline
            _release_torch_memory(self.accelerator.device)

    def update_summary(
        self,
        status: str,
        train_loss: float | None = None,
        val_loss: float | None = None,
        epoch: int = 0,
        **extra: Any,
    ) -> None:
        payload = {
            "status": status,
            "group_id": self.group.group_id,
            "group_name": self.group.display_name,
            "seed": self.seed,
            "run_dir": str(self.run_dir.resolve()),
            "best_train_loss": None if self.best_train_loss == float("inf") else self.best_train_loss,
            "best_val_loss": None if self.best_val_loss == float("inf") else self.best_val_loss,
            "best_fid": None if self.best_fid == float("inf") else self.best_fid,
            "best_fid_epoch": self.best_fid_epoch,
            "best_fid_checkpoint_path": self.best_fid_checkpoint_path,
            "best_fid_preview_path": self.best_fid_preview_path,
            "best_fid_eval_dir": self.best_fid_eval_dir,
            "best_fid_archive_dir": self.best_fid_archive_dir,
            "best_fid_learning_rate": self.best_fid_learning_rate,
            "quality_monitor_metric": self.config.quality_monitor_metric,
            "consecutive_quality_decline_epochs": self.consecutive_quality_decline_epochs,
            "best_fid_lr_recovery_count": self.best_fid_lr_recovery_count,
            "best_fid_lr_recovery_active": self.best_fid_lr_recovery_active,
            "latest_quality_recovery_triggered": self.latest_quality_recovery_triggered,
            "adversarial_cooldown_until_epoch": self.adversarial_cooldown_until_epoch,
            "collapse_guard_reason": self.collapse_guard_last_reason,
            "wgan_disabled_reason": self.wgan_disabled_reason,
            "wgan_disabled_epoch": self.wgan_disabled_epoch,
            "best_val_loss_checkpoint_path": self.best_val_loss_checkpoint_path,
            "latest_train_loss": train_loss,
            "latest_val_loss": val_loss,
            "epoch": epoch,
            "global_step": self.global_step,
            "total_optimizer_steps": self.total_optimizer_steps,
            "latest_preview_path": self.latest_preview_path,
            "environment_lock_path": str(self.environment_lock_path.resolve()) if self.environment_lock_path.exists() else "",
            "epoch_history_path": str((self.run_dir / "logs" / "epoch_history.json").resolve()),
            "validation_source": self.validation_source,
            "preview_source": self.preview_source,
            "validation_note": self.validation_note,
            "validation_selection_path": str(self.validation_selection_path.resolve()) if self.validation_selection_path.exists() else "",
            "params_m": self.params_m,
            "flops_g": self.flops_g,
            "train_gpu_memory_peak_gb": self.train_gpu_memory_peak_gb,
            "train_gpu_memory_reserved_peak_gb": self.train_gpu_memory_reserved_peak_gb,
            "train_cpu_memory_peak_gb": self.train_cpu_memory_peak_gb,
            "latest_eval_gpu_memory_peak_gb": self.latest_eval_gpu_memory_peak_gb,
            "latest_eval_gpu_memory_reserved_peak_gb": self.latest_eval_gpu_memory_reserved_peak_gb,
            "latest_eval_cpu_memory_peak_gb": self.latest_eval_cpu_memory_peak_gb,
            "latest_eval_archive_dir": self.latest_eval_archive_dir,
            "latest_epoch_time_seconds": self.latest_epoch_time_seconds,
            "latest_epoch_time_hms": self.latest_epoch_time_hms,
            "seed_started_at": self.seed_started_at_iso,
            "seed_elapsed_seconds": self.seed_elapsed_seconds,
            "seed_elapsed_hms": self.seed_elapsed_hms,
            "loss_curve_path": self.loss_curve_path,
            "lr_curve_path": self.lr_curve_path,
            "dashboard_summary_path": self.dashboard_summary_path,
            "latent_step_dashboard_path": self.latent_step_dashboard_path,
            "latent_epoch_dashboard_path": self.latent_epoch_dashboard_path,
            "latest_latent_snapshot_dir": self.latest_latent_snapshot_dir,
            "updated_at": now_iso(),
        }
        payload.update(extra)
        save_json(self.summary_path, payload)

    def train(self) -> Path:
        resume_info = self._discover_resume_checkpoint()
        if resume_info is None:
            self.reset_managed_run_outputs()
        else:
            self._restore_resume_runtime_state(resume_info)
        if self.config.record_environment_lock:
            self.environment_lock_path = ensure_environment_lock(self.config)
        self.write_metadata()
        self.write_status(state="initializing", message="Preparing models and dataloaders")
        split_bundle, train_dataset, val_dataset, train_loader, val_loader = self.build_dataloaders()
        weight_dtype = resolve_dtype(self.effective_mixed_precision)
        manager = ModelManager(self.config)
        components = manager.load_training_components(
            lora_rank=int(self.config.lora_rank),
            lora_alpha=int(self.config.lora_alpha),
            dtype=weight_dtype,
            resume_lora_dir=resume_info["lora_dir"] if resume_info is not None else None,
        )
        self._record_training_component_runtime_state(components)

        base_path = components["base_path"]
        tokenizer = components["tokenizer"]
        tokenizer_2 = components["tokenizer_2"]
        text_encoder = components["text_encoder"]
        text_encoder_2 = components["text_encoder_2"]
        vae = components["vae"]
        controlnet = components["controlnet"]
        unet = components["unet"]
        noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
        adapter = SXDLConditionAdapter(self.adapter_config, self.group.flags)
        critic = PatchImageCritic(base_channels=int(self.config.wgan_critic_channels)) if self.config.enable_wgan_gp else None
        if resume_info is not None:
            self._load_training_modules_from_checkpoint(Path(resume_info["checkpoint_dir"]), adapter, critic=critic)
        self.params_m = count_trainable_params_from_models(unet, adapter)
        self.flops_g = estimate_adapter_flops_g(self.run_dir, width=self.config.image_width, height=self.config.image_height)

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        controlnet.requires_grad_(False)

        optimizer = build_optimizer(
            self.config.optimizer_name,
            list(filter(lambda parameter: parameter.requires_grad, unet.parameters())) + list(adapter.parameters()),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        critic_optimizer = None
        if critic is not None:
            critic_optimizer = build_critic_optimizer(
                critic.parameters(),
                lr=float(self.config.wgan_critic_learning_rate),
                beta1=float(self.config.wgan_critic_beta1),
                beta2=float(self.config.wgan_critic_beta2),
            )

        steps_per_epoch = math.ceil(len(train_loader) / max(self.config.gradient_accumulation_steps, 1))
        self.total_optimizer_steps = int(self.config.epochs * max(steps_per_epoch, 1))
        lr_scheduler = get_scheduler(
            self.config.lr_scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=int(self.config.warmup_steps),
            num_training_steps=self.total_optimizer_steps,
        )
        if resume_info is not None and self.global_step > 0:
            for _ in range(min(self.global_step, self.total_optimizer_steps)):
                lr_scheduler.step()
            self._apply_best_fid_lr_if_needed(optimizer)
            self._enforce_learning_rate_cap(optimizer)

        resume_start_epoch = int(resume_info["epoch"]) + 1 if resume_info is not None else 1
        if resume_start_epoch > int(self.config.epochs):
            self.write_status(
                state="completed",
                message="Training already completed",
                epoch=self.config.epochs,
                global_step=self.global_step,
                total_optimizer_steps=self.total_optimizer_steps,
                seed_started_at=self.seed_started_at_iso,
                seed_elapsed_seconds=self.seed_elapsed_seconds,
                seed_elapsed_hms=self.seed_elapsed_hms,
            )
            self.update_summary(
                status="completed",
                train_loss=None if self.best_train_loss == float("inf") else self.best_train_loss,
                epoch=self.config.epochs,
            )
            sync_run_localized_outputs(self.run_dir)
            return self.run_dir

        vae.to(self.accelerator.device, dtype=weight_dtype)
        text_encoder.to(self.accelerator.device, dtype=weight_dtype)
        text_encoder_2.to(self.accelerator.device, dtype=weight_dtype)
        controlnet.to(self.accelerator.device, dtype=weight_dtype)
        unet.to(self.accelerator.device, dtype=weight_dtype)
        adapter.to(self.accelerator.device, dtype=torch.float32)
        if critic is not None:
            critic.to(self.accelerator.device, dtype=torch.float32)
        for parameter in unet.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

        if self.config.enable_gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        if critic is not None and critic_optimizer is not None:
            unet, adapter, critic, optimizer, critic_optimizer, train_loader, lr_scheduler = self.accelerator.prepare(
                unet,
                adapter,
                critic,
                optimizer,
                critic_optimizer,
                train_loader,
                lr_scheduler,
            )
        else:
            unet, adapter, optimizer, train_loader, lr_scheduler = self.accelerator.prepare(
                unet,
                adapter,
                optimizer,
                train_loader,
                lr_scheduler,
            )
        self._emit_training_startup_report(mode="train")

        progress_bar = tqdm(
            total=self.total_optimizer_steps,
            initial=min(self.global_step, self.total_optimizer_steps),
            disable=not self.accelerator.is_local_main_process,
            desc=f"{self.group.group_id}-seed{self.seed}",
        )
        seed_started_perf = time.perf_counter()
        seed_elapsed_offset_seconds = float(self.seed_elapsed_seconds or 0.0) if resume_info is not None else 0.0
        if not self.seed_started_at_iso:
            self.seed_started_at_iso = now_iso()
        self.write_status(
            state="running",
            message="Training resumed from checkpoint" if resume_info is not None else "Training started",
            epoch=max(0, resume_start_epoch - 1),
            global_step=self.global_step,
            total_optimizer_steps=self.total_optimizer_steps,
            dataset_size=len(train_dataset),
            val_size=len(val_dataset),
            seed_started_at=self.seed_started_at_iso,
            resumed_from_checkpoint=str(resume_info["checkpoint_dir"]) if resume_info is not None else "",
        )

        epoch_memory_monitor: PeakMemoryMonitor | None = None
        stopped_early = False
        stopped_early_reason = ""
        stable_export_checkpoint: Path | None = None
        final_epoch = 0
        try:
            for epoch in range(resume_start_epoch, int(self.config.epochs) + 1):
                _release_torch_memory(self.accelerator.device)
                wgan_memory_guard_triggered, wgan_memory_guard_reason = self._maybe_disable_wgan_for_memory(epoch)
                if wgan_memory_guard_triggered:
                    append_jsonl(
                        self.metric_log_path,
                        {
                            "timestamp": now_iso(),
                            "event": "wgan_memory_guard_triggered",
                            "group_id": self.group.group_id,
                            "seed": self.seed,
                            "epoch": epoch,
                            "reason": wgan_memory_guard_reason,
                        },
                    )
                epoch_started = time.perf_counter()
                epoch_memory_monitor = PeakMemoryMonitor(device=self.accelerator.device).start()
                self.latest_quality_recovery_triggered = False
                unet.train()
                adapter.train()
                epoch_losses: list[float] = []
                epoch_diffusion_losses: list[float] = []
                epoch_adversarial_losses: list[float] = []
                epoch_reconstruction_losses: list[float] = []
                epoch_flat_color_penalties: list[float] = []
                epoch_latent_reconstruction_losses: list[float] = []
                epoch_latent_distribution_losses: list[float] = []
                epoch_latent_local_variance_losses: list[float] = []
                epoch_latent_gradient_losses: list[float] = []
                epoch_latent_global_flatness_losses: list[float] = []
                epoch_image_gradient_losses: list[float] = []
                epoch_adapter_reconstruction_losses: list[float] = []
                epoch_adapter_kl_losses: list[float] = []
                epoch_adapter_raw_kl_losses: list[float] = []
                epoch_adapter_beta_values: list[float] = []
                epoch_kl_per_dim_means: list[float] = []
                epoch_kl_per_dim_maxes: list[float] = []
                epoch_free_bits_active_fractions: list[float] = []
                epoch_z_norm_l2_means: list[float] = []
                epoch_z_std_means: list[float] = []
                epoch_posterior_mu_abs_means: list[float] = []
                epoch_posterior_logvar_means: list[float] = []
                epoch_critic_losses: list[float] = []
                epoch_gradient_penalties: list[float] = []
                collapse_guard_triggered = False
                collapse_guard_reason = ""

                for batch in train_loader:
                    critic_loss_value: float | None = None
                    gradient_penalty_value: float | None = None
                    critic_real_score_value: float | None = None
                    critic_fake_score_value: float | None = None
                    adversarial_loss_value: float | None = None
                    reconstruction_l1_value: float | None = None
                    flat_color_penalty_value: float | None = None
                    latent_reconstruction_value: float | None = None
                    latent_distribution_value: float | None = None
                    latent_local_variance_value: float | None = None
                    latent_gradient_value: float | None = None
                    latent_global_flatness_value: float | None = None
                    image_gradient_value: float | None = None
                    adapter_reconstruction_value: float | None = None
                    adapter_kl_value: float | None = None
                    adapter_raw_kl_value: float | None = None
                    adapter_beta_value: float | None = None
                    kl_per_dim_mean_value: float | None = None
                    kl_per_dim_max_value: float | None = None
                    free_bits_active_fraction_value: float | None = None
                    z_norm_l2_mean_value: float | None = None
                    z_std_mean_value: float | None = None
                    posterior_mu_abs_mean_value: float | None = None
                    posterior_logvar_mean_value: float | None = None
                    diffusion_loss_value: float | None = None
                    effective_adversarial_weight_value = 0.0
                    effective_reconstruction_weight_value = 0.0
                    effective_flat_color_weight_value = 0.0
                    effective_image_gradient_weight_value = 0.0
                    recovery_weights = self._epoch_sanity_recovery_weights(epoch)
                    effective_latent_reconstruction_weight_value = float(getattr(self.config, "latent_reconstruction_weight", 0.0))
                    effective_latent_distribution_weight_value = float(getattr(self.config, "latent_distribution_weight", 0.0))
                    effective_latent_local_variance_weight_value = float(getattr(self.config, "latent_local_variance_weight", 0.0))
                    effective_latent_gradient_weight_value = float(getattr(self.config, "latent_gradient_weight", 0.0))
                    effective_latent_global_flatness_weight_value = float(getattr(self.config, "latent_global_flatness_weight", 0.0))
                    recovery_weight_multiplier_value = float(recovery_weights.get("weight_multiplier", 1.0))
                    effective_latent_reconstruction_weight_value = max(
                        effective_latent_reconstruction_weight_value,
                        float(recovery_weights["latent_reconstruction"]),
                    )
                    effective_latent_distribution_weight_value = max(
                        effective_latent_distribution_weight_value,
                        float(recovery_weights["latent_distribution"]),
                    )
                    effective_latent_local_variance_weight_value = max(
                        effective_latent_local_variance_weight_value,
                        float(recovery_weights["latent_local_variance"]),
                    )
                    effective_latent_gradient_weight_value = max(
                        effective_latent_gradient_weight_value,
                        float(recovery_weights["latent_gradient"]),
                    )
                    effective_latent_global_flatness_weight_value = max(
                        effective_latent_global_flatness_weight_value,
                        float(recovery_weights["latent_global_flatness"]),
                    )
                    effective_reconstruction_weight_value = max(
                        effective_reconstruction_weight_value,
                        float(recovery_weights["image_reconstruction"]),
                    )
                    effective_flat_color_weight_value = max(
                        effective_flat_color_weight_value,
                        float(recovery_weights["image_flat_color"]),
                    )
                    effective_image_gradient_weight_value = max(
                        effective_image_gradient_weight_value,
                        float(recovery_weights["image_gradient"]),
                    )
                    decoded_image_spatial_std_value: float | None = None
                    latent_spatial_std_value: float | None = None
                    generator_step_skipped = False
                    with self.accelerator.accumulate(unet, adapter):
                        color_images = _sanitize_tensor(batch["color"].to(self.accelerator.device, dtype=weight_dtype), nan=0.0, posinf=1.0, neginf=-1.0)
                        lineart_inputs = batch["lineart"].to(self.accelerator.device, dtype=torch.float32)
                        adapter_forward = adapter(lineart_inputs, return_stats=True)
                        if isinstance(adapter_forward, tuple):
                            adapted_lineart_images, adapter_stats = adapter_forward
                        else:
                            adapted_lineart_images, adapter_stats = adapter_forward, {}
                        lineart_images = _sanitize_tensor(adapted_lineart_images, nan=0.0, posinf=1.0, neginf=0.0).to(weight_dtype)

                        with torch.no_grad():
                            latents = vae.encode(color_images).latent_dist.sample()
                            latents = _sanitize_tensor(latents * vae.config.scaling_factor)

                        noise = _sanitize_tensor(torch.randn_like(latents), nan=0.0, posinf=1.0, neginf=-1.0)
                        batch_size = latents.shape[0]
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device).long()
                        noisy_latents = _sanitize_tensor(noise_scheduler.add_noise(latents, noise, timesteps))
                        encoder_hidden_states, added_cond_kwargs = self.encode_prompts(
                            tokenizer,
                            tokenizer_2,
                            text_encoder,
                            text_encoder_2,
                            batch_size=batch_size,
                            device=self.accelerator.device,
                            dtype=weight_dtype,
                        )
                        encoder_hidden_states = _sanitize_tensor(encoder_hidden_states)
                        added_cond_kwargs = {
                            key: _sanitize_tensor(value, nan=0.0, posinf=10.0, neginf=-10.0)
                            for key, value in added_cond_kwargs.items()
                        }
                        with torch.no_grad():
                            down_block_residuals, mid_block_residual = controlnet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=encoder_hidden_states,
                                controlnet_cond=lineart_images,
                                conditioning_scale=float(self.config.controlnet_conditioning_scale),
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )
                            down_block_residuals = tuple(_sanitize_tensor(item) for item in down_block_residuals)
                            mid_block_residual = _sanitize_tensor(mid_block_residual)
                        model_pred = unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            down_block_additional_residuals=down_block_residuals,
                            mid_block_additional_residual=mid_block_residual,
                            added_cond_kwargs=added_cond_kwargs,
                        ).sample
                        model_pred = _sanitize_tensor(model_pred)
                        diffusion_loss = F.mse_loss(model_pred.float(), noise.float())
                        if not torch.isfinite(diffusion_loss):
                            optimizer.zero_grad(set_to_none=True)
                            append_jsonl(
                                self.log_path,
                                {
                                    "timestamp": now_iso(),
                                    "event": "skipped_non_finite_train_loss",
                                    "group_id": self.group.group_id,
                                    "seed": self.seed,
                                    "epoch": epoch,
                                    "global_step": self.global_step,
                                },
                            )
                            continue
                        diffusion_loss_value = float(diffusion_loss.detach().item())
                        total_loss = diffusion_loss
                        adapter_variational_metrics = self._compute_adapter_variational_metrics(
                            adapter_stats,
                            optimizer_step_hint=self.global_step + 1,
                        )
                        if adapter_variational_metrics:
                            adapter_reconstruction_value = float(adapter_variational_metrics["reconstruction_loss"])
                            adapter_kl_value = float(adapter_variational_metrics["kl_loss"])
                            adapter_raw_kl_value = float(adapter_variational_metrics["raw_kl_loss"])
                            adapter_beta_value = float(adapter_variational_metrics["beta"])
                            kl_per_dim_mean_value = float(adapter_variational_metrics["kl_per_dim_mean"])
                            kl_per_dim_max_value = float(adapter_variational_metrics["kl_per_dim_max"])
                            free_bits_active_fraction_value = float(adapter_variational_metrics["free_bits_active_fraction"])
                            z_norm_l2_mean_value = float(adapter_variational_metrics["z_norm_l2_mean"])
                            z_std_mean_value = float(adapter_variational_metrics["z_std_mean"])
                            posterior_mu_abs_mean_value = float(adapter_variational_metrics["posterior_mu_abs_mean"])
                            posterior_logvar_mean_value = float(adapter_variational_metrics["posterior_logvar_mean"])
                            total_loss = total_loss + adapter_variational_metrics["total_loss_tensor"]
                        gan_active = critic is not None and critic_optimizer is not None and self._wgan_is_active(epoch)
                        latent_auxiliary_active = (
                            effective_latent_reconstruction_weight_value > 0.0
                            or effective_latent_distribution_weight_value > 0.0
                            or effective_latent_local_variance_weight_value > 0.0
                            or effective_latent_gradient_weight_value > 0.0
                            or effective_latent_global_flatness_weight_value > 0.0
                        )
                        image_auxiliary_active = (
                            effective_reconstruction_weight_value > 0.0
                            or effective_flat_color_weight_value > 0.0
                            or effective_image_gradient_weight_value > 0.0
                        )
                        needs_predicted_original = gan_active or latent_auxiliary_active or image_auxiliary_active
                        predicted_original_latents: torch.Tensor | None = None
                        decoded_fake_images: torch.Tensor | None = None
                        real_images: torch.Tensor | None = None

                        if needs_predicted_original:
                            predicted_original_latents = _predict_original_latents(
                                noise_scheduler,
                                noisy_latents,
                                model_pred,
                                timesteps,
                            )

                        if predicted_original_latents is not None and latent_auxiliary_active:
                            latent_reconstruction_loss, latent_distribution_loss, latent_spatial_std_value = self._compute_latent_consistency_losses(
                                predicted_original_latents,
                                latents,
                            )
                            latent_reconstruction_value = float(latent_reconstruction_loss.detach().item())
                            latent_distribution_value = float(latent_distribution_loss.detach().item())
                            if effective_latent_reconstruction_weight_value > 0.0:
                                total_loss = total_loss + effective_latent_reconstruction_weight_value * latent_reconstruction_loss
                            if effective_latent_distribution_weight_value > 0.0:
                                total_loss = total_loss + effective_latent_distribution_weight_value * latent_distribution_loss
                            if effective_latent_local_variance_weight_value > 0.0:
                                latent_local_variance_loss = self._compute_latent_local_variance_loss(
                                    predicted_original_latents,
                                    latents,
                                )
                                latent_local_variance_value = float(latent_local_variance_loss.detach().item())
                                total_loss = total_loss + effective_latent_local_variance_weight_value * latent_local_variance_loss
                            if effective_latent_gradient_weight_value > 0.0:
                                latent_gradient_loss = self._compute_latent_gradient_consistency_loss(
                                    predicted_original_latents,
                                    latents,
                                )
                                latent_gradient_value = float(latent_gradient_loss.detach().item())
                                total_loss = total_loss + effective_latent_gradient_weight_value * latent_gradient_loss
                            if effective_latent_global_flatness_weight_value > 0.0:
                                latent_global_flatness_loss = self._compute_latent_global_flatness_loss(
                                    predicted_original_latents,
                                    latents,
                                )
                                latent_global_flatness_value = float(latent_global_flatness_loss.detach().item())
                                total_loss = total_loss + effective_latent_global_flatness_weight_value * latent_global_flatness_loss

                        if predicted_original_latents is not None and (gan_active or image_auxiliary_active):
                            try:
                                decoded_fake_images = _decode_latents_to_images(vae, predicted_original_latents, weight_dtype)
                                real_images = color_images.float()
                            except RuntimeError as exc:
                                if _is_cuda_oom_error(exc) and image_auxiliary_active and not gan_active:
                                    self.recovery_image_aux_disabled_due_to_oom = True
                                    effective_reconstruction_weight_value = 0.0
                                    effective_flat_color_weight_value = 0.0
                                    effective_image_gradient_weight_value = 0.0
                                    decoded_fake_images = None
                                    real_images = None
                                    append_jsonl(
                                        self.metric_log_path,
                                        {
                                            "timestamp": now_iso(),
                                            "event": "recovery_image_aux_disabled_oom",
                                            "group_id": self.group.group_id,
                                            "seed": self.seed,
                                            "epoch": epoch,
                                            "global_step": self.global_step,
                                            "error": str(exc),
                                        },
                                    )
                                    _release_torch_memory(self.accelerator.device)
                                else:
                                    raise

                        if gan_active and critic is not None and critic_optimizer is not None and predicted_original_latents is not None and decoded_fake_images is not None and real_images is not None:
                            if self.accelerator.sync_gradients:
                                critic_input_real = resize_for_critic(real_images.detach(), int(self.config.critic_image_size))
                                critic_input_fake_detached = resize_for_critic(
                                    decoded_fake_images.detach(),
                                    int(self.config.critic_image_size),
                                )
                                _set_requires_grad(critic, True)
                                critic_steps = max(1, int(self.config.wgan_critic_steps))
                                for _ in range(critic_steps):
                                    critic_optimizer.zero_grad(set_to_none=True)
                                    critic_real_scores = critic(critic_input_real)
                                    critic_fake_scores = critic(critic_input_fake_detached)
                                    gradient_penalty = compute_gradient_penalty(
                                        critic,
                                        critic_input_real,
                                        critic_input_fake_detached,
                                    )
                                    critic_loss = (
                                        critic_fake_scores.mean()
                                        - critic_real_scores.mean()
                                        + float(self.config.wgan_gp_weight) * gradient_penalty
                                    )
                                    if torch.isfinite(critic_loss):
                                        self.accelerator.backward(critic_loss)
                                        critic_optimizer.step()
                                        critic_loss_value = float(critic_loss.detach().item())
                                        gradient_penalty_value = float(gradient_penalty.detach().item())
                                        critic_real_score_value = float(critic_real_scores.mean().detach().item())
                                        critic_fake_score_value = float(critic_fake_scores.mean().detach().item())
                                    critic_optimizer.zero_grad(set_to_none=True)

                            _set_requires_grad(critic, False)
                            decoded_image_spatial_std_value = self._decoded_image_spatial_std(decoded_fake_images.float())
                            (
                                effective_adversarial_weight_value,
                                gan_reconstruction_weight_value,
                                gan_flat_color_weight_value,
                            ) = self._compute_effective_adversarial_weights(
                                epoch=epoch,
                                critic_real_score=critic_real_score_value,
                                critic_fake_score=critic_fake_score_value,
                                decoded_image_spatial_std=decoded_image_spatial_std_value,
                            )
                            effective_reconstruction_weight_value = max(
                                float(effective_reconstruction_weight_value),
                                float(gan_reconstruction_weight_value),
                            )
                            effective_flat_color_weight_value = max(
                                float(effective_flat_color_weight_value),
                                float(gan_flat_color_weight_value),
                            )
                            critic_input_fake = resize_for_critic(decoded_fake_images, int(self.config.critic_image_size))
                            adversarial_loss = -critic(critic_input_fake).mean()
                            adversarial_loss_value = float(adversarial_loss.detach().item())
                            if effective_adversarial_weight_value > 0.0:
                                total_loss = total_loss + float(effective_adversarial_weight_value) * adversarial_loss
                            _set_requires_grad(critic, True)

                        if decoded_fake_images is not None and real_images is not None:
                            decoded_fake_images_float = decoded_fake_images.float()
                            if decoded_image_spatial_std_value is None:
                                decoded_image_spatial_std_value = self._decoded_image_spatial_std(decoded_fake_images_float)
                            if effective_reconstruction_weight_value > 0.0:
                                reconstruction_l1_loss = F.l1_loss(decoded_fake_images_float, real_images)
                                reconstruction_l1_value = float(reconstruction_l1_loss.detach().item())
                                total_loss = total_loss + float(effective_reconstruction_weight_value) * reconstruction_l1_loss
                            if effective_flat_color_weight_value > 0.0:
                                flat_color_penalty = compute_flat_color_penalty(
                                    decoded_fake_images_float,
                                    float(self.config.anti_flat_color_min_std),
                                )
                                flat_color_penalty_value = float(flat_color_penalty.detach().item())
                                total_loss = total_loss + float(effective_flat_color_weight_value) * flat_color_penalty
                            if effective_image_gradient_weight_value > 0.0:
                                image_gradient_loss = self._compute_image_gradient_consistency_loss(
                                    decoded_fake_images_float,
                                    real_images,
                                )
                                image_gradient_value = float(image_gradient_loss.detach().item())
                                total_loss = total_loss + float(effective_image_gradient_weight_value) * image_gradient_loss

                        if not torch.isfinite(total_loss):
                            optimizer.zero_grad(set_to_none=True)
                            if critic_optimizer is not None:
                                critic_optimizer.zero_grad(set_to_none=True)
                            append_jsonl(
                                self.log_path,
                                {
                                    "timestamp": now_iso(),
                                    "event": "skipped_non_finite_train_loss",
                                    "group_id": self.group.group_id,
                                    "seed": self.seed,
                                    "epoch": epoch,
                                    "global_step": self.global_step,
                                    "reason": "non_finite_total_loss",
                                },
                            )
                            continue
                        self.accelerator.backward(total_loss)
                        if self.accelerator.sync_gradients and self.config.max_grad_norm > 0:
                            self.accelerator.clip_grad_norm_(list(unet.parameters()) + list(adapter.parameters()), self.config.max_grad_norm)
                        optimizer.step()
                        generator_step_skipped = bool(getattr(optimizer, "step_was_skipped", False))
                        if not generator_step_skipped:
                            lr_scheduler.step()
                        self._apply_best_fid_lr_if_needed(optimizer)
                        self._enforce_learning_rate_cap(optimizer)
                        optimizer.zero_grad(set_to_none=True)

                    if self.accelerator.sync_gradients and generator_step_skipped:
                        recovery_image_aux_active = (
                            self._epoch_sanity_recovery_active(epoch)
                            and (
                                effective_reconstruction_weight_value > 0.0
                                or effective_flat_color_weight_value > 0.0
                                or effective_image_gradient_weight_value > 0.0
                            )
                        )
                        if recovery_image_aux_active:
                            self.recovery_skipped_step_streak += 1
                            if self.recovery_skipped_step_streak >= 1 and not self.recovery_image_aux_disabled_due_to_skips:
                                self.recovery_image_aux_disabled_due_to_skips = True
                                self.recovery_aux_disabled_due_to_skips = True
                                optimizer.zero_grad(set_to_none=True)
                                if critic_optimizer is not None:
                                    critic_optimizer.zero_grad(set_to_none=True)
                                _release_torch_memory(self.accelerator.device)
                                append_jsonl(
                                    self.metric_log_path,
                                    {
                                        "timestamp": now_iso(),
                                        "event": "recovery_image_aux_disabled_skipped_steps",
                                        "group_id": self.group.group_id,
                                        "seed": self.seed,
                                        "epoch": epoch,
                                        "global_step": self.global_step,
                                        "skip_streak": int(self.recovery_skipped_step_streak),
                                        "learning_rate": self._current_learning_rate(optimizer, lr_scheduler),
                                        "recovery_weight_multiplier": float(recovery_weight_multiplier_value),
                                        "image_reconstruction_weight": float(effective_reconstruction_weight_value),
                                        "image_flat_color_weight": float(effective_flat_color_weight_value),
                                        "image_gradient_weight": float(effective_image_gradient_weight_value),
                                    },
                                )
                                append_jsonl(
                                    self.metric_log_path,
                                    {
                                        "timestamp": now_iso(),
                                        "event": "epoch_sanity_recovery_disabled_skipped_steps",
                                        "group_id": self.group.group_id,
                                        "seed": self.seed,
                                        "epoch": epoch,
                                        "global_step": self.global_step,
                                        "skip_streak": int(self.recovery_skipped_step_streak),
                                        "learning_rate": self._current_learning_rate(optimizer, lr_scheduler),
                                    },
                                )
                        else:
                            self.recovery_skipped_step_streak = 0
                        current_lr = self._current_learning_rate(optimizer, lr_scheduler)
                        append_jsonl(
                            self.log_path,
                            {
                                "timestamp": now_iso(),
                                "event": "skipped_optimizer_step",
                                "group_id": self.group.group_id,
                                "seed": self.seed,
                                "epoch": epoch,
                                "global_step": self.global_step,
                                "loss": float(total_loss.detach().item()),
                                "lr": current_lr,
                            },
                        )
                        progress_bar.set_postfix(
                            loss=f"{float(total_loss.detach().item()):.4f}",
                            lr=f"{current_lr:.2e}",
                            skipped="yes",
                        )
                        continue

                    if self.accelerator.sync_gradients:
                        self.recovery_skipped_step_streak = 0
                        self.global_step += 1
                        loss_value = float(total_loss.detach().item())
                        epoch_losses.append(loss_value)
                        if diffusion_loss_value is not None:
                            epoch_diffusion_losses.append(diffusion_loss_value)
                        if adversarial_loss_value is not None:
                            epoch_adversarial_losses.append(adversarial_loss_value)
                        if reconstruction_l1_value is not None:
                            epoch_reconstruction_losses.append(reconstruction_l1_value)
                        if flat_color_penalty_value is not None:
                            epoch_flat_color_penalties.append(flat_color_penalty_value)
                        if latent_reconstruction_value is not None:
                            epoch_latent_reconstruction_losses.append(latent_reconstruction_value)
                        if latent_distribution_value is not None:
                            epoch_latent_distribution_losses.append(latent_distribution_value)
                        if latent_local_variance_value is not None:
                            epoch_latent_local_variance_losses.append(latent_local_variance_value)
                        if latent_gradient_value is not None:
                            epoch_latent_gradient_losses.append(latent_gradient_value)
                        if latent_global_flatness_value is not None:
                            epoch_latent_global_flatness_losses.append(latent_global_flatness_value)
                        if image_gradient_value is not None:
                            epoch_image_gradient_losses.append(image_gradient_value)
                        if adapter_reconstruction_value is not None:
                            epoch_adapter_reconstruction_losses.append(adapter_reconstruction_value)
                        if adapter_kl_value is not None:
                            epoch_adapter_kl_losses.append(adapter_kl_value)
                        if adapter_raw_kl_value is not None:
                            epoch_adapter_raw_kl_losses.append(adapter_raw_kl_value)
                        if adapter_beta_value is not None:
                            epoch_adapter_beta_values.append(adapter_beta_value)
                        if kl_per_dim_mean_value is not None:
                            epoch_kl_per_dim_means.append(kl_per_dim_mean_value)
                        if kl_per_dim_max_value is not None:
                            epoch_kl_per_dim_maxes.append(kl_per_dim_max_value)
                        if free_bits_active_fraction_value is not None:
                            epoch_free_bits_active_fractions.append(free_bits_active_fraction_value)
                        if z_norm_l2_mean_value is not None:
                            epoch_z_norm_l2_means.append(z_norm_l2_mean_value)
                        if z_std_mean_value is not None:
                            epoch_z_std_means.append(z_std_mean_value)
                        if posterior_mu_abs_mean_value is not None:
                            epoch_posterior_mu_abs_means.append(posterior_mu_abs_mean_value)
                        if posterior_logvar_mean_value is not None:
                            epoch_posterior_logvar_means.append(posterior_logvar_mean_value)
                        if critic_loss_value is not None:
                            epoch_critic_losses.append(critic_loss_value)
                        if gradient_penalty_value is not None:
                            epoch_gradient_penalties.append(gradient_penalty_value)
                        self.best_train_loss = min(self.best_train_loss, loss_value)
                        progress_bar.update(1)
                        payload = {
                            "timestamp": now_iso(),
                            "event": "train",
                            "group_id": self.group.group_id,
                            "seed": self.seed,
                            "epoch": epoch,
                            "global_step": self.global_step,
                            "loss": loss_value,
                            "lr": self._current_learning_rate(optimizer, lr_scheduler),
                            "adapter_bottleneck_reconstruction_loss": adapter_reconstruction_value,
                            "beta_vae_kl_loss": adapter_kl_value,
                            "beta_vae_kl_raw_loss": adapter_raw_kl_value,
                            "beta_vae_beta": adapter_beta_value,
                            "kl_per_dim_mean": kl_per_dim_mean_value,
                            "kl_per_dim_max": kl_per_dim_max_value,
                            "free_bits_active_fraction": free_bits_active_fraction_value,
                            "z_norm_l2_mean": z_norm_l2_mean_value,
                            "z_std_mean": z_std_mean_value,
                            "posterior_mu_abs_mean": posterior_mu_abs_mean_value,
                            "posterior_logvar_mean": posterior_logvar_mean_value,
                        }
                        append_jsonl(self.log_path, payload)
                        postfix = {
                            "loss": f"{loss_value:.4f}",
                            "lr": f"{self._current_learning_rate(optimizer, lr_scheduler):.2e}",
                        }
                        progress_bar.set_postfix(**postfix)

                        if self.global_step % max(1, int(self.config.checkpoint_interval_steps)) == 0:
                            self.save_checkpoint(unet, adapter, step=self.global_step, epoch=epoch, critic=critic)

                epoch_train_loss = float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else None
                epoch_diffusion_loss = (
                    float(sum(epoch_diffusion_losses) / len(epoch_diffusion_losses)) if epoch_diffusion_losses else None
                )
                epoch_adversarial_loss = (
                    float(sum(epoch_adversarial_losses) / len(epoch_adversarial_losses)) if epoch_adversarial_losses else None
                )
                epoch_reconstruction_loss = (
                    float(sum(epoch_reconstruction_losses) / len(epoch_reconstruction_losses))
                    if epoch_reconstruction_losses
                    else None
                )
                epoch_flat_color_penalty = (
                    float(sum(epoch_flat_color_penalties) / len(epoch_flat_color_penalties))
                    if epoch_flat_color_penalties
                    else None
                )
                epoch_latent_reconstruction_loss = (
                    float(sum(epoch_latent_reconstruction_losses) / len(epoch_latent_reconstruction_losses))
                    if epoch_latent_reconstruction_losses
                    else None
                )
                epoch_latent_distribution_loss = (
                    float(sum(epoch_latent_distribution_losses) / len(epoch_latent_distribution_losses))
                    if epoch_latent_distribution_losses
                    else None
                )
                epoch_latent_local_variance_loss = (
                    float(sum(epoch_latent_local_variance_losses) / len(epoch_latent_local_variance_losses))
                    if epoch_latent_local_variance_losses
                    else None
                )
                epoch_latent_gradient_loss = (
                    float(sum(epoch_latent_gradient_losses) / len(epoch_latent_gradient_losses))
                    if epoch_latent_gradient_losses
                    else None
                )
                epoch_latent_global_flatness_loss = (
                    float(sum(epoch_latent_global_flatness_losses) / len(epoch_latent_global_flatness_losses))
                    if epoch_latent_global_flatness_losses
                    else None
                )
                epoch_image_gradient_loss = (
                    float(sum(epoch_image_gradient_losses) / len(epoch_image_gradient_losses))
                    if epoch_image_gradient_losses
                    else None
                )
                epoch_adapter_reconstruction_loss = (
                    float(sum(epoch_adapter_reconstruction_losses) / len(epoch_adapter_reconstruction_losses))
                    if epoch_adapter_reconstruction_losses
                    else None
                )
                epoch_adapter_kl_loss = (
                    float(sum(epoch_adapter_kl_losses) / len(epoch_adapter_kl_losses))
                    if epoch_adapter_kl_losses
                    else None
                )
                epoch_adapter_raw_kl_loss = (
                    float(sum(epoch_adapter_raw_kl_losses) / len(epoch_adapter_raw_kl_losses))
                    if epoch_adapter_raw_kl_losses
                    else None
                )
                epoch_adapter_beta = (
                    float(sum(epoch_adapter_beta_values) / len(epoch_adapter_beta_values))
                    if epoch_adapter_beta_values
                    else None
                )
                epoch_kl_per_dim_mean = (
                    float(sum(epoch_kl_per_dim_means) / len(epoch_kl_per_dim_means))
                    if epoch_kl_per_dim_means
                    else None
                )
                epoch_kl_per_dim_max = (
                    float(sum(epoch_kl_per_dim_maxes) / len(epoch_kl_per_dim_maxes))
                    if epoch_kl_per_dim_maxes
                    else None
                )
                epoch_free_bits_active_fraction = (
                    float(sum(epoch_free_bits_active_fractions) / len(epoch_free_bits_active_fractions))
                    if epoch_free_bits_active_fractions
                    else None
                )
                epoch_z_norm_l2_mean = (
                    float(sum(epoch_z_norm_l2_means) / len(epoch_z_norm_l2_means))
                    if epoch_z_norm_l2_means
                    else None
                )
                epoch_z_std_mean = (
                    float(sum(epoch_z_std_means) / len(epoch_z_std_means))
                    if epoch_z_std_means
                    else None
                )
                epoch_posterior_mu_abs_mean = (
                    float(sum(epoch_posterior_mu_abs_means) / len(epoch_posterior_mu_abs_means))
                    if epoch_posterior_mu_abs_means
                    else None
                )
                epoch_posterior_logvar_mean = (
                    float(sum(epoch_posterior_logvar_means) / len(epoch_posterior_logvar_means))
                    if epoch_posterior_logvar_means
                    else None
                )
                epoch_critic_loss = (
                    float(sum(epoch_critic_losses) / len(epoch_critic_losses)) if epoch_critic_losses else None
                )
                epoch_gradient_penalty = (
                    float(sum(epoch_gradient_penalties) / len(epoch_gradient_penalties))
                    if epoch_gradient_penalties
                    else None
                )
                epoch_time = time.perf_counter() - epoch_started
                seed_elapsed_seconds = seed_elapsed_offset_seconds + (time.perf_counter() - seed_started_perf)
                epoch_memory = epoch_memory_monitor.stop()
                epoch_memory_monitor = None
                gpu_peak = epoch_memory.get("gpu_memory_peak_gb")
                gpu_reserved_peak = epoch_memory.get("gpu_memory_reserved_peak_gb")
                cpu_peak = epoch_memory.get("cpu_memory_peak_gb")
                self.train_gpu_memory_peak_gb = _max_optional(self.train_gpu_memory_peak_gb, gpu_peak)
                self.train_gpu_memory_reserved_peak_gb = _max_optional(self.train_gpu_memory_reserved_peak_gb, gpu_reserved_peak)
                self.train_cpu_memory_peak_gb = _max_optional(self.train_cpu_memory_peak_gb, cpu_peak)
                self.latest_epoch_time_seconds = round(epoch_time, 4)
                self.latest_epoch_time_hms = _format_duration(epoch_time)
                self.seed_elapsed_seconds = round(seed_elapsed_seconds, 4)
                self.seed_elapsed_hms = _format_duration(seed_elapsed_seconds)
                sanity_check_payload = self._run_epoch_sanity_check(
                    epoch=epoch,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    vae=vae,
                    controlnet=controlnet,
                    unet=unet,
                    adapter=adapter,
                    base_path=base_path,
                    dtype=weight_dtype,
                )
                sanity_warning_triggered = False
                sanity_warning_reason = ""
                if sanity_check_payload:
                    sanity_warning_triggered, sanity_warning_reason = self._detect_epoch_sanity_warning(sanity_check_payload)
                    if sanity_warning_triggered and not sanity_check_payload.get("collapsed"):
                        self._apply_epoch_sanity_recovery(
                            epoch=epoch,
                            reason=sanity_warning_reason,
                            sanity_check_payload=sanity_check_payload,
                            optimizer=optimizer,
                        )
                if sanity_check_payload.get("collapsed"):
                    collapse_guard_triggered = True
                    collapse_guard_reason = str(sanity_check_payload.get("reason", "")).strip()
                    self.collapse_guard_last_reason = collapse_guard_reason
                latent_snapshot_payload = self._save_epoch_latent_artifacts(epoch=epoch, adapter=adapter)
                metric_payload = {
                    "timestamp": now_iso(),
                    "event": "epoch_end",
                    "group_id": self.group.group_id,
                    "seed": self.seed,
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "diffusion_loss": epoch_diffusion_loss,
                    "adversarial_loss": epoch_adversarial_loss,
                    "reconstruction_l1_loss": epoch_reconstruction_loss,
                    "flat_color_penalty": epoch_flat_color_penalty,
                    "latent_reconstruction_loss": epoch_latent_reconstruction_loss,
                    "latent_distribution_loss": epoch_latent_distribution_loss,
                    "latent_local_variance_loss": epoch_latent_local_variance_loss,
                    "latent_gradient_loss": epoch_latent_gradient_loss,
                    "latent_global_flatness_loss": epoch_latent_global_flatness_loss,
                    "image_gradient_loss": epoch_image_gradient_loss,
                    "adapter_bottleneck_reconstruction_loss": epoch_adapter_reconstruction_loss,
                    "beta_vae_kl_loss": epoch_adapter_kl_loss,
                    "beta_vae_kl_raw_loss": epoch_adapter_raw_kl_loss,
                    "beta_vae_beta": epoch_adapter_beta,
                    "kl_per_dim_mean": epoch_kl_per_dim_mean,
                    "kl_per_dim_max": epoch_kl_per_dim_max,
                    "free_bits_active_fraction": epoch_free_bits_active_fraction,
                    "z_norm_l2_mean": epoch_z_norm_l2_mean,
                    "z_std_mean": epoch_z_std_mean,
                    "posterior_mu_abs_mean": epoch_posterior_mu_abs_mean,
                    "posterior_logvar_mean": epoch_posterior_logvar_mean,
                    "critic_loss": epoch_critic_loss,
                    "gradient_penalty": epoch_gradient_penalty,
                    "learning_rate": self._current_learning_rate(optimizer, lr_scheduler),
                    "adversarial_cooldown_until_epoch": self.adversarial_cooldown_until_epoch,
                    "collapse_guard_triggered": collapse_guard_triggered,
                    "collapse_guard_reason": collapse_guard_reason,
                    "wgan_disabled_reason": self.wgan_disabled_reason,
                    "wgan_disabled_epoch": self.wgan_disabled_epoch,
                    "gpu_memory_peak_gb": gpu_peak,
                    "gpu_memory_reserved_peak_gb": gpu_reserved_peak,
                    "cpu_memory_peak_gb": cpu_peak,
                    "epoch_time_seconds": round(epoch_time, 4),
                    "epoch_time_hms": _format_duration(epoch_time),
                    "seed_elapsed_seconds": round(seed_elapsed_seconds, 4),
                    "seed_elapsed_hms": _format_duration(seed_elapsed_seconds),
                    "epoch_sanity_check": sanity_check_payload,
                    "epoch_sanity_warning_triggered": sanity_warning_triggered,
                    "epoch_sanity_warning_reason": sanity_warning_reason,
                    **latent_snapshot_payload,
                }
                append_jsonl(self.metric_log_path, metric_payload)
                self.epoch_history.append(metric_payload)
                save_json(self.run_dir / "logs" / "epoch_history.json", {"epochs": self.epoch_history})
                curve_paths = export_training_curves(self.run_dir)
                self.loss_curve_path = curve_paths.get("loss_curve_path", self.loss_curve_path)
                self.lr_curve_path = curve_paths.get("lr_curve_path", self.lr_curve_path)
                self.dashboard_summary_path = curve_paths.get("dashboard_summary_path", self.dashboard_summary_path)
                self.latent_step_dashboard_path = curve_paths.get("latent_step_dashboard_path", self.latent_step_dashboard_path)
                self.latent_epoch_dashboard_path = curve_paths.get("latent_epoch_dashboard_path", self.latent_epoch_dashboard_path)
                self.write_status(
                    state="running",
                    message="Training in progress",
                    epoch=epoch,
                    global_step=self.global_step,
                    total_optimizer_steps=self.total_optimizer_steps,
                    train_loss=epoch_train_loss,
                    learning_rate=self._current_learning_rate(optimizer, lr_scheduler),
                    adversarial_cooldown_until_epoch=self.adversarial_cooldown_until_epoch,
                    collapse_guard_triggered=collapse_guard_triggered,
                    collapse_guard_reason=collapse_guard_reason,
                    gpu_memory_peak_gb=gpu_peak,
                    epoch_time_seconds=round(epoch_time, 4),
                    epoch_time_hms=_format_duration(epoch_time),
                    seed_elapsed_seconds=round(seed_elapsed_seconds, 4),
                    seed_elapsed_hms=_format_duration(seed_elapsed_seconds),
                    seed_started_at=self.seed_started_at_iso,
                    latest_latent_snapshot_dir=self.latest_latent_snapshot_dir,
                    dashboard_summary_path=self.dashboard_summary_path,
                )
                self.update_summary(
                    status="running",
                    train_loss=epoch_train_loss,
                    epoch=epoch,
                    collapse_guard_triggered=collapse_guard_triggered,
                    collapse_guard_reason=collapse_guard_reason,
                    latest_latent_snapshot_dir=self.latest_latent_snapshot_dir,
                    dashboard_summary_path=self.dashboard_summary_path,
                    latent_step_dashboard_path=self.latent_step_dashboard_path,
                    latent_epoch_dashboard_path=self.latent_epoch_dashboard_path,
                )
                sync_run_localized_outputs(self.run_dir)
                _release_torch_memory(self.accelerator.device)

                if self.config.save_every_epoch and not collapse_guard_triggered:
                    self.save_checkpoint(unet, adapter, step=self.global_step, epoch=epoch, critic=critic)
                final_epoch = epoch
                if collapse_guard_triggered and bool(getattr(self.config, "stop_training_on_epoch_sanity_failure", True)):
                    stopped_early = True
                    stopped_early_reason = collapse_guard_reason or "epoch_sanity_check_failed"
                    final_epoch = max(0, int(epoch) - 1)
                    if final_epoch > 0:
                        stable_export_checkpoint = self._latest_checkpoint_dir_for_epoch(final_epoch)
                        if stable_export_checkpoint is not None:
                            self.save_named_checkpoint_from_existing(stable_export_checkpoint, "best_stable", final_epoch)
                    break

            if stable_export_checkpoint is not None:
                self._export_final_artifacts_from_checkpoint(stable_export_checkpoint)
            else:
                self.save_final_artifacts(unet, adapter, critic=critic)
            generation_metrics: dict[str, Any] = {}
            posthoc_summary: dict[str, Any] = {}
            if bool(getattr(self.config, "defer_generation_metrics_until_seed_end", False)):
                if self.accelerator.is_local_main_process and self.metric_eval_records:
                    for epoch_record in self.epoch_history:
                        epoch = int(epoch_record.get("epoch", 0) or 0)
                        if epoch <= 0:
                            continue
                        eval_dir = self.run_dir / "evaluations" / "validation_epochs" / f"epoch_{epoch:03d}"
                        if not eval_dir.exists() or not list((eval_dir / "generated").glob("*.png")):
                            _release_torch_memory(self.accelerator.device)
                            self.generate_validation_images_for_epoch(
                                epoch=epoch,
                                text_encoder=text_encoder,
                                text_encoder_2=text_encoder_2,
                                tokenizer=tokenizer,
                                tokenizer_2=tokenizer_2,
                                vae=vae,
                                controlnet=controlnet,
                                unet=unet,
                                adapter=adapter,
                                base_path=base_path,
                                weight_dtype=weight_dtype,
                            )
                            _release_torch_memory(self.accelerator.device)
                posthoc_summary = self._finalize_deferred_generation_metrics(
                    vae=vae,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    controlnet=controlnet,
                    unet=unet,
                    adapter=adapter,
                    critic=critic,
                )
                latest_metrics = posthoc_summary.get("latest_metrics", {}) if isinstance(posthoc_summary, dict) else {}
                if isinstance(latest_metrics, dict) and latest_metrics:
                    generation_metrics = latest_metrics
                if bool(posthoc_summary.get("successful_epochs", 0)):
                    self.write_status(
                        state="running",
                        message="Deferred generation metrics completed",
                        epoch=self.config.epochs,
                        global_step=self.global_step,
                        total_optimizer_steps=self.total_optimizer_steps,
                        posthoc_metrics_completed=True,
                        posthoc_successful_epochs=posthoc_summary.get("successful_epochs", 0),
                    )

            curve_paths = export_training_curves(self.run_dir)
            self.loss_curve_path = curve_paths.get("loss_curve_path", self.loss_curve_path)
            self.lr_curve_path = curve_paths.get("lr_curve_path", self.lr_curve_path)
            final_seed_elapsed_seconds = round(seed_elapsed_offset_seconds + (time.perf_counter() - seed_started_perf), 4)
            self.seed_elapsed_seconds = final_seed_elapsed_seconds
            self.seed_elapsed_hms = _format_duration(final_seed_elapsed_seconds)
            completion_message = "Training completed"
            if stopped_early:
                completion_message = (
                    f"Training stopped early by epoch sanity check and preserved the last stable checkpoint. "
                    f"Reason: {stopped_early_reason}"
                )
            self.write_status(
                state="completed",
                message=completion_message,
                epoch=final_epoch if stopped_early else self.config.epochs,
                global_step=self.global_step,
                total_optimizer_steps=self.total_optimizer_steps,
                best_train_loss=self.best_train_loss,
                best_val_loss=None if self.best_val_loss == float("inf") else self.best_val_loss,
                params_m=self.params_m,
                flops_g=self.flops_g,
                seed_started_at=self.seed_started_at_iso,
                seed_elapsed_seconds=self.seed_elapsed_seconds,
                seed_elapsed_hms=self.seed_elapsed_hms,
                latest_epoch_time_seconds=self.latest_epoch_time_seconds,
                latest_epoch_time_hms=self.latest_epoch_time_hms,
                train_gpu_memory_peak_gb=self.train_gpu_memory_peak_gb,
                train_gpu_memory_reserved_peak_gb=self.train_gpu_memory_reserved_peak_gb,
                train_cpu_memory_peak_gb=self.train_cpu_memory_peak_gb,
                latest_eval_gpu_memory_peak_gb=self.latest_eval_gpu_memory_peak_gb,
                latest_eval_gpu_memory_reserved_peak_gb=self.latest_eval_gpu_memory_reserved_peak_gb,
                latest_eval_cpu_memory_peak_gb=self.latest_eval_cpu_memory_peak_gb,
                latest_eval_archive_dir=self.latest_eval_archive_dir,
                loss_curve_path=self.loss_curve_path,
                lr_curve_path=self.lr_curve_path,
                stopped_early_reason=stopped_early_reason if stopped_early else "",
                stable_checkpoint_path=str(stable_export_checkpoint.resolve()) if stable_export_checkpoint is not None else "",
            )
            self.update_summary(
                status="completed",
                train_loss=None if self.best_train_loss == float("inf") else self.best_train_loss,
                epoch=final_epoch if stopped_early else self.config.epochs,
                stopped_early_reason=stopped_early_reason if stopped_early else "",
                stable_checkpoint_path=str(stable_export_checkpoint.resolve()) if stable_export_checkpoint is not None else "",
            )
            sync_run_localized_outputs(self.run_dir)
            return self.run_dir
        except Exception as exc:
            if _is_cuda_oom_error(exc) and (self.epoch_history or self.best_fid != float("inf")):
                completed_epoch = int(self.epoch_history[-1].get("epoch", 0)) if self.epoch_history else max(0, int(self.best_fid_epoch))
                final_seed_elapsed_seconds = round(seed_elapsed_offset_seconds + (time.perf_counter() - seed_started_perf), 4)
                self.seed_elapsed_seconds = final_seed_elapsed_seconds
                self.seed_elapsed_hms = _format_duration(final_seed_elapsed_seconds)
                message = (
                    "Training stopped early after CUDA OOM; preserved the latest completed epoch and best checkpoint. "
                    f"Best FID epoch: {self.best_fid_epoch or completed_epoch}. Error: {exc}"
                )
                self.write_status(
                    state="completed",
                    message=message,
                    epoch=completed_epoch,
                    global_step=self.global_step,
                    total_optimizer_steps=self.total_optimizer_steps,
                    best_train_loss=None if self.best_train_loss == float("inf") else self.best_train_loss,
                    best_val_loss=None if self.best_val_loss == float("inf") else self.best_val_loss,
                    params_m=self.params_m,
                    flops_g=self.flops_g,
                    seed_started_at=self.seed_started_at_iso,
                    seed_elapsed_seconds=self.seed_elapsed_seconds,
                    seed_elapsed_hms=self.seed_elapsed_hms,
                    latest_epoch_time_seconds=self.latest_epoch_time_seconds,
                    latest_epoch_time_hms=self.latest_epoch_time_hms,
                    train_gpu_memory_peak_gb=self.train_gpu_memory_peak_gb,
                    train_gpu_memory_reserved_peak_gb=self.train_gpu_memory_reserved_peak_gb,
                    train_cpu_memory_peak_gb=self.train_cpu_memory_peak_gb,
                    latest_eval_gpu_memory_peak_gb=self.latest_eval_gpu_memory_peak_gb,
                    latest_eval_gpu_memory_reserved_peak_gb=self.latest_eval_gpu_memory_reserved_peak_gb,
                    latest_eval_cpu_memory_peak_gb=self.latest_eval_cpu_memory_peak_gb,
                    latest_eval_archive_dir=self.latest_eval_archive_dir,
                    loss_curve_path=self.loss_curve_path,
                    lr_curve_path=self.lr_curve_path,
                    stopped_early_reason=str(exc),
                )
                self.update_summary(
                    status="completed",
                    train_loss=None if self.best_train_loss == float("inf") else self.best_train_loss,
                    val_loss=None if self.best_val_loss == float("inf") else self.best_val_loss,
                    epoch=completed_epoch,
                    stopped_early_reason=str(exc),
                )
                sync_run_localized_outputs(self.run_dir)
                return self.run_dir
            failed_seed_elapsed_seconds = round(seed_elapsed_offset_seconds + (time.perf_counter() - seed_started_perf), 4)
            self.seed_elapsed_seconds = failed_seed_elapsed_seconds
            self.seed_elapsed_hms = _format_duration(failed_seed_elapsed_seconds)
            self.write_status(
                state="failed",
                message=str(exc),
                epoch=0,
                global_step=self.global_step,
                total_optimizer_steps=self.total_optimizer_steps,
                seed_started_at=self.seed_started_at_iso,
                seed_elapsed_seconds=self.seed_elapsed_seconds,
                seed_elapsed_hms=self.seed_elapsed_hms,
            )
            self.update_summary(status="failed", train_loss=None, val_loss=None, epoch=0)
            sync_run_localized_outputs(self.run_dir)
            raise
        finally:
            if epoch_memory_monitor is not None:
                epoch_memory_monitor.stop()
            progress_bar.close()
            _release_torch_memory(self.accelerator.device)

    def train_without_preview(self) -> Path:
        """Compatibility wrapper for pure training without any deferred generation metrics."""
        original_defer = bool(getattr(self.config, "defer_generation_metrics_until_seed_end", False))
        self.config.defer_generation_metrics_until_seed_end = False
        try:
            return self.train()
        finally:
            self.config.defer_generation_metrics_until_seed_end = original_defer

    def train_with_preview(self, preview_epochs: list[int] | None = None) -> Path:
        """
        Part 1: 训练 + 定期推理出图（不计算指标）

        Args:
            preview_epochs: 需要推理出图的epoch列表，如 [2, 4, 6, 8, 9, 10, 11, 12]
                           前8个epoch每2个epoch推理，后4个epoch每个都推理
        """
        if preview_epochs is None:
            preview_epochs = [2, 4, 6, 8, 9, 10, 11, 12]

        preview_epochs_set = set(preview_epochs)
        print(f"[train_with_preview] 训练开始，推理epoch: {sorted(preview_epochs_set)}")

        self.reset_managed_run_outputs()
        if self.config.record_environment_lock:
            self.environment_lock_path = ensure_environment_lock(self.config)
        self.write_metadata()
        self.write_status(state="initializing", message="Preparing models and dataloaders")
        split_bundle, train_dataset, val_dataset, train_loader, val_loader = self.build_dataloaders()
        weight_dtype = resolve_dtype(self.effective_mixed_precision)
        manager = ModelManager(self.config)
        components = manager.load_training_components(
            lora_rank=int(self.config.lora_rank),
            lora_alpha=int(self.config.lora_alpha),
            dtype=weight_dtype,
        )
        self._record_training_component_runtime_state(components)

        base_path = components["base_path"]
        tokenizer = components["tokenizer"]
        tokenizer_2 = components["tokenizer_2"]
        text_encoder = components["text_encoder"]
        text_encoder_2 = components["text_encoder_2"]
        vae = components["vae"]
        controlnet = components["controlnet"]
        unet = components["unet"]
        noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
        adapter = SXDLConditionAdapter(self.adapter_config, self.group.flags)
        critic = PatchImageCritic(base_channels=int(self.config.wgan_critic_channels)) if self.config.enable_wgan_gp else None
        self.params_m = count_trainable_params_from_models(unet, adapter)
        self.flops_g = estimate_adapter_flops_g(self.run_dir, width=self.config.image_width, height=self.config.image_height)

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        controlnet.requires_grad_(False)

        optimizer = build_optimizer(
            self.config.optimizer_name,
            list(filter(lambda parameter: parameter.requires_grad, unet.parameters())) + list(adapter.parameters()),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        critic_optimizer = None
        if critic is not None:
            critic_optimizer = build_critic_optimizer(
                critic.parameters(),
                lr=float(self.config.wgan_critic_learning_rate),
                beta1=float(self.config.wgan_critic_beta1),
                beta2=float(self.config.wgan_critic_beta2),
            )

        steps_per_epoch = math.ceil(len(train_loader) / max(self.config.gradient_accumulation_steps, 1))
        self.total_optimizer_steps = int(self.config.epochs * max(steps_per_epoch, 1))
        lr_scheduler = get_scheduler(
            self.config.lr_scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=int(self.config.warmup_steps),
            num_training_steps=self.total_optimizer_steps,
        )

        vae.to(self.accelerator.device, dtype=weight_dtype)
        text_encoder.to(self.accelerator.device, dtype=weight_dtype)
        text_encoder_2.to(self.accelerator.device, dtype=weight_dtype)
        controlnet.to(self.accelerator.device, dtype=weight_dtype)
        unet.to(self.accelerator.device, dtype=weight_dtype)
        adapter.to(self.accelerator.device, dtype=torch.float32)
        if critic is not None:
            critic.to(self.accelerator.device, dtype=torch.float32)
        for parameter in unet.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

        if self.config.enable_gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        if critic is not None and critic_optimizer is not None:
            unet, adapter, critic, optimizer, critic_optimizer, train_loader, lr_scheduler = self.accelerator.prepare(
                unet, adapter, critic, optimizer, critic_optimizer, train_loader, lr_scheduler,
            )
        else:
            unet, adapter, optimizer, train_loader, lr_scheduler = self.accelerator.prepare(
                unet, adapter, optimizer, train_loader, lr_scheduler,
            )
        self._emit_training_startup_report(mode="train_with_preview")

        progress_bar = tqdm(total=self.total_optimizer_steps, disable=not self.accelerator.is_local_main_process, desc=f"{self.group.group_id}-seed{self.seed}")
        seed_started_perf = time.perf_counter()
        self.seed_started_at_iso = now_iso()
        self.write_status(
            state="running",
            message="Training with preview started",
            epoch=0,
            global_step=0,
            total_optimizer_steps=self.total_optimizer_steps,
            dataset_size=len(train_dataset),
            val_size=len(val_dataset),
            seed_started_at=self.seed_started_at_iso,
            preview_epochs=list(preview_epochs_set),
        )

        epoch_memory_monitor: PeakMemoryMonitor | None = None
        try:
            for epoch in range(1, int(self.config.epochs) + 1):
                _release_torch_memory(self.accelerator.device)
                wgan_memory_guard_triggered, wgan_memory_guard_reason = self._maybe_disable_wgan_for_memory(epoch)
                if wgan_memory_guard_triggered:
                    append_jsonl(self.metric_log_path, {
                        "timestamp": now_iso(),
                        "event": "wgan_memory_guard_triggered",
                        "group_id": self.group.group_id,
                        "seed": self.seed,
                        "epoch": epoch,
                        "reason": wgan_memory_guard_reason,
                    })

                epoch_started = time.perf_counter()
                epoch_memory_monitor = PeakMemoryMonitor(device=self.accelerator.device).start()
                self.latest_quality_recovery_triggered = False
                unet.train()
                adapter.train()
                epoch_losses: list[float] = []
                epoch_diffusion_losses: list[float] = []
                epoch_adversarial_losses: list[float] = []
                epoch_reconstruction_losses: list[float] = []
                epoch_flat_color_penalties: list[float] = []
                epoch_latent_reconstruction_losses: list[float] = []
                epoch_latent_distribution_losses: list[float] = []
                epoch_latent_local_variance_losses: list[float] = []
                epoch_critic_losses: list[float] = []
                epoch_gradient_penalties: list[float] = []
                collapse_guard_triggered = False
                collapse_guard_reason = ""

                for batch in train_loader:
                    critic_loss_value: float | None = None
                    gradient_penalty_value: float | None = None
                    critic_real_score_value: float | None = None
                    critic_fake_score_value: float | None = None
                    adversarial_loss_value: float | None = None
                    reconstruction_l1_value: float | None = None
                    flat_color_penalty_value: float | None = None
                    latent_reconstruction_value: float | None = None
                    latent_distribution_value: float | None = None
                    latent_local_variance_value: float | None = None
                    diffusion_loss_value: float | None = None
                    effective_adversarial_weight_value = 0.0
                    effective_reconstruction_weight_value = 0.0
                    effective_flat_color_weight_value = 0.0
                    effective_latent_reconstruction_weight_value = float(getattr(self.config, "latent_reconstruction_weight", 0.0))
                    effective_latent_distribution_weight_value = float(getattr(self.config, "latent_distribution_weight", 0.0))
                    effective_latent_local_variance_weight_value = float(getattr(self.config, "latent_local_variance_weight", 0.0))
                    decoded_image_spatial_std_value: float | None = None
                    latent_spatial_std_value: float | None = None
                    generator_step_skipped = False

                    with self.accelerator.accumulate(unet, adapter):
                        color_images = _sanitize_tensor(batch["color"].to(self.accelerator.device, dtype=weight_dtype), nan=0.0, posinf=1.0, neginf=-1.0)
                        lineart_images = batch["lineart"].to(self.accelerator.device, dtype=torch.float32)
                        lineart_images = _sanitize_tensor(adapter(lineart_images), nan=0.0, posinf=1.0, neginf=0.0).to(weight_dtype)

                        with torch.no_grad():
                            latents = vae.encode(color_images).latent_dist.sample()
                            latents = _sanitize_tensor(latents * vae.config.scaling_factor)

                        noise = _sanitize_tensor(torch.randn_like(latents), nan=0.0, posinf=1.0, neginf=-1.0)
                        batch_size = latents.shape[0]
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device).long()
                        noisy_latents = _sanitize_tensor(noise_scheduler.add_noise(latents, noise, timesteps))
                        encoder_hidden_states, added_cond_kwargs = self.encode_prompts(
                            tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                            batch_size=batch_size, device=self.accelerator.device, dtype=weight_dtype,
                        )
                        encoder_hidden_states = _sanitize_tensor(encoder_hidden_states)
                        added_cond_kwargs = {
                            key: _sanitize_tensor(value, nan=0.0, posinf=10.0, neginf=-10.0)
                            for key, value in added_cond_kwargs.items()
                        }
                        with torch.no_grad():
                            down_block_residuals, mid_block_residual = controlnet(
                                noisy_latents, timesteps,
                                encoder_hidden_states=encoder_hidden_states,
                                controlnet_cond=lineart_images,
                                conditioning_scale=float(self.config.controlnet_conditioning_scale),
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )
                            down_block_residuals = tuple(_sanitize_tensor(item) for item in down_block_residuals)
                            mid_block_residual = _sanitize_tensor(mid_block_residual)
                        model_pred = unet(
                            noisy_latents, timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            down_block_additional_residuals=down_block_residuals,
                            mid_block_additional_residual=mid_block_residual,
                            added_cond_kwargs=added_cond_kwargs,
                        ).sample
                        model_pred = _sanitize_tensor(model_pred)
                        diffusion_loss = F.mse_loss(model_pred.float(), noise.float())

                        if not torch.isfinite(diffusion_loss):
                            optimizer.zero_grad(set_to_none=True)
                            append_jsonl(self.log_path, {
                                "timestamp": now_iso(),
                                "event": "skipped_non_finite_train_loss",
                                "group_id": self.group.group_id,
                                "seed": self.seed,
                                "epoch": epoch,
                                "global_step": self.global_step,
                            })
                            continue

                        diffusion_loss_value = float(diffusion_loss.detach().item())
                        total_loss = diffusion_loss
                        gan_active = critic is not None and critic_optimizer is not None and self._wgan_is_active(epoch)
                        needs_predicted_original = gan_active or self._latent_auxiliary_active()
                        predicted_original_latents: torch.Tensor | None = None
                        decoded_fake_images: torch.Tensor | None = None
                        real_images: torch.Tensor | None = None

                        if needs_predicted_original:
                            predicted_original_latents = _predict_original_latents(
                                noise_scheduler, noisy_latents, model_pred, timesteps,
                            )

                        if predicted_original_latents is not None and self._latent_auxiliary_active():
                            latent_reconstruction_loss, latent_distribution_loss, latent_spatial_std_value = self._compute_latent_consistency_losses(
                                predicted_original_latents,
                                latents,
                            )
                            latent_reconstruction_value = float(latent_reconstruction_loss.detach().item())
                            latent_distribution_value = float(latent_distribution_loss.detach().item())
                            if effective_latent_reconstruction_weight_value > 0.0:
                                total_loss = total_loss + effective_latent_reconstruction_weight_value * latent_reconstruction_loss
                            if effective_latent_distribution_weight_value > 0.0:
                                total_loss = total_loss + effective_latent_distribution_weight_value * latent_distribution_loss
                            if effective_latent_local_variance_weight_value > 0.0:
                                latent_local_variance_loss = self._compute_latent_local_variance_loss(
                                    predicted_original_latents,
                                    latents,
                                )
                                latent_local_variance_value = float(latent_local_variance_loss.detach().item())
                                total_loss = total_loss + effective_latent_local_variance_weight_value * latent_local_variance_loss

                        if gan_active and critic is not None and critic_optimizer is not None and predicted_original_latents is not None:
                            decoded_fake_images = _decode_latents_to_images(vae, predicted_original_latents, weight_dtype)
                            real_images = color_images.float()
                            if self.accelerator.sync_gradients:
                                critic_input_real = resize_for_critic(real_images.detach(), int(self.config.critic_image_size))
                                critic_input_fake_detached = resize_for_critic(
                                    decoded_fake_images.detach(), int(self.config.critic_image_size),
                                )
                                _set_requires_grad(critic, True)
                                critic_steps = max(1, int(self.config.wgan_critic_steps))
                                for _ in range(critic_steps):
                                    critic_optimizer.zero_grad(set_to_none=True)
                                    critic_real_scores = critic(critic_input_real)
                                    critic_fake_scores = critic(critic_input_fake_detached)
                                    gradient_penalty = compute_gradient_penalty(
                                        critic, critic_input_real, critic_input_fake_detached,
                                    )
                                    critic_loss = (
                                        critic_fake_scores.mean()
                                        - critic_real_scores.mean()
                                        + float(self.config.wgan_gp_weight) * gradient_penalty
                                    )
                                    if torch.isfinite(critic_loss):
                                        self.accelerator.backward(critic_loss)
                                        critic_optimizer.step()
                                        critic_loss_value = float(critic_loss.detach().item())
                                        gradient_penalty_value = float(gradient_penalty.detach().item())
                                        critic_real_score_value = float(critic_real_scores.mean().detach().item())
                                        critic_fake_score_value = float(critic_fake_scores.mean().detach().item())
                                    critic_optimizer.zero_grad(set_to_none=True)

                            _set_requires_grad(critic, False)
                            reconstruction_l1_loss = F.l1_loss(decoded_fake_images.float(), real_images)
                            flat_color_penalty = compute_flat_color_penalty(
                                decoded_fake_images.float(), float(self.config.anti_flat_color_min_std),
                            )
                            reconstruction_l1_value = float(reconstruction_l1_loss.detach().item())
                            flat_color_penalty_value = float(flat_color_penalty.detach().item())
                            decoded_image_spatial_std_value = self._decoded_image_spatial_std(decoded_fake_images.float())
                            (
                                effective_adversarial_weight_value,
                                effective_reconstruction_weight_value,
                                effective_flat_color_weight_value,
                            ) = self._compute_effective_adversarial_weights(
                                epoch=epoch,
                                critic_real_score=critic_real_score_value,
                                critic_fake_score=critic_fake_score_value,
                                decoded_image_spatial_std=decoded_image_spatial_std_value,
                            )
                            total_loss = (
                                total_loss
                                + float(effective_reconstruction_weight_value) * reconstruction_l1_loss
                                + float(effective_flat_color_weight_value) * flat_color_penalty
                            )
                            critic_input_fake = resize_for_critic(decoded_fake_images, int(self.config.critic_image_size))
                            adversarial_loss = -critic(critic_input_fake).mean()
                            adversarial_loss_value = float(adversarial_loss.detach().item())
                            if effective_adversarial_weight_value > 0.0:
                                total_loss = total_loss + float(effective_adversarial_weight_value) * adversarial_loss
                            _set_requires_grad(critic, True)

                        if not torch.isfinite(total_loss):
                            optimizer.zero_grad(set_to_none=True)
                            if critic_optimizer is not None:
                                critic_optimizer.zero_grad(set_to_none=True)
                            append_jsonl(self.log_path, {
                                "timestamp": now_iso(),
                                "event": "skipped_non_finite_train_loss",
                                "group_id": self.group.group_id,
                                "seed": self.seed,
                                "epoch": epoch,
                                "global_step": self.global_step,
                                "reason": "non_finite_total_loss",
                            })
                            continue

                        self.accelerator.backward(total_loss)
                        if self.accelerator.sync_gradients and self.config.max_grad_norm > 0:
                            self.accelerator.clip_grad_norm_(list(unet.parameters()) + list(adapter.parameters()), self.config.max_grad_norm)
                        optimizer.step()
                        generator_step_skipped = bool(getattr(optimizer, "step_was_skipped", False))
                        if not generator_step_skipped:
                            lr_scheduler.step()
                        self._apply_best_fid_lr_if_needed(optimizer)
                        optimizer.zero_grad(set_to_none=True)

                    if self.accelerator.sync_gradients and generator_step_skipped:
                        current_lr = self._current_learning_rate(optimizer, lr_scheduler)
                        append_jsonl(self.log_path, {
                            "timestamp": now_iso(),
                            "event": "skipped_optimizer_step",
                            "group_id": self.group.group_id,
                            "seed": self.seed,
                            "epoch": epoch,
                            "global_step": self.global_step,
                            "loss": float(total_loss.detach().item()),
                            "lr": current_lr,
                        })
                        progress_bar.set_postfix(loss=f"{float(total_loss.detach().item()):.4f}", lr=f"{current_lr:.2e}", skipped="yes")
                        continue

                    if self.accelerator.sync_gradients:
                        self.global_step += 1
                        loss_value = float(total_loss.detach().item())
                        epoch_losses.append(loss_value)
                        if diffusion_loss_value is not None:
                            epoch_diffusion_losses.append(diffusion_loss_value)
                        if adversarial_loss_value is not None:
                            epoch_adversarial_losses.append(adversarial_loss_value)
                        if reconstruction_l1_value is not None:
                            epoch_reconstruction_losses.append(reconstruction_l1_value)
                        if flat_color_penalty_value is not None:
                            epoch_flat_color_penalties.append(flat_color_penalty_value)
                        if latent_reconstruction_value is not None:
                            epoch_latent_reconstruction_losses.append(latent_reconstruction_value)
                        if latent_distribution_value is not None:
                            epoch_latent_distribution_losses.append(latent_distribution_value)
                        if latent_local_variance_value is not None:
                            epoch_latent_local_variance_losses.append(latent_local_variance_value)
                        if critic_loss_value is not None:
                            epoch_critic_losses.append(critic_loss_value)
                        if gradient_penalty_value is not None:
                            epoch_gradient_penalties.append(gradient_penalty_value)
                        self.best_train_loss = min(self.best_train_loss, loss_value)
                        progress_bar.update(1)
                        append_jsonl(self.log_path, {
                            "timestamp": now_iso(),
                            "event": "train",
                            "group_id": self.group.group_id,
                            "seed": self.seed,
                            "epoch": epoch,
                            "global_step": self.global_step,
                            "loss": loss_value,
                            "lr": self._current_learning_rate(optimizer, lr_scheduler),
                        })
                        progress_bar.set_postfix(
                            loss=f"{loss_value:.4f}",
                            lr=f"{self._current_learning_rate(optimizer, lr_scheduler):.2e}",
                        )

                        if self.global_step % max(1, int(self.config.checkpoint_interval_steps)) == 0:
                            self.save_checkpoint(unet, adapter, step=self.global_step, epoch=epoch, critic=critic)

                epoch_train_loss = float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else None
                epoch_diffusion_loss = float(sum(epoch_diffusion_losses) / len(epoch_diffusion_losses)) if epoch_diffusion_losses else None
                epoch_adversarial_loss = float(sum(epoch_adversarial_losses) / len(epoch_adversarial_losses)) if epoch_adversarial_losses else None
                epoch_reconstruction_loss = float(sum(epoch_reconstruction_losses) / len(epoch_reconstruction_losses)) if epoch_reconstruction_losses else None
                epoch_flat_color_penalty = float(sum(epoch_flat_color_penalties) / len(epoch_flat_color_penalties)) if epoch_flat_color_penalties else None
                epoch_latent_reconstruction_loss = (
                    float(sum(epoch_latent_reconstruction_losses) / len(epoch_latent_reconstruction_losses))
                    if epoch_latent_reconstruction_losses
                    else None
                )
                epoch_latent_distribution_loss = (
                    float(sum(epoch_latent_distribution_losses) / len(epoch_latent_distribution_losses))
                    if epoch_latent_distribution_losses
                    else None
                )
                epoch_latent_local_variance_loss = (
                    float(sum(epoch_latent_local_variance_losses) / len(epoch_latent_local_variance_losses))
                    if epoch_latent_local_variance_losses
                    else None
                )
                epoch_critic_loss = float(sum(epoch_critic_losses) / len(epoch_critic_losses)) if epoch_critic_losses else None
                epoch_gradient_penalty = float(sum(epoch_gradient_penalties) / len(epoch_gradient_penalties)) if epoch_gradient_penalties else None
                epoch_time = time.perf_counter() - epoch_started
                seed_elapsed_seconds = time.perf_counter() - seed_started_perf
                epoch_memory = epoch_memory_monitor.stop()
                epoch_memory_monitor = None
                gpu_peak = epoch_memory.get("gpu_memory_peak_gb")
                gpu_reserved_peak = epoch_memory.get("gpu_memory_reserved_peak_gb")
                cpu_peak = epoch_memory.get("cpu_memory_peak_gb")
                self.train_gpu_memory_peak_gb = _max_optional(self.train_gpu_memory_peak_gb, gpu_peak)
                self.train_gpu_memory_reserved_peak_gb = _max_optional(self.train_gpu_memory_reserved_peak_gb, gpu_reserved_peak)
                self.train_cpu_memory_peak_gb = _max_optional(self.train_cpu_memory_peak_gb, cpu_peak)
                self.latest_epoch_time_seconds = round(epoch_time, 4)
                self.latest_epoch_time_hms = _format_duration(epoch_time)
                self.seed_elapsed_seconds = round(seed_elapsed_seconds, 4)
                self.seed_elapsed_hms = _format_duration(seed_elapsed_seconds)

                metric_payload = {
                    "timestamp": now_iso(),
                    "event": "epoch_end",
                    "group_id": self.group.group_id,
                    "seed": self.seed,
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "diffusion_loss": epoch_diffusion_loss,
                    "adversarial_loss": epoch_adversarial_loss,
                    "reconstruction_l1_loss": epoch_reconstruction_loss,
                    "flat_color_penalty": epoch_flat_color_penalty,
                    "latent_reconstruction_loss": epoch_latent_reconstruction_loss,
                    "latent_distribution_loss": epoch_latent_distribution_loss,
                    "latent_local_variance_loss": epoch_latent_local_variance_loss,
                    "critic_loss": epoch_critic_loss,
                    "gradient_penalty": epoch_gradient_penalty,
                    "learning_rate": self._current_learning_rate(optimizer, lr_scheduler),
                    "adversarial_cooldown_until_epoch": self.adversarial_cooldown_until_epoch,
                    "collapse_guard_triggered": collapse_guard_triggered,
                    "collapse_guard_reason": collapse_guard_reason,
                    "wgan_disabled_reason": self.wgan_disabled_reason,
                    "wgan_disabled_epoch": self.wgan_disabled_epoch,
                    "gpu_memory_peak_gb": gpu_peak,
                    "gpu_memory_reserved_peak_gb": gpu_reserved_peak,
                    "cpu_memory_peak_gb": cpu_peak,
                    "epoch_time_seconds": round(epoch_time, 4),
                    "epoch_time_hms": _format_duration(epoch_time),
                    "seed_elapsed_seconds": round(seed_elapsed_seconds, 4),
                    "seed_elapsed_hms": _format_duration(seed_elapsed_seconds),
                }
                append_jsonl(self.metric_log_path, metric_payload)
                self.epoch_history.append(metric_payload)
                save_json(self.run_dir / "logs" / "epoch_history.json", {"epochs": self.epoch_history})

                self.write_status(
                    state="running",
                    message="Training in progress",
                    epoch=epoch,
                    global_step=self.global_step,
                    total_optimizer_steps=self.total_optimizer_steps,
                    train_loss=epoch_train_loss,
                    learning_rate=self._current_learning_rate(optimizer, lr_scheduler),
                    adversarial_cooldown_until_epoch=self.adversarial_cooldown_until_epoch,
                    collapse_guard_triggered=collapse_guard_triggered,
                    collapse_guard_reason=collapse_guard_reason,
                    gpu_memory_peak_gb=gpu_peak,
                    epoch_time_seconds=round(epoch_time, 4),
                    epoch_time_hms=_format_duration(epoch_time),
                    seed_elapsed_seconds=round(seed_elapsed_seconds, 4),
                    seed_elapsed_hms=_format_duration(seed_elapsed_seconds),
                    seed_started_at=self.seed_started_at_iso,
                )
                self.update_summary(
                    status="running",
                    train_loss=epoch_train_loss,
                    epoch=epoch,
                    collapse_guard_triggered=collapse_guard_triggered,
                    collapse_guard_reason=collapse_guard_reason,
                )
                sync_run_localized_outputs(self.run_dir)
                _release_torch_memory(self.accelerator.device)

                if self.config.save_every_epoch:
                    self.save_checkpoint(unet, adapter, step=self.global_step, epoch=epoch, critic=critic)

                # ========== 定期推理出图（不计算指标） ==========
                if epoch in preview_epochs_set and self.accelerator.is_local_main_process and self.metric_eval_records:
                    print(f"[Epoch {epoch}] 开始推理出图...")
                    _release_torch_memory(self.accelerator.device)
                    self.generate_validation_images_for_epoch(
                        epoch=epoch,
                        text_encoder=text_encoder,
                        text_encoder_2=text_encoder_2,
                        tokenizer=tokenizer,
                        tokenizer_2=tokenizer_2,
                        vae=vae,
                        controlnet=controlnet,
                        unet=unet,
                        adapter=adapter,
                        base_path=base_path,
                        weight_dtype=weight_dtype,
                    )
                    _release_torch_memory(self.accelerator.device)
                    print(f"[Epoch {epoch}] 推理完成，图片已保存（不计算指标）")
                # ========== 推理结束 ==========

            self.save_final_artifacts(unet, adapter, critic=critic)

            # 保存推理epoch信息
            preview_info = {
                "preview_epochs": sorted(list(preview_epochs_set)),
                "total_epochs": self.config.epochs,
                "epochs_with_preview": [e for e in range(1, self.config.epochs + 1) if e in preview_epochs_set],
            }
            save_json(self.run_dir / "preview_epochs.json", preview_info)

            curve_paths = export_training_curves(self.run_dir)
            self.loss_curve_path = curve_paths.get("loss_curve_path", self.loss_curve_path)
            self.lr_curve_path = curve_paths.get("lr_curve_path", self.lr_curve_path)
            final_seed_elapsed_seconds = round(time.perf_counter() - seed_started_perf, 4)
            self.seed_elapsed_seconds = final_seed_elapsed_seconds
            self.seed_elapsed_hms = _format_duration(final_seed_elapsed_seconds)

            self.write_status(
                state="completed",
                message="Training with preview completed (Part 1 finished)",
                epoch=self.config.epochs,
                global_step=self.global_step,
                total_optimizer_steps=self.total_optimizer_steps,
                best_train_loss=self.best_train_loss,
                best_val_loss=None if self.best_val_loss == float("inf") else self.best_val_loss,
                params_m=self.params_m,
                flops_g=self.flops_g,
                seed_started_at=self.seed_started_at_iso,
                seed_elapsed_seconds=self.seed_elapsed_seconds,
                seed_elapsed_hms=self.seed_elapsed_hms,
                latest_epoch_time_seconds=self.latest_epoch_time_seconds,
                latest_epoch_time_hms=self.latest_epoch_time_hms,
                train_gpu_memory_peak_gb=self.train_gpu_memory_peak_gb,
                train_gpu_memory_reserved_peak_gb=self.train_gpu_memory_reserved_peak_gb,
                train_cpu_memory_peak_gb=self.train_cpu_memory_peak_gb,
                loss_curve_path=self.loss_curve_path,
                lr_curve_path=self.lr_curve_path,
                preview_epochs=list(preview_epochs_set),
                part=1,
                note="Run analyze_results.py (or sci.py analyze) for post-hoc metric calculation",
            )
            self.update_summary(
                status="completed",
                train_loss=None if self.best_train_loss == float("inf") else self.best_train_loss,
                epoch=self.config.epochs,
                part=1,
            )
            sync_run_localized_outputs(self.run_dir)
            print(f"[完成] Part 1 训练+推理完成! 运行目录: {self.run_dir}")
            print(f"[提示] 运行 'python analyze_results.py --output-root {self.config.output_root}' 计算 Part 2 指标")
            return self.run_dir

        except Exception as exc:
            failed_seed_elapsed_seconds = round(time.perf_counter() - seed_started_perf, 4)
            self.seed_elapsed_seconds = failed_seed_elapsed_seconds
            self.seed_elapsed_hms = _format_duration(failed_seed_elapsed_seconds)
            self.write_status(
                state="failed",
                message=str(exc),
                epoch=0,
                global_step=self.global_step,
                total_optimizer_steps=self.total_optimizer_steps,
                seed_started_at=self.seed_started_at_iso,
                seed_elapsed_seconds=self.seed_elapsed_seconds,
                seed_elapsed_hms=self.seed_elapsed_hms,
            )
            self.update_summary(status="failed", train_loss=None, val_loss=None, epoch=0)
            sync_run_localized_outputs(self.run_dir)
            raise
        finally:
            if epoch_memory_monitor is not None:
                epoch_memory_monitor.stop()
            progress_bar.close()
            _release_torch_memory(self.accelerator.device)

        epoch_memory_monitor: PeakMemoryMonitor | None = None
        try:
            for epoch in range(1, int(self.config.epochs) + 1):
                _release_torch_memory(self.accelerator.device)
                wgan_memory_guard_triggered, wgan_memory_guard_reason = self._maybe_disable_wgan_for_memory(epoch)
                if wgan_memory_guard_triggered:
                    append_jsonl(
                        self.metric_log_path,
                        {
                            "timestamp": now_iso(),
                            "event": "wgan_memory_guard_triggered",
                            "group_id": self.group.group_id,
                            "seed": self.seed,
                            "epoch": epoch,
                            "reason": wgan_memory_guard_reason,
                        },
                    )
                epoch_started = time.perf_counter()
                epoch_memory_monitor = PeakMemoryMonitor(device=self.accelerator.device).start()
                self.latest_quality_recovery_triggered = False
                unet.train()
                adapter.train()
                epoch_losses: list[float] = []
                epoch_diffusion_losses: list[float] = []
                epoch_adversarial_losses: list[float] = []
                epoch_reconstruction_losses: list[float] = []
                epoch_flat_color_penalties: list[float] = []
                epoch_latent_reconstruction_losses: list[float] = []
                epoch_latent_distribution_losses: list[float] = []
                epoch_latent_local_variance_losses: list[float] = []
                epoch_critic_losses: list[float] = []
                epoch_gradient_penalties: list[float] = []
                collapse_guard_triggered = False
                collapse_guard_reason = ""

                for batch in train_loader:
                    critic_loss_value: float | None = None
                    gradient_penalty_value: float | None = None
                    critic_real_score_value: float | None = None
                    critic_fake_score_value: float | None = None
                    adversarial_loss_value: float | None = None
                    reconstruction_l1_value: float | None = None
                    flat_color_penalty_value: float | None = None
                    latent_reconstruction_value: float | None = None
                    latent_distribution_value: float | None = None
                    latent_local_variance_value: float | None = None
                    diffusion_loss_value: float | None = None
                    effective_adversarial_weight_value = 0.0
                    effective_reconstruction_weight_value = 0.0
                    effective_flat_color_weight_value = 0.0
                    effective_latent_reconstruction_weight_value = float(getattr(self.config, "latent_reconstruction_weight", 0.0))
                    effective_latent_distribution_weight_value = float(getattr(self.config, "latent_distribution_weight", 0.0))
                    effective_latent_local_variance_weight_value = float(getattr(self.config, "latent_local_variance_weight", 0.0))
                    decoded_image_spatial_std_value: float | None = None
                    latent_spatial_std_value: float | None = None
                    generator_step_skipped = False
                    with self.accelerator.accumulate(unet, adapter):
                        color_images = _sanitize_tensor(batch["color"].to(self.accelerator.device, dtype=weight_dtype), nan=0.0, posinf=1.0, neginf=-1.0)
                        lineart_images = batch["lineart"].to(self.accelerator.device, dtype=torch.float32)
                        lineart_images = _sanitize_tensor(adapter(lineart_images), nan=0.0, posinf=1.0, neginf=0.0).to(weight_dtype)

                        with torch.no_grad():
                            latents = vae.encode(color_images).latent_dist.sample()
                            latents = _sanitize_tensor(latents * vae.config.scaling_factor)

                        noise = _sanitize_tensor(torch.randn_like(latents), nan=0.0, posinf=1.0, neginf=-1.0)
                        batch_size = latents.shape[0]
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device).long()
                        noisy_latents = _sanitize_tensor(noise_scheduler.add_noise(latents, noise, timesteps))
                        encoder_hidden_states, added_cond_kwargs = self.encode_prompts(
                            tokenizer,
                            tokenizer_2,
                            text_encoder,
                            text_encoder_2,
                            batch_size=batch_size,
                            device=self.accelerator.device,
                            dtype=weight_dtype,
                        )
                        encoder_hidden_states = _sanitize_tensor(encoder_hidden_states)
                        added_cond_kwargs = {
                            key: _sanitize_tensor(value, nan=0.0, posinf=10.0, neginf=-10.0)
                            for key, value in added_cond_kwargs.items()
                        }
                        with torch.no_grad():
                            down_block_residuals, mid_block_residual = controlnet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=encoder_hidden_states,
                                controlnet_cond=lineart_images,
                                conditioning_scale=float(self.config.controlnet_conditioning_scale),
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )
                            down_block_residuals = tuple(_sanitize_tensor(item) for item in down_block_residuals)
                            mid_block_residual = _sanitize_tensor(mid_block_residual)
                        model_pred = unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            down_block_additional_residuals=down_block_residuals,
                            mid_block_additional_residual=mid_block_residual,
                            added_cond_kwargs=added_cond_kwargs,
                        ).sample
                        model_pred = _sanitize_tensor(model_pred)
                        diffusion_loss = F.mse_loss(model_pred.float(), noise.float())
                        if not torch.isfinite(diffusion_loss):
                            optimizer.zero_grad(set_to_none=True)
                            append_jsonl(
                                self.log_path,
                                {
                                    "timestamp": now_iso(),
                                    "event": "skipped_non_finite_train_loss",
                                    "group_id": self.group.group_id,
                                    "seed": self.seed,
                                    "epoch": epoch,
                                    "global_step": self.global_step,
                                },
                            )
                            continue
                        diffusion_loss_value = float(diffusion_loss.detach().item())
                        total_loss = diffusion_loss
                        gan_active = critic is not None and critic_optimizer is not None and self._wgan_is_active(epoch)
                        needs_predicted_original = gan_active or self._latent_auxiliary_active()
                        predicted_original_latents: torch.Tensor | None = None
                        decoded_fake_images: torch.Tensor | None = None
                        real_images: torch.Tensor | None = None

                        if needs_predicted_original:
                            predicted_original_latents = _predict_original_latents(
                                noise_scheduler,
                                noisy_latents,
                                model_pred,
                                timesteps,
                            )

                        if predicted_original_latents is not None and self._latent_auxiliary_active():
                            latent_reconstruction_loss, latent_distribution_loss, latent_spatial_std_value = self._compute_latent_consistency_losses(
                                predicted_original_latents,
                                latents,
                            )
                            latent_reconstruction_value = float(latent_reconstruction_loss.detach().item())
                            latent_distribution_value = float(latent_distribution_loss.detach().item())
                            if effective_latent_reconstruction_weight_value > 0.0:
                                total_loss = total_loss + effective_latent_reconstruction_weight_value * latent_reconstruction_loss
                            if effective_latent_distribution_weight_value > 0.0:
                                total_loss = total_loss + effective_latent_distribution_weight_value * latent_distribution_loss
                            if effective_latent_local_variance_weight_value > 0.0:
                                latent_local_variance_loss = self._compute_latent_local_variance_loss(
                                    predicted_original_latents,
                                    latents,
                                )
                                latent_local_variance_value = float(latent_local_variance_loss.detach().item())
                                total_loss = total_loss + effective_latent_local_variance_weight_value * latent_local_variance_loss

                        if gan_active and critic is not None and critic_optimizer is not None and predicted_original_latents is not None:
                            decoded_fake_images = _decode_latents_to_images(vae, predicted_original_latents, weight_dtype)
                            real_images = color_images.float()
                            if self.accelerator.sync_gradients:
                                critic_input_real = resize_for_critic(real_images.detach(), int(self.config.critic_image_size))
                                critic_input_fake_detached = resize_for_critic(
                                    decoded_fake_images.detach(),
                                    int(self.config.critic_image_size),
                                )
                                _set_requires_grad(critic, True)
                                critic_steps = max(1, int(self.config.wgan_critic_steps))
                                for _ in range(critic_steps):
                                    critic_optimizer.zero_grad(set_to_none=True)
                                    critic_real_scores = critic(critic_input_real)
                                    critic_fake_scores = critic(critic_input_fake_detached)
                                    gradient_penalty = compute_gradient_penalty(
                                        critic,
                                        critic_input_real,
                                        critic_input_fake_detached,
                                    )
                                    critic_loss = (
                                        critic_fake_scores.mean()
                                        - critic_real_scores.mean()
                                        + float(self.config.wgan_gp_weight) * gradient_penalty
                                    )
                                    if torch.isfinite(critic_loss):
                                        self.accelerator.backward(critic_loss)
                                        critic_optimizer.step()
                                        critic_loss_value = float(critic_loss.detach().item())
                                        gradient_penalty_value = float(gradient_penalty.detach().item())
                                        critic_real_score_value = float(critic_real_scores.mean().detach().item())
                                        critic_fake_score_value = float(critic_fake_scores.mean().detach().item())
                                    critic_optimizer.zero_grad(set_to_none=True)

                            _set_requires_grad(critic, False)
                            reconstruction_l1_loss = F.l1_loss(decoded_fake_images.float(), real_images)
                            flat_color_penalty = compute_flat_color_penalty(
                                decoded_fake_images.float(),
                                float(self.config.anti_flat_color_min_std),
                            )
                            reconstruction_l1_value = float(reconstruction_l1_loss.detach().item())
                            flat_color_penalty_value = float(flat_color_penalty.detach().item())
                            decoded_image_spatial_std_value = self._decoded_image_spatial_std(decoded_fake_images.float())
                            (
                                effective_adversarial_weight_value,
                                effective_reconstruction_weight_value,
                                effective_flat_color_weight_value,
                            ) = self._compute_effective_adversarial_weights(
                                epoch=epoch,
                                critic_real_score=critic_real_score_value,
                                critic_fake_score=critic_fake_score_value,
                                decoded_image_spatial_std=decoded_image_spatial_std_value,
                            )
                            total_loss = (
                                total_loss
                                + float(effective_reconstruction_weight_value) * reconstruction_l1_loss
                                + float(effective_flat_color_weight_value) * flat_color_penalty
                            )
                            critic_input_fake = resize_for_critic(decoded_fake_images, int(self.config.critic_image_size))
                            adversarial_loss = -critic(critic_input_fake).mean()
                            adversarial_loss_value = float(adversarial_loss.detach().item())
                            if effective_adversarial_weight_value > 0.0:
                                total_loss = total_loss + float(effective_adversarial_weight_value) * adversarial_loss
                            _set_requires_grad(critic, True)

                        if not torch.isfinite(total_loss):
                            optimizer.zero_grad(set_to_none=True)
                            if critic_optimizer is not None:
                                critic_optimizer.zero_grad(set_to_none=True)
                            append_jsonl(
                                self.log_path,
                                {
                                    "timestamp": now_iso(),
                                    "event": "skipped_non_finite_train_loss",
                                    "group_id": self.group.group_id,
                                    "seed": self.seed,
                                    "epoch": epoch,
                                    "global_step": self.global_step,
                                    "reason": "non_finite_total_loss",
                                },
                            )
                            continue
                        self.accelerator.backward(total_loss)
                        if self.accelerator.sync_gradients and self.config.max_grad_norm > 0:
                            self.accelerator.clip_grad_norm_(list(unet.parameters()) + list(adapter.parameters()), self.config.max_grad_norm)
                        optimizer.step()
                        generator_step_skipped = bool(getattr(optimizer, "step_was_skipped", False))
                        if not generator_step_skipped:
                            lr_scheduler.step()
                        self._apply_best_fid_lr_if_needed(optimizer)
                        optimizer.zero_grad(set_to_none=True)

                    if self.accelerator.sync_gradients and generator_step_skipped:
                        current_lr = self._current_learning_rate(optimizer, lr_scheduler)
                        append_jsonl(
                            self.log_path,
                            {
                                "timestamp": now_iso(),
                                "event": "skipped_optimizer_step",
                                "group_id": self.group.group_id,
                                "seed": self.seed,
                                "epoch": epoch,
                                "global_step": self.global_step,
                                "loss": float(total_loss.detach().item()),
                                "lr": current_lr,
                            },
                        )
                        progress_bar.set_postfix(
                            loss=f"{float(total_loss.detach().item()):.4f}",
                            lr=f"{current_lr:.2e}",
                            skipped="yes",
                        )
                        continue

                    if self.accelerator.sync_gradients:
                        self.global_step += 1
                        loss_value = float(total_loss.detach().item())
                        epoch_losses.append(loss_value)
                        if diffusion_loss_value is not None:
                            epoch_diffusion_losses.append(diffusion_loss_value)
                        if adversarial_loss_value is not None:
                            epoch_adversarial_losses.append(adversarial_loss_value)
                        if reconstruction_l1_value is not None:
                            epoch_reconstruction_losses.append(reconstruction_l1_value)
                        if flat_color_penalty_value is not None:
                            epoch_flat_color_penalties.append(flat_color_penalty_value)
                        if latent_reconstruction_value is not None:
                            epoch_latent_reconstruction_losses.append(latent_reconstruction_value)
                        if latent_distribution_value is not None:
                            epoch_latent_distribution_losses.append(latent_distribution_value)
                        if latent_local_variance_value is not None:
                            epoch_latent_local_variance_losses.append(latent_local_variance_value)
                        if critic_loss_value is not None:
                            epoch_critic_losses.append(critic_loss_value)
                        if gradient_penalty_value is not None:
                            epoch_gradient_penalties.append(gradient_penalty_value)
                        self.best_train_loss = min(self.best_train_loss, loss_value)
                        progress_bar.update(1)
                        payload = {
                            "timestamp": now_iso(),
                            "event": "train",
                            "group_id": self.group.group_id,
                            "seed": self.seed,
                            "epoch": epoch,
                            "global_step": self.global_step,
                            "loss": loss_value,
                            "lr": self._current_learning_rate(optimizer, lr_scheduler),
                        }
                        append_jsonl(self.log_path, payload)
                        postfix = {
                            "loss": f"{loss_value:.4f}",
                            "lr": f"{self._current_learning_rate(optimizer, lr_scheduler):.2e}",
                        }
                        progress_bar.set_postfix(**postfix)

                        if self.global_step % max(1, int(self.config.checkpoint_interval_steps)) == 0:
                            self.save_checkpoint(unet, adapter, step=self.global_step, epoch=epoch, critic=critic)

                epoch_train_loss = float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else None
                epoch_diffusion_loss = (
                    float(sum(epoch_diffusion_losses) / len(epoch_diffusion_losses)) if epoch_diffusion_losses else None
                )
                epoch_adversarial_loss = (
                    float(sum(epoch_adversarial_losses) / len(epoch_adversarial_losses)) if epoch_adversarial_losses else None
                )
                epoch_reconstruction_loss = (
                    float(sum(epoch_reconstruction_losses) / len(epoch_reconstruction_losses))
                    if epoch_reconstruction_losses
                    else None
                )
                epoch_flat_color_penalty = (
                    float(sum(epoch_flat_color_penalties) / len(epoch_flat_color_penalties))
                    if epoch_flat_color_penalties
                    else None
                )
                epoch_latent_reconstruction_loss = (
                    float(sum(epoch_latent_reconstruction_losses) / len(epoch_latent_reconstruction_losses))
                    if epoch_latent_reconstruction_losses
                    else None
                )
                epoch_latent_distribution_loss = (
                    float(sum(epoch_latent_distribution_losses) / len(epoch_latent_distribution_losses))
                    if epoch_latent_distribution_losses
                    else None
                )
                epoch_latent_local_variance_loss = (
                    float(sum(epoch_latent_local_variance_losses) / len(epoch_latent_local_variance_losses))
                    if epoch_latent_local_variance_losses
                    else None
                )
                epoch_critic_loss = (
                    float(sum(epoch_critic_losses) / len(epoch_critic_losses)) if epoch_critic_losses else None
                )
                epoch_gradient_penalty = (
                    float(sum(epoch_gradient_penalties) / len(epoch_gradient_penalties))
                    if epoch_gradient_penalties
                    else None
                )
                epoch_time = time.perf_counter() - epoch_started
                seed_elapsed_seconds = time.perf_counter() - seed_started_perf
                epoch_memory = epoch_memory_monitor.stop()
                epoch_memory_monitor = None
                gpu_peak = epoch_memory.get("gpu_memory_peak_gb")
                gpu_reserved_peak = epoch_memory.get("gpu_memory_reserved_peak_gb")
                cpu_peak = epoch_memory.get("cpu_memory_peak_gb")
                self.train_gpu_memory_peak_gb = _max_optional(self.train_gpu_memory_peak_gb, gpu_peak)
                self.train_gpu_memory_reserved_peak_gb = _max_optional(self.train_gpu_memory_reserved_peak_gb, gpu_reserved_peak)
                self.train_cpu_memory_peak_gb = _max_optional(self.train_cpu_memory_peak_gb, cpu_peak)
                self.latest_epoch_time_seconds = round(epoch_time, 4)
                self.latest_epoch_time_hms = _format_duration(epoch_time)
                self.seed_elapsed_seconds = round(seed_elapsed_seconds, 4)
                self.seed_elapsed_hms = _format_duration(seed_elapsed_seconds)
                metric_payload = {
                    "timestamp": now_iso(),
                    "event": "epoch_end",
                    "group_id": self.group.group_id,
                    "seed": self.seed,
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "diffusion_loss": epoch_diffusion_loss,
                    "adversarial_loss": epoch_adversarial_loss,
                    "reconstruction_l1_loss": epoch_reconstruction_loss,
                    "flat_color_penalty": epoch_flat_color_penalty,
                    "latent_reconstruction_loss": epoch_latent_reconstruction_loss,
                    "latent_distribution_loss": epoch_latent_distribution_loss,
                    "latent_local_variance_loss": epoch_latent_local_variance_loss,
                    "critic_loss": epoch_critic_loss,
                    "gradient_penalty": epoch_gradient_penalty,
                    "learning_rate": self._current_learning_rate(optimizer, lr_scheduler),
                    "adversarial_cooldown_until_epoch": self.adversarial_cooldown_until_epoch,
                    "collapse_guard_triggered": collapse_guard_triggered,
                    "collapse_guard_reason": collapse_guard_reason,
                    "wgan_disabled_reason": self.wgan_disabled_reason,
                    "wgan_disabled_epoch": self.wgan_disabled_epoch,
                    "gpu_memory_peak_gb": gpu_peak,
                    "gpu_memory_reserved_peak_gb": gpu_reserved_peak,
                    "cpu_memory_peak_gb": cpu_peak,
                    "epoch_time_seconds": round(epoch_time, 4),
                    "epoch_time_hms": _format_duration(epoch_time),
                    "seed_elapsed_seconds": round(seed_elapsed_seconds, 4),
                    "seed_elapsed_hms": _format_duration(seed_elapsed_seconds),
                }
                append_jsonl(self.metric_log_path, metric_payload)
                self.epoch_history.append(metric_payload)
                save_json(self.run_dir / "logs" / "epoch_history.json", {"epochs": self.epoch_history})
                self.write_status(
                    state="running",
                    message="Training in progress",
                    epoch=epoch,
                    global_step=self.global_step,
                    total_optimizer_steps=self.total_optimizer_steps,
                    train_loss=epoch_train_loss,
                    learning_rate=self._current_learning_rate(optimizer, lr_scheduler),
                    adversarial_cooldown_until_epoch=self.adversarial_cooldown_until_epoch,
                    collapse_guard_triggered=collapse_guard_triggered,
                    collapse_guard_reason=collapse_guard_reason,
                    gpu_memory_peak_gb=gpu_peak,
                    epoch_time_seconds=round(epoch_time, 4),
                    epoch_time_hms=_format_duration(epoch_time),
                    seed_elapsed_seconds=round(seed_elapsed_seconds, 4),
                    seed_elapsed_hms=_format_duration(seed_elapsed_seconds),
                    seed_started_at=self.seed_started_at_iso,
                )
                self.update_summary(
                    status="running",
                    train_loss=epoch_train_loss,
                    epoch=epoch,
                    collapse_guard_triggered=collapse_guard_triggered,
                    collapse_guard_reason=collapse_guard_reason,
                )
                sync_run_localized_outputs(self.run_dir)
                _release_torch_memory(self.accelerator.device)

                if self.config.save_every_epoch:
                    self.save_checkpoint(unet, adapter, step=self.global_step, epoch=epoch, critic=critic)

            self.save_final_artifacts(unet, adapter, critic=critic)
            generation_metrics: dict[str, Any] = {}
            posthoc_summary: dict[str, Any] = {}
            if bool(getattr(self.config, "defer_generation_metrics_until_seed_end", False)):
                if self.accelerator.is_local_main_process and self.metric_eval_records:
                    for epoch_record in self.epoch_history:
                        epoch = int(epoch_record.get("epoch", 0) or 0)
                        if epoch <= 0:
                            continue
                        eval_dir = self.run_dir / "evaluations" / "validation_epochs" / f"epoch_{epoch:03d}"
                        if not eval_dir.exists() or not list((eval_dir / "generated").glob("*.png")):
                            _release_torch_memory(self.accelerator.device)
                            self.generate_validation_images_for_epoch(
                                epoch=epoch,
                                text_encoder=text_encoder,
                                text_encoder_2=text_encoder_2,
                                tokenizer=tokenizer,
                                tokenizer_2=tokenizer_2,
                                vae=vae,
                                controlnet=controlnet,
                                unet=unet,
                                adapter=adapter,
                                base_path=base_path,
                                weight_dtype=weight_dtype,
                            )
                            _release_torch_memory(self.accelerator.device)
                posthoc_summary = self._finalize_deferred_generation_metrics(
                    vae=vae,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    controlnet=controlnet,
                    unet=unet,
                    adapter=adapter,
                    critic=critic,
                )
                latest_metrics = posthoc_summary.get("latest_metrics", {}) if isinstance(posthoc_summary, dict) else {}
                if isinstance(latest_metrics, dict) and latest_metrics:
                    generation_metrics = latest_metrics
                if bool(posthoc_summary.get("successful_epochs", 0)):
                    self.write_status(
                        state="running",
                        message="Deferred generation metrics completed",
                        epoch=self.config.epochs,
                        global_step=self.global_step,
                        total_optimizer_steps=self.total_optimizer_steps,
                        posthoc_metrics_completed=True,
                        posthoc_successful_epochs=posthoc_summary.get("successful_epochs", 0),
                    )

            curve_paths = export_training_curves(self.run_dir)
            self.loss_curve_path = curve_paths.get("loss_curve_path", self.loss_curve_path)
            self.lr_curve_path = curve_paths.get("lr_curve_path", self.lr_curve_path)
            final_seed_elapsed_seconds = round(time.perf_counter() - seed_started_perf, 4)
            self.seed_elapsed_seconds = final_seed_elapsed_seconds
            self.seed_elapsed_hms = _format_duration(final_seed_elapsed_seconds)
            self.write_status(
                state="completed",
                message="Training completed",
                epoch=self.config.epochs,
                global_step=self.global_step,
                total_optimizer_steps=self.total_optimizer_steps,
                best_train_loss=self.best_train_loss,
                best_val_loss=None if self.best_val_loss == float("inf") else self.best_val_loss,
                params_m=self.params_m,
                flops_g=self.flops_g,
                seed_started_at=self.seed_started_at_iso,
                seed_elapsed_seconds=self.seed_elapsed_seconds,
                seed_elapsed_hms=self.seed_elapsed_hms,
                latest_epoch_time_seconds=self.latest_epoch_time_seconds,
                latest_epoch_time_hms=self.latest_epoch_time_hms,
                train_gpu_memory_peak_gb=self.train_gpu_memory_peak_gb,
                train_gpu_memory_reserved_peak_gb=self.train_gpu_memory_reserved_peak_gb,
                train_cpu_memory_peak_gb=self.train_cpu_memory_peak_gb,
                latest_eval_gpu_memory_peak_gb=self.latest_eval_gpu_memory_peak_gb,
                latest_eval_gpu_memory_reserved_peak_gb=self.latest_eval_gpu_memory_reserved_peak_gb,
                latest_eval_cpu_memory_peak_gb=self.latest_eval_cpu_memory_peak_gb,
                latest_eval_archive_dir=self.latest_eval_archive_dir,
                loss_curve_path=self.loss_curve_path,
                lr_curve_path=self.lr_curve_path,
            )
            self.update_summary(
                status="completed",
                train_loss=None if self.best_train_loss == float("inf") else self.best_train_loss,
                epoch=self.config.epochs,
            )
            sync_run_localized_outputs(self.run_dir)
            return self.run_dir
        except Exception as exc:
            if _is_cuda_oom_error(exc) and (self.epoch_history or self.best_fid != float("inf")):
                completed_epoch = int(self.epoch_history[-1].get("epoch", 0)) if self.epoch_history else max(0, int(self.best_fid_epoch))
                final_seed_elapsed_seconds = round(time.perf_counter() - seed_started_perf, 4)
                self.seed_elapsed_seconds = final_seed_elapsed_seconds
                self.seed_elapsed_hms = _format_duration(final_seed_elapsed_seconds)
                message = (
                    "Training stopped early after CUDA OOM; preserved the latest completed epoch and best checkpoint. "
                    f"Best FID epoch: {self.best_fid_epoch or completed_epoch}. Error: {exc}"
                )
                self.write_status(
                    state="completed",
                    message=message,
                    epoch=completed_epoch,
                    global_step=self.global_step,
                    total_optimizer_steps=self.total_optimizer_steps,
                    best_train_loss=None if self.best_train_loss == float("inf") else self.best_train_loss,
                    best_val_loss=None if self.best_val_loss == float("inf") else self.best_val_loss,
                    params_m=self.params_m,
                    flops_g=self.flops_g,
                    seed_started_at=self.seed_started_at_iso,
                    seed_elapsed_seconds=self.seed_elapsed_seconds,
                    seed_elapsed_hms=self.seed_elapsed_hms,
                    latest_epoch_time_seconds=self.latest_epoch_time_seconds,
                    latest_epoch_time_hms=self.latest_epoch_time_hms,
                    train_gpu_memory_peak_gb=self.train_gpu_memory_peak_gb,
                    train_gpu_memory_reserved_peak_gb=self.train_gpu_memory_reserved_peak_gb,
                    train_cpu_memory_peak_gb=self.train_cpu_memory_peak_gb,
                    latest_eval_gpu_memory_peak_gb=self.latest_eval_gpu_memory_peak_gb,
                    latest_eval_gpu_memory_reserved_peak_gb=self.latest_eval_gpu_memory_reserved_peak_gb,
                    latest_eval_cpu_memory_peak_gb=self.latest_eval_cpu_memory_peak_gb,
                    latest_eval_archive_dir=self.latest_eval_archive_dir,
                    loss_curve_path=self.loss_curve_path,
                    lr_curve_path=self.lr_curve_path,
                    stopped_early_reason=str(exc),
                )
                self.update_summary(
                    status="completed",
                    train_loss=None if self.best_train_loss == float("inf") else self.best_train_loss,
                    val_loss=None if self.best_val_loss == float("inf") else self.best_val_loss,
                    epoch=completed_epoch,
                    stopped_early_reason=str(exc),
                )
                sync_run_localized_outputs(self.run_dir)
                return self.run_dir
            failed_seed_elapsed_seconds = round(time.perf_counter() - seed_started_perf, 4)
            self.seed_elapsed_seconds = failed_seed_elapsed_seconds
            self.seed_elapsed_hms = _format_duration(failed_seed_elapsed_seconds)
            self.write_status(
                state="failed",
                message=str(exc),
                epoch=0,
                global_step=self.global_step,
                total_optimizer_steps=self.total_optimizer_steps,
                seed_started_at=self.seed_started_at_iso,
                seed_elapsed_seconds=self.seed_elapsed_seconds,
                seed_elapsed_hms=self.seed_elapsed_hms,
            )
            self.update_summary(status="failed", train_loss=None, val_loss=None, epoch=0)
            sync_run_localized_outputs(self.run_dir)
            raise
        finally:
            if epoch_memory_monitor is not None:
                epoch_memory_monitor.stop()
            progress_bar.close()
            _release_torch_memory(self.accelerator.device)
