from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetPipeline,
    UniPCMultistepScheduler,
    UNet2DConditionModel,
)
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from .ablation import ModuleFlags
from .adapter import AdapterConfig, SXDLConditionAdapter
from .config import InferenceConfig, TrainerConfig
from .paths import resolve_repo_or_path


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    return torch.float16


def load_scheduler(base_model_path: str, scheduler_name: str):
    if scheduler_name == "euler_a":
        return EulerAncestralDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")
    if scheduler_name == "dpmpp_2m":
        base_scheduler = UniPCMultistepScheduler.from_pretrained(base_model_path, subfolder="scheduler")
        return DPMSolverMultistepScheduler.from_config(
            base_scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
    return UniPCMultistepScheduler.from_pretrained(base_model_path, subfolder="scheduler")


def enable_xformers_for_modules(enabled: bool, **modules: Any) -> dict[str, Any]:
    summary = {
        "requested": bool(enabled),
        "enabled": False,
        "status": "disabled_by_config",
        "enabled_modules": [],
        "failed_modules": {},
        "skipped_modules": [],
    }
    if not enabled:
        return summary

    enabled_modules: list[str] = []
    failed_modules: dict[str, str] = {}
    skipped_modules: list[str] = []
    for module_name, module in modules.items():
        if module is None:
            skipped_modules.append(str(module_name))
            continue
        enable_fn = getattr(module, "enable_xformers_memory_efficient_attention", None)
        if enable_fn is None:
            skipped_modules.append(str(module_name))
            continue
        try:
            enable_fn()
            enabled_modules.append(str(module_name))
        except Exception as exc:
            failed_modules[str(module_name)] = str(exc)

    summary["enabled_modules"] = enabled_modules
    summary["failed_modules"] = failed_modules
    summary["skipped_modules"] = skipped_modules
    summary["enabled"] = bool(enabled_modules)
    if enabled_modules and not failed_modules:
        summary["status"] = "enabled" if not skipped_modules else "enabled_with_skips"
    elif enabled_modules:
        summary["status"] = "partially_enabled"
    elif failed_modules:
        summary["status"] = "failed"
    else:
        summary["status"] = "unsupported"
    return summary


def maybe_enable_xformers(pipeline: StableDiffusionXLControlNetPipeline, enabled: bool) -> dict[str, Any]:
    return enable_xformers_for_modules(enabled, pipeline=pipeline)


def _resolve_lora_target_modules(profile: str) -> list[str]:
    normalized = str(profile or "").strip().lower()
    if normalized in {"extended", "full", "all"}:
        return [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "ff.net.0.proj",
            "ff.net.2.proj",
            "proj_in",
            "proj_out",
        ]
    if normalized in {"attention_plus_ff", "attn_ff"}:
        return [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "ff.net.0.proj",
            "ff.net.2.proj",
        ]
    return [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
    ]


def create_lora_unet(
    unet: UNet2DConditionModel,
    rank: int,
    alpha: int,
    *,
    dropout: float = 0.0,
    target_profile: str = "attention_only",
):
    target_modules = _resolve_lora_target_modules(target_profile)
    lora_config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        target_modules=target_modules,
        bias="none",
        lora_dropout=float(dropout),
    )
    return get_peft_model(unet, lora_config)


class ModelManager:
    def __init__(self, config: TrainerConfig | InferenceConfig, control_type: str = "scribble") -> None:
        self.config = config
        self.control_type = control_type

    def resolved_model_paths(self) -> tuple[str, str]:
        base = resolve_repo_or_path(getattr(self.config, "base_model", ""), "base")
        control_role = "canny" if "canny" in getattr(self.config, "controlnet_model", "").lower() else self.control_type
        control = resolve_repo_or_path(getattr(self.config, "controlnet_model", ""), control_role)
        return base, control

    def load_training_components(
        self,
        lora_rank: int,
        lora_alpha: int,
        dtype: torch.dtype,
        resume_lora_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        base_path, controlnet_path = self.resolved_model_paths()
        tokenizer = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", use_fast=False)
        tokenizer_2 = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", use_fast=False)
        text_encoder = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=dtype, use_safetensors=True)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            base_path,
            subfolder="text_encoder_2",
            torch_dtype=dtype,
            use_safetensors=True,
        )
        vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=dtype, use_safetensors=True)
        if hasattr(vae, "enable_vae_slicing"):
            vae.enable_vae_slicing()
        if hasattr(vae, "enable_vae_tiling"):
            vae.enable_vae_tiling()
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype, use_safetensors=True)
        unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=dtype, use_safetensors=True)
        xformers_state = enable_xformers_for_modules(
            bool(getattr(self.config, "enable_xformers", False)),
            vae=vae,
            controlnet=controlnet,
            unet=unet,
        )
        resume_lora_path = Path(resume_lora_dir) if resume_lora_dir else None
        if resume_lora_path is not None and resume_lora_path.exists():
            unet = PeftModel.from_pretrained(unet, str(resume_lora_path), is_trainable=True)
        else:
            unet = create_lora_unet(
                unet,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=float(getattr(self.config, "lora_dropout", 0.0)),
                target_profile=str(getattr(self.config, "lora_target_profile", "attention_only")),
            )
        scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
        return {
            "base_path": base_path,
            "controlnet_path": controlnet_path,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "vae": vae,
            "controlnet": controlnet,
            "unet": unet,
            "noise_scheduler": scheduler,
            "xformers_state": xformers_state,
        }


def load_run_metadata(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class InferenceEngine:
    def __init__(
        self,
        run_dir: str | Path,
        checkpoint_dir: str | Path | None = None,
        scheduler_name: str = "unipc",
        device: str = "cuda",
        dtype: str = "fp16",
        cpu_offload: bool = False,
        enable_xformers: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.metadata = load_run_metadata(self.run_dir)
        if not self.metadata:
            raise FileNotFoundError(f"run_metadata.json not found in {self.run_dir}")
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        trainer_cfg = TrainerConfig(**self.metadata["trainer_config"])
        adapter_cfg = AdapterConfig(**self.metadata["adapter_config"])
        flags = ModuleFlags(**self.metadata["flags"])
        manager = ModelManager(trainer_cfg, control_type=_infer_control_type(trainer_cfg.controlnet_model))

        self.base_model_path, self.controlnet_model_path = manager.resolved_model_paths()
        weight_dtype = resolve_dtype(dtype)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, subfolder="tokenizer", use_fast=False)
        tokenizer_2 = AutoTokenizer.from_pretrained(self.base_model_path, subfolder="tokenizer_2", use_fast=False)
        text_encoder = CLIPTextModel.from_pretrained(
            self.base_model_path,
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            use_safetensors=True,
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.base_model_path,
            subfolder="text_encoder_2",
            torch_dtype=weight_dtype,
            use_safetensors=True,
        )
        vae = AutoencoderKL.from_pretrained(self.base_model_path, subfolder="vae", torch_dtype=weight_dtype, use_safetensors=True)
        unet = UNet2DConditionModel.from_pretrained(
            self.base_model_path,
            subfolder="unet",
            torch_dtype=weight_dtype,
            use_safetensors=True,
        )
        lora_dir = _resolve_adapter_dir(self.run_dir, self.checkpoint_dir)
        if lora_dir is not None:
            unet = PeftModel.from_pretrained(unet, str(lora_dir))

        controlnet = ControlNetModel.from_pretrained(self.controlnet_model_path, torch_dtype=weight_dtype, use_safetensors=True)
        scheduler = load_scheduler(self.base_model_path, scheduler_name)
        self.pipeline = StableDiffusionXLControlNetPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            add_watermarker=False,
        )
        self.pipeline.enable_vae_slicing()
        if hasattr(self.pipeline, "enable_vae_tiling"):
            self.pipeline.enable_vae_tiling()
        self.pipeline.enable_attention_slicing()
        maybe_enable_xformers(self.pipeline, enable_xformers)
        if cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline = self.pipeline.to(device)

        self.device = device
        self.adapter = SXDLConditionAdapter(adapter_cfg, flags)
        adapter_path = _resolve_adapter_file(self.run_dir, self.checkpoint_dir)
        self.adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"))
        self.adapter.eval().to(device)

        self.controlnet_name = trainer_cfg.controlnet_model or self.controlnet_model_path
        self.inference_defaults = self.metadata.get("inference_defaults", asdict(InferenceConfig()))

    @torch.no_grad()
    def colorize(
        self,
        lineart_image: np.ndarray,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        controlnet_scale: float,
        seed: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        width = max(64, (int(width) // 8) * 8)
        height = max(64, (int(height) // 8) * 8)
        if lineart_image.ndim == 2:
            control_rgb = np.repeat(lineart_image[:, :, None], 3, axis=2)
        elif lineart_image.shape[2] == 4:
            control_rgb = cv2.cvtColor(lineart_image, cv2.COLOR_RGBA2RGB)
        else:
            control_rgb = lineart_image[:, :, :3]
        control_rgb = cv2.resize(control_rgb.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        control_tensor = torch.from_numpy(control_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        control_tensor = control_tensor.to(self.device)
        adapted = self.adapter(control_tensor).clamp(0.0, 1.0)
        control_pil = Image.fromarray((adapted.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))

        if seed < 0:
            generator = None
        else:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_pil,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(controlnet_scale),
            generator=generator,
            width=width,
            height=height,
        ).images[0]
        result_np = np.array(result)
        return result_np


def _infer_control_type(controlnet_model: str) -> str:
    return "canny" if "canny" in (controlnet_model or "").lower() else "scribble"


def _resolve_adapter_dir(run_dir: Path, checkpoint_dir: Path | None) -> Path | None:
    if checkpoint_dir is not None:
        lora_dir = checkpoint_dir / "lora"
        if lora_dir.exists():
            return lora_dir
    final_dir = run_dir / "lora"
    if final_dir.exists():
        return final_dir
    candidates = sorted((run_dir / "checkpoints").glob("step_*"), key=lambda item: item.name)
    for candidate in reversed(candidates):
        lora_dir = candidate / "lora"
        if lora_dir.exists():
            return lora_dir
    return None


def _resolve_adapter_file(run_dir: Path, checkpoint_dir: Path | None) -> Path:
    if checkpoint_dir is not None and (checkpoint_dir / "adapter.pt").exists():
        return checkpoint_dir / "adapter.pt"
    if (run_dir / "adapter.pt").exists():
        return run_dir / "adapter.pt"
    candidates = sorted((run_dir / "checkpoints").glob("step_*"), key=lambda item: item.name)
    for candidate in reversed(candidates):
        adapter_path = candidate / "adapter.pt"
        if adapter_path.exists():
            return adapter_path
    raise FileNotFoundError(f"No adapter.pt found under {run_dir}")
