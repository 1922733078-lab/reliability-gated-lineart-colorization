from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainerConfig:
    dataset_root: str = "data/train"
    color_dir_name: str = "color"
    lineart_dir_name: str = "lineart"
    validation_dataset_root: str = "data/validation"
    validation_color_dir_name: str = "color"
    validation_lineart_dir_name: str = "lineart"
    prefer_external_validation_dataset: bool = True
    helper_tools_root: str = "tools/helper_metrics"
    output_root: str = "outputs"
    inference_archive_root: str = "artifacts/inference_archive"
    run_name: str = ""
    use_all_training_pairs_for_training: bool = True
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    image_width: int = 768
    image_height: int = 1024
    batch_size: int = 1
    epochs: int = 12
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.5e-5
    weight_decay: float = 1e-4
    optimizer_name: str = "adamw"
    lr_scheduler_name: str = "cosine"
    warmup_steps: int = 150
    enable_best_fid_lr_recovery: bool = True
    quality_decline_patience_epochs: int = 2
    quality_monitor_metric: str = "fid"
    enable_wgan_gp: bool = False
    wgan_critic_learning_rate: float = 5e-5
    wgan_critic_beta1: float = 0.0
    wgan_critic_beta2: float = 0.9
    wgan_critic_channels: int = 64
    wgan_critic_steps: int = 3
    wgan_start_epoch: int = 3
    wgan_warmup_epochs: int = 3
    wgan_balance_penalty_factor: float = 0.25
    wgan_gp_weight: float = 10.0
    wgan_generator_weight: float = 0.02
    wgan_reconstruction_weight: float = 0.25
    anti_flat_color_weight: float = 0.2
    anti_flat_color_min_std: float = 0.18
    latent_reconstruction_weight: float = 0.0
    latent_distribution_weight: float = 0.0
    latent_local_variance_weight: float = 0.0
    latent_gradient_weight: float = 0.0
    latent_global_flatness_weight: float = 0.0
    latent_global_flatness_target_std_ratio: float = 1.0
    latent_global_flatness_target_range_ratio: float = 1.0
    latent_local_variance_kernel_size: int = 3
    latent_local_variance_target_ratio: float = 0.95
    enable_wgan_memory_guard: bool = True
    wgan_memory_guard_ratio: float = 0.9
    collapse_guard_enabled: bool = True
    collapse_guard_adversarial_scale: float = 0.1
    collapse_guard_reconstruction_boost: float = 1.5
    collapse_guard_flat_boost: float = 2.0
    collapse_guard_cooldown_epochs: int = 1
    collapse_guard_color_bleeding_threshold: float = 0.85
    collapse_guard_ssim_threshold: float = 0.22
    collapse_guard_precision_threshold: float = 0.01
    collapse_guard_fid_ratio_threshold: float = 1.35
    critic_image_size: int = 256
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"
    num_workers: int = 2
    seed: int = 42
    seed_list: list[int] = field(default_factory=lambda: [42, 123, 456])
    checkpoint_interval_steps: int = 200
    save_every_epoch: bool = True
    eval_every_epoch: bool = True
    defer_generation_metrics_until_seed_end: bool = True
    max_eval_samples: int = 16
    preview_every_epoch: bool = True
    preview_samples: int = 2
    horizontal_flip: bool = True
    random_crop: bool = False
    color_jitter: bool = False
    final_eval_on_test: bool = True
    record_environment_lock: bool = True
    prompt_template: str = (
        "masterpiece, best quality, lineart colorization, harmonious colors, rich details, "
        "traditional art palette, clean cel shading"
    )
    negative_prompt: str = (
        "monochrome, grayscale, lowres, blurry, deformed, oversaturated, muddy colors, "
        "text, watermark, signature"
    )
    base_model: str = ""
    controlnet_model: str = ""
    controlnet_conditioning_scale: float = 1.0
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200
    fixed_threshold_value: float = 0.08
    adapter_channels: int = 32
    adapter_blocks: int = 2
    enable_adapter_variational_bottleneck: bool = True
    adapter_variational_latent_channels: int = 12
    adapter_variational_bottleneck_channels: int = 24
    adapter_variational_decoder_channels: int = 16
    adapter_variational_dropout: float = 0.05
    adapter_variational_logvar_min: float = -6.0
    adapter_variational_logvar_max: float = 2.0
    adapter_variational_reconstruction_weight: float = 0.08
    adapter_variational_beta_start: float = 0.0005
    adapter_variational_beta_end: float = 0.02
    adapter_variational_anneal_epochs: int = 4
    adapter_variational_free_bits: float = 0.02
    latent_snapshot_samples: int = 3
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_profile: str = "attention_only"
    enable_epoch_sanity_check: bool = True
    epoch_sanity_check_samples: int = 1
    epoch_sanity_check_inference_steps: int = 10
    epoch_sanity_check_guidance_scale: float = 5.5
    epoch_sanity_min_std: float = 18.0
    epoch_sanity_min_dynamic_range: float = 72.0
    epoch_sanity_min_gradient_mean: float = 1.25
    epoch_sanity_target_std_ratio: float = 0.35
    epoch_sanity_target_dynamic_range_ratio: float = 0.4
    epoch_sanity_target_gradient_ratio: float = 0.35
    epoch_sanity_warning_enabled: bool = True
    epoch_sanity_warning_min_ratio: float = 1.2
    epoch_sanity_warning_drop_ratio: float = 0.12
    epoch_sanity_recovery_lr_scale: float = 0.4
    epoch_sanity_recovery_severe_lr_scale: float = 0.3
    epoch_sanity_recovery_min_lr: float = 2e-6
    epoch_sanity_recovery_epochs: int = 3
    epoch_sanity_recovery_latent_reconstruction_weight: float = 0.06
    epoch_sanity_recovery_latent_distribution_weight: float = 0.05
    epoch_sanity_recovery_latent_local_variance_weight: float = 0.08
    epoch_sanity_recovery_latent_gradient_weight: float = 0.06
    epoch_sanity_recovery_latent_global_flatness_weight: float = 0.26
    epoch_sanity_recovery_image_reconstruction_weight: float = 0.008
    epoch_sanity_recovery_image_flat_color_weight: float = 0.03
    epoch_sanity_recovery_image_gradient_weight: float = 0.015
    epoch_sanity_recovery_weight_multiplier_step: float = 0.25
    epoch_sanity_recovery_max_weight_multiplier: float = 1.75
    stop_training_on_epoch_sanity_failure: bool = True
    enable_gradient_checkpointing: bool = True
    enable_xformers: bool = True
    cpu_offload: bool = False
    device: str = "cuda"


@dataclass
class InferenceConfig:
    prompt: str = (
        "masterpiece, best quality, lineart colorization, harmonious colors, rich details, "
        "traditional art palette, clean cel shading"
    )
    negative_prompt: str = (
        "monochrome, grayscale, lowres, blurry, deformed, oversaturated, muddy colors, "
        "text, watermark, signature"
    )
    num_inference_steps: int = 35
    guidance_scale: float = 6.5
    controlnet_scale: float = 1.1
    seed: int = -1
    width: int = 768
    height: int = 1024
    scheduler: str = "unipc"
    device: str = "cuda"
    dtype: str = "fp16"
    cpu_offload: bool = False
    enable_xformers: bool = False


@dataclass
class ExperimentConfig:
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        trainer = TrainerConfig(**payload.get("trainer", {}))
        inference = InferenceConfig(**payload.get("inference", {}))
        return cls(trainer=trainer, inference=inference)
