from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.utils import get_column_letter


FILE_NAME_MAP: dict[str, str] = {
    "run_summary.json": "运行摘要.json",
    "train_status.json": "训练状态.json",
    "run_metadata.json": "运行元数据.json",
    "dataset_split.json": "数据集划分.json",
    "validation_selection.json": "验证样本选择.json",
    "epoch_history.json": "轮次历史.json",
    "checkpoint.json": "检查点信息.json",
    "metrics.json": "评估指标.json",
    "per_sample_metrics.csv": "逐图评估指标.csv",
    "pr_curve_points.csv": "PR曲线数据.csv",
    "subgroup_metrics.json": "分组指标.json",
    "helper_tool_metrics.json": "辅助工具指标.json",
    "archive_manifest.json": "归档清单.json",
    "train.jsonl": "训练步日志.jsonl",
    "metrics.jsonl": "轮次指标日志.jsonl",
    "last_batch_report.csv": "最后一批报告.csv",
    "analysis_summary.json": "分析汇总.json",
    "dashboard_summary.json": "训练仪表盘摘要.json",
    "per_run_summary.csv": "单次运行汇总.csv",
    "group_summary.csv": "实验组汇总.csv",
    "seed_average_summary.csv": "Seed平均汇总.csv",
    "best_checkpoints.csv": "最优检查点汇总.csv",
    "single_module_contributions.csv": "单模块贡献.csv",
    "interaction_effects.csv": "交互效应.csv",
    "paired_t_tests.csv": "配对T检验.csv",
    "train_step_logs.csv": "训练步日志.csv",
    "epoch_metric_logs.csv": "轮次指标日志.csv",
    "per_sample_metrics_all.csv": "逐图评估总表.csv",
    "experiment_summary.xlsx": "实验汇总.xlsx",
    "experiment_config.lock.json": "实验配置锁定.json",
    "latent_step_dashboard.png": "潜变量步级仪表盘.png",
    "latent_epoch_dashboard.png": "潜变量轮次仪表盘.png",
    "latent_snapshot_manifest.json": "潜变量快照清单.json",
}


FILE_DESCRIPTION_MAP: dict[str, str] = {
    "run_summary.json": "单次实验运行摘要，集中记录当前 seed 的最佳结果、路径、资源占用和曲线文件位置。",
    "train_status.json": "训练状态快照，适合在训练过程中查看当前 epoch、step、学习率和最近一次评估结果。",
    "run_metadata.json": "运行元数据，记录实验组配置、模块开关、训练配置和默认推理参数。",
    "dataset_split.json": "数据划分结果，记录训练集、验证集、测试集的样本归属和划分元信息。",
    "validation_selection.json": "验证样本选择记录，说明本次训练用哪些样本做指标验证和预览。",
    "epoch_history.json": "按 epoch 汇总的历史记录，包含训练损失、验证损失、FID、LPIPS、SSIM、耗时和显存等信息。",
    "checkpoint.json": "单个检查点的保存信息，记录保存时的 epoch、step 和时间。",
    "metrics.json": "单次评估汇总结果，记录该评估目录下的主指标、显存占用、样本路径和辅助工具结果。",
    "per_sample_metrics.csv": "逐张图片的评估结果，每一行对应一张验证图片的生成效果指标。",
    "pr_curve_points.csv": "PR 曲线数据点，每一行记录一个邻域参数下的 Precision、Recall 和 F-score。",
    "subgroup_metrics.json": "按线稿密度、颜色复杂度、区域尺度分组后的指标均值，用于消融细分分析。",
    "helper_tool_metrics.json": "辅助工具计算得到的 LPIPS、SSIM、FID 结果及其报告路径。",
    "archive_manifest.json": "推理归档清单，说明归档目录来源、归档类型、所属实验组和 seed 信息。",
    "train.jsonl": "训练 step 日志，每一行是一条 step 级记录，包含 loss 与学习率。",
    "metrics.jsonl": "训练过程中的 epoch 指标日志，每一行是一条 epoch 级记录。",
    "last_batch_report.csv": "辅助工具对最近一批验证图片计算得到的原始逐图报告。",
    "analysis_summary.json": "分析阶段的汇总结论，记录单模块贡献、交互效应和主要输出文件路径。",
    "dashboard_summary.json": "训练实时仪表盘摘要，集中记录当前 loss、KL、z 范数以及关联图表路径。",
    "per_run_summary.csv": "单次运行汇总表，每行对应一个 group + seed 的最终汇总结果。",
    "group_summary.csv": "实验组汇总表，按实验组聚合多个 seed 的均值和标准差。",
    "seed_average_summary.csv": "Seed 平均汇总表，便于直接引用组级均值结果。",
    "best_checkpoints.csv": "最优检查点表，集中给出每个运行的 best_fid 和 best_val_loss 检查点位置。",
    "single_module_contributions.csv": "单模块贡献分析结果，用于衡量各模块单独带来的效果变化。",
    "interaction_effects.csv": "模块交互效应分析结果，用于衡量双模块联合作用。",
    "paired_t_tests.csv": "配对 T 检验结果，比较 E_FULL 与其他实验组在相同 seed 下的差异显著性。",
    "train_step_logs.csv": "汇总后的训练 step 日志表。",
    "epoch_metric_logs.csv": "汇总后的 epoch 指标日志表。",
    "per_sample_metrics_all.csv": "汇总后的逐图评估总表，聚合所有运行和所有评估目录下的逐图结果。",
    "experiment_summary.xlsx": "面向人工查看的实验总表工作簿，包含所有关键结果页和字段说明页。",
    "experiment_config.lock.json": "实验环境与超参数锁定文件，用于保证不同实验组使用一致的环境与核心配置。",
    "latent_step_dashboard.png": "按 step 展示潜变量相关指标的仪表盘图。",
    "latent_epoch_dashboard.png": "按 epoch 展示潜变量相关指标的仪表盘图。",
    "latent_snapshot_manifest.json": "每个 epoch 保存的潜变量快照和统计汇总清单。",
}


EXACT_FIELD_NAME_MAP: dict[str, str] = {
    "status": "状态",
    "state": "运行状态",
    "message": "消息",
    "group_id": "实验组编号",
    "group_name": "实验组名称",
    "display_name": "显示名称",
    "description": "描述",
    "category": "类别",
    "seed": "随机种子",
    "seeds": "种子列表",
    "seed_count": "种子数量",
    "run_dir": "运行目录",
    "source_file": "源文件",
    "created_at": "创建时间",
    "updated_at": "更新时间",
    "saved_at": "保存时间",
    "timestamp": "时间戳",
    "event": "事件",
    "epoch": "轮次",
    "global_step": "全局步数",
    "total_optimizer_steps": "总优化步数",
    "step": "步数",
    "name": "名称",
    "code": "编码",
    "group": "实验组配置",
    "environment": "环境信息",
    "hyperparameters": "超参数",
    "data": "数据配置",
    "flags": "模块开关",
    "trainer_config": "训练配置",
    "adapter_config": "适配器配置",
    "inference_defaults": "默认推理配置",
    "best_train_loss": "最佳训练损失",
    "best_val_loss": "最佳验证损失",
    "best_fid": "最佳FID",
    "best_fid_epoch": "最佳FID所在轮次",
    "best_fid_checkpoint_path": "最佳FID检查点路径",
    "best_fid_preview_path": "最佳FID预览图路径",
    "best_fid_eval_dir": "最佳FID评估目录",
    "best_fid_archive_dir": "最佳FID归档目录",
    "best_fid_learning_rate": "最佳FID学习率",
    "best_val_loss_checkpoint_path": "最佳验证损失检查点路径",
    "latest_preview_path": "最新预览图路径",
    "latest_train_loss": "最新训练损失",
    "latest_val_loss": "最新验证损失",
    "environment_lock_path": "环境锁定文件路径",
    "epoch_history_path": "轮次历史路径",
    "validation_source": "验证来源",
    "preview_source": "预览来源",
    "validation_note": "验证说明",
    "validation_selection_path": "验证样本选择路径",
    "params_m": "参数量(M)",
    "flops_g": "FLOPs(G)",
    "train_gpu_memory_peak_gb": "训练GPU峰值显存(GB)",
    "train_gpu_memory_reserved_peak_gb": "训练GPU预留峰值显存(GB)",
    "train_cpu_memory_peak_gb": "训练CPU峰值内存(GB)",
    "latest_eval_gpu_memory_peak_gb": "最近评估GPU峰值显存(GB)",
    "latest_eval_gpu_memory_reserved_peak_gb": "最近评估GPU预留峰值显存(GB)",
    "latest_eval_cpu_memory_peak_gb": "最近评估CPU峰值内存(GB)",
    "latest_eval_archive_dir": "最近评估归档目录",
    "latest_epoch_time_seconds": "最近轮次耗时(秒)",
    "latest_epoch_time_hms": "最近轮次耗时(时分秒)",
    "seed_started_at": "当前Seed开始时间",
    "seed_elapsed_seconds": "当前Seed累计耗时(秒)",
    "seed_elapsed_hms": "当前Seed累计耗时(时分秒)",
    "loss_curve_path": "Loss曲线路径",
    "lr_curve_path": "学习率曲线路径",
    "dashboard_summary_path": "训练仪表盘摘要路径",
    "latent_step_dashboard_path": "潜变量步级仪表盘路径",
    "latent_epoch_dashboard_path": "潜变量轮次仪表盘路径",
    "latest_latent_snapshot_dir": "最近潜变量快照目录",
    "metric_records": "指标评估样本",
    "preview_records": "预览样本",
    "source": "来源",
    "has_reference": "是否有参考图",
    "external_root": "外部数据根目录",
    "external_paired_count": "外部成对样本数",
    "external_lineart_only_count": "仅线稿样本数",
    "fallback_count": "回退样本数",
    "note": "备注",
    "train": "训练集",
    "val": "验证集",
    "test": "测试集",
    "_metadata": "元数据",
    "loss": "损失",
    "lr": "学习率",
    "learning_rate": "学习率",
    "train_loss": "训练损失",
    "val_loss": "验证损失",
    "fid": "FID",
    "precision": "Precision",
    "recall": "Recall",
    "f_score": "F-score",
    "pr_curve_auc": "PR曲线面积",
    "pr_curve_csv": "PR曲线数据路径",
    "pr_curve_plot": "PR曲线路径",
    "pr_metrics_error": "PR指标错误",
    "lpips": "LPIPS",
    "ssim": "SSIM",
    "edge_consistency": "边缘一致性",
    "edge_consistency_f1": "边缘一致性F1",
    "color_bleeding_rate": "颜色渗漏率",
    "histogram_correlation": "直方图相关性",
    "inference_time_ms": "推理耗时(ms)",
    "subgroup_metrics": "分组指标",
    "per_sample_rows": "逐图记录",
    "eval_dir": "评估目录",
    "generated_dir": "生成图目录",
    "target_dir": "目标图目录",
    "lineart_dir": "线稿图目录",
    "checkpoint_dir": "检查点目录",
    "archive_dir": "归档目录",
    "archive_kind": "归档类型",
    "source_eval_dir": "源评估目录",
    "split_name": "划分名称",
    "checkpoint_label": "检查点标签",
    "archived_at": "归档时间",
    "fid_computed": "是否计算FID",
    "gpu_memory_peak_gb": "GPU峰值显存(GB)",
    "gpu_memory_reserved_peak_gb": "GPU预留峰值显存(GB)",
    "cpu_memory_peak_gb": "CPU峰值内存(GB)",
    "memory_unit": "内存单位",
    "gpu_memory_peak": "GPU峰值显存",
    "cpu_memory_peak": "CPU峰值内存",
    "helper_metrics_available": "辅助指标是否可用",
    "helper_metric_reports": "辅助指标报告",
    "helper_metric_errors": "辅助指标错误",
    "fid_internal": "内部FID",
    "fid_source": "FID来源",
    "lpips_internal": "内部LPIPS",
    "lpips_mean": "LPIPS均值",
    "lpips_std": "LPIPS标准差",
    "lpips_source": "LPIPS来源",
    "ssim_internal": "内部SSIM",
    "ssim_mean": "SSIM均值",
    "ssim_std": "SSIM标准差",
    "ssim_source": "SSIM来源",
    "split": "数据划分",
    "num_samples": "样本数",
    "edge_consistency_mean": "边缘一致性均值",
    "edge_consistency_std": "边缘一致性标准差",
    "color_bleeding_rate_mean": "颜色渗漏率均值",
    "color_bleeding_rate_std": "颜色渗漏率标准差",
    "histogram_correlation_mean": "直方图相关性均值",
    "histogram_correlation_std": "直方图相关性标准差",
    "inference_time_ms_mean": "推理耗时均值(ms)",
    "inference_time_ms_std": "推理耗时标准差(ms)",
    "image_id": "图像编号",
    "generated_path": "生成图路径",
    "target_path": "目标图路径",
    "lineart_path": "线稿图路径",
    "per_sample_metrics_path": "逐图评估表路径",
    "eval_split": "评估划分",
    "selected_metrics_path": "选用指标文件路径",
    "line_density": "线稿密度",
    "color_complexity": "颜色复杂度",
    "region_scale": "区域尺度",
    "line_density_group": "线稿密度分组",
    "color_complexity_group": "颜色复杂度分组",
    "region_scale_group": "区域尺度分组",
    "loss_curve_smoothness": "Loss曲线平滑度",
    "complexity_performance_gap": "复杂度性能差距",
    "scale_performance_gap": "尺度性能差距",
    "quality_monitor_metric": "质量监控指标",
    "consecutive_quality_decline_epochs": "连续质量下降轮数",
    "best_fid_lr_recovery_count": "最佳FID学习率恢复次数",
    "best_fid_lr_recovery_active": "最佳FID学习率恢复是否激活",
    "latest_quality_recovery_triggered": "本轮是否触发学习率恢复",
    "enable_wgan_gp": "启用WGAN-GP",
    "wgan_critic_learning_rate": "WGAN判别器学习率",
    "wgan_critic_beta1": "WGAN判别器Beta1",
    "wgan_critic_beta2": "WGAN判别器Beta2",
    "wgan_critic_channels": "WGAN判别器通道数",
    "wgan_critic_steps": "WGAN判别器步数",
    "wgan_start_epoch": "WGAN启用起始轮次",
    "wgan_warmup_epochs": "WGAN对抗预热轮数",
    "wgan_balance_penalty_factor": "判别器失衡时对抗降权系数",
    "wgan_gp_weight": "WGAN梯度惩罚权重",
    "wgan_generator_weight": "WGAN生成器对抗权重",
    "wgan_reconstruction_weight": "生成图L1重建权重",
    "anti_flat_color_weight": "抗纯色塌缩权重",
    "anti_flat_color_min_std": "最小颜色标准差阈值",
    "latent_reconstruction_weight": "潜空间重建权重",
    "latent_distribution_weight": "潜空间分布约束权重",
    "latent_local_variance_weight": "潜空间局部方差约束权重",
    "latent_local_variance_kernel_size": "潜空间局部方差窗口大小",
    "latent_local_variance_target_ratio": "潜空间局部方差目标比例",
    "collapse_guard_enabled": "启用坍塌保护",
    "collapse_guard_adversarial_scale": "坍塌保护时对抗缩放系数",
    "collapse_guard_reconstruction_boost": "坍塌保护时重建增强系数",
    "collapse_guard_flat_boost": "坍塌保护时纯色惩罚增强系数",
    "collapse_guard_cooldown_epochs": "坍塌保护冷却轮数",
    "collapse_guard_color_bleeding_threshold": "坍塌保护颜色渗漏阈值",
    "collapse_guard_ssim_threshold": "坍塌保护SSIM阈值",
    "collapse_guard_precision_threshold": "坍塌保护Precision阈值",
    "collapse_guard_fid_ratio_threshold": "坍塌保护FID倍数阈值",
    "adversarial_cooldown_until_epoch": "对抗冷却截止轮次",
    "collapse_guard_triggered": "是否触发坍塌保护",
    "collapse_guard_reason": "坍塌保护触发原因",
    "effective_adversarial_weight": "实际对抗权重",
    "effective_reconstruction_weight": "实际重建权重",
    "effective_flat_color_weight": "实际纯色惩罚权重",
    "decoded_image_spatial_std": "生成图空间标准差",
    "critic_image_size": "判别器图像尺寸",
    "diffusion_loss": "扩散损失",
    "adversarial_loss": "对抗损失",
    "reconstruction_l1_loss": "重建L1损失",
    "flat_color_penalty": "纯色塌缩惩罚",
    "latent_reconstruction_loss": "潜空间重建损失",
    "latent_distribution_loss": "潜空间分布约束损失",
    "latent_local_variance_loss": "潜空间局部方差约束损失",
    "latent_gradient_loss": "潜空间梯度一致性损失",
    "latent_global_flatness_loss": "潜空间全局抗平坦损失",
    "image_gradient_loss": "图像梯度一致性损失",
    "adapter_bottleneck_reconstruction_loss": "适配器瓶颈重建损失",
    "beta_vae_kl_loss": "Beta-VAE KL损失",
    "beta_vae_kl_raw_loss": "Beta-VAE原始KL损失",
    "beta_vae_beta": "Beta-VAE当前Beta",
    "kl_per_dim_mean": "每维KL均值",
    "kl_per_dim_max": "每维KL最大值",
    "free_bits_active_fraction": "自由比特激活比例",
    "z_norm_l2_mean": "潜变量L2范数均值",
    "z_std_mean": "潜变量标准差均值",
    "posterior_mu_abs_mean": "后验mu绝对值均值",
    "posterior_logvar_mean": "后验logvar均值",
    "latent_snapshot_dir": "潜变量快照目录",
    "latent_snapshot_manifest_path": "潜变量快照清单路径",
    "latent_snapshot_samples": "潜变量快照样本数",
    "latent_npz_path": "潜变量NPZ路径",
    "snapshot_path": "快照图路径",
    "critic_loss": "判别器损失",
    "gradient_penalty": "梯度惩罚",
    "critic_real_score": "判别器真实图得分",
    "critic_fake_score": "判别器生成图得分",
    "latest_diffusion_loss": "最新扩散损失",
    "latest_adversarial_loss": "最新对抗损失",
    "latest_reconstruction_l1_loss": "最新重建L1损失",
    "latest_flat_color_penalty": "最新纯色塌缩惩罚",
    "latest_critic_loss": "最新判别器损失",
    "latest_gradient_penalty": "最新梯度惩罚",
    "latest_precision": "最新Precision",
    "latest_recall": "最新Recall",
    "latest_f_score": "最新F-score",
    "latest_pr_curve_auc": "最新PR曲线面积",
    "optimizer": "优化器",
    "optimizer_name": "优化器",
    "weight_decay": "权重衰减",
    "lr_scheduler": "学习率调度器",
    "lr_scheduler_name": "学习率调度器",
    "warmup_steps": "预热步数",
    "batch_size": "批大小",
    "gradient_clip": "梯度裁剪阈值",
    "gradient_accumulation_steps": "梯度累积步数",
    "epochs": "训练轮数",
    "total_epochs": "总训练轮数",
    "max_grad_norm": "最大梯度范数",
    "ema_decay": "EMA衰减",
    "dataset": "数据集名称",
    "dataset_root": "训练数据根目录",
    "validation_dataset_root": "验证数据根目录",
    "color_dir_name": "彩色图目录名",
    "lineart_dir_name": "线稿目录名",
    "validation_color_dir_name": "验证彩色图目录名",
    "validation_lineart_dir_name": "验证线稿目录名",
    "train_split": "训练集比例",
    "val_split": "验证集比例",
    "test_split": "测试集比例",
    "train_ratio": "训练集比例",
    "val_ratio": "验证集比例",
    "test_ratio": "测试集比例",
    "split_seed": "划分随机种子",
    "augmentation": "数据增强",
    "horizontal_flip": "水平翻转增强",
    "random_crop": "随机裁剪增强",
    "color_jitter": "颜色扰动增强",
    "image_width": "图像宽度",
    "image_height": "图像高度",
    "mixed_precision": "混合精度",
    "effective_training_precision": "实际训练精度",
    "device": "设备",
    "cpu_offload": "CPU卸载",
    "enable_xformers": "启用xFormers",
    "record_environment_lock": "记录环境锁",
    "eval_every_epoch": "每轮评估",
    "preview_every_epoch": "每轮预览",
    "save_every_epoch": "每轮保存检查点",
    "checkpoint_interval_steps": "检查点步数间隔",
    "output_root": "输出根目录",
    "run_name": "运行名称",
    "prompt_template": "提示词模板",
    "negative_prompt": "负向提示词",
    "controlnet_conditioning_scale": "ControlNet条件强度",
    "controlnet_version": "ControlNet版本",
    "controlnet_model": "ControlNet模型",
    "adapter_channels": "适配器通道数",
    "adapter_blocks": "适配器块数",
    "enable_adapter_variational_bottleneck": "启用适配器变分瓶颈",
    "adapter_variational_latent_channels": "适配器变分潜变量通道数",
    "adapter_variational_bottleneck_channels": "适配器变分瓶颈通道数",
    "adapter_variational_decoder_channels": "适配器低容量解码器通道数",
    "adapter_variational_dropout": "适配器变分瓶颈Dropout",
    "adapter_variational_logvar_min": "适配器变分logvar下限",
    "adapter_variational_logvar_max": "适配器变分logvar上限",
    "adapter_variational_reconstruction_weight": "适配器变分重建权重",
    "adapter_variational_beta_start": "适配器KL退火起始Beta",
    "adapter_variational_beta_end": "适配器KL退火结束Beta",
    "adapter_variational_anneal_epochs": "适配器KL退火轮数",
    "adapter_variational_free_bits": "适配器自由比特阈值",
    "latent_snapshot_samples": "每轮潜变量快照样本数",
    "fixed_threshold_value": "固定阈值",
    "lora_rank": "LoRA秩",
    "lora_alpha": "LoRA Alpha",
    "helper_tools_root": "辅助工具根目录",
    "inference_archive_root": "推理归档根目录",
    "max_eval_samples": "最大评估样本数",
    "num_workers": "数据加载线程数",
    "enable_gradient_checkpointing": "启用梯度检查点",
    "prefer_external_validation_dataset": "优先使用外部验证集",
    "use_all_training_pairs_for_training": "训练数据全部用于训练",
    "final_eval_on_test": "训练结束后在测试集评估",
    "seed_list": "种子列表",
    "enable_best_fid_lr_recovery": "启用最佳FID学习率恢复",
    "quality_decline_patience_epochs": "质量下降容忍轮数",
    "gpu": "GPU型号",
    "gpu_memory_gb": "GPU显存(GB)",
    "cuda": "CUDA版本",
    "python": "Python版本",
    "torch": "PyTorch版本",
    "diffusers": "Diffusers版本",
    "platform": "平台信息",
    "analysis_dir": "分析目录",
    "analysis_summary_json": "分析汇总JSON路径",
    "per_run_summary": "单次运行汇总路径",
    "group_summary": "实验组汇总路径",
    "seed_average_summary": "Seed平均汇总路径",
    "best_checkpoints": "最优检查点路径",
    "train_step_logs": "训练步日志路径",
    "epoch_metric_logs": "轮次指标日志路径",
    "per_sample_metrics_all": "逐图评估总表路径",
    "single_module_contributions": "单模块贡献路径",
    "interaction_effects": "交互效应路径",
    "paired_t_tests": "配对T检验路径",
    "excel_workbook": "实验总表路径",
    "experiment_summary_chinese": "中文实验总表路径",
    "summary_json": "汇总JSON路径",
    "lpips_csv": "LPIPS报告CSV路径",
    "ssim_csv": "SSIM报告CSV路径",
    "eval_gpu_memory_peak_gb": "评估GPU峰值显存(GB)",
    "eval_gpu_memory_reserved_peak_gb": "评估GPU预留峰值显存(GB)",
    "eval_cpu_memory_peak_gb": "评估CPU峰值内存(GB)",
    "dynamic_weight": "动态权重",
    "adaptive_threshold": "自适应阈值",
    "adaptive_rf": "自适应感受野",
    "adaptive_norm": "自适应归一化",
    "available": "是否可用",
    "errors": "错误信息",
    "reports": "报告文件",
    "raw_outputs": "原始输出",
    "single_module_contributions": "单模块贡献",
    "interaction_effects": "交互效应",
    "report_files": "报告文件列表",
    "section": "分区",
    "item": "项目",
    "value": "值",
    "module": "模块",
    "contribution": "贡献值",
    "module_pair": "模块组合",
    "interaction": "交互效应值",
    "metric": "指标",
    "neighborhood": "邻域参数",
    "t_statistic": "T统计量",
    "p_value": "P值",
    "n": "样本数",
}


TOKEN_NAME_MAP: dict[str, str] = {
    "best": "最佳",
    "latest": "最新",
    "train": "训练",
    "val": "验证",
    "test": "测试",
    "eval": "评估",
    "epoch": "轮次",
    "step": "步",
    "loss": "损失",
    "fid": "FID",
    "lpips": "LPIPS",
    "ssim": "SSIM",
    "mean": "均值",
    "std": "标准差",
    "gpu": "GPU",
    "cpu": "CPU",
    "memory": "内存",
    "peak": "峰值",
    "reserved": "预留",
    "time": "耗时",
    "seconds": "秒",
    "hms": "时分秒",
    "learning": "学习",
    "rate": "率",
    "quality": "质量",
    "monitor": "监控",
    "recovery": "恢复",
    "count": "数量",
    "source": "来源",
    "path": "路径",
    "dir": "目录",
    "archive": "归档",
    "preview": "预览",
    "generated": "生成图",
    "target": "目标图",
    "lineart": "线稿图",
    "split": "划分",
    "image": "图像",
    "id": "编号",
    "file": "文件",
    "summary": "汇总",
    "selected": "选用",
    "seed": "Seed",
    "group": "实验组",
    "name": "名称",
    "status": "状态",
    "metric": "指标",
}


VALUE_TRANSLATION_MAP: dict[str, str] = {
    "completed": "已完成",
    "running": "运行中",
    "failed": "失败",
    "initializing": "初始化中",
    "Training completed": "训练已完成",
    "Training started": "训练已开始",
    "Training in progress": "训练进行中",
    "Preparing models and dataloaders": "正在准备模型与数据加载器",
    "train": "训练",
    "epoch_end": "轮次结束",
    "best_fid_lr_recovery_triggered": "触发最佳FID学习率恢复",
    "skipped_non_finite_train_loss": "跳过非有限训练损失",
    "external_paired_validation": "外部成对验证集",
    "external_lineart_only": "外部仅线稿验证集",
    "split_val": "划分验证集",
    "split_test": "划分测试集",
    "training_validation": "训练过程验证",
    "standalone_evaluation": "独立评估推理",
    "helper_tool": "辅助工具",
    "internal": "内部计算",
    "GB": "GB",
    "Baseline": "基线组",
    "Full": "完整模块组",
    "baseline": "基线实验",
    "full": "完整实验",
    "single": "单模块消融",
    "pair": "双模块消融",
    "sparse_lineart": "稀疏线稿",
    "dense_lineart": "密集线稿",
    "simple_color": "简单配色",
    "complex_color": "复杂配色",
    "small_region": "小区域",
    "large_region": "大区域",
    "Using paired external validation dataset for LPIPS/SSIM/FID and other generation metrics.": "当前使用外部成对验证集来计算 LPIPS、SSIM、FID 及其他生成质量指标。",
}


EXACT_FIELD_EXPLANATION_MAP: dict[str, str] = {
    "status": "当前运行的最终状态或最近状态。",
    "state": "训练器当前处于的阶段状态。",
    "group_id": "实验组编号，例如 E0、E_FULL。",
    "group_name": "实验组的人类可读名称。",
    "seed": "本次重复实验使用的随机种子。",
    "best_train_loss": "训练过程中观测到的最小训练损失。",
    "best_val_loss": "验证过程中观测到的最小验证损失。",
    "best_fid": "训练或评估过程中得到的最优 FID，越低越好。",
    "best_fid_epoch": "取得最佳 FID 时对应的 epoch 编号。",
    "best_fid_learning_rate": "取得最佳 FID 时记录下来的学习率，用于质量回退后的恢复训练。",
    "best_fid_lr_recovery_count": "因连续质量下降而回退到最佳 FID 学习率的累计次数。",
    "best_fid_lr_recovery_active": "当前是否处于按最佳 FID 学习率继续训练的恢复模式。",
    "latest_quality_recovery_triggered": "当前这轮是否刚刚触发了学习率恢复。",
    "adversarial_cooldown_until_epoch": "当前对抗训练冷却会持续到哪个 epoch 为止。",
    "collapse_guard_triggered": "当前轮次是否检测到了明显的坍塌风险并触发保护。",
    "collapse_guard_reason": "触发坍塌保护的具体指标原因说明。",
    "quality_monitor_metric": "用于判断质量是否连续下降的监控指标。",
    "consecutive_quality_decline_epochs": "连续发生质量下降的 epoch 数。",
    "neighborhood": "PR 指标计算时使用的邻域大小参数。",
    "params_m": "可训练参数量，单位为百万参数。",
    "flops_g": "估算计算量，单位为 GFLOPs。",
    "train_gpu_memory_peak_gb": "整个训练过程中 GPU 峰值显存占用，单位 GB。",
    "train_gpu_memory_reserved_peak_gb": "整个训练过程中 GPU 预留峰值显存，单位 GB。",
    "train_cpu_memory_peak_gb": "整个训练过程中 CPU 峰值内存占用，单位 GB。",
    "latest_eval_gpu_memory_peak_gb": "最近一次评估时 GPU 峰值显存占用，单位 GB。",
    "latest_eval_gpu_memory_reserved_peak_gb": "最近一次评估时 GPU 预留峰值显存，单位 GB。",
    "latest_eval_cpu_memory_peak_gb": "最近一次评估时 CPU 峰值内存占用，单位 GB。",
    "latest_epoch_time_seconds": "最近完成的一个 epoch 的耗时，单位秒。",
    "seed_elapsed_seconds": "当前 seed 从开始训练到当前时刻的累计耗时，单位秒。",
    "latest_epoch_time_hms": "最近完成的一个 epoch 的耗时，格式为时:分:秒。",
    "seed_elapsed_hms": "当前 seed 的累计耗时，格式为时:分:秒。",
    "fid": "Fréchet Inception Distance，衡量生成图与真实图分布差异，越低越好。",
    "precision": "生成样本落在真实样本流形内的比例，越高越好。",
    "recall": "真实样本被生成样本流形覆盖到的比例，越高越好。",
    "f_score": "Precision 与 Recall 的调和平均，越高越好。",
    "pr_curve_auc": "PR 曲线下面积，用于综合衡量分布覆盖与样本保真度，越高越好。",
    "pr_metrics_error": "PR 指标计算失败时记录的错误信息。",
    "lpips": "LPIPS 感知距离，衡量感知相似性差异，越低越好。",
    "lpips_mean": "LPIPS 的平均值，越低越好。",
    "lpips_std": "LPIPS 的标准差，用于衡量不同样本之间的波动。",
    "ssim": "SSIM 结构相似性，越高越好。",
    "ssim_mean": "SSIM 的平均值，越高越好。",
    "ssim_std": "SSIM 的标准差，用于衡量不同样本之间的波动。",
    "edge_consistency": "生成图与线稿边缘的一致程度，越高越好。",
    "edge_consistency_f1": "边缘一致性的 F1 分数，越高越好。",
    "color_bleeding_rate": "颜色渗漏率，表示颜色越界程度，通常越低越好。",
    "histogram_correlation": "生成图与目标图颜色直方图的一致程度，越高越好。",
    "inference_time_ms": "单张图像推理耗时，单位毫秒。",
    "loss": "当前训练 step 的损失值。",
    "train_loss": "当前 epoch 的平均训练损失。",
    "val_loss": "当前验证阶段的平均损失。",
    "learning_rate": "当前优化器使用的学习率。",
    "lr": "当前 step 的学习率。",
    "diffusion_loss": "扩散噪声预测主损失，用于保持扩散模型训练目标。",
    "adversarial_loss": "WGAN-GP 生成器对抗损失，用于约束生成图像分布不发生塌缩。",
    "effective_adversarial_weight": "当前 step 实际参与总损失计算的对抗权重，可能因预热或坍塌保护而降低。",
    "effective_reconstruction_weight": "当前 step 实际参与总损失计算的重建权重。",
    "effective_flat_color_weight": "当前 step 实际参与总损失计算的纯色惩罚权重。",
    "reconstruction_l1_loss": "解码图像与真实图像之间的 L1 重建损失。",
    "flat_color_penalty": "用于抑制输出退化为大面积纯色色块的惩罚项。",
    "decoded_image_spatial_std": "生成图在空间维度上的平均标准差，越低越接近大面积纯色。",
    "critic_loss": "WGAN-GP 判别器损失。",
    "gradient_penalty": "WGAN-GP 中的梯度惩罚项，用于稳定判别器训练。",
    "critic_real_score": "判别器对真实图像的平均打分。",
    "critic_fake_score": "判别器对生成图像的平均打分。",
    "epoch_time_seconds": "当前 epoch 耗时，单位秒。",
    "epoch_time_hms": "当前 epoch 耗时，格式为时:分:秒。",
    "image_id": "样本图像的唯一标识。",
    "generated_path": "生成结果图片的路径。",
    "target_path": "参考真值图片的路径。",
    "lineart_path": "输入线稿图片的路径。",
    "eval_dir": "本次评估结果所在目录。",
    "generated_dir": "本次评估生成图所在目录。",
    "target_dir": "本次评估参考图所在目录。",
    "lineart_dir": "本次评估线稿图所在目录。",
    "archive_dir": "评估结果在推理归档目录中的存放位置。",
    "helper_metric_reports": "辅助工具生成的原始报告文件路径集合。",
    "helper_metric_errors": "辅助工具执行失败时记录的错误信息。",
    "validation_source": "本次验证指标使用的数据来源。",
    "preview_source": "本次预览图使用的数据来源。",
    "validation_note": "对当前验证策略的补充说明。",
    "dataset_root": "训练数据集的根目录路径。",
    "validation_dataset_root": "外部验证数据集的根目录路径。",
    "train_ratio": "训练集占总体样本的比例。",
    "val_ratio": "验证集占总体样本的比例。",
    "test_ratio": "测试集占总体样本的比例。",
    "split_seed": "进行数据划分时固定使用的随机种子。",
    "horizontal_flip": "是否对训练样本启用水平翻转增强。",
    "random_crop": "是否对训练样本启用随机裁剪增强。",
    "color_jitter": "是否对训练样本启用颜色扰动增强。",
    "dynamic_weight": "是否启用动态权重模块。",
    "adaptive_threshold": "是否启用自适应阈值模块。",
    "adaptive_rf": "是否启用自适应感受野模块。",
    "adaptive_norm": "是否启用自适应归一化模块。",
    "wgan_start_epoch": "从第几个 epoch 开始启用 WGAN-GP 对抗训练。",
    "wgan_warmup_epochs": "WGAN-GP 启用后，生成器对抗权重从小到大预热的轮数。",
    "wgan_balance_penalty_factor": "当判别器出现 fake 分数高于 real 分数时，对抗损失额外缩小的比例。",
    "collapse_guard_enabled": "是否启用基于 FID、SSIM、颜色渗漏率等指标的坍塌保护逻辑。",
    "collapse_guard_adversarial_scale": "进入坍塌保护冷却期后，对抗权重会再乘上的缩放系数。",
    "collapse_guard_reconstruction_boost": "进入坍塌保护冷却期后，重建损失权重的放大系数。",
    "collapse_guard_flat_boost": "进入坍塌保护冷却期后，纯色惩罚权重的放大系数。",
    "collapse_guard_cooldown_epochs": "触发坍塌保护后，额外维持保守对抗训练的 epoch 数。",
    "collapse_guard_color_bleeding_threshold": "颜色渗漏率高于该阈值时，会被视作坍塌风险信号之一。",
    "collapse_guard_ssim_threshold": "SSIM 低于该阈值时，会被视作坍塌风险信号之一。",
    "collapse_guard_precision_threshold": "Precision 或 Recall 低于该阈值时，会被视作坍塌风险信号之一。",
    "collapse_guard_fid_ratio_threshold": "当当前 FID 超过历史最佳 FID 的该倍数时，会被视作坍塌风险信号之一。",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def localized_file_name(file_name: str) -> str:
    if file_name in FILE_NAME_MAP:
        return FILE_NAME_MAP[file_name]
    path = Path(file_name)
    translated_stem = translate_identifier(path.stem)
    suffix = "".join(path.suffixes)
    return f"{translated_stem}{suffix}" if translated_stem else file_name


def describe_file(file_name: str) -> str:
    return FILE_DESCRIPTION_MAP.get(file_name, f"{localized_file_name(file_name)} 的中文版导出文件。")


def translate_identifier(name: str) -> str:
    text = str(name)
    if text in EXACT_FIELD_NAME_MAP:
        return EXACT_FIELD_NAME_MAP[text]
    if "." in text:
        return ".".join(translate_identifier(part) for part in text.split("."))
    parts = [part for part in text.split("_") if part]
    if parts and all(part.lower() in TOKEN_NAME_MAP for part in parts):
        return "".join(TOKEN_NAME_MAP[part.lower()] for part in parts)
    return text


def explain_identifier(name: str) -> str:
    text = str(name)
    if text in EXACT_FIELD_EXPLANATION_MAP:
        return EXACT_FIELD_EXPLANATION_MAP[text]
    translated = translate_identifier(text)
    if text.endswith("_path"):
        return f"{translated}，记录对应文件的绝对路径。"
    if text.endswith("_dir"):
        return f"{translated}，记录对应目录的绝对路径。"
    if text.endswith("_gb"):
        return f"{translated}，单位为 GB。"
    if text.endswith("_seconds"):
        return f"{translated}，单位为秒。"
    if text.endswith("_ms"):
        return f"{translated}，单位为毫秒。"
    if text.endswith("_hms"):
        return f"{translated}，采用时:分:秒格式。"
    if text.endswith("_mean"):
        return f"{translated}，表示该指标在当前统计范围内的平均值。"
    if text.endswith("_std"):
        return f"{translated}，表示该指标在当前统计范围内的标准差。"
    if text.endswith("_count"):
        return f"{translated}，表示数量统计结果。"
    if text.endswith("_source"):
        return f"{translated}，说明该结果来自哪个数据源或计算来源。"
    if text.endswith("_records"):
        return f"{translated}，记录本次使用到的样本清单。"
    return f"{translated}。"


def translate_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return VALUE_TRANSLATION_MAP.get(value, value)
    return value


def _translate_structure(value: Any) -> Any:
    if isinstance(value, dict):
        return {translate_identifier(str(key)): _translate_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_translate_structure(item) for item in value]
    return translate_scalar(value)


def _try_parse_structured_string(text: str) -> Any:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{(":
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(stripped)
        except Exception:
            continue
    return None


def _translate_cell(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, str):
        parsed = _try_parse_structured_string(value)
        if parsed is not None:
            return json.dumps(_translate_structure(parsed), ensure_ascii=False)
        return translate_scalar(value)
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(_translate_structure(value), ensure_ascii=False)
    return value


def translate_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        translated = frame.copy()
        translated.columns = [translate_identifier(str(column)) for column in translated.columns]
        return translated
    translated = frame.copy()
    translated.columns = [translate_identifier(str(column)) for column in translated.columns]
    for column in translated.columns:
        translated[column] = translated[column].map(_translate_cell)
    return translated


def _build_json_field_rows(payload: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    def visit(value: Any, raw_path: str = "", cn_path: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key)
                next_raw = f"{raw_path}.{key_text}" if raw_path else key_text
                next_cn = f"{cn_path}.{translate_identifier(key_text)}" if cn_path else translate_identifier(key_text)
                if next_raw not in seen:
                    rows.append(
                        {
                            "原始字段路径": next_raw,
                            "中文字段路径": next_cn,
                            "中文解释": explain_identifier(key_text),
                        }
                    )
                    seen.add(next_raw)
                visit(item, next_raw, next_cn)
            return
        if isinstance(value, list):
            sample_dicts = [item for item in value if isinstance(item, dict)]
            if sample_dicts:
                sample = sample_dicts[0]
                list_raw = f"{raw_path}[]" if raw_path else "[]"
                list_cn = f"{cn_path}[]" if cn_path else "[]"
                for key, item in sample.items():
                    key_text = str(key)
                    next_raw = f"{list_raw}.{key_text}"
                    next_cn = f"{list_cn}.{translate_identifier(key_text)}"
                    if next_raw not in seen:
                        rows.append(
                            {
                                "原始字段路径": next_raw,
                                "中文字段路径": next_cn,
                                "中文解释": explain_identifier(key_text),
                            }
                        )
                        seen.add(next_raw)
                    visit(item, next_raw, next_cn)

    visit(payload)
    return rows


def _build_dataframe_field_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "原始字段名": str(column),
            "中文字段名": translate_identifier(str(column)),
            "中文解释": explain_identifier(str(column)),
        }
        for column in frame.columns
    ]
    return pd.DataFrame(rows)


def _auto_fit_sheet(worksheet, frame: pd.DataFrame) -> None:
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    sampled = frame.head(200)
    for column_index, column_name in enumerate(frame.columns, start=1):
        width = len(str(column_name)) + 2
        if not sampled.empty:
            lengths = sampled[column_name].map(lambda value: len(str(value)) if value is not None else 0)
            if not lengths.empty:
                width = max(width, int(lengths.max()) + 2)
        worksheet.column_dimensions[get_column_letter(column_index)].width = min(max(width, 10), 80)


def _write_excel_sheet(writer: pd.ExcelWriter, sheet_name: str, frame: pd.DataFrame) -> None:
    frame.to_excel(writer, sheet_name=sheet_name, index=False)
    _auto_fit_sheet(writer.sheets[sheet_name], frame)


def export_localized_json_artifact(path: str | Path, payload: Any | None = None) -> Path | None:
    raw_path = Path(path)
    if not raw_path.exists() and payload is None:
        return None
    if payload is None:
        try:
            with raw_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return None

    localized_path = raw_path.with_name(localized_file_name(raw_path.name))
    export_payload = {
        "文件说明": describe_file(raw_path.name),
        "原始文件名": raw_path.name,
        "原始文件路径": str(raw_path.resolve()),
        "导出时间": _now_iso(),
        "字段说明": _build_json_field_rows(payload),
        "数据": _translate_structure(payload),
    }
    localized_path.parent.mkdir(parents=True, exist_ok=True)
    with localized_path.open("w", encoding="utf-8") as handle:
        json.dump(export_payload, handle, ensure_ascii=False, indent=2)
    return localized_path


def export_localized_csv_artifact(path: str | Path, frame: pd.DataFrame | None = None) -> dict[str, str]:
    raw_path = Path(path)
    if frame is None:
        if not raw_path.exists():
            return {}
        try:
            frame = pd.read_csv(raw_path)
        except Exception:
            return {}

    translated = translate_dataframe(frame)
    localized_csv_path = raw_path.with_name(localized_file_name(raw_path.name))
    localized_xlsx_path = localized_csv_path.with_suffix(".xlsx")
    translated.to_csv(localized_csv_path, index=False)

    overview = pd.DataFrame(
        [
            {"项目": "文件说明", "内容": describe_file(raw_path.name)},
            {"项目": "原始文件名", "内容": raw_path.name},
            {"项目": "原始文件路径", "内容": str(raw_path.resolve())},
            {"项目": "导出时间", "内容": _now_iso()},
            {"项目": "行数", "内容": len(frame)},
            {"项目": "列数", "内容": len(frame.columns)},
        ]
    )
    field_rows = _build_dataframe_field_rows(frame)
    with pd.ExcelWriter(localized_xlsx_path, engine="openpyxl") as writer:
        _write_excel_sheet(writer, "说明", overview)
        _write_excel_sheet(writer, "数据", translated)
        _write_excel_sheet(writer, "字段说明", field_rows)
    return {
        "csv": str(localized_csv_path.resolve()),
        "xlsx": str(localized_xlsx_path.resolve()),
    }


def export_localized_jsonl_artifact(path: str | Path) -> dict[str, str]:
    raw_path = Path(path)
    if not raw_path.exists():
        return {}
    rows: list[dict[str, Any]] = []
    try:
        with raw_path.open("r", encoding="utf-8") as handle:
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
        return {}

    localized_jsonl_path = raw_path.with_name(localized_file_name(raw_path.name))
    localized_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with localized_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_translate_structure(row), ensure_ascii=False) + "\n")

    frame = pd.DataFrame(rows)
    translated = translate_dataframe(frame)
    localized_csv_path = localized_jsonl_path.with_suffix(".csv")
    localized_xlsx_path = localized_jsonl_path.with_suffix(".xlsx")
    translated.to_csv(localized_csv_path, index=False)

    overview = pd.DataFrame(
        [
            {"项目": "文件说明", "内容": describe_file(raw_path.name)},
            {"项目": "原始文件名", "内容": raw_path.name},
            {"项目": "原始文件路径", "内容": str(raw_path.resolve())},
            {"项目": "导出时间", "内容": _now_iso()},
            {"项目": "记录数", "内容": len(rows)},
            {"项目": "字段数", "内容": len(frame.columns)},
        ]
    )
    field_rows = _build_dataframe_field_rows(frame)
    with pd.ExcelWriter(localized_xlsx_path, engine="openpyxl") as writer:
        _write_excel_sheet(writer, "说明", overview)
        _write_excel_sheet(writer, "数据", translated)
        _write_excel_sheet(writer, "字段说明", field_rows)
    return {
        "jsonl": str(localized_jsonl_path.resolve()),
        "csv": str(localized_csv_path.resolve()),
        "xlsx": str(localized_xlsx_path.resolve()),
    }


def sync_run_localized_outputs(run_dir: str | Path) -> None:
    root = Path(run_dir)
    for relative in [
        "run_summary.json",
        "train_status.json",
        "run_metadata.json",
        "dataset_split.json",
        "validation_selection.json",
        "logs/epoch_history.json",
    ]:
        export_localized_json_artifact(root / relative)
    for relative in ["logs/train.jsonl", "logs/metrics.jsonl"]:
        export_localized_jsonl_artifact(root / relative)


def sync_evaluation_localized_outputs(eval_dir: str | Path) -> None:
    root = Path(eval_dir)
    for relative in ["metrics.json", "subgroup_metrics.json", "helper_tool_metrics.json"]:
        export_localized_json_artifact(root / relative)
    export_localized_csv_artifact(root / "per_sample_metrics.csv")
    helper_reports_dir = root / "helper_tools_runtime" / "reports"
    if helper_reports_dir.exists():
        for csv_path in sorted(helper_reports_dir.rglob("*.csv")):
            export_localized_csv_artifact(csv_path)


def sync_archive_localized_outputs(archive_dir: str | Path) -> None:
    root = Path(archive_dir)
    export_localized_json_artifact(root / "archive_manifest.json")
    reports_dir = root / "reports"
    if reports_dir.exists():
        for json_path in sorted(reports_dir.glob("*.json")):
            export_localized_json_artifact(json_path)
        for csv_path in sorted(reports_dir.glob("*.csv")):
            export_localized_csv_artifact(csv_path)
    helper_reports_dir = root / "helper_tools_runtime" / "reports"
    if helper_reports_dir.exists():
        for csv_path in sorted(helper_reports_dir.rglob("*.csv")):
            export_localized_csv_artifact(csv_path)


def sync_analysis_localized_outputs(analysis_dir: str | Path) -> Path | None:
    root = Path(analysis_dir)
    if not root.exists():
        return None

    sheet_specs = [
        ("单次运行汇总", "per_run_summary.csv"),
        ("实验组汇总", "group_summary.csv"),
        ("Seed平均汇总", "seed_average_summary.csv"),
        ("最优检查点汇总", "best_checkpoints.csv"),
        ("训练步日志", "train_step_logs.csv"),
        ("轮次指标日志", "epoch_metric_logs.csv"),
        ("逐图评估总表", "per_sample_metrics_all.csv"),
        ("单模块贡献", "single_module_contributions.csv"),
        ("交互效应", "interaction_effects.csv"),
        ("配对T检验", "paired_t_tests.csv"),
    ]

    workbook_rows: list[dict[str, Any]] = [
        {"项目": "文件说明", "内容": describe_file("experiment_summary.xlsx")},
        {"项目": "分析目录", "内容": str(root.resolve())},
        {"项目": "导出时间", "内容": _now_iso()},
    ]
    field_rows: list[dict[str, str]] = []
    localized_workbook_path = root / localized_file_name("experiment_summary.xlsx")

    with pd.ExcelWriter(localized_workbook_path, engine="openpyxl") as writer:
        for sheet_name, file_name in sheet_specs:
            raw_path = root / file_name
            if not raw_path.exists():
                continue
            try:
                frame = pd.read_csv(raw_path)
            except Exception:
                continue
            export_localized_csv_artifact(raw_path, frame)
            translated = translate_dataframe(frame)
            _write_excel_sheet(writer, sheet_name, translated)
            workbook_rows.append({"项目": sheet_name, "内容": str(raw_path.resolve())})
            for column in frame.columns:
                field_rows.append(
                    {
                        "所属Sheet": sheet_name,
                        "原始字段名": str(column),
                        "中文字段名": translate_identifier(str(column)),
                        "中文解释": explain_identifier(str(column)),
                    }
                )

        _write_excel_sheet(writer, "说明", pd.DataFrame(workbook_rows))
        _write_excel_sheet(writer, "字段说明", pd.DataFrame(field_rows))

    export_localized_json_artifact(root / "analysis_summary.json")
    return localized_workbook_path


def sync_output_root_localized_outputs(output_root: str | Path) -> None:
    root = Path(output_root)
    export_localized_json_artifact(root / "experiment_config.lock.json")
