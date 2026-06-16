from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .ablation import ModuleFlags


@dataclass(frozen=True)
class AdapterConfig:
    in_channels: int = 3
    hidden_channels: int = 32
    num_blocks: int = 2
    fixed_threshold_value: float = 0.08
    enable_variational_bottleneck: bool = True
    latent_channels: int = 12
    bottleneck_channels: int = 24
    decoder_channels: int = 16
    bottleneck_dropout: float = 0.05
    logvar_min: float = -6.0
    logvar_max: float = 2.0


class AdaptiveNorm2d(nn.Module):
    def __init__(self, channels: int, enabled: bool) -> None:
        super().__init__()
        self.enabled = enabled
        self.norm = nn.BatchNorm2d(channels)
        if enabled:
            self.context = nn.Sequential(
                nn.Linear(channels, channels * 2),
                nn.SiLU(),
                nn.Linear(channels * 2, channels * 2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if not self.enabled:
            return h
        pooled = x.mean(dim=(2, 3))
        scale, shift = self.context(pooled).chunk(2, dim=1)
        return h * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)


class ReceptiveFieldMixer(nn.Module):
    def __init__(self, channels: int, enabled: bool) -> None:
        super().__init__()
        self.enabled = enabled
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.Conv2d(channels, channels, kernel_size=5, padding=2),
                nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            ]
        )
        if enabled:
            self.router = nn.Sequential(
                nn.Linear(channels, channels),
                nn.SiLU(),
                nn.Linear(channels, len(self.branches)),
            )
        else:
            self.fallback = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.fallback(x)
        pooled = x.mean(dim=(2, 3))
        weights = torch.softmax(self.router(pooled), dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        branches = torch.stack([branch(x) for branch in self.branches], dim=1)
        return torch.sum(branches * weights, dim=1)


class DynamicWeighting(nn.Module):
    def __init__(self, channels: int, enabled: bool) -> None:
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.gate = nn.Sequential(
                nn.Linear(channels, channels),
                nn.SiLU(),
                nn.Linear(channels, channels * 2),
            )

    def forward(self, skip: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return 0.5 * skip + 0.5 * residual
        pooled = residual.mean(dim=(2, 3))
        skip_gate, residual_gate = self.gate(pooled).chunk(2, dim=1)
        skip_gate = torch.sigmoid(skip_gate).unsqueeze(-1).unsqueeze(-1)
        residual_gate = torch.sigmoid(residual_gate).unsqueeze(-1).unsqueeze(-1)
        return skip * skip_gate + residual * residual_gate


class AdaptiveThreshold(nn.Module):
    def __init__(self, channels: int, enabled: bool, fixed_threshold_value: float) -> None:
        super().__init__()
        self.enabled = enabled
        self.fixed_threshold_value = float(fixed_threshold_value)
        if enabled:
            self.threshold = nn.Sequential(
                nn.Linear(channels, channels),
                nn.SiLU(),
                nn.Linear(channels, channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            pooled = x.mean(dim=(2, 3))
            threshold = F.softplus(self.threshold(pooled)).unsqueeze(-1).unsqueeze(-1)
        else:
            threshold = x.new_full((x.shape[0], x.shape[1], 1, 1), self.fixed_threshold_value)
        return torch.sign(x) * F.relu(torch.abs(x) - threshold)


class ConditionAdapterBlock(nn.Module):
    def __init__(self, channels: int, flags: ModuleFlags, fixed_threshold_value: float) -> None:
        super().__init__()
        self.norm1 = AdaptiveNorm2d(channels, enabled=flags.adaptive_norm)
        self.rf = ReceptiveFieldMixer(channels, enabled=flags.adaptive_rf)
        self.norm2 = AdaptiveNorm2d(channels, enabled=flags.adaptive_norm)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.threshold = AdaptiveThreshold(
            channels,
            enabled=flags.adaptive_threshold,
            fixed_threshold_value=fixed_threshold_value,
        )
        self.weighting = DynamicWeighting(channels, enabled=flags.dynamic_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.rf(F.silu(self.norm1(x)))
        residual = self.conv(F.silu(self.norm2(residual)))
        residual = self.threshold(residual)
        return self.weighting(x, residual)


class LowCapacityVariationalBottleneck(nn.Module):
    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.enabled = bool(config.enable_variational_bottleneck)
        if not self.enabled:
            return
        hidden = int(config.hidden_channels)
        bottleneck = max(8, int(config.bottleneck_channels))
        latent = max(4, int(config.latent_channels))
        decoder_channels = max(4, int(config.decoder_channels))
        dropout = max(0.0, float(config.bottleneck_dropout))
        self.logvar_min = float(config.logvar_min)
        self.logvar_max = float(config.logvar_max)
        self.encoder = nn.Sequential(
            nn.Conv2d(hidden, bottleneck, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(bottleneck, bottleneck, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.to_mu = nn.Conv2d(bottleneck, latent, kernel_size=1)
        self.to_logvar = nn.Conv2d(bottleneck, latent, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent, decoder_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(decoder_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.enabled:
            return x, {}
        encoded = self.encoder(x)
        mu = self.to_mu(encoded)
        logvar = self.to_logvar(encoded).clamp(min=self.logvar_min, max=self.logvar_max)
        std = torch.exp(0.5 * logvar)
        if self.training:
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        decoded = self.decoder(z)
        if decoded.shape[-2:] != x.shape[-2:]:
            decoded = F.interpolate(decoded, size=x.shape[-2:], mode="bilinear", align_corners=False)
        reconstruction_loss = F.smooth_l1_loss(decoded, x.detach())
        return decoded, {
            "posterior_mu": mu,
            "posterior_logvar": logvar,
            "latent_sample": z,
            "bottleneck_reconstruction_loss": reconstruction_loss,
        }


class SXDLConditionAdapter(nn.Module):
    def __init__(self, config: AdapterConfig, flags: ModuleFlags) -> None:
        super().__init__()
        self.config = config
        self.flags = flags
        hidden = config.hidden_channels
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ConditionAdapterBlock(hidden, flags=flags, fixed_threshold_value=config.fixed_threshold_value) for _ in range(config.num_blocks)]
        )
        self.variational_bottleneck = LowCapacityVariationalBottleneck(config)
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, config.in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, return_stats: bool = False):
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        bottleneck_stats: dict[str, torch.Tensor] = {}
        if self.variational_bottleneck.enabled:
            h, bottleneck_stats = self.variational_bottleneck(h)
        residual = torch.tanh(self.head(h)) * 0.25
        adapted = (x + residual).clamp(0.0, 1.0)
        if not return_stats:
            return adapted
        stats = dict(bottleneck_stats)
        stats["adapted_control"] = adapted
        return adapted, stats
