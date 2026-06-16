from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PatchImageCritic(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        channels = max(16, int(base_channels))
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 4, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(images)
        return logits.flatten(start_dim=1).mean(dim=1)


def resize_for_critic(images: torch.Tensor, image_size: int) -> torch.Tensor:
    target_size = max(64, int(image_size))
    return F.interpolate(images, size=(target_size, target_size), mode="bilinear", align_corners=False)


def compute_gradient_penalty(
    critic: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
) -> torch.Tensor:
    batch_size = real_images.shape[0]
    alpha = torch.rand(batch_size, 1, 1, 1, device=real_images.device, dtype=real_images.dtype)
    interpolated = (alpha * real_images + (1.0 - alpha) * fake_images).requires_grad_(True)
    critic_scores = critic(interpolated)
    grad_outputs = torch.ones_like(critic_scores)
    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.flatten(start_dim=1)
    gradient_norm = gradients.norm(2, dim=1)
    return ((gradient_norm - 1.0) ** 2).mean()


def compute_flat_color_penalty(images: torch.Tensor, min_std: float) -> torch.Tensor:
    spatial_std = images.flatten(start_dim=2).std(dim=2, unbiased=False).mean(dim=1)
    threshold = images.new_full(spatial_std.shape, float(min_std))
    return F.relu(threshold - spatial_std).mean()
