# Ultralytics YOLO, GPL-3.0 license
"""RecoveryBranch v1 modules for auxiliary dehazing and P3 feature recovery."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """A small Conv-BN-SiLU block used by RecoveryBranch."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class RecoveryBranch(nn.Module):
    """Lightweight recovery branch.

    Input:
        x: [B, 3, H, W]

    Output:
        {
            "dehaze_img": [B, 3, H, W],
            "r3": [B, r3_channels, H/8, W/8]
        }
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32, r3_channels: int = 64):
        super().__init__()
        self.stem1 = ConvBNAct(in_channels, base_channels, 3, 2)
        self.stem2 = ConvBNAct(base_channels, base_channels * 2, 3, 2)
        self.stem3 = ConvBNAct(base_channels * 2, r3_channels, 3, 2)
        self.refine = nn.Sequential(
            ConvBNAct(r3_channels, r3_channels, 3, 1),
            ConvBNAct(r3_channels, r3_channels, 3, 1),
        )
        self.rgb_head = nn.Sequential(
            ConvBNAct(r3_channels, base_channels, 3, 1),
            nn.Conv2d(base_channels, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        r3 = self.stem3(self.stem2(self.stem1(x)))
        r3 = self.refine(r3)
        dehaze_low = self.rgb_head(r3)
        dehaze_img = F.interpolate(dehaze_low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"dehaze_img": dehaze_img, "r3": r3}
