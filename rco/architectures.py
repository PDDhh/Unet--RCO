"""
Memory-friendly 6-level U-Net / U-Net++ architectures for RCO segmentation.

v4 keeps a six-level NestedUNet structure and adds GroupNorm support:
- configurable base_channels / filters
- optional gradient checkpointing
- no forward debug printing
- configurable normalization: batch/group/instance/none
- constructor is compatible with scripts.train_rco / scripts.predict_rco config loading

For 1024x1024 grayscale RCO segmentation, recommended starting settings:
    --base_channels 8 --batch_size 1 --amp True --use_checkpoint True --grad_accumulation 2 --norm_type group
Equivalent filters: 8,16,32,64,128,256
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ["UNet", "NestedUNet"]


def _parse_filters(filters: Optional[Union[str, Sequence[int]]], base_channels: int) -> List[int]:
    """Return a 6-level channel list for six-level U-Net/U-Net++."""
    if filters is None or filters == "":
        return [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
            base_channels * 32,
        ]

    if isinstance(filters, str):
        parsed = [int(x.strip()) for x in filters.split(",") if x.strip()]
    else:
        parsed = [int(x) for x in filters]

    if len(parsed) != 6:
        raise ValueError(
            "filters must contain exactly 6 integers for the six-level UNet++ structure, "
            f"but got {parsed}. Example: --filters 8,16,32,64,128,256"
        )
    return parsed


def _make_norm(num_channels: int, norm_type: str = "batch", num_groups: int = 8) -> nn.Module:
    norm_type = (norm_type or "none").lower()
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "group":
        groups = min(int(num_groups), int(num_channels))
        while groups > 1 and num_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type in {"none", "identity", "false", "0"}:
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}. Use batch, group, instance, or none.")


class VGGBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        use_bn: bool = True,
        norm_type: str = "batch",
        num_groups: int = 8,
    ):
        super().__init__()
        if not use_bn:
            norm_type = "none"
        use_bias = (norm_type or "none").lower() in {"none", "identity", "false", "0"}

        layers = [nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, bias=use_bias)]
        norm = _make_norm(middle_channels, norm_type=norm_type, num_groups=num_groups)
        if not isinstance(norm, nn.Identity):
            layers.append(norm)
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1, bias=use_bias))
        norm = _make_norm(out_channels, norm_type=norm_type, num_groups=num_groups)
        if not isinstance(norm, nn.Identity):
            layers.append(norm)
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _CheckpointMixin:
    use_checkpoint: bool

    def _block(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)


class UNet(nn.Module, _CheckpointMixin):
    """Six-level U-Net, compatible with the original constructor style."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        deep_supervision: bool = False,
        base_channels: int = 8,
        filters: Optional[Union[str, Sequence[int]]] = None,
        use_checkpoint: bool = False,
        use_bn: bool = True,
        norm_type: str = "batch",
        num_groups: int = 8,
        align_corners: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.use_checkpoint = use_checkpoint
        self.align_corners = align_corners
        nb = _parse_filters(filters, base_channels)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = VGGBlock(input_channels, nb[0], nb[0], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv1_0 = VGGBlock(nb[0], nb[1], nb[1], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv2_0 = VGGBlock(nb[1], nb[2], nb[2], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv3_0 = VGGBlock(nb[2], nb[3], nb[3], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv4_0 = VGGBlock(nb[3], nb[4], nb[4], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv5_0 = VGGBlock(nb[4], nb[5], nb[5], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)

        self.conv4_1 = VGGBlock(nb[4] + nb[5], nb[4], nb[4], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv3_2 = VGGBlock(nb[3] + nb[4], nb[3], nb[3], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv2_3 = VGGBlock(nb[2] + nb[3], nb[2], nb[2], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv1_4 = VGGBlock(nb[1] + nb[2], nb[1], nb[1], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv0_5 = VGGBlock(nb[0] + nb[1], nb[0], nb[0], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)

        self.final = nn.Conv2d(nb[0], num_classes, kernel_size=1)

    def _up_like(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=self.align_corners)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x0_0 = self._block(self.conv0_0, input)
        x1_0 = self._block(self.conv1_0, self.pool(x0_0))
        x2_0 = self._block(self.conv2_0, self.pool(x1_0))
        x3_0 = self._block(self.conv3_0, self.pool(x2_0))
        x4_0 = self._block(self.conv4_0, self.pool(x3_0))
        x5_0 = self._block(self.conv5_0, self.pool(x4_0))

        x4_1 = self._block(self.conv4_1, torch.cat([x4_0, self._up_like(x5_0, x4_0)], dim=1))
        x3_2 = self._block(self.conv3_2, torch.cat([x3_0, self._up_like(x4_1, x3_0)], dim=1))
        x2_3 = self._block(self.conv2_3, torch.cat([x2_0, self._up_like(x3_2, x2_0)], dim=1))
        x1_4 = self._block(self.conv1_4, torch.cat([x1_0, self._up_like(x2_3, x1_0)], dim=1))
        x0_5 = self._block(self.conv0_5, torch.cat([x0_0, self._up_like(x1_4, x0_0)], dim=1))

        return self.final(x0_5)


class NestedUNet(nn.Module, _CheckpointMixin):
    """
    Six-level U-Net++ for RCO segmentation.

    Default channel schedule with base_channels=8:
        8, 16, 32, 64, 128, 256

    Memory-saving recommendations for 1024x1024:
        deep_supervision=False
        base_channels=8
        use_checkpoint=True
        AMP enabled in training
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        deep_supervision: bool = False,
        base_channels: int = 8,
        filters: Optional[Union[str, Sequence[int]]] = None,
        use_checkpoint: bool = False,
        use_bn: bool = True,
        norm_type: str = "batch",
        num_groups: int = 8,
        align_corners: bool = False,
        **kwargs,
    ):
        super().__init__()
        nb = _parse_filters(filters, base_channels)

        self.deep_supervision = deep_supervision
        self.use_checkpoint = use_checkpoint
        self.align_corners = align_corners
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder nodes
        self.conv0_0 = VGGBlock(input_channels, nb[0], nb[0], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv1_0 = VGGBlock(nb[0], nb[1], nb[1], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv2_0 = VGGBlock(nb[1], nb[2], nb[2], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv3_0 = VGGBlock(nb[2], nb[3], nb[3], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv4_0 = VGGBlock(nb[3], nb[4], nb[4], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv5_0 = VGGBlock(nb[4], nb[5], nb[5], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)

        # j = 1
        self.conv0_1 = VGGBlock(nb[0] + nb[1], nb[0], nb[0], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv1_1 = VGGBlock(nb[1] + nb[2], nb[1], nb[1], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv2_1 = VGGBlock(nb[2] + nb[3], nb[2], nb[2], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv3_1 = VGGBlock(nb[3] + nb[4], nb[3], nb[3], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv4_1 = VGGBlock(nb[4] + nb[5], nb[4], nb[4], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)

        # j = 2
        self.conv0_2 = VGGBlock(nb[0] * 2 + nb[1], nb[0], nb[0], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv1_2 = VGGBlock(nb[1] * 2 + nb[2], nb[1], nb[1], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv2_2 = VGGBlock(nb[2] * 2 + nb[3], nb[2], nb[2], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv3_2 = VGGBlock(nb[3] * 2 + nb[4], nb[3], nb[3], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)

        # j = 3
        self.conv0_3 = VGGBlock(nb[0] * 3 + nb[1], nb[0], nb[0], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv1_3 = VGGBlock(nb[1] * 3 + nb[2], nb[1], nb[1], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv2_3 = VGGBlock(nb[2] * 3 + nb[3], nb[2], nb[2], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)

        # j = 4
        self.conv0_4 = VGGBlock(nb[0] * 4 + nb[1], nb[0], nb[0], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)
        self.conv1_4 = VGGBlock(nb[1] * 4 + nb[2], nb[1], nb[1], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)

        # j = 5
        self.conv0_5 = VGGBlock(nb[0] * 5 + nb[1], nb[0], nb[0], use_bn=use_bn, norm_type=norm_type, num_groups=num_groups)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final5 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb[0], num_classes, kernel_size=1)

    def _up_like(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=self.align_corners)

    def forward(self, input: torch.Tensor):
        x0_0 = self._block(self.conv0_0, input)

        x1_0 = self._block(self.conv1_0, self.pool(x0_0))
        x0_1 = self._block(self.conv0_1, torch.cat([x0_0, self._up_like(x1_0, x0_0)], dim=1))

        x2_0 = self._block(self.conv2_0, self.pool(x1_0))
        x1_1 = self._block(self.conv1_1, torch.cat([x1_0, self._up_like(x2_0, x1_0)], dim=1))
        x0_2 = self._block(self.conv0_2, torch.cat([x0_0, x0_1, self._up_like(x1_1, x0_0)], dim=1))

        x3_0 = self._block(self.conv3_0, self.pool(x2_0))
        x2_1 = self._block(self.conv2_1, torch.cat([x2_0, self._up_like(x3_0, x2_0)], dim=1))
        x1_2 = self._block(self.conv1_2, torch.cat([x1_0, x1_1, self._up_like(x2_1, x1_0)], dim=1))
        x0_3 = self._block(self.conv0_3, torch.cat([x0_0, x0_1, x0_2, self._up_like(x1_2, x0_0)], dim=1))

        x4_0 = self._block(self.conv4_0, self.pool(x3_0))
        x3_1 = self._block(self.conv3_1, torch.cat([x3_0, self._up_like(x4_0, x3_0)], dim=1))
        x2_2 = self._block(self.conv2_2, torch.cat([x2_0, x2_1, self._up_like(x3_1, x2_0)], dim=1))
        x1_3 = self._block(self.conv1_3, torch.cat([x1_0, x1_1, x1_2, self._up_like(x2_2, x1_0)], dim=1))
        x0_4 = self._block(self.conv0_4, torch.cat([x0_0, x0_1, x0_2, x0_3, self._up_like(x1_3, x0_0)], dim=1))

        x5_0 = self._block(self.conv5_0, self.pool(x4_0))
        x4_1 = self._block(self.conv4_1, torch.cat([x4_0, self._up_like(x5_0, x4_0)], dim=1))
        x3_2 = self._block(self.conv3_2, torch.cat([x3_0, x3_1, self._up_like(x4_1, x3_0)], dim=1))
        x2_3 = self._block(self.conv2_3, torch.cat([x2_0, x2_1, x2_2, self._up_like(x3_2, x2_0)], dim=1))
        x1_4 = self._block(self.conv1_4, torch.cat([x1_0, x1_1, x1_2, x1_3, self._up_like(x2_3, x1_0)], dim=1))
        x0_5 = self._block(self.conv0_5, torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self._up_like(x1_4, x0_0)], dim=1))

        if self.deep_supervision:
            return [
                self.final1(x0_1),
                self.final2(x0_2),
                self.final3(x0_3),
                self.final4(x0_4),
                self.final5(x0_5),
            ]
        return self.final(x0_5)
