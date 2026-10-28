"""PyTorch U-Net components and categorical segmentation loss.

The architectural code is reused from this project's earlier learning prototype;
this clean implementation restart does not reuse its trained weights or scores.
The model/loss definitions below are retained to avoid silently changing the
fixed architecture while rebuilding the surrounding data and execution pipeline.

The 2D baseline uses the encoder/decoder and skip-connection principle of
Ronneberger et al. (2015), https://arxiv.org/abs/1505.04597.
The 3D model is an inspired adaptation of residual units and deep supervision
discussed by Isensee et al., https://arxiv.org/abs/1802.10508. It is NOT an exact
reproduction of that paper or nnU-Net. This compact implementation uses three
downsampling stages, GroupNorm, and two auxiliary segmentation heads. Our
adaptation uses max-pooling rather than strided convolution, GroupNorm rather
than InstanceNorm, and no dropout. Deep supervision applies independent
weighted losses at native head resolutions rather than summing segmentation
logits as in the cited source. These differences are deliberate and explicit.

No HipMRI label mapping is assumed: callers must set num_classes from an
audited dataset specification. Heads emit mutually exclusive categorical
logits; predictions are logits.argmax(dim=1), not independent sigmoid masks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _check_configuration(num_classes: int, base_channels: int) -> None:
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two.")
    if not isinstance(base_channels, int) or base_channels < 1:
        raise ValueError("base_channels must be a positive integer.")


def _check_input(images: Tensor, spatial_dimensions: int) -> None:
    if images.ndim != spatial_dimensions + 2 or images.shape[1] != 1:
        raise ValueError(
            f"Expected (batch, 1, {', '.join(['spatial'] * spatial_dimensions)}) input."
        )
    if images.shape[0] < 1 or min(images.shape[2:]) < 8:
        raise ValueError("Input batch must be nonempty and each spatial size must be >= 8.")
    if not images.is_floating_point():
        raise ValueError("Model input must be a floating-point tensor.")


class _DoubleConv2D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class UNet2D(nn.Module):
    """Standard three-level 2D U-Net baseline with a single categorical head."""

    def __init__(self, num_classes: int = 6, base_channels: int = 16):
        super().__init__()
        _check_configuration(num_classes, base_channels)
        widths = [base_channels * 2**level for level in range(4)]
        self.encoders = nn.ModuleList(
            [_DoubleConv2D(1, widths[0])]
            + [_DoubleConv2D(widths[i - 1], widths[i]) for i in range(1, 4)]
        )
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in (2, 1, 0)]
        )
        self.decoders = nn.ModuleList(
            [_DoubleConv2D(2 * widths[i], widths[i]) for i in (2, 1, 0)]
        )
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, images: Tensor) -> dict[str, Tensor | list[Tensor]]:
        _check_input(images, 2)
        skips = []
        features = images
        for index, encoder in enumerate(self.encoders):
            features = encoder(features if index == 0 else self.pool(features))
            if index < 3:
                skips.append(features)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            features = up(features)
            if features.shape[2:] != skip.shape[2:]:
                features = F.interpolate(features, size=skip.shape[2:], mode="bilinear", align_corners=False)
            features = decoder(torch.cat((features, skip), dim=1))
        return {"logits": self.head(features), "aux_logits": []}


def _group_norm(channels: int) -> nn.GroupNorm:
    # Retain at least two channels per group at the deepest, potentially 1^3,
    # feature map. GroupNorm does not depend on batch statistics.
    groups = min(8, max(1, channels // 2))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _ResidualUnit3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            _group_norm(out_channels),
        )
        self.shortcut = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, features: Tensor) -> Tensor:
        return self.activation(self.layers(features) + self.shortcut(features))


class ImprovedUNet3D(nn.Module):
    """Residual 3D U-Net with batch-one-safe normalization and deep supervision.

    Auxiliary heads are returned in order: approximately half resolution, then
    quarter resolution. The primary head always matches the input spatial size,
    including odd dimensions. Three floor-pooling operations require sizes >= 8.
    """

    def __init__(self, num_classes: int = 6, base_channels: int = 16):
        super().__init__()
        _check_configuration(num_classes, base_channels)
        widths = [base_channels * 2**level for level in range(4)]
        self.encoders = nn.ModuleList(
            [_ResidualUnit3D(1, widths[0])]
            + [_ResidualUnit3D(widths[i - 1], widths[i]) for i in range(1, 4)]
        )
        self.pool = nn.MaxPool3d(2)
        self.ups = nn.ModuleList(
            [nn.Conv3d(widths[i + 1], widths[i], 3, padding=1, bias=False) for i in (2, 1, 0)]
        )
        self.decoders = nn.ModuleList(
            [_ResidualUnit3D(2 * widths[i], widths[i]) for i in (2, 1, 0)]
        )
        self.head = nn.Conv3d(widths[0], num_classes, 1)
        self.aux_heads = nn.ModuleList(
            [nn.Conv3d(widths[1], num_classes, 1), nn.Conv3d(widths[2], num_classes, 1)]
        )

    def forward(self, images: Tensor) -> dict[str, Tensor | list[Tensor]]:
        _check_input(images, 3)
        skips = []
        features = images
        for index, encoder in enumerate(self.encoders):
            features = encoder(features if index == 0 else self.pool(features))
            if index < 3:
                skips.append(features)
        decoder_features = []
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            features = up(F.interpolate(features, size=skip.shape[2:], mode="trilinear", align_corners=False))
            features = decoder(torch.cat((features, skip), dim=1))
            decoder_features.append(features)
        return {
            "logits": self.head(features),
            "aux_logits": [
                self.aux_heads[0](decoder_features[1]),
                self.aux_heads[1](decoder_features[0]),
            ],
        }


def _categorical_head_loss(
    logits: Tensor, target: Tensor, ce_weight: float, dice_weight: float, smooth: float,
    class_weights: Tensor | None = None,
) -> Tensor:
    if tuple(target.shape[1:]) != tuple(logits.shape[2:]):
        # Resize class IDs only with nearest-neighbour interpolation.
        target = F.interpolate(
            target.unsqueeze(1).float(), size=logits.shape[2:], mode="nearest"
        ).squeeze(1).long()
    # Keep the caller's device; only match the head's floating-point dtype.
    head_weights = class_weights.to(dtype=logits.dtype) if class_weights is not None else None
    ce = F.cross_entropy(logits, target, weight=head_weights)
    probabilities = torch.softmax(logits.float(), dim=1)
    one_hot = F.one_hot(target, num_classes=logits.shape[1]).movedim(-1, 1).float()
    axes = (0,) + tuple(range(2, logits.ndim))
    intersection = (probabilities * one_hot).sum(dim=axes)
    denominator = probabilities.sum(dim=axes) + one_hot.sum(dim=axes)
    dice_loss = 1 - ((2 * intersection + smooth) / (denominator + smooth)).mean()
    return ce_weight * ce + dice_weight * dice_loss


def segmentation_loss(
    output: dict[str, Tensor | list[Tensor]],
    target: Tensor,
    aux_weights: Sequence[float] = (0.5, 0.25),
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    smooth: float = 1e-5,
    class_weights: Tensor | None = None,
) -> Tensor:
    """CE + soft multiclass Dice, averaged over every class including background.

    The main/half/quarter head weights are 1/0.5/0.25 by default. The weighted
    sum is divided by the sum of active weights. Baseline output has no auxiliary
    heads, so its loss is simply CE + Dice. Integer target labels are validated
    before nearest-neighbour resizing to each head. No independent binary masks
    or sigmoid losses are used. Optional class_weights apply to CE at every
    head, leaving the equally weighted multiclass Dice term unchanged. Callers
    should derive weights from training data only and reuse the same vector for
    each model in a controlled comparison.
    """
    logits = output.get("logits")
    auxiliary = output.get("aux_logits", [])
    if not isinstance(logits, Tensor) or logits.ndim not in (4, 5):
        raise ValueError("logits must be a 2D or 3D segmentation tensor.")
    if not isinstance(auxiliary, list) or not all(isinstance(head, Tensor) for head in auxiliary):
        raise ValueError("aux_logits must be a list of tensors.")
    if logits.shape[1] < 2:
        raise ValueError("Categorical segmentation requires at least two classes.")
    if class_weights is not None:
        if not isinstance(class_weights, Tensor):
            raise ValueError("class_weights must be a floating-point torch tensor.")
        if class_weights.ndim != 1 or class_weights.numel() != logits.shape[1]:
            raise ValueError("class_weights must be 1D with one entry per class.")
        if class_weights.device != logits.device:
            raise ValueError("class_weights and logits must be on the same device.")
        if not class_weights.is_floating_point():
            raise ValueError("class_weights must be a floating-point torch tensor.")
        if not bool(torch.isfinite(class_weights).all().item()):
            raise ValueError("class_weights must contain only finite values.")
        if bool((class_weights < 0).any().item()) or not bool((class_weights > 0).any().item()):
            raise ValueError("class_weights must be nonnegative and not all zero.")
    if target.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("target must contain integer class IDs, not float or one-hot labels.")
    if target.ndim != logits.ndim - 1 or target.shape[0] != logits.shape[0] or target.numel() == 0:
        raise ValueError("target must have shape (batch, *spatial) and match the logits batch.")
    if target.device != logits.device:
        raise ValueError("target and logits must be on the same device.")
    if int(target.min().item()) < 0 or int(target.max().item()) >= logits.shape[1]:
        raise ValueError(f"target class IDs must be in [0, {logits.shape[1] - 1}].")
    if not all(math.isfinite(value) and value >= 0 for value in (ce_weight, dice_weight)) or ce_weight + dice_weight == 0:
        raise ValueError("CE and Dice weights must be finite, nonnegative, and not both zero.")
    if not math.isfinite(smooth) or smooth <= 0:
        raise ValueError("smooth must be positive and finite.")
    weights = tuple(aux_weights) if auxiliary else ()
    if len(weights) != len(auxiliary) or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Provide one positive, finite aux weight per auxiliary head.")
    target = target.long()
    total = _categorical_head_loss(logits, target, ce_weight, dice_weight, smooth, class_weights)
    for head, weight in zip(auxiliary, weights):
        if head.ndim != logits.ndim or head.shape[:2] != logits.shape[:2] or head.device != logits.device:
            raise ValueError("Every auxiliary head must match the main batch, classes, dimensions, and device.")
        total = total + weight * _categorical_head_loss(head, target, ce_weight, dice_weight, smooth, class_weights)
    return total / (1.0 + sum(weights))
