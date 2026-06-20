from __future__ import annotations

import math
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinarySEHead(nn.Module):
    def __init__(self, in_channels: int = 384):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, in_channels // 16)
        self.fc2 = nn.Linear(in_channels // 16, in_channels)
        self.fc_out = nn.Linear(in_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 4:
            avg = self.avg_pool(features).flatten(1)
            maximum = self.max_pool(features).flatten(1)
        elif features.dim() == 3:
            token_count = features.size(1)
            if token_count > 1 and int(math.sqrt(token_count - 1)) ** 2 == token_count - 1:
                features = features[:, 1:, :]
            avg = features.mean(dim=1)
            maximum = features.amax(dim=1)
        else:
            avg = maximum = features

        scale = torch.sigmoid(
            self.fc2(self.relu(self.fc1(avg)))
            + self.fc2(self.relu(self.fc1(maximum)))
        )
        if features.dim() == 4:
            output = self.avg_pool(features * scale[:, :, None, None]).flatten(1)
        elif features.dim() == 3:
            output = (features * scale[:, None, :]).mean(dim=1)
        else:
            output = features * scale
        return self.fc_out(output)


class BinaryClassifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_config = config["model"]
        self.backbone = timm.create_model(
            model_config["architecture"],
            pretrained=False,
            num_classes=0,
            img_size=model_config.get("backbone_image_size", 518),
            dynamic_img_size=True,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        self.head = BinarySEHead(384) if model_config.get("use_se_head", True) else nn.Linear(384, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))


class MulticlassSEHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, (2 * in_dim) // reduction)
        self.fc1 = nn.Linear(2 * in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2 * in_dim)
        self.act = nn.ReLU(inplace=True)
        self.cls = nn.Linear(2 * in_dim, num_classes)

    def forward(self, tokens: torch.Tensor, num_prefix_tokens: int = 1) -> torch.Tensor:
        if tokens.dim() == 2:
            avg = maximum = tokens
        else:
            patches = tokens[:, num_prefix_tokens:, :] if tokens.size(1) > num_prefix_tokens else tokens
            avg = patches.mean(dim=1)
            maximum = patches.amax(dim=1)
        features = torch.cat([avg, maximum], dim=1)
        scale = torch.sigmoid(self.fc2(self.act(self.fc1(features))))
        return self.cls(features * scale)


class MulticlassClassifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_config = config["model"]
        self.backbone = timm.create_model(
            model_config["architecture"],
            pretrained=False,
            num_classes=len(config["data"]["class_names"]),
            img_size=model_config.get("backbone_image_size", 518),
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        embed_dim = getattr(self.backbone, "embed_dim", self.backbone.num_features)
        self.head = MulticlassSEHead(embed_dim, len(config["data"]["class_names"]))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.backbone.forward_features(images)
        if isinstance(output, dict):
            output = next(
                (output[key] for key in ("x", "last_hidden_state", "features") if key in output),
                next(iter(output.values())),
            )
        elif isinstance(output, (list, tuple)):
            output = output[-1]
        return self.head(output, getattr(self.backbone, "num_prefix_tokens", 1))


def build_model(config: dict) -> nn.Module:
    task = config["task"]
    if task == "binary":
        return BinaryClassifier(config)
    if task == "multiclass":
        return MulticlassClassifier(config)
    raise ValueError(f"Unsupported task: {task}")


def load_checkpoint(model: nn.Module, checkpoint: str | Path) -> tuple[list[str], list[str]]:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    allowed_unexpected = (
        "backbone.mask_token",
        "blocks.",
        "norm.",
    )
    bad_unexpected = [key for key in unexpected if not key.startswith(allowed_unexpected)]
    if missing or bad_unexpected:
        raise RuntimeError(
            f"Checkpoint is incompatible. missing={missing}, unexpected={bad_unexpected}"
        )
    return list(missing), list(unexpected)
