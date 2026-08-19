from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import os
import sys

import torch
from torch import nn
import yaml

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SmallCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 4,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        layers = []
        channels = base_channels
        current_in = in_channels
        for _ in range(num_blocks):
            layers.extend(
                [
                    nn.Conv2d(current_in, channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                ]
            )
            current_in = channels
            channels = min(channels * 2, base_channels * 4)
        self.features = nn.Sequential(*layers)
        self.out_channels = current_in
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_in, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward_head(self, feats: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(feats)
        return self.classifier(torch.flatten(pooled, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.forward_head(feats)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        children = list(backbone.children())
        # up to the last conv block (exclude avgpool and fc)
        self.feature_extractor = nn.Sequential(*children[:-2])
        self.out_channels = backbone.fc.in_features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        pooled = self.pool(feats)
        return torch.flatten(pooled, 1)


class CLIPFeatureExtractor(nn.Module):
    def __init__(self, visual, preprocess, resize: int = 224) -> None:
        super().__init__()
        self.visual = visual
        self.preprocess = preprocess
        self.target_size = resize

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.target_size and x.shape[-2] == self.target_size:
            return x
        return torch.nn.functional.interpolate(x, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resize(x)
        if hasattr(self.visual, "stem") and hasattr(self.visual, "layer4"):
            h = self.visual.stem(x)
            h = self.visual.layer1(h)
            h = self.visual.layer2(h)
            h = self.visual.layer3(h)
            h = self.visual.layer4(h)
            return h
        feats = self.visual.forward(x)
        return feats.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return feats.mean(dim=(-2, -1))


class _FirstStageEncoder(nn.Module):
    """Encoder wrapper that mimics encode_first_stage without sampling."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        self.post_quant_conv = vae.post_quant_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.quant_conv(h)
        return self.post_quant_conv(h)


@dataclass
class ModelConfig:
    constructor: type

    def build(self, **kwargs):
        return self.constructor(**kwargs)


def _build_resnet50(**kwargs) -> nn.Module:
    try:
        from torchvision import models
    except ImportError as exc:
        raise ImportError("torchvision is required to use the resnet50 MPE function.") from exc
    # Support both old (pretrained=True) and new weights enum APIs.
    if hasattr(models, "ResNet50_Weights"):
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)
    else:
        backbone = models.resnet50(pretrained=True)
    return ResNetFeatureExtractor(backbone)


def _build_clip_resnet50(**kwargs) -> nn.Module:
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError("open_clip_torch is required to use clip_resnet50.") from exc

    model, _, preprocess = open_clip.create_model_and_transforms("RN50", pretrained="openai")
    visual = model.visual
    extractor = CLIPFeatureExtractor(visual, preprocess)
    channels = getattr(getattr(visual.layer4[-1], "bn3", visual), "num_features", getattr(visual, "output_dim", 1024))
    extractor.out_channels = channels
    return extractor


AVAILABLE_MODELS: Dict[str, ModelConfig] = {
    "small_cnn": ModelConfig(SmallCNN),
    "resnet50": ModelConfig(_build_resnet50),
    "clip_resnet50": ModelConfig(_build_clip_resnet50),
}


def build_model(name: str, **kwargs) -> nn.Module:
    key = name.lower()
    if key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(AVAILABLE_MODELS)}")
    return AVAILABLE_MODELS[key].build(**kwargs)


def load_checkpoint_encoder(checkpoint_path: str, device: Optional[torch.device] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device or "cpu")
    arch = ckpt.get("arch", "small_cnn")
    channels = ckpt.get("channels", 3)
    base_channels = ckpt.get("base_channels", 64)
    num_blocks = ckpt.get("num_blocks", 4)
    num_classes = ckpt.get("num_classes", 4)
    model = build_model(
        arch,
        in_channels=channels,
        base_channels=base_channels,
        num_blocks=num_blocks,
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt["model_state"])
    if device is not None:
        model.to(device)
    model.eval()
    metadata = dict(ckpt)
    metadata.update(
        {
            "arch": arch,
            "channels": channels,
            "base_channels": base_channels,
            "num_blocks": num_blocks,
            "num_classes": num_classes,
            "checkpoint": checkpoint_path,
        }
    )
    return model, metadata


def load_named_encoder(model_name: str, device: Optional[torch.device] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    lower = model_name.lower()
    if lower == "resnet50":
        model = build_model("resnet50")
        metadata = {
            "arch": "resnet50",
            "channels": 3,
            "image_size": 256,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        }
    elif lower == "clip_resnet50":
        model = build_model("clip_resnet50")
        metadata = {
            "arch": "clip_resnet50",
            "channels": 3,
            "image_size": 224,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        }
    else:
        raise ValueError(f"Unsupported named model '{model_name}'.")
    if device is not None:
        model.to(device)
    model.eval()
    return model, metadata


LDM_ROOT_ENV = "EQUIREG_LDM_ROOT"


def _default_ldm_root() -> Optional[Path]:
    """Locate a CompVis latent-diffusion checkout.

    This repository vendors no upstream code, so the LDM package must come from
    the user's own clone. Set EQUIREG_LDM_ROOT to the directory that CONTAINS
    the `ldm` package (i.e. the repository root of latent-diffusion or PSLD),
    or pass ldm_root= explicitly.
    """
    env_root = os.environ.get(LDM_ROOT_ENV)
    if env_root:
        candidate = Path(env_root).expanduser()
        if (candidate / "ldm").is_dir():
            return candidate
        raise FileNotFoundError(
            f"{LDM_ROOT_ENV}={env_root!r} does not contain an 'ldm' package directory."
        )
    return None


def load_ldm_autoencoder(
    config_path: str,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    ldm_root: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    search_root = Path(ldm_root).expanduser() if ldm_root else _default_ldm_root()
    if search_root:
        resolved_root = search_root.resolve()
        if not resolved_root.exists():
            raise FileNotFoundError(f"Specified ldm_root '{resolved_root}' does not exist.")
        if str(resolved_root) not in sys.path:
            sys.path.append(str(resolved_root))
    try:
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config
    except ImportError as exc:
        raise ImportError(
            "Loading an LDM autoencoder MPE function requires a CompVis "
            "latent-diffusion checkout on sys.path. Set EQUIREG_LDM_ROOT to the "
            "repository root of your latent-diffusion or PSLD clone, "
            "or pass ldm_root=. EquiReg vendors no upstream code — see "
            "the README."
        ) from exc

    config = OmegaConf.load(config_path)
    pl_sd = torch.load(checkpoint_path, map_location="cpu")
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    vae = model.first_stage_model
    encoder = _FirstStageEncoder(vae)
    encoder.eval()
    encoder.requires_grad_(False)
    if device is not None:
        encoder.to(device)
    # release reference to the heavy diffusion model
    del model
    metadata = {
        "arch": "ldm_autoencoder",
        "config": config_path,
        "checkpoint": checkpoint_path,
    }
    return encoder, metadata


def _resolve_path(path_value: str, base_dir: Path) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def load_mpe_from_config(config_path: str, device: Optional[torch.device] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    config_file = Path(config_path).expanduser().resolve()
    base_dir = config_file.parent
    with open(config_file, "r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle) or {}
    mpe_type = spec.get("type", "named").lower()

    if mpe_type == "named":
        name = spec.get("name")
        if not name:
            raise ValueError("Named MPE config requires a 'name' field.")
        return load_named_encoder(name, device)

    if mpe_type == "checkpoint":
        checkpoint = spec.get("checkpoint")
        if not checkpoint:
            raise ValueError("Checkpoint MPE config requires a 'checkpoint' field.")
        checkpoint_path = _resolve_path(checkpoint, base_dir)
        return load_checkpoint_encoder(checkpoint_path, device)

    if mpe_type in {"ldm_autoencoder", "ldm_encoder"}:
        config_file_path = spec.get("config")
        checkpoint = spec.get("checkpoint")
        if not config_file_path or not checkpoint:
            raise ValueError("LDM autoencoder config requires 'config' and 'checkpoint' fields.")
        resolved_config = _resolve_path(config_file_path, base_dir)
        resolved_ckpt = _resolve_path(checkpoint, base_dir)
        ldm_root = spec.get("ldm_root")
        resolved_root = _resolve_path(ldm_root, base_dir) if ldm_root else None
        return load_ldm_autoencoder(resolved_config, resolved_ckpt, device, ldm_root=resolved_root)

    raise ValueError(f"Unsupported MPE config type '{mpe_type}'.")
