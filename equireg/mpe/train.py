from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .models import build_model
from .data import (
    build_contrastive_dataset,
    build_dataset,
    limit_dataset,
    split_dataset,
    wrap_with_group,
    wrap_with_rotation,
)
from ..groups import get_group


def _expand_stat(values: List[float] | None, channels: int, default: float) -> List[float]:
    if values is None:
        return [default] * channels
    if len(values) == 1:
        return values * channels
    if len(values) != channels:
        raise ValueError(f"Expected 1 or {channels} stats, got {len(values)}.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN to act as an MPE function.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (imagenet, mnist, ffhq256, ...).")
    parser.add_argument("--data_path", type=str, default=None, help="Directory that stores the dataset images.")
    parser.add_argument("--output_dir", type=str, default="models/MPE", help="Directory to save checkpoints.")
    parser.add_argument("--save_name", type=str, default=None, help="Optional checkpoint name.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--group", type=str, default="rot90", help="Transformation group used for training.")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit training dataset size.")
    parser.add_argument(
        "--limit_strategy",
        type=str,
        default="random",
        choices=["random", "first"],
        help="How to pick samples when limiting the training set.",
    )
    parser.add_argument("--model", type=str, default="small_cnn")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalize_mean", type=float, nargs="+", default=None)
    parser.add_argument("--normalize_std", type=float, nargs="+", default=None)
    parser.add_argument(
        "--objective",
        type=str,
        choices=["classification", "equivariance", "rotation", "contrastive"],
        default="classification",
        help="Training objective for the CNN.",
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for contrastive loss.")
    parser.add_argument("--projection_dim", type=int, default=128, help="Projection head output dimension.")
    parser.add_argument("--projection_hidden_dim", type=int, default=256, help="Projection head hidden size.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch_classification(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples


def train_epoch_equivariance(model, loader, optimizer, mse_loss, group, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images in loader:
        if isinstance(images, (list, tuple)):
            images = images[0]
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad()
        base_feats = model.forward_features(images)
        if group.order > 1:
            op_idx = torch.randint(1, group.order, (1,)).item()
        else:
            op_idx = 0
        transformed_inputs = group.apply_to_input(images, op_idx)
        transformed_feats = model.forward_features(transformed_inputs)
        projected_feats = group.apply_to_feature(base_feats, op_idx)
        loss = mse_loss(transformed_feats, projected_feats)
        loss.backward()
        optimizer.step()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples


def info_nce_loss(z1, z2, temperature):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    similarity = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, -float("inf"))
    labels = torch.arange(2 * batch_size, device=z.device)
    positives = (labels + batch_size) % (2 * batch_size)
    loss = F.cross_entropy(similarity, positives)
    return loss


def train_epoch_contrastive(model, projector, loader, optimizer, temperature, device):
    model.train()
    projector.train()
    total_loss = 0.0
    total_samples = 0
    for view1, view2 in loader:
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        optimizer.zero_grad()
        feats1 = model.forward_features(view1)
        feats2 = model.forward_features(view2)
        z1 = projector(feats1)
        z2 = projector(feats2)
        loss = info_nce_loss(z1, z2, temperature)
        loss.backward()
        optimizer.step()
        batch_size = view1.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples


def evaluate_contrastive(model, projector, loader, temperature, device):
    if loader is None:
        return None
    model.eval()
    projector.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for view1, view2 in loader:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            feats1 = model.forward_features(view1)
            feats2 = model.forward_features(view2)
            z1 = projector(feats1)
            z2 = projector(feats2)
            loss = info_nce_loss(z1, z2, temperature)
            batch_size = view1.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples


def evaluate_classification(model, loader, criterion, device):
    if loader is None:
        return None, None
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size
    return total_loss / total_samples, total_correct / total_samples


def evaluate_equivariance(model, loader, mse_loss, group, device):
    if loader is None:
        return None
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images in loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device, non_blocking=True)
            base_feats = model.forward_features(images)
            if group.order > 1:
                op_idx = torch.randint(1, group.order, (1,)).item()
            else:
                op_idx = 0
            transformed_inputs = group.apply_to_input(images, op_idx)
            transformed_feats = model.forward_features(transformed_inputs)
            projected_feats = group.apply_to_feature(base_feats, op_idx)
            loss = mse_loss(transformed_feats, projected_feats)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    group = get_group(args.group)
    if args.objective == "rotation" and group.order != 4:
        raise ValueError("Rotation objective requires a 4-element rotational group (e.g., rot90).")

    mean_vals = _expand_stat(args.normalize_mean, args.channels, 0.5)
    std_vals = _expand_stat(args.normalize_std, args.channels, 0.5)

    if args.objective == "contrastive":
        base_dataset = build_contrastive_dataset(
            name=args.dataset,
            data_path=args.data_path,
            image_size=args.image_size,
            channels=args.channels,
            mean=mean_vals,
            std=std_vals,
        )
    else:
        base_dataset = build_dataset(
            name=args.dataset,
            data_path=args.data_path,
            image_size=args.image_size,
            channels=args.channels,
            mean=mean_vals,
            std=std_vals,
        )
    base_dataset = limit_dataset(
        base_dataset, args.max_train_samples, seed=args.seed, strategy=args.limit_strategy
    )
    train_dataset, val_dataset = split_dataset(base_dataset, args.val_fraction, seed=args.seed)
    val_loader = None
    if args.objective == "classification":
        train_dataset = wrap_with_group(train_dataset, args.group)
        if val_dataset is not None and len(val_dataset) > 0:
            val_dataset = wrap_with_group(val_dataset, args.group)
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
            )
    elif args.objective == "rotation":
        train_dataset = wrap_with_rotation(train_dataset)
        if val_dataset is not None and len(val_dataset) > 0:
            val_dataset = wrap_with_rotation(val_dataset)
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
            )
    elif args.objective == "contrastive" and val_dataset is not None and len(val_dataset) > 0:
        drop_last_val = len(val_dataset) >= args.batch_size
        if not drop_last_val:
            print(
                f"[WARN] Contrastive val dataset has only {len(val_dataset)} samples; "
                f"disabling drop_last so evaluation can run."
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=drop_last_val,
        )
    elif val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
        )

    drop_last_train = args.objective == "contrastive" and len(train_dataset) >= args.batch_size
    if args.objective == "contrastive" and not drop_last_train:
        print(
            f"[WARN] Contrastive train dataset has only {len(train_dataset)} samples "
            f"(< batch size {args.batch_size}); disabling drop_last."
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last_train,
    )

    model = build_model(
        args.model,
        in_channels=args.channels,
        num_classes=group.order,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    projector = None
    if args.objective == "contrastive":
        projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(model.out_channels, args.projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.projection_hidden_dim, args.projection_dim),
        ).to(device)
        trainable_parameters = list(model.parameters()) + list(projector.parameters())
    else:
        trainable_parameters = model.parameters()
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        if args.objective == "classification":
            train_loss, train_acc = train_epoch_classification(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate_classification(model, val_loader, criterion, device)
            log_msg = f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%"
            if val_loss is not None:
                log_msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        elif args.objective == "rotation":
            train_loss, train_acc = train_epoch_classification(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate_classification(model, val_loader, criterion, device)
            log_msg = f"[Epoch {epoch:03d}] Train Rot Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%"
            if val_loss is not None:
                log_msg += f" | Val Rot Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        elif args.objective == "contrastive":
            train_loss = train_epoch_contrastive(model, projector, train_loader, optimizer, args.temperature, device)
            val_loss = evaluate_contrastive(model, projector, val_loader, args.temperature, device)
            log_msg = f"[Epoch {epoch:03d}] Train Contrastive Loss: {train_loss:.6f}"
            if val_loss is not None:
                log_msg += f" | Val Contrastive Loss: {val_loss:.6f}"
        else:
            train_loss = train_epoch_equivariance(model, train_loader, optimizer, mse_loss, group, device)
            val_loss = evaluate_equivariance(model, val_loader, mse_loss, group, device)
            log_msg = f"[Epoch {epoch:03d}] Train Equiv Loss: {train_loss:.6f}"
            if val_loss is not None:
                log_msg += f" | Val Equiv Loss: {val_loss:.6f}"
        print(log_msg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = args.save_name or f"{args.dataset}_CNN.pt"
    ckpt_path = output_dir / ckpt_name
    checkpoint = {
        "model_state": model.state_dict(),
        "arch": args.model,
        "dataset": args.dataset,
        "group": args.group,
        "num_classes": group.order,
        "objective": args.objective,
        "temperature": args.temperature,
        "projection_dim": args.projection_dim,
        "projection_hidden_dim": args.projection_hidden_dim,
        "image_size": args.image_size,
        "channels": args.channels,
        "mean": mean_vals,
        "std": std_vals,
        "base_channels": args.base_channels,
        "num_blocks": args.num_blocks,
    }
    if projector is not None:
        checkpoint["projection_state"] = projector.state_dict()
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
