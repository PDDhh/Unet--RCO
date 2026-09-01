"""
Training script for Region + Center + Offset (RCO) U-Net++.

The default settings in this public version are aligned with the training
configuration reported in the manuscript:
- 1024 x 1024 grayscale input
- U-Net++ / NestedUNet with four RCO output channels
- 5-fold image-level cross-validation
- 200 epochs, batch size 2
- AdamW (lr=1e-3, weight_decay=1e-4)
- CosineAnnealingLR (eta_min=1e-5)
- GroupNorm (8 groups), base_channels=32
- Region / Center / Offset loss weights = 1.0 / 1.0 / 1.0
- Training-only augmentation: horizontal/vertical flips, brightness/contrast
  perturbation, and Gaussian noise

The test split is never used for model selection. The checkpoint with the
highest validation IoU is retained as the default model for inference.
"""

from __future__ import annotations

import argparse
import os
import random
from contextlib import nullcontext
from collections import OrderedDict
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from rco import architectures


# ------------------ utils ------------------ #
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_seed(seed: int = 41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.cnt += n

    @property
    def avg(self):
        return self.sum / self.cnt if self.cnt > 0 else 0.0


# ------------------ loss and metrics ------------------ #
def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


class RegionCenterOffsetLoss(nn.Module):
    def __init__(
        self,
        lambda_region=1.0,
        lambda_center=1.0,
        lambda_offset=1.0,
        region_dice_weight=0.5,
        region_pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.lambda_region = lambda_region
        self.lambda_center = lambda_center
        self.lambda_offset = lambda_offset
        self.region_dice_weight = float(region_dice_weight)

        if region_pos_weight is None or region_pos_weight <= 0:
            region_pos_weight = 1.0
        self.register_buffer("region_pos_weight", torch.tensor([float(region_pos_weight)], dtype=torch.float32))

    def forward(self, output, target):
        """
        output: (B, 4, H, W) logits
        target: (B, 4, H, W) [region, center, offset_y, offset_x]
        """
        region_logit = output[:, 0:1]
        center_logit = output[:, 1:2]
        offset_pred = output[:, 2:]

        region_gt = target[:, 0:1]
        center_gt = target[:, 1:2]
        offset_gt = target[:, 2:]

        bce_region = F.binary_cross_entropy_with_logits(
            region_logit,
            region_gt,
            pos_weight=self.region_pos_weight.to(output.device),
        )
        dice_region = dice_loss_with_logits(region_logit, region_gt)
        # The paper uses an equally weighted combination when region_dice_weight=0.5.
        loss_region = (1.0 - self.region_dice_weight) * bce_region + self.region_dice_weight * dice_region

        center_pred = torch.sigmoid(center_logit)
        region_mask = (region_gt > 0.5).float()
        denom = region_mask.sum().clamp_min(1.0)

        diff_center = (center_pred - center_gt) * region_mask
        loss_center = diff_center.abs().sum() / denom

        region_mask2 = region_mask.expand_as(offset_pred)
        denom2 = region_mask2.sum().clamp_min(1.0)
        diff_offset = (offset_pred - offset_gt) * region_mask2
        loss_offset = (diff_offset ** 2).sum() / denom2

        loss = (
            self.lambda_region * loss_region
            + self.lambda_center * loss_center
            + self.lambda_offset * loss_offset
        )

        return loss, {
            "loss_region": loss_region.detach(),
            "loss_region_bce": bce_region.detach(),
            "loss_region_dice": dice_region.detach(),
            "loss_center": loss_center.detach(),
            "loss_offset": loss_offset.detach(),
        }


def region_metrics(output, target, thr=0.5, eps=1e-6):
    region_logit = output[:, 0:1]
    region_gt = target[:, 0:1]

    prob = torch.sigmoid(region_logit)
    pred = (prob > thr).float()
    gt = (region_gt > 0.5).float()

    inter = (pred * gt).flatten(1).sum(dim=1)
    pred_sum = pred.flatten(1).sum(dim=1)
    gt_sum = gt.flatten(1).sum(dim=1)
    union = pred_sum + gt_sum - inter

    iou = ((inter + eps) / (union + eps)).mean().item()
    dice = ((2.0 * inter + eps) / (pred_sum + gt_sum + eps)).mean().item()
    return iou, dice


def compute_region_pos_weight(img_ids: List[str], inst_dir: str) -> float:
    pos_count = 0.0
    neg_count = 0.0
    for img_id in img_ids:
        data = np.load(os.path.join(inst_dir, img_id + ".npz"))
        region = data["region"] > 0.5
        pos_count += float(region.sum())
        neg_count += float(region.size - region.sum())
    if pos_count <= 0:
        return 1.0
    return max(neg_count / pos_count, 1e-6)


# ------------------ dataset ------------------ #
class InstanceDataset(Dataset):
    """Dataset for grayscale SEM images and RCO supervision targets.

    Random augmentation is applied only when ``training=True``. Geometric
    transformations are applied synchronously to the image and all RCO target
    maps. Offset signs are corrected after flips so that the vector field
    remains physically consistent.
    """

    def __init__(
        self,
        img_ids,
        img_dir,
        inst_dir,
        img_ext=".png",
        input_h=1024,
        input_w=1024,
        training=False,
        hflip_prob=0.25,
        vflip_prob=0.25,
        intensity_aug_prob=0.80,
        contrast_min=0.85,
        contrast_max=1.20,
        brightness_min=-18.0,
        brightness_max=18.0,
        noise_prob=0.25,
        noise_std_min=2.0,
        noise_std_max=8.0,
    ):
        self.img_ids = list(img_ids)
        self.img_dir = img_dir
        self.inst_dir = inst_dir
        self.img_ext = img_ext
        self.input_h = input_h
        self.input_w = input_w
        self.training = bool(training)

        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.intensity_aug_prob = float(intensity_aug_prob)
        self.contrast_min = float(contrast_min)
        self.contrast_max = float(contrast_max)
        self.brightness_min = float(brightness_min)
        self.brightness_max = float(brightness_max)
        self.noise_prob = float(noise_prob)
        self.noise_std_min = float(noise_std_min)
        self.noise_std_max = float(noise_std_max)

    def __len__(self):
        return len(self.img_ids)

    def _augment(self, img, region, center, offset_y, offset_x):
        # Horizontal flip: x-direction offset changes sign.
        if random.random() < self.hflip_prob:
            img = np.fliplr(img)
            region = np.fliplr(region)
            center = np.fliplr(center)
            offset_y = np.fliplr(offset_y)
            offset_x = -np.fliplr(offset_x)

        # Vertical flip: y-direction offset changes sign.
        if random.random() < self.vflip_prob:
            img = np.flipud(img)
            region = np.flipud(region)
            center = np.flipud(center)
            offset_y = -np.flipud(offset_y)
            offset_x = np.flipud(offset_x)

        # Brightness / contrast perturbation on the 8-bit image.
        if random.random() < self.intensity_aug_prob:
            alpha = random.uniform(self.contrast_min, self.contrast_max)
            beta = random.uniform(self.brightness_min, self.brightness_max)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0.0, 255.0)
        else:
            img = img.astype(np.float32)

        # Zero-mean Gaussian noise.
        if random.random() < self.noise_prob:
            sigma = random.uniform(self.noise_std_min, self.noise_std_max)
            noise = np.random.normal(0.0, sigma, size=img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 255.0)

        # np.flip* may produce negative strides, which torch.from_numpy rejects.
        return tuple(np.ascontiguousarray(x) for x in (img, region, center, offset_y, offset_x))

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        npz_path = os.path.join(self.inst_dir, img_id + ".npz")

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        data = np.load(npz_path)
        region = data["region"].astype(np.float32)
        center = data["center"].astype(np.float32)
        offset_y = data["offset_y"].astype(np.float32)
        offset_x = data["offset_x"].astype(np.float32)

        if not (region.shape == center.shape == offset_y.shape == offset_x.shape == img.shape):
            raise ValueError(
                f"Shape mismatch for {img_id}: img{img.shape}, region{region.shape}, "
                f"center{center.shape}, offset_y{offset_y.shape}, offset_x{offset_x.shape}"
            )

        h, w = img.shape
        if (h != self.input_h) or (w != self.input_w):
            raise ValueError(
                f"Image/target size {img.shape} != expected ({self.input_h}, {self.input_w}). "
                "Crop images and targets to the configured input size before training."
            )

        if self.training:
            img, region, center, offset_y, offset_x = self._augment(
                img, region, center, offset_y, offset_x
            )
        else:
            img = img.astype(np.float32)

        # Manuscript normalization: 8-bit -> [0, 1] -> approximately [-1, 1].
        img = img / 255.0
        img = (img - 0.5) / 0.5
        img = np.ascontiguousarray(img[None, ...], dtype=np.float32)

        target = np.concatenate(
            [
                region[None, ...],
                center[None, ...],
                offset_y[None, ...],
                offset_x[None, ...],
            ],
            axis=0,
        )
        target = np.ascontiguousarray(target, dtype=np.float32)

        return torch.from_numpy(img), torch.from_numpy(target), img_id


# ------------------ arguments ------------------ #
def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Optional YAML file with training argument defaults.")

    parser.add_argument("--name", default=None, help="model name: default dataset+arch+RCO")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("-b", "--batch_size", default=2, type=int)

    # model
    parser.add_argument("--arch", "-a", default="NestedUNet", help="Architecture name from rco.architectures.")
    parser.add_argument("--deep_supervision", default=False, type=str2bool)
    parser.add_argument("--input_channels", default=1, type=int)
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--input_w", default=1024, type=int)
    parser.add_argument("--input_h", default=1024, type=int)

    # architecture memory controls
    parser.add_argument("--base_channels", default=32, type=int, help="Base channel width for U-Net++.")
    parser.add_argument("--filters", default=None, type=str, help="Override channels, e.g. 8,16,32,64,128,256")
    parser.add_argument("--use_checkpoint", default=True, type=str2bool, help="Gradient checkpointing to reduce memory.")
    parser.add_argument("--use_bn", default=True, type=str2bool)
    parser.add_argument("--norm_type", default="group", choices=["batch", "group", "instance", "none"],
                        help="Normalization used in rco.architectures.")
    parser.add_argument("--num_groups", default=8, type=int, help="Group count for GroupNorm.")
    parser.add_argument("--align_corners", default=False, type=str2bool)

    # dataset
    parser.add_argument("--dataset", default="gamma_gamma_prime", help="Dataset folder under inputs/.")
    parser.add_argument("--img_ext", default=".png")
    parser.add_argument("--inst_dir_name", default="inst_targets")
    parser.add_argument("--test_size", default=0.2, type=float)
    parser.add_argument("--val_size", default=0.1, type=float, help="Validation ratio inside the train_val split.")
    parser.add_argument("--random_state", default=41, type=int)

    # training-only data augmentation
    parser.add_argument("--hflip_prob", default=0.25, type=float)
    parser.add_argument("--vflip_prob", default=0.25, type=float)
    parser.add_argument("--intensity_aug_prob", default=0.80, type=float)
    parser.add_argument("--contrast_min", default=0.85, type=float)
    parser.add_argument("--contrast_max", default=1.20, type=float)
    parser.add_argument("--brightness_min", default=-18.0, type=float)
    parser.add_argument("--brightness_max", default=18.0, type=float)
    parser.add_argument("--noise_prob", default=0.25, type=float)
    parser.add_argument("--noise_std_min", default=2.0, type=float)
    parser.add_argument("--noise_std_max", default=8.0, type=float)

    # cross-validation / external split controls
    parser.add_argument("--kfold", default=True, type=str2bool,
                        help="Use KFold image-level split. Run one fold at a time with --fold.")
    parser.add_argument("--n_splits", default=5, type=int, help="Number of folds for KFold.")
    parser.add_argument("--fold", default=1, type=int, help="Fold index, 1-based. Example: 1..5.")
    parser.add_argument("--append_fold_to_name", default=True, type=str2bool,
                        help="Append _foldN to model name when --kfold True to avoid overwriting.")
    parser.add_argument("--train_id_list", default=None, type=str,
                        help="Optional txt file containing train image ids, one id per line, without extension.")
    parser.add_argument("--val_id_list", default=None, type=str,
                        help="Optional txt file containing val image ids, one id per line, without extension.")
    parser.add_argument("--test_id_list", default=None, type=str,
                        help="Optional txt file containing test image ids, one id per line, without extension.")

    # optimizer
    parser.add_argument("--optimizer", default="AdamW", choices=["Adam", "AdamW", "SGD"])
    parser.add_argument("--lr", "--learning_rate", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--nesterov", default=False, type=str2bool)

    # scheduler
    parser.add_argument(
        "--scheduler",
        default="CosineAnnealingLR",
        choices=["CosineAnnealingLR", "ReduceLROnPlateau", "MultiStepLR", "ConstantLR"],
    )
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--factor", default=0.1, type=float)
    parser.add_argument("--patience", default=5, type=int, help="Scheduler patience for ReduceLROnPlateau.")
    parser.add_argument("--milestones", default="50,80", type=str)
    parser.add_argument("--gamma", default=0.1, type=float)
    parser.add_argument("--early_stopping", default=-1, type=int, help="Early stopping patience based on val loss/dice/iou improvement.")
    parser.add_argument("--model_selection", default="best_val_iou",
                        choices=["best_val_iou", "best_val_dice", "best_val_loss"],
                        help="Default checkpoint type suggested for prediction.")

    # memory/performance
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--amp", default=True, type=str2bool, help="Use mixed precision on CUDA.")
    parser.add_argument("--channels_last", default=True, type=str2bool)
    parser.add_argument("--grad_accumulation", default=1, type=int)
    parser.add_argument("--clip_grad", default=0.0, type=float)

    # loss weights
    parser.add_argument("--lambda_region", default=1.0, type=float)
    parser.add_argument("--lambda_center", default=1.0, type=float)
    parser.add_argument("--lambda_offset", default=1.0, type=float)
    parser.add_argument("--region_dice_weight", default=0.5, type=float, help="0=BCE only, 1=Dice only, 0.5=equally weighted BCE and Dice.")
    parser.add_argument("--auto_pos_weight", default=False, type=str2bool)
    parser.add_argument("--region_pos_weight", default=1.0, type=float)

    if config_args.config:
        with open(config_args.config, "r", encoding="utf-8") as f:
            defaults = yaml.safe_load(f) or {}
        if not isinstance(defaults, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {config_args.config}")
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(defaults) - valid_keys)
        if unknown_keys:
            raise ValueError(f"Unknown keys in config file {config_args.config}: {unknown_keys}")
        parser.set_defaults(**defaults)

    return parser.parse_args()


def _config_to_dict(config) -> Dict[str, Any]:
    return vars(config).copy()


def load_id_list(path: str) -> List[str]:
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if not item:
                continue
            # supports csv-like files with one id per line; ignores a simple header if present
            item = item.split(",")[0].strip()
            if item.lower() in {"id", "image_id", "train_images", "val_images", "test_images"}:
                continue
            ids.append(Path(item).stem)
    return ids


def validate_id_list(ids: List[str], all_ids: List[str], list_name: str):
    all_set = set(all_ids)
    missing = [i for i in ids if i not in all_set]
    if missing:
        raise ValueError(f"{list_name} contains ids that are not matched in images/inst_targets, e.g. {missing[:5]}")


def split_ids(ids: List[str], test_size: float, random_state: int):
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if len(ids) < 2:
        raise ValueError("Need at least 2 samples to create a split.")

    rng = np.random.RandomState(random_state)
    indices = rng.permutation(len(ids))
    num_test = int(np.ceil(len(ids) * test_size))
    if num_test <= 0 or num_test >= len(ids):
        raise ValueError(f"test_size={test_size} leaves an empty split for {len(ids)} samples.")

    test_indices = indices[:num_test]
    train_indices = indices[num_test:]
    return [ids[i] for i in train_indices], [ids[i] for i in test_indices]


def kfold_indices(num_samples: int, n_splits: int, random_state: int):
    indices = np.arange(num_samples)
    rng = np.random.RandomState(random_state)
    rng.shuffle(indices)
    fold_sizes = np.full(n_splits, num_samples // n_splits, dtype=int)
    fold_sizes[: num_samples % n_splits] += 1

    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        yield train_indices, test_indices
        current = stop


def build_split_ids(config, img_ids: List[str]):
    """
    Return train_ids, val_ids, test_ids using one of three modes:
    1) external id lists;
    2) KFold image-level split;
    3) original random train/val/test split.
    """
    external_lists = [config.train_id_list, config.val_id_list, config.test_id_list]
    if any(external_lists) and not all(external_lists):
        raise ValueError("--train_id_list, --val_id_list, and --test_id_list must be provided together.")

    if all(external_lists):
        train_ids = load_id_list(config.train_id_list)
        val_ids = load_id_list(config.val_id_list)
        test_ids = load_id_list(config.test_id_list)
        validate_id_list(train_ids, img_ids, "train_id_list")
        validate_id_list(val_ids, img_ids, "val_id_list")
        validate_id_list(test_ids, img_ids, "test_id_list")
        split_mode = "external_id_lists"
        return train_ids, val_ids, test_ids, split_mode

    if bool(config.kfold):
        if config.n_splits < 2:
            raise ValueError("--n_splits must be >= 2")
        if not (1 <= config.fold <= config.n_splits):
            raise ValueError(f"--fold must be in [1, {config.n_splits}], got {config.fold}")
        if len(img_ids) < config.n_splits:
            raise ValueError(f"Number of samples ({len(img_ids)}) must be >= n_splits ({config.n_splits})")

        folds = list(kfold_indices(len(img_ids), config.n_splits, config.random_state))
        train_val_idx, test_idx = folds[config.fold - 1]
        train_val_ids = [img_ids[i] for i in train_val_idx]
        test_ids = [img_ids[i] for i in test_idx]

        if len(train_val_ids) < 2:
            raise ValueError("Not enough train_val samples to create train/val split.")
        train_ids, val_ids = split_ids(
            train_val_ids,
            test_size=config.val_size,
            random_state=config.random_state + config.fold,
        )
        split_mode = f"kfold_{config.fold}_of_{config.n_splits}"
        return sorted(train_ids), sorted(val_ids), sorted(test_ids), split_mode

    train_val_ids, test_ids = split_ids(
        img_ids,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    train_ids, val_ids = split_ids(
        train_val_ids,
        test_size=config.val_size,
        random_state=config.random_state,
    )
    split_mode = "random_train_val_test"
    return sorted(train_ids), sorted(val_ids), sorted(test_ids), split_mode


def get_amp_context(device, amp_enabled: bool):
    # Old PyTorch compatible AMP context.
    if device.type == "cuda" and amp_enabled:
        return torch.cuda.amp.autocast()
    return nullcontext()


def _arch_kwargs(config) -> Dict[str, Any]:
    keys = ["deep_supervision", "base_channels", "filters", "use_checkpoint", "use_bn", "norm_type", "num_groups", "align_corners"]
    return {k: getattr(config, k) for k in keys if hasattr(config, k)}


def create_model(config):
    print(f"=> creating model {config.arch}")
    if not hasattr(architectures, config.arch):
        raise ValueError(f"rco.architectures does not contain architecture: {config.arch}")
    model_cls = getattr(architectures, config.arch)
    return model_cls(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        **_arch_kwargs(config),
    )


def get_model_output(output):
    if isinstance(output, (list, tuple)):
        return output[-1]
    return output


# ------------------ train / validate ------------------ #
def train_one_epoch(config, train_loader, model, criterion, optimizer, scaler, epoch, device):
    model.train()
    meter_loss = AverageMeter()
    meter_iou = AverageMeter()
    meter_dice = AverageMeter()
    meter_region = AverageMeter()
    meter_center = AverageMeter()
    meter_offset = AverageMeter()
    use_amp = bool(config.amp and device.type == "cuda")
    grad_accumulation = max(1, int(config.grad_accumulation))

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=len(train_loader), desc=f"Train {epoch}")
    for step, (img, target, _) in enumerate(train_loader, start=1):
        img = img.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if config.channels_last and device.type == "cuda":
            img = img.contiguous(memory_format=torch.channels_last)

        with get_amp_context(device, use_amp):
            output = get_model_output(model(img))
            loss, loss_dict = criterion(output, target)
            loss_for_backward = loss / grad_accumulation

        if scaler is not None:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        if step % grad_accumulation == 0 or step == len(train_loader):
            if config.clip_grad and config.clip_grad > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        iou, dice = region_metrics(output.detach(), target.detach())
        batch_size = img.size(0)
        meter_loss.update(loss.item(), batch_size)
        meter_iou.update(iou, batch_size)
        meter_dice.update(dice, batch_size)
        meter_region.update(loss_dict["loss_region"].item(), batch_size)
        meter_center.update(loss_dict["loss_center"].item(), batch_size)
        meter_offset.update(loss_dict["loss_offset"].item(), batch_size)

        pbar.set_postfix(OrderedDict([
            ("loss", f"{meter_loss.avg:.4f}"),
            ("iou", f"{meter_iou.avg:.4f}"),
            ("dice", f"{meter_dice.avg:.4f}"),
        ]))
        pbar.update(1)
    pbar.close()

    return OrderedDict([
        ("loss", meter_loss.avg),
        ("iou", meter_iou.avg),
        ("dice", meter_dice.avg),
        ("loss_region", meter_region.avg),
        ("loss_center", meter_center.avg),
        ("loss_offset", meter_offset.avg),
    ])


@torch.inference_mode()
def validate(config, val_loader, model, criterion, epoch, device, desc_prefix="Val"):
    model.eval()
    meter_loss = AverageMeter()
    meter_iou = AverageMeter()
    meter_dice = AverageMeter()
    meter_region = AverageMeter()
    meter_center = AverageMeter()
    meter_offset = AverageMeter()
    use_amp = bool(config.amp and device.type == "cuda")

    pbar = tqdm(total=len(val_loader), desc=f"{desc_prefix} {epoch}")
    for img, target, _ in val_loader:
        img = img.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if config.channels_last and device.type == "cuda":
            img = img.contiguous(memory_format=torch.channels_last)

        with get_amp_context(device, use_amp):
            output = get_model_output(model(img))
            loss, loss_dict = criterion(output, target)

        iou, dice = region_metrics(output, target)
        batch_size = img.size(0)
        meter_loss.update(loss.item(), batch_size)
        meter_iou.update(iou, batch_size)
        meter_dice.update(dice, batch_size)
        meter_region.update(loss_dict["loss_region"].item(), batch_size)
        meter_center.update(loss_dict["loss_center"].item(), batch_size)
        meter_offset.update(loss_dict["loss_offset"].item(), batch_size)

        pbar.set_postfix(OrderedDict([
            ("loss", f"{meter_loss.avg:.4f}"),
            ("iou", f"{meter_iou.avg:.4f}"),
            ("dice", f"{meter_dice.avg:.4f}"),
        ]))
        pbar.update(1)
    pbar.close()

    return OrderedDict([
        ("loss", meter_loss.avg),
        ("iou", meter_iou.avg),
        ("dice", meter_dice.avg),
        ("loss_region", meter_region.avg),
        ("loss_center", meter_center.avg),
        ("loss_offset", meter_offset.avg),
    ])


def save_checkpoint(path, model, optimizer, scheduler, epoch, config, train_log, val_log, best_metric_name, best_metric_value):
    checkpoint = {
        "epoch": epoch,
        "arch": config.arch,
        "num_classes": config.num_classes,
        "input_channels": config.input_channels,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": _config_to_dict(config),
        "train_log": dict(train_log),
        "val_log": dict(val_log),
        "best_metric_name": best_metric_name,
        "best_metric_value": float(best_metric_value),
    }
    torch.save(checkpoint, path)


def save_curves(log_df: pd.DataFrame, save_dir: Path):
    if log_df.empty:
        return

    curve_dir = save_dir / "training_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(log_df["epoch"], log_df["loss"], label="train_loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_dir / "loss_curve.png", dpi=300)
    plt.savefig(save_dir / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(log_df["epoch"], log_df["dice"], label="train_dice")
    plt.plot(log_df["epoch"], log_df["val_dice"], label="val_dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_dir / "dice_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(log_df["epoch"], log_df["iou"], label="train_iou")
    plt.plot(log_df["epoch"], log_df["val_iou"], label="val_iou")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_dir / "iou_curve.png", dpi=300)
    plt.savefig(save_dir / "metric_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(log_df["epoch"], log_df["lr"], label="learning_rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_dir / "learning_rate_curve.png", dpi=300)
    plt.close()

    component_cols = [
        ("loss_region", "train_region"),
        ("val_loss_region", "val_region"),
        ("loss_center", "train_center"),
        ("val_loss_center", "val_center"),
        ("loss_offset", "train_offset"),
        ("val_loss_offset", "val_offset"),
    ]
    if all(col in log_df.columns for col, _ in component_cols):
        plt.figure(figsize=(8, 5))
        for col, label in component_cols:
            plt.plot(log_df["epoch"], log_df[col], label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Component loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(curve_dir / "component_loss_curve.png", dpi=300)
        plt.close()


# ------------------ main ------------------ #
def main():
    config = parse_args()
    set_seed(config.random_state)

    if config.name is None:
        tag = "RCO"
        ds_tag = "wDS" if config.deep_supervision else "woDS"
        config.name = f"{config.dataset}_{config.arch}_{tag}_{ds_tag}_bc{config.base_channels}"

    if bool(config.kfold) and bool(config.append_fold_to_name):
        fold_suffix = f"_fold{config.fold}"
        if not str(config.name).endswith(fold_suffix):
            config.name = f"{config.name}{fold_suffix}"

    save_dir = Path("models") / config.name
    split_dir = save_dir / "splits"
    save_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    print("-" * 20)
    for k, v in vars(config).items():
        print(f"{k}: {v}")
    print("-" * 20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    img_dir = Path("inputs") / config.dataset / "images"
    inst_dir = Path("inputs") / config.dataset / config.inst_dir_name

    img_paths = glob(str(img_dir / ("*" + config.img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_paths]
    img_ids = sorted([i for i in img_ids if (inst_dir / f"{i}.npz").exists()])
    if len(img_ids) < 3:
        raise RuntimeError("Need at least 3 matched image/npz pairs for train/val/test split.")

    train_ids, val_ids, test_ids, split_mode = build_split_ids(config, img_ids)

    pd.Series(train_ids, name="image_id").to_csv(split_dir / "train_ids.txt", index=False, header=False)
    pd.Series(val_ids, name="image_id").to_csv(split_dir / "val_ids.txt", index=False, header=False)
    pd.Series(test_ids, name="image_id").to_csv(split_dir / "test_ids.txt", index=False, header=False)
    with open(split_dir / "split_info.yml", "w", encoding="utf-8") as f:
        yaml.dump({
            "split_mode": split_mode,
            "random_state": config.random_state,
            "kfold": bool(config.kfold),
            "n_splits": int(config.n_splits),
            "fold": int(config.fold),
            "val_size_inside_train_val": float(config.val_size),
            "num_total_matched": len(img_ids),
            "num_train": len(train_ids),
            "num_val": len(val_ids),
            "num_test": len(test_ids),
        }, f, sort_keys=False)
    print(f"Split mode: {split_mode}")
    print(f"Train images: {len(train_ids)} | Val images: {len(val_ids)} | Test images: {len(test_ids)}")

    if config.auto_pos_weight:
        config.region_pos_weight = compute_region_pos_weight(train_ids, str(inst_dir))
        print(f"Auto region_pos_weight: {config.region_pos_weight:.4f}")

    with open(save_dir / "config.yml", "w", encoding="utf-8") as f:
        yaml.dump(_config_to_dict(config), f, sort_keys=False)

    model = create_model(config).to(device)
    if config.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    criterion = RegionCenterOffsetLoss(
        lambda_region=config.lambda_region,
        lambda_center=config.lambda_center,
        lambda_offset=config.lambda_offset,
        region_dice_weight=config.region_dice_weight,
        region_pos_weight=config.region_pos_weight,
    ).to(device)

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "AdamW":
        optimizer = optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "SGD":
        optimizer = optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            nesterov=config.nesterov,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError

    if config.scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)
    elif config.scheduler == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=config.factor, patience=config.patience, min_lr=config.min_lr
        )
    elif config.scheduler == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(e) for e in config.milestones.split(",")],
            gamma=config.gamma,
        )
    elif config.scheduler == "ConstantLR":
        scheduler = None
    else:
        raise NotImplementedError

    dataset_common = dict(
        img_dir=str(img_dir),
        inst_dir=str(inst_dir),
        img_ext=config.img_ext,
        input_h=config.input_h,
        input_w=config.input_w,
    )
    train_dataset = InstanceDataset(
        train_ids,
        training=True,
        hflip_prob=config.hflip_prob,
        vflip_prob=config.vflip_prob,
        intensity_aug_prob=config.intensity_aug_prob,
        contrast_min=config.contrast_min,
        contrast_max=config.contrast_max,
        brightness_min=config.brightness_min,
        brightness_max=config.brightness_max,
        noise_prob=config.noise_prob,
        noise_std_min=config.noise_std_min,
        noise_std_max=config.noise_std_max,
        **dataset_common,
    )
    val_dataset = InstanceDataset(val_ids, training=False, **dataset_common)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(config.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(config.num_workers > 0),
    )

    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=True) if amp_enabled else None

    log = OrderedDict([
        ("epoch", []),
        ("lr", []),
        ("loss", []),
        ("iou", []),
        ("dice", []),
        ("loss_region", []),
        ("loss_center", []),
        ("loss_offset", []),
        ("val_loss", []),
        ("val_iou", []),
        ("val_dice", []),
        ("val_loss_region", []),
        ("val_loss_center", []),
        ("val_loss_offset", []),
    ])

    best_val_iou = -1.0
    best_val_dice = -1.0
    best_val_loss = float("inf")
    trigger = 0

    for epoch in range(1, config.epochs + 1):
        print(f"Epoch [{epoch}/{config.epochs}]")
        train_log = train_one_epoch(config, train_loader, model, criterion, optimizer, scaler, epoch, device)
        val_log = validate(config, val_loader, model, criterion, epoch, device, desc_prefix="Val")

        if scheduler is not None:
            if config.scheduler == "ReduceLROnPlateau":
                scheduler.step(val_log["loss"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            "lr %.6f - loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f"
            % (current_lr, train_log["loss"], train_log["iou"], train_log["dice"], val_log["loss"], val_log["iou"], val_log["dice"])
        )

        log["epoch"].append(epoch)
        log["lr"].append(current_lr)
        log["loss"].append(train_log["loss"])
        log["iou"].append(train_log["iou"])
        log["dice"].append(train_log["dice"])
        log["loss_region"].append(train_log["loss_region"])
        log["loss_center"].append(train_log["loss_center"])
        log["loss_offset"].append(train_log["loss_offset"])
        log["val_loss"].append(val_log["loss"])
        log["val_iou"].append(val_log["iou"])
        log["val_dice"].append(val_log["dice"])
        log["val_loss_region"].append(val_log["loss_region"])
        log["val_loss_center"].append(val_log["loss_center"])
        log["val_loss_offset"].append(val_log["loss_offset"])
        log_df = pd.DataFrame(log)
        log_df.to_csv(save_dir / "log.csv", index=False)
        save_curves(log_df, save_dir)

        save_checkpoint(
            save_dir / "model_latest.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            config,
            train_log,
            val_log,
            "latest",
            val_log["loss"],
        )

        improved = False
        if val_log["iou"] > best_val_iou:
            best_val_iou = val_log["iou"]
            trigger = 0
            improved = True
            save_checkpoint(
                save_dir / "model_best_val_iou.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                train_log,
                val_log,
                "val_iou",
                best_val_iou,
            )
            torch.save(model.state_dict(), save_dir / "model.pth")
            print(f"=> saved best val_iou model: {best_val_iou:.4f}")

        if val_log["dice"] > best_val_dice:
            best_val_dice = val_log["dice"]
            trigger = 0
            improved = True
            save_checkpoint(
                save_dir / "model_best_val_dice.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                train_log,
                val_log,
                "val_dice",
                best_val_dice,
            )
            torch.save(model.state_dict(), save_dir / "model_val_dice.pth")
            print(f"=> saved best val_dice model: {best_val_dice:.4f}")

        if val_log["loss"] < best_val_loss:
            best_val_loss = val_log["loss"]
            trigger = 0
            improved = True
            save_checkpoint(
                save_dir / "model_best_val_loss.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                train_log,
                val_log,
                "val_loss",
                best_val_loss,
            )
            torch.save(model.state_dict(), save_dir / "model_val_loss.pth")
            print(f"=> saved best val_loss model: {best_val_loss:.4f}")

        if not improved:
            trigger += 1

        if config.early_stopping >= 0 and trigger >= config.early_stopping:
            print("=> early stopping")
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("Training finished.")
    print(f"Best val_iou:  {best_val_iou:.4f}")
    print(f"Best val_dice: {best_val_dice:.4f}")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Suggested model_selection for prediction: {config.model_selection}")
    print(f"Splits saved to: {split_dir}")
    print(f"Training curves saved to: {save_dir / 'training_curves'}")


if __name__ == "__main__":
    main()
