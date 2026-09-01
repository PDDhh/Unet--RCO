"""
Optimized prediction script for Region + Center + Offset U-Net++.

Main upgrades:
- Rebuilds the model directly from models/<name>/config.yml, including architecture parameters.
- Can automatically read train/val/test split files saved by scripts.train_rco.
- Saves phase masks by default: gamma-prime = black 0, gamma = white 255.
- Optional gt-dir evaluation writes per-image Precision/Recall/Dice/IoU and mean/std summaries.
- Optional validation threshold search selects and applies the best threshold by Dice or IoU.
"""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from glob import glob
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from rco import architectures


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for region+center+offset UNet++")
    parser.add_argument("--name", required=True, help="Model folder name under models/")
    parser.add_argument("--model", default="auto", help="Checkpoint file name/full path. Use 'auto' with --model-selection.")
    parser.add_argument("--model-selection", default=None,
                        choices=["best_val_iou", "best_val_dice", "best_val_loss", "latest"],
                        help="Checkpoint selected when --model auto. Defaults to config.yml model_selection or best_val_iou.")
    parser.add_argument("--predict-dir", type=str, required=True, help="Directory containing images for prediction")
    parser.add_argument("--save-dir", default=None, help="Default: outputs/<name>/<model_stem>/<split_name>")
    parser.add_argument("--split-name", default="predict", help="train, val, test, or predict")
    parser.add_argument("--id-list", default=None, help="Optional txt/csv id list. Use 'auto' to read models/<name>/splits/<split>_ids.txt")
    parser.add_argument("--gt-dir", default=None, help="Optional gt dir. Prefer inst_targets with <id>.npz containing region.")

    parser.add_argument("--region-thr", type=float, default=0.5)
    parser.add_argument("--min-size", type=int, default=50)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true", default=True)
    parser.add_argument("--no-channels-last", action="store_false", dest="channels_last")

    parser.add_argument("--no-enhance", action="store_true", help="Disable brightness/contrast enhancement")
    parser.add_argument("--save-phase-mask", action="store_true", default=True)
    parser.add_argument("--no-save-phase-mask", action="store_false", dest="save_phase_mask")
    parser.add_argument("--save-prob", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--save-npz", action="store_true")

    parser.add_argument("--uncertain-low", type=float, default=0.35)
    parser.add_argument("--uncertain-high", type=float, default=0.65)
    parser.add_argument("--aux-alpha", type=float, default=0.15)
    parser.add_argument("--aux-beta", type=float, default=0.15)
    parser.add_argument("--offset-kernel", type=int, default=11)

    parser.add_argument("--search-threshold", action="store_true", help="Search best region threshold. Requires --gt-dir.")
    parser.add_argument("--thr-min", type=float, default=0.30)
    parser.add_argument("--thr-max", type=float, default=0.70)
    parser.add_argument("--thr-step", type=float, default=0.05)
    parser.add_argument("--select-metric", choices=["Dice", "IoU"], default="IoU")
    return parser.parse_args()


# -----------------------------
# config / model loading
# -----------------------------
def get_arch_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["deep_supervision", "base_channels", "filters", "use_checkpoint", "use_bn", "norm_type", "num_groups", "align_corners"]
    return {k: config[k] for k in keys if k in config}


def resolve_model_path(name: str, model_arg: str, model_selection: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Path:
    if str(model_arg).lower() == "auto":
        selection = model_selection or (config or {}).get("model_selection", "best_val_iou")
        mapping = {
            "best_val_iou": "model_best_val_iou.pth",
            "best_val_dice": "model_best_val_dice.pth",
            "best_val_loss": "model_best_val_loss.pth",
            "latest": "model_latest.pth",
        }
        if selection not in mapping:
            raise ValueError(f"Unknown model selection: {selection}")
        model_arg = mapping[selection]
        print(f"Auto-selected checkpoint: {model_arg} ({selection})")

    p = Path(model_arg)
    if p.exists():
        return p
    return Path("models") / name / model_arg


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint
    return checkpoint, None


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        out[k] = v
    return out


def load_model(config: Dict[str, Any], model_path: Path, device: torch.device, channels_last: bool = True):
    arch_name = config["arch"]
    if not hasattr(architectures, arch_name):
        raise ValueError(f"rco.architectures does not contain architecture: {arch_name}")

    model_cls = getattr(architectures, arch_name)
    print(f"=> Creating model {arch_name} with arch kwargs: {get_arch_kwargs(config)}")
    model = model_cls(
        num_classes=config["num_classes"],
        input_channels=config["input_channels"],
        **get_arch_kwargs(config),
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict, metadata = extract_state_dict(checkpoint)
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if device.type == "cuda" and channels_last:
        model = model.to(memory_format=torch.channels_last)

    if metadata is not None:
        metric_name = metadata.get("best_metric_name")
        metric_value = metadata.get("best_metric_value")
        print(
            "Loaded checkpoint: "
            f"epoch={metadata.get('epoch')}, "
            f"{metric_name}={metric_value}"
        )
    return model


# -----------------------------
# dataset
# -----------------------------
def enhance_brightness_contrast(img, gamma=3.0, target_mean=90, target_std=90):
    img = img.astype(np.float32) / 255.0
    img = np.power(img, 1.0 / gamma)
    img = img * 255.0
    mean, std = img.mean(), img.std()
    img = (img - mean) * (target_std / (std + 1e-6)) + target_mean
    return np.clip(img, 0, 255).astype(np.uint8)


def read_id_list(path: Optional[str]) -> Optional[set]:
    if path is None:
        return None
    ids = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            item = line.strip().split(",")[0]
            if item and item.lower() not in {"image_id", "train_images", "val_images", "test_images"}:
                ids.append(Path(item).stem)
    return set(ids)


class PredictDataset(Dataset):
    def __init__(self, image_paths, input_h, input_w, enhance=True):
        self.image_paths = image_paths
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.enhance = enhance

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        if self.enhance:
            img = enhance_brightness_contrast(img, gamma=3.0, target_mean=90, target_std=90)
        img = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.ascontiguousarray(img[None, ...], dtype=np.float32)
        return torch.from_numpy(img), img_name


def build_loader(image_paths, config, batch_size=1, num_workers=0, enhance=True):
    ds = PredictDataset(
        image_paths,
        input_h=config["input_h"],
        input_w=config["input_w"],
        enhance=enhance,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


@torch.inference_mode()
def predict_batch(batch_tensor, model, device, use_amp=True, channels_last=True):
    x = batch_tensor.to(device, non_blocking=True)
    if device.type == "cuda":
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
    else:
        amp_ctx = nullcontext()
    with amp_ctx:
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
    return out.float().cpu().numpy()


# -----------------------------
# postprocess
# -----------------------------
def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def normalize01(x):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def compute_region_gradient(region_prob):
    gx = cv2.Sobel(region_prob.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(region_prob.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return normalize01(np.sqrt(gx * gx + gy * gy))


def compute_offset_inconsistency(offset_y, offset_x, kernel_size=9):
    mean_y = cv2.blur(offset_y.astype(np.float32), (kernel_size, kernel_size))
    mean_x = cv2.blur(offset_x.astype(np.float32), (kernel_size, kernel_size))
    dev = np.sqrt((offset_y - mean_y) ** 2 + (offset_x - mean_x) ** 2)
    dev = cv2.GaussianBlur(dev, (0, 0), sigmaX=1.2, sigmaY=1.2)
    return normalize01(dev)


def refine_region_boundary_with_auxiliary_cues(
    region_prob,
    center_prob,
    offset_y,
    offset_x,
    region_thr=0.5,
    uncertain_low=0.35,
    uncertain_high=0.65,
    aux_alpha=0.15,
    aux_beta=0.15,
    offset_kernel=9,
):
    grad_map = compute_region_gradient(region_prob)
    offset_inconsistency = compute_offset_inconsistency(offset_y, offset_x, kernel_size=offset_kernel)
    boundary_conf = 0.50 * grad_map + 0.30 * (1.0 - center_prob) + 0.20 * offset_inconsistency
    boundary_conf = normalize01(boundary_conf)
    refined_score = region_prob.astype(np.float32).copy()
    uncertain_mask = (region_prob >= uncertain_low) & (region_prob <= uncertain_high)
    refined_score[uncertain_mask] = (
        region_prob[uncertain_mask]
        + aux_alpha * (center_prob[uncertain_mask] - 0.5)
        - aux_beta * boundary_conf[uncertain_mask]
    )
    refined_score = np.clip(refined_score, 0.0, 1.0)
    return refined_score, boundary_conf, grad_map, offset_inconsistency


def fast_connected_components(binary, min_size=50):
    binary_u8 = binary.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=4)
    out = np.zeros_like(labels, dtype=np.int32)
    new_id = 1
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_size:
            out[labels == lab] = new_id
            new_id += 1
    return out


def colorize_instances(instance_mask):
    h, w = instance_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    max_id = int(instance_mask.max())
    if max_id <= 0:
        return vis
    rng = np.random.default_rng(1234)
    colors = rng.integers(0, 255, size=(max_id + 1, 3), dtype=np.uint8)
    colors[0] = 0
    return colors[instance_mask]


def binary_from_prediction(
    region_prob,
    center_prob,
    offset_y,
    offset_x,
    region_thr=0.5,
    min_size=50,
    uncertain_low=0.35,
    uncertain_high=0.65,
    aux_alpha=0.15,
    aux_beta=0.15,
    offset_kernel=9,
):
    refined_score, boundary_conf, grad_map, offset_inconsistency = refine_region_boundary_with_auxiliary_cues(
        region_prob, center_prob, offset_y, offset_x, region_thr, uncertain_low, uncertain_high,
        aux_alpha, aux_beta, offset_kernel
    )
    binary = refined_score > region_thr

    binary = fast_connected_components(binary, min_size=min_size) > 0
    return binary.astype(np.uint8), refined_score, boundary_conf, grad_map, offset_inconsistency


def postprocess_logits(logits, args, region_thr=None):
    if region_thr is None:
        region_thr = args.region_thr
    logit_region = logits[0]
    logit_center = logits[1]
    offset_y = logits[2]
    offset_x = logits[3]
    region_prob = sigmoid_np(logit_region)
    center_prob = sigmoid_np(logit_center)

    binary, refined_score, boundary_conf, grad_map, offset_inconsistency = binary_from_prediction(
        region_prob=region_prob,
        center_prob=center_prob,
        offset_y=offset_y,
        offset_x=offset_x,
        region_thr=region_thr,
        min_size=args.min_size,
        uncertain_low=args.uncertain_low,
        uncertain_high=args.uncertain_high,
        aux_alpha=args.aux_alpha,
        aux_beta=args.aux_beta,
        offset_kernel=args.offset_kernel,
    )

    labels = fast_connected_components(binary, min_size=args.min_size)
    phase_mask = np.where(binary > 0, 0, 255).astype(np.uint8)

    return {
        "region_prob": region_prob,
        "center_prob": center_prob,
        "offset_y": offset_y,
        "offset_x": offset_x,
        "binary": binary,
        "labels": labels,
        "phase_mask": phase_mask,
        "refined_score": refined_score,
        "boundary_conf": boundary_conf,
        "grad_map": grad_map,
        "offset_inconsistency": offset_inconsistency,
    }


def save_prediction_outputs(pred: Dict[str, np.ndarray], img_name: str, save_root: Path, args):
    phase_dir = save_root / "phase_masks"
    prob_dir = save_root / "prob_maps"
    vis_dir = save_root / "visualization"
    npy_dir = save_root / "npy"
    npz_dir = save_root / "npz"

    if args.save_phase_mask:
        phase_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(phase_dir / f"{img_name}_phase_mask.png"), pred["phase_mask"])

    if args.save_prob:
        prob_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(prob_dir / f"{img_name}_region_prob.png"), (np.clip(pred["region_prob"], 0, 1) * 255).astype(np.uint8))
        cv2.imwrite(str(prob_dir / f"{img_name}_center_prob.png"), (np.clip(pred["center_prob"], 0, 1) * 255).astype(np.uint8))
        cv2.imwrite(str(prob_dir / f"{img_name}_refined_score.png"), (np.clip(pred["refined_score"], 0, 1) * 255).astype(np.uint8))
        cv2.imwrite(str(prob_dir / f"{img_name}_boundary_conf.png"), (np.clip(pred["boundary_conf"], 0, 1) * 255).astype(np.uint8))

    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_dir / f"{img_name}_gamma_region_binary.png"), (pred["binary"] * 255).astype(np.uint8))
        labels = pred["labels"]
        if labels.max() > 0:
            inst_id_vis = (labels.astype(np.float32) / labels.max() * 255).astype(np.uint8)
        else:
            inst_id_vis = np.zeros(labels.shape, dtype=np.uint8)
        cv2.imwrite(str(vis_dir / f"{img_name}_instances_id.png"), inst_id_vis)
        cv2.imwrite(str(vis_dir / f"{img_name}_instances_color.png"), colorize_instances(labels))

    if args.save_npy:
        npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_dir / f"{img_name}_instances.npy"), pred["labels"].astype(np.int32))

    if args.save_npz:
        npz_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(npz_dir / f"{img_name}_pred.npz"),
            region_prob=pred["region_prob"].astype(np.float32),
            center_prob=pred["center_prob"].astype(np.float32),
            offset_y=pred["offset_y"].astype(np.float32),
            offset_x=pred["offset_x"].astype(np.float32),
            refined_score=pred["refined_score"].astype(np.float32),
            labels=pred["labels"].astype(np.int32),
        )


# -----------------------------
# metrics
# -----------------------------
def read_gt_region(gt_dir: str, img_name: str) -> Optional[np.ndarray]:
    gt_dir = Path(gt_dir)
    npz_path = gt_dir / f"{img_name}.npz"
    if npz_path.exists():
        data = np.load(str(npz_path))
        if "region" in data:
            return (data["region"] > 0.5)
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        p = gt_dir / f"{img_name}{ext}"
        if p.exists():
            gt = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                # Supports phase mask convention: gamma-prime black = foreground.
                return gt <= 127
    return None


def calculate_metrics(pred_bool, gt_bool):
    pred = pred_bool.astype(bool)
    gt = gt_bool.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return precision, recall, dice, iou


def summarize_metrics(rows, save_root: Path, prefix="metrics"):
    if not rows:
        return
    metric_dir = save_root / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(metric_dir / f"{prefix}_per_image.csv", index=False, encoding="utf-8-sig")
    overall = pd.DataFrame([
        {
            "Type": "mean",
            "Precision": df["Precision"].mean(),
            "Recall": df["Recall"].mean(),
            "Dice": df["Dice"].mean(),
            "IoU": df["IoU"].mean(),
        },
        {
            "Type": "std",
            "Precision": df["Precision"].std(),
            "Recall": df["Recall"].std(),
            "Dice": df["Dice"].std(),
            "IoU": df["IoU"].std(),
        },
    ])
    overall.to_csv(metric_dir / f"{prefix}_overall.csv", index=False, encoding="utf-8-sig")


def threshold_values(args):
    vals = np.arange(args.thr_min, args.thr_max + args.thr_step * 0.5, args.thr_step)
    return [float(round(v, 6)) for v in vals]


# -----------------------------
# main
# -----------------------------
def collect_threshold_search_rows(
    model,
    loader,
    device,
    args,
    thresholds,
):
    rows = []
    use_amp = (device.type == "cuda") and (not args.disable_amp)

    for batch_imgs, batch_names in tqdm(loader, desc="Searching threshold"):
        logits_batch = predict_batch(
            batch_tensor=batch_imgs,
            model=model,
            device=device,
            use_amp=use_amp,
            channels_last=args.channels_last,
        )

        for logits, img_name in zip(logits_batch, batch_names):
            gt = read_gt_region(args.gt_dir, img_name) if args.gt_dir else None
            if gt is None:
                continue

            region_prob = sigmoid_np(logits[0])
            center_prob = sigmoid_np(logits[1])
            offset_y, offset_x = logits[2], logits[3]

            for thr in thresholds:
                binary, _, _, _, _ = binary_from_prediction(
                    region_prob=region_prob,
                    center_prob=center_prob,
                    offset_y=offset_y,
                    offset_x=offset_x,
                    region_thr=thr,
                    min_size=args.min_size,
                    uncertain_low=args.uncertain_low,
                    uncertain_high=args.uncertain_high,
                    aux_alpha=args.aux_alpha,
                    aux_beta=args.aux_beta,
                    offset_kernel=args.offset_kernel,
                )

                if gt.shape != binary.shape:
                    gt_eval = cv2.resize(
                        gt.astype(np.uint8),
                        (binary.shape[1], binary.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                else:
                    gt_eval = gt

                p, r, d, i = calculate_metrics(binary > 0, gt_eval)
                rows.append({
                    "Image": img_name,
                    "Threshold": thr,
                    "Precision": p,
                    "Recall": r,
                    "Dice": d,
                    "IoU": i,
                })

    return rows


def run_final_prediction(
    model,
    loader,
    device,
    args,
    save_root: Path,
    final_threshold: float,
):
    rows = []
    use_amp = (device.type == "cuda") and (not args.disable_amp)

    for batch_imgs, batch_names in tqdm(loader, desc=f"Predicting thr={final_threshold:.3f}"):
        logits_batch = predict_batch(
            batch_tensor=batch_imgs,
            model=model,
            device=device,
            use_amp=use_amp,
            channels_last=args.channels_last,
        )

        for logits, img_name in zip(logits_batch, batch_names):
            pred = postprocess_logits(logits, args, region_thr=final_threshold)
            save_prediction_outputs(pred, img_name, save_root, args)

            gt = read_gt_region(args.gt_dir, img_name) if args.gt_dir else None
            if gt is not None:
                if gt.shape != pred["binary"].shape:
                    gt = cv2.resize(
                        gt.astype(np.uint8),
                        (pred["binary"].shape[1], pred["binary"].shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)

                p, r, d, i = calculate_metrics(pred["binary"] > 0, gt)
                rows.append({
                    "Image": img_name,
                    "Threshold": final_threshold,
                    "Precision": p,
                    "Recall": r,
                    "Dice": d,
                    "IoU": i,
                })

    return rows


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    config_path = Path("models") / args.name / "config.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["name"] = args.name

    model_path = resolve_model_path(
        args.name,
        args.model,
        model_selection=args.model_selection,
        config=config,
    )
    model_stem = model_path.stem

    print(f"Loaded config: {config_path}")
    print(f"Loaded model:  {model_path}")
    print(f"Device: {device}")

    model = load_model(config, model_path, device, channels_last=args.channels_last)

    # Auto-read split list.
    id_list_path = args.id_list
    if (id_list_path is None or str(id_list_path).lower() == "auto") and args.split_name in {"train", "val", "test"}:
        candidate = Path("models") / args.name / "splits" / f"{args.split_name}_ids.txt"
        if candidate.exists():
            id_list_path = str(candidate)
            print(f"Using split id list: {id_list_path}")
        elif str(args.id_list).lower() == "auto":
            raise FileNotFoundError(f"Cannot find split id list: {candidate}")

    selected_ids = read_id_list(id_list_path) if id_list_path is not None else None

    image_paths = sorted(glob(os.path.join(args.predict_dir, "*")))
    if selected_ids is not None:
        image_paths = [p for p in image_paths if Path(p).stem in selected_ids]
    print(f"Found {len(image_paths)} images for prediction")
    if not image_paths:
        return

    if args.save_dir is None:
        save_root = Path("outputs") / args.name / model_stem / args.split_name
    else:
        save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    loader = build_loader(
        image_paths=image_paths,
        config=config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        enhance=not args.no_enhance,
    )

    final_threshold = float(args.region_thr)

    if args.search_threshold:
        if not args.gt_dir:
            raise ValueError("--search-threshold requires --gt-dir so validation metrics can be computed.")

        thresholds = threshold_values(args)
        search_rows = collect_threshold_search_rows(
            model=model,
            loader=loader,
            device=device,
            args=args,
            thresholds=thresholds,
        )

        metric_dir = save_root / "metrics"
        metric_dir.mkdir(parents=True, exist_ok=True)

        search_df = pd.DataFrame(search_rows)
        search_df.to_csv(metric_dir / "threshold_search_per_image.csv", index=False, encoding="utf-8-sig")

        summary = search_df.groupby("Threshold")[["Precision", "Recall", "Dice", "IoU"]].mean().reset_index()
        summary.to_csv(metric_dir / "threshold_search_summary.csv", index=False, encoding="utf-8-sig")

        best_idx = summary[args.select_metric].idxmax()
        best = summary.loc[best_idx].to_dict()
        final_threshold = float(best["Threshold"])

        with open(metric_dir / "best_threshold.yml", "w", encoding="utf-8") as f:
            yaml.dump({"select_metric": args.select_metric, "best": best}, f, sort_keys=False)

        print(f"Best threshold by {args.select_metric}: {final_threshold:.4f}")

        # Rebuild loader to avoid any exhausted iterator state.
        loader = build_loader(
            image_paths=image_paths,
            config=config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            enhance=not args.no_enhance,
        )

    metric_rows = run_final_prediction(
        model=model,
        loader=loader,
        device=device,
        args=args,
        save_root=save_root,
        final_threshold=final_threshold,
    )

    summarize_metrics(metric_rows, save_root, prefix=f"{args.split_name}_{final_threshold:.2f}")

    with open(save_root / "prediction_settings.yml", "w", encoding="utf-8") as f:
        yaml.dump({
            "model": str(model_path),
            "split_name": args.split_name,
            "region_thr": final_threshold,
            "threshold_was_searched": bool(args.search_threshold),
            "select_metric": args.select_metric if args.search_threshold else None,
            "brightness_contrast_enhancement": not args.no_enhance,
            "channels_last": bool(args.channels_last),
            "rco_boundary_refinement": True,
            "min_size": int(args.min_size),
            "uncertain_low": float(args.uncertain_low),
            "uncertain_high": float(args.uncertain_high),
            "aux_alpha": float(args.aux_alpha),
            "aux_beta": float(args.aux_beta),
            "offset_kernel": int(args.offset_kernel),
            "phase_mask_rule": "gamma_prime=0_black, gamma=255_white",
        }, f, sort_keys=False)

    print(f"Prediction completed. Final threshold = {final_threshold:.4f}")
    print(f"Results saved to: {save_root}")


if __name__ == "__main__":
    main()
