"""Generate region, center, and offset supervision targets from binary masks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np


def load_binary_mask(mask_path: Union[str, Path], thr: int = 50, invert: bool = True) -> np.ndarray:
    """
    读取单通道 mask 并二值化。
    """
    mask_path = str(mask_path)
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")

    binary = (m <= thr) if invert else (m > thr)
    return binary.astype(bool)


def compute_instance_labels(
    binary: np.ndarray,
    min_size: int = 10,
    connectivity: int = 1,
) -> np.ndarray:
    """
    从二值 mask 生成 instance label。
    """
    cv_connectivity = 4 if connectivity == 1 else 8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=cv_connectivity,
    )
    inst = np.zeros_like(labels, dtype=np.int32)
    new_id = 1
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_size:
            inst[labels == label_id] = new_id
            new_id += 1
    return inst


def compute_region_gt(instance_labels: np.ndarray) -> np.ndarray:
    return (instance_labels > 0).astype(np.uint8)


def compute_center_gt(instance_labels: np.ndarray) -> np.ndarray:
    """
    整张图只做一次距离变换，再按实例归一化。
    """
    center_gt = np.zeros(instance_labels.shape, dtype=np.float32)
    max_id = int(instance_labels.max())
    if max_id == 0:
        return center_gt

    fg = instance_labels > 0
    dist = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5)

    ys, xs = np.nonzero(fg)
    ids = instance_labels[ys, xs]
    region_max = np.zeros(max_id + 1, dtype=np.float32)
    np.maximum.at(region_max, ids, dist[ys, xs])
    denom = np.maximum(region_max[ids], 1e-6)

    center_gt[ys, xs] = dist[ys, xs] / denom
    return center_gt


def compute_offset_gt(
    instance_labels: np.ndarray,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    一次性计算所有实例中心，再批量回填 offset。
    """
    h, w = instance_labels.shape
    offset_y = np.zeros((h, w), dtype=np.float32)
    offset_x = np.zeros((h, w), dtype=np.float32)

    max_id = int(instance_labels.max())
    if max_id == 0:
        return offset_y, offset_x

    ys, xs = np.nonzero(instance_labels > 0)
    ids = instance_labels[ys, xs]
    counts = np.bincount(ids, minlength=max_id + 1).astype(np.float32)
    cy_all = np.bincount(ids, weights=ys, minlength=max_id + 1) / np.maximum(counts, 1.0)
    cx_all = np.bincount(ids, weights=xs, minlength=max_id + 1) / np.maximum(counts, 1.0)

    dy = cy_all[ids] - ys.astype(np.float32)
    dx = cx_all[ids] - xs.astype(np.float32)

    if normalize:
        norm = np.sqrt(dx * dx + dy * dy)
        valid = norm > 1e-6
        dx[valid] /= norm[valid]
        dy[valid] /= norm[valid]
        dx[~valid] = 0.0
        dy[~valid] = 0.0

    offset_y[ys, xs] = dy
    offset_x[ys, xs] = dx
    return offset_y, offset_x


def validate_targets(
    region_gt: np.ndarray,
    center_gt: np.ndarray,
    offset_y: np.ndarray,
    offset_x: np.ndarray,
) -> dict:
    fg = region_gt == 1
    bg = ~fg
    offset_norm = np.sqrt(offset_x * offset_x + offset_y * offset_y)

    return {
        "center_fg_max": float(center_gt[fg].max()) if np.any(fg) else 0.0,
        "center_bg_max": float(center_gt[bg].max()) if np.any(bg) else 0.0,
        "offset_fg_mean_norm": float(offset_norm[fg].mean()) if np.any(fg) else 0.0,
        "offset_bg_mean_norm": float(offset_norm[bg].mean()) if np.any(bg) else 0.0,
    }


def process_one_mask(
    mask_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    thr: int = 50,
    invert: bool = True,
    min_size: int = 10,
    connectivity: int = 1,
    normalize_offset: bool = True,
    compressed: bool = True,
    verbose: bool = False,
) -> Path:
    """
    处理单张 mask 并保存。
    """
    mask_path = Path(mask_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    binary = load_binary_mask(mask_path, thr=thr, invert=invert)
    instance_labels = compute_instance_labels(
        binary,
        min_size=min_size,
        connectivity=connectivity,
    )
    region_gt = compute_region_gt(instance_labels)
    center_gt = compute_center_gt(instance_labels)
    offset_y, offset_x = compute_offset_gt(instance_labels, normalize=normalize_offset)

    out_path = out_dir / f"{mask_path.stem}.npz"
    save_fn = np.savez_compressed if compressed else np.savez

    save_fn(
        out_path,
        region=region_gt.astype(np.uint8),
        center=center_gt.astype(np.float32),
        offset_y=offset_y.astype(np.float32),
        offset_x=offset_x.astype(np.float32),
    )

    if verbose:
        stats = validate_targets(region_gt, center_gt, offset_y, offset_x)
        print(
            f"[OK] {mask_path.name} | instances={int(instance_labels.max())} | "
            f"center_fg_max={stats['center_fg_max']:.4f}, "
            f"center_bg_max={stats['center_bg_max']:.4f}, "
            f"offset_fg_mean_norm={stats['offset_fg_mean_norm']:.4f}, "
            f"offset_bg_mean_norm={stats['offset_bg_mean_norm']:.4f}"
        )

    return out_path


def _worker(task: tuple) -> tuple:
    """
    子进程执行函数。
    返回: (success, file_name, message)
    """
    (
        mask_path,
        out_root,
        thr,
        invert,
        min_size,
        connectivity,
        normalize_offset,
        compressed,
        verbose,
        skip_existing,
    ) = task

    mask_path = Path(mask_path)
    out_root = Path(out_root)
    out_path = out_root / f"{mask_path.stem}.npz"

    try:
        if skip_existing and out_path.exists():
            return True, mask_path.name, f"skip -> {out_path.name}"

        process_one_mask(
            mask_path=mask_path,
            out_dir=out_root,
            thr=thr,
            invert=invert,
            min_size=min_size,
            connectivity=connectivity,
            normalize_offset=normalize_offset,
            compressed=compressed,
            verbose=verbose,
        )
        return True, mask_path.name, f"saved -> {out_path.name}"
    except Exception as e:
        return False, mask_path.name, str(e)


def batch_process_masks_parallel(
    mask_root: Union[str, Path],
    out_root: Union[str, Path],
    *,
    img_ext: str = ".png",
    thr: int = 50,
    invert: bool = True,
    min_size: int = 10,
    connectivity: int = 1,
    normalize_offset: bool = True,
    compressed: bool = True,
    verbose: bool = False,
    skip_existing: bool = False,
    num_workers: Optional[int] = None,
) -> None:
    """
    多进程并行批量处理。
    """
    mask_root = Path(mask_root)
    out_root = Path(out_root)
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(mask_root.glob(f"*{img_ext}"))
    total = len(mask_paths)
    print(f"Found {total} masks in: {mask_root}")

    if total == 0:
        return

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 1) - 1)
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")

    tasks = [
        (
            str(mp),
            str(out_root),
            thr,
            invert,
            min_size,
            connectivity,
            normalize_offset,
            compressed,
            verbose,
            skip_existing,
        )
        for mp in mask_paths
    ]

    ok_cnt = 0
    fail_cnt = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]

        for idx, future in enumerate(as_completed(futures), start=1):
            success, file_name, msg = future.result()
            if success:
                ok_cnt += 1
                print(f"[{idx}/{total}] Done: {file_name} | {msg}")
            else:
                fail_cnt += 1
                print(f"[{idx}/{total}] Failed: {file_name} | {msg}")

    print(
        f"Finished. total={total}, success={ok_cnt}, failed={fail_cnt}, "
        f"workers={num_workers}, out_dir={out_root}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate region, center, and offset targets from binary masks."
    )
    parser.add_argument("--mask-dir", required=True, help="Directory containing binary masks.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated .npz targets.")
    parser.add_argument("--img-ext", default=".png", help="Mask extension, for example .png or .tif.")
    parser.add_argument("--threshold", type=int, default=50, help="Threshold used to binarize masks.")
    parser.add_argument(
        "--foreground",
        choices=["dark", "light"],
        default="dark",
        help="Whether foreground pixels are darker or lighter than the threshold.",
    )
    parser.add_argument("--min-size", type=int, default=20, help="Discard connected regions smaller than this area.")
    parser.add_argument("--connectivity", type=int, choices=[1, 2], default=1)
    parser.add_argument("--no-normalize-offset", action="store_false", dest="normalize_offset")
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    batch_process_masks_parallel(
        mask_root=args.mask_dir,
        out_root=args.output_dir,
        img_ext=args.img_ext,
        thr=args.threshold,
        invert=args.foreground == "dark",
        min_size=args.min_size,
        connectivity=args.connectivity,
        normalize_offset=args.normalize_offset,
        compressed=not args.no_compress,
        verbose=args.verbose,
        skip_existing=args.skip_existing,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
