"""Standard U-Net++ baseline for full-image phase segmentation."""

import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# 基本设置
# ============================================================
IMAGE_SIZE = (1024, 1024)
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


# ============================================================
# 随机种子
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ============================================================
# 读取灰度图像
# ============================================================
def read_gray_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if image.shape != IMAGE_SIZE:
        raise ValueError(
            f"{image_path} size should be 1024×1024, but got {image.shape}"
        )

    return image


# ============================================================
# 寻找对应 mask
# ============================================================
def find_mask_path(mask_dir, image_path):
    mask_dir = Path(mask_dir)

    exact_mask_path = mask_dir / image_path.name
    if exact_mask_path.exists():
        return exact_mask_path

    for ext in IMAGE_EXTENSIONS:
        candidate = mask_dir / f"{image_path.stem}{ext}"
        if candidate.exists():
            return candidate

    return None


# ============================================================
# U-Net++ 模型
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="batch"):
        super().__init__()

        if norm_type == "group":
            groups1 = min(8, out_channels)
            while out_channels % groups1 != 0:
                groups1 -= 1
            norm1 = nn.GroupNorm(groups1, out_channels)

            groups2 = min(8, out_channels)
            while out_channels % groups2 != 0:
                groups2 -= 1
            norm2 = nn.GroupNorm(groups2, out_channels)
        else:
            norm1 = nn.BatchNorm2d(out_channels)
            norm2 = nn.BatchNorm2d(out_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm1,
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm2,
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """
    标准 U-Net++ 二分类分割模型。

    输入：灰度图，in_channels=1
    输出：单通道 logits，out_channels=1
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=16, norm_type="batch"):
        super().__init__()

        filters = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = DoubleConv(in_channels, filters[0], norm_type=norm_type)
        self.conv1_0 = DoubleConv(filters[0], filters[1], norm_type=norm_type)
        self.conv2_0 = DoubleConv(filters[1], filters[2], norm_type=norm_type)
        self.conv3_0 = DoubleConv(filters[2], filters[3], norm_type=norm_type)
        self.conv4_0 = DoubleConv(filters[3], filters[4], norm_type=norm_type)

        self.conv0_1 = DoubleConv(filters[0] + filters[1], filters[0], norm_type=norm_type)

        self.conv1_1 = DoubleConv(filters[1] + filters[2], filters[1], norm_type=norm_type)
        self.conv0_2 = DoubleConv(filters[0] * 2 + filters[1], filters[0], norm_type=norm_type)

        self.conv2_1 = DoubleConv(filters[2] + filters[3], filters[2], norm_type=norm_type)
        self.conv1_2 = DoubleConv(filters[1] * 2 + filters[2], filters[1], norm_type=norm_type)
        self.conv0_3 = DoubleConv(filters[0] * 3 + filters[1], filters[0], norm_type=norm_type)

        self.conv3_1 = DoubleConv(filters[3] + filters[4], filters[3], norm_type=norm_type)
        self.conv2_2 = DoubleConv(filters[2] * 2 + filters[3], filters[2], norm_type=norm_type)
        self.conv1_3 = DoubleConv(filters[1] * 3 + filters[2], filters[1], norm_type=norm_type)
        self.conv0_4 = DoubleConv(filters[0] * 4 + filters[1], filters[0], norm_type=norm_type)

        self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        return self.final(x0_4)


# ============================================================
# 整图 Dataset
# ============================================================
class GammaPhaseFullImageDataset(Dataset):
    """
    整张 1024×1024 图像输入训练。

    当前规则：
    γ′ 相 = 黑色 0，标签为 1
    γ 相  = 白色 255，标签为 0
    """

    def __init__(self, image_paths, mask_dir, augment=False, repeat_factor=1):
        self.image_paths = list(image_paths)
        self.mask_dir = Path(mask_dir)
        self.augment = augment
        self.repeat_factor = repeat_factor
        self.valid_pairs = []

        for image_path in self.image_paths:
            mask_path = find_mask_path(self.mask_dir, image_path)
            if mask_path is None:
                print(f"Warning: mask not found for {image_path.name}")
                continue
            self.valid_pairs.append((image_path, mask_path))

        if len(self.valid_pairs) == 0:
            raise ValueError("No valid image-mask pairs found.")

    def __len__(self):
        return len(self.valid_pairs) * self.repeat_factor

    def __getitem__(self, idx):
        real_idx = idx % len(self.valid_pairs)
        image_path, mask_path = self.valid_pairs[real_idx]

        image = read_gray_image(image_path)
        gt_mask = read_gray_image(mask_path)

        image = image.astype(np.float32) / 255.0

        # γ′ 相 = 黑色区域，标签为 1
        # γ 相  = 白色区域，标签为 0
        gt_mask = (gt_mask > 127).astype(np.uint8) * 255
        label = (gt_mask <= 127).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                image = np.flip(image, axis=1)
                label = np.flip(label, axis=1)
            if random.random() < 0.5:
                image = np.flip(image, axis=0)
                label = np.flip(label, axis=0)
            if random.random() < 0.5:
                k = random.randint(1, 3)
                image = np.rot90(image, k)
                label = np.rot90(label, k)

        image = image.copy()
        label = label.copy()

        image_tensor = torch.from_numpy(image[None, :, :]).float()
        label_tensor = torch.from_numpy(label[None, :, :]).float()

        return image_tensor, label_tensor


# ============================================================
# 正类权重
# ============================================================
def compute_pos_weight(image_paths, mask_dir):
    pos_count = 0
    neg_count = 0

    for image_path in image_paths:
        mask_path = find_mask_path(mask_dir, image_path)
        if mask_path is None:
            continue

        gt_mask = read_gray_image(mask_path)
        gt_mask = (gt_mask > 127).astype(np.uint8) * 255

        pos_count += np.sum(gt_mask <= 127)
        neg_count += np.sum(gt_mask > 127)

    if pos_count == 0:
        return 1.0

    return float(neg_count / pos_count)


# ============================================================
# Loss：BCE + Dice Loss
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=1.0, dice_weight=0.5):
        super().__init__()
        self.register_buffer(
            "pos_weight_tensor",
            torch.tensor([pos_weight], dtype=torch.float32)
        )
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight_tensor.to(logits.device)
        )
        dice_loss = self.dice(logits, targets)
        return (1.0 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss


# ============================================================
# 训练阶段 torch 指标：默认看 γ′ 相
# ============================================================
def torch_binary_metrics_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    gt = (targets >= 0.5).float()

    dims = tuple(range(1, pred.ndim))
    tp = (pred * gt).sum(dim=dims)
    fp = (pred * (1 - gt)).sum(dim=dims)
    fn = ((1 - pred) * gt).sum(dim=dims)

    precision = ((tp + eps) / (tp + fp + eps)).mean().item()
    recall = ((tp + eps) / (tp + fn + eps)).mean().item()
    dice = ((2 * tp + eps) / (2 * tp + fp + fn + eps)).mean().item()
    iou = ((tp + eps) / (tp + fp + fn + eps)).mean().item()

    return precision, recall, dice, iou


# ============================================================
# 标准训练：Train 更新参数，Val 保存最优模型
# ============================================================
def train_unetplusplus_standard(
    model,
    train_loader,
    val_loader,
    device,
    model_dir,
    log_save_path,
    pos_weight=1.0,
    epochs=50,
    learning_rate=1e-4,
    dice_weight=0.5,
    patience=10,
    amp=True,
    grad_accumulation=1,
    monitor_threshold=0.5
):
    model.to(device)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    criterion = BCEDiceLoss(pos_weight=pos_weight, dice_weight=dice_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    amp_enabled = bool(amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_loss = float("inf")
    best_val_dice = -1.0
    best_val_iou = -1.0
    no_improve_count = 0
    history = []

    latest_path = model_dir / "unetplusplus_latest.pth"
    best_loss_path = model_dir / "unetplusplus_best_val_loss.pth"
    best_dice_path = model_dir / "unetplusplus_best_val_dice.pth"
    best_iou_path = model_dir / "unetplusplus_best_val_iou.pth"

    for epoch in range(1, epochs + 1):
        # =========================
        # Train
        # =========================
        model.train()
        train_loss_sum = 0.0
        train_dice_sum = 0.0
        train_iou_sum = 0.0
        train_batches = 0

        optimizer.zero_grad(set_to_none=True)

        for step, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
                loss_to_backward = loss / grad_accumulation

            scaler.scale(loss_to_backward).backward()

            if step % grad_accumulation == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            _, _, train_dice, train_iou = torch_binary_metrics_from_logits(
                logits.detach(), masks.detach(), threshold=monitor_threshold
            )

            train_loss_sum += loss.item()
            train_dice_sum += train_dice
            train_iou_sum += train_iou
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        train_dice = train_dice_sum / max(train_batches, 1)
        train_iou = train_iou_sum / max(train_batches, 1)

        # =========================
        # Validation
        # =========================
        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        val_iou_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits = model(images)
                    loss = criterion(logits, masks)

                _, _, val_dice, val_iou = torch_binary_metrics_from_logits(
                    logits.detach(), masks.detach(), threshold=monitor_threshold
                )

                val_loss_sum += loss.item()
                val_dice_sum += val_dice
                val_iou_sum += val_iou
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        val_dice = val_dice_sum / max(val_batches, 1)
        val_iou = val_iou_sum / max(val_batches, 1)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "Epoch": epoch,
            "Train_Loss": train_loss,
            "Train_Dice": train_dice,
            "Train_IoU": train_iou,
            "Val_Loss": val_loss,
            "Val_Dice": val_dice,
            "Val_IoU": val_iou,
            "Learning_Rate": current_lr,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(log_save_path, index=False, encoding="utf-8-sig")

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.6f} | Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.6f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | "
            f"LR: {current_lr:.6e}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_dice": train_dice,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
        }
        torch.save(checkpoint, latest_path)

        improved = False

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, best_loss_path)
            print(f"Best val loss model saved: epoch {epoch}, val_loss = {val_loss:.6f}")
            improved = True

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(checkpoint, best_dice_path)
            print(f"Best val Dice model saved: epoch {epoch}, val_dice = {val_dice:.4f}")
            improved = True

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(checkpoint, best_iou_path)
            print(f"Best val IoU model saved: epoch {epoch}, val_iou = {val_iou:.4f}")
            improved = True

        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(log_save_path, index=False, encoding="utf-8-sig")

    print("\nTraining finished.")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best val dice: {best_val_dice:.4f}")
    print(f"Best val iou:  {best_val_iou:.4f}")

    return model, history_df


# ============================================================
# 概率图预测 / mask 预测
# ============================================================
def predict_prob_map_unetplusplus_full(model, image, device, amp=True):
    model.eval()
    image_float = image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_float[None, None, :, :]).float().to(device)

    amp_enabled = bool(amp and device.type == "cuda")

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(image_tensor)
            prob = torch.sigmoid(logits)

    return prob.squeeze().float().cpu().numpy()


def prob_map_to_phase_mask(prob_map, threshold=0.5):
    # prob >= threshold 表示 γ′ 相
    pred_gamma_prime = prob_map >= threshold
    # 输出规则：γ′ 相 = 黑色 0，γ 相 = 白色 255
    pred_mask = np.where(pred_gamma_prime, 0, 255).astype(np.uint8)
    return pred_mask


def predict_one_image_unetplusplus_full(model, image, device, threshold=0.5, amp=True):
    prob_map = predict_prob_map_unetplusplus_full(model, image, device, amp=amp)
    return prob_map_to_phase_mask(prob_map, threshold=threshold)


# ============================================================
# numpy 指标计算
# ============================================================
def calculate_metrics(pred_bool, gt_bool):
    TP = np.logical_and(pred_bool, gt_bool).sum()
    FP = np.logical_and(pred_bool, ~gt_bool).sum()
    FN = np.logical_and(~pred_bool, gt_bool).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return precision, recall, dice, iou


def evaluate_one_image(pred_mask, gt_mask, image_name, dataset_type):
    pred_mask = (pred_mask > 127).astype(np.uint8) * 255
    gt_mask = (gt_mask > 127).astype(np.uint8) * 255

    pred_gamma_prime = pred_mask <= 127
    gt_gamma_prime = gt_mask <= 127

    pred_gamma = pred_mask > 127
    gt_gamma = gt_mask > 127

    gp_precision, gp_recall, gp_dice, gp_iou = calculate_metrics(
        pred_gamma_prime, gt_gamma_prime
    )

    g_precision, g_recall, g_dice, g_iou = calculate_metrics(
        pred_gamma, gt_gamma
    )

    rows = [
        {
            "Dataset": dataset_type,
            "Image": image_name,
            "Phase": "gamma_prime",
            "Precision": gp_precision,
            "Recall": gp_recall,
            "Dice": gp_dice,
            "IoU": gp_iou,
        },
        {
            "Dataset": dataset_type,
            "Image": image_name,
            "Phase": "gamma",
            "Precision": g_precision,
            "Recall": g_recall,
            "Dice": g_dice,
            "IoU": g_iou,
        },
    ]
    return rows


# ============================================================
# 阈值自动搜索
# ============================================================
def search_best_threshold_on_val(
    model,
    val_paths,
    mask_dir,
    device,
    thresholds=None,
    select_phase="gamma_prime",
    select_metric="IoU",
    save_path=None,
    amp=True,
):
    """
    在验证集上自动搜索最佳 threshold。

    select_phase:
        "gamma_prime"、"gamma" 或 "overall_mean"

    select_metric:
        "Precision"、"Recall"、"Dice"、"IoU"
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.71, 0.05)

    search_results = []

    # 为了减少重复前向，先缓存验证集概率图和 GT
    cached_items = []
    for image_path in val_paths:
        mask_path = find_mask_path(mask_dir, image_path)
        image = read_gray_image(image_path)
        gt_mask = read_gray_image(mask_path)
        gt_mask = (gt_mask > 127).astype(np.uint8) * 255
        prob_map = predict_prob_map_unetplusplus_full(model, image, device, amp=amp)
        cached_items.append((image_path.name, prob_map, gt_mask))

    for threshold in thresholds:
        all_rows = []
        for image_name, prob_map, gt_mask in cached_items:
            pred_mask = prob_map_to_phase_mask(prob_map, threshold=threshold)
            rows = evaluate_one_image(
                pred_mask=pred_mask,
                gt_mask=gt_mask,
                image_name=image_name,
                dataset_type="val",
            )
            all_rows.extend(rows)

        results_df = pd.DataFrame(all_rows)
        mean_by_phase = results_df.groupby("Phase")[["Precision", "Recall", "Dice", "IoU"]].mean().reset_index()

        if select_phase == "overall_mean":
            score_values = results_df[["Precision", "Recall", "Dice", "IoU"]].mean()
            row = {
                "Threshold": float(threshold),
                "Select_Phase": select_phase,
                "Select_Metric": select_metric,
                "Precision": score_values["Precision"],
                "Recall": score_values["Recall"],
                "Dice": score_values["Dice"],
                "IoU": score_values["IoU"],
                "Score": score_values[select_metric],
            }
        else:
            target_row = mean_by_phase[mean_by_phase["Phase"] == select_phase].iloc[0]
            row = {
                "Threshold": float(threshold),
                "Select_Phase": select_phase,
                "Select_Metric": select_metric,
                "Precision": target_row["Precision"],
                "Recall": target_row["Recall"],
                "Dice": target_row["Dice"],
                "IoU": target_row["IoU"],
                "Score": target_row[select_metric],
            }

        search_results.append(row)

    threshold_df = pd.DataFrame(search_results)
    best_idx = threshold_df["Score"].idxmax()
    best_threshold = float(threshold_df.loc[best_idx, "Threshold"])

    if save_path is not None:
        threshold_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print("Threshold search finished.")
    print("==============================")
    print(threshold_df)
    print(f"\nBest threshold = {best_threshold:.2f}")
    print(
        f"Best {select_phase} {select_metric} = "
        f"{threshold_df.loc[best_idx, 'Score']:.4f}"
    )

    return best_threshold, threshold_df


# ============================================================
# 保存预测对比图
# ============================================================
def save_comparison_figure(image, gt_mask, pred_mask, save_path, title):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask, cmap="gray")
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("U-Net++ prediction")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# 预测并评价某个数据集
# ============================================================
def predict_and_evaluate_dataset(
    model,
    image_paths,
    mask_dir,
    pred_mask_dir,
    comparison_dir,
    dataset_type,
    device,
    threshold=0.5,
    amp=True,
):
    pred_mask_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for image_path in image_paths:
        mask_path = find_mask_path(mask_dir, image_path)

        image = read_gray_image(image_path)
        gt_mask = read_gray_image(mask_path)
        gt_mask = (gt_mask > 127).astype(np.uint8) * 255

        pred_mask = predict_one_image_unetplusplus_full(
            model=model,
            image=image,
            device=device,
            threshold=threshold,
            amp=amp,
        )

        cv2.imwrite(str(pred_mask_dir / image_path.name), pred_mask)

        comparison_path = comparison_dir / f"{image_path.stem}_comparison.png"
        save_comparison_figure(
            image=image,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            save_path=comparison_path,
            title=f"{dataset_type}: {image_path.name} | threshold={threshold:.2f}",
        )

        rows = evaluate_one_image(
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            image_name=image_path.name,
            dataset_type=dataset_type,
        )
        all_results.extend(rows)

        print(f"{dataset_type} predicted and evaluated: {image_path.name}")

    return pd.DataFrame(all_results)


# ============================================================
# 表格和曲线
# ============================================================
def make_wide_table(results_df):
    wide_df = results_df.pivot(
        index=["Dataset", "Image"],
        columns="Phase",
        values=["Precision", "Recall", "Dice", "IoU"],
    )

    wide_df.columns = [f"{phase}_{metric}" for metric, phase in wide_df.columns]
    wide_df = wide_df.reset_index()

    column_order = [
        "Dataset",
        "Image",
        "gamma_prime_Precision",
        "gamma_prime_Recall",
        "gamma_prime_Dice",
        "gamma_prime_IoU",
        "gamma_Precision",
        "gamma_Recall",
        "gamma_Dice",
        "gamma_IoU",
    ]
    return wide_df[column_order]


def summarize_metrics(results_df, decimal_places=4):
    metric_columns = ["Precision", "Recall", "Dice", "IoU"]

    per_image_long_df = results_df.copy()
    per_image_wide_df = make_wide_table(results_df)

    mean_by_phase_df = results_df.groupby(["Dataset", "Phase"])[metric_columns].mean().reset_index()
    std_by_phase_df = results_df.groupby(["Dataset", "Phase"])[metric_columns].std().reset_index()

    overall_rows = []
    for dataset_name in results_df["Dataset"].unique():
        subset = results_df[results_df["Dataset"] == dataset_name]
        overall_rows.append({
            "Dataset": dataset_name,
            "Type": "overall_mean",
            "Precision": subset["Precision"].mean(),
            "Recall": subset["Recall"].mean(),
            "Dice": subset["Dice"].mean(),
            "IoU": subset["IoU"].mean(),
        })
        overall_rows.append({
            "Dataset": dataset_name,
            "Type": "overall_std",
            "Precision": subset["Precision"].std(),
            "Recall": subset["Recall"].std(),
            "Dice": subset["Dice"].std(),
            "IoU": subset["IoU"].std(),
        })

    overall_mean_std_df = pd.DataFrame(overall_rows)

    per_image_long_df[metric_columns] = per_image_long_df[metric_columns].round(decimal_places)
    mean_by_phase_df[metric_columns] = mean_by_phase_df[metric_columns].round(decimal_places)
    std_by_phase_df[metric_columns] = std_by_phase_df[metric_columns].round(decimal_places)
    overall_mean_std_df[metric_columns] = overall_mean_std_df[metric_columns].round(decimal_places)

    wide_metric_columns = [c for c in per_image_wide_df.columns if c not in ["Dataset", "Image"]]
    per_image_wide_df[wide_metric_columns] = per_image_wide_df[wide_metric_columns].round(decimal_places)

    return per_image_long_df, per_image_wide_df, mean_by_phase_df, std_by_phase_df, overall_mean_std_df


def plot_training_curves(history_df, curve_dir):
    curve_dir = Path(curve_dir)
    curve_dir.mkdir(parents=True, exist_ok=True)

    def save_curve(y_cols, title, ylabel, filename):
        plt.figure(figsize=(7, 5))
        for col in y_cols:
            if col in history_df.columns:
                plt.plot(history_df["Epoch"], history_df[col], label=col)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(curve_dir / filename, dpi=300)
        plt.close()

    save_curve(["Train_Loss", "Val_Loss"], "Loss Curve", "Loss", "loss_curve.png")
    save_curve(["Train_Dice", "Val_Dice"], "Dice Curve", "Dice", "dice_curve.png")
    save_curve(["Train_IoU", "Val_IoU"], "IoU Curve", "IoU", "iou_curve.png")
    save_curve(["Learning_Rate"], "Learning Rate Curve", "Learning Rate", "learning_rate_curve.png")


# ============================================================
# U-Net++ 主流程
# ============================================================
def unetplusplus_full_image_pipeline(
    image_dir,
    mask_dir,
    output_dir,
    test_size=0.2,
    val_size=0.2,
    batch_size=1,
    epochs=100,
    learning_rate=1e-4,
    base_channels=16,
    repeat_factor=1,
    dice_weight=0.6,
    patience=15,
    random_state=42,
    decimal_places=4,
    norm_type="group",
    amp=True,
    grad_accumulation=2,
    model_selection="best_val_iou",  # best_val_iou / best_val_dice / best_val_loss / latest
    search_threshold=True,
    threshold=0.5,
    threshold_min=0.30,
    threshold_max=0.70,
    threshold_step=0.05,
    threshold_select_phase="gamma_prime",  # gamma_prime / gamma / overall_mean
    threshold_select_metric="IoU",         # Precision / Recall / Dice / IoU
):
    set_seed(random_state)

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    model_dir = output_dir / "model"
    curve_dir = output_dir / "training_curves"

    train_pred_mask_dir = output_dir / "train_pred_masks"
    val_pred_mask_dir = output_dir / "val_pred_masks"
    test_pred_mask_dir = output_dir / "test_pred_masks"

    train_comparison_dir = output_dir / "train_comparison_figures"
    val_comparison_dir = output_dir / "val_comparison_figures"
    test_comparison_dir = output_dir / "test_comparison_figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in sorted(image_dir.glob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]

    valid_image_paths = []
    for image_path in image_paths:
        mask_path = find_mask_path(mask_dir, image_path)
        if mask_path is not None:
            valid_image_paths.append(image_path)
        else:
            print(f"Warning: mask not found for {image_path.name}")

    if len(valid_image_paths) < 3:
        raise ValueError("有效图像数量太少，至少需要 train / val / test 三部分。")

    train_val_paths, test_paths = train_test_split(
        valid_image_paths,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    train_paths, val_paths = train_test_split(
        train_val_paths,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
    )

    print("\n==============================")
    print("Dataset split")
    print("==============================")
    print(f"Train images: {len(train_paths)}")
    print(f"Val images:   {len(val_paths)}")
    print(f"Test images:  {len(test_paths)}")

    pd.DataFrame({"Train_images": [p.name for p in train_paths]}).to_csv(
        output_dir / "train_image_list.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"Val_images": [p.name for p in val_paths]}).to_csv(
        output_dir / "val_image_list.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"Test_images": [p.name for p in test_paths]}).to_csv(
        output_dir / "test_image_list.csv", index=False, encoding="utf-8-sig"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    train_dataset = GammaPhaseFullImageDataset(
        image_paths=train_paths,
        mask_dir=mask_dir,
        augment=True,
        repeat_factor=repeat_factor,
    )
    val_dataset = GammaPhaseFullImageDataset(
        image_paths=val_paths,
        mask_dir=mask_dir,
        augment=False,
        repeat_factor=1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    pos_weight = compute_pos_weight(train_paths, mask_dir)
    print(f"pos_weight: {pos_weight:.4f}")

    model = UNetPlusPlus(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        norm_type=norm_type,
    )

    log_path = model_dir / "unetplusplus_training_log.csv"

    print("\n==============================")
    print("Training U-Net++")
    print("==============================")

    model, history_df = train_unetplusplus_standard(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_dir=model_dir,
        log_save_path=log_path,
        pos_weight=pos_weight,
        epochs=epochs,
        learning_rate=learning_rate,
        dice_weight=dice_weight,
        patience=patience,
        amp=amp,
        grad_accumulation=grad_accumulation,
        monitor_threshold=0.5,
    )

    plot_training_curves(history_df, curve_dir)

    model_paths = {
        "best_val_loss": model_dir / "unetplusplus_best_val_loss.pth",
        "best_val_dice": model_dir / "unetplusplus_best_val_dice.pth",
        "best_val_iou": model_dir / "unetplusplus_best_val_iou.pth",
        "latest": model_dir / "unetplusplus_latest.pth",
    }

    if model_selection not in model_paths:
        raise ValueError(f"model_selection must be one of {list(model_paths.keys())}")

    selected_model_path = model_paths[model_selection]
    checkpoint = torch.load(selected_model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("\n==============================")
    print("Selected model loaded")
    print("==============================")
    print(f"Model selection: {model_selection}")
    print(f"Loaded model: {selected_model_path}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Val Loss: {checkpoint.get('val_loss', np.nan):.6f}")
    print(f"Val Dice: {checkpoint.get('val_dice', np.nan):.4f}")
    print(f"Val IoU:  {checkpoint.get('val_iou', np.nan):.4f}")

    if search_threshold:
        thresholds = np.arange(threshold_min, threshold_max + 1e-8, threshold_step)
        threshold_search_path = output_dir / "unetplusplus_threshold_search.csv"
        best_threshold, threshold_df = search_best_threshold_on_val(
            model=model,
            val_paths=val_paths,
            mask_dir=mask_dir,
            device=device,
            thresholds=thresholds,
            select_phase=threshold_select_phase,
            select_metric=threshold_select_metric,
            save_path=threshold_search_path,
            amp=amp,
        )
        threshold = best_threshold
    else:
        threshold_df = pd.DataFrame()

    print(f"\nFinal prediction threshold = {threshold:.2f}")

    print("\n==============================")
    print("Predicting train images")
    print("==============================")
    train_results_df = predict_and_evaluate_dataset(
        model=model,
        image_paths=train_paths,
        mask_dir=mask_dir,
        pred_mask_dir=train_pred_mask_dir,
        comparison_dir=train_comparison_dir,
        dataset_type="train",
        device=device,
        threshold=threshold,
        amp=amp,
    )

    print("\n==============================")
    print("Predicting val images")
    print("==============================")
    val_results_df = predict_and_evaluate_dataset(
        model=model,
        image_paths=val_paths,
        mask_dir=mask_dir,
        pred_mask_dir=val_pred_mask_dir,
        comparison_dir=val_comparison_dir,
        dataset_type="val",
        device=device,
        threshold=threshold,
        amp=amp,
    )

    print("\n==============================")
    print("Predicting test images")
    print("==============================")
    test_results_df = predict_and_evaluate_dataset(
        model=model,
        image_paths=test_paths,
        mask_dir=mask_dir,
        pred_mask_dir=test_pred_mask_dir,
        comparison_dir=test_comparison_dir,
        dataset_type="test",
        device=device,
        threshold=threshold,
        amp=amp,
    )

    all_results_df = pd.concat(
        [train_results_df, val_results_df, test_results_df],
        ignore_index=True,
    )

    (
        per_image_long_df,
        per_image_wide_df,
        mean_by_phase_df,
        std_by_phase_df,
        overall_mean_std_df,
    ) = summarize_metrics(all_results_df, decimal_places=decimal_places)

    train_long_df = per_image_long_df[per_image_long_df["Dataset"] == "train"]
    val_long_df = per_image_long_df[per_image_long_df["Dataset"] == "val"]
    test_long_df = per_image_long_df[per_image_long_df["Dataset"] == "test"]

    train_wide_df = per_image_wide_df[per_image_wide_df["Dataset"] == "train"]
    val_wide_df = per_image_wide_df[per_image_wide_df["Dataset"] == "val"]
    test_wide_df = per_image_wide_df[per_image_wide_df["Dataset"] == "test"]

    train_overall_df = overall_mean_std_df[overall_mean_std_df["Dataset"] == "train"]
    val_overall_df = overall_mean_std_df[overall_mean_std_df["Dataset"] == "val"]
    test_overall_df = overall_mean_std_df[overall_mean_std_df["Dataset"] == "test"]

    train_long_df.to_csv(output_dir / "unetplusplus_train_per_image_metrics_long.csv", index=False, encoding="utf-8-sig")
    val_long_df.to_csv(output_dir / "unetplusplus_val_per_image_metrics_long.csv", index=False, encoding="utf-8-sig")
    test_long_df.to_csv(output_dir / "unetplusplus_test_per_image_metrics_long.csv", index=False, encoding="utf-8-sig")

    train_wide_df.to_csv(output_dir / "unetplusplus_train_per_image_metrics_wide.csv", index=False, encoding="utf-8-sig")
    val_wide_df.to_csv(output_dir / "unetplusplus_val_per_image_metrics_wide.csv", index=False, encoding="utf-8-sig")
    test_wide_df.to_csv(output_dir / "unetplusplus_test_per_image_metrics_wide.csv", index=False, encoding="utf-8-sig")

    train_overall_df.to_csv(output_dir / "unetplusplus_train_overall_mean_metrics.csv", index=False, encoding="utf-8-sig")
    val_overall_df.to_csv(output_dir / "unetplusplus_val_overall_mean_metrics.csv", index=False, encoding="utf-8-sig")
    test_overall_df.to_csv(output_dir / "unetplusplus_test_overall_mean_metrics.csv", index=False, encoding="utf-8-sig")

    mean_by_phase_df.to_csv(output_dir / "unetplusplus_mean_metrics_by_phase_train_val_test.csv", index=False, encoding="utf-8-sig")
    std_by_phase_df.to_csv(output_dir / "unetplusplus_std_metrics_by_phase_train_val_test.csv", index=False, encoding="utf-8-sig")
    overall_mean_std_df.to_csv(output_dir / "unetplusplus_overall_mean_std_train_val_test.csv", index=False, encoding="utf-8-sig")

    excel_path = output_dir / "unetplusplus_auto_threshold_train_val_test_metrics.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        history_df.to_excel(writer, sheet_name="Training_Log", index=False)
        if not threshold_df.empty:
            threshold_df.to_excel(writer, sheet_name="Threshold_Search", index=False)

        train_long_df.to_excel(writer, sheet_name="Train_Long", index=False)
        val_long_df.to_excel(writer, sheet_name="Val_Long", index=False)
        test_long_df.to_excel(writer, sheet_name="Test_Long", index=False)

        train_wide_df.to_excel(writer, sheet_name="Train_Wide", index=False)
        val_wide_df.to_excel(writer, sheet_name="Val_Wide", index=False)
        test_wide_df.to_excel(writer, sheet_name="Test_Wide", index=False)

        mean_by_phase_df.to_excel(writer, sheet_name="Mean_By_Phase", index=False)
        std_by_phase_df.to_excel(writer, sheet_name="Std_By_Phase", index=False)
        overall_mean_std_df.to_excel(writer, sheet_name="Overall_Mean_Std", index=False)

    print("\n==============================")
    print("U-Net++ train/val/test evaluation finished.")
    print("==============================")
    print(f"Model selection: {model_selection}")
    print(f"Final threshold: {threshold:.2f}")
    print("\nMean metrics by phase:")
    print(mean_by_phase_df)
    print("\nOverall mean and std:")
    print(overall_mean_std_df)
    print(f"\nExcel file saved to: {excel_path}")

    return (
        model,
        history_df,
        threshold_df,
        train_results_df,
        val_results_df,
        test_results_df,
        mean_by_phase_df,
        std_by_phase_df,
        overall_mean_std_df,
    )


# ============================================================
# 主程序入口
# ============================================================
if __name__ == "__main__":
    (
        model,
        history_df,
        threshold_df,
        train_results_df,
        val_results_df,
        test_results_df,
        mean_by_phase_df,
        std_by_phase_df,
        overall_mean_std_df,
    ) = unetplusplus_full_image_pipeline(
        image_dir="dataset/images",
        mask_dir="dataset/masks",
        output_dir="dataset/results_unetplusplus_auto_threshold",

        # 数据划分：总体约 train 64%, val 16%, test 20%
        test_size=0.2,
        val_size=0.2,

        # 整图训练建议 batch_size=1；需要等效大 batch 用 grad_accumulation
        batch_size=4,
        grad_accumulation=2,

        # 训练参数
        epochs=200,
        learning_rate=1e-4,
        base_channels=16,
        repeat_factor=1,
        dice_weight=0.6,
        patience=15,

        # 小 batch 下推荐 group；如果想完全沿用原版，可改为 batch
        norm_type="group",

        # 混合精度，CUDA 下可降低显存
        amp=True,

        # 选择最终用于预测的模型
        model_selection="best_val_iou",  # best_val_iou / best_val_dice / best_val_loss / latest

        # 验证集自动搜索阈值
        search_threshold=True,
        threshold=0.5,
        threshold_min=0.30,
        threshold_max=0.70,
        threshold_step=0.05,
        threshold_select_phase="overall_mean",  # gamma_prime / gamma / overall_mean
        threshold_select_metric="IoU",         # Precision / Recall / Dice / IoU

        random_state=42,
        decimal_places=4,
    )
