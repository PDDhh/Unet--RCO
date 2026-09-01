"""Standard U-Net baseline for full-image phase segmentation."""

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


# ==============================
# 基本设置
# ==============================
IMAGE_SIZE = (1024, 1024)
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


# ==============================
# 随机种子
# ==============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ==============================
# 读取灰度图像
# ==============================
def read_gray_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if image.shape != IMAGE_SIZE:
        raise ValueError(
            f"{image_path} size should be 1024×1024, but got {image.shape}"
        )

    return image


# ==============================
# 寻找对应 mask
# ==============================
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


# ==============================
# U-Net 模型
# ==============================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=16):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


# ==============================
# 整图 Dataset
# ==============================
class GammaPhaseFullImageDataset(Dataset):
    """
    整张 1024×1024 图像输入训练。

    当前规则：
    γ′ 相 = 黑色 0，标签为 1
    γ 相  = 白色 255，标签为 0
    """

    def __init__(self, image_paths, mask_dir, augment=True, repeat_factor=1):
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


# ==============================
# 计算正类权重
# ==============================
def compute_pos_weight(image_paths, mask_dir):
    """
    γ′ 相 = 正类 1
    γ 相 = 负类 0
    """

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

    return neg_count / pos_count


# ==============================
# Loss：BCE + Dice
# ==============================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

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


# ==============================
# 训练阶段监控指标：γ′ 相 Dice / IoU
# ==============================
def torch_dice_iou_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    gt = (targets >= 0.5).float()

    pred = pred.view(pred.size(0), -1)
    gt = gt.view(gt.size(0), -1)

    inter = (pred * gt).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    gt_sum = gt.sum(dim=1)

    dice = (2.0 * inter + eps) / (pred_sum + gt_sum + eps)
    iou = (inter + eps) / (pred_sum + gt_sum - inter + eps)

    return dice.mean().item(), iou.mean().item()


# ==============================
# 单个 epoch：训练 / 验证
# ==============================
def run_one_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    threshold=0.5,
    use_amp=True
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_n = 0

    amp_enabled = (device.type == "cuda") and use_amp

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)

            if is_train:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = images.size(0)
        dice, iou = torch_dice_iou_from_logits(
            logits.detach(),
            masks.detach(),
            threshold=threshold
        )

        total_loss += loss.item() * batch_size
        total_dice += dice * batch_size
        total_iou += iou * batch_size
        total_n += batch_size

    return {
        "loss": total_loss / total_n,
        "dice": total_dice / total_n,
        "iou": total_iou / total_n
    }


# ==============================
# 标准训练流程：train 更新参数，val 保存最优模型
# ==============================
def train_unet_standard(
    model,
    train_loader,
    val_loader,
    device,
    model_dir,
    pos_weight=1.0,
    epochs=50,
    learning_rate=1e-4,
    dice_weight=0.5,
    threshold=0.5,
    patience=10,
    use_amp=True
):
    model.to(device)

    criterion = BCEDiceLoss(
        pos_weight=pos_weight,
        dice_weight=dice_weight
    )

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

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and use_amp))

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_dice = 0.0
    best_epoch_loss = 0
    best_epoch_dice = 0
    no_improve_count = 0

    history = []

    best_loss_path = model_dir / "unet_best_val_loss.pth"
    best_dice_path = model_dir / "unet_best_val_dice.pth"
    latest_path = model_dir / "unet_latest.pth"

    for epoch in range(1, epochs + 1):
        train_log = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            threshold=threshold,
            use_amp=use_amp
        )

        val_log = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            threshold=threshold,
            use_amp=use_amp
        )

        scheduler.step(val_log["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_log["loss"],
            "train_dice": train_log["dice"],
            "train_iou": train_log["iou"],
            "val_loss": val_log["loss"],
            "val_dice": val_log["dice"],
            "val_iou": val_log["iou"],
        }
        history.append(row)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_log['loss']:.6f} | Train Dice: {train_log['dice']:.4f} | Train IoU: {train_log['iou']:.4f} | "
            f"Val Loss: {val_log['loss']:.6f} | Val Dice: {val_log['dice']:.4f} | Val IoU: {val_log['iou']:.4f} | "
            f"LR: {current_lr:.6e}"
        )

        # 每轮保存 latest，方便中断恢复或排查
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_log["loss"],
                "val_dice": val_log["dice"],
                "config": {
                    "pos_weight": pos_weight,
                    "learning_rate": learning_rate,
                    "dice_weight": dice_weight,
                    "threshold": threshold,
                }
            },
            latest_path
        )

        improved = False

        # 按验证集 loss 保存最优模型
        if val_log["loss"] < best_val_loss:
            best_val_loss = val_log["loss"]
            best_epoch_loss = epoch
            no_improve_count = 0
            improved = True

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_dice": val_log["dice"],
                    "val_iou": val_log["iou"],
                },
                best_loss_path
            )
            print(f"=> saved best val loss model: epoch {epoch}, val_loss = {best_val_loss:.6f}")

        # 同时按验证集 Dice 保存一个最优模型，便于对比
        if val_log["dice"] > best_val_dice:
            best_val_dice = val_log["dice"]
            best_epoch_dice = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_dice": best_val_dice,
                    "val_loss": val_log["loss"],
                    "val_iou": val_log["iou"],
                },
                best_dice_path
            )
            print(f"=> saved best val dice model: epoch {epoch}, val_dice = {best_val_dice:.4f}")

        if not improved:
            no_improve_count += 1

        if patience > 0 and no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(model_dir / "training_log.csv", index=False, encoding="utf-8-sig")

    print("\nTraining finished.")
    print(f"Best val loss: epoch {best_epoch_loss}, loss = {best_val_loss:.6f}")
    print(f"Best val dice: epoch {best_epoch_dice}, dice = {best_val_dice:.4f}")

    return model, history_df, best_loss_path, best_dice_path


# ==============================
# 整图预测
# ==============================
def predict_one_image_unet_full(model, image, device, threshold=0.5, use_amp=True):
    """
    整张 1024×1024 图像直接预测。

    输出规则：
    γ′ 相 = 黑色 0
    γ 相  = 白色 255
    """

    model.eval()

    image_float = image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_float[None, None, :, :]).float().to(device)

    amp_enabled = (device.type == "cuda") and use_amp

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(image_tensor)
            prob = torch.sigmoid(logits)

    prob_map = prob.squeeze().float().cpu().numpy()
    pred_gamma_prime = prob_map >= threshold
    pred_mask = np.where(pred_gamma_prime, 0, 255).astype(np.uint8)

    return pred_mask


# ==============================
# 计算评价指标
# ==============================
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
    """
    当前规则：
    γ′ 相 = 黑色 0
    γ 相  = 白色 255
    """

    pred_mask = (pred_mask > 127).astype(np.uint8) * 255
    gt_mask = (gt_mask > 127).astype(np.uint8) * 255

    # γ′ 相：黑色区域
    pred_gamma_prime = pred_mask <= 127
    gt_gamma_prime = gt_mask <= 127

    # γ 相：白色区域
    pred_gamma = pred_mask > 127
    gt_gamma = gt_mask > 127

    gp_precision, gp_recall, gp_dice, gp_iou = calculate_metrics(
        pred_gamma_prime,
        gt_gamma_prime
    )

    g_precision, g_recall, g_dice, g_iou = calculate_metrics(
        pred_gamma,
        gt_gamma
    )

    rows = [
        {
            "Dataset": dataset_type,
            "Image": image_name,
            "Phase": "gamma_prime",
            "Precision": gp_precision,
            "Recall": gp_recall,
            "Dice": gp_dice,
            "IoU": gp_iou
        },
        {
            "Dataset": dataset_type,
            "Image": image_name,
            "Phase": "gamma",
            "Precision": g_precision,
            "Recall": g_recall,
            "Dice": g_dice,
            "IoU": g_iou
        }
    ]

    return rows


# ==============================
# 保存预测对比图
# ==============================
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
    plt.title("U-Net prediction")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ==============================
# 绘制训练曲线
# ==============================
def save_training_curves(history_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # loss 曲线
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()

    # Dice 曲线
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch"], history_df["train_dice"], label="Train Dice")
    plt.plot(history_df["epoch"], history_df["val_dice"], label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "dice_curve.png", dpi=300)
    plt.close()

    # IoU 曲线
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch"], history_df["train_iou"], label="Train IoU")
    plt.plot(history_df["epoch"], history_df["val_iou"], label="Val IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "iou_curve.png", dpi=300)
    plt.close()


# ==============================
# 每张图一行宽表
# ==============================
def make_wide_table(results_df):
    wide_df = results_df.pivot(
        index=["Dataset", "Image"],
        columns="Phase",
        values=["Precision", "Recall", "Dice", "IoU"]
    )

    wide_df.columns = [
        f"{phase}_{metric}"
        for metric, phase in wide_df.columns
    ]

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
        "gamma_IoU"
    ]

    wide_df = wide_df[column_order]

    return wide_df


# ==============================
# 汇总平均值和标准差
# ==============================
def summarize_metrics(results_df, decimal_places=4):
    metric_columns = ["Precision", "Recall", "Dice", "IoU"]

    per_image_long_df = results_df.copy()
    per_image_wide_df = make_wide_table(results_df)

    mean_by_phase_df = results_df.groupby(["Dataset", "Phase"])[
        metric_columns
    ].mean().reset_index()

    std_by_phase_df = results_df.groupby(["Dataset", "Phase"])[
        metric_columns
    ].std().reset_index()

    overall_rows = []

    for dataset_name in results_df["Dataset"].unique():
        subset = results_df[results_df["Dataset"] == dataset_name]

        overall_rows.append({
            "Dataset": dataset_name,
            "Type": "overall_mean",
            "Precision": subset["Precision"].mean(),
            "Recall": subset["Recall"].mean(),
            "Dice": subset["Dice"].mean(),
            "IoU": subset["IoU"].mean()
        })

        overall_rows.append({
            "Dataset": dataset_name,
            "Type": "overall_std",
            "Precision": subset["Precision"].std(),
            "Recall": subset["Recall"].std(),
            "Dice": subset["Dice"].std(),
            "IoU": subset["IoU"].std()
        })

    overall_mean_std_df = pd.DataFrame(overall_rows)

    per_image_long_df[metric_columns] = per_image_long_df[metric_columns].round(decimal_places)
    mean_by_phase_df[metric_columns] = mean_by_phase_df[metric_columns].round(decimal_places)
    std_by_phase_df[metric_columns] = std_by_phase_df[metric_columns].round(decimal_places)
    overall_mean_std_df[metric_columns] = overall_mean_std_df[metric_columns].round(decimal_places)

    wide_metric_columns = [
        c for c in per_image_wide_df.columns
        if c not in ["Dataset", "Image"]
    ]

    per_image_wide_df[wide_metric_columns] = per_image_wide_df[wide_metric_columns].round(decimal_places)

    return (
        per_image_long_df,
        per_image_wide_df,
        mean_by_phase_df,
        std_by_phase_df,
        overall_mean_std_df
    )


# ==============================
# 预测并评价训练集 / 验证集 / 测试集
# ==============================
def predict_and_evaluate_dataset(
    model,
    image_paths,
    mask_dir,
    pred_mask_dir,
    comparison_dir,
    dataset_type,
    device,
    threshold=0.5,
    use_amp=True
):
    pred_mask_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for image_path in image_paths:
        mask_path = find_mask_path(mask_dir, image_path)

        image = read_gray_image(image_path)
        gt_mask = read_gray_image(mask_path)
        gt_mask = (gt_mask > 127).astype(np.uint8) * 255

        pred_mask = predict_one_image_unet_full(
            model=model,
            image=image,
            device=device,
            threshold=threshold,
            use_amp=use_amp
        )

        cv2.imwrite(
            str(pred_mask_dir / image_path.name),
            pred_mask
        )

        comparison_path = comparison_dir / f"{image_path.stem}_comparison.png"

        save_comparison_figure(
            image=image,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            save_path=comparison_path,
            title=f"{dataset_type}: {image_path.name}"
        )

        rows = evaluate_one_image(
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            image_name=image_path.name,
            dataset_type=dataset_type
        )

        all_results.extend(rows)
        print(f"{dataset_type} predicted and evaluated: {image_path.name}")

    return pd.DataFrame(all_results)


# ==============================
# U-Net 标准训练主流程
# ==============================
def unet_standard_pipeline(
    image_dir,
    mask_dir,
    output_dir,
    test_size=0.2,
    val_size=0.2,
    batch_size=1,
    epochs=50,
    learning_rate=1e-4,
    base_channels=16,
    threshold=0.5,
    repeat_factor=1,
    dice_weight=0.5,
    patience=10,
    use_amp=True,
    random_state=42,
    decimal_places=4,
    best_model_type="val_loss"
):
    """
    标准训练流程：
    1. train set：更新模型参数
    2. val set：监控 loss/Dice，并保存最优模型
    3. test set：加载最优模型后，只做最终评价

    val_size 是在 train_val 集合中再次划分的比例。
    默认 test_size=0.2, val_size=0.2，整体约为 train 64%, val 16%, test 20%。
    """
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
    curve_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in sorted(image_dir.glob("*"))
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    valid_image_paths = []

    for image_path in image_paths:
        mask_path = find_mask_path(mask_dir, image_path)

        if mask_path is not None:
            valid_image_paths.append(image_path)
        else:
            print(f"Warning: mask not found for {image_path.name}")

    if len(valid_image_paths) < 3:
        raise ValueError("有效图像数量太少，至少需要 train / val / test 三部分。")

    # 先划出 test，test 不参与模型选择
    train_val_paths, test_paths = train_test_split(
        valid_image_paths,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # 再从 train_val 中划出 val
    train_paths, val_paths = train_test_split(
        train_val_paths,
        test_size=val_size,
        random_state=random_state,
        shuffle=True
    )

    print("\n==============================")
    print("Dataset split")
    print("==============================")
    print(f"Train images: {len(train_paths)}")
    print(f"Val images:   {len(val_paths)}")
    print(f"Test images:  {len(test_paths)}")

    pd.DataFrame({"Train_images": [p.name for p in train_paths]}).to_csv(
        output_dir / "train_image_list.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame({"Val_images": [p.name for p in val_paths]}).to_csv(
        output_dir / "val_image_list.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame({"Test_images": [p.name for p in test_paths]}).to_csv(
        output_dir / "test_image_list.csv",
        index=False,
        encoding="utf-8-sig"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # ==============================
    # 构建数据集
    # ==============================
    train_dataset = GammaPhaseFullImageDataset(
        image_paths=train_paths,
        mask_dir=mask_dir,
        augment=True,
        repeat_factor=repeat_factor
    )

    val_dataset = GammaPhaseFullImageDataset(
        image_paths=val_paths,
        mask_dir=mask_dir,
        augment=False,
        repeat_factor=1
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    pos_weight = compute_pos_weight(train_paths, mask_dir)
    print(f"pos_weight: {pos_weight:.4f}")

    # ==============================
    # 构建模型
    # ==============================
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels
    )

    print("\n==============================")
    print("Training U-Net with standard train/val/test flow")
    print("==============================")

    model, history_df, best_loss_path, best_dice_path = train_unet_standard(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_dir=model_dir,
        pos_weight=pos_weight,
        epochs=epochs,
        learning_rate=learning_rate,
        dice_weight=dice_weight,
        threshold=threshold,
        patience=patience,
        use_amp=use_amp
    )

    history_path = model_dir / "training_log.csv"
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")
    save_training_curves(history_df, curve_dir)

    # ==============================
    # 加载最优模型
    # ==============================
    if best_model_type == "val_dice":
        best_model_path = best_dice_path
    else:
        best_model_path = best_loss_path

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("\n==============================")
    print("Best model loaded")
    print("==============================")
    print(f"Best model type: {best_model_type}")
    print(f"Best model path: {best_model_path}")
    print(f"Best epoch: {checkpoint['epoch']}")

    # ==============================
    # 训练集预测与评价
    # ==============================
    print("\n==============================")
    print("Predicting training images")
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
        use_amp=use_amp
    )

    # ==============================
    # 验证集预测与评价
    # ==============================
    print("\n==============================")
    print("Predicting validation images")
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
        use_amp=use_amp
    )

    # ==============================
    # 测试集预测与评价：最终报告重点看这里
    # ==============================
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
        use_amp=use_amp
    )

    all_results_df = pd.concat(
        [train_results_df, val_results_df, test_results_df],
        ignore_index=True
    )

    (
        per_image_long_df,
        per_image_wide_df,
        mean_by_phase_df,
        std_by_phase_df,
        overall_mean_std_df
    ) = summarize_metrics(
        results_df=all_results_df,
        decimal_places=decimal_places
    )

    # ==============================
    # 分开保存结果
    # ==============================
    train_long_df = per_image_long_df[per_image_long_df["Dataset"] == "train"]
    val_long_df = per_image_long_df[per_image_long_df["Dataset"] == "val"]
    test_long_df = per_image_long_df[per_image_long_df["Dataset"] == "test"]

    train_wide_df = per_image_wide_df[per_image_wide_df["Dataset"] == "train"]
    val_wide_df = per_image_wide_df[per_image_wide_df["Dataset"] == "val"]
    test_wide_df = per_image_wide_df[per_image_wide_df["Dataset"] == "test"]

    train_overall_df = overall_mean_std_df[overall_mean_std_df["Dataset"] == "train"]
    val_overall_df = overall_mean_std_df[overall_mean_std_df["Dataset"] == "val"]
    test_overall_df = overall_mean_std_df[overall_mean_std_df["Dataset"] == "test"]

    train_long_df.to_csv(output_dir / "unet_train_per_image_metrics_long.csv", index=False, encoding="utf-8-sig")
    val_long_df.to_csv(output_dir / "unet_val_per_image_metrics_long.csv", index=False, encoding="utf-8-sig")
    test_long_df.to_csv(output_dir / "unet_test_per_image_metrics_long.csv", index=False, encoding="utf-8-sig")

    train_wide_df.to_csv(output_dir / "unet_train_per_image_metrics_wide.csv", index=False, encoding="utf-8-sig")
    val_wide_df.to_csv(output_dir / "unet_val_per_image_metrics_wide.csv", index=False, encoding="utf-8-sig")
    test_wide_df.to_csv(output_dir / "unet_test_per_image_metrics_wide.csv", index=False, encoding="utf-8-sig")

    train_overall_df.to_csv(output_dir / "unet_train_overall_mean_metrics.csv", index=False, encoding="utf-8-sig")
    val_overall_df.to_csv(output_dir / "unet_val_overall_mean_metrics.csv", index=False, encoding="utf-8-sig")
    test_overall_df.to_csv(output_dir / "unet_test_overall_mean_metrics.csv", index=False, encoding="utf-8-sig")

    mean_by_phase_df.to_csv(output_dir / "unet_mean_metrics_by_phase_train_val_test.csv", index=False, encoding="utf-8-sig")
    std_by_phase_df.to_csv(output_dir / "unet_std_metrics_by_phase_train_val_test.csv", index=False, encoding="utf-8-sig")
    overall_mean_std_df.to_csv(output_dir / "unet_overall_mean_std_train_val_test.csv", index=False, encoding="utf-8-sig")

    # ==============================
    # 保存 Excel
    # ==============================
    excel_path = output_dir / "unet_standard_train_val_test_metrics.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        history_df.to_excel(writer, sheet_name="Training_Log", index=False)

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
    print("U-Net standard train/val/test evaluation finished.")
    print("==============================")

    print("\nMean metrics by phase:")
    print(mean_by_phase_df)

    print("\nStd metrics by phase:")
    print(std_by_phase_df)

    print("\nOverall mean and std:")
    print(overall_mean_std_df)

    print(f"\nBest model saved to: {best_model_path}")
    print(f"Training log saved to: {history_path}")
    print(f"Training curves saved to: {curve_dir}")

    print(f"\nTrain prediction masks saved to: {train_pred_mask_dir}")
    print(f"Val prediction masks saved to:   {val_pred_mask_dir}")
    print(f"Test prediction masks saved to:  {test_pred_mask_dir}")

    print(f"\nTrain comparison figures saved to: {train_comparison_dir}")
    print(f"Val comparison figures saved to:   {val_comparison_dir}")
    print(f"Test comparison figures saved to:  {test_comparison_dir}")

    print(f"\nExcel file saved to: {excel_path}")

    return (
        model,
        history_df,
        train_results_df,
        val_results_df,
        test_results_df,
        mean_by_phase_df,
        std_by_phase_df,
        overall_mean_std_df
    )


# ==============================
# 主程序入口
# ==============================
if __name__ == "__main__":

    (
        model,
        history_df,
        train_results_df,
        val_results_df,
        test_results_df,
        mean_by_phase_df,
        std_by_phase_df,
        overall_mean_std_df
    ) = unet_standard_pipeline(
        image_dir="dataset/images",
        mask_dir="dataset/masks",
        output_dir="dataset/results_unet_standard",

        # 数据划分
        # 先划出 20% 作为 test，再从剩余 train_val 中划出 20% 作为 val
        # 总体约为 train 64%, val 16%, test 20%
        test_size=0.2,
        val_size=0.2,

        # 整图训练建议 batch_size=1
        batch_size=8,

        # 训练参数
        epochs=200,
        learning_rate=1e-4,

        # 1024×1024 整图训练时，建议先用 16；显存够再改 32
        base_channels=16,

        # 预测阈值
        threshold=0.5,

        # 每个 epoch 中每张训练图重复增强次数
        # 1 表示每张训练图每轮出现 1 次；数据少时可以设为 3 或 5
        repeat_factor=1,

        # BCE + Dice Loss 中 Dice Loss 的权重
        dice_weight=0.5,

        # Early stopping，按 val_loss 监控
        patience=10,

        # CUDA 下使用混合精度，降低显存占用
        use_amp=True,

        # 随机种子
        random_state=42,

        # 指标保留小数位
        decimal_places=4,

        # 可选："val_loss" 或 "val_dice"
        # 论文中更建议默认使用 val_loss 选择模型，然后 test 只做最终评价
        best_model_type="val_loss"
    )
