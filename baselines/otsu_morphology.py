"""Otsu thresholding with morphology baseline."""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import ndimage as ndi
from skimage.morphology import disk, binary_opening, binary_closing, remove_small_objects


# ==============================
# 基本设置
# ==============================
IMAGE_SIZE = (1024, 1024)

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


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
# 寻找对应人工标记 mask
# ==============================
def find_mask_path(mask_dir, image_path):
    """
    优先寻找与原图完全同名的 mask；
    如果没有，则寻找相同 stem 的不同格式 mask。
    """
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
# Otsu + morphology 分割
# ==============================
def otsu_morphology_segmentation(
    image,
    gamma_prime_is_dark=True,
    gaussian_kernel=5,
    morph_radius=2,
    min_area=64
):
    """
    基于 Otsu + morphology 的 γ′ 相 / γ 相识别。

    当前约定：
    γ′ 相 = 黑色 0
    γ 相  = 白色 255

    Parameters
    ----------
    image : ndarray
        输入灰度图像，尺寸为 1024×1024

    gamma_prime_is_dark : bool
        True 表示原始图像中 γ′ 相为暗相/黑色
        False 表示原始图像中 γ′ 相为亮相/白色

    gaussian_kernel : int
        高斯滤波核大小，一般取 3 或 5

    morph_radius : int
        形态学结构元素半径，一般取 1~3

    min_area : int
        去除小区域面积阈值，单位为像素

    Returns
    -------
    pred_mask : ndarray
        预测结果 mask：
        γ′ 相 = 黑色 0
        γ 相  = 白色 255
    """

    # 1. 高斯滤波降噪
    blurred = cv2.GaussianBlur(
        image,
        (gaussian_kernel, gaussian_kernel),
        0
    )

    # 2. Otsu 自动阈值分割
    # binary 中亮区域为 255，暗区域为 0
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 3. 提取 γ′ 相区域
    # 如果 γ′ 是暗相，则 binary == 0 为 γ′
    # 如果 γ′ 是亮相，则 binary == 255 为 γ′
    if gamma_prime_is_dark:
        gamma_prime = binary == 0
    else:
        gamma_prime = binary == 255

    # 4. 形态学开运算：去除孤立噪点
    selem = disk(morph_radius)
    gamma_prime = binary_opening(gamma_prime, selem)

    # 5. 形态学闭运算：平滑边界、连接局部断裂
    gamma_prime = binary_closing(gamma_prime, selem)

    # 6. 孔洞填充
    gamma_prime = ndi.binary_fill_holes(gamma_prime)

    # 7. 去除小区域
    gamma_prime = remove_small_objects(
        gamma_prime.astype(bool),
        min_size=min_area
    )

    # 8. 输出 mask，保持你的标记规则：
    # γ′ 相 = 黑色 0
    # γ 相  = 白色 255
    pred_mask = np.where(gamma_prime, 0, 255).astype(np.uint8)

    return pred_mask


# ==============================
# 计算指标
# ==============================
def calculate_metrics(pred_bool, gt_bool):
    """
    计算 Precision、Recall、Dice、IoU。

    pred_bool : bool ndarray
        预测中属于当前相的区域

    gt_bool : bool ndarray
        人工标记中属于当前相的区域
    """

    TP = np.logical_and(pred_bool, gt_bool).sum()
    FP = np.logical_and(pred_bool, ~gt_bool).sum()
    FN = np.logical_and(~pred_bool, gt_bool).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return precision, recall, dice, iou


# ==============================
# 单张图像评价
# ==============================
def evaluate_one_image(pred_mask, gt_mask, image_name):
    """
    根据规则：
    γ′ 相 = 黑色 0
    γ 相  = 白色 255

    分别计算 γ′ 相和 γ 相的四个指标。
    """

    # 二值化，避免 mask 中存在 0/255 以外的灰度值
    pred_mask = (pred_mask > 127).astype(np.uint8) * 255
    gt_mask = (gt_mask > 127).astype(np.uint8) * 255

    # γ′ 相：黑色区域
    pred_gamma_prime = pred_mask <= 127
    gt_gamma_prime = gt_mask <= 127

    # γ 相：白色区域
    pred_gamma = pred_mask > 127
    gt_gamma = gt_mask > 127

    # γ′ 相指标
    gp_precision, gp_recall, gp_dice, gp_iou = calculate_metrics(
        pred_gamma_prime,
        gt_gamma_prime
    )

    # γ 相指标
    g_precision, g_recall, g_dice, g_iou = calculate_metrics(
        pred_gamma,
        gt_gamma
    )

    rows = [
        {
            "Image": image_name,
            "Phase": "gamma_prime",
            "Precision": gp_precision,
            "Recall": gp_recall,
            "Dice": gp_dice,
            "IoU": gp_iou
        },
        {
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
# 转换为每张图一行的宽表
# ==============================
def make_wide_table(results_df):
    """
    将两行一张图的结果转换为一行一张图。
    """

    wide_df = results_df.pivot(
        index="Image",
        columns="Phase",
        values=["Precision", "Recall", "Dice", "IoU"]
    )

    wide_df.columns = [
        f"{phase}_{metric}"
        for metric, phase in wide_df.columns
    ]

    wide_df = wide_df.reset_index()

    column_order = [
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
# 批量处理
# ==============================
def batch_process(
    image_dir,
    mask_dir,
    output_dir,
    gamma_prime_is_dark=True,
    gaussian_kernel=5,
    morph_radius=2,
    min_area=64,
    decimal_places=4
):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    pred_mask_dir = output_dir / "pred_masks"

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_mask_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    image_paths = [
        p for p in sorted(image_dir.glob("*"))
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {image_dir}")

    for image_path in image_paths:
        mask_path = find_mask_path(mask_dir, image_path)

        if mask_path is None:
            print(f"Warning: mask not found for {image_path.name}")
            continue

        # 读取原始图像和人工标记
        image = read_gray_image(image_path)
        gt_mask = read_gray_image(mask_path)

        # 人工标记二值化：
        # γ′ 相 = 黑色 0
        # γ 相  = 白色 255
        gt_mask = (gt_mask > 127).astype(np.uint8) * 255

        # Otsu + morphology 分割
        pred_mask = otsu_morphology_segmentation(
            image=image,
            gamma_prime_is_dark=gamma_prime_is_dark,
            gaussian_kernel=gaussian_kernel,
            morph_radius=morph_radius,
            min_area=min_area
        )

        # 保存预测 mask
        # 保存结果仍然是：
        # γ′ 相 = 黑色 0
        # γ 相  = 白色 255
        cv2.imwrite(
            str(pred_mask_dir / image_path.name),
            pred_mask
        )

        # 计算当前图像指标
        rows = evaluate_one_image(
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            image_name=image_path.name
        )

        all_results.extend(rows)

        print(f"Processed: {image_path.name}")

    # ==============================
    # 生成结果表
    # ==============================
    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        raise ValueError("No valid image-mask pairs were processed.")

    # 每张图两行：γ′ 相一行，γ 相一行
    per_image_long_df = results_df.copy()

    # 每张图一行：适合论文制表
    per_image_wide_df = make_wide_table(results_df)

    # 按相类别求平均值
    mean_by_phase_df = results_df.groupby("Phase")[
        ["Precision", "Recall", "Dice", "IoU"]
    ].mean().reset_index()


    overall_mean_df = pd.DataFrame([
        {
            "Type": "overall_mean",
            "Precision": results_df["Precision"].mean(),
            "Recall": results_df["Recall"].mean(),
            "Dice": results_df["Dice"].mean(),
            "IoU": results_df["IoU"].mean()
        },
        {
            "Type": "overall_std",
            "Precision": results_df["Precision"].std(),
            "Recall": results_df["Recall"].std(),
            "Dice": results_df["Dice"].std(),
            "IoU": results_df["IoU"].std()
        }
    ])

    # 保留小数位
    metric_columns = ["Precision", "Recall", "Dice", "IoU"]

    per_image_long_df[metric_columns] = per_image_long_df[metric_columns].round(decimal_places)
    mean_by_phase_df[metric_columns] = mean_by_phase_df[metric_columns].round(decimal_places)
    overall_mean_df[metric_columns] = overall_mean_df[metric_columns].round(decimal_places)

    wide_metric_columns = [
        c for c in per_image_wide_df.columns
        if c != "Image"
    ]

    per_image_wide_df[wide_metric_columns] = per_image_wide_df[wide_metric_columns].round(decimal_places)

    # ==============================
    # 保存 CSV
    # ==============================
    per_image_long_df.to_csv(
        output_dir / "per_image_metrics_long.csv",
        index=False,
        encoding="utf-8-sig"
    )

    per_image_wide_df.to_csv(
        output_dir / "per_image_metrics_wide.csv",
        index=False,
        encoding="utf-8-sig"
    )

    mean_by_phase_df.to_csv(
        output_dir / "mean_metrics_by_phase.csv",
        index=False,
        encoding="utf-8-sig"
    )

    overall_mean_df.to_csv(
        output_dir / "overall_mean_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ==============================
    # 保存 Excel
    # ==============================
    excel_path = output_dir / "otsu_morphology_metrics.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        per_image_long_df.to_excel(
            writer,
            sheet_name="Per_Image_Long",
            index=False
        )

        per_image_wide_df.to_excel(
            writer,
            sheet_name="Per_Image_Wide",
            index=False
        )

        mean_by_phase_df.to_excel(
            writer,
            sheet_name="Mean_By_Phase",
            index=False
        )

        overall_mean_df.to_excel(
            writer,
            sheet_name="Overall_Mean",
            index=False
        )

    print("\n==============================")
    print("Processing finished.")
    print("==============================")

    print("\nPer-image metrics:")
    print(per_image_long_df)

    print("\nMean metrics by phase:")
    print(mean_by_phase_df)

    print("\nOverall mean metrics:")
    print(overall_mean_df)

    print(f"\nPredicted masks saved to: {pred_mask_dir}")
    print(f"Results saved to: {output_dir}")
    print(f"Excel file saved to: {excel_path}")

    return per_image_long_df, per_image_wide_df, mean_by_phase_df, overall_mean_df


# ==============================
# 主程序入口
# ==============================
if __name__ == "__main__":

    per_image_long_df, per_image_wide_df, mean_by_phase_df, overall_mean_df = batch_process(
        image_dir="dataset/images",
        mask_dir="dataset/masks",
        output_dir="dataset/results",


        gamma_prime_is_dark=True,

        # Otsu + morphology 参数
        gaussian_kernel=5,
        morph_radius=2,
        min_area=64,

        # 结果保留 4 位小数
        decimal_places=4
    )
