"""Random forest baseline for full-image phase segmentation."""

import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from scipy import ndimage as ndi
from skimage.filters import sobel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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
# 像素级特征提取
# ==============================
def extract_pixel_features(image):
    """
    提取像素级特征：
    1. 原始灰度
    2. Gaussian sigma=1
    3. Gaussian sigma=2
    4. Sobel 边缘
    5. Laplacian 边缘
    6. 局部均值
    7. 局部标准差
    """

    image_float = image.astype(np.float32) / 255.0

    gray = image_float
    gaussian_1 = ndi.gaussian_filter(image_float, sigma=1)
    gaussian_2 = ndi.gaussian_filter(image_float, sigma=2)
    sobel_edge = sobel(image_float)
    laplacian = cv2.Laplacian(image_float, cv2.CV_32F)

    local_mean = ndi.uniform_filter(image_float, size=7)
    local_mean_sq = ndi.uniform_filter(image_float ** 2, size=7)
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean ** 2, 0))

    feature_stack = np.stack(
        [
            gray,
            gaussian_1,
            gaussian_2,
            sobel_edge,
            laplacian,
            local_mean,
            local_std
        ],
        axis=-1
    )

    features = feature_stack.reshape(-1, feature_stack.shape[-1])

    return features


# ==============================
# 采样训练像素
# ==============================
def sample_training_pixels(
    image,
    gt_mask,
    samples_per_class=5000,
    random_state=42
):
    """
    当前规则：
    γ′ 相 = 黑色 0，对应标签 1
    γ 相  = 白色 255，对应标签 0
    """

    rng = np.random.default_rng(random_state)

    features = extract_pixel_features(image)

    gt_mask = (gt_mask > 127).astype(np.uint8) * 255

    gamma_prime_pixels = np.where(gt_mask.reshape(-1) <= 127)[0]
    gamma_pixels = np.where(gt_mask.reshape(-1) > 127)[0]

    n_gp = min(samples_per_class, len(gamma_prime_pixels))
    n_g = min(samples_per_class, len(gamma_pixels))

    if n_gp == 0 or n_g == 0:
        raise ValueError("Mask 中某一类别像素数量为 0，请检查人工标记。")

    selected_gp = rng.choice(gamma_prime_pixels, size=n_gp, replace=False)
    selected_g = rng.choice(gamma_pixels, size=n_g, replace=False)

    selected_indices = np.concatenate([selected_gp, selected_g])

    X = features[selected_indices]

    y_gp = np.ones(n_gp, dtype=np.uint8)
    y_g = np.zeros(n_g, dtype=np.uint8)
    y = np.concatenate([y_gp, y_g])

    return X, y


# ==============================
# 构建训练数据
# ==============================
def build_training_dataset(
    image_paths,
    mask_dir,
    samples_per_class=5000,
    random_state=42
):
    X_list = []
    y_list = []

    for i, image_path in enumerate(image_paths):
        mask_path = find_mask_path(mask_dir, image_path)

        if mask_path is None:
            print(f"Warning: mask not found for {image_path.name}")
            continue

        image = read_gray_image(image_path)
        gt_mask = read_gray_image(mask_path)

        X, y = sample_training_pixels(
            image=image,
            gt_mask=gt_mask,
            samples_per_class=samples_per_class,
            random_state=random_state + i
        )

        X_list.append(X)
        y_list.append(y)

        print(f"Training samples extracted: {image_path.name}")

    if len(X_list) == 0:
        raise ValueError("No training data was generated.")

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    return X_train, y_train


# ==============================
# 训练 Random Forest
# ==============================
def train_random_forest(
    X_train,
    y_train,
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs
    )

    model.fit(X_train, y_train)

    return model


# ==============================
# 单张图像预测
# ==============================
def predict_one_image(model, image):
    """
    输出规则：
    γ′ 相 = 黑色 0
    γ 相  = 白色 255
    """

    features = extract_pixel_features(image)

    pred_label = model.predict(features)
    pred_label = pred_label.reshape(image.shape)

    pred_mask = np.where(pred_label == 1, 0, 255).astype(np.uint8)

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


# ==============================
# 单张图像评价
# ==============================
def evaluate_one_image(pred_mask, gt_mask, image_name, dataset_type):
    """
    当前规则：
    γ′ 相 = 黑色 0
    γ 相  = 白色 255
    """

    pred_mask = (pred_mask > 127).astype(np.uint8) * 255
    gt_mask = (gt_mask > 127).astype(np.uint8) * 255

    pred_gamma_prime = pred_mask <= 127
    gt_gamma_prime = gt_mask <= 127

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
    """
    保存三联图：
    原始图像 / 人工标记 / RF预测结果
    """

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
    plt.title("RF prediction")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
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
# 保存某一数据集的统计结果
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

    overall_mean_std_df = results_df.groupby("Dataset")[
        metric_columns
    ].agg(["mean", "std"])

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
# 对训练集或测试集进行预测和评价
# ==============================
def predict_and_evaluate_dataset(
    model,
    image_paths,
    mask_dir,
    pred_mask_dir,
    comparison_dir,
    dataset_type
):
    pred_mask_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for image_path in image_paths:
        mask_path = find_mask_path(mask_dir, image_path)

        image = read_gray_image(image_path)
        gt_mask = read_gray_image(mask_path)

        gt_mask = (gt_mask > 127).astype(np.uint8) * 255

        pred_mask = predict_one_image(
            model=model,
            image=image
        )

        # 保存预测 mask
        cv2.imwrite(
            str(pred_mask_dir / image_path.name),
            pred_mask
        )

        # 保存预测对比图
        comparison_path = comparison_dir / f"{image_path.stem}_comparison.png"

        save_comparison_figure(
            image=image,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            save_path=comparison_path,
            title=f"{dataset_type}: {image_path.name}"
        )

        # 计算指标
        rows = evaluate_one_image(
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            image_name=image_path.name,
            dataset_type=dataset_type
        )

        all_results.extend(rows)

        print(f"{dataset_type} predicted and evaluated: {image_path.name}")

    results_df = pd.DataFrame(all_results)

    return results_df


# ==============================
# 主流程
# ==============================
def random_forest_pipeline(
    image_dir,
    mask_dir,
    output_dir,
    test_size=0.3,
    samples_per_class=5000,
    n_estimators=200,
    max_depth=None,
    random_state=42,
    decimal_places=4
):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    model_dir = output_dir / "model"

    train_pred_mask_dir = output_dir / "train_pred_masks"
    test_pred_mask_dir = output_dir / "test_pred_masks"

    train_comparison_dir = output_dir / "train_comparison_figures"
    test_comparison_dir = output_dir / "test_comparison_figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

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

    if len(valid_image_paths) < 2:
        raise ValueError("有效图像数量太少，无法划分训练集和测试集。")

    train_paths, test_paths = train_test_split(
        valid_image_paths,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    print("\n==============================")
    print("Dataset split")
    print("==============================")
    print(f"Train images: {len(train_paths)}")
    print(f"Test images:  {len(test_paths)}")

    # 保存训练集/测试集文件名，方便论文记录
    pd.DataFrame({"Train_images": [p.name for p in train_paths]}).to_csv(
        output_dir / "train_image_list.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame({"Test_images": [p.name for p in test_paths]}).to_csv(
        output_dir / "test_image_list.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ==============================
    # 构建训练样本
    # ==============================
    print("\n==============================")
    print("Building training dataset")
    print("==============================")

    X_train, y_train = build_training_dataset(
        image_paths=train_paths,
        mask_dir=mask_dir,
        samples_per_class=samples_per_class,
        random_state=random_state
    )

    print(f"Training feature shape: {X_train.shape}")
    print(f"Training label shape:   {y_train.shape}")

    # ==============================
    # 训练模型
    # ==============================
    print("\n==============================")
    print("Training Random Forest")
    print("==============================")

    rf_model = train_random_forest(
        X_train=X_train,
        y_train=y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )

    model_path = model_dir / "random_forest_gamma_prime_gamma.joblib"
    joblib.dump(rf_model, model_path)

    print(f"Model saved to: {model_path}")

    # ==============================
    # 训练集预测与评价
    # ==============================
    print("\n==============================")
    print("Predicting training images")
    print("==============================")

    train_results_df = predict_and_evaluate_dataset(
        model=rf_model,
        image_paths=train_paths,
        mask_dir=mask_dir,
        pred_mask_dir=train_pred_mask_dir,
        comparison_dir=train_comparison_dir,
        dataset_type="train"
    )

    # ==============================
    # 测试集预测与评价
    # ==============================
    print("\n==============================")
    print("Predicting test images")
    print("==============================")

    test_results_df = predict_and_evaluate_dataset(
        model=rf_model,
        image_paths=test_paths,
        mask_dir=mask_dir,
        pred_mask_dir=test_pred_mask_dir,
        comparison_dir=test_comparison_dir,
        dataset_type="test"
    )

    # 合并结果
    all_results_df = pd.concat(
        [train_results_df, test_results_df],
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
    # 分开保存训练集和测试集结果
    # ==============================
    train_long_df = per_image_long_df[per_image_long_df["Dataset"] == "train"]
    test_long_df = per_image_long_df[per_image_long_df["Dataset"] == "test"]

    train_wide_df = per_image_wide_df[per_image_wide_df["Dataset"] == "train"]
    test_wide_df = per_image_wide_df[per_image_wide_df["Dataset"] == "test"]

    train_overall_df = overall_mean_std_df[overall_mean_std_df["Dataset"] == "train"]
    test_overall_df = overall_mean_std_df[overall_mean_std_df["Dataset"] == "test"]

    train_long_df.to_csv(
        output_dir / "rf_train_per_image_metrics_long.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test_long_df.to_csv(
        output_dir / "rf_test_per_image_metrics_long.csv",
        index=False,
        encoding="utf-8-sig"
    )

    train_wide_df.to_csv(
        output_dir / "rf_train_per_image_metrics_wide.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test_wide_df.to_csv(
        output_dir / "rf_test_per_image_metrics_wide.csv",
        index=False,
        encoding="utf-8-sig"
    )

    train_overall_df.to_csv(
        output_dir / "rf_train_overall_mean_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test_overall_df.to_csv(
        output_dir / "rf_test_overall_mean_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    mean_by_phase_df.to_csv(
        output_dir / "rf_mean_metrics_by_phase_train_test.csv",
        index=False,
        encoding="utf-8-sig"
    )

    std_by_phase_df.to_csv(
        output_dir / "rf_std_metrics_by_phase_train_test.csv",
        index=False,
        encoding="utf-8-sig"
    )

    overall_mean_std_df.to_csv(
        output_dir / "rf_overall_mean_std_train_test.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ==============================
    # 保存 Excel
    # ==============================
    excel_path = output_dir / "random_forest_train_test_metrics.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        train_long_df.to_excel(writer, sheet_name="Train_Long", index=False)
        test_long_df.to_excel(writer, sheet_name="Test_Long", index=False)

        train_wide_df.to_excel(writer, sheet_name="Train_Wide", index=False)
        test_wide_df.to_excel(writer, sheet_name="Test_Wide", index=False)

        mean_by_phase_df.to_excel(writer, sheet_name="Mean_By_Phase", index=False)
        std_by_phase_df.to_excel(writer, sheet_name="Std_By_Phase", index=False)
        overall_mean_std_df.to_excel(writer, sheet_name="Overall_Mean_Std", index=False)

    print("\n==============================")
    print("Random Forest train/test evaluation finished.")
    print("==============================")

    print("\nMean metrics by phase:")
    print(mean_by_phase_df)

    print("\nOverall mean and std:")
    print(overall_mean_std_df)

    print(f"\nTrain prediction masks saved to: {train_pred_mask_dir}")
    print(f"Test prediction masks saved to:  {test_pred_mask_dir}")
    print(f"Train comparison figures saved to: {train_comparison_dir}")
    print(f"Test comparison figures saved to:  {test_comparison_dir}")
    print(f"Excel file saved to: {excel_path}")

    return (
        rf_model,
        train_results_df,
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
        rf_model,
        train_results_df,
        test_results_df,
        mean_by_phase_df,
        std_by_phase_df,
        overall_mean_std_df
    ) = random_forest_pipeline(
        image_dir="dataset/images",
        mask_dir="dataset/masks",
        output_dir="dataset/results_rf",

        # 测试集比例
        test_size=0.2,

        # 每张训练图中，每一类采样像素数
        samples_per_class=50000,

        # Random Forest 参数
        n_estimators=200,
        max_depth=None,

        # 随机种子
        random_state=42,

        # 小数位数
        decimal_places=4
    )
