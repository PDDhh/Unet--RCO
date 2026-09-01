# U-Net++-RCO

Research code for **U-Net++ with Region, Center, and Offset (RCO) supervision** for segmentation and quantitative characterization of γ′ precipitates in grayscale SEM images.

This repository accompanies the manuscript:

**Deep learning segmentation with quality control for γ/γ′ microstructure characterization in Ni3Al-based single-crystal superalloys**

## Method overview

U-Net++-RCO predicts four output maps:

| Channel | Output | Description |
|---|---|---|
| 0 | `region` | Foreground γ′ region probability |
| 1 | `center` | Normalized center score |
| 2 | `offset_y` | Normalized vertical direction toward the instance center |
| 3 | `offset_x` | Normalized horizontal direction toward the instance center |

During inference, the region prediction is refined using the center and offset outputs in uncertain pixels. The default paper inference settings are:

- Region threshold: `0.5`
- Uncertain interval: `0.35–0.65`
- `alpha = 0.15`
- `beta = 0.15`

A deterministic grayscale brightness/contrast enhancement is enabled by default during inference. It can be disabled with `--no-enhance`.

## Repository structure

```text
.
├── baselines/
│   ├── otsu_morphology.py
│   ├── random_forest.py
│   ├── unet.py
│   └── unetplusplus.py
├── configs/
│   ├── unetpp_rco_paper.yaml
│   └── unetpp_rco_example.yaml
├── inputs/
│   └── README.md
├── rco/
│   └── architectures.py
├── scripts/
│   ├── generate_rco_targets.py
│   ├── train_rco.py
│   └── predict_rco.py
├── environment.yml
├── requirements.txt
├── LICENSE
└── README.md
```

Generated model files and prediction outputs are written to directories such as `models/` and `outputs/`, which are excluded from the repository.

## Installation

Using Conda:

```bash
conda env create -f environment.yml
conda activate unetpp-rco
```

Alternatively, install the core dependencies in an existing Python environment:

```bash
pip install -r requirements.txt
```

Users may need to install a PyTorch build appropriate for their own CUDA/CPU environment.

## Data preparation

The original SEM images, annotations, and pretrained model weights used in the paper are **not included** in this repository.

Users should prepare their own grayscale SEM images and corresponding binary annotations. A typical dataset layout is:

```text
inputs/<dataset_name>/
├── images/
│   └── sample_001.tif
└── masks/
    └── sample_001.tif
```

Image and mask file stems should match.

Generate RCO supervision targets with:

```bash
python -m scripts.generate_rco_targets \
  --mask-dir inputs/<dataset_name>/masks \
  --output-dir inputs/<dataset_name>/inst_targets \
  --img-ext .tif \
  --threshold 50 \
  --foreground dark \
  --min-size 20
```

Each generated `.npz` file contains:

- `region`
- `center`
- `offset_y`
- `offset_x`

## Training

The configuration used for the paper is:

```text
configs/unetpp_rco_paper.yaml
```

Run one fold at a time:

```bash
python -m scripts.train_rco \
  --config configs/unetpp_rco_paper.yaml \
  --dataset <dataset_name> \
  --img_ext .tif \
  --fold 1
```

Repeat with `--fold 2`, `--fold 3`, `--fold 4`, and `--fold 5`.

The paper configuration uses:

- Input size: `1024 × 1024`
- Grayscale input: `1` channel
- RCO outputs: `4` channels
- Base channels: `32`
- GroupNorm groups: `8`
- Deep supervision: disabled
- Epochs: `200`
- Batch size: `2`
- Optimizer: AdamW
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Scheduler: CosineAnnealingLR
- Minimum learning rate: `1e-5`
- Five-fold image-level cross-validation
- Random seed: `41`
- `lambda_region = lambda_center = lambda_offset = 1.0`
- Best checkpoint selected according to validation IoU

For the 110-image dataset used in the study, each fold contained approximately:

- 79 training images
- 9 validation images
- 22 test images

The exact experimental SEM dataset is not distributed with this repository.

## Prediction

Predict a directory of SEM images with:

```bash
python -m scripts.predict_rco \
  --name <model_name> \
  --predict-dir inputs/<dataset_name>/images \
  --save-vis \
  --save-prob
```

The default inference workflow uses the RCO refinement described above and includes deterministic grayscale enhancement.

To disable the grayscale enhancement:

```bash
python -m scripts.predict_rco \
  --name <model_name> \
  --predict-dir inputs/<dataset_name>/images \
  --no-enhance
```

## Validation threshold search

If reference targets are available, validation threshold search can be performed with:

```bash
python -m scripts.predict_rco \
  --name <model_name> \
  --predict-dir inputs/<dataset_name>/images \
  --split-name val \
  --id-list auto \
  --gt-dir inputs/<dataset_name>/inst_targets \
  --search-threshold
```

## Baselines

Reference implementations used for comparison are included in `baselines/`:

| Script | Method |
|---|---|
| `otsu_morphology.py` | Otsu thresholding |
| `random_forest.py` | Random Forest |
| `unet.py` | U-Net |
| `unetplusplus.py` | U-Net++ |

These baseline scripts retain their experiment-oriented entry points and may require additional Python packages depending on the selected method.

## Data and model availability

The original experimental SEM dataset and pretrained model weights are not distributed in this repository. Users should provide their own appropriately prepared SEM images and annotations.

## License

This repository is released under the MIT License.
