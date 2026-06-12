# U-Net++-RCO

Official-style research code layout for U-Net++ with region, center, and offset
(RCO) supervision for grayscale phase segmentation.

The repository contains the proposed U-Net++-RCO workflow and four reference
baselines used for comparison. The core workflow is designed to run from the
repository root without editing source files.

## Method Overview

U-Net++-RCO predicts four maps:

| Channel | Name | Description |
| --- | --- | --- |
| 0 | `region` | Foreground region probability |
| 1 | `center` | Normalized distance-to-boundary center score |
| 2 | `offset_y` | Normalized vertical direction toward the instance center |
| 3 | `offset_x` | Normalized horizontal direction toward the instance center |

Inference combines the region, center, and offset maps to refine uncertain
boundary pixels. This RCO refinement is part of the default inference path. It
does not apply morphology. Small isolated predictions are filtered by connected
component area.

See [docs/method.md](docs/method.md) for the target definitions and inference
details.

## Repository Layout

```text
.
|-- rco/
|   |-- architectures.py
|-- scripts/
|   |-- generate_rco_targets.py
|   |-- train_rco.py
|   |-- predict_rco.py
|-- baselines/
|   |-- otsu_morphology.py
|   |-- random_forest.py
|   |-- unet.py
|   |-- unetplusplus.py
|-- configs/
|   |-- unetpp_rco_example.yaml
|-- docs/
|   |-- method.md
|   |-- reproducibility.md
|-- inputs/
|   |-- README.md
```

Generated checkpoints are written to `models/`. Prediction outputs are written
to `outputs/`. Both directories are ignored by Git.

## Installation

The workflow was verified with Conda on Windows. The clean reconstruction is
defined in `environment.yml`:

- Python `3.8.0`
- PyTorch `2.1.0+cu121`
- CUDA runtime `12.1`
- cuDNN `8801`
- NVIDIA GeForce RTX 4060 Laptop GPU

The environment file intentionally installs one OpenCV distribution. Avoid
installing `opencv-python`, `opencv-contrib-python`, and
`opencv-python-headless` together because they provide the same `cv2` package.

Create the environment:

```bash
conda env create -f environment.yml
conda activate unetpp-rco
```

To use an existing environment, install the core dependencies:

```bash
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA version if a different
system configuration is used. To run the supplementary baselines, install:

```bash
pip install -r requirements-baselines.txt
```

## Data Preparation

Place images and binary masks under:

```text
inputs/<dataset_name>/
|-- images/
|   |-- sample_001.tif
|-- masks/
|   |-- sample_001.tif
```

The file stems must match. Generate the RCO supervision targets:

```bash
python -m scripts.generate_rco_targets \
  --mask-dir inputs/<dataset_name>/masks \
  --output-dir inputs/<dataset_name>/inst_targets \
  --img-ext .tif \
  --threshold 50 \
  --foreground dark \
  --min-size 20
```

Each generated `.npz` file contains `region`, `center`, `offset_y`, and
`offset_x`. By default, connected foreground pixels are treated as one instance.
If touching objects must remain separate, provide instance-aware annotations or
extend the target generation step.

## Training

Review [configs/unetpp_rco_example.yaml](configs/unetpp_rco_example.yaml), then
run:

```bash
python -m scripts.train_rco \
  --config configs/unetpp_rco_example.yaml \
  --dataset <dataset_name> \
  --img_ext .tif
```

The training script stores:

- `models/<name>/config.yml`
- `models/<name>/splits/train_ids.txt`
- `models/<name>/splits/val_ids.txt`
- `models/<name>/splits/test_ids.txt`
- Best checkpoints selected by validation IoU, Dice, and loss
- Training curves and logs

For a paper release, commit the exact split ID files used for the reported
results under `paper_splits/<dataset_name>/`.

## Prediction And Evaluation

Predict a directory:

```bash
python -m scripts.predict_rco \
  --name <model_name> \
  --predict-dir inputs/<dataset_name>/images \
  --save-vis \
  --save-prob
```

Evaluate the saved test split and search the validation threshold:

```bash
python -m scripts.predict_rco \
  --name <model_name> \
  --predict-dir inputs/<dataset_name>/images \
  --split-name val \
  --id-list auto \
  --gt-dir inputs/<dataset_name>/inst_targets \
  --search-threshold
```

Prediction settings are recorded in `prediction_settings.yml`, including the
RCO refinement parameters and connected-component size threshold.

Brightness and contrast enhancement is enabled by default during prediction to
preserve the reference workflow. Use `--no-enhance` for an ablation or when the
training protocol does not use this preprocessing choice.

## Baselines

Reference implementations are stored in `baselines/`:

| Script | Method |
| --- | --- |
| `otsu_morphology.py` | Otsu thresholding with morphology |
| `random_forest.py` | Pixel-level random forest |
| `unet.py` | Standard U-Net |
| `unetplusplus.py` | Standard U-Net++ |

These scripts retain their original experiment entry points and default
`dataset/` paths so the reported baseline behavior stays intact. Before a paper
release, archive the exact baseline commands and result files alongside the
reported tables.

## Reproducibility

The example config is a starting point, not a claim about the final paper
settings. Before publication:

1. Replace it with the exact configuration used for the paper.
2. Commit the fixed train, validation, and test split files.
3. Record the Python, CUDA, and GPU versions.
4. Add the released checkpoint URL and dataset access instructions.
5. Update `CITATION.cff` with the paper authors, title, DOI, and repository URL.

See [docs/reproducibility.md](docs/reproducibility.md) for a release checklist.

## Citation

If this repository supports a publication, update `CITATION.cff` before the
public release. GitHub will expose the completed metadata through its
**Cite this repository** interface.

## License

Released under the MIT License. See [LICENSE](LICENSE).
