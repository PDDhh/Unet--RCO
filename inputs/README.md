# Input Data Layout

Datasets are not committed to this repository.

Create one directory per dataset:

```text
inputs/<dataset_name>/
|-- images/
|-- masks/
|-- inst_targets/
```

`images/` and `masks/` must use matching file stems. Generate `inst_targets/`
with:

```bash
python -m scripts.generate_rco_targets \
  --mask-dir inputs/<dataset_name>/masks \
  --output-dir inputs/<dataset_name>/inst_targets
```
