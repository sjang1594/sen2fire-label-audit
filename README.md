# The Labels Are the Ceiling: A Model-Free Audit of Sen2Fire

[![Checkpoints DOI](https://img.shields.io/badge/checkpoints-10.5281%2Fzenodo.24083276-blue)](https://doi.org/10.5281/zenodo.24083276)
[![License: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Seungho Jang<sup>1</sup>, Rumandeep Singh<sup>1</sup>, Sanjiv K. Bhatia<sup>1</sup>, Sunil Bhaskaran<sup>2</sup>

<sup>1</sup> University of Missouri–St. Louis &nbsp; <sup>2</sup> The Graduate Center, City University of New York

*Under review.* Code, result tables and trained checkpoints for the paper.

## Overview

Sentinel-2 imagery offers finer spatial detail than operational fire products, but the
[Sen2Fire](https://zenodo.org/records/10881058) wildfire segmentation benchmark maps coarse, same-day
active fire detections onto its 10 m imagery. We audit it across spectral inputs, loss functions,
augmentation and network architectures, with a blind visual census of its test labels and a
within-task control on [CaBuAr](https://doi.org/10.1109/MGRS.2023.3292467), whose labels are 20 m
burn perimeters.

**Key findings**

- DeepLabV3 achieves 36.6 % pooled pixel F1, 8.5 percentage points above the published baseline.
- In a blind census of the 99 labeled test patches, no fire or burn scar is visible within 1 km of
  the label in 49.5 % of cases.
- Per-patch macro-F1 stays near 25 % across the tested architectures and losses, below the 40.2 % of
  predicting fire everywhere.
- On CaBuAr, the same models reach 62–71 %.
- Degrading CaBuAr's labels toward Sen2Fire's measured properties reproduces 0.77 (95 % interval
  0.55–1.13) of the score gap against an implementation-matched Sen2Fire reference.

## Repository structure

```
sen2fire-label-audit/
├── src/sen2fire_audit/
│   ├── data/          Sen2Fire and CaBuAr datasets, label-degradation arms
│   ├── models/        U-Net, Attention U-Net, DeepLabV3, SegFormer
│   ├── configs/       one YAML file per experiment
│   ├── reproduce/     one module per analysis of the Supplement
│   ├── losses.py  train.py  evaluate.py  experiment.py
├── results/           result tables of every experiment and the census sheets
├── checkpoints/       MANIFEST.csv of the released checkpoints
└── docs/              training, reproduction, result tables, checkpoints
```

## Installation

```bash
git clone https://github.com/sjang1594/sen2fire-label-audit.git
cd sen2fire-label-audit
pip install -e .                 # add [segformer] for the SegFormer runs
```

Python 3.10 or later; PyTorch and torchvision are installed as dependencies.

## Data preparation

The datasets are not redistributed. Download them and arrange them under one data root:

```
data/
├── sen2fire/
│   ├── scene1/ scene2/ scene3/ scene4/      released patches (*.npz)
│   └── train.txt val.txt test.txt           split lists
└── cabuar/
    └── raw/patched/512x512.hdf5, chabud_test.h5
```

- **Sen2Fire** (Xu et al., IGARSS 2024): <https://zenodo.org/records/10881058>. Write the split
  lists (scenes 1–2 train, 3 validation, 4 test) with
  `python -c "from sen2fire_audit.data.sen2fire import write_split_lists; write_split_lists('data/sen2fire')"`.
- **CaBuAr** (Cambrin et al., IEEE GRSM 2023): <https://doi.org/10.1109/MGRS.2023.3292467>.
  Folds 0 / 1 / 2–4 are test / validation / train; the chabud subset is excluded.

## Usage

**Reproduce an analysis of the Supplement** from the released result tables:

```bash
python -m sen2fire_audit.reproduce supplement_tables --data data --results results
```

All commands are listed in [docs/reproduce.md](docs/reproduce.md).

**Train** the runs of a configuration (16 configurations, listed in [docs/training.md](docs/training.md)):

```python
from sen2fire_audit.experiment import run_experiment
run_experiment('cabuar_attunet', data_dir='data/cabuar', out_dir='runs', device='cuda', seeds=[42])
```

**Evaluate a released checkpoint** on the Sen2Fire test split:

```python
import torch
from sen2fire_audit import evaluate as E
from sen2fire_audit.data import sen2fire
from sen2fire_audit.models import DeepLabV3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
test = sen2fire.Sen2FireDataset.from_strategy('data/sen2fire', 'test', 'Vanilla', augment=False)
model = DeepLabV3(sen2fire.STRATEGIES['Vanilla'][2], init=None)
model.load_state_dict(torch.load('sen2fire_deeplab_dice_Vanilla_aug_seed42.pt', map_location='cpu', weights_only=True))
rows = E.score_patches(model.to(device).eval(), test, device)
print(E.pooled_metrics(rows), E.macro_f1(rows, min_cov=0.001))
```

## Trained checkpoints

The 303 checkpoints of the 16 configurations and the 30 single runs behind Tables I–III are on
Zenodo: [10.5281/zenodo.24083276](https://doi.org/10.5281/zenodo.24083276), one archive per
configuration. [`checkpoints/MANIFEST.csv`](checkpoints/MANIFEST.csv) lists every file with its
SHA-256; see [docs/checkpoints.md](docs/checkpoints.md).

## Result tables

`results/` holds the per-run and per-patch results of every experiment, the tables the reproduction
commands read and write, and the census sheets; see [docs/results.md](docs/results.md).

## Citation

```bibtex
@article{jang2026labels,
  title   = {The Labels Are the Ceiling: A Model-Free Audit of {Sen2Fire}},
  author  = {Jang, Seungho and Singh, Rumandeep and Bhatia, Sanjiv K. and Bhaskaran, Sunil},
  journal = {IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.},
  year    = {2026},
  note    = {Under review}
}

@dataset{jang2026checkpoints,
  title     = {Trained checkpoints and result tables for "The Labels Are the Ceiling: A Model-Free Audit of Sen2Fire"},
  author    = {Jang, Seungho and Singh, Rumandeep and Bhatia, Sanjiv K. and Bhaskaran, Sunil},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24083276}
}
```

## License

This repository (code and result tables) is released under the MIT License. The checkpoints on
Zenodo are released under CC BY 4.0, except the SegFormer checkpoints, which are fine-tuned from
NVIDIA's MiT-B2 weights and distributed under the NVIDIA Source Code License for SegFormer
(non-commercial use). The datasets keep their own licenses.

## Acknowledgments

We thank the authors of [Sen2Fire](https://zenodo.org/records/10881058) and
[CaBuAr](https://doi.org/10.1109/MGRS.2023.3292467) for releasing their datasets.
