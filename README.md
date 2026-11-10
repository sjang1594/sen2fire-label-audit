# sen2fire-label-audit

Code and result tables for

> S. Jang, R. Singh, S. K. Bhatia, S. Bhaskaran,
> *The Labels Are the Ceiling: A Model-Free Audit of Sen2Fire* (under review).

The repository holds the training code and configurations of every reported experiment, the result
tables behind the paper and its Supplement, and one command per reported analysis. The commands keep
the computation and the order of random draws of the scripts that produced the published numbers, so
bootstrap intervals reproduce to the printed digit.

## Archives

| Item | Where | DOI |
|---|---|---|
| Code and result tables (this repository, release `v1.0.0`) | GitHub, archived on Zenodo | assigned on release |
| Trained checkpoints of every reported configuration, the result tables and a copy of the `v1.0.0` code | Zenodo | [10.5281/zenodo.24083276](https://doi.org/10.5281/zenodo.24083276) |

## Layout

| Path | Contents |
|---|---|
| `src/sen2fire_audit/` | Python package: data, models, losses, training, evaluation |
| `src/sen2fire_audit/configs/` | One YAML file per experiment (see *Training*) |
| `src/sen2fire_audit/reproduce/` | One module per reported analysis (see *Reproducing the Supplement*) |
| `results/census/` | Census key and response sheets |
| `results/analysis/` | Tables written by the analysis commands, including the Supplement table rows |
| `results/<config>/` | Training results of one configuration: `results.csv` per run, `per_patch.csv` per test patch |
| `checkpoints/MANIFEST.csv` | Every released checkpoint: archive, file, config, run, size, SHA-256 |

## Installation

```bash
git clone https://github.com/sjang1594/sen2fire-label-audit.git
cd sen2fire-label-audit
pip install -e .                 # add [segformer] for the SegFormer runs
```

Python 3.10 or later.

## Data

The datasets are not redistributed here. The commands expect this layout under one data root:

```
data/
  sen2fire/
    scene1/ scene2/ scene3/ scene4/      # the released Sen2Fire patches (*.npz)
    train.txt val.txt test.txt           # split lists, one scene*/patch path per line
  cabuar/
    raw/patched/512x512.hdf5
    raw/patched/chabud_test.h5
```

- **Sen2Fire**: Xu et al., IGARSS 2024, <https://doi.org/10.1109/igarss53475.2024.10641441>;
  data at <https://zenodo.org/records/10881058>. The split lists (scenes 1-2 train, 3 validation,
  4 test, sorted by file name) are written by `sen2fire_audit.data.sen2fire.write_split_lists(data_dir)`.
- **CaBuAr**: Cambrin et al., IEEE GRSM 11(3), 2023, <https://doi.org/10.1109/MGRS.2023.3292467>.
  Folds: test = 0, validation = 1, train = 2-4; the chabud subset is excluded.

## Reproducing the Supplement

```bash
python -m sen2fire_audit.reproduce <name> --data data --results results [--out DIR]
```

`--out` defaults to `results/analysis`, so a run regenerates the released tables in place. Commands
marked *data* also read the Sen2Fire (and, for the degradation commands, CaBuAr) images; the others
read only `results/`.

| Command | Reproduces | Needs |
|---|---|---|
| `supplement_tables` | Census tables (agreement, rule, uncertainty, sensitivity, item-level responses) and the post hoc tables (empty-label negatives, fire-only training, threshold sweep, pre-fire probe) | data |
| `census_misplaced_reference` | Misplaced-label reference for the active-fire overlap | data |
| `active_fire_rule_sensitivity` | Active-fire rule variants against the census | data |
| `audit_rule_intervals` | Block bootstrap of the labeled-pixel spectral-rule rates | |
| `audit_strata_intervals` | Block bootstrap of every stratum of the companion audit | |
| `audit_strata_difference` | Non-fire minus labeled-fire burn-evidence rate | |
| `block_size_sensitivity` | Sensitivity of the spatial block bootstrap to the block size | |
| `seed_paired_contrasts` | Seed-paired and Welch analysis of the ten-seed comparisons | |
| `label_evidence_coupling` | Coupling of labels to image evidence, CaBuAr vs Sen2Fire | data |
| `sen2fire_deeplab_init_by_census_location` | Sen2Fire DeepLabV3 initialization table | |
| `cabuar_deeplab_init_contrasts` | CaBuAr DeepLabV3 initialization table | |

Shared analyses, also runnable on their own:

| Command | Computes | Needs |
|---|---|---|
| `census_prevalence` | Census rates from the first author's blind answers | data |
| `census_agreement` | Agreement of the three annotators on the 25 shared patches | data |
| `degradation_oracle_ceiling` | Label-degradation arms and their training-free oracle ceiling | data |
| `degradation_gap_share` | Gap share G and learning ceiling of the degradation arms | data |
| `degradation_gap_matched` | Gap share with the matched Sen2Fire reference | |

`supplement_tables` reads the outputs of `active_fire_rule_sensitivity` and `audit_strata_intervals`;
run those first when regenerating everything. On a CPU, `supplement_tables` takes about ten minutes;
the others take seconds to two minutes.

## Training

Each configuration expands to its runs (strategies, augmentation, width, arm and seed crossed).
A run's results go to `<out_dir>/<config>/`; runs already in its `results.csv` are skipped, and an
existing valid checkpoint is evaluated, never trained further.

```python
from sen2fire_audit.experiment import run_experiment
run_experiment('sen2fire_deeplab_init', data_dir='data/sen2fire', out_dir='runs', device='cuda')
run_experiment('cabuar_attunet', data_dir='data/cabuar', out_dir='runs', device='cuda', seeds=[42])
```

| Config | Runs | Model | Loss | Batch | Epoch cap |
|---|---:|---|---|---:|---:|
| `sen2fire_attunet_dice` | 24 | Attention U-Net | Dice (batch, both classes) | 8 | 100 |
| `sen2fire_attunet_combo` | 39 | Attention U-Net | weighted CE + Dice | 8 | 50 |
| `sen2fire_unet_wce` | 27 | U-Net | weighted CE | 8 | 50 |
| `sen2fire_deeplab_dice` | 9 | DeepLabV3 (COCO) | Dice (batch, both classes) | 4 | 100 |
| `sen2fire_deeplab_combo` | 3 | DeepLabV3 (COCO) | weighted CE + Dice | 4 | 50 |
| `sen2fire_segformer_dice` | 9 | SegFormer MiT-B2 | Dice (batch, both classes) | 2 | 100 |
| `sen2fire_attunet_dropband` | 50 | Attention U-Net, bands removed | Dice (batch, both classes) | 8 | 100 |
| `sen2fire_attunet_oversample` | 30 | Attention U-Net, fire-heavy oversampling | Dice (batch, both classes) | 8 | 100 |
| `sen2fire_attunet_fireonly` | 8 | Attention U-Net, no empty-label patches | Dice (batch, both classes) | 8 | 100 |
| `sen2fire_attunet_cabuar_impl` | 10 | Attention U-Net, CaBuAr implementation | Dice (per image, fire) | 4 | 100 |
| `sen2fire_deeplab_init` | 9 | DeepLabV3: COCO / ImageNet backbone / scratch | Dice (batch, both classes) | 4 | 100 |
| `cabuar_attunet` | 10 | Attention U-Net | Dice (per image, fire) | 4 | 100 |
| `cabuar_deeplab` | 10 | DeepLabV3 (COCO) | Dice (per image, fire) | 4 | 100 |
| `cabuar_deeplab_init` | 30 | DeepLabV3: COCO / ImageNet backbone / scratch | Dice (per image, fire) | 4 | 100 |
| `cabuar_degrade` | 25 | Attention U-Net, degraded labels | Dice (per image, fire) | 4 | 100 |
| `cabuar_degrade_negatives` | 10 | Attention U-Net, degraded labels + empty-label patches | Dice (per image, fire) | 4 | 100 |

All runs use Adam (learning rate 1e-4; pretrained backbones 1e-5) and early stopping on the
validation loss with patience 10. The two Dice definitions are kept separate on purpose: the
Sen2Fire and CaBuAr experiments used different ones. Each config records the experiment label of
the original notebooks in `legacy_id`, which is also how some analysis outputs name the runs.

Per-patch macro-F1 is the mean fire-class F1 over test patches whose label covers more than 0.1 %
(99 Sen2Fire and 115 CaBuAr patches). Two Supplement results use all 102 Sen2Fire patches with any
labeled fire: the fire-only table and the oversampling macro-F1 rows of the seed-paired table. The
`macro_f1` column of `results/sen2fire_attunet_cabuar_impl/results.csv` also averages all 102 patches;
the value in the paper (32.5) is the 99-patch mean computed from that configuration's `per_patch.csv`.

## Census sheets

`results/census/` holds the blind key (`key.csv`), the first author's answers on the 99 patches
(`first_author.csv`), the two co-annotators' answers on the 25 shared patches (`annotator_a.csv`,
`annotator_b.csv`, as Co-annotator A and B in the Supplement), and the three-way location of each
patch (`location.csv`). The notes of `first_author.csv` record the two entries completed after the
first pass (S073, S079).

## Checkpoints

The 303 checkpoints of the configurations above and the 30 single runs behind Tables I-III are on
Zenodo, one archive per configuration (`<config>.zip`, files named `<run_id>.pt`) and
`reference_single_run.zip`. `checkpoints/MANIFEST.csv` lists each file with its size and SHA-256.
Each file is a `state_dict`; every configuration checkpoint loads with `strict=True` into the model
of this package (build DeepLabV3 with `init=None` to skip the pretrained download):

```python
import torch
from sen2fire_audit.models import AttentionUNet
model = AttentionUNet(12, base_ch=32)
model.load_state_dict(torch.load('cabuar_attunet_aug_ch32_seed42.pt', map_location='cpu', weights_only=True))
```

The Table I-III runs were saved by the original single-run notebook, whose layer names differ from
this package; they are released as recorded.

## Not yet in this repository

- Commands for the main-text tables and figures, and for the input tables in `results/analysis/`
  that the commands above read (`audit_rule_rates`, `audit_rule_validation_counts_*`,
  `score_by_census_location*`, `prefire_swap_filter`, `cabuar_degrade_threshold_sweep`,
  `cabuar_prefire_probe*`) and for `results/census/location.csv`.
- Colab notebooks for training and checkpoint evaluation.
- The notebook that produced Tables I-III.

## License

Code: MIT (see `LICENSE`). Checkpoints: see the Zenodo record. The datasets keep their own licenses.
