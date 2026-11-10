# Result tables

| Path | Contents |
|---|---|
| `results/<config>/results*.csv` | Test scores per run; `_{noaug,aug}_ch{32,64}` splits augmentation and width |
| `results/<config>/per_patch.csv`, `per_patch/` | Test F1 per patch |
| `results/<config>/threshold_*.csv` | Decision-threshold sweep |
| `results/<config>/macro_f1.csv` | Per-patch macro-F1 of the DeepLabV3 runs |
| `results/analysis/` | Outputs of the reproduction commands |
| `results/census/` | Census key (`key.csv`) and answers: first author (99 patches), co-annotators A and B (25) |

**Macro-F1** averages patches with more than 0.1 % labeled fire (99 Sen2Fire, 115 CaBuAr). All 102
Sen2Fire fire patches are used in the fire-only table, the oversampling rows of the seed-paired table, and
the `macro_f1` column of `sen2fire_attunet_cabuar_impl/results.csv` (the paper's 32.5 % uses 99).

**Experiments `E1` and `E2`** in `sen2fire_unet_wce/results_noaug_ch32.csv` are two independent runs of
the same SWIR configuration; the paper reports both.

**Census notes**: `first_author.csv` records the two entries completed after the first pass (S073, S079).
