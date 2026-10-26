# sen2fire-label-audit

Code, result tables and checkpoint manifest for

> S. Jang, R. Singh, S. K. Bhatia, S. Bhaskaran,
> *The Labels Are the Ceiling: A Model-Free Audit of Sen2Fire* (under review).

The repository contains the label-evidence audit of the Sen2Fire benchmark, the
per-patch macro-F1 evaluation, the training notebooks for every reported
experiment, and the result tables behind every table and figure of the paper and
its supplement.

## Archives

| Item | Where | DOI |
|---|---|---|
| src (this repository, release `v1.0`) | GitHub, archived on Zenodo | TBD |
| Trained checkpoints and census panel images | Zenodo | TBD |

## Layout

| Path | Contents |
|---|---|
| `src/` | All scripts and notebooks, kept flat as they were run: several scripts import one another and resolve paths relative to their own folder |
| `src/label_audit/` | Result tables the scripts read and write (census panel images are on Zenodo) |
| `checkpoints/MANIFEST.csv` | Manifest of the released checkpoints (the files themselves are on Zenodo) |

The section *Reproducing the paper* below maps each table and figure to the
script that produces it and the result file it writes.

## Data

The datasets are not redistributed here.

- **Sen2Fire**: Xu et al., 2024, <https://doi.org/10.5281/zenodo.10881058>
- **CaBuAr**: TBD

## Reproducing the paper

TBD: one entry per table and figure, giving the script or notebook and the
result file it produces.

## License

Code: MIT (see `LICENSE`). Checkpoints: see the Zenodo record.
