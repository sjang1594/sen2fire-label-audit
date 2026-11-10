# Reproduction commands

```bash
python -m sen2fire_audit.reproduce <name> --data data --results results [--out DIR]
```

Outputs go to `results/analysis` unless `--out` is given. Random draws follow the original scripts, so
bootstrap intervals match the paper to the printed digit.

| Command | Reproduces | Reads images |
|---|---|:-:|
| `supplement_tables` | Census and post hoc tables of the Supplement | ✓ |
| `census_misplaced_reference` | Misplaced-label reference for the active-fire overlap | ✓ |
| `active_fire_rule_sensitivity` | Active-fire rule variants against the census | ✓ |
| `audit_rule_intervals` | Block bootstrap of the labeled-pixel rule rates | |
| `audit_strata_intervals` | Block bootstrap of every companion-audit stratum | |
| `audit_strata_difference` | Non-fire minus labeled-fire burn-evidence rate | |
| `block_size_sensitivity` | Block-size sensitivity of the spatial bootstrap | |
| `seed_paired_contrasts` | Seed-paired and Welch analysis of the ten-seed comparisons | |
| `label_evidence_coupling` | Label-to-evidence coupling, CaBuAr vs Sen2Fire | ✓ |
| `sen2fire_deeplab_init_by_census_location` | Sen2Fire DeepLabV3 initialization table | |
| `cabuar_deeplab_init_contrasts` | CaBuAr DeepLabV3 initialization table | |
| `census_prevalence` | Census rates (first author) | ✓ |
| `census_agreement` | Agreement of the three annotators | ✓ |
| `degradation_oracle_ceiling` | Degradation arms and their oracle ceiling | ✓ |
| `degradation_gap_share` | Gap share G of the degradation arms | ✓ |
| `degradation_gap_matched` | Gap share with the matched Sen2Fire reference | |

Run `active_fire_rule_sensitivity` and `audit_strata_intervals` before `supplement_tables`, which reads
their outputs (about ten minutes on a CPU).
