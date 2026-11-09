"""Paired block bootstrap of the non-fire minus labelled-fire burn-evidence rate, Sen2Fire."""
import sys

import numpy as np
import pandas as pd

from . import _paths as P

sys.stdout.reconfigure(encoding='utf-8')
d = pd.read_csv(P.analysis('audit_rule_validation_counts_sen2fire.csv'))
d = d[~d.clear_only & (d.n > 0)].copy()
rc = d.patch.str.extract(r'patch_(\d+)_(\d+)').astype(int)
d['block'] = (rc[0] // 4).astype(str) + '_' + (rc[1] // 4).astype(str)

for st in ('nonfire_le1km', 'nonfire_1to3km', 'nonfire_gt3km', 'nonfire_firefree_patch'):
    g = d[d.stratum.isin(['fire_label', st])]
    blocks = g.block.unique()
    A = g[g.stratum == 'fire_label'].groupby('block')[['sup_noB12', 'n']].sum().reindex(blocks, fill_value=0)
    B = g[g.stratum == st].groupby('block')[['sup_noB12', 'n']].sum().reindex(blocks, fill_value=0)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(blocks), size=(5000, len(blocks)))
    a = A.sup_noB12.to_numpy()[idx].sum(1) / A.n.to_numpy()[idx].sum(1)
    b = B.sup_noB12.to_numpy()[idx].sum(1) / B.n.to_numpy()[idx].sum(1)
    diff = (b - a) * 100
    pt = (B.sup_noB12.sum() / B.n.sum() - A.sup_noB12.sum() / A.n.sum()) * 100
    print(f'{st:24} diff {pt:+.1f} pp  block 95% [{np.nanpercentile(diff, 2.5):+.1f}, {np.nanpercentile(diff, 97.5):+.1f}]'
          f'  P(diff<=0) {np.nanmean(diff <= 0):.3f}  blocks {len(blocks)}')