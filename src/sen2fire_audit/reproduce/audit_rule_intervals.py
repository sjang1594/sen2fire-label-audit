"""Patch and 4 x 4-block bootstrap of the pooled spectral-rule rates of the labelled pixels."""
import sys

import numpy as np
import pandas as pd

from . import _paths as P

sys.stdout.reconfigure(encoding='utf-8')
d = pd.read_csv(P.analysis('audit_rule_rates.csv'))
assert len(d) == 99
d['n'] = d.fire_pct / 100 * 512 * 512
rc = d.path.str.extract(r'patch_(\d+)_(\d+)').astype(int)
d['block'] = (rc[0] // 4).astype(str) + '_' + (rc[1] // 4).astype(str)


def pooled(s, col):
    return (s[col] * s.n).sum() / s.n.sum()


rng = np.random.default_rng(0)
blocks = d.block.unique()
for col, name in (('support_frac', 'burn evidence (NBR < 0)'), ('veg_unsup_frac', 'healthy vegetation')):
    pt = pooled(d, col)
    bb = [pooled(pd.concat([d[d.block == b] for b in rng.choice(blocks, len(blocks))]), col) for _ in range(5000)]
    pb = [pooled(d.iloc[rng.choice(len(d), len(d))], col) for _ in range(5000)]
    print(f'{name:24} pooled {pt:.1f}%  patch bootstrap [{np.percentile(pb, 2.5):.1f}, {np.percentile(pb, 97.5):.1f}]  '
          f'block bootstrap [{np.percentile(bb, 2.5):.1f}, {np.percentile(bb, 97.5):.1f}]  ({len(blocks)} blocks)')
