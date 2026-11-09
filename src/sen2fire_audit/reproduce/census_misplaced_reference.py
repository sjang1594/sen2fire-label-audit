"""Misplaced-label reference for the active-fire overlap, and its contrast with the observed share."""
import sys

import numpy as np
import pandas as pd

from . import _paths as P
from . import census_prevalence as cr

sys.stdout.reconfigure(encoding='utf-8')
rng = np.random.default_rng(0)
a = pd.read_csv(P.census('first_author.csv'), dtype=str).fillna('')
a = a.merge(pd.read_csv(P.census('key.csv')), on='id')
a = a.merge(cr.rule_features(a.idx.tolist()), on='idx')
s = a[a.active_fire != 'none'].copy()
s['human_in'] = s.active_fire.isin(['inside', 'both']).astype(float)
s['rule_in'] = (s.hot_in >= cr.HOT_MIN_PX).astype(float)
rc = s.patch.str.extract(r'patch_(\d+)_(\d+)').astype(int)
s['block'] = (rc[0] // 4).astype(str) + '_' + (rc[1] // 4).astype(str)
print(f'n = {len(s)} patches with active fire; with >= {cr.HOT_MIN_PX} rule hot pixels: {(s.hot_all >= cr.HOT_MIN_PX).sum()}; '
      f'blocks: {s.block.nunique()}')
print(f'reference (misplaced label) {s.null_hit.mean() * 100:.1f}%; human inside {s.human_in.mean() * 100:.1f}%; '
      f'rule inside true label {s.rule_in.mean() * 100:.1f}%')

# above: rule as first specified; below: adopted rule (+ B12 > B11)
lines = [l.strip() for l in open(cr.DATA / 'test.txt') if l.strip()]
adopted = []

for i in s.idx:
    d = np.load(cr.DATA / lines[i].replace('\\', cr.os.sep))
    img = d['image'].astype(np.float32) / 10000.0
    gt = d['label'] == 1
    hot = (img[11] > 0.5) & (img[11] > 1.5 * img[7]) & (img[11] > img[10])
    adopted.append((np.mean([(hot & t(gt)).sum() >= cr.HOT_MIN_PX for t in cr.NULL_TRANSFORMS]),
                    float((hot & gt).sum() >= cr.HOT_MIN_PX), int(hot.sum() >= cr.HOT_MIN_PX)))

s['null_adopted'], s['rule_in_adopted'], s['any_adopted'] = zip(*adopted)
print(f'adopted rule: patches with >= {cr.HOT_MIN_PX} hot pixels {int(s.any_adopted.sum())}; reference '
      f'{s.null_adopted.mean() * 100:.1f}%; rule inside true label {s.rule_in_adopted.mean() * 100:.1f}%')

blocks = s.block.unique()
idx = {b: np.where(s.block.values == b)[0] for b in blocks}

for name, col, ref in (('human inside - reference', 'human_in', 'null_hit'), ('rule inside - reference', 'rule_in', 'null_hit'), ('human inside - adopted-rule reference', 'human_in', 'null_adopted')):
    d = (s[col] - s[ref]).values * 100
    pb = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    bb = [d[np.concatenate([idx[b] for b in rng.choice(blocks, len(blocks))])].mean() for _ in range(10000)]
    print(f'{name}: {d.mean():+.1f} pp; patch bootstrap [{np.percentile(pb, 2.5):+.1f}, {np.percentile(pb, 97.5):+.1f}]; '
          f'block bootstrap [{np.percentile(bb, 2.5):+.1f}, {np.percentile(bb, 97.5):+.1f}]')