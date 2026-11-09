"""Active-fire (hot-pixel) rule variants against the first author's census answers.
Writes active_fire_rule_sensitivity.csv (default sheet only)."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import _paths as P
from . import census_prevalence as cr

VARIANTS = [(t, r, b11) for b11 in (False, True) for t in (0.3, 0.4, 0.5, 0.6, 0.8) for r in (1.5, 2.0)]
NULLS = cr.NULL_TRANSFORMS

raw = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else P.census('first_author.csv'), dtype=str).fillna('')
raw.loc[raw.active_fire == '', 'active_fire'] = 'none'     # S073, confirmed by the author
census = raw.merge(pd.read_csv(P.census('key.csv')), on='id').set_index('idx')

lines = [l.strip() for l in open(cr.DATA / 'test.txt') if l.strip()]
acc = {v: dict(hot=0, hot_in=0, null_in=0.0, lab=0, rows=[]) for v in VARIANTS}
for idx, rel in enumerate(lines):
    d = np.load(cr.DATA / rel.replace('\\', os.sep))
    gt = d['label'] == 1
    if gt.mean() * 100 <= 0.1:
        continue
    img = d['image'].astype(np.float32) / 10000.0
    b8, b11, b12 = img[7], img[10], img[11]
    for v in VARIANTS:
        t, r, use_b11 = v
        hot = (b12 > t) & (b12 > r * b8)
        if use_b11:
            hot &= b12 > b11
        a = acc[v]
        a['hot'] += int(hot.sum()); a['hot_in'] += int((hot & gt).sum()); a['lab'] += int(gt.sum())
        a['null_in'] += np.mean([(hot & f(gt)).sum() for f in NULLS])
        if idx in census.index:
            a['rows'].append((idx, int((hot & gt).sum()), int(hot.sum())))

out = []
for v, a in acc.items():
    t, r, use_b11 = v
    m = census.join(pd.DataFrame(a['rows'], columns=['idx', 'hot_in', 'hot_all']).set_index('idx'), how='inner')
    row = {'B12>': t, 'ratio': r, 'B12>B11': use_b11, 'hot_px': a['hot'],
           'share_in_label': 100 * a['hot_in'] / max(a['hot'], 1),
           'share_in_misplaced': 100 * a['null_in'] / max(a['hot'], 1),
           'pct_of_label_area': 100 * a['hot_in'] / a['lab']}
    for name, h, rr in (('in', m.active_fire.isin(['inside', 'both']), m.hot_in >= cr.HOT_MIN_PX),
                        ('any', m.active_fire != 'none', m.hot_all >= cr.HOT_MIN_PX)):
        tp = int((h & rr).sum()); fp = int((~h & rr).sum()); fn = int((h & ~rr).sum())
        row[f'kappa_{name}'] = cr.kappa(h.map({True: 'y', False: 'n'}), rr.map({True: 'y', False: 'n'}), ['y', 'n'])
        row[f'prec_{name}'] = tp / max(tp + fp, 1); row[f'rec_{name}'] = tp / max(tp + fn, 1)
        row[f'fp_{name}'] = fp
    out.append(row)
res = pd.DataFrame(out)
pd.set_option('display.width', 200)
print(res.round(3).to_string(index=False))
if len(sys.argv) == 1:
    res.to_csv(P.out('active_fire_rule_sensitivity.csv'), index=False)
