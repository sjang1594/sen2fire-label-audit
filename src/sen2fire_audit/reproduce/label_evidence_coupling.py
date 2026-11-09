"""Share of label pixels within 1.5 km of image evidence: CaBuAr degraded labels vs Sen2Fire labels."""
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

from . import _paths as P
from . import degradation_oracle_ceiling as do

RADIUS_KM = 1.5
S2F = P.sen2fire()

def near_frac(label, evidence, px_m):
    d = distance_transform_edt(~evidence) * px_m / 1000.0
    return float((d[label] <= RADIUS_KM).mean())

def boot_ci(x, rng, n=4000):
    x = np.asarray(x)
    m = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])

def main():
    rng = np.random.default_rng(0)
    spec = do.arm_spec()
    res = {}
    for split in ('train', 'test'):
        ks = [k for k in spec['keys'] if k in spec['fire'] and do.SPLIT[spec['fold'][k]] == split]
        for name, lab in (('inflate_area', spec['inflate']), ('resolution', spec['one_km'])):
            f = [near_frac(lab[k], spec['burn'][k], 20.0) for k in ks]
            sh = spearmanr([lab[k].mean() for k in ks], [spec['burn'][k].mean() for k in ks])[0]
            res[('CaBuAr', name, split)] = (np.array(f), sh)

    for split in ('train', 'test'):
        f, ls, es = [], [], []
        for rel in [l.strip() for l in open(S2F / f'{split}.txt') if l.strip()]:
            d = np.load(S2F / rel.replace('\\', os.sep))
            lab = d['label'] == 1
            if lab.mean() <= do.MIN_COV:
                continue
            x = d['image'].astype(np.float32) / 10000.0
            b8, b11, b12 = x[7], x[10], x[11]
            hot = (b12 > 0.5) & (b12 > 1.5 * b8) & (b12 > b11)
            if hot.sum() < 20:
                continue
            f.append(near_frac(lab, hot, 10.0)); ls.append(lab.mean()); es.append(hot.mean())
        res[('Sen2Fire', 'as released', split)] = (np.array(f), spearmanr(ls, es)[0])

    print(f'Fraction of label pixels within {RADIUS_KM} km of evidence (fire patches with evidence)')
    print(f'{"dataset":9}{"label":14}{"split":7}{"n":>5}{"mean":>8}   95% CI       Spearman(share)')
    for (ds, name, split), (f, sh) in res.items():
        lo, hi = boot_ci(f, rng)
        print(f'{ds:9}{name:14}{split:7}{len(f):5}{f.mean():8.3f}   [{lo:.3f}, {hi:.3f}]   {sh:.2f}')

    for split in ('train', 'test'):
        a, b = res[('CaBuAr', 'inflate_area', split)][0], res[('Sen2Fire', 'as released', split)][0]
        diff = rng.choice(a, (4000, len(a))).mean(1) - rng.choice(b, (4000, len(b))).mean(1)
        lo, hi = np.percentile(diff, [2.5, 97.5])
        d = a.mean() - b.mean()
        v = ('inflation decouples more than Sen2Fire' if d <= -0.2 and hi < 0 else
             'CaBuAr more coupled (inconclusive: Sen2Fire under-estimated)' if lo > 0 else 'no clear difference')
        print(f'{split}: CaBuAr inflate_area - Sen2Fire = {d:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {v}')

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
