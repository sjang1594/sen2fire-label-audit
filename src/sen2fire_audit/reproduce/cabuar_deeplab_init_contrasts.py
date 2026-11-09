"""CaBuAr DeepLabV3 initialization arms against each other and the stored Attention U-Net.
Run labels in the output: E10 = cabuar_attunet / cabuar_deeplab."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import _paths as P

def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    d = a.mean() - b.mean()
    return d, d - h, d + h, stats.ttest_ind(a, b, equal_var=False).pvalue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablation', type=Path, default=P.run_table('cabuar_deeplab_init', ''))
    args = ap.parse_args()

    abl = pd.read_csv(args.ablation / 'results.csv').drop_duplicates(['init', 'seed'], keep='last')
    e10 = P.cabuar_architectures('results').drop_duplicates(['arch', 'seed'], keep='last')
    runs = pd.concat([abl.assign(arm='DLv3 ' + abl.init),
                      e10[e10.arch == 'AttUNet'].assign(arm='AttUNet (E10)'),
                      e10[e10.arch == 'DeepLabV3'].assign(arm='DLv3 coco (E10, stored)')])
    order = ['AttUNet (E10)', 'DLv3 coco (E10, stored)', 'DLv3 coco', 'DLv3 imagenet', 'DLv3 scratch']

    print('seed level (%): mean +/- sd over seeds')
    print(f'{"arm":26} {"n":>3} {"macro-F1":>13} {"pooled F1":>13} {"prec":>6} {"rec":>6} {"stop ep":>7}')
    for a in order:
        s = runs[runs.arm == a]
        stop = pd.to_numeric(s.stop_epoch, errors='coerce').mean()
        print(f'{a:26} {len(s):3} {s.macro_f1.mean():6.1f} +/- {s.macro_f1.std():3.1f} {s.f1.mean():6.1f} +/- {s.f1.std():3.1f}'
              f' {s.prec.mean():6.1f} {s.rec.mean():6.1f} {stop:7.0f}')

    print('\nseed-level macro-F1 differences (Welch 95 % interval, p)')
    pairs = [('DLv3 imagenet', 'DLv3 coco'), ('DLv3 scratch', 'DLv3 coco'), ('DLv3 scratch', 'DLv3 imagenet'),
             ('AttUNet (E10)', 'DLv3 coco'), ('AttUNet (E10)', 'DLv3 imagenet'), ('AttUNet (E10)', 'DLv3 scratch'),
             ('DLv3 coco', 'DLv3 coco (E10, stored)')]
    for x, y in pairs:
        d, lo, hi, p = welch(runs[runs.arm == x].macro_f1, runs[runs.arm == y].macro_f1)
        print(f'  {x:15} - {y:24} {d:+5.1f}  [{lo:+5.1f}, {hi:+5.1f}]  p = {p:.3g}')
    for x, y in pairs[:3]:
        d, lo, hi, p = welch(runs[runs.arm == x].f1, runs[runs.arm == y].f1)
        print(f'  pooled F1: {x:15} - {y:13} {d:+5.1f}  [{lo:+5.1f}, {hi:+5.1f}]  p = {p:.3g}')

    pp = pd.read_csv(args.ablation / 'per_patch.csv').drop_duplicates(['init', 'seed', 'patch_idx'], keep='last')
    pe = P.cabuar_architectures('per_patch').drop_duplicates(['arch', 'seed', 'patch_idx'], keep='last')
    pe = pe[pe.arch == 'AttUNet'].rename(columns={'arch': 'init'}).assign(init='AttUNet')
    m = pd.concat([pp, pe]).groupby(['init', 'patch_idx']).f1.mean().unstack(0)
    groups = pp.drop_duplicates('patch_idx').set_index('patch_idx').group
    assert len(m) == 115 and not m.isna().any().any(), (len(m), m.isna().sum())
    fire = groups.loc[m.index].str.rsplit('_', n=1).str[0].values
    fires = np.unique(fire)
    idx = {f: np.where(fire == f)[0] for f in fires}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([idx[f] for f in rng.choice(fires, len(fires))]) for _ in range(10000)]
    print(f'\npatch level: seed-mean per-patch F1 on {len(m)} fire patches from {len(fires)} fires; '
          f'95 % interval resampling fires')
    for x, y in [('imagenet', 'coco'), ('scratch', 'coco'), ('scratch', 'imagenet'),
                 ('AttUNet', 'coco'), ('AttUNet', 'imagenet'), ('AttUNet', 'scratch')]:
        d = ((m[x] - m[y]) * 100).values
        bs = [d[s].mean() for s in draws]
        print(f'  {x:8} - {y:8} {d.mean():+5.1f}  [{np.percentile(bs, 2.5):+5.1f}, {np.percentile(bs, 97.5):+5.1f}]'
              f'  {x} better on {(d > 0).sum()}/{len(d)} patches')

if __name__ == '__main__':
    main()
