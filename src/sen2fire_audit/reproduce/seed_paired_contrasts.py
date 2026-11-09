"""Seed-paired t, exact Wilcoxon and Welch intervals of every ten-seed comparison.
Writes seed_paired_contrasts.tex.
Run labels in the output: E7 = sen2fire_attunet_oversample baseline, E13 = sen2fire_attunet_cabuar_impl."""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from . import _paths as P

sys.stdout.reconfigure(encoding='utf-8')

def paired(d):
    d = np.asarray(d, float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    h = stats.t.ppf(0.975, len(d) - 1) * se
    return d.mean(), d.mean() - h, d.mean() + h, d.mean() / se, stats.ttest_1samp(d, 0).pvalue

def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    return a.mean() - b.mean() - h, a.mean() - b.mean() + h

def line(name, x, y):
    d = (x - y).values
    m, lo, hi, t, p = paired(d)
    wl, wh = welch(x, y)
    wx = stats.wilcoxon(d, method='exact').pvalue
    return (f'   {name:16} {m:+.2f} [{lo:+.2f}, {hi:+.2f}] t={t:+.2f} p={p:.2g} Wilcoxon exact p={wx:.2g} '
            f'positive {int((d > 0).sum())}/{len(d)} | Welch [{wl:+.2f}, {wh:+.2f}]')

print('CaBuAr per-patch macro-F1 (pp), AttUNet - DeepLabV3')
c = P.cabuar_architectures('per_patch')
c = c[c.coverage > 0.001]
sm = (c.groupby(['arch', 'seed']).f1.mean().unstack(0) * 100).dropna()
A, D = [x for x in sm.columns if 'Att' in x][0], [x for x in sm.columns if 'Att' not in x][0]
print(line('ten seeds', sm[A], sm[D]))
d = (sm[A] - sm[D]).values
k = int((d > 0).sum())
print(f'   exact two-sided sign test: {stats.binomtest(k, len(d), 0.5).pvalue:.6f}')
o3 = sm.loc[[s for s in (42, 1, 2) if s in sm.index]]
print(line('original 3 seeds', o3[A], o3[D]))
# the six three-seed runs reported first: rows with an empty note (DeepLabV3 seed 1 was retrained later)
r = P.cabuar_architectures('results')
r = r[r.seed.isin([42, 1, 2]) & r.note.isna()].set_index(['arch', 'seed']).macro_f1.unstack(0)
assert r.shape == (3, 2) and np.isclose(r.loc[1, D], 61.47), r
print(line('submitted runs', r[A], r[D]))
ra, rd = r[A].values, r[D].values
print(f'   submitted runs: means {ra.mean():.2f} / {rd.mean():.2f}, sd {ra.std(ddof=1):.2f} / {rd.std(ddof=1):.2f}; '
      f'Welch t = {(ra.mean() - rd.mean()) / np.sqrt(ra.var(ddof=1) / 3 + rd.var(ddof=1) / 3):.2f} '
      f'(first report: 70.9 +/- 1.0, 63.8 +/- 3.2, t ~ 3.7)')

print('\nChannel ablation (pp), arm - full')
e = pd.read_csv(P.run_table('sen2fire_attunet_dropband', 'results.csv'))
for col in ('f1', 'prec', 'rec'):
    w = e.pivot(index='seed', columns='drop', values=col).dropna()
    for arm in [a for a in w.columns if a != 'full']:
        print(line(f'{col} {arm}', w[arm], w['full']))

print('\nOversampling (pp), arm - baseline')
o = pd.read_csv(P.run_table('sen2fire_attunet_oversample', 'results.csv'))
for col in ('f1', 'macro_f1', 'prec', 'rec', 'sup_rec', 'veg_rec'):
    w = o.pivot(index='seed', columns='arm', values=col).dropna()
    for name, x in (('over50', w['over50']), ('over75', w['over75']), ('both', (w['over50'] + w['over75']) / 2)):
        print(line(f'{col} {name}', x, w['baseline']))
    if col == 'f1':
        o3 = w.loc[[s for s in (42, 1, 2) if s in w.index]]
        n7 = w.drop(index=[s for s in (42, 1, 2) if s in w.index])
        print(line('f1 both, orig 3', (o3['over50'] + o3['over75']) / 2, o3['baseline']))
        print(line('f1 both, new 7', (n7['over50'] + n7['over75']) / 2, n7['baseline']))

print('\nSen2Fire Attention U-Net, CaBuAr training implementation (E13) minus original (E7 baseline), '
      'per-patch macro-F1 (pp) on the 99 patches behind S')
from .degradation_gap_matched import s2f_table  # noqa: E402
p13 = pd.read_csv(P.run_table('sen2fire_attunet_cabuar_impl', 'per_patch.csv'))
p7 = pd.read_csv(P.run_table('sen2fire_attunet_oversample', 'per_patch.csv'))
p7 = p7[p7.arm == 'baseline']
m13, m7 = s2f_table(p13).mean() * 100, s2f_table(p7).mean() * 100
m13, m7 = m13.loc[m13.index.intersection(m7.index)], m7.loc[m13.index.intersection(m7.index)]
assert len(m13) == 10
print(line('E13 - E7', m13, m7))

def hedges(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (1 - 3 / (4 * (na + nb - 2) - 1)) * (a.mean() - b.mean()) / sp

def num(v):
    return f'${v:+.2f}$'

def ci(lo, hi):
    return f'[{num(lo)}, {num(hi)}]'

def row(label, metric, x, y):
    d = (x - y).values
    m, lo, hi, _, _ = paired(d)
    wl, wh = welch(x, y)
    wx = stats.wilcoxon(d, method='exact').pvalue
    return f'{label} & {metric} & {num(m)} & {num(hedges(x, y))} & {ci(lo, hi)} & {wx:.3f} & {ci(wl, wh)} \\\\'

rows = [r'\multicolumn{7}{l}{\textit{CaBuAr, Attention U-Net minus DeepLabV3}} \\',
        row('', r'macro-$F_1$', sm[A], sm[D]),
        r'\midrule', r'\multicolumn{7}{l}{\textit{Channel ablation, configuration minus all bands}} \\']
names = {'dropB12': 'Without B12', 'dropB1': 'Without B1', 'dropB9': 'Without B9', 'dropB1B9': 'Without B1 and B9'}
for arm in ('dropB12', 'dropB1', 'dropB9', 'dropB1B9'):
    for col, mname in (('f1', r'pixel $F_1$'), ('prec', 'precision'), ('rec', 'recall')):
        if arm != 'dropB12' and col != 'f1':
            continue
        w = e.pivot(index='seed', columns='drop', values=col).dropna()
        rows.append(row(names[arm] if col == 'f1' else '', mname, w[arm], w['full']))
rows += [r'\midrule', r'\multicolumn{7}{l}{\textit{Oversampling, setting minus baseline}} \\']
mnames = {'f1': r'pixel $F_1$', 'macro_f1': r'macro-$F_1$', 'prec': 'precision', 'rec': 'recall',
          'sup_rec': 'recall, burn evidence', 'veg_rec': 'recall, vegetation'}
for name, key in ((r'50\,\% of draws', 'over50'), (r'75\,\% of draws', 'over75'), ('Average of the two', 'avg')):
    for i, (col, mname) in enumerate(mnames.items()):
        w = o.pivot(index='seed', columns='arm', values=col).dropna()
        x = (w['over50'] + w['over75']) / 2 if key == 'avg' else w[key]
        rows.append(row(name if i == 0 else '', mname, x, w['baseline']))
rows += [r'\midrule', r'\multicolumn{7}{l}{\textit{Sen2Fire Attention U-Net, training implementation (CaBuAr minus original)}} \\',
         row('', r'macro-$F_1$', m13, m7)]
out = P.out('seed_paired_contrasts.tex')
open(out, 'w', encoding='utf-8').write('\n'.join(rows) + '\n')
print(f'\nwrote {out} ({len(rows)} lines)')
