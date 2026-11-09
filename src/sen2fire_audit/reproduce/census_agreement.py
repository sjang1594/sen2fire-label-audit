"""Agreement of the first author and co-annotators A and B on the 25 shared census patches, and with
the adopted active-fire rule."""
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import _paths as P
from . import census_prevalence as cr

HUMANS = ('author', 'annotator_a', 'annotator_b')
SHEETS = {
    'author': P.census('first_author.csv'),
    'annotator_a': P.census('annotator_a.csv'),
    'annotator_b': P.census('annotator_b.csv'),
}
ITEMS = [('active_fire', sorted(cr.ALLOWED['active_fire']), None),
         ('fire_in_label', cr.FIRE_ORDER, 'linear'),
         ('scar_outside', sorted(cr.ALLOWED['scar_outside']), None),
         ('label_verdict', sorted(cr.ALLOWED['label_verdict']), None),
         ('vgroup', sorted(set(cr.VERDICT_GROUP.values())), None),
         ('loc3', ['inside', 'near', 'none'], None),
         ('label_main', sorted(cr.ALLOWED['label_main']), None),
         ('green_main', ['y', 'n'], None)]


def load(path, ids):
    s = pd.read_csv(path, dtype=str).fillna('')
    s = s[s.id.isin(ids)].set_index('id').loc[ids]
    for col in cr.ALLOWED:
        s[col] = s[col].str.strip().str.lower()
    s.loc[(s.assessable == 'yes') & (s.active_fire == ''), 'active_fire'] = 'none'
    s['loc3'] = cr.location3(s)
    s['vgroup'] = s.label_verdict.map(cr.VERDICT_GROUP).fillna('')
    s['fire_in'] = np.where(s.active_fire.isin(['inside', 'both']), 'y', 'n')
    s['fire_any'] = np.where(s.active_fire != 'none', 'y', 'n')
    s['green_main'] = np.where(s.label_main.isin(['dense_veg', 'open_veg']), 'y', 'n')
    s['little_evidence'] = np.where(s.fire_in_label.isin(['none', 'lt10']), 'y', 'n')
    return s


def rule_presence(ids):
    key = pd.read_csv(P.census('key.csv')).set_index('id').loc[ids]
    lines = [l.strip() for l in open(cr.DATA / 'test.txt') if l.strip()]
    rows = {}
    for pid, idx in key.idx.items():
        d = np.load(cr.DATA / lines[idx].replace('\\', os.sep))
        img = d['image'].astype(np.float32) / 10000.0
        gt = d['label'] == 1
        b8, b11, b12 = img[7], img[10], img[11]
        hot = (b12 > 0.5) & (b12 > 1.5 * b8) & (b12 > b11)
        rows[pid] = dict(label_px=int(gt.sum()),
                         fire_in='y' if (hot & gt).sum() >= cr.HOT_MIN_PX else 'n',
                         fire_any='y' if hot.sum() >= cr.HOT_MIN_PX else 'n')
    return pd.DataFrame(rows).T


def fleiss(sheets, col, labs):
    m = np.ones(len(next(iter(sheets.values()))), bool)
    for k in HUMANS:
        m &= (sheets[k].assessable == 'yes').values & (sheets[k][col] != '').values
    counts = np.array([[sum(sheets[k][col].values[i] == l for k in HUMANS) for l in labs]
                       for i in np.where(m)[0]], float)
    n = len(HUMANS)
    p_j = counts.sum(0) / counts.sum()
    p_i = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    pe = (p_j ** 2).sum()
    return m.sum(), (p_i.mean() - pe) / (1 - pe)


def main():
    ids = pd.read_csv(SHEETS['annotator_a'], dtype=str).id.tolist()
    sheets = {k: load(p, ids) for k, p in SHEETS.items()}
    rule = rule_presence(ids)
    print('not assessable:', {k: s.index[s.assessable != 'yes'].tolist() for k, s in sheets.items()})

    pairs = list(combinations(HUMANS, 2))
    print('\nPairwise: n, agreement %, kappa (linear-weighted for fire_in_label)')
    print('item'.ljust(14) + ''.join(f'{a}-{b}'.rjust(22) for a, b in pairs))
    for col, labs, w in ITEMS:
        row = col.ljust(14)
        for a, b in pairs:
            m = (sheets[a].assessable == 'yes') & (sheets[b].assessable == 'yes') & \
                (sheets[a][col] != '') & (sheets[b][col] != '')
            x, y = sheets[a][col][m], sheets[b][col][m]
            row += f'{m.sum():>8} {100 * (x == y).mean():>4.0f}% {cr.kappa(x, y, labs, w):>6.2f}'
        print(row)

    print('\nFleiss kappa, author + annotator_a + annotator_b')
    for col, labs, _ in ITEMS:
        n, k = fleiss(sheets, col, labs)
        print(f'   {col:14} n={n:2}  kappa {k:.2f}')

    print('\nAnnotator vs hot-pixel rule (B12 > B11 variant)')
    for k, s in sheets.items():
        m = s.assessable == 'yes'
        for name in ('fire_in', 'fire_any'):
            h, r = s[name][m], rule[name][m]
            agree = int((h == r).sum())
            lo, hi = cr.wilson(agree, int(m.sum()))
            print(f'   {k:9} {name:8} n={m.sum():2} agree {agree}/{m.sum()} [{lo * 100:.0f}, {hi * 100:.0f}]  '
                  f'kappa {cr.kappa(h, r, ["y", "n"]):.2f}  rule-only {list(h.index[(r == "y") & (h == "n")])}  '
                  f'human-only {list(h.index[(h == "y") & (r == "n")])}')

    print('\nHeadline quantities on the 25 (assessable only)')
    for k, s in sheets.items():
        s = s[s.assessable == 'yes']
        print(f'   {k:9} n={len(s):2}  no fire/scar inside or within 1 km {(s.loc3 == "none").sum():2}  '
              f'little evidence in label {(s.little_evidence == "y").sum():2}  green main cover {(s.green_main == "y").sum():2}')

    print('\nPer-patch 3-way location where the humans disagree (author / annotator_a / annotator_b; rule fire_in)')
    for pid in ids:
        v = [sheets[k].loc[pid, 'loc3'] if sheets[k].loc[pid, 'assessable'] == 'yes' else 'NA'
             for k in HUMANS]
        if len(set(x for x in v[:3] if x != 'NA')) > 1 or 'NA' in v[:3]:
            print(f'   {pid}: {" / ".join(v)}; rule {rule.loc[pid, "fire_in"]}; label px {rule.loc[pid, "label_px"]}')


if __name__ == '__main__':
    main()
