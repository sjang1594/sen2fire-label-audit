"""Census rates from the first author's blind answers on the 99 test fire patches.
Usage: python -m sen2fire_audit.reproduce census_prevalence [sheet.csv] [second_sheet.csv]"""
from math import sqrt
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

from . import _paths as P

DATA = P.sen2fire()
FIRE_ORDER = ['none', 'lt10', '10to50', 'gt50']
ALLOWED = {
    'assessable': {'yes', 'no_cloud', 'no_smoke', 'no_other'},
    'active_fire': {'none', 'inside', 'outside', 'both'},
    # green = dense + open
    # label_main: plurality class
    'label_main': {'dense_veg', 'open_veg', 'dry_veg', 'orange_brown', 'veg_scar_mix', 'burn_scar', 'active_fire_area', 'bare_or_crop'},
    'label_verdict': {'fire_in_label', 'oversized', 'fire_nearby', 'mislabeled', 'unsure'},
    'label_second': {'dense_veg', 'open_veg', 'dry_veg', 'orange_brown', 'veg_scar_mix', 'burn_scar', 'active_fire_area', 'bare_or_crop', 'none'},
    'fire_in_label': set(FIRE_ORDER),
    'scar_outside': {'none', 'adjacent', 'far', 'both'},
    'confidence': {'high', 'medium', 'low'},
}
GATED = ['active_fire', 'label_main', 'label_second', 'fire_in_label', 'scar_outside', 'label_verdict']
RULE_SUPPORT_THRESHOLD = 33.0
HOT_MIN_PX = 20
RNG = np.random.default_rng(0)


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def kappa(a, b, labels, weights=None):
    """Cohen's kappa; weights='linear' for ordinal labels."""
    k = len(labels); ix = {l: i for i, l in enumerate(labels)}
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[ix[x], ix[y]] += 1
    n = m.sum()
    if n == 0:
        return float('nan')
    w = np.ones((k, k)) - np.eye(k) if weights is None else \
        np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    exp = np.outer(m.sum(1), m.sum(0)) / n
    return 1 - (w * m).sum() / (w * exp).sum() if (w * exp).sum() > 0 else float('nan')


def auc(score, y):
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    return np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :])


def rate(k, n, label):
    lo, hi = wilson(k, n)
    pct = k / n * 100 if n else float('nan')
    print(f'   {label}: {k}/{n} = {pct:.1f}%  Wilson 95% CI [{lo * 100:.1f}, {hi * 100:.1f}]')


def load_sheet(path):
    s = pd.read_csv(path, dtype=str).fillna('')
    for col, ok in ALLOWED.items():
        s[col] = s[col].str.strip().str.lower()
        bad = s[(s[col] != '') & ~s[col].isin(ok)]
        if len(bad):
            raise SystemExit(f'{path}: invalid {col} values {sorted(set(bad[col]))} at IDs {bad.id.tolist()[:5]}')
    missing = s[(s.assessable == 'yes') & (s[GATED] == '').any(axis=1)]
    if len(missing):
        raise SystemExit(f'{path}: assessable=yes rows missing answers: {missing.id.tolist()[:10]}')
    return s, s[s.assessable != '']


NULL_TRANSFORMS = (np.flipud, np.fliplr, np.transpose, lambda a: np.rot90(a, 1),
                   lambda a: np.rot90(a, 2), lambda a: np.rot90(a, 3))


def rule_features(idx_list):
    """Per patch: label share, hot pixels inside the label and in the patch, misplaced-label hit rate."""
    lines = [l.strip() for l in open(DATA / 'test.txt') if l.strip()]
    rows = []
    for idx in idx_list:
        d = np.load(DATA / lines[idx].replace('\\', os.sep))
        img = d['image'].astype(np.float32) / 10000.0
        gt = d['label'] == 1
        hot = (img[11] > 0.5) & (img[11] > 1.5 * img[7])
        null_hit = np.mean([(hot & t(gt)).sum() >= HOT_MIN_PX for t in NULL_TRANSFORMS])
        rows.append({'idx': idx, 'label_share': gt.mean(), 'hot_in': int((hot & gt).sum()),
                     'hot_all': int(hot.sum()), 'null_hit': null_hit})
    return pd.DataFrame(rows)


def human_vs_rule(m, tag):
    for name, h, r in (
            ('inside the label', m.active_fire.isin(['inside', 'both']), m.hot_in >= HOT_MIN_PX),
            ('anywhere in patch', m.active_fire != 'none', m.hot_all >= HOT_MIN_PX)):
        tp = int((r & h).sum()); fp = int((r & ~h).sum()); fn = int((~r & h).sum())
        k = kappa(h.map({True: 'y', False: 'n'}), r.map({True: 'y', False: 'n'}), ['y', 'n'])
        print(f'   {tag} active fire {name}: agreement {(h == r).mean() * 100:.0f}%  kappa {k:.2f}  '
              f'rule precision {tp / max(tp + fp, 1):.2f}  recall {tp / max(tp + fn, 1):.2f}  '
              f'(tp {tp}, fp {fp}, fn {fn}, n={len(m)})')


def active_location(ass, tag):
    seen = ass[ass.active_fire != 'none']
    rate(int(seen.active_fire.isin(['inside', 'both']).sum()), len(seen),
         f'{tag} active fire inside the label (inside/both) among patches with active fire')
    if len(seen):
        ruled = seen[seen.hot_all >= HOT_MIN_PX]
        print(f'   {tag} chance reference (misplaced label, rule hot pixels, n={len(ruled)} of these patches): '
              f'{ruled.null_hit.mean() * 100:.1f}% would still show fire inside; '
              f'correct label: {(ruled.hot_in >= HOT_MIN_PX).mean() * 100:.1f}%  '
              f'(label area share {seen.label_share.mean() * 100:.1f}%)')


def main(path_a, path_b):
    a_all, a = load_sheet(path_a)
    print(f'sheet A: {len(a)}/{len(a_all)} rows filled')
    if (a.assessable == 'yes').sum() == 0:
        print('no assessable verdicts yet; fill the sheet first')
        return
    key = pd.read_csv(P.census('key.csv'))
    a = a.merge(key, on='id')
    a = a.merge(rule_features(a.idx.tolist()), on='idx')
    ass = a[a.assessable == 'yes']
    print(f'assessable: {len(ass)}  not assessable: {(a.assessable != "yes").sum()} '
          f'{a[a.assessable != "yes"].assessable.value_counts().to_dict()}')

    print('\nA. active fire:', ass.active_fire.value_counts().to_dict())
    active_location(ass, 'all')

    print('\nB. label_main:', ass.label_main.value_counts().to_dict())
    grp = ass.label_main.replace({'dense_veg': 'green_veg', 'open_veg': 'green_veg'})
    print('   grouped (green_veg = dense + open):', grp.value_counts().to_dict(),
          '-> green_veg most common:', bool(len(grp) and grp.value_counts().idxmax() == 'green_veg'))
    green = {'dense_veg', 'open_veg'}
    print('   label_second:', ass.label_second.value_counts().to_dict())
    print(f'   green vegetation as most extensive class: {ass.label_main.isin(green).sum()}/{len(ass)};'
          f' in the top two: {(ass.label_main.isin(green) | ass.label_second.isin(green)).sum()}/{len(ass)}')
    print('   fire_in_label:', ass.fire_in_label.value_counts().reindex(FIRE_ORDER, fill_value=0).to_dict())
    rate(int(ass.fire_in_label.isin(['none', 'lt10']).sum()), len(ass), 'label largely without fire evidence (none/lt10)')

    print('\nB2. label_verdict:', ass.label_verdict.value_counts().to_dict())
    rate(int((ass.label_verdict == 'mislabeled').sum()), len(ass), 'clear mislabel (no fire evidence in or within 1 km of the label)')
    rate(int(ass.label_verdict.isin(['fire_in_label', 'oversized']).sum()), len(ass), 'fire evidence inside the label (located correctly)')
    rate(int((ass.label_verdict == 'oversized').sum()), len(ass), 'located correctly but label mostly unrelated to fire (oversized)')
    # flag, do not correct
    bad = ass[(ass.label_verdict == 'mislabeled') & ((ass.active_fire != 'none') | (ass.fire_in_label != 'none') | (ass.scar_outside == 'adjacent'))]
    if len(bad):
        print('   check: mislabeled verdict but fire/scar recorded nearby ->', bad.id.tolist())
    print('\nC. human vs hot-pixel rule')
    human_vs_rule(ass, 'all')

    ev = pd.read_csv(P.analysis('audit_rule_rates.csv'))[['idx', 'support_frac']]
    m = ass.merge(ev, on='idx', how='left').dropna(subset=['support_frac'])
    y = m.fire_in_label.isin(['10to50', 'gt50']).values
    sc = m.support_frac.values.astype(float)
    boots = [auc(sc[i], y[i]) for i in (RNG.integers(0, len(y), len(y)) for _ in range(2000))]
    boots = [b for b in boots if not np.isnan(b)]
    print(f'\nD. human positives (fire evidence >= 10% of label): {y.sum()}/{len(y)}')
    if boots:
        print(f'   ROC AUC of audit support fraction: {auc(sc, y):.2f}  '
              f'95% CI [{np.percentile(boots, 2.5):.2f}, {np.percentile(boots, 97.5):.2f}]')
    pred = sc >= RULE_SUPPORT_THRESHOLD
    tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum()
    print(f'   at support >= {RULE_SUPPORT_THRESHOLD}%: precision {tp / max(tp + fp, 1):.2f}  '
          f'recall {tp / max(tp + fn, 1):.2f}  (tp {tp}, fp {fp}, fn {fn})')

    print('\nE. scar_outside:', ass.scar_outside.value_counts().to_dict())
    sc_any = ass[ass.scar_outside != 'none']
    rate(int(sc_any.scar_outside.isin(['adjacent', 'both']).sum()), len(sc_any),
         'scar outside the label lies within 1 km of it (adjacent/both)')

    # location from the fire items, not from label_verdict
    print('\nG. location of fire relative to the label (from the item answers)')
    for name, inside in (('active fire only', ass.active_fire.isin(['inside', 'both'])),
                         ('active fire or scar', ass.active_fire.isin(['inside', 'both']) | (ass.fire_in_label != 'none'))):
        near = ~inside & ((ass.label_verdict == 'fire_nearby') | ass.scar_outside.isin(['adjacent', 'both']))
        print(f'   [{name}]')
        rate(int(inside.sum()), len(ass), 'fire inside the label')
        rate(int(near.sum()), len(ass), 'none inside, fire or scar within 1 km')
        rate(int((~inside & ~near).sum()), len(ass), 'none inside or within 1 km')
    tags = (a.get('notes', pd.Series('', index=a.index)).fillna('') + ' ' +
            a.get('changed_from_author', pd.Series('', index=a.index)).fillna(''))
    for t in ('INVERTED_LABEL', 'AUTHOR_SKETCH', 'CONSISTENCY_FLAG', 'PENDING'):
        ids = a.id[tags.str.contains(t)].tolist()
        print(f'   tag {t}: {len(ids)} {ids}')

    hi = ass[ass.confidence == 'high']
    print(f'\nRobustness, confidence == high (n={len(hi)})')
    if len(hi):
        active_location(hi, 'high')
        human_vs_rule(hi, 'high')

    if path_b and Path(path_b).exists():
        b_all, b = load_sheet(path_b)
        j = a_all.merge(b, on='id', suffixes=('_a', '_b'))
        j = j[(j.assessable_a != '') & (j.assessable_b != '')]
        print(f'\nF. overlap with both filled: {len(j)}')
        if len(j):
            print(f'   assessable: agreement {(j.assessable_a == j.assessable_b).mean() * 100:.0f}%  '
                  f'kappa {kappa(j.assessable_a, j.assessable_b, sorted(ALLOWED["assessable"])):.2f}')
            jj = j[(j.assessable_a == 'yes') & (j.assessable_b == 'yes')]
            for col in ('active_fire', 'label_main', 'scar_outside'):
                if len(jj):
                    print(f'   {col} (n={len(jj)}): agreement {(jj[col + "_a"] == jj[col + "_b"]).mean() * 100:.0f}%  '
                          f'kappa {kappa(jj[col + "_a"], jj[col + "_b"], sorted(ALLOWED[col])):.2f}')
            if len(jj):
                print(f'   fire_in_label (n={len(jj)}): agreement {(jj.fire_in_label_a == jj.fire_in_label_b).mean() * 100:.0f}%  '
                      f'linear-weighted kappa {kappa(jj.fire_in_label_a, jj.fire_in_label_b, FIRE_ORDER, "linear"):.2f}')
    else:
        print('\nF. sheet B not provided or not found; inter-annotator agreement skipped')


VERDICT_GROUP = {'fire_in_label': 'caught', 'oversized': 'caught', 'fire_nearby': 'missed_near',
                 'mislabeled': 'missed', 'unsure': 'unsure'}


def location3(s):
    inside = s.active_fire.isin(['inside', 'both']) | (s.fire_in_label != 'none')
    near = ~inside & ((s.label_verdict == 'fire_nearby') | s.scar_outside.isin(['adjacent', 'both']))
    return np.where(inside, 'inside', np.where(near, 'near', 'none'))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else P.census('first_author.csv'),
         sys.argv[2] if len(sys.argv) > 2 else None)
