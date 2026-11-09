"""LaTeX table rows of the Supplement, written to <out>/supplement_tables/.
Stops if a census agreement value differs from the reported one."""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import _paths as P
from . import census_agreement as sb
from . import census_prevalence as cr

sys.stdout.reconfigure(encoding='utf-8')
HUM = ('author', 'annotator_a', 'annotator_b')
NAME = {'author': 'First author', 'annotator_a': 'Co-annotator A', 'annotator_b': 'Co-annotator B'}
LATE = ['S073', 'S079']  # completed after the first pass

def write(name, body):
    p = P.out('supplement_tables', name)
    p.write_text(body, encoding='utf-8')
    print('wrote', p)

def landis(k):
    for lo, word in ((0.81, 'almost perfect'), (0.61, 'substantial'), (0.41, 'moderate'),
                     (0.21, 'fair'), (0.0, 'slight')):
        if k >= lo:
            return word
    return 'poor'

def blocks_of(patch):
    rc = patch.str.extract(r'patch_(\d+)_(\d+)').astype(int)
    return (rc[0] // 4).astype(str) + '_' + (rc[1] // 4).astype(str)

def block_boot(df, stat, rng, n=5000):
    b = df.block.unique()
    out = []
    for _ in range(n):
        s = pd.concat([df[df.block == x] for x in rng.choice(b, len(b))])
        v = stat(s)
        if np.isfinite(v):
            out.append(v)
    return np.percentile(out, [2.5, 97.5])

a = pd.read_csv(P.census('first_author.csv'), dtype=str).fillna('')
a = a.merge(pd.read_csv(P.census('key.csv')), on='id')
a['loc'] = cr.location3(a)
a['af_in'] = a.active_fire.isin(['inside', 'both'])
pp = pd.read_csv(P.run_table('sen2fire_attunet_oversample', 'per_patch.csv'))
base = pp[pp.arm == 'baseline'].groupby('patch').f1.mean().rename('f1').reset_index()
m = a.merge(base, on='patch', how='left')
m['block'] = blocks_of(m.patch)
assert len(m) == 99 and m.f1.notna().all() and (m.assessable == 'yes').all()

# The order of the draws below fixes the reported intervals; do not reorder.
rng = np.random.default_rng(0)
ins, non = m[m['loc'] == 'inside'], m[m['loc'] == 'none']
r4_block = []
blocks = m.block.unique()
for _ in range(5000):
    s = pd.concat([m[m.block == b] for b in rng.choice(blocks, len(blocks))])
    i, n_ = s[s['loc'] == 'inside'].f1, s[s['loc'] == 'none'].f1
    if len(i) and len(n_):
        r4_block.append(i.mean() - n_.mean())
r4_block = np.percentile(r4_block, [2.5, 97.5])

ids = pd.read_csv(sb.SHEETS['annotator_a'], dtype=str).id.tolist()
sheets = {k: sb.load(sb.SHEETS[k], ids) for k in HUM}

ITEM_TEXT = {
    'active_fire': 'Active-fire location relative to the label',
    'loc3': 'Fire or scar location (inside / within 1 km / neither)',
    'fire_in_label': 'Share of the label with fire evidence',
    'vgroup': 'Verdict, fire-inside categories merged',
    'label_verdict': 'Verdict on the label',
    'scar_outside': 'Burn-scar location outside the label',
    'green_main': 'Main cover: green vegetation vs other',
    'label_main': 'Main cover class',
}

kap = {}

for col, labs, _ in sb.ITEMS:
    n, k = sb.fleiss(sheets, col, labs)
    ks = []
    for _ in range(2000):
        pick = rng.choice(len(ids), len(ids))
        sub = {h: sheets[h].iloc[pick].reset_index(drop=True) for h in sheets}
        try:
            ks.append(sb.fleiss(sub, col, labs)[1])
        except Exception:
            pass
    ks = np.array(ks)
    ks = ks[np.isfinite(ks)]
    kap[col] = (n, k, np.percentile(ks, 2.5), np.percentile(ks, 97.5))

# reported values; stop if they change
for col, (k, lo, hi) in {'active_fire': (0.82, 0.66, 0.96), 'loc3': (0.86, 0.67, 1.00),
                         'fire_in_label': (0.48, 0.25, 0.67), 'scar_outside': (0.28, 0.09, 0.44),
                         'label_verdict': (0.38, 0.13, 0.58)}.items():
    assert np.allclose(np.round(kap[col][1:], 2), (k, lo, hi)), (col, kap[col])

rows = []
for col in ITEM_TEXT:
    n, k, lo, hi = kap[col]
    rows.append(f'{ITEM_TEXT[col]} & {n} & {k:.2f} & [{lo:.2f}, {hi:.2f}] & {landis(k)} \\\\')
write('tab_kappa.tex', '\n'.join(rows) + '\n')

# pairwise
pairs = list(combinations(HUM, 2))
rows = []
for col, labs, w in sb.ITEMS:
    cells = []
    for x, y in pairs:
        mk = (sheets[x].assessable == 'yes') & (sheets[y].assessable == 'yes') & \
             (sheets[x][col] != '') & (sheets[y][col] != '')
        u, v = sheets[x][col][mk], sheets[y][col][mk]
        cells.append(f'{mk.sum()} & {100 * (u == v).mean():.0f} & {cr.kappa(u, v, labs, w):.2f}')
    star = '$^{\\ast}$' if w else ''
    rows.append(f'{ITEM_TEXT[col]}{star} & ' + ' & '.join(cells) + ' \\\\')
write('tab_pairwise.tex', '\n'.join(rows) + '\n')

# per-annotator proportions on the shared patches
rows = []
for h in HUM:
    s = sheets[h][sheets[h].assessable == 'yes']
    n = len(s)
    le, no, gr = (s.little_evidence == 'y').sum(), (s.loc3 == 'none').sum(), (s.green_main == 'y').sum()
    af = s[s.active_fire != 'none']
    rows.append(f'{NAME[h]} & {n} & {le}/{n} ({100 * le / n:.0f}\\%) & {no}/{n} ({100 * no / n:.0f}\\%) & '
                f'{gr}/{n} ({100 * gr / n:.0f}\\%) & {af.active_fire.isin(["inside", "both"]).sum()}/{len(af)} \\\\')
write('tab_annotator.tex', '\n'.join(rows) + '\n')

# rule vs annotator
rule = sb.rule_presence(ids)
rows = []
for h in HUM:
    s = sheets[h]
    mk = s.assessable == 'yes'
    cells = []
    for nm in ('fire_in', 'fire_any'):
        u, r = s[nm][mk], rule[nm][mk]
        ag = int((u == r).sum())
        lo, hi = cr.wilson(ag, int(mk.sum()))
        cells.append(f'{ag}/{mk.sum()} & [{100 * lo:.0f}, {100 * hi:.0f}] & {cr.kappa(u, r, ["y", "n"]):.2f}')
    rows.append(f'{NAME[h]} & ' + ' & '.join(cells) + ' \\\\')
write('tab_rule.tex', '\n'.join(rows) + '\n')

# rule as first specified vs adopted, all 99 patches
hr = pd.read_csv(P.analysis('active_fire_rule_sensitivity.csv'))
rows = []
for b11, label in ((False, 'Original: $B_{12}>0.5$, $B_{12}>1.5\\,B_8$'), (True, 'Adopted: $+\\,B_{12}>B_{11}$')):
    r = hr[(hr['B12>'] == 0.5) & (hr.ratio == 1.5) & (hr['B12>B11'] == b11)].iloc[0]
    rows.append(f'{label} & {r.share_in_label:.1f} & {r.share_in_misplaced:.1f} & {r.pct_of_label_area:.2f} & '
                f'{r.prec_in:.2f} & {r.rec_in:.2f} & {r.kappa_in:.2f} & '
                f'{r.prec_any:.2f} & {r.rec_any:.2f} & {r.kappa_any:.2f} \\\\')
write('tab_rule_original.tex', '\n'.join(rows) + '\n')


def r1(s): 
    return s.fire_in_label.isin(['none', 'lt10']).mean()

def r2(s):
    t = s[s.active_fire != 'none']
    return t.af_in.mean() if len(t) else np.nan

def r3(s): 
    return (s['loc'] == 'none').mean()

# one generator; this order fixes the reported intervals
rng3 = np.random.default_rng(0)
pb = [rng3.choice(ins.f1.values, len(ins)).mean() - rng3.choice(non.f1.values, len(non)).mean() for _ in range(5000)]
bi = {}

for name, f in (('r1', r1), ('r3', r3), ('inside', lambda s: (s['loc'] == 'inside').mean()), ('r2', r2)):
    bi[name] = block_boot(m, f, rng3)

rows = []
for label, key, k, n in (('Little or no fire evidence in the label', 'r1', r1(m) * 99, 99),
                         ('Active fire inside the label, of patches with active fire', 'r2',
                          m[m.active_fire != 'none'].af_in.sum(), (m.active_fire != 'none').sum()),
                         ('No fire or scar inside or within 1 km', 'r3', r3(m) * 99, 99)):
    k = int(round(k))
    lo, hi = cr.wilson(k, n)
    blo, bhi = bi[key]
    rows.append(f'{label} & {k}/{n} = {100 * k / n:.1f}\\% & [{100 * lo:.1f}, {100 * hi:.1f}] & '
                f'[{100 * blo:.1f}, {100 * bhi:.1f}] \\\\')

d = ins.f1.mean() - non.f1.mean()
rows.append(f'Per-patch $F_1$, inside minus neither (46 vs 49 patches), Attention U-Net, ten seeds & {d:.2f} & '
            f'[{np.percentile(pb, 2.5):.2f}, {np.percentile(pb, 97.5):.2f}] & [{r4_block[0]:.2f}, {r4_block[1]:.2f}] \\\\')

sm = pd.read_csv(P.analysis('score_by_census_location.csv')).set_index('model')
for arch in ('DeepLabV3', 'SegFormer-B2'):
    r = sm.loc[arch]
    rows.append(f'\\quad same, {arch}, three seeds ({r.inside_f1:.2f} vs {r.none_f1:.2f}) & {r["diff"]:.2f} & '
                f'[{r.patch_lo:.2f}, {r.patch_hi:.2f}] & [{r.block_lo:.2f}, {r.block_hi:.2f}] \\\\')

ev = pd.read_csv(P.analysis('audit_rule_rates.csv'))
ev['n'] = ev.fire_pct / 100 * 512 * 512
ev['block'] = blocks_of(ev.path)
rng4 = np.random.default_rng(0)
for col, label in (('support_frac', 'Labelled pixels with burn evidence (spectral rule)'),
                   ('veg_unsup_frac', 'Labelled pixels of healthy vegetation (spectral rule)')):
    pooled = lambda s, c=col: (s[c] * s.n).sum() / s.n.sum()
    eb = ev.block.unique()
    bb = [pooled(pd.concat([ev[ev.block == b] for b in rng4.choice(eb, len(eb))])) for _ in range(5000)]
    pt = [pooled(ev.iloc[rng4.choice(len(ev), len(ev))]) for _ in range(5000)]
    rows.append(f'{label} & {pooled(ev):.1f}\\% & [{np.percentile(pt, 2.5):.1f}, {np.percentile(pt, 97.5):.1f}] & '
                f'[{np.percentile(bb, 2.5):.1f}, {np.percentile(bb, 97.5):.1f}] \\\\')

# companion audit strata (audit_strata_intervals); CaBuAr blocks are fires
sb_ = pd.read_csv(P.analysis('audit_strata_intervals.csv')).set_index(['dataset', 'stratum', 'rule'])
rows.append('\\midrule')
for (ds, st, rule), label in (
        (('sen2fire', 'nonfire_le1km', 'burn evidence'), 'Burn evidence, non-fire-labelled pixels within 1 km of a label'),
        (('sen2fire', 'nonfire_1to3km', 'burn evidence'), '\\quad 1--3 km from a label'),
        (('sen2fire', 'nonfire_gt3km', 'burn evidence'), '\\quad more than 3 km from a label'),
        (('sen2fire', 'nonfire_firefree_patch', 'burn evidence'), '\\quad in patches with no labelled fire'),
        (('cabuar', 'burn_label', 'burn evidence'), 'CaBuAr, burn evidence inside the CAL FIRE perimeters$^{c}$'),
        (('cabuar', 'unburned', 'burn evidence'), 'CaBuAr, burn evidence outside the perimeters$^{c}$'),
        (('cabuar', 'burn_label', 'healthy vegetation'), 'CaBuAr, healthy vegetation inside the perimeters$^{c}$')):
    r = sb_.loc[(ds, st, rule)]
    rows.append(f'{label} & {r.pct:.1f}\\% & [{r.patch_lo:.1f}, {r.patch_hi:.1f}] & '
                f'[{r.block_lo:.1f}, {r.block_hi:.1f}] \\\\')
write('tab_uncertainty.tex', '\n'.join(rows) + '\n')

# subsets: all; without late entries; without S001-S005 / S001-S008 (rubric revision)
REVISION5 = [f'S00{i}' for i in range(1, 6)]
REVISION8 = [f'S00{i}' for i in range(1, 9)]
subsets = [m, m[~m.id.isin(LATE)], m[~m.id.isin(REVISION5)], m[~m.id.isin(REVISION8)]]

def frac(k, n):
    return f'{k}/{n} = {100 * k / n:.1f}\\%'

rows = []
for label, fn in (('Little or no fire evidence in the label', lambda d: (int(r1(d) * len(d) + .5), len(d))),
                  ('Active fire inside the label, of patches with active fire',
                   lambda d: (int(d[d.active_fire != 'none'].af_in.sum()), int((d.active_fire != 'none').sum()))),
                  ('No fire or scar inside or within 1 km', lambda d: (int((d['loc'] == 'none').sum()), len(d)))):
    rows.append(f'{label} & ' + ' & '.join(frac(*fn(d)) for d in subsets) + ' \\\\')
rows.append('Per-patch $F_1$, inside / neither (Attention U-Net, ten seeds) & '
            + ' & '.join(f"{d[d['loc'] == 'inside'].f1.mean():.2f} / {d[d['loc'] == 'none'].f1.mean():.2f}" for d in subsets)
            + ' \\\\')
write('tab_sensitivity.tex', '\n'.join(rows) + '\n')

# item-level responses
AB = {'none': 'none', 'inside': 'in', 'outside': 'out', 'both': 'both',
      'dense_veg': 'DV', 'open_veg': 'OV', 'dry_veg': 'DR', 'orange_brown': 'OB', 'veg_scar_mix': 'VS',
      'burn_scar': 'BS', 'active_fire_area': 'AF', 'bare_or_crop': 'BC',
      'lt10': '$<$10', '10to50': '10--50', 'gt50': '$>$50', 'adjacent': 'adj', 'far': 'far',
      'fire_in_label': 'on fire', 'oversized': 'oversized', 'fire_nearby': 'nearby', 'mislabeled': 'no fire',
      'unsure': 'unsure', 'high': 'H', 'medium': 'M', 'low': 'L', 'near': 'near', '': '--'}

ab = lambda v: AB.get(v, v)
rows = []
for _, r in m.sort_values('id').iterrows():
    pid = r.id + ('$^{\\dagger}$' if r.id in LATE else '')
    pt = r.patch.replace('scene_4_patch_', '').replace('.npz', '').replace('_', ',')
    rows.append(f'{pid} & ({pt}) & {ab(r.active_fire)} & {ab(r.label_main)} & {ab(r.label_second)} & '
                f'{ab(r.fire_in_label)} & {ab(r.scar_outside)} & {ab(r.label_verdict)} & {ab(r.confidence)} & '
                f'{ab(r["loc"])} & {r.f1:.2f} \\\\')
write('tab_items_A.tex', '\n'.join(rows) + '\n')

rows = []
for pid in sorted(ids):
    cells = []
    for col in ('active_fire', 'fire_in_label', 'scar_outside', 'label_verdict', 'loc3'):
        v = [ab(sheets[h].loc[pid, col]) if sheets[h].loc[pid, 'assessable'] == 'yes' else 'n/a' for h in HUM]
        cells.append('/'.join(v))
    rows.append(f'{pid} & ' + ' & '.join(cells) + ' \\\\')
write('tab_items_B.tex', '\n'.join(rows) + '\n')

# post hoc experiments
def ci(x):  # mean and seed standard deviation
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1)


neg = pd.read_csv(P.run_table('cabuar_degrade_negatives', 'per_patch.csv'))
e11 = pd.read_csv(P.run_table('cabuar_degrade', 'per_patch.csv'))
e11r = pd.read_csv(P.run_table('cabuar_degrade', 'results.csv'))
negr = pd.read_csv(P.run_table('cabuar_degrade_negatives', 'results.csv'))
rows = []
for label, pp_, rr, arm in (('Matched, $e=0.35$ (control arm)', e11, e11r, 'matched_0.35'),
                            ('Matched, $e=0.35$, + 59 empty-label pre-fire patches', neg, negr, 'matched_0.35_preneg'),
                            ('Clean 20\\,m + the same empty-label patches', neg, negr, '20m_preneg')):
    p = pp_[(pp_.arm == arm) & (pp_.cov_eval > 0.001)]
    r = rr[rr.arm == arm]
    sw = p[p.swapped == 1].groupby('seed').f1_degraded.mean() * 100
    us = p[p.swapped == 0].groupby('seed').f1_degraded.mean() * 100
    share = p[p.swapped == 1].groupby('seed').pred_share.mean() * 100 if 'pred_share' in p and len(sw) else []
    mac, h = ci(r.macro_deg_primary)
    rows.append(f'{label} & {len(r)} & {mac:.1f} $\\pm$ {h:.1f} & '
                f'{sw.mean():.1f} & {us.mean():.1f} & ' + (f'{np.mean(share):.1f}' if len(share) else '--') +
                f' & {r.macro_clean_unswapped.mean():.1f} \\\\'.replace('nan', '--'))
write('tab_empty_label_negatives.tex', '\n'.join(rows).replace('nan', '--') + '\n')

fo = pd.read_csv(P.run_table('sen2fire_attunet_fireonly', 'results.csv'))
pf = pd.read_csv(P.run_table('sen2fire_attunet_fireonly', 'per_patch.csv')).merge(pd.read_csv(P.census('location.csv')), on='patch')
rows = []
for label, arm in (('Baseline, rerun (natural sampling, 89.5\\% empty)', 'baseline_bridge'),
                   ('Fire patches only (no empty label seen)', 'fireonly')):
    r = fo[fo.arm == arm]
    g = pf[pf.arm == arm].groupby(['loc', 'seed']).agg(f1=('f1', 'mean'), ps=('pred_share', 'mean')).groupby('loc').mean() * 100
    mac, h = ci(r.macro_f1)
    rows.append(f'{label} & {len(r)} & {mac:.1f} $\\pm$ {h:.1f} & {r.prec.mean():.1f} & {r.rec.mean():.1f} & '
                f'{g.loc["inside", "f1"]:.1f} & {g.loc["none", "f1"]:.1f} & {g.loc["none", "ps"]:.1f} \\\\')
write('tab_fireonly.tex', '\n'.join(rows) + '\n')

sw = pd.read_csv(P.analysis('cabuar_degrade_threshold_sweep.csv'))
taus = [c for c in sw.columns if c.startswith('f1_')]
e10 = sw[sw.family == '20m_E10']
rows = []
for fam, lab in (('resolution', 'resolution only'), ('inflate_area', 'area inflation'),
                 ('noevid_0.35', 'no-evidence swaps, $e=0.35$'), ('matched_0.20', 'matched, $e=0.20$'),
                 ('matched_0.35', 'matched, $e=0.35$')):
    arm = sw[(sw.family == fam) & (sw.swapped == 0)]
    keys = set(arm.key)
    ref = e10[e10.key.isin(keys)]
    ca = arm.groupby('seed')[taus].mean().mean() * 100
    cr_ = ref.groupby('seed')[taus].mean().mean() * 100
    ta, tr = ca.idxmax(), cr_.idxmax()
    rows.append(f'{lab} & {len(keys)} & {cr_.max():.1f} ({tr[3:]}) & {ca.max():.1f} ({ta[3:]}) & '
                f'{cr_.max() - ca.max():+.1f} & {ref.groupby("seed").auroc.mean().mean():.3f} & '
                f'{arm.groupby("seed").auroc.mean().mean():.3f} \\\\')
write('tab_threshold_sweep.tex', '\n'.join(rows) + '\n')

pr = pd.concat([pd.read_csv(P.analysis('cabuar_prefire_probe.csv')), pd.read_csv(P.analysis('cabuar_prefire_probe_extended.csv'))])
FAM = [('E10_20m', 'clean labels', 'no', '0'), ('E12_20m_preneg', 'clean + empty-label patches', 'no', '0'),
       ('E11_resolution', 'resolution only', 'no', '0'), ('E11_noevid_0.35', 'no-evidence swaps', 'no', '0.35'),
       ('E11_inflate_area', 'area inflation', 'yes', '0'), ('E11_matched_0.20', 'matched', 'yes', '0.20'),
       ('E11_matched_0.35', 'matched', 'yes', '0.35'),
       ('E12_matched_0.35_preneg', 'matched + empty-label patches', 'yes', '0.35')]
rows = []
for fam, lab, infl, e in FAM:
    g = pr[pr.family == fam].groupby('seed')[['pred_pre', 'pred_post', 'auroc_pre', 'f1_post']].mean()
    rows.append(f'{lab} & {infl} & {e} & {len(g)} & {100 * g.pred_pre.mean():.1f} '
                f'[{100 * g.pred_pre.min():.1f}, {100 * g.pred_pre.max():.1f}] & {100 * g.pred_post.mean():.1f} & '
                f'{g.auroc_pre.mean():.2f} & {100 * g.f1_post.mean():.1f} \\\\')
write('tab_prefire_probe.tex', '\n'.join(rows) + '\n')
