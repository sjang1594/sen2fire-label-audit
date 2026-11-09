"""Label-degradation arms on CaBuAr and their training-free oracle ceiling.
Writes degradation_oracle_examples.png; --scope pooled ranks inflate targets over all patches."""
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import rankdata
import pandas as pd
import matplotlib
from . import _paths as P

H5 = P.cabuar_h5()
S2F = P.sen2fire()
BLOCK = 50                      # 1 km at 20 m
MIN_COV = 0.001                 # fire-patch rule
SEED = 20261024
SPLIT = {0: 'test', 1: 'val', 2: 'train', 3: 'train', 4: 'train'}
B4, B8, B12 = 3, 7, 11
FILTER_CSV = P.analysis('prefire_swap_filter.csv')
E_VALUES = (0.20, 0.25, 0.30, 0.35)
E_PRIMARY, E_SENSITIVITY = 0.35, 0.20     # chosen from the census, not from the oracle
# inflate targets ranked per split (fixed before training; pooled ranking biased test upward)
TARGET_SCOPE = 'split'          # 'pooled': earlier version, for comparison
ARMS = (['clean', 'resolution', 'inflate_area'] + [f'noevid_{e:.2f}' for e in E_VALUES]
        + [f'matched_{e:.2f}' for e in E_VALUES])
SWAP_OF = {a: float(a.split('_')[1]) for a in ARMS if a.startswith(('noevid_', 'matched_'))}

def degrade(lbl, block):
    """Block-max to a block x block grid and back (block 1 = unchanged)."""
    lbl = np.asarray(lbl) > 0
    if block == 1:
        return lbl.copy()
    h, w = lbl.shape
    hp, wp = -(-h // block) * block, -(-w // block) * block
    pad = np.zeros((hp, wp), dtype=bool)
    pad[:h, :w] = lbl
    coarse = pad.reshape(hp // block, block, wp // block, block).max(axis=(1, 3))
    up = np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)
    return up[:h, :w]

def coarse_grid(lbl):
    h, w = lbl.shape
    hp, wp = -(-h // BLOCK) * BLOCK, -(-w // BLOCK) * BLOCK
    pad = np.zeros((hp, wp), bool)
    pad[:h, :w] = lbl
    return pad.reshape(hp // BLOCK, BLOCK, wp // BLOCK, BLOCK).max(axis=(1, 3))

def upsample(grid, shape):
    return np.repeat(np.repeat(grid, BLOCK, axis=0), BLOCK, axis=1)[:shape[0], :shape[1]]

def dilate8(g):
    out = g.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sh = np.zeros_like(g)
            ys = slice(max(dy, 0), g.shape[0] + min(dy, 0))
            yd = slice(max(-dy, 0), g.shape[0] + min(-dy, 0))
            xs = slice(max(dx, 0), g.shape[1] + min(dx, 0))
            xd = slice(max(-dx, 0), g.shape[1] + min(-dx, 0))
            sh[yd, xd] = g[ys, xs]
            out |= sh
    return out

def purity(lab, burn):
    n = lab.sum()
    return (lab & burn).sum() / n if n else float('nan')

def grow(burn, done):
    """Dilate the 1 km label ring by ring until done(label); returns (label, rings, fully_covered)."""
    g = coarse_grid(burn)
    lab = upsample(g, burn.shape)
    rings = 0
    while not done(lab):
        g2 = dilate8(g)
        if g2.sum() == g.sum():
            return lab, rings, True
        g, rings = g2, rings + 1
        lab = upsample(g, burn.shape)
    return lab, rings, bool(lab.all())

def grow_closest(burn, target):
    """Ring (>= 0) whose label share is closest to target; returns (label, rings)."""
    g = coarse_grid(burn)
    best, best_d, best_r, r = g, abs(upsample(g, burn.shape).mean() - target), 0, 0
    while upsample(g, burn.shape).mean() < target:
        g2 = dilate8(g)
        if g2.sum() == g.sum():
            break
        g, r = g2, r + 1
        d = abs(upsample(g, burn.shape).mean() - target)
        if d < best_d:
            best, best_d, best_r = g, d, r
    return upsample(best, burn.shape), best_r

def f1(pred, gt):
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    return 2 * tp / (2 * tp + fp + fn + 1e-8), tp, fp, fn

def q(a):
    a = np.asarray(a, float)
    return f'mean {np.nanmean(a):5.1f}  Q1 {np.nanpercentile(a, 25):5.1f}  median {np.nanmedian(a):5.1f}  Q3 {np.nanpercentile(a, 75):5.1f}'

def sen2fire_shares():
    lines = [l.strip() for l in open(S2F / 'test.txt') if l.strip()]
    out = []
    for rel in lines:
        lab = np.load(S2F / rel.replace('\\', os.sep))['label'] == 1
        if lab.mean() > MIN_COV:
            out.append(lab.mean())
    return np.array(out)

def nbr_neg(img, mask):
    img = img.astype(np.float32) / 10000.0
    b8, b12 = img[..., B8], img[..., B12]
    nbr = (b8 - b12) / (b8 + b12 + 1e-6)
    valid = mask & ((b8 + b12) > 0)
    return (nbr[valid] < 0).mean() if valid.any() else float('nan')

def arm_spec(verbose=False, scope=None):
    """Per-patch arm specification shared by the oracle and the degradation runs."""

    scope = scope or TARGET_SCOPE
    with h5py.File(H5, 'r') as h5:
        keys = sorted(h5.keys())
        burn = {k: np.asarray(h5[k]['mask'])[..., 0] > 0 for k in keys}
        fold = {k: int(h5[k].attrs['fold']) for k in keys}
        has_pre_raw = {k: 'pre_fire' in h5[k] for k in keys}
        blank_pre = {k for k in keys if has_pre_raw[k] and np.asarray(h5[k]['pre_fire']).max() == 0}
        has_pre = {k: has_pre_raw[k] and k not in blank_pre for k in keys}
        attr_names = sorted({a for k in keys for a in h5[k].attrs})
    fire = {k for k in keys if burn[k].mean() > MIN_COV}
    one_km = {k: degrade(burn[k], BLOCK) for k in keys}

    filt = pd.read_csv(FILTER_CSV)
    passed = set(filt.loc[filt['pass'], 'key'])
    rng = np.random.default_rng(SEED)
    sel = {e: set() for e in E_VALUES}

    if verbose:
        print('No-evidence arm = pre-fire swap; pool = prefire_swap_filter pass (seasonal-offset dNBR >= 0.1 on '
              '>= 70% of perimeter px, nodata < 1%); same e for every split; nested selection')

    for s in ('train', 'val', 'test'):
        fk = sorted(k for k in fire if SPLIT[fold[k]] == s)
        pool = list(rng.permutation(sorted(k for k in fk if k in passed)))
        if verbose:
            print(f'  {s:5}: fire patches {len(fk):3}, filtered pool {len(pool):3} (cap e {len(pool) / len(fk):.3f})')
        for e in E_VALUES:
            want = round(e * len(fk))
            got = pool[:min(want, len(pool))]
            sel[e] |= set(got)
            if verbose:
                print(f'      e={e:.2f}: target {want:3}, swapped {len(got):3} -> achieved e {len(got) / len(fk):.3f}'
                      + ('  (POOL TOO SMALL)' if len(pool) < want else ''))
    assert all(sel[a] <= sel[b] for a, b in zip(E_VALUES, E_VALUES[1:])), 'nested selection'

    s2f = sen2fire_shares()
    groups = ([keys] if scope == 'pooled' else
              [[k for k in keys if SPLIT[fold[k]] == s] for s in ('train', 'val', 'test')])
    assert scope in ('split', 'pooled') and sum(map(len, groups)) == len(keys)

    target = {}
    for grp in groups:
        share1 = np.array([one_km[k].mean() for k in grp])
        quant = (rankdata(share1, method='average') - 0.5) / len(grp)
        target.update(zip(grp, np.quantile(s2f, quant)))

    inflate, rings = {}, {}
    for k in keys:
        inflate[k], rings[k] = grow_closest(burn[k], target[k])
    return dict(keys=keys, burn=burn, fold=fold, has_pre=has_pre, blank_pre=blank_pre,
                attr_names=attr_names, fire=fire, one_km=one_km, sel=sel, s2f=s2f,
                target=target, inflate=inflate, rings=rings)

def main(scope=None):
    scope = scope or TARGET_SCOPE
    print(f'inflate target scope: {scope}')
    spec = arm_spec(verbose=True, scope=scope)
    keys, burn, fold, has_pre, blank_pre = spec['keys'], spec['burn'], spec['fold'], spec['has_pre'], spec['blank_pre']
    attr_names, fire, one_km, sel, s2f = spec['attr_names'], spec['fire'], spec['one_km'], spec['sel'], spec['s2f']
    print(f'  HDF5 per-patch attrs: {attr_names}; acquisition dates: not available')
    print(f'  all-zero (nodata) pre_fire images excluded from the pool: {len(blank_pre)} '
          f'({sum(k in fire for k in blank_pre)} fire patches)')

    # selection bias: fire patches with vs without pre_fire
    w = [k for k in fire if has_pre[k]]; wo = [k for k in fire if not has_pre[k]]
    print(f'\nSelection bias (fire patches, all splits): with pre_fire n={len(w)}, without n={len(wo)}')
    
    for name, f in (('burn-area share %', lambda k: burn[k].mean() * 100),
                    ('1 km label share %', lambda k: one_km[k].mean() * 100)):
        print(f'  {name:18} with: {q([f(k) for k in w])}')
        print(f'  {"":18} w/o : {q([f(k) for k in wo])}')

    # pre-fire images show no burn: NBR < 0 share inside the clean perimeter
    pre_n, post_n, blank = [], [], 0
    with h5py.File(H5, 'r') as h5:
        for k in sorted(sel[max(E_VALUES)]):
            pre = np.asarray(h5[k]['pre_fire']); post = np.asarray(h5[k]['post_fire'])
            if pre.max() == 0:
                blank += 1
            pre_n.append(nbr_neg(pre, burn[k])); post_n.append(nbr_neg(post, burn[k]))
    
    pre_n, post_n = np.array(pre_n) * 100, np.array(post_n) * 100
    print(f'\nNBR < 0 inside the clean perimeter, swapped patches (e={max(E_VALUES):.2f}, n={len(pre_n)}): '
          f'pre_fire median {np.nanmedian(pre_n):.1f}% [Q1 {np.nanpercentile(pre_n, 25):.1f}, Q3 {np.nanpercentile(pre_n, 75):.1f}], '
          f'post_fire median {np.nanmedian(post_n):.1f}% [Q1 {np.nanpercentile(post_n, 25):.1f}, Q3 {np.nanpercentile(post_n, 75):.1f}]; '
          f'pre > post in {(pre_n > post_n).sum()}; all-zero pre_fire images {blank}; '
          f'pre_fire NBR<0 > 20% in {(pre_n > 20).sum()} patches')

    labels = {a: {} for a in ARMS}
    full = {'inflate_area': 0}
    rings = []
    
    for k in keys:
        b = burn[k]
        labels['clean'][k] = b
        labels['resolution'][k] = one_km[k]
        lab, r = spec['inflate'][k], spec['rings'][k]
        labels['inflate_area'][k] = lab; full['inflate_area'] += lab.all(); rings.append(r)
        for e in E_VALUES:
            labels[f'noevid_{e:.2f}'][k] = one_km[k]
            labels[f'matched_{e:.2f}'][k] = lab
    print(f'inflate_area rings added: median {np.median(rings):.0f}, max {max(rings)}, zero in {sum(r == 0 for r in rings)}/534')

    # sanity checks
    assert all(np.array_equal(degrade(burn[k], 1), burn[k]) for k in keys), '20 m identity'
    assert all(((one_km[k] | burn[k]) == one_km[k]).all() for k in keys), 'block-max containment'
    assert all(labels['inflate_area'][k].sum() >= one_km[k].sum() for k in keys), 'never shrink'
    print('\nchecks: 20 m identity OK, block-max containment OK, inflate_area never shrinks OK')
    fk = [k for k in keys if k in fire]
    print(f'\nLabel share % (fire patches): Sen2Fire test (n={len(s2f)}): {q(s2f * 100)}')
    
    for a in ('clean', 'resolution', 'inflate_area'):
        print(f'  CaBuAr {a:13} (n={len(fk)}): {q([labels[a][k].mean() * 100 for k in fk])}')

    print('Label purity % (fire patches, against the post-fire perimeter)')    
    for a in ('resolution', 'inflate_area'):
        print(f'  CaBuAr {a:13}: {q([purity(labels[a][k], burn[k]) * 100 for k in fk])}')
    
    print(f'fully covered patches (all 534): inflate_area {full["inflate_area"]}')
    print('inflate_area label share % by split (fire patches): mean / median / Q3; Sen2Fire 29.4 / 23.2 / 45.1')
    for s in ('train', 'val', 'test'):
        v = np.array([labels['inflate_area'][k].mean() * 100 for k in fk if SPLIT[fold[k]] == s])
        print(f'  {s:5} (n={len(v)}): {v.mean():5.1f} / {np.median(v):5.1f} / {np.percentile(v, 75):5.1f}')

    # oracle on test
    test = [k for k in keys if fold[k] == 0]
    clean115 = [k for k in test if k in fire]
    print(f'\nOracle on the TEST split (fold 0); prediction = clean perimeter, or empty on swapped patches')
    print(f'{"arm":14}{"macro primary":>14}{"n":>5}{"macro clean-115":>16}{"pooled F1":>11}'
          f'{"purity*":>9}{"label share":>13}{"e test":>8}')
    
    for a in ARMS:
        e = SWAP_OF.get(a)
        swapped = sel[e] if e else set()
        def pred(k):
            return np.zeros_like(burn[k]) if k in swapped else burn[k]
        prim, TP = [], [0, 0, 0]
        for k in test:
            f, tp, fp, fn = f1(pred(k), labels[a][k])
            TP[0] += tp; TP[1] += fp; TP[2] += fn
            if labels[a][k].mean() > MIN_COV:
                prim.append(f)
        sec = [f1(pred(k), labels[a][k])[0] for k in clean115]
        pooled = 2 * TP[0] / (2 * TP[0] + TP[1] + TP[2]) * 100
        pur = np.nanmean([purity(labels[a][k], pred(k)) for k in clean115]) * 100
        share = np.mean([labels[a][k].mean() for k in clean115]) * 100
        e_test = len(swapped & set(clean115)) / len(clean115)
        mark = '  <- primary' if e == E_PRIMARY else ('  <- sensitivity' if e == E_SENSITIVITY else '')
        print(f'{a:14}{np.mean(prim) * 100:14.1f}{len(prim):5d}{np.mean(sec) * 100:16.1f}{pooled:11.1f}'
              f'{pur:9.1f}{share:13.1f}{e_test:8.3f}{mark}')
    print('* purity against what the input shows (swapped patches count as 0); fire patches of test.')
    print('reference: Sen2Fire macro-F1 ~25 (AttUNet); CaBuAr AttUNet 20 m macro-F1 71.0 (n=10)')

    if scope != TARGET_SCOPE:
        return

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def swir(img):
        x = img[..., [B12, B8, B4]].astype(np.float32) / 10000.0
        return np.clip(x / 0.4, 0, 1)

    big = max((k for k in test if k not in sel[E_PRIMARY] and 0.2 < burn[k].mean() < 0.6), key=lambda k: burn[k].mean())
    mid = [k for k in test if k in sel[E_PRIMARY] and 0.05 < burn[k].mean() < 0.3][0]
    with h5py.File(H5, 'r') as h5:
        post, pre = np.asarray(h5[mid]['post_fire']), np.asarray(h5[mid]['pre_fire'])
    cols = ['clean', 'resolution', 'inflate_area', f'matched_{E_SENSITIVITY:.2f}', f'matched_{E_PRIMARY:.2f}']
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for c, a in enumerate(cols):
        ax = axes[0, c]
        ax.imshow(labels[a][big], cmap='gray', vmin=0, vmax=1)
        ax.contour(burn[big].astype(float), levels=[0.5], colors='red', linewidths=0.6)
        ax.set_title(f'{a}\nshare {labels[a][big].mean() * 100:.0f}%  purity {purity(labels[a][big], burn[big]) * 100:.0f}%', fontsize=8)
    axes[0, 0].set_ylabel(f'large burn\n{big[:8]} ({burn[big].mean() * 100:.0f}%)', fontsize=8)
    panels = [(swir(post), 'post_fire SWIR (not used)'), (swir(pre), 'pre_fire SWIR = model input'),
              (burn[mid], 'clean perimeter'), (labels['resolution'][mid], 'no-evidence label (1 km)'),
              (labels[f'matched_{E_PRIMARY:.2f}'][mid], f'matched_{E_PRIMARY:.2f} label')]
    for c, (im, t) in enumerate(panels):
        ax = axes[1, c]
        ax.imshow(im, cmap=None if im.ndim == 3 else 'gray', vmin=0, vmax=1)
        ax.contour(burn[mid].astype(float), levels=[0.5], colors='red', linewidths=0.6)
        ax.set_title(t, fontsize=8)
    axes[1, 0].set_ylabel(f'swapped\n{mid[:8]}', fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Degradation arms: label (white), post-fire perimeter (red)', fontsize=10)
    out_png = P.out('degradation_oracle_examples.png')
    fig.tight_layout(); fig.savefig(out_png, dpi=80)
    print('saved', out_png)

if __name__ == '__main__':
    main('pooled' if '--scope' in sys.argv and sys.argv[-1] == 'pooled' else None)
