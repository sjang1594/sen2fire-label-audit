"""Block-size sensitivity of the spatial block bootstrap, and the CaBuAr fire-and-seed interval.
Part 1 draws from one generator in a fixed order (k = 5 has its own); part 2 repeats k = 2..6
with generator seeds 0 and 1."""
import sys

import numpy as np
import pandas as pd

from . import _paths as P


def part_one():
    rng = np.random.default_rng(0)

    pf = pd.read_csv(P.analysis('score_by_census_location_per_patch.csv'))
    rc = pf.patch.str.extract(r'patch_(\d+)_(\d+)').astype(int)
    dd = (pf['DeepLabV3'] - pf['Attention U-Net']).values * 100
    print(f'1. Sen2Fire DL - AttUNet {dd.mean():+.2f} pp (99 patches; grid rows {rc[0].min()}-{rc[0].max()}, cols {rc[1].min()}-{rc[1].max()})')
    
    for k in (1, 2, 3, 4, 6, 8):
        blk = ((rc[0] // k).astype(str) + '_' + (rc[1] // k).astype(str)).values
        ub = np.unique(blk)
        idx = [np.where(blk == b)[0] for b in ub]
        bb = [dd[np.concatenate([idx[i] for i in rng.choice(len(ub), len(ub))])].mean() for _ in range(10000)]
        print(f'   {k} x {k} blocks: {len(ub):3} blocks with fire patches, sizes {min(map(len, idx))}-{max(map(len, idx))}; '
              f'95% [{np.percentile(bb, 2.5):+.2f}, {np.percentile(bb, 97.5):+.2f}]')

    c = P.cabuar_architectures('per_patch')
    c = c[c.coverage > 0.001].copy()
    c['fire_id'] = c.group.str.rsplit('_', n=1).str[0]
    w = c.pivot_table(index=['seed', 'group'], columns='arch', values='f1').reset_index()
    w['d'] = (w['AttUNet'] - w['DeepLabV3']) * 100
    w['fire_id'] = w.group.str.rsplit('_', n=1).str[0]
    seeds, fires = w.seed.unique(), w.fire_id.unique()
    by = {(s, f): g.d.values for (s, f), g in w.groupby(['seed', 'fire_id'])}
    bb = []
    
    for _ in range(10000):
        ss, ff = rng.choice(seeds, len(seeds)), rng.choice(fires, len(fires))
        bb.append(np.mean([np.concatenate([by[(s, f)] for f in ff]).mean() for s in ss]))
    print(f'\n2. CaBuAr AttUNet - DL patch macro {w.groupby("seed").d.mean().mean():+.2f} pp ({len(fires)} fires, '
          f'{w.group.nunique()} patches, {len(seeds)} seeds); fires and seeds resampled 95% '
          f'[{np.percentile(bb, 2.5):+.2f}, {np.percentile(bb, 97.5):+.2f}]')

    # gap difference: CaBuAr fire+seed draws minus Sen2Fire 4 x 4-block draws
    blk = ((rc[0] // 4).astype(str) + '_' + (rc[1] // 4).astype(str)).values
    ub = np.unique(blk)
    idx = [np.where(blk == b)[0] for b in ub]
    s2f = [-dd[np.concatenate([idx[i] for i in rng.choice(len(ub), len(ub))])].mean() for _ in range(10000)]
    diff = np.array(bb) - np.array(s2f)
    print(f'   gap difference CaBuAr - Sen2Fire {w.groupby("seed").d.mean().mean() + dd.mean():+.2f} pp; 95% '
          f'[{np.percentile(diff, 2.5):+.2f}, {np.percentile(diff, 97.5):+.2f}] '
          f'(Sen2Fire AttUNet - DL block interval [{np.percentile(s2f, 2.5):+.2f}, {np.percentile(s2f, 97.5):+.2f}])')

    tp, fp, fn = np.array([50, 0, 5]), np.array([10, 30, 0]), np.array([20, 0, 40])
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0)
    wt = (2 * tp + fp + fn) / (2 * tp + fp + fn).sum()
    print(f'\n3. pooled {2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()):.4f} = weighted mean {(wt * f1).sum():.4f}')

    # k = 5 on its own generator, so the draws of parts 1-2 stay unchanged
    rng5 = np.random.default_rng(5)
    blk = ((rc[0] // 5).astype(str) + '_' + (rc[1] // 5).astype(str)).values
    ub = np.unique(blk)
    idx = [np.where(blk == b)[0] for b in ub]
    bb5 = [dd[np.concatenate([idx[i] for i in rng5.choice(len(ub), len(ub))])].mean() for _ in range(10000)]
    
    print(f'\n4. Sen2Fire DL - AttUNet, 5 x 5 blocks: {len(ub):3} blocks with fire patches, sizes {min(map(len, idx))}-{max(map(len, idx))}; '
          f'95% [{np.percentile(bb5, 2.5):+.2f}, {np.percentile(bb5, 97.5):+.2f}]')

def part_two():
    N_BOOT, SEEDS, REF4 = 10_000, (0, 1), (-1.27, 3.81)

    pf = pd.read_csv(P.analysis('score_by_census_location_per_patch.csv'))
    rc = pf.patch.str.extract(r'patch_(\d+)_(\d+)').astype(int)
    dd = (pf['DeepLabV3'] - pf['Attention U-Net']).values * 100
    assert len(dd) == 99
    print(f'DeepLabV3 - Attention U-Net, per-patch macro-F1: {dd.mean():+.2f} pp over {len(dd)} patches')
    uppers, lows = [], []
    for k in (2, 3, 4, 5, 6):
        block = ((rc[0] // k).astype(str) + '_' + (rc[1] // k).astype(str)).values
        ub = np.unique(block)
        idx = {b: np.where(block == b)[0] for b in ub}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            bb = [dd[np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])].mean() for _ in range(N_BOOT)]
            lo, hi = np.percentile(bb, 2.5), np.percentile(bb, 97.5)
            uppers.append(hi); lows.append(lo)
            ref = f'   (score_by_census_location: [{REF4[0]:+.2f}, {REF4[1]:+.2f}])' if k == 4 else ''
            print(f'{k}x{k}: {len(ub):2} blocks, rng {seed}: [{lo:+.2f}, {hi:+.2f}]{ref}')
    print(f'upper bound over block sizes and seeds: {min(uppers):+.2f} to {max(uppers):+.2f} pp; '
          f'every interval includes zero: {all(l < 0 < u for l, u in zip(lows, uppers))}')

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    part_one()
    print()
    part_two()
