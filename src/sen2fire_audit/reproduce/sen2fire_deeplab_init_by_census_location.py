"""Per-patch F1 of the DeepLabV3 initialization arms by census location, with
block-bootstrap contrasts. Writes sen2fire_deeplab_init_by_census_location.csv."""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import _paths as P

GROUPS = ('inside', 'near', 'none')
N_BOOT = 10000

def load_arms(per_patch_dir, census):
    arms = {}
    for p in sorted(Path(per_patch_dir).glob('DeepLabV3_*_seed*.csv')):
        m = re.match(r'DeepLabV3_(\w+?)_(discriminative|uniform)_Vanilla_aug_seed(\d+)\.csv', p.name)
        if not m:
            continue
        d = pd.read_csv(p).set_index('patch')
        arms.setdefault(f'{m[1]}_{m[2]}', []).append(d.loc[census.index, ['f1', 'pred_share']])
    out = {}
    for arm, runs in arms.items():
        out[arm] = dict(seeds=len(runs),
                        f1=np.mean([r.f1.values for r in runs], axis=0),
                        share=np.mean([r.pred_share.values for r in runs], axis=0),
                        seed_macro=[r.f1.mean() * 100 for r in runs])
    return out

def block_ids(patches):
    rc = pd.Series(patches).str.extract(r'patch_(\d+)_(\d+)').astype(int)
    return ((rc[0] // 4).astype(str) + '_' + (rc[1] // 4).astype(str)).values

def block_ci(diff, mask, blocks, rng):
    ub = np.unique(blocks)
    idx = {b: np.where(blocks == b)[0] for b in ub}
    draws = []
    for _ in range(N_BOOT):
        s = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
        s = s[mask[s]]
        if len(s):
            draws.append(diff[s].mean())
    return np.percentile(draws, [2.5, 97.5])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, default=P.run_table('sen2fire_deeplab_init', ''))
    ap.add_argument('--census', type=Path, default=P.census('location.csv'))
    args = ap.parse_args()

    census = pd.read_csv(args.census).set_index('patch')
    assert len(census) == 99, len(census)
    loc = census['loc'].values
    blocks = block_ids(census.index)
    arms = load_arms(args.runs / 'per_patch', census)
    assert arms, f'no per-patch CSVs in {args.runs / "per_patch"}'

    rows = []
    print(f'{"arm":24} seeds  macro   ' + '  '.join(f'{g:>6} F1 (n)  marked' for g in GROUPS))
    for arm, a in arms.items():
        line = f'{arm:24} {a["seeds"]:5}  {np.mean(a["seed_macro"]):5.1f}  '
        row = dict(arm=arm, seeds=a['seeds'], macro=np.mean(a['seed_macro']),
                   macro_sd=np.std(a['seed_macro'], ddof=1) if a['seeds'] > 1 else np.nan)
        for g in GROUPS:
            k = loc == g
            row.update({f'{g}_n': int(k.sum()), f'{g}_f1': a['f1'][k].mean() * 100,
                        f'{g}_marked': a['share'][k].mean() * 100})
            line += f'  {a["f1"][k].mean() * 100:6.1f} ({k.sum():2})  {a["share"][k].mean() * 100:5.1f}%'
        print(line)
        rows.append(row)

    rng = np.random.default_rng(0)
    print('\npaired contrasts (pp; 95% interval from the 4 x 4-patch block bootstrap)')
    names = list(arms)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            diff = (arms[x]['f1'] - arms[y]['f1']) * 100
            for g, mask in (('all 99', np.ones(len(loc), bool)), ('inside', loc == 'inside'), ('none', loc == 'none')):
                lo, hi = block_ci(diff, mask, blocks, rng)
                print(f'  {x} - {y}  {g:7} {diff[mask].mean():+6.2f}  [{lo:+.2f}, {hi:+.2f}]')
                rows.append(dict(arm=f'{x} - {y}', group=g, diff=diff[mask].mean(), block_lo=lo, block_hi=hi))
    out = P.out('sen2fire_deeplab_init_by_census_location.csv')
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print('\nwrote', out)

if __name__ == '__main__':
    main()
