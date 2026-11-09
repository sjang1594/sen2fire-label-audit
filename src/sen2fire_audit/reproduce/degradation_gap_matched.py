"""Gap share G with the matched Sen2Fire reference (config sen2fire_attunet_cabuar_impl).
Run labels in the output: E7 = sen2fire_attunet_oversample baseline, E10 = cabuar_attunet / cabuar_deeplab,
E11 = cabuar_degrade, E13 = sen2fire_attunet_cabuar_impl."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import _paths as P
from . import degradation_oracle_ceiling as do
from .degradation_gap_share import fire_of

C_REG, S_REG, ARM = 71.0, 25.0, 'matched_0.35'
N_BOOT = 4000

def s2f_table(pp):
    """patch x seed F1 on the 99 patches behind S = 25.0."""
    q = pp[pp.coverage > do.MIN_COV].drop_duplicates(['seed', 'patch'], keep='last')
    t = q.pivot(index='patch', columns='seed', values='f1')
    assert len(t) == 99 and not t.isna().any().any(), (len(t), int(t.isna().sum().sum()))
    return t

def boot_mean(T, groups, rng):
    """One replicate of the macro (in %): clusters resampled, then seeds."""
    cl = np.array(sorted(set(groups)))
    w = groups.map(pd.Series(rng.choice(cl, len(cl))).value_counts()).fillna(0).values
    s = rng.choice(T.columns, len(T.columns))
    return (T[s].values * w[:, None]).sum(0).mean() / w.sum() * 100

def joint_boot_cm(T10, A, rng):
    """CaBuAr: fires resampled jointly for both tables, then seeds within each. Returns (C*, M*)."""
    fa, fb = fire_of(T10.index), fire_of(A.index)
    fires = np.array(sorted(set(fa) | set(fb)))
    w = pd.Series(rng.choice(fires, len(fires))).value_counts()
    wa, wb = fa.map(w).fillna(0).values, fb.map(w).fillna(0).values
    sa, sb = rng.choice(T10.columns, len(T10.columns)), rng.choice(A.columns, len(A.columns))
    c = (T10[sa].values * wa[:, None]).sum(0).mean() / wa.sum() * 100
    m = (A[sb].values * wb[:, None]).sum(0).mean() / wb.sum() * 100
    return c, m

def main(dry_run):
    census = pd.read_csv(P.census('location.csv')).set_index('patch')['loc']
    blocks = pd.read_csv(P.analysis('score_by_census_location_per_patch.csv')).set_index('patch')['block']
    p7 = pd.read_csv(P.run_table('sen2fire_attunet_oversample', 'per_patch.csv'))
    p7 = p7[p7.arm == 'baseline']
    if dry_run:
        print('DRY RUN: the E7 baseline stands in for E13. Numbers below are a code check, not E13.\n')
        p13 = p7.assign(pred_share=np.nan)
    else:
        p13 = pd.read_csv(P.run_table('sen2fire_attunet_cabuar_impl', 'per_patch.csv'))
    T13, T7 = s2f_table(p13), s2f_table(p7)
    s13, s7 = T13.mean() * 100, T7.mean() * 100
    print(f"S' = {s13.mean():.2f} (n={len(s13)} seeds, sd {s13.std(ddof=1):.2f}); "
          f"E7 baseline on the same 99 patches = {s7.mean():.2f} (n={len(s7)})")
    common = s13.index.intersection(s7.index)
    d = s13[common] - s7[common]
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    print(f"S' - E7, seed-paired (n={len(d)}): {d.mean():+.2f} pp [{d.mean() - h:+.2f}, {d.mean() + h:+.2f}]")

    r = pd.read_csv(P.run_table('cabuar_degrade', 'results.csv')).drop_duplicates(['arm', 'seed'], keep='last')
    p = pd.read_csv(P.run_table('cabuar_degrade', 'per_patch.csv'))
    pp10 = P.cabuar_architectures('per_patch')
    pp10 = pp10[pp10.arch == 'AttUNet']
    T10 = pp10[pp10.coverage > do.MIN_COV].drop_duplicates(['seed', 'group'], keep='last').pivot(
        index='group', columns='seed', values='f1')
    A = p[(p.arm == ARM) & (p.cov_eval > do.MIN_COV)].pivot(index='key', columns='seed', values='f1_degraded')
    M = r[r.arm == ARM].macro_deg_primary.mean()
    Sp = s13.mean()
    print(f'\nM ({ARM}) = {M:.2f}; C registered {C_REG}, C on the clean-115 table {T10.mean().mean() * 100:.2f}')

    g_orig = (C_REG - M) / (C_REG - S_REG)
    g_match = (C_REG - M) / (C_REG - Sp)
    rng = np.random.default_rng(0)
    gb = pd.Series(blocks.reindex(T13.index).values, index=T13.index)
    assert gb.notna().all()

    orig, match, match_cfix = np.empty(N_BOOT), np.empty(N_BOOT), np.empty(N_BOOT)

    for i in range(N_BOOT):
        c, m = joint_boot_cm(T10, A, rng)
        s = boot_mean(T13, gb, rng)
        orig[i] = (c - m) / (C_REG - S_REG)
        match[i] = (c - m) / (c - s)
        match_cfix[i] = (c - m) / (C_REG - s)

    q = lambda x: f'[{np.percentile(x, 2.5):.3f}, {np.percentile(x, 97.5):.3f}]'
    print(f'G_original = {g_orig:.3f}  fire+seed 95% {q(orig)}  (registered; paper 0.64 [0.43, 0.89])')
    print(f'G_matched  = {g_match:.3f}  joint 95% {q(match)}  P(G >= 2/3) = {(match >= 2 / 3).mean():.3f}')
    print(f'           sensitivity, denominator 71.0 - S\'*: {q(match_cfix)}')
    print(f'           {len(set(gb))} Sen2Fire blocks, {len(set(fire_of(T10.index)) | set(fire_of(A.index)))} CaBuAr fires, '
          f'{N_BOOT} draws')

    if not dry_run:
        q2 = p13[p13.patch.isin(census.index) & (p13.coverage > do.MIN_COV)].copy()
        q2['loc'] = q2.patch.map(census)
        g = q2.groupby(['seed', 'loc']).agg(f1=('f1', 'mean'), ps=('pred_share', 'mean')).unstack().mean() * 100
        print('\nSecondary (seed mean): ' + '  '.join(
            f"{c}: F1 {g[('f1', c)]:.1f}, predicted share {g[('ps', c)]:.1f}%" for c in ('inside', 'near', 'none')))
        # registered: share >= 5 % and F1 near 0 -> implicated, share <= 2 % -> not; F1_NEAR0 was set
        # after the result (the verdict is the same for any cutoff below 22)
        F1_NEAR0 = 5.0
        ps, f1 = g[('ps', 'none')], g[('f1', 'none')]
        v = ('implicated' if ps >= 5 and f1 < F1_NEAR0 else 'not implicated' if ps <= 2 else
             'inconclusive (positive clause not met)' if ps >= 5 else 'inconclusive')
        print(f'"nothing within 1 km": predicted share {ps:.1f}%, F1 {f1:.1f} (E7: 0.9%, F1 0.0; implicated if share >= 5 '
              f'and F1 near 0 [< {F1_NEAR0:.0f}], not if share <= 2) -> {v}')

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main('--dry-run' in sys.argv)
