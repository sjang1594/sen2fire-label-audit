"""Gap share G and learning ceiling of the label-degradation arms (config cabuar_degrade).
Run labels in the output: E10 = cabuar_attunet / cabuar_deeplab, E11 = cabuar_degrade."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import _paths as P
from . import degradation_oracle_ceiling as do

ARMS = ['resolution', 'inflate_area', 'noevid_0.35', 'matched_0.35', 'matched_0.20']
S2F_REF = 25.0
N_BOOT = 4000

def fire_of(keys):
    return pd.Series(pd.Index(keys).str.rsplit('_', n=1).str[0], index=keys)

def fire_seed_boot(A, B, rng, n=N_BOOT):
    """macro(A) - macro(B) in pp; fires resampled jointly, then seeds per table."""
    fa, fb = fire_of(A.index), fire_of(B.index)
    fires = np.array(sorted(set(fa) | set(fb)))
    out = np.empty(n)
    for i in range(n):
        w = pd.Series(rng.choice(fires, len(fires))).value_counts()
        wa, wb = fa.map(w).fillna(0).values, fb.map(w).fillna(0).values
        sa = rng.choice(A.columns, len(A.columns)); sb = rng.choice(B.columns, len(B.columns))
        ma = (A[sa].values * wa[:, None]).sum(0).mean() / wa.sum()
        mb = (B[sb].values * wb[:, None]).sum(0).mean() / wb.sum()
        out[i] = (ma - mb) * 100
    return out, len(fires)

def main():
    r = pd.read_csv(P.run_table('cabuar_degrade', 'results.csv')).drop_duplicates(['arm', 'seed'], keep='last')
    p = pd.read_csv(P.run_table('cabuar_degrade', 'per_patch.csv'))
    e10 = P.cabuar_architectures('results').drop_duplicates(['arch', 'seed'], keep='last')
    e10 = e10[e10.arch == 'AttUNet']
    pp10 = P.cabuar_architectures('per_patch')
    pp10 = pp10[pp10.arch == 'AttUNet']
    ref = e10.macro_f1.values
    denom = 71.0 - S2F_REF
    print(f'E10 AttUNet 20 m macro: n={len(ref)} mean {ref.mean():.2f} sd {ref.std(ddof=1):.2f}')

    rng = np.random.default_rng(0)
    print('\nG per arm (pre-registered for matched_0.35 / matched_0.20; descriptive for the rest)')

    for arm in ARMS:
        a = r[r.arm == arm].macro_deg_primary.values
        g = np.array([(rng.choice(ref, len(ref)).mean() - rng.choice(a, len(a)).mean()) / denom
                      for _ in range(20000)])
        print(f'   {arm:13} G = {(ref.mean() - a.mean()) / denom:.3f}  seed bootstrap 95% '
              f'[{np.percentile(g, 2.5):.3f}, {np.percentile(g, 97.5):.3f}]  P(G >= 2/3) = {(g >= 2 / 3).mean():.3f}')
    print("\nPost hoc: G with a fire + seed bootstrap (E10 clean-115 set vs the arm's primary set)")

    T10 = pp10[pp10.coverage > do.MIN_COV].drop_duplicates(['seed', 'group'], keep='last').pivot(
        index='group', columns='seed', values='f1')

    for arm in ARMS:
        A = p[(p.arm == arm) & (p.cov_eval > do.MIN_COV)].pivot(index='key', columns='seed', values='f1_degraded')
        d, nf = fire_seed_boot(T10, A, rng)
        g = d / denom
        print(f'   {arm:13} G = {(T10.mean().mean() - A.mean().mean()) * 100 / denom:.3f}  fire+seed 95% '
              f'[{np.percentile(g, 2.5):.3f}, {np.percentile(g, 97.5):.3f}]  P(G >= 2/3) = {(g >= 2 / 3).mean():.3f}  '
              f'({nf} fires)')

    m = r[r.arm == 'matched_0.35'].macro_deg_primary.mean()
    print('   G(matched_0.35) vs the Sen2Fire reference: ' + ', '.join(
        f'{s}: {(ref.mean() - m) / (ref.mean() - s):.3f}' for s in (23, 24, 25, 26, 27)))

    print('\nPost hoc: all-fire predictor, mean patch F1 (primary set: label share > 0.1%)')

    q = p[(p.seed == 42) & (p.cov_eval > do.MIN_COV)].copy()
    q['blank'] = 2 * q.cov_eval / (1 + q.cov_eval)
    s2f = np.asarray(do.sen2fire_shares())

    for arm in ARMS:
        a = q[q.arm == arm]
        model = r[r.arm == arm].macro_deg_primary.mean()
        by = a.groupby('swapped').blank.mean()
        print(f'   {arm:13} all-fire {a.blank.mean() * 100:5.1f}  (unswapped {by.get(0, np.nan) * 100:5.1f}, '
              f'swapped {by.get(1, np.nan) * 100:5.1f})  trained model {model:5.1f}')
    print(f'   Sen2Fire test (n={len(s2f)}, mean label share {s2f.mean():.3f}): all-fire '
          f'{(2 * s2f / (1 + s2f)).mean() * 100:.1f}; Sen2Fire AttUNet ~{S2F_REF:.0f}')
    print(f'   G = 2/3 corresponds to M = {71.0 - 2 / 3 * denom:.1f}')

    print('\nPost hoc: mean patch F1 by swapped flag (seed mean)')

    for col, cov in (('f1_degraded', 'cov_eval'), ('f1_clean', 'cov_clean')):
        t = p[p[cov] > do.MIN_COV].groupby(['arm', 'swapped'])[col].mean().unstack() * 100
        print(f'   {col}\n' + t.round(1).reindex(ARMS).to_string())
    print('\nPixel precision / recall on degraded labels, pixel F1 on clean labels (seed mean)')
    print(r.groupby('arm')[['prec_deg', 'rec_deg', 'f1_deg', 'f1_clean']].mean().reindex(ARMS).round(1).to_string())

    print('\nSeed-level learning ceiling (clean labels, unswapped clean fire patches)')
    for arm in ARMS:
        keys = set(p[(p.arm == arm) & (p.swapped == 0) & (p.cov_clean > do.MIN_COV)].key)
        e = pp10[pp10.group.isin(keys) & (pp10.coverage > do.MIN_COV)].groupby('seed').f1.mean() * 100
        a = r[r.arm == arm].macro_clean_unswapped.values
        t = stats.ttest_ind(e.values, a, equal_var=False)
        A = p[(p.arm == arm) & p.key.isin(keys)].pivot(index='key', columns='seed', values='f1_clean')
        d, nf = fire_seed_boot(T10[T10.index.isin(keys)], A, rng)
        print(f'   {arm:13} 20 m {e.mean():.1f} (n={len(e)}, {len(keys)} patches)  arm {a.mean():.1f}  '
              f'diff {e.mean() - a.mean():+.1f}  Welch p = {t.pvalue:.2g}  '
              f'fire+seed 95% [{np.percentile(d, 2.5):+.1f}, {np.percentile(d, 97.5):+.1f}] ({nf} fires)')
    print(f'\nGPU-hours {r.gpu_hours.sum():.2f}')

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
