"""Block bootstrap of every companion-audit stratum (Sen2Fire 4 x 4-patch blocks, CaBuAr fires).
Writes audit_strata_intervals.csv and audit_strata_intervals_cabuar_paired.csv."""
import sys

import numpy as np
import pandas as pd

from . import _paths as P

from scipy.stats import wilcoxon


sys.stdout.reconfigure(encoding='utf-8')
N_BOOT = 5000

def boot(g, unit, col, rng):
    agg = g.groupby(unit)[[col, 'n']].sum()
    num, den = agg[col].to_numpy(float), agg['n'].to_numpy(float)
    idx = rng.integers(0, len(den), size=(N_BOOT, len(den)))
    b = num[idx].sum(1) / den[idx].sum(1) * 100
    return np.percentile(b, 2.5), np.percentile(b, 97.5), len(den)

s2 = pd.read_csv(P.analysis('audit_rule_validation_counts_sen2fire.csv'))
s2 = s2[~s2.clear_only].copy()
rc = s2.patch.str.extract(r'patch_(\d+)_(\d+)').astype(int)
s2['block'] = (rc[0] // 4).astype(str) + '_' + (rc[1] // 4).astype(str)
s2['unit_patch'] = s2.patch
cb = pd.read_csv(P.analysis('audit_rule_validation_counts_cabuar.csv'))
cb = cb[~cb.clear_only].copy()
cb['block'] = cb.patch.str.replace(r'_\d+$', '', regex=True)
cb['unit_patch'] = cb.patch

rows = []
for ds, d, strata in (('sen2fire', s2, ['nonfire_le1km', 'nonfire_1to3km', 'nonfire_gt3km',
                                         'nonfire_firefree_patch']),
                      ('cabuar', cb, ['burn_label', 'unburned'])):
    for st in strata:
        g = d[(d.stratum == st) & (d.n > 0)]
        for col, name in (('sup_noB12', 'burn evidence'), ('veg_noB12', 'healthy vegetation')):
            pt = g[col].sum() / g.n.sum() * 100
            blo, bhi, nb = boot(g, 'block', col, np.random.default_rng(0))
            plo, phi, npch = boot(g, 'unit_patch', col, np.random.default_rng(0))
            rows.append(dict(dataset=ds, stratum=st, rule=name, pct=round(pt, 2), patches=npch,
                             units=nb, block_lo=round(blo, 2), block_hi=round(bhi, 2),
                             patch_lo=round(plo, 2), patch_hi=round(phi, 2)))

out = pd.DataFrame(rows)
out.to_csv(P.out('audit_strata_intervals.csv'), index=False)
print(out.to_string(index=False))

# CaBuAr per patch: burn-evidence share inside vs outside the perimeters, paired

cb['r'] = cb.sup_noB12 / cb.n
pp = cb[cb.n > 0].pivot_table(index='patch', columns='stratum', values='r').dropna(subset=['burn_label', 'unburned'])
hi = int((pp.burn_label > pp.unburned).sum())
res = dict(patches=len(pp), inside_higher=hi, median_inside=round(pp.burn_label.median() * 100, 1),
           median_outside=round(pp.unburned.median() * 100, 1), wilcoxon_p=wilcoxon(pp.burn_label, pp.unburned).pvalue)
pd.DataFrame([res]).to_csv(P.out('audit_strata_intervals_cabuar_paired.csv'), index=False)
print(res)
