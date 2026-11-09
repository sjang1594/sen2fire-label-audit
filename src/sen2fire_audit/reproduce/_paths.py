"""Data, result and output roots of the reproduction commands; every released file name is defined here."""
import os
from pathlib import Path
import pandas as pd

_ROOTS = {
    'data': Path(os.environ.get('SEN2FIRE_AUDIT_DATA', 'data')),
    'results': Path(os.environ.get('SEN2FIRE_AUDIT_RESULTS', 'results')),
    'out': None,
}

def configure(data=None, results=None, out=None):
    if data is not None:
        _ROOTS['data'] = Path(data)
    if results is not None:
        _ROOTS['results'] = Path(results)
    if out is not None:
        _ROOTS['out'] = Path(out)

def sen2fire():
    return _ROOTS['data'] / 'sen2fire'

def cabuar_h5():
    return _ROOTS['data'] / 'cabuar' / 'raw' / 'patched' / '512x512.hdf5'

def results(*parts):
    return _ROOTS['results'].joinpath(*parts)

def out(*parts):
    root = _ROOTS['out'] if _ROOTS['out'] is not None else _ROOTS['results'] / 'analysis'
    p = root.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def census(name):
    """census/: key.csv, first_author.csv (99 patches), annotator_a.csv, annotator_b.csv (25 shared), location.csv."""
    return results('census', name)

def analysis(name):
    """analysis/: tables written by the reproduction commands and read by later ones."""
    return results('analysis', name)

def run_table(config, name):
    """<config>/: training results of one configuration (results.csv, per_patch.csv, per_patch/)."""
    return results(config, name)

def cabuar_architectures(kind):
    """The CaBuAr Attention U-Net and DeepLabV3 runs, as one table (kind: 'results' or 'per_patch')."""
    return pd.concat([pd.read_csv(run_table('cabuar_attunet', f'{kind}.csv')),
                      pd.read_csv(run_table('cabuar_deeplab', f'{kind}.csv'))], ignore_index=True)
