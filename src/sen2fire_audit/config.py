"""Experiment configs: one YAML per experiment; each `runs` entry overrides the top-level keys and
its list values are crossed into runs."""
from dataclasses import dataclass, field
from importlib import resources
from itertools import product
from typing import Optional

import yaml

CROSSED = ('strategy', 'augment', 'base_ch', 'arm', 'seed')

@dataclass(frozen=True)
class Run:
    config: str
    legacy_id: str
    dataset: str
    model: str
    loss: str
    optimizer: str
    batch_size: int
    max_epochs: int
    macro_min_cov: float
    augment: bool
    seed: int
    strategy: Optional[str] = None
    arm: Optional[str] = None
    base_ch: Optional[int] = None
    init: Optional[str] = None
    fire_weight: Optional[float] = None
    pretrained_key: Optional[str] = None
    drop_bands: tuple = field(default=())
    oversample_target: Optional[float] = None
    burn_free_weight: Optional[float] = None
    persistent_workers: bool = False
    evidence_recall: bool = False
    degrade: bool = False
    empty_label_negatives: bool = False

    @property
    def run_id(self):
        """Unique within a config. Holds only what varies between runs, in a fixed order."""
        
        parts = [self.config]
        if self.arm:
            parts.append(self.arm)
        if self.strategy:
            parts.append(self.strategy.replace(' + ', '_').replace('-', ''))
        parts.append('aug' if self.augment else 'noaug')
        if self.base_ch:
            parts.append(f'ch{self.base_ch}')
        parts.append(f'seed{self.seed}')
        return '_'.join(parts)

def config_names():
    return sorted(p.name[:-5] for p in resources.files('sen2fire_audit.configs').iterdir() if p.name.endswith('.yaml'))

def load_config(name):
    with resources.files('sen2fire_audit.configs').joinpath(f'{name}.yaml').open(encoding='utf-8') as f:
        return yaml.safe_load(f)

def expand(name):
    """All runs of an experiment."""
    
    cfg = load_config(name)
    runs_spec = cfg.pop('runs')
    cfg_name, legacy = cfg.pop('name'), cfg.pop('legacy_id')
    assert cfg_name == name, f'{name}.yaml declares name {cfg_name}'
    runs = []
    for spec in runs_spec:
        merged = {**cfg, **spec}
        crossed = {k: merged.pop(k) for k in CROSSED if isinstance(merged.get(k), list)}
        for values in product(*crossed.values()):
            kw = {**merged, **dict(zip(crossed, values))}
            if 'drop_bands' in kw:
                kw['drop_bands'] = tuple(kw['drop_bands'])
            runs.append(Run(config=cfg_name, legacy_id=str(legacy), **kw))
    ids = [r.run_id for r in runs]
    assert len(ids) == len(set(ids)), f'{name}: duplicate run ids'
    return runs
