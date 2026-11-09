"""Sen2Fire split lists, band strategies, Dataset and training samplers."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .augment import augment_pair

BAND_NAMES = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
BAND_INDEX = {b: i for i, b in enumerate(BAND_NAMES)}
SCALE = 10000.0
FIRE_HEAVY_SHARE = 0.01      # fire-heavy: >= 1 % labeled fire
SPLIT_SCENES = {'train': ('scene1', 'scene2'), 'val': ('scene3',), 'test': ('scene4',)}
SPLIT_SIZES = {'train': 1458, 'val': 504, 'test': 504}

def _rgb(img):
    return img[[3, 2, 1]]

def _swir(img):
    return img[[11, 7, 3]]

def _nbr(img):
    nir, swir = img[7], img[11]
    return torch.stack([(nir - swir) / (nir + swir + 1e-6), img[3], img[2]])

def _ndvi(img):
    nir, red = img[7], img[3]
    return torch.stack([(nir - red) / (nir + red + 1e-6), img[3], img[2]])

def _all_indices(img):
    nir, swir, red = img[7], img[11], img[3]
    nbr = (nir - swir) / (nir + swir + 1e-6)
    ndvi = (nir - red) / (nir + red + 1e-6)
    return torch.stack([img[3], img[2], img[1], nir, swir, nbr, ndvi])

def _vanilla(img):
    return img

STRATEGIES = {
    'RGB': (_rgb, False, 3),
    'SWIR': (_swir, False, 3),
    'NBR': (_nbr, False, 3),
    'NDVI': (_ndvi, False, 3),
    'All-indices': (_all_indices, False, 7),
    'Vanilla': (_vanilla, False, 12),
    'RGB + AOD': (_rgb, True, 4),
    'SWIR + AOD': (_swir, True, 4),
    'NBR + AOD': (_nbr, True, 4),
    'NDVI + AOD': (_ndvi, True, 4),
    'All-indices + AOD': (_all_indices, True, 8),
    'Vanilla + AOD': (_vanilla, True, 13),
}

def drop_bands(names):
    """Band function that removes the named bands, and the resulting number of input channels."""
    dropped = [BAND_INDEX[b] for b in names]
    keep = [i for i in range(12) if i not in dropped]

    def band_fn(img):
        return img[keep]
    return band_fn, len(keep)

def split_paths(data_dir, split):
    """Patch files of an official split. List files written on Windows hold backslashes."""
    data_dir = Path(data_dir)
    with open(data_dir / f'{split}.txt') as f:
        return [data_dir / line.strip().replace('\\', '/') for line in f if line.strip()]

def write_split_lists(data_dir):
    """Official split lists (scenes 1-2 train, 3 validation, 4 test), sorted by file name;
    the order fixes which patch a shuffled index maps to."""
    data_dir = Path(data_dir)
    for split, scenes in SPLIT_SCENES.items():
        names = [f'{s}/{p.name}' for s in scenes for p in sorted((data_dir / s).glob('*.npz'), key=lambda x: x.name)]
        (data_dir / f'{split}.txt').write_text('\n'.join(names) + '\n')

def check_split_lists(data_dir):
    """Fail early on an incomplete or misordered dataset."""
    for split, n in SPLIT_SIZES.items():
        paths = split_paths(data_dir, split)
        assert len(paths) == n, f'{split}: {len(paths)} patches, expected {n}'
        rel = [(p.parent.name, p.name) for p in paths]
        assert rel == sorted(rel), f'{split}: list is not sorted by scene and file name'
        missing = [p for p in paths if not p.exists()]
        assert not missing, f'{split}: {len(missing)} files missing, e.g. {missing[0]}'

class Sen2FireDataset(Dataset):
    def __init__(self, data_dir, split, band_fn=_vanilla, include_aerosol=False, *, augment):
        self.paths = split_paths(data_dir, split)
        self.band_fn = band_fn
        self.include_aerosol = include_aerosol
        self.augment = augment

    @classmethod
    def from_strategy(cls, data_dir, split, strategy, *, augment):
        band_fn, include_aerosol, _ = STRATEGIES[strategy]
        return cls(data_dir, split, band_fn, include_aerosol, augment=augment)

    def __len__(self):
        return len(self.paths)

    def key(self, idx):
        return self.paths[idx].name

    def reflectance(self, idx):
        """The 12 bands as reflectance, numpy (bands, H, W)."""
        return np.load(self.paths[idx])['image'].astype(np.float32) / SCALE

    def __getitem__(self, idx):
        d = np.load(self.paths[idx])
        img = torch.from_numpy(d['image'].astype(np.float32)) / SCALE
        lbl = torch.from_numpy(d['label'].astype(np.int64))
        bands = self.band_fn(img)

        if self.include_aerosol:
            aod = torch.from_numpy(d['aerosol'].astype(np.float32)).unsqueeze(0)
            bands = torch.cat([bands, aod], dim=0)
        if self.augment:
            bands, lbl = augment_pair(bands, lbl)
        return bands, lbl

def train_fire_shares(data_dir):
    """Labeled-fire share of every training patch, in split order."""
    return np.array([(np.load(p)['label'] == 1).mean() for p in split_paths(data_dir, 'train')], dtype=np.float64)

def fire_heavy_sampler(shares, target):
    """Sampler giving fire-heavy patches about `target` of the draws (oversampling experiment)."""
    heavy = shares >= FIRE_HEAVY_SHARE
    natural = heavy.mean()
    k = (target / (1 - target)) * ((1 - natural) / natural)
    w = np.where(heavy, k, 1.0).astype(np.float64)
    return WeightedRandomSampler(torch.as_tensor(w), num_samples=len(w), replacement=True)

def burn_free_sampler(shares, burn_free_weight=0.0):
    """Sampler with a weight on training patches that hold no labeled fire (0: they are never drawn)."""
    w = np.where(shares > 0, 1.0, burn_free_weight).astype(np.float64)
    return WeightedRandomSampler(torch.as_tensor(w), num_samples=len(w), replacement=True)

def make_loader(dataset, batch_size, *, shuffle, sampler=None, num_workers=2, persistent_workers=False):
    """`shuffle` is required. persistent_workers changes the draws after the first epoch
    (used only by the DeepLabV3 initialization runs) and is ignored without workers."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                      pin_memory=True, persistent_workers=persistent_workers and num_workers > 0)
