"""CaBuAr label-degradation arms (resolution, inflate_area, noevid, matched, 20m).
Per-patch rings and swap flags are fixed in tables/degradation_arms.csv."""
from importlib import resources

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset

from .augment import augment_pair
from .cabuar import PRE_KEY, POST_KEY, H5Dataset, read_image, read_mask

BLOCK = 50     # 1 km at 20 m

ARMS = {
    'resolution': (False, None),
    'inflate_area': (True, None),
    'noevid_0.35': (False, 0.35),
    'matched_0.35': (True, 0.35),
    'matched_0.20': (True, 0.20),
    '20m': (False, None),
}

def _table(name):
    return resources.files('sen2fire_audit.data').joinpath('tables', name)

def load_arm_table():
    with _table('degradation_arms.csv').open() as f:
        return pd.read_csv(f).set_index('key')

def load_empty_label_negatives():
    """Training and validation patches whose pre-fire image is added with an empty label."""
    with _table('empty_label_negatives.csv').open() as f:
        return pd.read_csv(f)

class ArmSpec:
    """Per-patch rings and swap sets from the arm table."""
    def __init__(self, table=None):
        table = load_arm_table() if table is None else table
        self.table = table
        self.rings = table.inflate_rings.astype(int).to_dict()
        self.swap = {0.35: set(table.index[table.swap_035 == 1]), 0.20: set(table.index[table.swap_020 == 1])}

    def check_against(self, entries):
        """The table must describe exactly the patches of the file, with the frozen swap counts."""
        t = self.table
        folds = {grp: fold for _, grp, fold in entries if fold.isdigit()}
        assert set(folds) == set(t.index), 'arm table and HDF5 groups differ'
        assert all(int(folds[k]) == int(t.fold[k]) for k in folds), 'fold mismatch between arm table and HDF5'
        counts = {e: [sum(t.split[k] == s for k in self.swap[e]) for s in ('train', 'val', 'test')] for e in self.swap}
        assert counts[0.35] == [89, 30, 40] and counts[0.20] == [51, 17, 23], f'swap counts {counts}'
        assert self.swap[0.20] <= self.swap[0.35], 'the 20 % swap must be nested in the 35 % swap'

    def is_swapped(self, arm, key):
        share = ARMS[arm][1]
        return share is not None and key in self.swap[share]

    def label(self, mask, arm, key):
        """The arm's boolean label for one patch."""
        if arm == '20m':
            return np.asarray(mask) > 0
        g = coarse_grid(np.asarray(mask) > 0)
        if ARMS[arm][0]:
            for _ in range(self.rings[key]):
                g = dilate8(g)
        return upsample(g, mask.shape)

def coarse_grid(lbl):
    """1 km block-max; zero padding cannot create fire."""
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

class DegradeDataset(H5Dataset):
    """A patch under an arm. With return_clean=True (evaluation), also the clean label and the swap flag."""
    def __init__(self, entries, arm, spec=None, *, augment, return_clean=False):
        super().__init__(entries, augment=augment)
        self.arm = arm
        self.spec = spec or ArmSpec()
        self.return_clean = return_clean

    def __getitem__(self, idx):
        key = self.key(idx)
        g = self.group(idx)
        swapped = self.spec.is_swapped(self.arm, key)
        img = read_image(g, PRE_KEY if swapped else POST_KEY)
        mask = read_mask(g)
        lbl = torch.from_numpy(self.spec.label(mask, self.arm, key).astype(np.int64))
        if self.return_clean:
            return img, lbl, torch.from_numpy(mask.astype(np.int64)), swapped
        if self.augment:
            img, lbl = augment_pair(img, lbl)
        return img, lbl


class EmptyLabelDataset(H5Dataset):
    """Pre-fire image with an empty label, augmented like the positives."""
    def __getitem__(self, idx):
        img = read_image(self.group(idx), PRE_KEY)
        lbl = torch.zeros(img.shape[-2:], dtype=torch.int64)
        if self.augment:
            img, lbl = augment_pair(img, lbl)
        return img, lbl

def with_empty_label_negatives(train_entries, val_entries, arm, spec=None):
    """Train and validation sets with the empty-label negatives appended (validation negatives affect early stopping)."""
    neg = load_empty_label_negatives()
    by_group = {e[1]: e for e in train_entries + val_entries}
    neg_train = [by_group[k] for k in neg.key[neg.split == 'train']]
    neg_val = [by_group[k] for k in neg.key[neg.split == 'val']]
    assert (len(neg_train), len(neg_val)) == (59, 3), (len(neg_train), len(neg_val))
    train = ConcatDataset([DegradeDataset(train_entries, arm, spec, augment=True),
                           EmptyLabelDataset(neg_train, augment=True)])
    val = ConcatDataset([DegradeDataset(val_entries, arm, spec, augment=False),
                         EmptyLabelDataset(neg_val, augment=False)])
    return train, val
