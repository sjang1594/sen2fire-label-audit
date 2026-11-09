"""CaBuAr patches from the 512 x 512 HDF5 file: fold split and Dataset."""
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import augment_pair

POST_KEY, PRE_KEY, MASK_KEY = 'post_fire', 'pre_fire', 'mask'
IN_CHANNELS = 12
SCALE = 10000.0
TEST_FOLD, VAL_FOLD = '0', '1'
SPLIT_SIZES = (300, 104, 130)     # train, validation, test
HDF5_NAMES = ('512x512.hdf5', 'chabud_test.h5')

def hdf5_files(root):
    files = [Path(root) / 'raw' / 'patched' / name for name in HDF5_NAMES]
    missing = [f for f in files if not f.exists()]
    assert not missing, f'missing CaBuAr files: {missing}'
    return files

def load_entries(hdf5_paths):
    entries = []
    for path in hdf5_paths:
        with h5py.File(path, 'r') as h5:
            for grp in h5.keys():
                g = h5[grp]
                if POST_KEY in g and MASK_KEY in g:
                    entries.append((str(path), grp, str(g.attrs['fold'])))
    groups = [e[1] for e in entries]
    assert len(groups) == len(set(groups)), 'duplicate group names across files'
    return entries

def fold_splits(entries):
    train = [e for e in entries if e[2].isdigit() and e[2] not in (TEST_FOLD, VAL_FOLD)]
    val = [e for e in entries if e[2] == VAL_FOLD]
    test = [e for e in entries if e[2] == TEST_FOLD]
    sizes = (len(train), len(val), len(test))
    assert sizes == SPLIT_SIZES, f'split sizes {sizes}, expected {SPLIT_SIZES}'
    return train, val, test

def read_mask(group):
    """Boolean (H, W) mask."""
    mask = np.asarray(group[MASK_KEY]) > 0
    if mask.ndim == 3:
        mask = mask[..., 0] if mask.shape[-1] < mask.shape[0] else mask[0]
    return mask

def read_image(group, key=POST_KEY):
    return torch.from_numpy((np.asarray(group[key]).astype(np.float32) / SCALE).transpose(2, 0, 1))

class H5Dataset(Dataset):
    def __init__(self, entries, *, augment):
        self.entries = entries
        self.augment = augment
        self._h5 = {}
        self._pid = None

    def __len__(self):
        return len(self.entries)

    def key(self, idx):
        return self.entries[idx][1]

    def group(self, idx):
        path, grp, _ = self.entries[idx]
        pid = os.getpid()
        if pid != self._pid:
            self._h5 = {}
            self._pid = pid
        if path not in self._h5:
            self._h5[path] = h5py.File(path, 'r')
        return self._h5[path][grp]

class CaBuArDataset(H5Dataset):
    def __getitem__(self, idx):
        g = self.group(idx)
        img, lbl = read_image(g), torch.from_numpy(read_mask(g).astype(np.int64))
        if self.augment:
            img, lbl = augment_pair(img, lbl)
        return img, lbl
