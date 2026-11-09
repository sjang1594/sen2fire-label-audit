"""Train and evaluate one run of an experiment configuration."""
import csv
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from . import evaluate as E
from . import losses, train as T
from .config import Run, expand
from .data import cabuar, degradation, sen2fire
from .models import AttentionUNet, DeepLabV3, UNet

SEN2FIRE_KEYS = ('prec', 'rec', 'f1', 'miou')

def build_model(run: Run, in_channels):
    if run.model == 'attunet':
        return AttentionUNet(in_channels, base_ch=run.base_ch)
    if run.model == 'unet':
        return UNet(in_channels, base_ch=run.base_ch)
    if run.model == 'deeplab':
        return DeepLabV3(in_channels, init=run.init)
    if run.model == 'segformer':
        from .models.segformer import SegFormer
        return SegFormer(in_channels)
    raise ValueError(run.model)

def build_loss(run: Run):
    if run.loss == 'dice_batch_both_classes':
        return losses.dice_batch_both_classes
    if run.loss == 'dice_per_image_fire':
        return losses.dice_per_image_fire
    if run.loss == 'weighted_ce':
        return losses.weighted_ce(run.fire_weight)
    if run.loss == 'combo':
        return losses.combo(run.fire_weight)
    raise ValueError(run.loss)

def build_optimizer(run: Run, model):
    if run.optimizer == 'uniform':
        return T.uniform_optimizer(model)
    if run.optimizer == 'discriminative':
        return T.discriminative_optimizer(model, run.pretrained_key)
    raise ValueError(run.optimizer)

def _smoke_train_paths(paths, shares, n):
    """Smoke test only: keep both kinds of patch, so that the samplers can draw."""
    fire = np.argsort(-shares)[:n // 2]
    empty = np.where(shares == 0)[0][:n - len(fire)]
    keep = sorted(fire.tolist() + empty.tolist())
    return [paths[i] for i in keep], shares[keep]

def sen2fire_data(run: Run, data_dir, num_workers, limit=None):
    if run.drop_bands or run.strategy is None:
        band_fn, in_ch = sen2fire.drop_bands(list(run.drop_bands))
        make = lambda split, aug: sen2fire.Sen2FireDataset(data_dir, split, band_fn, augment=aug)
    else:
        in_ch = sen2fire.STRATEGIES[run.strategy][2]
        make = lambda split, aug: sen2fire.Sen2FireDataset.from_strategy(data_dir, split, run.strategy, augment=aug)
    train_ds, val_ds, test_ds = make('train', run.augment), make('val', False), make('test', False)
    sampler, shares = None, None
    if run.oversample_target is not None or run.burn_free_weight is not None or limit:
        shares = sen2fire.train_fire_shares(data_dir)
    if limit:
        train_ds.paths, shares = _smoke_train_paths(train_ds.paths, shares, limit)
        val_ds.paths, test_ds.paths = val_ds.paths[:limit], test_ds.paths[:limit]
    if run.oversample_target is not None:
        sampler = sen2fire.fire_heavy_sampler(shares, run.oversample_target)
    elif run.burn_free_weight is not None:
        sampler = sen2fire.burn_free_sampler(shares, run.burn_free_weight)
    train_loader = sen2fire.make_loader(train_ds, run.batch_size, shuffle=sampler is None, sampler=sampler,
                                        num_workers=num_workers, persistent_workers=run.persistent_workers)
    val_loader = sen2fire.make_loader(val_ds, run.batch_size, shuffle=False, num_workers=num_workers,
                                      persistent_workers=run.persistent_workers)
    return train_loader, val_loader, test_ds, in_ch

def cabuar_data(run: Run, data_dir, num_workers, limit=None):
    entries = cabuar.load_entries(cabuar.hdf5_files(data_dir))
    tr, va, te = cabuar.fold_splits(entries)
    if run.degrade:
        spec = degradation.ArmSpec()
        spec.check_against(entries)
        if run.empty_label_negatives:
            train_ds, val_ds = degradation.with_empty_label_negatives(tr, va, run.arm, spec)
        else:
            train_ds = degradation.DegradeDataset(tr, run.arm, spec, augment=run.augment)
            val_ds = degradation.DegradeDataset(va, run.arm, spec, augment=False)
        test_ds = degradation.DegradeDataset(te, run.arm, spec, augment=False, return_clean=True)
    else:
        train_ds, val_ds = cabuar.CaBuArDataset(tr, augment=run.augment), cabuar.CaBuArDataset(va, augment=False)
        test_ds = cabuar.CaBuArDataset(te, augment=False)
    if limit:
        for ds in (train_ds, val_ds, test_ds):
            for member in (ds.datasets if isinstance(ds, ConcatDataset) else [ds]):
                member.entries = member.entries[:limit]
            if isinstance(ds, ConcatDataset):
                ds.cumulative_sizes = ConcatDataset.cumsum(ds.datasets)
    train_loader = sen2fire.make_loader(train_ds, run.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = sen2fire.make_loader(val_ds, run.batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_ds, cabuar.IN_CHANNELS

def score(run: Run, model, test_ds, device):
    """(summary dict, per-patch rows) of the test split."""
    if run.degrade:
        rows = E.score_degraded_arm(model, test_ds, device)
        flat = [dict(key=r['key'], swapped=r['swapped'], cov_eval=r['degraded']['coverage'], cov_clean=r['clean']['coverage'],
                     f1_degraded=r['degraded']['f1'], f1_clean=r['clean']['f1'], pred_share=r['degraded']['pred_share'])
                for r in rows]
        return E.degraded_arm_summary(rows, run.macro_min_cov), flat
    rows = E.score_patches(model, test_ds, device)
    fire = E.fire_rows(rows, run.macro_min_cov)
    summary = dict(E.pooled_metrics(rows), macro_f1=float(np.mean([r['f1'] for r in fire])) if fire else float('nan'),
                   n_fire_patches=len(fire))
    if run.evidence_recall:
        summary.update(E.recall_by_evidence(model, test_ds, device))
    return summary, rows

def _append(path, row):
    new = not path.exists() or path.stat().st_size == 0
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)

def run_one(run: Run, data_dir, out_dir, device, num_workers=2, limit=None, max_epochs=None, log=print):
    """Train and evaluate one run into <out_dir>/<config>/; a valid checkpoint is only evaluated.
    `limit` and `max_epochs` are for smoke tests."""
    out = Path(out_dir) / run.config
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out / 'per_patch').mkdir(exist_ok=True)
    final = out / 'checkpoints' / f'{run.run_id}.pt'
    T.set_seed(run.seed)
    t0 = time.time()
    data = sen2fire_data if run.dataset == 'sen2fire' else cabuar_data
    train_loader, val_loader, test_ds, in_ch = data(run, data_dir, num_workers, limit)
    model = build_model(run, in_ch).to(device)
    record = dict(best_epoch='', stop_epoch='', best_val='')
    if T.checkpoint_is_valid(final):
        log(f'{run.run_id}: valid checkpoint, evaluating only')
        model.load_state_dict(torch.load(final, map_location=device))
    else:
        # per-epoch saves stay local; the final place gets one verified copy
        local = Path(tempfile.gettempdir()) / f'{run.run_id}.pt'
        fit = T.fit(model, train_loader, val_loader, build_loss(run), build_optimizer(run, model), local, device,
                    max_epochs=max_epochs or run.max_epochs, log=log)
        T.publish_checkpoint(local, final)
        record = dict(best_epoch=fit.best_epoch, stop_epoch=fit.stop_epoch, best_val=fit.best_val)
    summary, rows = score(run, model, test_ds, device)
    with open(out / 'per_patch' / f'{run.run_id}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    row = dict(run_id=run.run_id, arm=run.arm, strategy=run.strategy, augment=run.augment, base_ch=run.base_ch, seed=run.seed,
               **summary, **record, minutes=round((time.time() - t0) / 60, 2), torch=torch.__version__,
               device=torch.cuda.get_device_name(0) if torch.device(device).type == 'cuda' else 'cpu')
    _append(out / 'results.csv', row)
    return row

def done_run_ids(out_dir, config):
    path = Path(out_dir) / config / 'results.csv'
    if not path.exists():
        return set()
    with open(path) as f:
        return {r['run_id'] for r in csv.DictReader(f)}

def run_experiment(config, data_dir, out_dir, device, seeds=None, num_workers=2, log=print):
    """All runs of a config that are not yet in its results file."""
    done = done_run_ids(out_dir, config)
    for run in expand(config):
        if run.run_id in done or (seeds is not None and run.seed not in seeds):
            continue
        log(f'=== {run.run_id}')
        row = run_one(run, data_dir, out_dir, device, num_workers=num_workers, log=log)
        log(f'    {row}')

def describe(run: Run):
    return {k: v for k, v in asdict(run).items() if v not in (None, (), False)}