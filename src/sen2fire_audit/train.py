import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def use_amp(device):
    """AMP on CUDA Only"""
    return torch.device(device).type == 'cuda'

def uniform_optimizer(model, lr=1e-4):
    return torch.optim.Adam(model.parameters(), lr=lr)

def discriminative_optimizer(model, pretrained_key, pretrained_lr=1e-5, head_lr=1e-4):
    """Lower learning rate for parameters named `pretrained_key` ('backbone', 'segformer')."""
    pretrained = [p for n, p in model.named_parameters() if pretrained_key in n]
    head = [p for n, p in model.named_parameters() if pretrained_key not in n]
    return torch.optim.Adam([{'params': pretrained, 'lr': pretrained_lr}, {'params': head, 'lr': head_lr}])

def train_epoch(model, loader, optimizer, scaler, loss_fn, device):
    model.train()
    total = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=torch.device(device).type, enabled=use_amp(device)):
            loss = loss_fn(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def eval_loss(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.autocast(device_type=torch.device(device).type, enabled=use_amp(device)):
            total += loss_fn(model(imgs), labels).item()
    return total / len(loader)

@dataclass
class FitResult:
    best_val: float
    best_epoch: int
    stop_epoch: int

def fit(model, train_loader, val_loader, loss_fn, optimizer, checkpoint, device, *, max_epochs, patience=10, log=print):
    """Train until early stopping; the best state is saved to `checkpoint` and reloaded.
    `max_epochs` is required (100 for Dice, 50 for Combo and WCE)."""
    checkpoint = Path(checkpoint)
    scaler = torch.amp.GradScaler(enabled=use_amp(device))
    best_val, best_epoch, waited, epoch = float('inf'), 0, 0, 0
    for epoch in range(1, max_epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, scaler, loss_fn, device)
        val = eval_loss(model, val_loader, loss_fn, device)
        if val < best_val:
            best_val, best_epoch, waited = val, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            waited += 1
        log(f'  epoch {epoch:3d}  train={tr:.4f}  val={val:.4f}{"  *" if best_epoch == epoch else ""}')
        if waited >= patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return FitResult(best_val, best_epoch, epoch)

def checkpoint_is_valid(path):
    """A save to a network drive can silently truncate the file."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        torch.load(path, map_location='cpu')
        return True
    except Exception as e:
        print(f'  unloadable checkpoint {path.name}: {e}')
        return False

def publish_checkpoint(local_path, final_path):
    """Copy the finished checkpoint once to its final place and verify every tensor."""
    local_path, final_path = Path(local_path), Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local_path, final_path)
    assert final_path.stat().st_size == local_path.stat().st_size, f'size mismatch after copy: {final_path}'
    a = torch.load(local_path, map_location='cpu')
    b = torch.load(final_path, map_location='cpu')
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), f'tensor mismatch after copy: {final_path}'