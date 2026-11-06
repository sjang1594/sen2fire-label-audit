"""Training losses"""
import torch
import torch.nn.functional as F


def dice_batch_both_classes(logits, targets, smooth=1e-6):
    """Sen2Fire experiments: both classes, summed over the mini-batch and pixels, mean over the classes."""
    probs = F.softmax(logits, dim=1)
    targets_oh = F.one_hot(targets, 2).permute(0, 3, 1, 2).float()
    inter = (probs * targets_oh).sum(dim=(0, 2, 3))
    union = (probs + targets_oh).sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def dice_per_image_fire(logits, targets, smooth=1e-6):
    """CaBuAr experiments: fire class only, computed per image, mean over the mini-batch."""
    probs = F.softmax(logits, dim=1)[:, 1]
    t = (targets == 1).float()
    inter = (probs * t).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
    return (1 - (2 * inter + smooth) / (union + smooth)).mean()


def weighted_ce(fire_weight):
    def loss(logits, targets):
        w = torch.tensor([1.0, float(fire_weight)], device=logits.device)
        return F.cross_entropy(logits, targets, weight=w)
    return loss

def combo(fire_weight):
    """Weighted cross-entropy plus the batch Dice, used on Sen2Fire only."""
    wce = weighted_ce(fire_weight)

    def loss(logits, targets):
        return wce(logits, targets) + dice_batch_both_classes(logits, targets)
    return loss
