"""Random flips and 90-degree rotations, applied identically to image and label."""
import torch

def augment_pair(img, lbl):
    if torch.rand(1) > 0.5:
        img = torch.flip(img, dims=[-1])
        lbl = torch.flip(lbl, dims=[-1])
    if torch.rand(1) > 0.5:
        img = torch.flip(img, dims=[-2])
        lbl = torch.flip(lbl, dims=[-2])
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        img = torch.rot90(img, k, dims=[-2, -1])
        lbl = torch.rot90(lbl, k, dims=[-2, -1])
    return img, lbl
