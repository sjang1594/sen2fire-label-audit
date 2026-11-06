"""SegFormer with an ImageNet-pretrained MiT encoder and an N-band patch-embedding stem. Needs `transformers`."""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_VARIANT = 'nvidia/mit-b2'


def build_segformer(in_channels, num_classes=2, variant=DEFAULT_VARIANT):
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    # Setting num_labels on the config rebuilds id2label to the right length.
    cfg = SegformerConfig.from_pretrained(variant)
    cfg.num_labels = num_classes
    model = SegformerForSemanticSegmentation.from_pretrained(variant, config=cfg, ignore_mismatched_sizes=True)

    # The stem is the 3-channel Conv2d with the largest kernel. It takes the mean of the RGB filters,
    # broadcast to all input bands, as in DeepLabV3.
    stem_name, stem = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.in_channels == 3:
            if stem is None or mod.kernel_size[0] > stem.kernel_size[0]:
                stem_name, stem = name, mod
    assert stem is not None, 'could not locate the 3-channel patch-embedding stem'
    new = nn.Conv2d(in_channels, stem.out_channels, kernel_size=stem.kernel_size, stride=stem.stride,
                    padding=stem.padding, bias=stem.bias is not None)
    with torch.no_grad():
        new.weight.copy_(stem.weight.mean(dim=1, keepdim=True).expand(-1, in_channels, -1, -1))
        if stem.bias is not None:
            new.bias.copy_(stem.bias)
    parent, _, child = stem_name.rpartition('.')
    setattr(model.get_submodule(parent), child, new)
    return model


class SegFormer(nn.Module):
    """ forward(x) returns logits upsampled to the input size 
        (Note: the Hugging Face model outputs H/4)."""

    def __init__(self, in_channels, num_classes=2, variant=DEFAULT_VARIANT):
        super().__init__()
        self.model = build_segformer(in_channels, num_classes, variant)

    def forward(self, x):
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
