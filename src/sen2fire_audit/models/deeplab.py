"""DeepLabV3-ResNet50 with an N-band first conv. init: 'coco' (torchvision DEFAULT,
COCO_WITH_VOC_LABELS_V1), 'imagenet' (backbone only), 'scratch', or None (architecture only)."""

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

INITS = ('coco', 'imagenet', 'scratch')

def _base_model(init):
    if init == 'coco':
        return deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    if init == 'imagenet':
        return deeplabv3_resnet50(weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V1, aux_loss=True)
    if init == 'scratch':
        return deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=True)
    if init is None:     # architecture only, for loading a trained state dict
        return deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=True)
    raise ValueError(f'init must be one of {INITS} or None, got {init!r}')

class DeepLabV3(nn.Module):
    """Returns the main logits; the first conv starts from the mean RGB filter for every band."""

    def __init__(self, in_channels, num_classes=2, init='coco'):
        super().__init__()
        self.model = _base_model(init)
        old = self.model.backbone.conv1
        new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=old.bias is not None)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True).expand(-1, in_channels, -1, -1))
        self.model.backbone.conv1 = new
        self.model.classifier[4] = nn.Conv2d(self.model.classifier[4].in_channels, num_classes, 1)
        self.model.aux_classifier[4] = nn.Conv2d(self.model.aux_classifier[4].in_channels, num_classes, 1)

    def forward(self, x):
        return self.model(x)['out']
