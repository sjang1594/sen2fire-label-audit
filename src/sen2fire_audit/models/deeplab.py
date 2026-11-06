"""DeepLabV3 (ResNet-50) with an N-band first convolution.

init='coco' loads torchvision's DEFAULT weights, COCO_WITH_VOC_LABELS_V1: backbone and ASPP head are both
pretrained for semantic segmentation, and no separate ImageNet backbone is loaded.
init='imagenet' loads a ResNet-50 IMAGENET1K_V1 backbone with a randomly initialized ASPP head.
init='scratch' uses no pretrained weights.

The auxiliary head exists in every case, so that state-dict keys match, but it receives no gradient
because forward() returns only the main output.
"""
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
    """ forward(x) returns the main logits. The first convolution takes the mean of the RGB filters,
        broadcast to all input bands, so every band starts from the same filter."""

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
