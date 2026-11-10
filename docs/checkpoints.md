# Checkpoints

Zenodo: [10.5281/zenodo.24083276](https://doi.org/10.5281/zenodo.24083276), one archive per config
(`<config>.zip` of `<run_id>.pt`) plus `reference_single_run.zip` (Tables I–III).
[`MANIFEST.csv`](../checkpoints/MANIFEST.csv) lists every file with its SHA-256.

```python
import torch
from sen2fire_audit.models import AttentionUNet
model = AttentionUNet(12, base_ch=32)
model.load_state_dict(torch.load('cabuar_attunet_aug_ch32_seed42.pt', map_location='cpu', weights_only=True))
```

- Every config checkpoint loads with `strict=True`; build DeepLabV3 with `init=None`.
- Table I–III runs come from the original notebook and use its layer names.
- `sen2fire_unet_wce`, SWIR without augmentation: only the `E2` checkpoints exist; `E1` saved to the same
  file names and was overwritten.
- `sen2fire_segformer_dice.zip`: NVIDIA Source Code License for SegFormer (non-commercial); other
  archives CC BY 4.0.
