# Training

```python
from sen2fire_audit.experiment import run_experiment
run_experiment('cabuar_attunet', data_dir='data/cabuar', out_dir='runs', device='cuda', seeds=[42])
```

Results go to `<out_dir>/<config>/`; finished runs are skipped. All runs use Adam (1e-4; pretrained
backbones 1e-5) and early stopping on validation loss (patience 10).

| Config | Runs | Model | Loss | Batch | Epochs (max) |
|---|---:|---|---|---:|---:|
| `sen2fire_attunet_dice` | 24 | Attention U-Net | Dice (batch) | 8 | 100 |
| `sen2fire_attunet_combo` | 39 | Attention U-Net | weighted CE + Dice | 8 | 50 |
| `sen2fire_unet_wce` | 27 | U-Net | weighted CE | 8 | 50 |
| `sen2fire_deeplab_dice` | 9 | DeepLabV3 (COCO) | Dice (batch) | 4 | 100 |
| `sen2fire_deeplab_combo` | 3 | DeepLabV3 (COCO) | weighted CE + Dice | 4 | 50 |
| `sen2fire_segformer_dice` | 9 | SegFormer MiT-B2 | Dice (batch) | 2 | 100 |
| `sen2fire_attunet_dropband` | 50 | Attention U-Net, bands removed | Dice (batch) | 8 | 100 |
| `sen2fire_attunet_oversample` | 30 | Attention U-Net, oversampling | Dice (batch) | 8 | 100 |
| `sen2fire_attunet_fireonly` | 8 | Attention U-Net, fire patches only | Dice (batch) | 8 | 100 |
| `sen2fire_attunet_cabuar_impl` | 10 | Attention U-Net, CaBuAr implementation | Dice (per image) | 4 | 100 |
| `sen2fire_deeplab_init` | 9 | DeepLabV3, three initializations | Dice (batch) | 4 | 100 |
| `cabuar_attunet` | 10 | Attention U-Net | Dice (per image) | 4 | 100 |
| `cabuar_deeplab` | 10 | DeepLabV3 (COCO) | Dice (per image) | 4 | 100 |
| `cabuar_deeplab_init` | 30 | DeepLabV3, three initializations | Dice (per image) | 4 | 100 |
| `cabuar_degrade` | 25 | Attention U-Net, degraded labels | Dice (per image) | 4 | 100 |
| `cabuar_degrade_negatives` | 10 | Attention U-Net, degraded labels + empty patches | Dice (per image) | 4 | 100 |

Dice (batch) is computed over the batch for both classes; Dice (per image) per image for the fire class.
`legacy_id` in each config names the original notebook experiment.
