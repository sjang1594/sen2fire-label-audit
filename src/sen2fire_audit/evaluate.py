"""Model scores on a test split, from one pass over the patches.

`score_patches` returns per-patch counts. Every score is then computed from those rows, so that pooled and
per-patch numbers come from the same predictions.

Which patches enter a per-patch macro-F1 is a choice the caller must state, because it changes the number:
  min_cov=0.001  more than 0.1 % labeled fire: 99 Sen2Fire test patches, 115 CaBuAr test patches
  min_cov=0      any labeled fire: 102 Sen2Fire test patches

Precision follows the device: fp16 autocast on CUDA, fp32 on CPU. Per-patch F1 can differ slightly between them.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .train import use_amp

EPS = 1e-8
B4_RED, B8_NIR, B12_SWIR2 = 3, 7, 11

@torch.no_grad()
def fire_probability(model, imgs, device):
    """Softmax probability of class 1 for a batch."""
    with torch.autocast(device_type=torch.device(device).type, enabled=use_amp(device)):
        return F.softmax(model(imgs.to(device)), dim=1)[:, 1]

def predict_fire(model, imgs, device, threshold=0.5):
    """Boolean fire masks for a batch: probability >= threshold."""
    return fire_probability(model, imgs, device) >= threshold

def patch_row(pred, label):
    """Counts for one patch. `pred` and `label` are boolean tensors of the same shape"""
    tp = int((pred & label).sum())
    fp = int((pred & ~label).sum())
    fn = int((~pred & label).sum())
    n = label.numel()
    return dict(tp=tp, fp=fp, fn=fn, n_pixels=n, coverage=(tp + fn) / n, pred_share=(tp + fp) / n,
                f1=2 * tp / (2 * tp + fp + fn + EPS))

def score_patches(model, dataset, device, threshold=0.5):
    """One row per patch of `dataset`, in dataset order. Each row also holds the patch `key`."""
    model.eval()
    rows = []
    for i in range(len(dataset)):
        img, lbl = dataset[i][:2]
        pred = predict_fire(model, img.unsqueeze(0), device, threshold)[0]
        rows.append(dict(key=dataset.key(i), **patch_row(pred, (lbl == 1).to(device))))
    return rows

def score_degraded_arm(model, dataset, device, threshold=0.5):
    """ One pass over a DegradeDataset built with return_clean=True. 
        Each patch is predicted from the arm's input,
        which is the pre-fire image for a swapped patch, 
        and scored against both the arm's label and the clean label."""
    
    model.eval()
    rows = []
    for i in range(len(dataset)):
        img, lbl, clean, swapped = dataset[i]
        pred = predict_fire(model, img.unsqueeze(0), device, threshold)[0]
        rows.append(dict(key=dataset.key(i), swapped=int(swapped),
                         degraded=patch_row(pred, (lbl == 1).to(device)),
                         clean=patch_row(pred, (clean == 1).to(device))))
    return rows

def degraded_arm_summary(rows, min_cov=0.001):
    """ Scores of one arm. Three patch sets: above `min_cov` 
        under the arm's label; above it under the clean label;
        and the latter without swapped patches."""
    
    deg, cln = [r['degraded'] for r in rows], [r['clean'] for r in rows]
    primary = [r['degraded']['f1'] for r in rows if r['degraded']['coverage'] > min_cov]
    fire = [r for r in rows if r['clean']['coverage'] > min_cov]
    unswapped = [r for r in fire if not r['swapped']]
    pd_, pc = pooled_metrics(deg), pooled_metrics(cln)
    return dict(prec_deg=pd_['prec'], rec_deg=pd_['rec'], f1_deg=pd_['f1'],
                macro_deg_primary=float(np.mean(primary)), n_deg_primary=len(primary),
                macro_deg_clean115=float(np.mean([r['degraded']['f1'] for r in fire])), n_clean115=len(fire),
                f1_clean=pc['f1'], macro_clean_all=float(np.mean([r['clean']['f1'] for r in fire])),
                macro_clean_unswapped=float(np.mean([r['clean']['f1'] for r in unswapped])),
                n_clean_unswapped=len(unswapped), n_swapped_test=sum(r['swapped'] for r in rows))

def pooled_metrics(rows):
    """Pixel precision, recall, F1 and fire-class IoU over all the given patches."""
    
    tp = sum(r['tp'] for r in rows)
    fp = sum(r['fp'] for r in rows)
    fn = sum(r['fn'] for r in rows)
    return dict(prec=tp / (tp + fp + EPS), rec=tp / (tp + fn + EPS),
                f1=2 * tp / (2 * tp + fp + fn + EPS), miou=tp / (tp + fp + fn + EPS))

def fire_rows(rows, min_cov):
    return [r for r in rows if r['coverage'] > min_cov]

def macro_f1(rows, min_cov):
    """Mean per-patch F1 over the patches with labeled-fire coverage above `min_cov`"""
    return float(np.mean([r['f1'] for r in fire_rows(rows, min_cov)]))

def evidence_strata(image):
    """ Spectral strata used for the oversampling readout, 
        from reflectance in (bands, H, W): burn evidence
        (NBR < 0 or B12 > 1), vegetation (NDVI > 0.3, NBR > 0.1, B12 < 1), 
        and ambiguous (neither).
        
        These are the rules as the training readout computed them. 
        The B12 terms are kept for that reason; on the
        labeled pixels of the Sen2Fire test split they change no pixel's class"""
    
    b4, b8, b12 = image[B4_RED], image[B8_NIR], image[B12_SWIR2]
    ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
    nbr = (b8 - b12) / (b8 + b12 + 1e-6)
    burn = (nbr < 0.0) | (b12 > 1.0)
    vegetation = (ndvi > 0.3) & (nbr > 0.1) & (b12 < 1.0)
    return burn, vegetation, ~burn & ~vegetation

def recall_by_evidence(model, dataset, device, threshold=0.5):
    """ Recall on labeled pixels of each spectral stratum, 
        over test patches with any labeled fire.
        `dataset[i]` must yield model input and label, 
        and `dataset.reflectance(i)` the 12-band image the rule reads."""
    
    model.eval()
    hit, total = np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)
    for i in range(len(dataset)):
        img, lbl = dataset[i][:2]
        gt = (lbl == 1).numpy()
        if not gt.any():
            continue
        pred = predict_fire(model, img.unsqueeze(0), device, threshold)[0].cpu().numpy()
        for s, stratum in enumerate(evidence_strata(dataset.reflectance(i))):
            mask = gt & stratum
            total[s] += int(mask.sum())
            hit[s] += int((pred & mask).sum())
    rec = hit / (total + EPS)
    return dict(burn_rec=rec[0], vegetation_rec=rec[1], ambiguous_rec=rec[2],
                burn_n=int(total[0]), vegetation_n=int(total[1]), ambiguous_n=int(total[2]))