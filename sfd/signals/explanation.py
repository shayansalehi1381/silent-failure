"""Explanation-drift signals from Grad-CAM (Selvaraju et al. 2017), predicted class.

One forward + one backward pass gives, per image: logits, 512-d features and Grad-CAMs at layer4 (PRIMARY, the
standard definition) and layer3 (sensitivity). CAMs are ReLU'd, resized to a common GRID x GRID map (bilinear) and
normalised to unit mass. Reference per class = mean CAM of correctly classified source-train images, plus a PCA
embedding with a shared Ledoit-Wolf Gaussian. Normal CAM variability is measured under label-preserving transforms
(flips, rot90, +-10 degree rotations, +-10% brightness), each CAM mapped back to the original orientation.

Scores (higher = more drift): js_raw = JS(CAM, class mean CAM); js_norm (PRIMARY) = js_raw robust-z-scored against
the transform-induced JS distribution of the predicted class; pca_maha = Mahalanobis distance in the PCA space.
"""
from __future__ import annotations

import numpy as np

GRID = 8
LAYERS = ("layer4", "layer3")
TRANSFORMS = ("hflip", "vflip", "rot90", "rot180", "rot270", "rot+10", "rot-10", "bright+10", "bright-10")


def _maps_to_grid(acts, grads, grid: int = GRID):
    import torch
    import torch.nn.functional as F
    w = grads.mean((2, 3), keepdim=True)
    cam = F.relu((w * acts).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(grid, grid), mode="bilinear", align_corners=False)[:, 0]
    s = cam.flatten(1).sum(1, keepdim=True)
    flat = cam.flatten(1)
    uniform = torch.full_like(flat, 1.0 / (grid * grid))
    return torch.where(s > 0, flat / s.clamp_min(1e-12), uniform)            # unit mass (uniform if all zero)


def gradcam_batch(model, x, target=None, layers=LAYERS, grid: int = GRID):
    """x: normalised NCHW batch on the model device. Returns dict with logits [N,C], features [N,512] and, per layer,
    unit-mass CAMs [N, GRID*GRID] for `target` (default: predicted class). Single forward/backward pass."""
    import torch
    m = model.net
    store = {}
    h = [getattr(m, ln).register_forward_hook(lambda mod, i, o, ln=ln: store.__setitem__(ln, o)) for ln in layers]
    try:
        with torch.enable_grad():
            x = x.detach().requires_grad_(False)
            feats = model.features(x)
            logits = m.fc(feats)
            t = logits.argmax(1) if target is None else target
            score = logits.gather(1, t[:, None]).sum()
            grads = torch.autograd.grad(score, [store[ln] for ln in layers])
        cams = {ln: _maps_to_grid(store[ln].detach(), g.detach(), grid).cpu().numpy() for ln, g in zip(layers, grads)}
    finally:
        for hh in h:
            hh.remove()
    return {"logits": logits.detach().float().cpu().numpy(), "features": feats.detach().half().cpu().numpy(), **cams}


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise Jensen-Shannon divergence (base 2, in [0, 1]) between unit-mass maps; q may be one row (broadcast)."""
    p = np.clip(np.asarray(p, np.float64), 1e-12, None)
    q = np.clip(np.asarray(q, np.float64), 1e-12, None)
    p, q = p / p.sum(-1, keepdims=True), q / q.sum(-1, keepdims=True)
    m = 0.5 * (p + q)
    return 0.5 * (p * np.log2(p / m)).sum(-1) + 0.5 * (q * np.log2(q / m)).sum(-1)


# ------------------------------------------------------------------ transforms ---------------
def apply_transform(x, name: str):
    """Label-preserving transform of a float NCHW batch in [0,1] (before normalisation)."""
    import torch
    if name == "hflip":
        return x.flip(3)
    if name == "vflip":
        return x.flip(2)
    if name.startswith("rot") and name[3:].isdigit():
        return torch.rot90(x, int(name[3:]) // 90, (2, 3))
    if name in ("rot+10", "rot-10"):
        return _rotate(x, 10.0 if name == "rot+10" else -10.0)
    if name.startswith("bright"):
        return (x * (1.1 if name == "bright+10" else 0.9)).clamp(0, 1)
    raise ValueError(name)


def invert_cam(cam, name: str, grid: int = GRID):
    """Map a [N, GRID*GRID] CAM of the transformed image back to the original orientation (unit mass kept)."""
    import torch
    c = torch.as_tensor(cam).view(-1, 1, grid, grid)
    if name == "hflip":
        c = c.flip(3)
    elif name == "vflip":
        c = c.flip(2)
    elif name.startswith("rot") and name[3:].isdigit():
        c = torch.rot90(c, -(int(name[3:]) // 90), (2, 3))
    elif name in ("rot+10", "rot-10"):
        c = _rotate(c, -10.0 if name == "rot+10" else 10.0)
    flat = c.flatten(1).clamp_min(0)
    return (flat / flat.sum(1, keepdim=True).clamp_min(1e-12)).numpy()


def _rotate(x, deg: float):
    import math
    import torch
    import torch.nn.functional as F
    a = math.radians(deg)
    theta = torch.tensor([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0]], dtype=x.dtype,
                         device=x.device).expand(x.shape[0], 2, 3)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


# ------------------------------------------------------------------ reference -----------------
def fit_cam_reference(cams: np.ndarray, labels: np.ndarray, pred: np.ndarray, n_classes: int, max_pc: int = 16) -> dict:
    """Class mean CAMs (correctly classified reference images) + PCA embedding with a shared Ledoit-Wolf Gaussian."""
    from sklearn.covariance import LedoitWolf
    ok = labels == pred
    means = np.stack([cams[ok & (labels == c)].mean(0) if (ok & (labels == c)).any() else cams[labels == c].mean(0)
                      for c in range(n_classes)])
    mu = cams[ok].mean(0)
    u, s, vt = np.linalg.svd(cams[ok] - mu, full_matrices=False)
    var = s ** 2 / max(1, ok.sum() - 1)
    k = int(min(max_pc, np.searchsorted(np.cumsum(var) / var.sum(), 0.95) + 1))
    comps = vt[:k]
    z = (cams[ok] - mu) @ comps.T
    cls_means = np.stack([z[labels[ok] == c].mean(0) for c in range(n_classes)])
    lw = LedoitWolf().fit(z - cls_means[labels[ok]])
    return {"mean_cam": means.astype(np.float32), "pca_mu": mu.astype(np.float32), "pca_components": comps.astype(np.float32),
            "pca_class_means": cls_means.astype(np.float32), "pca_precision": lw.precision_.astype(np.float32),
            "pca_k": k, "pca_explained": float(np.cumsum(var)[k - 1] / var.sum())}


def explanation_scores(cams: np.ndarray, pred: np.ndarray, ref: dict, tstats: dict) -> dict:
    """js_raw, js_norm (primary) and pca_maha for one layer's CAMs."""
    js = js_divergence(cams, ref["mean_cam"][pred])
    med, mad = tstats["median"][pred], tstats["mad"][pred]
    z = (cams - ref["pca_mu"]) @ ref["pca_components"].T
    d = z - ref["pca_class_means"][pred]
    maha = np.sqrt(np.maximum(np.einsum("nd,de,ne->n", d, ref["pca_precision"], d), 0))
    return {"js_raw": js, "js_norm": (js - med) / np.maximum(1.4826 * mad, 1e-6), "pca_maha": maha}

