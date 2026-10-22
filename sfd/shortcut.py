"""Family C: a synthetic shortcut cue (class-specific 16x16 patch in a fixed corner).

Training data: the cue matches the true label with probability p_aligned (0.95); otherwise it is the cue of a
random WRONG class, so the real tissue signal stays learnable. The cue is pasted AFTER the GPU augmentation, so it
always sits at the same position. Test states on the same images: aligned (cue = label), removed (no cue) and
reversed (cue = derangement[label], a fixed seeded permutation without fixed points).
"""
from __future__ import annotations

import numpy as np

# 9 well-separated colours (none of them H&E pink/purple-like), one per class
COLOURS = np.array([(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
                    (70, 240, 240), (0, 0, 0), (255, 255, 255), (128, 128, 0)], dtype=np.uint8)


def make_palette(n_classes: int = 9, px: int = 16) -> np.ndarray:
    """uint8 [n_classes, px, px, 3]: cue k = colour k, with pattern k % 3 (solid / 4-px checker / 4-px stripes)
    drawn with a contrasting grey, so cues differ in colour AND texture."""
    assert n_classes <= len(COLOURS)
    pal = np.zeros((n_classes, px, px, 3), np.uint8)
    yy, xx = np.mgrid[0:px, 0:px]
    masks = [np.zeros((px, px), bool), ((yy // 4 + xx // 4) % 2) == 1, (yy // 4) % 2 == 1]
    for k in range(n_classes):
        pal[k] = COLOURS[k]
        alt = np.array([110, 110, 110], np.uint8) if COLOURS[k].mean() > 100 else np.array([200, 200, 200], np.uint8)
        pal[k][masks[k % 3]] = alt
    return pal


def make_palette_v3(n_classes: int = 9, px: int = 32, frame_px: int = 2) -> np.ndarray:
    """Family C v2 / S3 cues: colour k AND a texture unique to class k (9 distinct patterns), inside a frame_px
    mid-grey safety frame, so the cue is unambiguous against tissue. Shape [n, px + 2*frame, px + 2*frame, 3]."""
    assert n_classes <= len(COLOURS)
    yy, xx = np.mgrid[0:px, 0:px]
    s = max(2, px // 8)
    pats = [np.zeros((px, px), bool),                               # solid
            ((yy // s + xx // s) % 2) == 1,                         # checker
            (yy // s) % 2 == 1,                                     # horizontal stripes
            (xx // s) % 2 == 1,                                     # vertical stripes
            ((yy + xx) // s) % 2 == 1,                              # diagonal stripes
            ((yy % (2 * s)) < s // 2 + 1) & ((xx % (2 * s)) < s // 2 + 1),   # dots
            (np.abs(yy - px / 2) < s) | (np.abs(xx - px / 2) < s),          # cross
            np.abs(np.hypot(yy - px / 2, xx - px / 2) - px / 3) < s,         # ring
            ((yy // (2 * s) + xx // (2 * s)) % 2) == 1]             # coarse checker
    P = px + 2 * frame_px
    pal = np.full((n_classes, P, P, 3), 128, np.uint8)
    for k in range(n_classes):
        cue = np.empty((px, px, 3), np.uint8)
        cue[:] = COLOURS[k]
        alt = np.array([110, 110, 110], np.uint8) if COLOURS[k].mean() > 100 else np.array([200, 200, 200], np.uint8)
        cue[pats[k]] = alt
        pal[k, frame_px:frame_px + px, frame_px:frame_px + px] = cue
    return pal


def palette_for(setting: dict, n_classes: int = 9) -> np.ndarray:
    """Palette for a shortcut setting dict {style: v1|v3, cue_px, frame_px}."""
    if setting.get("style", "v1") == "v3":
        return make_palette_v3(n_classes, setting["cue_px"], setting.get("frame_px", 2))
    return make_palette(n_classes, setting["cue_px"])


def derangement(n: int, seed: int) -> np.ndarray:
    """Seeded permutation of 0..n-1 with no fixed point (the 'reversed' cue mapping)."""
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def cue_assignment(labels: np.ndarray, p_aligned: float, n_classes: int, seed: int) -> np.ndarray:
    """Per-image cue id: the label with prob p_aligned, else a uniformly drawn wrong class (fixed per image)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    cue = labels.copy()
    wrong = rng.random(len(labels)) >= p_aligned
    shift = rng.integers(1, n_classes, len(labels))                    # 1..n-1 -> never the label
    cue[wrong] = (labels[wrong] + shift[wrong]) % n_classes
    return cue.astype(np.int64)


def state_cues(labels: np.ndarray, state: str, perm: np.ndarray) -> np.ndarray:
    """Cue ids for a test state: aligned -> label, removed -> -1 (no cue), reversed -> perm[label]."""
    labels = np.asarray(labels)
    if state == "aligned":
        return labels.astype(np.int64)
    if state == "removed":
        return np.full(len(labels), -1, np.int64)
    if state == "reversed":
        return perm[labels].astype(np.int64)
    raise ValueError(state)


def apply_cue_uint8(imgs: np.ndarray, cue_ids: np.ndarray, palette: np.ndarray, offset: int) -> np.ndarray:
    """Copy of uint8 NHWC images with the cue pasted at (offset, offset); cue id -1 leaves the image untouched."""
    out = np.array(imgs, copy=True)
    px = palette.shape[1]
    m = np.asarray(cue_ids) >= 0
    out[m, offset:offset + px, offset:offset + px] = palette[np.asarray(cue_ids)[m]]
    return out


def apply_cue_tensor(x, cue_ids, palette_t, offset: int):
    """In-place on a float NCHW tensor in [0,1] (after augmentation, before normalisation).
    palette_t: float tensor [n_classes, 3, px, px] on the same device."""
    px = palette_t.shape[-1]
    m = cue_ids >= 0
    if m.any():
        x[m, :, offset:offset + px, offset:offset + px] = palette_t[cue_ids[m]]
    return x
