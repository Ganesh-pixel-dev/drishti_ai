import io
import os
import uuid

import cv2
import numpy as np
from PIL import Image

from ._common import load_bgr, result


def ela_difference(bgr, quality=90):
    """Absolute per-pixel difference between the image and a JPEG re-save at `quality`.

    The re-save happens in memory at full size. Resizing first would destroy the
    compression traces this check depends on.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = np.asarray(Image.open(buf).convert("RGB"))
    return np.abs(rgb.astype(np.int16) - resaved.astype(np.int16)).astype(np.float32)


def perform_ela(image_path, quality=90, heatmap_dir=None):
    """Error level analysis. Writes a heatmap to `heatmap_dir` when one is given."""
    bgr = load_bgr(image_path)
    diff = ela_difference(bgr, quality)
    gray = diff.mean(axis=2)
    h, w = gray.shape

    mean_diff = float(gray.mean())
    global_score = min(mean_diff / 10.0, 1.0)

    # Compare an 8x8 grid of regions: a pasted region that was compressed
    # differently from its surroundings shows up as a block far from the rest.
    block_h, block_w = max(h // 8, 1), max(w // 8, 1)
    means = np.array([
        gray[r:r + block_h, c:c + block_w].mean()
        for r in range(0, h - block_h + 1, block_h)
        for c in range(0, w - block_w + 1, block_w)
    ])
    if len(means) >= 4 and means.mean() > 1e-6:
        spread_score = min(float(means.std() / means.mean()), 1.0)
        hot_ratio = float(np.mean(means > means.mean() * 1.8))
        hotspot_score = min(hot_ratio * 3.0, 1.0)
    else:
        spread_score = hotspot_score = 0.0

    top = float(gray.max())
    scaled = (gray * (255.0 / top)).astype(np.uint8) if top > 0 else gray.astype(np.uint8)
    edge_score = min(float(np.mean(cv2.Canny(scaled, 50, 150) > 0)) * 10.0, 1.0)

    score = (global_score * 0.15 + spread_score * 0.35 +
             hotspot_score * 0.30 + edge_score * 0.20)

    heatmap_path = None
    if heatmap_dir:
        os.makedirs(heatmap_dir, exist_ok=True)
        heat = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        heatmap_path = os.path.join(heatmap_dir, f"ela_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(heatmap_path, heat)

    return result(score, mean_diff=mean_diff, block_spread=spread_score,
                  heatmap_path=heatmap_path)
