import cv2
import numpy as np

from ._common import load_gray, not_applicable, result


def detect_luminance_gradient(image_path):
    """Unevenness of gradient strength across a 10x10 grid plus gradient-direction entropy."""
    gray = load_gray(image_path).astype(np.float32)
    h, w = gray.shape
    if h < 20 or w < 20:
        return not_applicable("Image too small")

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    norm = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

    block_h, block_w = max(h // 10, 1), max(w // 10, 1)
    spreads = np.array([
        norm[r:r + block_h, c:c + block_w].std()
        for r in range(0, h - block_h + 1, block_h)
        for c in range(0, w - block_w + 1, block_w)
    ])
    inconsistency = (spreads.max() - spreads.mean()) / (spreads.max() + 1e-6)

    hist, _ = np.histogram(angle, bins=8, range=(0, 360), weights=magnitude)
    hist = hist / (hist.sum() + 1e-6)
    entropy = -np.sum(hist * np.log2(hist + 1e-6))
    entropy_score = max(0.0, (3.0 - entropy) / 3.0)

    return result(inconsistency * 0.6 + entropy_score * 0.4,
                  inconsistency=inconsistency, orientation_entropy=entropy)
