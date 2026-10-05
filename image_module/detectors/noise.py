import cv2
import numpy as np

from ._common import load_gray, not_applicable, result


def detect_noise(image_path):
    """Noise level and how unevenly it is spread across a 6x6 grid.

    A spliced region often carries noise from a different camera or processing
    chain, so block-to-block noise levels differ more than in an untouched photo.
    """
    gray = load_gray(image_path)
    h, w = gray.shape
    if h < 24 or w < 24:
        return not_applicable("Image too small")

    noise = cv2.absdiff(gray, cv2.GaussianBlur(gray, (5, 5), 0)).astype(np.float32)
    global_score = noise.mean() / 255.0

    block_h, block_w = max(h // 6, 1), max(w // 6, 1)
    levels = np.array([
        noise[r:r + block_h, c:c + block_w].mean()
        for r in range(0, h - block_h + 1, block_h)
        for c in range(0, w - block_w + 1, block_w)
    ])
    variance_score = min(float(levels.std()) / 15.0, 1.0)
    return result(global_score * 0.4 + variance_score * 0.6,
                  mean_noise=noise.mean(), block_spread=levels.std())
