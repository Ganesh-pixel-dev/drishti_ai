import cv2
import numpy as np

from ._common import load_bgr, not_applicable, result


def detect_cfa_artifacts(image_path):
    """Looks for the 2x2 periodicity that demosaicing leaves in the green channel.

    Camera images predict well from their neighbours at some positions of the 2x2
    grid and badly at others. Resized, regenerated or heavily re-processed images
    tend to lose that imbalance. Higher score means the pattern is weaker.
    """
    img = load_bgr(image_path)
    h, w = img.shape[:2]
    if h < 16 or w < 16:
        return not_applicable("Image too small")

    green = img[:, :, 1].astype(np.float32)
    kernel = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], dtype=np.float32)
    residual = np.abs(green - cv2.filter2D(green, -1, kernel))

    h2, w2 = (h // 2) * 2, (w // 2) * 2
    res = residual[:h2, :w2]
    pos_means = np.array([res[0::2, 0::2].mean(), res[0::2, 1::2].mean(),
                          res[1::2, 0::2].mean(), res[1::2, 1::2].mean()])
    consistency = pos_means.std() / (pos_means.max() + 1e-6)
    return result(1.0 - min(consistency * 10.0, 1.0), consistency_index=consistency)
