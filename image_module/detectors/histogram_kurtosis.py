import cv2
import numpy as np
from scipy.stats import kurtosis

from ._common import load_bgr, result


def detect_histogram_anomaly(image_path):
    """Channel kurtosis and empty bins inside each channel's active range.

    Gaps in the histogram point to tone curves, resampling or quantisation after
    the image was captured. Both numbers are noisy on their own.
    """
    img = load_bgr(image_path)

    ks = []
    for i in range(3):
        k = float(kurtosis(img[:, :, i].ravel()))
        ks.append(abs(k) if np.isfinite(k) else 0.0)

    gap_total = 0.0
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256]).ravel()
        active = np.where(hist > 0)[0]
        if len(active) > 1:
            inside = hist[active[0]:active[-1] + 1]
            gap_total += float(np.sum(inside == 0)) / len(inside)

    avg_k = float(np.mean(ks))
    score = min(avg_k / 10.0, 1.0) * 0.4 + min(gap_total * 5.0, 1.0) * 0.6
    return result(score, kurtosis=avg_k, gap_ratio=gap_total)
