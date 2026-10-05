import numpy as np
import pywt

from ._common import load_gray, not_applicable, result


def detect_wavelet_anomalies(image_path):
    """Diagonal-subband energy and horizontal/vertical imbalance of a Haar wavelet transform."""
    gray = load_gray(image_path)
    if min(gray.shape) < 16:
        return not_applicable("Image too small")

    _, (lh, hl, hh) = pywt.dwt2(gray.astype(np.float32), "db1")
    lh_mean, hl_mean, hh_mean = (float(np.mean(np.abs(b))) for b in (lh, hl, hh))

    anisotropy = abs(lh_mean - hl_mean) / (lh_mean + hl_mean + 1e-6)
    energy_ratio = hh_mean / (lh_mean + hl_mean + 1e-6)
    score = min(anisotropy * 5.0, 1.0) * 0.4 + min(energy_ratio * 2.0, 1.0) * 0.6
    return result(score, hh_energy_ratio=energy_ratio, anisotropy_index=anisotropy)
