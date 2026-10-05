import cv2
import numpy as np
from scipy.stats import kurtosis

from ._common import load_gray, not_applicable, result


def detect_adversarial_noise(image_path):
    """Kurtosis and spectral spikes of the Laplacian (high-pass) residual.

    Structured, non-Gaussian high-frequency patterns (including the periodic
    grids some generators leave) show up as heavy tails and sharp peaks in the
    spectrum of this residual.
    """
    gray = load_gray(image_path)
    if min(gray.shape) < 32:
        return not_applicable("Image too small")

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    k = abs(float(kurtosis(lap.ravel())))
    k_score = min(k / 50.0, 1.0) if np.isfinite(k) else 0.0

    mag = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(lap))) + 1e-6)
    f_gap = float(mag.max() - mag.mean())
    f_score = min(f_gap / 100.0, 1.0)

    return result(k_score * 0.5 + f_score * 0.5, noise_kurtosis=k, freq_gap=f_gap)
