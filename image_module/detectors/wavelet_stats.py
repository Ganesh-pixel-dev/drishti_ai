import cv2
import numpy as np
import pywt

def detect_wavelet_anomalies(image_path):
    """
    Analyzes wavelets for statistical anomalies in high-frequency subbands.
    Deepfakes often show specific distributions in HL, LH, and HH subbands.
    """
    img = cv2.imread(image_path, 0)
    if img is None:
        return {"score": 0.0, "details": "File not found"}

    # Discrete Wavelet Transform (using Daubechies 1)
    coeffs2 = pywt.dwt2(img, 'db1')
    LL, (LH, HL, HH) = coeffs2

    # Feature: Mean and Variance of coefficients
    # In natural images, HH (diagonal high frequency) has very low energy and specific distribution
    lh_mean = np.mean(np.abs(LH))
    hl_mean = np.mean(np.abs(HL))
    hh_mean = np.mean(np.abs(HH))

    lh_std = np.std(LH)
    hl_std = np.std(HL)
    hh_std = np.std(HH)

    # Forgery indicators: 
    # 1. Unusually high energy in HH subband (sharpening or AI synthesis noise)
    # 2. Inconsistency between LH and HL (non-isotropic editing/stretching)
    
    anisotropy = abs(lh_mean - hl_mean) / (lh_mean + hl_mean + 1e-6)
    energy_level = hh_mean / (lh_mean + hl_mean + 1e-6)

    # Calibrate: 
    # Higher energy in HH relative to others = sus
    # Significant anisotropy = sus
    score = (min(anisotropy * 5.0, 1.0) * 0.4) + (min(energy_level * 2.0, 1.0) * 0.6)

    return {
        "score": float(score),
        "hh_energy_ratio": float(energy_level),
        "anisotropy_index": float(anisotropy)
    }
