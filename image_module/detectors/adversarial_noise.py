import cv2
import numpy as np

def detect_adversarial_noise(image_path):
    """
    Adversarial Armour: Detects non-random digital noise patterns.
    Catching pixels designed to fool AI (Adversarial Attacks).
    """
    img = cv2.imread(image_path, 0)
    if img is None: return {"score": 0.0, "details": "File not found"}

    # Adversarial noise is often high-frequency but spatially correlated 
    # to target specific neurons.
    
    # 1. High-Pass Residual
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    
    # 2. Statistical Kurtosis of the Noise
    # Natural noise is Gaussian (bell-shaped). 
    # Adversarial noise has extreme outliers or specific distribution peaks.
    from scipy.stats import kurtosis
    k = abs(kurtosis(laplacian.flatten()))
    
    # Normalized score: high kurtosis in the high-pass residual = sus
    k_score = min(k / 50.0, 1.0)
    
    # 3. FFT Entropy of Noise
    fft = np.fft.fft2(laplacian)
    fft_shift = np.fft.fftshift(fft)
    mag = 20 * np.log(np.abs(fft_shift) + 1e-6)
    
    # Standard noise is evenly spread. Adversarial noise often has 'spikes' in 
    # the frequency domain to represent the pattern.
    max_f = np.max(mag)
    avg_f = np.mean(mag)
    f_gap = max_f - avg_f
    
    f_score = min(f_gap / 100.0, 1.0)

    final_score = (k_score * 0.5) + (f_score * 0.5)

    return {
        "score": float(final_score),
        "noise_kurtosis": float(k),
        "freq_gap": float(f_gap)
    }
