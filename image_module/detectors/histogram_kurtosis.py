import cv2
import numpy as np
from scipy.stats import kurtosis

def detect_histogram_anomaly(image_path):
    """
    Analyzes the color histogram for AI-generated signatures.
    AI models often exhibit 'spectral combing' or unnatural kurtosis in specific channels.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"score": 0.0, "details": "File not found"}

    # Run analysis on RGB and Lab spaces
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    
    scores = []
    
    # Check Kurtosis (tailedness) of color channels
    # AI pixels are often more clustered or exhibit 'perfect' distributions
    for i in range(3):
        channel = img[:,:,i].flatten()
        k = kurtosis(channel)
        # Highly positive kurtosis (> 3) or negative kurtosis (< -1) is suspicious for 'natural' photography
        scores.append(abs(k))

    # Check for 'Gaps' in the histogram (Quantization artifacts from generation)
    gap_score = 0
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256]).flatten()
        # Find index of zero bins within the active range
        active_range = np.where(hist > 0)[0]
        if len(active_range) > 0:
            sub_hist = hist[active_range[0]:active_range[-1]]
            zeros = np.sum(sub_hist == 0)
            gap_score += zeros / len(sub_hist)

    avg_kurtosis = np.mean(scores)
    # Calibrate: natural kurtosis is usually -1 to 2. 
    # AI/Edited images often have values > 10 or very specific clusters.
    k_normalized = min(avg_kurtosis / 10.0, 1.0)
    g_normalized = min(gap_score * 5.0, 1.0) # Gaps are a strong indicator

    final_score = (k_normalized * 0.4) + (g_normalized * 0.6)

    return {
        "score": float(final_score),
        "kurtosis": float(avg_kurtosis),
        "gap_ratio": float(gap_score)
    }
