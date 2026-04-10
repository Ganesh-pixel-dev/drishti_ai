import cv2
import numpy as np

def detect_median_filtering(image_path):
    """
    Detects fingerprints of median filtering.
    Median filtering leaves specific statistical patterns in the pixel difference histogram.
    """
    img = cv2.imread(image_path, 0)
    if img is None:
        return {"score": 0.0, "details": "File not found"}

    # Calculate horizontal and vertical pixel differences
    diff_h = np.abs(img[:, :-1].astype(np.int16) - img[:, 1:].astype(np.int16))
    diff_v = np.abs(img[:-1, :].astype(np.int16) - img[1:, :].astype(np.int16))

    # In median-filtered images, the probability of zero difference (flat regions) 
    # and specific patterns in the 'first-order difference' histogram is higher.
    def get_feature(diff_map):
        hist, _ = np.histogram(diff_map, bins=256, range=(0, 256), density=True)
        # Check for 'Comb' effect in low differences
        # P(0) in median filtered is usually much higher than P(1), P(2)
        if hist[0] > 0:
            ratio = hist[0] / (hist[1] + hist[2] + 1e-6)
            return ratio
        return 0.0

    score_h = get_feature(diff_h)
    score_v = get_feature(diff_v)

    # Normalize score: Natural images have ratio ~1.0-1.5. Filtered images have 5.0+.
    final_ratio = (score_h + score_v) / 2.0
    
    # Sigmoid-like scaling to 0-1
    # We calibrate so that a ratio of 4.0 starts looking suspicious (0.5)
    normalized_score = 1.0 / (1.0 + np.exp(-(final_ratio - 4.0)))

    return {
        "score": float(normalized_score),
        "ratio": float(final_ratio)
    }
