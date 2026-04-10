import cv2
import numpy as np

def detect_cfa_artifacts(image_path):
    """
    Detects inconsistencies in CFA (Color Filter Array) artifacts.
    Demosaicing introduces a periodic correlation between neighboring pixels. 
    Forged regions (AI/Spliced) usually break this periodicity.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"score": 0.0, "details": "File not found"}

    # We use the 'Green' channel which has the highest density in Bayer patterns
    green = img[:, :, 1].astype(np.float32)

    # 1. Prediction Residuals
    # In authentic demosaiced images, a pixel can be predicted accurately from its 'Bayer-neighbors'
    # Simple bilinear kernel to estimate interpolation error
    kernel = np.array([[ 0, 0.25, 0],
                       [0.25, 0, 0.25],
                       [ 0, 0.25, 0]])
    
    predicted = cv2.filter2D(green, -1, kernel)
    residual = np.abs(green - predicted)

    # 2. Periodicity Analysis
    # Original demosaicing has 2x2 or 4x4 periodicity
    # We measure the variance of residuals in a 2x2 grid
    h, w = residual.shape
    h_adj, w_adj = (h // 2) * 2, (w // 2) * 2
    res_crop = residual[:h_adj, :w_adj]
    
    # Reshape into 2x2 blocks
    blocks = res_crop.reshape(h_adj // 2, 2, w_adj // 2, 2).swapaxes(1, 2)
    
    # Mean residual for each of the 4 positions in the 2x2 grid
    pos_means = [
        np.mean(blocks[:, :, 0, 0]),
        np.mean(blocks[:, :, 0, 1]),
        np.mean(blocks[:, :, 1, 0]),
        np.mean(blocks[:, :, 1, 1])
    ]
    
    # In authentic images, one or more positions will have significantly lower residual error
    # than others because they were the 'interpolated' ones.
    pos_means = np.array(pos_means)
    std_pos = np.std(pos_means)
    max_pos = np.max(pos_means)
    
    # CFA Score: Low variance between grid positions means the pattern is missing (AI/Forged)
    # High variance = Authentic signature preserved.
    cfa_consistency = std_pos / (max_pos + 1e-6)
    
    # For forensic mapping: where is the consistency lowest?
    # (High score here means high probability of being AI/Forged)
    final_score = 1.0 - min(cfa_consistency * 10.0, 1.0)

    return {
        "score": float(final_score),
        "consistency_index": float(cfa_consistency),
    }
