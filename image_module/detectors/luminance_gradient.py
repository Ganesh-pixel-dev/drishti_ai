import cv2
import numpy as np

def detect_luminance_gradient(image_path):
    """
    Analyzes the consistency of light gradients across the image.
    Forged regions often have inconsistent gradient directions or magnitudes.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"score": 0.0, "details": "File not found"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Compute gradients using Sobel operators
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Calculate magnitude and direction
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

    # Normalize magnitude for statistical analysis
    norm_magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    # Analysis: Regional Gradient Variance
    # Spliced regions often have sharp, unnatural gradient spikes
    h, w = norm_magnitude.shape
    block_h, block_w = max(h // 10, 1), max(w // 10, 1)
    
    grid_scores = []
    for r in range(0, h - block_h + 1, block_h):
        for c in range(0, w - block_w + 1, block_w):
            block = norm_magnitude[r:r+block_h, c:c+block_w]
            grid_scores.append(np.std(block))

    if not grid_scores:
        return {"score": 0.0, "details": "Image too small"}

    grid_scores = np.array(grid_scores)
    
    # 1. Gradient Spikiness (Kurtosis-like)
    # Natural images have smooth gradients. Forgeries have sharp changes.
    avg_variance = np.mean(grid_scores)
    max_variance = np.max(grid_scores)
    
    inconsistency_score = (max_variance - avg_variance) / (max_variance + 1e-6)
    
    # 2. Histogram of Orientations (HOG-Lite)
    # AI generated images often have a "preferred" gradient direction due to convolution biases
    hist, _ = np.histogram(angle, bins=8, range=(0, 360), weights=magnitude)
    hist = hist / (np.sum(hist) + 1e-6)
    entropy = -np.sum(hist * np.log2(hist + 1e-6))
    
    # Low entropy in gradient directions = suspicious regularity (AI/Digital)
    # Average entropy for natural edges is high.
    entropy_score = max(0, (4.0 - entropy) / 4.0)

    final_score = (inconsistency_score * 0.6) + (entropy_score * 0.4)
    
    return {
        "score": float(np.clip(final_score, 0, 1)),
        "inconsistency": float(inconsistency_score),
        "entropy": float(entropy_score)
    }
