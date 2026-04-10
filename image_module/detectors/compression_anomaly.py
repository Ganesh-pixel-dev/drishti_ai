import cv2
import numpy as np

def detect_compression_anomaly(image_path):
    """
    Detects Block Artifact Grid (BAG) inconsistencies.
    JPEG compression happens in 8x8 blocks. If an image was spliced,
    the 8x8 grids of the original and the splice will likely be misaligned.
    """
    img = cv2.imread(image_path, 0)
    if img is None:
        return {"score": 0.0, "details": "File not found"}

    # Compute a simple 'Blocking' signal
    # We look at the difference across 8x8 boundaries vs internal pixels
    h, w = img.shape
    if h < 16 or w < 16:
        return {"score": 0.0, "details": "Image too small"}

    # Grid alignment check
    def get_grid_strength(offset_y, offset_x):
        # Slice image into the 8x8 grid with given offset
        rows = range(offset_y + 7, h - 8, 8)
        cols = range(offset_x + 7, w - 8, 8)
        if not rows or not cols: return 0
        
        # Horizontal boundaries
        h_diff = np.abs(img[rows, :].astype(np.int32) - img[np.array(rows)+1, :].astype(np.int32))
        # Vertical boundaries
        v_diff = np.abs(img[:, cols].astype(np.int32) - img[:, np.array(cols)+1].astype(np.int32))
        
        return np.mean(h_diff) + np.mean(v_diff)

    # Test all 64 possible offsets to find the 'true' grid
    strengths = []
    for y in range(8):
        for x in range(8):
            strengths.append(get_grid_strength(y, x))
    
    strengths = np.array(strengths).reshape(8, 8)
    
    # In a clean JPEG, one offset will have a much higher 'strength' (boundary difference)
    # In a forged image or a PNG, the grid will be weak or multiple grids will conflict.
    max_strength = np.max(strengths)
    avg_strength = np.mean(strengths)
    
    grid_dominance = (max_strength - avg_strength) / (avg_strength + 1e-6)
    
    # Final Score: 
    # High grid dominance = likely authentic JPEG (consistent grid).
    # Low grid dominance = suspicious (no grid, multicompressed, or AI-generated).
    if grid_dominance < 0.2:
        score = 0.5 # Suspicious
    elif grid_dominance < 0.1:
        score = 0.8 # Highly Likely Forged/AI
    else:
        score = 0.0 # Consistent
        
    return {
        "score": float(np.clip(score, 0, 1)),
        "grid_dominance": float(grid_dominance)
    }
