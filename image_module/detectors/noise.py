import cv2
import numpy as np

def detect_noise(image_path):
    image = cv2.imread(image_path, 0)

    blur = cv2.GaussianBlur(image, (5, 5), 0)
    noise = cv2.absdiff(image, blur)

    global_score = np.mean(noise) / 255.0
    
    # Regional noise analysis — spliced regions have different noise profiles
    h, w = noise.shape
    block_h, block_w = max(h // 6, 1), max(w // 6, 1)
    
    block_noise_levels = []
    for r in range(0, h - block_h + 1, block_h):
        for c in range(0, w - block_w + 1, block_w):
            block = noise[r:r+block_h, c:c+block_w]
            block_noise_levels.append(np.mean(block))
    
    if len(block_noise_levels) >= 4:
        block_noise_levels = np.array(block_noise_levels)
        noise_std = np.std(block_noise_levels)
        # High std = some regions are much noisier than others = splice indicator
        variance_score = min(noise_std / 15.0, 1.0)
    else:
        variance_score = 0.0
    
    score = (global_score * 0.4) + (variance_score * 0.6)

    return {
        "score": float(score),
        "map": None
    }