from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import os
import cv2
import uuid
from django.core.files.storage import FileSystemStorage

def perform_ela(image_path, quality=90):
    temp_path = f"temp_ela_{uuid.uuid4().hex}.jpg"

    original = Image.open(image_path).convert('RGB')
    original.save(temp_path, 'JPEG', quality=quality)

    compressed = Image.open(temp_path)

    ela_image = ImageChops.difference(original, compressed)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1

    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    ela_array = np.array(ela_image)

    # --- ENHANCED SCORING ---
    # 1. Global mean (basic overall tampering)
    global_mean = np.mean(ela_array) / 255.0
    
    # 2. Regional variance analysis (catches localized splices like face swaps)
    # Split image into a grid and compare region-level ELA intensities
    gray_ela = np.mean(ela_array, axis=2)  # collapse to grayscale
    h, w = gray_ela.shape
    block_h, block_w = max(h // 8, 1), max(w // 8, 1)
    
    block_means = []
    for r in range(0, h - block_h + 1, block_h):
        for c in range(0, w - block_w + 1, block_w):
            block = gray_ela[r:r+block_h, c:c+block_w]
            block_means.append(np.mean(block))
    
    if len(block_means) >= 4:
        block_means = np.array(block_means)
        block_std = np.std(block_means)
        block_max = np.max(block_means)
        block_min = np.min(block_means)
        block_range = (block_max - block_min) / 255.0
        
        # High variance between blocks = different regions were compressed differently = SPLICE
        regional_variance_score = min(block_std / 40.0, 1.0)
        
        # Hot-spot ratio: what fraction of blocks are significantly brighter than average?
        avg = np.mean(block_means)
        hot_blocks = np.sum(block_means > avg * 1.8)
        hot_ratio = hot_blocks / len(block_means)
        hotspot_score = min(hot_ratio * 3.0, 1.0)
    else:
        regional_variance_score = 0.0
        hotspot_score = 0.0
        block_range = 0.0
    
    # 3. Edge discontinuity: look for sharp compression boundaries
    # A pasted region creates a hard edge in the ELA map
    edges = cv2.Canny(gray_ela.astype(np.uint8), 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    edge_score = min(edge_density * 10.0, 1.0)
    
    # Combine scores: weight regional analysis heavily for splice detection
    score = (global_mean * 0.15) + (regional_variance_score * 0.35) + (hotspot_score * 0.30) + (edge_score * 0.20)
    score = min(score, 1.0)

    # Generate Visual Heatmap
    ela_bgr = cv2.cvtColor(ela_array, cv2.COLOR_RGB2BGR)
    ela_norm = cv2.normalize(ela_bgr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(ela_norm, cv2.COLORMAP_JET)

    # Save Heatmap
    heatmap_filename = f"ela_{uuid.uuid4().hex}.jpg"
    fs = FileSystemStorage()
    heatmap_path = os.path.join(fs.location, heatmap_filename)
    cv2.imwrite(heatmap_path, heatmap)
    
    heatmap_url = fs.url(heatmap_filename)

    os.remove(temp_path)

    return {
        "score": float(score),
        "heatmap_url": heatmap_url
    }