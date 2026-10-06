import cv2
import numpy as np

def analyze_temporal_noise(video_path, samples=15, jitter=1.0):
    """
    Analyzes noise consistency across frames.
    [ARMADA v2] Jitter-aware: If the camera is stable (low jitter),
    stationary noise is EXPECTED and should not be flagged as AI.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"score": 0.0, "details": "Could not open video"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // samples)
    
    noise_maps = []
    
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Extract high-frequency noise using a high-pass filter (Laplacian)
        noise = cv2.Laplacian(gray, cv2.CV_64F)
        noise_maps.append(noise)
        
        if len(noise_maps) >= samples: break
    
    cap.release()
    
    if len(noise_maps) < 2:
        return {"score": 0.0, "details": "Not enough frames"}

    # Correlation Analysis
    correlations = []
    for i in range(len(noise_maps) - 1):
        n1 = noise_maps[i]
        n2 = noise_maps[i+1]
        c = np.corrcoef(n1.flatten(), n2.flatten())[0, 1]
        correlations.append(c)

    avg_corr = float(np.nanmean(correlations)) if not np.all(np.isnan(correlations)) else 0.0
    
    # [ARMADA v2] Absolute Physics Override
    # High correlation + Low Jitter = Authentic Static Video
    # High correlation + High Jitter = Synthetic Baked-in Noise
    
    if jitter < 0.05:
        # On a tripod, noise MUST be consistent. 
        # If it's NOT consistent on a tripod, THAT'S actually suspicious (flickering).
        score = 0.0 if avg_corr > 0.3 else 0.4
    else:
        # Standard moving video logic
        if avg_corr > 0.4:
            score = 0.9
        elif avg_corr > 0.2:
            score = 0.5
        else:
            score = 0.0

    return {
        "score": float(score),
        "avg_noise_correlation": float(avg_corr),
        "jitter_context": jitter
    }
