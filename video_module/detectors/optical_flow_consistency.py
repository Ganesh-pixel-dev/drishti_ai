import cv2
import numpy as np

def analyze_optical_flow(video_path, frames_to_check=10):
    """
    Analyzes the consistency of movement between frames using Optical Flow.
    Deepfakes often have 'micro-twitches' or edges that don't move 
    consistently with the rest of the head/body.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"score": 0.0, "details": "Could not open video"}

    ret, prev_frame = cap.read()
    if not ret: return {"score": 0.0, "details": "Empty video"}
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    anomalies = []
    
    for _ in range(frames_to_check):
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate Dense Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Compute flow magnitude
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Feature: Magnitude variance
        # In natural motion, movement is locally smooth.
        # In deepfakes, specific regions (eyes, mouth edges) often have divergent flow.
        h, w = mag.shape
        block_h, block_w = max(h // 8, 1), max(w // 8, 1)
        
        regional_mags = []
        for r in range(0, h - block_h + 1, block_h):
            for c in range(0, w - block_w + 1, block_w):
                block = mag[r:r+block_h, c:c+block_w]
                regional_mags.append(np.mean(block))
        
        if len(regional_mags) > 0:
            regional_mags = np.array(regional_mags)
            avg_mag = np.mean(regional_mags)
            if avg_mag > 0.1: # Only analyze frames with movement
                # Measure how much the busiest region deviates from the average
                deviation = np.max(regional_mags) / (avg_mag + 1e-6)
                anomalies.append(deviation)
        
        prev_gray = gray

    cap.release()
    
    if not anomalies:
        return {"score": 0.0, "details": "No significant movement detected."}
        
    avg_anomaly = np.mean(anomalies)
    
    # Natural motion usually has max/avg ratio < 5.0. 
    # Deepfake flickering can reach 20+.
    score = min(max(0, (avg_anomaly - 5.0) / 15.0), 1.0)

    return {
        "score": float(score),
        "flow_divergence_index": float(avg_anomaly)
    }
