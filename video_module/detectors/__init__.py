from .temporal_noise import analyze_temporal_noise
from .optical_flow_consistency import analyze_optical_flow
from .heartbeat_detector import detect_heartbeat
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_temporal_jitter(video_path, samples=20):
    """
    Mathematical engine to detect Face-Swaps & Splices in video.
    Extracts consecutive frames and measures inter-frame structural variance.
    Deepfake splices suffer from micro-jitter and edge bleeding frame-to-frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < samples * 2: return 0.0

    start_frame = int(total_frames // 2 - (samples // 2))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))

    frames = []
    for _ in range(samples):
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()

    if len(frames) < 2: return 0.0

    jitter_scores = []
    for i in range(1, len(frames)):
        edges1 = cv2.Canny(frames[i - 1], 100, 200)
        edges2 = cv2.Canny(frames[i], 100, 200)
        diff = cv2.absdiff(edges1, edges2)
        h, w = diff.shape
        block_h, block_w = max(h // 6, 1), max(w // 6, 1)
        block_diffs = []
        for r in range(0, h - block_h + 1, block_h):
            for c in range(0, w - block_w + 1, block_w):
                block = diff[r:r+block_h, c:c+block_w]
                block_diffs.append(np.sum(block > 0) / (block_h * block_w))
        if block_diffs:
            block_diffs = np.array(block_diffs)
            variance = np.std(block_diffs)
            max_change = np.max(block_diffs)
            frame_jitter = (variance * 0.7) + (max_change * 0.3)
            jitter_scores.append(frame_jitter)

    if not jitter_scores: return 0.0
    return min(float(np.mean(jitter_scores) * 3.5), 1.0)
