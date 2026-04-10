import cv2
import os
import logging
import json
import numpy as np
from PIL import Image
from image_module.utils import run_full_analysis
from .detectors import analyze_temporal_noise, analyze_optical_flow, detect_temporal_jitter, detect_heartbeat

logger = logging.getLogger(__name__)

MAX_SAMPLE_FRAMES = 8 # Reduced samples for deeper local analysis per frame

def check_video_metadata(video_path):
    brands = [b"Apple", b"Samsung", b"Sony", b"Nikon", b"Canon", b"GoPro", b"DJI", b"Xiaomi", b"Google", b"Huawei"]
    try:
        with open(video_path, 'rb') as f:
            header = f.read(16384)
            for brand in brands:
                if brand in header:
                    return brand.decode()
    except: pass
    return None

def evaluate_video_final(video_path, max_duration=60):
    """
    Overhauled 'Local-First' Video Forensic Engine.
    Uses 20+ detectors per frame + temporal video-only detectors.
    """
    if not os.path.exists(video_path):
        return {"error": "Video file not found."}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_secs = total_frames / fps if fps > 0 else 0

    if duration_secs <= 0:
        cap.release()
        return {"error": "Could not read video duration."}

    # 1. Hardware Audit
    camera_brand = check_video_metadata(video_path)
    metadata_val = camera_brand or "Generic/Webcam"
    
    num_samples = min(MAX_SAMPLE_FRAMES, max(3, int(duration_secs // 2)))
    step = duration_secs / num_samples
    timestamps = [step * i for i in range(num_samples)]
    
    # 🧬 PHASE 0: TEMPORAL STABILITY (The 'Supreme Court')
    # We calculate Jitter FIRST because physics never lies.
    jitter_score = detect_temporal_jitter(video_path)

    frame_paths = []
    frame_results = []
    
    # 🧪 PHASE 1: DEEP PER-FRAME FORENSICS (20+ Detectors each)
    for i, ts in enumerate(timestamps):
        frame_id = int(fps * ts)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            f_path = os.path.join(os.path.dirname(video_path), f"_vframe_{i}.jpg")
            cv2.imwrite(f_path, frame)
            frame_paths.append(f_path)
            
            # 🔥 RUN 20+ IMAGE DETECTORS ON THIS FRAME
            # Now with Physics-Primacy (Jitter) passed down
            analysis = run_full_analysis(f_path, metadata=metadata_val, jitter=jitter_score)
            frame_results.append(analysis)
    
    cap.release()

    if not frame_results:
        return {"error": "Deep analysis failed to process frames."}

    # 🧬 PHASE 2: REMAINING TEMPORAL DETECTORS
    noise_result = analyze_temporal_noise(video_path, jitter=jitter_score)
    flow_result = analyze_optical_flow(video_path)
    
    # HEAVY ARMOUR: BIOLOGICAL PULSE SCAN
    from .detectors.heartbeat_detector import detect_heartbeat
    from .detectors.blink_consistency import analyze_blink_consistency
    from .detectors.landmark_jitter import analyze_landmark_jitter
    from .detectors.spectral_fingerprint import analyze_spectral_fingerprint
    
    heart_result = detect_heartbeat(video_path)
    heart_score = heart_result.get("score", 0.0)
    
    blink_result = analyze_blink_consistency(video_path)
    blink_score = blink_result.get("score", 0.0)
    
    volume_result = analyze_landmark_jitter(video_path)
    volume_score = volume_result.get("score", 0.0)
    
    spectral_result = analyze_spectral_fingerprint(video_path)
    spectral_score = spectral_result.get("score", 0.0)

    # 🧠 PHASE 3: MULTI-LEVEL CONSENSUS
    timeline = []
    total_ai_votes = 0
    total_suspicious_votes = 0
    
    for i, res in enumerate(frame_results):
        v = res.get("verdict", "")
        is_ai = "AI" in v or "Forged" in v
        is_suspicious = "Suspicious" in v
        
        if is_ai: total_ai_votes += 1
        if is_suspicious: total_suspicious_votes += 1
        
        timeline.append({
            "sec": round(timestamps[i], 1),
            "verdict": v,
            "confidence": res.get("confidence", 0.0),
            "note": res['explanation'][0] if res['explanation'] else "Scan Complete"
        })

    # FINAL AGGREGATION
    ai_ratio = total_ai_votes / len(frame_results)
    suspicious_ratio = (total_ai_votes + total_suspicious_votes) / len(frame_results)
    
    # WEBCAM-RESISTANT AGGRESSIVE WEIGHTING
    # Temporal Noise and Jitter are the hardest for AI to hide.
    # [ARMADA v4] ABSOLUTE BIOLOGICAL SILENCE:
    # Webcams almost always fail heartbeat scans. We now ignore this entirely.
    heart_weight = 0.2
    if metadata_val == "Generic/Webcam":
        heart_weight = 0.0
        heart_score = 0.0 # Force zero for webcams
    
    temporal_score = (
        (jitter_score * 0.45) + 
        (noise_result['score'] * 0.45) + 
        (flow_result['score'] * 0.10) + 
        (heart_score * heart_weight)
    )
    
    # MAJOR ANOMALY OVERDRIVE
    # If any single video-detector is screaming 'AI' (> 0.45), we boost the final score.
    # CRITICAL FIX: Ignore Heartbeat Void for Overdrive if it's a Generic/Webcam.
    if metadata_val == "Generic/Webcam":
        max_detector_anomaly = max(jitter_score, noise_result['score'], blink_score, volume_score, spectral_score)
    else:
        max_detector_anomaly = max(jitter_score, noise_result['score'], heart_score, blink_score, volume_score, spectral_score)
    
    if max_detector_anomaly > 0.45:
        temporal_score = max(temporal_score, max_detector_anomaly)
    
    # Decide Verdict based on Consensus
    # [ARMADA v4] Final Surgical Wedge calibration
    # Deepfakes rarely trigger 100% frame failure on stable shots. 
    # If even 30% of the frames scream AI, it's a Deepfake.
    ai_threshold = 0.30 
    
    if ai_ratio >= ai_threshold or temporal_score > 0.65:
        overall_verdict = f"Highly Likely Deepfake (Consensus AI Ratio: {ai_ratio*100:.0f}%)"
        final_conf = max(ai_ratio, temporal_score)
    elif suspicious_ratio > 0.40 or temporal_score > 0.40 or ai_ratio > 0.10:
        overall_verdict = "Suspicious (Inconsistent Forensic Markers)"
        final_conf = max(suspicious_ratio, temporal_score, ai_ratio)
    else:
        overall_verdict = "Likely Authentic"
        if camera_brand: overall_verdict = "AUTHENTIC (Verified Hardware Signature)"
        final_conf = 1.0 - max(ai_ratio, temporal_score)

    if blink_score > 0.8: overall_verdict += " [True Sight: Reptilian Gaze]"
    if volume_score > 0.7: overall_verdict += " [True Sight: Volume Morphing]"
    if spectral_score > 0.8: overall_verdict += " [True Sight: Generative Pulse]"

    # Cleanup
    for p in frame_paths:
        try: os.remove(p)
        except: pass

    notes = [
        f"Temporal Jitter: {jitter_score:.2f}", 
        f"Noise Consist: {noise_result['score']:.2f}", 
        f"Biological Void: {heart_score:.2f}",
        f"Flow Divergence: {flow_result['score']:.2f}",
        f"Blink Anomaly: {blink_score:.2f}",
        f"Volume Jitter: {volume_score:.2f}",
        f"Spectral Spike: {spectral_score:.2f}"
    ]
    if camera_brand: notes.append(f"Header: {camera_brand}")

    return {
        "verdict": overall_verdict,
        "ai_ratio_percent": round(ai_ratio * 100, 2),
        "avg_confidence": round(final_conf * 100, 2),
        "jitter_score": jitter_score,
        "duration": round(duration_secs, 1),
        "frames_analyzed": len(frame_results),
        "timeline": timeline,
        "notes": " | ".join(notes),
        "metadata": camera_brand or "Generic/Webcam"
    }
