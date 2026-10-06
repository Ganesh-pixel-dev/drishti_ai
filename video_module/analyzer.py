"""Video checks. Every number here is an unvalidated heuristic: none of these
detectors has been benchmarked. Per-frame image flags come from the image models,
which were trained on photos, not on video frames."""
import logging
import os
import tempfile

import cv2

from image_module.analysis import analyze_image

from .detectors import (analyze_optical_flow, analyze_temporal_noise,
                        detect_heartbeat, detect_temporal_jitter)
from .detectors.blink_consistency import analyze_blink_consistency
from .detectors.landmark_jitter import analyze_landmark_jitter
from .detectors.spectral_fingerprint import analyze_spectral_fingerprint

logger = logging.getLogger(__name__)

SAMPLE_FRAMES = 8
MAX_DURATION_SECONDS = 600


def evaluate_video_final(video_path, max_duration=60):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps and fps > 0 else 0
    if not cap.isOpened() or duration <= 0:
        cap.release()
        return {"error": "Could not read this video."}
    if duration > MAX_DURATION_SECONDS:
        cap.release()
        return {"error": f"Video is longer than {MAX_DURATION_SECONDS // 60} minutes."}

    count = min(SAMPLE_FRAMES, max(3, int(duration // 2)))
    step = duration / count
    timeline = []
    with tempfile.TemporaryDirectory(prefix="drishti_v_") as tmp:
        for i in range(count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * step * i))
            ok, frame = cap.read()
            if not ok:
                continue
            frame_path = os.path.join(tmp, f"frame_{i}.png")
            cv2.imwrite(frame_path, frame)
            try:
                tasks = analyze_image(frame_path)["tasks"]
            except Exception:
                logger.exception("Frame analysis failed")
                continue
            timeline.append({
                "sec": round(step * i, 1),
                "edit": round(tasks["edit"]["probability"], 2) if "edit" in tasks else None,
                "ai": round(tasks["ai"]["probability"], 2) if "ai" in tasks else None,
                "flagged": any(t["flagged"] for t in tasks.values()),
            })
    cap.release()
    if not timeline:
        return {"error": "No frames could be analysed."}

    checks = {
        "Temporal jitter": detect_temporal_jitter(video_path),
        "Noise consistency": analyze_temporal_noise(video_path)["score"],
        "Optical flow": analyze_optical_flow(video_path)["score"],
        "Heartbeat signal (rPPG)": detect_heartbeat(video_path)["score"],
        "Blink rate": analyze_blink_consistency(video_path)["score"],
        "Landmark jitter": analyze_landmark_jitter(video_path)["score"],
        "Spectral spikes": analyze_spectral_fingerprint(video_path)["score"],
    }
    flagged_share = sum(f["flagged"] for f in timeline) / len(timeline)
    strongest = max(checks, key=checks.get)
    if flagged_share >= 0.5 or checks[strongest] > 0.65:
        verdict = "Flagged for human review (unvalidated heuristics)"
    else:
        verdict = "Not flagged (unvalidated heuristics)"
    return {
        "verdict": verdict,
        "duration": round(duration, 1),
        "frames_analyzed": len(timeline),
        "flagged_frame_percent": round(flagged_share * 100, 1),
        "checks": {k: round(float(v), 2) for k, v in checks.items()},
        "strongest_check": strongest,
        "timeline": timeline,
    }
