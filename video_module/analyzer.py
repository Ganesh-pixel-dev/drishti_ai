import cv2
import os
import logging
import json
from PIL import Image
from image_module.ai_model import predict_image, get_api_key
import statistics

logger = logging.getLogger(__name__)

MAX_SAMPLE_FRAMES = 12 

def check_video_metadata(video_path):
    """
    Scans the binary header of a video file for common camera manufacturer strings.
    A poor-man's ffprobe to distinguish real hardware from generic AI output.
    """
    brands = [b"Apple", b"Samsung", b"Sony", b"Nikon", b"Canon", b"GoPro", b"DJI", b"Xiaomi", b"Google", b"Huawei"]
    try:
        with open(video_path, 'rb') as f:
            header = f.read(16384) # Read first 16KB
            for brand in brands:
                if brand in header:
                    return brand.decode()
    except:
        pass
    return None

def evaluate_video_final(video_path, max_duration=60):
    """
    Extracts keyframes, runs Dual-Model analysis (Gemini + Local Texture),
    and applies 'Webcam Shield' for generic hardware.
    """
    if not os.path.exists(video_path):
        return {"error": "Video file not found."}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_secs = total_frames / fps if fps > 0 else 0

    if duration_secs <= 0:
        cap.release()
        return {"error": "Could not read video duration."}

    if duration_secs > max_duration:
        duration_secs = max_duration

    # 1. Extract & Detect Hardware Profile
    camera_brand = check_video_metadata(video_path)
    # Universal Generic Shield: Any source without a brand tag (webcams, desktop recording, etc.)
    is_webcam = not camera_brand
    
    num_samples = min(MAX_SAMPLE_FRAMES, max(4, int(duration_secs)))
    step = duration_secs / num_samples
    timestamps = [step * i for i in range(num_samples)]

    frame_paths = []
    texture_results = []
    
    for i, ts in enumerate(timestamps):
        frame_id = int(fps * ts)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            path = os.path.join(os.path.dirname(video_path), f"_vframe_{i}.jpg")
            cv2.imwrite(path, frame)
            frame_paths.append((i, ts, path))
            
            try:
                t_verdict, t_conf, _ = predict_image(path)
                # In Webcam Mode, we treat very high ISO grain more conservatively
                texture_results.append({
                    "frame": i + 1,
                    "is_ai_texture": "AI" in t_verdict.upper(),
                    "conf": t_conf
                })
            except:
                texture_results.append({"frame": i + 1, "is_ai_texture": False, "conf": 0})
    cap.release()

    if not frame_paths:
        return {"error": "Could not extract frames."}

    # 2. RUN WEBCAM-AWARE GEMINI AUDIT
    api_key = get_api_key()
    gemini_data = {}
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            contents = []
            for idx, ts, path in frame_paths:
                img = Image.open(path).convert("RGB")
                contents.append(img)
                contents.append(f"[Frame {idx+1}]")

            # Updated Prompt with Webcam Intelligence
            prompt = """Analyze these keyframes as a Digital Forensics Expert.
NOTE: This is LOW-QUALITY footage likely from a WEBCAM.
1. DO NOT flag sensor noise, 'salt & pepper' grain, or compression blockiness as AI.
2. DO NOT flag soft facial shadows as AI. These are webcam artifacts.
3. ONLY flag 'Structural Paradoxes': floating limbs or background morphing.
Unless you see a definitive physical paradox, verdict must be REAL.
Respond EXCLUSIVELY in JSON: {"overall_verdict": "DEEPFAKE"|"REAL", "overall_confidence": 0.0-1.0, "analysis_notes": "...", "frames": [{"frame": 1, "verdict": "AI"|"REAL", "confidence": 0.0-1.0, "notes": "..."}]}"""
            
            contents.append(prompt)
            response = client.models.generate_content(model='gemini-2.5-flash', contents=contents)
            gemini_data = json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())
        except Exception as e:
            logger.warning(f"Gemini failed: {e}")

    # 3. AGGREGATED ANALYSIS (The "Webcam Shield" Consensus)
    timeline = []
    gemini_frames = {f['frame']: f for f in gemini_data.get('frames', [])}
    
    ai_texture_votes = 0
    ai_vision_votes = 0
    total_analyzed = len(frame_paths)

    ov_gemini = gemini_data.get("overall_verdict", "").upper()
    oc_gemini = float(gemini_data.get("overall_confidence", 0.0))
    is_vision_solid_real = ("REAL" in ov_gemini) and oc_gemini > 0.8

    for i in range(total_analyzed):
        frame_idx = i + 1
        g = gemini_frames.get(frame_idx, {"verdict": "UNKNOWN", "confidence": 0.0, "notes": ""})
        t = texture_results[i]
        
        # WEBCAM SHIELD FORCE-FIELD:
        # We increase the bar for the Texture model if we know it's a webcam
        texture_cap = 0.98 if is_webcam else 0.88
        
        vision_says_ai = "AI" in g['verdict'].upper() and g['confidence'] > 0.9
        texture_says_ai = t['is_ai_texture'] and t['conf'] > texture_cap
        
        if vision_says_ai: ai_vision_votes += 1
        if texture_says_ai: ai_texture_votes += 1

        timeline.append({
            "sec": frame_idx,
            "verdict": "AI" if (vision_says_ai or texture_says_ai) else "Real",
            "vision_conf": round(g['confidence'] * 100, 1),
            "texture_conf": round(t['conf'] * 100, 1),
            "note": g['notes'] if vision_says_ai else ("Texture Anomaly" if texture_says_ai else "Clean Math")
        })

    # --- FINAL VERDICT (CALIBRATED) ---
    precision_vision = ai_vision_votes / total_analyzed
    precision_texture = ai_texture_votes / total_analyzed

    from .detectors import detect_temporal_jitter
    jitter_score = detect_temporal_jitter(video_path, samples=20)

    overall_verdict = "Likely Authentic"
    diagnostic_notes = []

    # THE ATOMIC PRECISION OVERRIDE:
    # If the system identifies a webcam/generic source AND the visual audit (Gemini) 
    # finds ZERO structural evidence of AI, we MUST ignore the texture math entirely.
    if is_webcam and precision_vision < 0.2:
        precision_texture = 0.0
        overall_verdict = "AUTHENTIC (Webcam Shield Enabled)"
        diagnostic_notes.append("Shield: Web/Generic sensor optimization active.")
        # SYNC TIMELINE: Ensure all frames match the authentic decision
        for item in timeline:
            item["verdict"] = "Real"
            item["note"] = "Webcam Artifact Filtered"
    else:
        # Standard consensus logic
        consensus_trigger = 0.55 if is_webcam else 0.25
        
        if precision_vision >= consensus_trigger and precision_texture >= consensus_trigger:
            overall_verdict = "Highly Likely Deepfake (Consensus Identified)"
            diagnostic_notes.append("Dual-model verification confirmed AI origin.")
        elif precision_texture > 0.85 and not is_vision_solid_real:
            overall_verdict = "Highly Likely Deepfake (Texture Anomaly)"
            diagnostic_notes.append("Mathematical noise patterns match AI diffusion.")
        elif precision_vision > 0.4:
            overall_verdict = "Highly Likely Deepfake (Visual Anomaly)"
            diagnostic_notes.append("Major physical inconsistencies detected.")
        elif (precision_vision > 0.2 or precision_texture > 0.2) and not is_webcam:
            overall_verdict = "Suspicious (Inconsistent Markers)"
            diagnostic_notes.append("Forensic anomalies detected.")

    if camera_brand:
        diagnostic_notes.append(f"Header: Valid {camera_brand} signature.")

    # Cleanup
    for _, _, path in frame_paths:
        try: os.remove(path)
        except: pass

    notes = gemini_data.get("analysis_notes", "Forensic check complete.")
    if diagnostic_notes: notes = " | ".join(diagnostic_notes) + " -- " + notes

    return {
        "verdict": overall_verdict,
        "shield_token": "", # Cleaned for production
        "ai_ratio_percent": round(max(precision_vision, precision_texture) * 100, 2),
        "avg_confidence": round(oc_gemini * 100, 2),
        "texture_confidence": round(precision_texture * 100, 2),
        "jitter_score": jitter_score,
        "duration": round(duration_secs, 1),
        "frames_analyzed": total_analyzed,
        "timeline": timeline,
        "notes": notes,
        "metadata": camera_brand or ("Webcam/Generic" if is_webcam else "Generic/No Signature")
    }
