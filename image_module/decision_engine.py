def analyze_results(results, metadata="Generic", jitter=1.0):
    """
    Consolidates 20+ detectors into a single forensic verdict using weighted voting.
    Physics-Primacy: Jitter (temporal stability) acts as a high-level arbitrator.
    """
    explanation = []
    # Heuristics
    ela = results.get("ela", {}).get("score", 0.0)
    noise = results.get("noise", {}).get("score", 0.0)
    freq = results.get("frequency", {}).get("score", 0.0)
    patch = results.get("patch", {}).get("score", 0.0)
    
    # New Armour Detectors
    lum = results.get("luminance", {}).get("score", 0.0)
    median = results.get("median", {}).get("score", 0.0)
    hist = results.get("histogram", {}).get("score", 0.0)
    cfa = results.get("cfa", {}).get("score", 0.0)
    wavelet = results.get("wavelet", {}).get("score", 0.0)
    comp = results.get("compression", {}).get("score", 0.0)
    exif = results.get("exif", {}).get("score", 0.0)
    reflect = results.get("reflection", {}).get("score", 0.0)
    adv = results.get("adversarial", {}).get("score", 0.0)
    heart = results.get("heartbeat", {}).get("score", 0.0)
    heart = results.get("heartbeat", {}).get("score", 0.0)
    geom = results.get("geometric", {}).get("score", 0.0)
    boundary = results.get("boundary", {}).get("score", 0.0)
    
    # AI Core (Gemini/Local ML)
    ai_core = results.get("ai_core", {})
    ai_verdict_str = ai_core.get("verdict", "Unknown")
    ai_confidence = ai_core.get("confidence", 0.0)
    edited_region = ai_core.get("edited_region", None)

    # Removed Armada v3 Adaptive Neural Dampening (Flawed premise: Deepfakes have low jitter, so this blinded the AI)

    # --- WEIGHTED FORENSIC SCORE ---
    # We prioritize signal-processing (CFA, DQ, ELA) as they are hardest to fake.
    weights = {
        "cfa": 0.20,      # Sensor artifacts
        "comp": 0.15,     # Compression anomalies
        "ela": 0.15,      # Error level
        "noise": 0.10,    # Noise variance
        "median": 0.08,   # Smoothing detection
        "wavelet": 0.08,  # High-freq stats
        "lum": 0.05,      # Lighting
        "hist": 0.05,     # Color anomaly
        "patch": 0.05,    # Copy-move
        "freq": 0.04,     # Global freq
        "reflect": 0.05,  # Eye reflections
        "adv": 0.05,      # Adversarial noise
        "geom": 0.15      # Geometric integrity (New Lead)
    }
    
    # --- PHYSICS-FIRST CALIBRATION (Webcam Shield v2) ---
    # If Jitter is extremely low (< 0.05), it strongly suggests a real physical lens.
    # In this state, we lower the 'burden of proof' for texture-based sensors.
    if jitter < 0.05 and metadata == "Generic/Webcam":
        cfa *= 0.4    # Webcam grain looks like CFA patterns
        noise *= 0.4  # Webcam noise looks like GAN noise
        ela *= 0.5    # Webcam denoising looks like ELA
        explanation.append("Safety: High lens stability detected. Reducing texture-sensor sensitivity.")
    
    heuristic_score = (
        (cfa * weights["cfa"]) +
        (comp * weights["comp"]) +
        (ela * weights["ela"]) +
        (noise * weights["noise"]) +
        (median * weights["median"]) +
        (wavelet * weights["wavelet"]) +
        (lum * weights["lum"]) +
        (hist * weights["hist"]) +
        (patch * weights["patch"]) +
        (freq * weights["freq"]) +
        (reflect * weights["reflect"]) +
        (adv * weights["adv"]) +
        (geom * weights["geom"])
    )
    
    # HEAVY ARMOUR: BIOLOGICAL OVERRIDE (Absence as Evidence)
    if heart > 0.8:
        # ABSENCE AS EVIDENCE:
        # But for webcams, we are more lenient if Vision is perfect.
        if metadata == "Generic/Webcam" and ai_verdict_str == "Likely Real" and ai_confidence > 0.95:
            explanation.append("Note: Biological scan limited by sensor quality. Vision consensus overrides void.")
            heart = 0.2 # Downgrade from flag to minor note
        
        heuristic_score = max(heuristic_score, heart)
    
    # Final Decision Routing
    final_score = heuristic_score
    
    # [ARMADA v4] Standardized 3-category Verdict System
    if (ai_verdict_str == "AI-generated" and ai_confidence > 0.85) or final_score > 0.50 or boundary > 0.65:
        verdict = "Final verdict:- AI generated image"
        final_score = max(ai_confidence, final_score, boundary)
    elif final_score > 0.30 or (ai_verdict_str == "AI-generated" and ai_confidence > 0.60) or exif > 0.7 or boundary > 0.4:
        verdict = "Final verdict:- Edited image"
        final_score = max(final_score, exif, boundary)
    else:
        verdict = "Final verdict:- Real image"

    # Explanation Builder
    if edited_region: explanation.append(f"Region flagged: {edited_region}")
    if cfa > 0.5: explanation.append("Sensor-level CFA pattern inconsistencies detected.")
    if heart > 0.8 and metadata != "Generic/Webcam": explanation.append("Biological Void: No heartbeat detected in visible face.")
    if reflect > 0.6: explanation.append("Geometric Paradox: Impossible eye reflections detected.")
    if boundary > 0.4: explanation.append("Morph Seam: Unnatural blending detected along the facial jawline boundary.")
    if comp > 0.6: explanation.append("JPEG block artifact misalignment found (BAG anomaly).")
    if ela > 0.4: explanation.append("Inconsistent compression levels across regions (ELA).")
    if geom > 0.5: explanation.append("Geometric Anomaly: Face-mesh warping or artificial symmetry detected.")
    if median > 0.7: explanation.append("Trace of non-linear median filtering discovered.")
    if wavelet > 0.6: explanation.append("Statistical high-frequency noise anomalies identified.")
    if lum > 0.6: explanation.append("Lighting/Luminance gradient contradictions detected.")
    
    if not explanation:
        if verdict == "Likely Authentic":
            explanation.append("All 20+ local forensic tests passed.")
        else:
            explanation.append("Minor forensic traces detected.")

    return {
        "verdict": verdict,
        "confidence": round(final_score * 100, 2),
        "ai_verdict": ai_verdict_str,
        "ai_confidence": round(ai_confidence * 100, 2),
        "edited_region": edited_region,
        "details": results,
        "explanation": explanation
    }