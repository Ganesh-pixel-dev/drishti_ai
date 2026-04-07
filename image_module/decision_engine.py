def analyze_results(results):
    ela = results.get("ela", {}).get("score", 0.0)
    noise = results.get("noise", {}).get("score", 0.0)
    freq = results.get("frequency", {}).get("score", 0.0)
    patch = results.get("patch", {}).get("score", 0.0)
    
    ai_core = results.get("ai_core", {})
    ai_verdict_str = ai_core.get("verdict", "Unknown")
    ai_confidence = ai_core.get("confidence", 0.0)
    edited_region = ai_core.get("edited_region", None)

    # Weighted heuristic score (ELA and noise are most important for splices)
    heuristic_score = (ela * 0.35) + (noise * 0.30) + (freq * 0.20) + (patch * 0.15)
    
    # === DECISION ROUTING ===
    
    # 1. AI-Generated detection (Gemini/HF model says AI)
    if ai_verdict_str == "AI-generated" and ai_confidence > 0.70:
        verdict = "Highly Likely AI-Generated"
        final_score = ai_confidence
    
    # 2. Photoshop/Splice detection (heuristics catch face swaps, copy-paste)
    elif heuristic_score > 0.35:
        verdict = "Highly Likely Forged (Edited/Spliced)"
        final_score = heuristic_score
    
    # 3. Moderate AI suspicion
    elif ai_verdict_str == "AI-generated" and ai_confidence > 0.50:
        verdict = "Suspicious (Possible AI or Heavy Editing)"
        final_score = max(ai_confidence, heuristic_score)
    
    # 4. Moderate heuristic suspicion
    elif heuristic_score > 0.20:
        verdict = "Suspicious (Possible Editing Detected)"
        final_score = heuristic_score
    
    # 5. Low-confidence AI model — don't trust blindly
    elif ai_verdict_str == "Likely Real" and ai_confidence < 0.65:
        verdict = "Suspicious (Low Confidence — Manual Review Recommended)"
        final_score = ai_confidence
    
    # 6. Clean bill of health
    else:
        verdict = "Likely Authentic"
        if ai_verdict_str == "Likely Real":
            final_score = ai_confidence
        else:
            final_score = max(1.0 - heuristic_score, 0.5)

    # Build Explanation
    explanation = []
    
    if edited_region:
        explanation.insert(0, f"Specific region flagged as manipulated: {edited_region}")

    if ai_verdict_str == "AI-generated" and ai_confidence > 0.7:
        explanation.append("AI model detected synthetic generation artifacts.")
    if ela > 0.35:
        explanation.append(f"ELA detected compression inconsistencies (score: {ela:.2f}) — possible splice/edit.")
    if noise > 0.25:
        explanation.append(f"Noise analysis found regional inconsistencies (score: {noise:.2f}) — possible pasted region.")
    if freq > 0.5:
        explanation.append("Unnatural frequency/pixel distribution patterns.")
    if patch > 0.01:
        explanation.append("Repeated regions detected (Copy-Move).")
        
    if not explanation:
        explanation.append("No significant manipulation detected.")

    return {
        "verdict": verdict,
        "confidence": round(final_score * 100, 2),
        "ai_verdict": ai_verdict_str,
        "ai_confidence": round(ai_confidence * 100, 2),
        "edited_region": edited_region,
        "details": results,
        "explanation": explanation
    }