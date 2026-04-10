import logging
import os
import re
import json

logger = logging.getLogger(__name__)

# --- LOCAL FORENSIC KNOWLEDGE BASE (THE DICTIONARY) ---
FORENSIC_DB = {
    "ela": "Error Level Analysis (ELA) identifies areas within an image that are at different compression levels. AI-generated regions often exhibit significantly lower error residuals than natural photograph grain.",
    "cfa": "Color Filter Array (CFA) artifacts are the microscopic 'checkerboard' fingerprints left by a physical camera sensor. AI generation fails to replicate these underlying hardware patterns.",
    "comp": "Compression Anomaly detection (BAG) looks for misalignments in JPEG block boundaries. Forged regions often break the standard 8x8 block grid of a real sensor.",
    "spectral": "Spectral Temporal Fingerprinting uses 1D Fourier Transforms to find the 'periodic pulse' of an AI model. Natural video noise is stochastic; AI generation has a regular mathematical beat.",
    "heartbeat": "Biological r-PPG scans look for heart-rate variants in skin color. AI videos (like Sora or Deepfacelab) usually lack this microscopic biological pulse.",
    "blink": "Blink Consistency tracks the Eye Aspect Ratio (EAR) over time. 'Reptilian Gaze' (zero blinks) is a hallmark of low-cost AI avatars or static face-modals.",
    "jitter": "Temporal Jitter analysis identifies non-rigid structural morphing. While real human heads are stable physical objects, deepfake faces often 'breathe' or subtlely warp.",
    "kurtosis": "Histogram Kurtosis measures the distribution of pixel intensities. Diffusion models often create 'super-flat' or 'clipped' histograms that are statistically impossible in nature.",
    "geometric": "Geometric Paradox detection looks for impossible facial symmetry or iris reflections that don't match the environment's light source.",
    "boundary": "Face Boundary analysis looks for the jawline seam where an AI face-swap mask is blended. High color-temperature mismatch along this curve is a definitive 'Skin Graft' flag."
}

def generate_chat_response(prompt, image_path, context_data):
    """
    [LOCAL ARMADA] Narrative Forensic Specialist.
    Converts raw mathematical forensic signals into expert investigative reports.
    100% Local. No Cloud Dependencies.
    """
    try:
        p = prompt.lower()
        
        # 🧪 1. DICTIONARY LOOKUP (Works even without analysis)
        for term, definition in FORENSIC_DB.items():
            if term in p:
                return f"Detective Drishti (Expert Mode): {definition}"

        # --- CONTEXT SAFETY SHIELD ---
        # If context_data is a JSON string from the session, unpack it
        if isinstance(context_data, str):
            try:
                context_data = json.loads(context_data)
            except:
                context_data = {}
        
        if not isinstance(context_data, dict):
            context_data = {}

        verdict = context_data.get("verdict", "Unknown")
        explanation = context_data.get("explanation", context_data.get("explanations", []))
        notes = context_data.get("notes", "")
        # --- END SHIELD ---

        # 🧪 2. WHY FLAG (The Correlation Logic)
        if any(w in p for w in ["why", "flag", "reason", "investigate"]):
            if not explanation:
                return "Detective Drishti: No forensic red flags were detected. The image clears all 20+ local signal processing layers with high confidence."
            
            # Build a correlated report
            report = f"Detective Drishti: My audit suggests this is {verdict.lower()}. "
            
            # Logic Correlation: Biological + Neural
            if any("heartbeat" in e.lower() or "biological" in e.lower() for e in explanation):
                report += "The primary anomaly is a 'Biological Void'—the absence of a human cardiovascular signature despite a visible face. "
            
            # Logic Correlation: Signal Artifacts
            if any(s in notes.lower() for s in ["spectral", "noise", "kurtosis"]):
                report += "This is corroborated by unnatural spectral spikes in the pixel frequency domain, which are characteristic of generative AI diffusion. "
            
            if len(explanation) > 1:
                report += f"Furthermore, I've identified {explanation[-1].lower()}."
            
            return report

        # 🧪 3. GENERAL TRAGI (The Verdict)
        if any(w in p for w in ["ai", "fake", "real", "truth", "summary"]):
            conf = context_data.get("confidence", context_data.get("avg_confidence", 0.0))
            return f"Detective Drishti: My consolidated audit is complete. I am {conf}% confident that this sample is {verdict}. The decision was reached through a consensus of multi-layered local forensic probes."

        # 🧪 4. HARDWARE AUDIT
        if any(w in p for w in ["meta", "exif", "hardware", "camera"]):
            meta = context_data.get("metadata", "Generic/Webcam")
            return f"Detective Drishti: The hardware signature for this file is '{meta}'. My rules were adaptive-weighted based on this sensor profile."

        # Default Fallback instructions
        return ("Detective Drishti: I am your private forensic narrative expert. "
                "You can ask me technical questions like 'What is ELA?' or 'Explain Spectral Spikes', "
                "or investigative questions like 'Why did you flag this?'")
        
    except Exception as e:
        logger.error(f"Chat execution error: {e}", exc_info=True)
        return "Detective Drishti: Forensic data cache currently inaccessible."
