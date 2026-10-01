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
    "boundary": "Face Boundary analysis looks for the jawline seam where an AI face-swap mask is blended. High color-temperature mismatch along this curve is a definitive 'Skin Graft' flag.",
    "temporal noise": "Temporal Noise is the microscopic continuous film-grain of a video sensor. AI generators struggle to keep this stable, usually causing 'frozen grain' or wildly erratic pixel static between frames.",
    "texture": "Texture fidelity measures the micro-contrast of a surface (like skin pores). AI smoothing often obliterates high-frequency texture data, returning mathematically 'flat' surfaces."
}

def parse_context(context_string):
    """Safely extracts dictionary from session string."""
    if isinstance(context_string, str):
        try:
            return json.loads(context_string)
        except:
            return {}
    if isinstance(context_string, dict):
        return context_string
    return {}

def generate_chat_response(prompt, image_path, context_data):
    """
    [LOCAL ARMADA] Narrative Forensic Specialist.
    Converts raw mathematical forensic signals into expert investigative reports.
    100% Local. Enhanced Regex NLP Engine. No Cloud Dependencies.
    """
    try:
        p = prompt.lower()
        
        # Unpack Context State
        ctx = parse_context(context_data)
        verdict = ctx.get("verdict", "No evidence analyzed")
        explanation = ctx.get("explanation", ctx.get("explanations", []))
        notes = ctx.get("notes", "")
        conf = ctx.get("confidence", ctx.get("avg_confidence", 0.0))
        is_edited = "AI" in verdict or "Edited" in verdict

        # 🧪 1. GREETINGS & IDENTITY
        if re.search(r'\b(hello|hi|hey|howdy|greetings|who are you|what are you|are you ai)\b', p):
            return "Detective Drishti: Greetings. I am the Drishti AI Forensic Specialist—a 100% localized, offline diagnostic engine designed to interpret raw forensic data. How can I assist your investigation?"

        # 🧪 2. WHY FLAG / JUSTIFICATION
        if re.search(r'\b(why|reason|how do you know|fake|real|flagged|explain the result|what is wrong)\b', p):
            if not is_edited:
                return f"Detective Drishti: All forensic multi-layered probes passed. I am {conf}% confident this media is Authentic. The temporal consistency and sensor noise patterns behave exactly as expected from physical camera hardware with no signs of digital morphing."
            
            # Identify it's fake
            report = f"Detective Drishti: My analysis flagged this as '{verdict}'. "
            
            # Attach evidence based on module
            if explanation: # Image module typically returns list of explanations
                if isinstance(explanation, list):
                    report += f"I detected the following critical anomalies: {', '.join(explanation).lower()}."
                else:
                    report += f"Primary anomaly found: {explanation}."
            elif notes: # Video module typically returns string 'notes'
                report += f"The temporal engine flagged these metrics: {notes}."
            else:
                report += "The primary signal confidence indices fell below the threshold of natural physical physics."
            
            report += " These traits are physically impossible for a genuine hardware sensor to produce natively."
            return report

        # 🧪 3. DICTIONARY & JARGON LOOKUP
        # Matches "what is ELA", "explain jitter", "meaning of cfa"
        if re.search(r'\b(what is|what does|explain|meaning of|define)\b', p):
            for term, definition in FORENSIC_DB.items():
                if term in p:
                    return f"Detective Drishti (Forensic Lexicon): {definition}"
            return "Detective Drishti: I detected an inquiry regarding a forensic term, but it is not currently logged in my local Lexicon. My primary diagnostic focuses on noise, compression, frequency, and temporal logic."

        # Fallback exact word match just in case they typed "ela" without "what is"
        for term, definition in FORENSIC_DB.items():
            if re.search(rf'\b{term}\b', p):
                return f"Detective Drishti: {definition}"

        # 🧪 4. ACTIONABLE / EVIDENCE QUESTIONS
        if re.search(r'\b(prove|evidence|report|show me|where)\b', p):
            return "Detective Drishti: You can reference the 'Signal Confidence Indices' on the dashboard UI. For formal investigations, note the specific sensor anomalies flagged in this session and capture a snapshot of any visual heatmaps provided."

        # 🧪 5. DEFAULT FALLBACK
        return ("Detective Drishti: I am your private forensic narrative expert. "
                "You can ask me technical questions like 'What is Temporal Jitter?' or 'Explain ELA', "
                "or investigative questions like 'Why did you flag this file?'")
        
    except Exception as e:
        logger.error(f"Chat execution error: {e}", exc_info=True)
        return "Detective Drishti: Forensic data cache currently inaccessible due to an internal system error."
