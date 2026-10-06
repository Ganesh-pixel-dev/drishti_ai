"""Rule-based assistant ("Detective Drishti"). No model, no network: it matches keywords
and explains the current result and the forensic terms. It cannot reason about the image."""
import json
import re

GLOSSARY = {
    "ela": "Error level analysis re-saves the image as a JPEG and looks at what changed. Regions that were last saved at a different quality than their surroundings can stand out. It also reacts to ordinary things like sharp edges and noise.",
    "cfa": "Camera sensors record one colour per pixel and software fills in the rest (demosaicing), which leaves a faint 2x2 pattern. Resizing and re-processing weaken it. Absence is not proof of editing.",
    "compression": "JPEG compresses 8x8 pixel blocks. An untouched JPEG has one consistent grid. Resizing, cropping or pasting from another image can break it.",
    "block": "JPEG compresses 8x8 pixel blocks. An untouched JPEG has one consistent grid. Resizing, cropping or pasting from another image can break it.",
    "noise": "Every camera and every processing step leaves noise with a certain texture. A pasted region may carry different noise from its surroundings.",
    "copy-move": "Copy-move means part of an image was cloned elsewhere in the same image. The check matches ORB keypoints that appear twice, far apart.",
    "patch": "Copy-move means part of an image was cloned elsewhere in the same image. The check matches ORB keypoints that appear twice, far apart.",
    "wavelet": "Wavelet statistics compare high-frequency detail in horizontal, vertical and diagonal directions.",
    "histogram": "The histogram check looks at the shape of the colour distribution and for empty bins that point to tone curves or resampling.",
    "exif": "EXIF is metadata stored in the file (camera, software, time). It is easy to strip or fake, so it is a weak hint only.",
    "kurtosis": "Kurtosis measures how heavy the tails of a distribution are. Here it is applied to pixel values and to a high-pass residual.",
}


def _context(raw):
    try:
        return json.loads(raw) if isinstance(raw, str) else (raw or {})
    except json.JSONDecodeError:
        return {}


def generate_chat_response(prompt, context_data):
    p = prompt.lower()
    ctx = _context(context_data)
    tasks = ctx.get("tasks", {})

    if re.search(r"\b(hello|hi|hey|who are you|what are you)\b", p):
        return ("I am a keyword-matching helper built into this page. I can explain the checks "
                "and summarise the current result. I do not look at the image myself.")

    if re.search(r"\b(why|flag|flagged|result|verdict|explain the result|summary)\b", p):
        if not tasks:
            return "There is no result yet. Upload an image first."
        parts = []
        for task in tasks.values():
            evidence = ", ".join(task["evidence"]) or "no single check dominated"
            parts.append(f"{task['verdict']} (score {task['probability']:.2f}). "
                         f"Checks that counted most: {evidence}.")
        return " ".join(parts) + " This is a flag for human review, not proof."

    for term, text in GLOSSARY.items():
        if re.search(rf"\b{re.escape(term)}", p):
            return text

    return ("Ask me about a check (for example 'what is ELA') or ask 'why was this flagged'.")
