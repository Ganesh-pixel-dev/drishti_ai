import os
from PIL import Image
from PIL.ExifTags import TAGS
import datetime

def analyze_exif(image_path):
    """
    Analyzes EXIF metadata for consistency.
    Detects use of editing software (Photoshop, GIMP) or strange timestamp logic.
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        score = 0.0
        details = []
        
        if not exif_data:
            # Most AI-generated or stripped images have NO exif
            return {"score": 0.4, "details": "No EXIF data found (common in web-dl or AI)."}

        exif = {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
        
        # 1. Check for Software Tags
        software = str(exif.get("Software", "")).lower()
        editing_tools = ["photoshop", "gimp", "adobe", "pixelmator", "midjourney", "stable diffusion"]
        for tool in editing_tools:
            if tool in software:
                score += 0.5
                details.append(f"Editing software signature found: {software}")

        # 2. DateTime Consistency
        # DateTimeOriginal vs DateTimeDigitized vs File Mod Time
        dt_orig = exif.get("DateTimeOriginal")
        dt_file = datetime.datetime.fromtimestamp(os.path.getmtime(image_path)).strftime("%Y:%m:%d %H:%M:%S")
        
        if dt_orig and dt_file:
            # If the file was modified BEFORE it was allegedly taken, that's impossible.
            if dt_file < dt_orig:
                score += 0.3
                details.append("Temporal Paradox: File modification date precedes photo capture date.")

        # 3. Model vs Maker
        model = exif.get("Model")
        make = exif.get("Make")
        if (model and not make) or (make and not model):
            score += 0.2
            details.append("Incomplete camera identity tags.")

        final_score = min(score, 1.0)
        return {
            "score": float(final_score),
            "details": " | ".join(details) if details else "EXIF tags look consistent with hardware acquisition.",
            "software": software,
            "camera": f"{make} {model}" if make or model else "Unknown"
        }

    except Exception as e:
        return {"score": 0.0, "details": f"EXIF parse error: {e}"}
