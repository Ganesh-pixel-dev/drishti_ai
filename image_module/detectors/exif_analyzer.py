from PIL import Image
from PIL.ExifTags import TAGS

from ._common import result

_EDITING_TOOLS = ("photoshop", "gimp", "adobe", "pixelmator", "midjourney", "stable diffusion")


def analyze_exif(image_path):
    """Metadata check: missing EXIF, editing-software tags, half-filled camera tags.

    Metadata is trivially stripped or forged, so this is a weak hint. Images that
    went through a web upload or a dataset conversion have no EXIF at all, and the
    detector then reports a fixed score for every image.
    """
    with Image.open(image_path) as img:
        raw = img.getexif()
        if not raw:
            return {"score": 0.4, "applicable": False,
                    "details": "No EXIF data (common for web images, screenshots and AI output)."}
        exif = {TAGS.get(tag, tag): value for tag, value in raw.items()}

    score = 0.0
    notes = []
    software = str(exif.get("Software", "")).lower()
    if any(tool in software for tool in _EDITING_TOOLS):
        score += 0.5
        notes.append(f"Editing software tag: {software}")

    make, model = exif.get("Make"), exif.get("Model")
    if bool(make) != bool(model):
        score += 0.2
        notes.append("Camera make or model tag is missing.")

    return result(score, details=" | ".join(notes) or "EXIF tags look unremarkable.",
                  software=software, camera=f"{make or ''} {model or ''}".strip() or "Unknown")
