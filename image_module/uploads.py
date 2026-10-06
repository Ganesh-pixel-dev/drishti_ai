import base64
import io
import os
import tempfile
from contextlib import contextmanager

from django.conf import settings
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}


class UploadError(ValueError):
    """Message is safe to show to the user."""


def validate_image(upload):
    """Check extension, size and that Pillow can really decode it. Raises UploadError."""
    ext = os.path.splitext(upload.name)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        raise UploadError("Unsupported file type. Use JPG, PNG, WebP, BMP or TIFF.")
    if upload.size > settings.MAX_IMAGE_UPLOAD_BYTES:
        mb = settings.MAX_IMAGE_UPLOAD_BYTES // (1024 * 1024)
        raise UploadError(f"File is larger than {mb} MB.")
    try:
        with Image.open(upload) as im:
            if im.format not in IMAGE_FORMATS:
                raise UploadError("The file contents are not a supported image format.")
            if im.width * im.height > settings.MAX_IMAGE_PIXELS:
                raise UploadError("Image has too many pixels (limit 50 megapixels).")
            im.verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise UploadError("Could not read this file as an image.")
    finally:
        upload.seek(0)
    return ext


@contextmanager
def saved_temp(upload, suffix):
    """Write an upload to a temp directory and delete it afterwards."""
    with tempfile.TemporaryDirectory(prefix="drishti_") as directory:
        path = os.path.join(directory, "upload" + suffix)
        with open(path, "wb") as out:
            for chunk in upload.chunks():
                out.write(chunk)
        yield directory, path


def data_uri(path, max_side=1400):
    """Downscaled JPEG preview as a data URI (for display only, never analysed)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
