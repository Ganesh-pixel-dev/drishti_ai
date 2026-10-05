"""The two Hugging Face image classifiers, loaded lazily, returning P(AI-generated).

dima806/deepfake_vs_real_image_detection  labels: Real / Fake
Organika/sdxl-detector                    labels: artificial / human

The label names differ, so each model has its own mapping. (An earlier version
searched the label for "ai", which never matched "artificial" and made the second
model always vote "real".)
"""
import logging

from PIL import Image

logger = logging.getLogger(__name__)

MODELS = {
    "vit_deepfake": ("dima806/deepfake_vs_real_image_detection", "fake"),
    "swin_sdxl": ("Organika/sdxl-detector", "artificial"),
}

_pipelines = {}


def _get_pipeline(key):
    if key not in _pipelines:
        import torch
        from transformers import pipeline
        model_id, _ = MODELS[key]
        try:
            device = 0 if torch.cuda.is_available() else -1
            _pipelines[key] = pipeline("image-classification", model=model_id,
                                       top_k=None, device=device)
        except Exception as exc:
            logger.warning("Could not load %s: %s", model_id, exc)
            _pipelines[key] = None
    return _pipelines[key]


def prob_ai(key, image):
    """P(AI-generated) from one classifier for a PIL image, or None if it is unavailable."""
    clf = _get_pipeline(key)
    if clf is None:
        return None
    _, ai_label = MODELS[key]
    for item in clf(image.convert("RGB")):
        if item["label"].lower() == ai_label:
            return float(item["score"])
    return None


def all_probs(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        return {key: prob_ai(key, img) for key in MODELS}
