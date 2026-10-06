"""Analyse one image file: run detectors, apply the learned combiner for each task."""
import io
import os
import tempfile

from PIL import Image

from . import combiner
from .pipeline import DETECTORS, LABELS, run_detectors

# Which trained weight file serves each task. Names match scripts/train_combiner.py output.
MODEL_FILES = {"edit": "edit_raw_jpgonly", "ai": "ai_jpeg90"}

TASK_TITLES = {
    "edit": "Edited or spliced?",
    "ai": "AI-generated?",
}


def _load(task):
    return combiner.load_model(MODEL_FILES[task])


def _jpeg90_copy(path, directory):
    """Same normalisation the AI model was trained with: re-save as JPEG quality 90."""
    target = os.path.join(directory, "normalised.jpg")
    with Image.open(path) as im:
        im.convert("RGB").save(target, "JPEG", quality=90)
    return target


def _classifier_scores(model, path):
    wanted = [f for f in model["features"] if f.startswith("clf_")]
    if not wanted:
        return {}
    from .classifiers import MODELS, prob_ai
    with Image.open(path) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=90)
        buf.seek(0)
        img = Image.open(buf)
        return {f"clf_{key}": prob_ai(key, img) for key in MODELS if f"clf_{key}" in wanted}


def _task_report(task, model, scores):
    pred = combiner.predict(model, scores)
    threshold = model.get("threshold", 0.5)
    rows = []
    for c in pred["contributions"]:
        name = c["name"]
        label = LABELS.get(name, {"clf_vit_deepfake": "ViT deepfake classifier",
                                  "clf_swin_sdxl": "Swin SDXL classifier"}.get(name, name))
        rows.append({"name": name, "label": label, "value": c["value"], "weight": c["weight"],
                     "contribution": c["contribution"],
                     "direction": "toward flagged" if c["contribution"] > 0 else "toward not flagged"})
    return {
        "task": task,
        "title": TASK_TITLES[task],
        "probability": pred["probability"],
        "flagged": pred["probability"] >= threshold,
        "threshold": threshold,
        "verdict": combiner.verdict_text(task, pred["probability"], threshold),
        "rows": rows,
        "dropped": model.get("dropped_no_signal", []),
        "test_roc_auc": model.get("test_roc_auc"),
        "test_accuracy": model.get("test_accuracy"),
    }


def analyze_image(path, heatmap_dir=None):
    """Return per-task reports plus every detector's raw score.

    A task is omitted if its weight file is missing (train it with scripts/train_combiner.py).
    """
    raw = run_detectors(path, heatmap_dir=heatmap_dir) if heatmap_dir else run_detectors(path)
    raw_scores = {name: float(r["score"]) for name, r in raw.items()}

    tasks = {}
    for task in MODEL_FILES:
        model = _load(task)
        if model is None:
            continue
        scores = dict(raw_scores)
        if model.get("condition") == "jpeg90":
            with tempfile.TemporaryDirectory() as tmp:
                copy = _jpeg90_copy(path, tmp)
                normalised = run_detectors(copy)
                scores = {n: float(r["score"]) for n, r in normalised.items()}
                scores.update(_classifier_scores(model, path))
        tasks[task] = _task_report(task, model, scores)

    detectors = [{"name": n, "label": LABELS[n], "score": raw_scores[n],
                  "applicable": raw[n].get("applicable", True), "error": raw[n].get("error"),
                  "details": raw[n].get("details")} for n in DETECTORS]
    return {"tasks": tasks, "detectors": detectors,
            "heatmap_path": raw.get("ela", {}).get("heatmap_path")}
