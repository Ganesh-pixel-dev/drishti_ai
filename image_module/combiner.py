"""Learned combiner: logistic regression over detector scores, one model per task.

Weights live in image_module/weights/<task>.json and are written by
scripts/train_combiner.py. Each feature's contribution to the final log-odds is
coef * (score - mean) / scale, so the explanation is exact, not an approximation.
"""
import json
import math
from functools import lru_cache
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
TASKS = ("edit", "ai")


@lru_cache(maxsize=None)
def load_model(task):
    path = WEIGHTS_DIR / f"{task}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))


def predict(model, scores):
    """Probability that the image is edited/AI-generated, plus per-feature contributions.

    `scores` maps feature name to value. Features the model gave zero weight, and
    features missing from `scores`, contribute nothing.
    """
    z = model["intercept"]
    contributions = []
    for name, mean, scale, coef in zip(model["features"], model["mean"], model["scale"], model["coef"]):
        if coef == 0.0 or name not in scores or scores[name] is None:
            continue
        value = float(scores[name])
        part = coef * (value - mean) / scale
        z += part
        contributions.append({"name": name, "value": value, "weight": coef, "contribution": part})
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return {"probability": _sigmoid(z), "logit": z, "contributions": contributions}


def verdict_text(task, probability, threshold):
    """Wording for the report. Never claims proof."""
    flagged = probability >= threshold
    if task == "edit":
        return ("Flagged for human review: possible editing or splicing" if flagged
                else "Not flagged: no strong editing signal found")
    return ("Flagged for human review: possibly AI-generated" if flagged
            else "Not flagged: no strong sign of AI generation")
