"""Fit the per-task logistic-regression combiner and measure it on held-out data.

    python scripts/train_combiner.py edit
    python scripts/train_combiner.py ai --condition raw

Reads data/scores_<task>_<condition>.csv (from extract_scores.py).

Procedure (all choices made on the calibration split only):
  1. Per-detector ROC AUC on the calibration split. A detector is kept only if its
     AUC is at least MIN_EDGE away from 0.5 and its score is not constant.
  2. Standardise the kept scores, fit L2 logistic regression with balanced class
     weights. C is picked by 5-fold cross-validation on the calibration split.
  3. (ai task) Repeat with the two Hugging Face classifiers as extra features and use
     them only if cross-validated AUC on the calibration split improves by CV_GAIN.
  4. Evaluate once on the held-out test split (and, for edit, on the external sets).
Writes image_module/weights/<task>_<condition>.json and metrics/<task>_<condition>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from image_module.pipeline import DETECTORS  # noqa: E402

SEED = 1337
MIN_EDGE = 0.03
CV_GAIN = 0.01
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
CLASSIFIERS = ["clf_vit_deepfake", "clf_swin_sdxl"]


def auc(y, s):
    y = np.asarray(y)
    if len(set(y)) < 2 or np.nanstd(s) == 0:
        return float("nan")
    return float(roc_auc_score(y, s))


def boot_ci(y, s, n=1000):
    rng = np.random.default_rng(SEED)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], s[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def fit(X, y, c):
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=SEED)
    model.fit(scaler.transform(X), y)
    return scaler, model


def pick_c(X, y):
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best = None
    for c in C_GRID:
        pipe_scores = np.zeros(len(y))
        for tr, va in folds.split(X, y):
            scaler, model = fit(X[tr], y[tr], c)
            pipe_scores[va] = model.decision_function(scaler.transform(X[va]))
        score = roc_auc_score(y, pipe_scores)
        if best is None or score > best[1] + 1e-9:
            best = (c, score)
    return best


def summarise(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "n_positive": int(np.sum(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": auc(y, prob), "roc_auc_ci95": boot_ci(y, prob),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "baseline_accuracy_majority_class": float(max(np.mean(y), 1 - np.mean(y))),
    }


def metadata_baseline(cal, test):
    """How well file facts alone (original extension, size) separate the classes."""
    def feats(df):
        ext = pd.get_dummies(df["orig_ext"].astype(str)).reindex(
            columns=sorted(set(cal["orig_ext"].astype(str))), fill_value=0)
        return np.column_stack([ext.values, np.log(df["width"]), np.log(df["height"])])
    scaler, model = fit(feats(cal), cal["label"].values, 1.0)
    prob = model.predict_proba(scaler.transform(feats(test)))[:, 1]
    return summarise(test["label"].values, prob)


def run(task, condition, only_ext=None):
    df = pd.read_csv(ROOT / "data" / f"scores_{task}_{condition}.csv")
    tag = condition
    if only_ext:
        # Keep only images whose original file was this format (calibration and test),
        # so a model cannot win by learning "was this ever a JPEG".
        df = df[(df["orig_ext"] == only_ext) | (df["split"] == "external_test")]
        tag = f"{condition}_{only_ext}only"
    cal = df[df["split"] == "cal"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    external = df[df["split"] == "external_test"].reset_index(drop=True)
    names = list(DETECTORS)

    per_detector = {}
    kept = []
    for name in names:
        cal_auc = auc(cal["label"], cal[name])
        test_auc = auc(test["label"], test[name])
        ext_auc = auc(external["label"], external[name]) if len(external) else None
        keep = bool(np.isfinite(cal_auc) and abs(cal_auc - 0.5) >= MIN_EDGE)
        per_detector[name] = {"auc_cal": cal_auc, "auc_test": test_auc, "auc_external": ext_auc,
                              "kept": keep, "inapplicable_share_cal": float(cal.get(f"na_{name}", pd.Series([0])).mean())}
        if keep:
            kept.append(name)

    def evaluate_set(features):
        X = cal[features].values.astype(float)
        c, cv_auc = pick_c(X, cal["label"].values)
        return c, cv_auc

    variants = {"detectors": kept}
    if task == "ai":
        present = [c for c in CLASSIFIERS if c in df and df[c].notna().all()]
        clf_per = {c: {"auc_cal": auc(cal["label"], cal[c]), "auc_test": auc(test["label"], test[c])} for c in present}
        # A classifier is only eligible if it points the right way on the calibration split.
        clf_cols = [c for c in present if clf_per[c]["auc_cal"] >= 0.5 + MIN_EDGE]
        variants["detectors+classifiers"] = kept + clf_cols
        variants["classifiers_only"] = clf_cols
    chosen_name, chosen_feats, chosen_c, cv_results = "detectors", kept, None, {}
    for vname, feats in variants.items():
        if not feats:
            continue
        c, cv_auc = evaluate_set(feats)
        cv_results[vname] = {"features": feats, "C": c, "cv_auc_cal": float(cv_auc)}
    if not cv_results:
        raise SystemExit("No detector carries signal on the calibration split.")
    chosen_name = max(cv_results, key=lambda k: cv_results[k]["cv_auc_cal"])
    if task == "ai" and chosen_name != "detectors":
        # classifiers must beat the best detector-only model by CV_GAIN to be kept
        base = cv_results.get("detectors", {"cv_auc_cal": 0})["cv_auc_cal"]
        if cv_results[chosen_name]["cv_auc_cal"] < base + CV_GAIN:
            chosen_name = "detectors"
    chosen = cv_results[chosen_name]
    chosen_feats, chosen_c = chosen["features"], chosen["C"]

    scaler, model = fit(cal[chosen_feats].values.astype(float), cal["label"].values, chosen_c)

    def prob(frame):
        return model.predict_proba(scaler.transform(frame[chosen_feats].values.astype(float)))[:, 1]

    metrics = {
        "task": task, "condition": condition, "only_ext": only_ext, "seed": SEED,
        "split": {"cal": int(len(cal)), "test": int(len(test)), "external_test": int(len(external))},
        "sources": sorted(df["source"].unique().tolist()),
        "selection": {"min_auc_edge_from_0.5": MIN_EDGE, "variants_cv": cv_results, "chosen": chosen_name},
        "per_detector": per_detector,
        "coefficients": {f: float(w) for f, w in zip(chosen_feats, model.coef_[0])},
        "intercept": float(model.intercept_[0]),
        "cal_in_sample": summarise(cal["label"].values, prob(cal)),
        "test": summarise(test["label"].values, prob(test)),
        "metadata_only_baseline_test": metadata_baseline(cal, test),
    }
    if task == "ai":
        metrics["per_classifier"] = clf_per
        metrics["test_per_generator_recall"] = {
            g: float((prob(test[test.group == g]) >= 0.5).mean())
            for g in sorted(test[test.label == 1]["group"].unique())}
        metrics["test_real_specificity"] = float((prob(test[test.label == 0]) < 0.5).mean())
    if len(external):
        metrics["external_test"] = {}
        for src, sub in external.groupby("source"):
            metrics["external_test"][src] = summarise(sub["label"].values, prob(sub))
        metrics["external_test"]["all"] = summarise(external["label"].values, prob(external))
    if task == "edit":
        casia_test = test[test.source == "casia"]
        metrics["test_recall_by_tamper_type"] = {
            g: float((prob(casia_test[casia_test.group == g]) >= 0.5).mean())
            for g in sorted(casia_test[casia_test.label == 1]["group"].unique())}

    out_dir = ROOT / "metrics"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{task}_{tag}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    weights = {
        "task": task, "condition": condition, "only_ext": only_ext, "seed": SEED,
        "features": chosen_feats, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": [float(w) for w in model.coef_[0]], "intercept": float(model.intercept_[0]),
        "threshold": 0.5,
        "dropped_no_signal": [n for n in names if n not in kept],
        "test_roc_auc": metrics["test"]["roc_auc"], "test_accuracy": metrics["test"]["accuracy"],
    }
    wdir = ROOT / "image_module" / "weights"
    wdir.mkdir(exist_ok=True)
    (wdir / f"{task}_{tag}.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("selection", "test")}, indent=2)[:3000])
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["edit", "ai"])
    ap.add_argument("--condition", default="raw")
    ap.add_argument("--only-ext", default=None, help="restrict calibration/test to one original file format")
    a = ap.parse_args()
    run(a.task, a.condition, a.only_ext)
