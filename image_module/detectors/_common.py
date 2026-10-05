"""Shared helpers for the image detectors.

Every detector takes a file path and returns a dict with a "score" in [0, 1]
(higher means more suspicious) and an "applicable" flag. A detector that cannot
say anything about an image (no face found, image too small) returns
applicable=False and a score of 0.0, so callers can tell "clean" from "no opinion".
"""
import math

import cv2
import numpy as np

_FACE_CASCADE = None
_EYE_CASCADE = None


def load_bgr(path):
    """Read an image as uint8 BGR without resizing. Works with non-ASCII Windows paths."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def load_gray(path):
    return cv2.cvtColor(load_bgr(path), cv2.COLOR_BGR2GRAY)


def clip01(value):
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return min(max(value, 0.0), 1.0)


def result(score, **details):
    out = {"score": clip01(score), "applicable": True}
    for key, value in details.items():
        out[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
    return out


def not_applicable(reason):
    return {"score": 0.0, "applicable": False, "details": reason}


def face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _FACE_CASCADE


def eye_cascade():
    global _EYE_CASCADE
    if _EYE_CASCADE is None:
        _EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    return _EYE_CASCADE


def largest_face(gray):
    faces = face_cascade().detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])
