import cv2
import numpy as np

from ._common import largest_face, load_gray, not_applicable, result


def detect_geometric_warping(image_path):
    """Left-right symmetry of the detected face.

    A face crop is compared with its mirror image. Very low error (an almost
    perfect mirror) or very high error (heavy distortion) is flagged. This is a
    crude proxy: it also reacts to lighting and head pose.
    """
    gray = load_gray(image_path)
    face = largest_face(gray)
    if face is None:
        return not_applicable("No face detected")
    x, y, w, h = [int(v) for v in face]
    roi = cv2.resize(gray[y:y + h, x:x + w], (256, 256)).astype(np.float32)
    mse = float(np.mean((roi - roi[:, ::-1]) ** 2))

    if mse < 50.0:
        return result(0.75, mse=mse, details="Almost perfectly mirrored face.")
    if mse > 4000.0:
        return result(0.85, mse=mse, details="Extreme left-right difference.")
    return result(0.0, mse=mse, details="Normal asymmetry.")
