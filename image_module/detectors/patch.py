import cv2
import numpy as np

from ._common import load_gray, not_applicable, result

_MAX_SIDE = 1024
_MIN_SEPARATION = 40


def detect_patch_repetition(image_path):
    """Copy-move check: ORB keypoints that match another place in the same image.

    Matches closer than 40 px are ignored (repeating texture, not a clone). Score
    rises with the number of distinct far-apart matches.
    """
    gray = load_gray(image_path)
    h, w = gray.shape
    if min(h, w) < 64:
        return not_applicable("Image too small")
    scale = min(1.0, _MAX_SIDE / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    orb = cv2.ORB_create(nfeatures=2000)
    keypoints, desc = orb.detectAndCompute(gray, None)
    if desc is None or len(keypoints) < 10:
        return not_applicable("Too few keypoints")

    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc, desc, k=3)
    points = np.array([kp.pt for kp in keypoints])

    pairs = set()
    for m in matches:
        if len(m) < 3:
            continue
        # m[0] is the keypoint matching itself; m[1] is its best other match.
        first, second = m[1], m[2]
        if first.distance >= 0.6 * second.distance or first.distance > 48:
            continue
        if np.linalg.norm(points[first.queryIdx] - points[first.trainIdx]) < _MIN_SEPARATION:
            continue
        pairs.add(tuple(sorted((first.queryIdx, first.trainIdx))))

    return result(len(pairs) / 30.0, matched_pairs=len(pairs))
