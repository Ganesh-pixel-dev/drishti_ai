import cv2
import numpy as np

from ._common import largest_face, load_bgr, not_applicable, result


def analyze_face_boundary(image_path):
    """Colour mismatch between the inner face and the skin band around it.

    Face swaps often leave a shift in skin tone where the pasted face meets the
    host. Compares Cr/Cb histograms of the inner face against the surrounding ring
    (the Haar face box padded by 10%, minus the inner face).
    """
    img = load_bgr(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = largest_face(gray)
    if face is None:
        return not_applicable("No face detected")
    x, y, w, h = [int(v) for v in face]
    H, W = gray.shape

    ix1, iy1 = int(x + w * 0.2), int(y + h * 0.2)
    ix2, iy2 = int(x + w * 0.8), int(y + h * 0.8)
    pad = int(w * 0.1)
    ox1, oy1 = max(0, x - pad), max(0, y - pad)
    ox2, oy2 = min(W, x + w + pad), min(H, y + h + pad)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    inner = ycrcb[iy1:iy2, ix1:ix2]
    outer = ycrcb[oy1:oy2, ox1:ox2]
    ring = np.ones(outer.shape[:2], dtype=np.uint8)
    ring[iy1 - oy1:iy2 - oy1, ix1 - ox1:ix2 - ox1] = 0
    if inner.size == 0 or ring.sum() == 0:
        return not_applicable("Face region too small")

    shift = 0.0
    for channel in (1, 2):  # Cr, Cb in YCrCb order
        a = cv2.calcHist([inner], [channel], None, [32], [0, 256])
        b = cv2.calcHist([outer], [channel], ring, [32], [0, 256])
        cv2.normalize(a, a)
        cv2.normalize(b, b)
        shift += cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)

    return result(shift / 0.8, colour_shift=shift)
