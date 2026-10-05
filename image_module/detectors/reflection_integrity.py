import cv2

from ._common import eye_cascade, largest_face, load_bgr, not_applicable, result


def detect_reflection_inconsistency(image_path):
    """Checks whether both eyes show a bright specular highlight.

    Looks at the two largest eye boxes in the upper half of the largest face.
    Scores 0.4 when neither eye has a highlight and 0.3 when only one has.
    Photos taken without a light source or at low resolution trigger this too.
    """
    img = load_bgr(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = largest_face(gray)
    if face is None:
        return not_applicable("No face detected")
    x, y, w, h = [int(v) for v in face]

    upper = gray[y:y + h // 2 + h // 8, x:x + w]
    eyes = sorted(eye_cascade().detectMultiScale(upper), key=lambda e: e[2] * e[3], reverse=True)
    # Two distinct eyes: second box must sit well to one side of the first.
    pair = None
    for i, a in enumerate(eyes):
        for b in eyes[i + 1:]:
            if abs((a[0] + a[2] / 2) - (b[0] + b[2] / 2)) > w * 0.2:
                pair = (a, b)
                break
        if pair:
            break
    if pair is None:
        return not_applicable("Two eyes not found")

    def has_highlight(box):
        ex, ey, ew, eh = [int(v) for v in box]
        patch = upper[ey:ey + eh, ex:ex + ew]
        return patch.size > 0 and bool((patch >= 230).any())

    lit = [has_highlight(b) for b in pair]
    if not any(lit):
        return result(0.4, details="Neither eye shows a specular highlight.")
    if not all(lit):
        return result(0.3, details="Only one eye shows a specular highlight.")
    return result(0.0, details="Both eyes show a highlight.")
