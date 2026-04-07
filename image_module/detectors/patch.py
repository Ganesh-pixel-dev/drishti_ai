import cv2
import numpy as np

def detect_patch_repetition(image_path):
    image = cv2.imread(image_path, 0)

    result = cv2.matchTemplate(image, image, cv2.TM_CCOEFF_NORMED)

    repetitions = np.sum(result > 0.9)

    score = repetitions / (image.shape[0] * image.shape[1])

    return {
        "score": float(score),
        "map": result.tolist()
    }