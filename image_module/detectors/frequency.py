import cv2
import numpy as np

def frequency_analysis(image_path):
    image = cv2.imread(image_path, 0)

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    magnitude = 20 * np.log(np.abs(fshift) + 1)

    score = np.mean(magnitude) / 255.0

    return {
        "score": float(score),
        "map": magnitude.tolist()
    }