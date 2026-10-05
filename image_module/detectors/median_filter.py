import numpy as np

from ._common import load_gray, not_applicable, result


def detect_median_filtering(image_path):
    """Excess of zero pixel differences relative to differences of 1 and 2.

    Median filtering flattens small steps, which raises P(0) against P(1)+P(2) in
    the first-order difference histogram.
    """
    img = load_gray(image_path).astype(np.int16)
    if min(img.shape) < 16:
        return not_applicable("Image too small")

    ratios = []
    for diff in (np.abs(img[:, :-1] - img[:, 1:]), np.abs(img[:-1, :] - img[1:, :])):
        hist = np.bincount(diff.ravel(), minlength=3) / diff.size
        ratios.append(hist[0] / (hist[1] + hist[2] + 1e-6))
    ratio = float(np.mean(ratios))

    # Sigmoid centred on a ratio of 4, clipped so the exponent cannot overflow.
    score = 1.0 / (1.0 + np.exp(-np.clip(ratio - 4.0, -50, 50)))
    return result(score, ratio=ratio)
