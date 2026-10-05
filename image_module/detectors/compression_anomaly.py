import numpy as np

from ._common import load_gray, not_applicable, result


def detect_compression_anomaly(image_path):
    """Measures how strongly the image shows a single 8x8 JPEG block grid.

    A clean JPEG has one grid offset where pixel steps across block borders are
    larger than elsewhere. Spliced, resized or never-JPEG images show a weaker
    or conflicting grid. Higher score means the grid is weaker.
    """
    img = load_gray(image_path).astype(np.int32)
    h, w = img.shape
    if h < 32 or w < 32:
        return not_applicable("Image too small")

    # Mean absolute step across every row border and every column border.
    row_steps = np.abs(img[1:, :] - img[:-1, :]).mean(axis=1)
    col_steps = np.abs(img[:, 1:] - img[:, :-1]).mean(axis=0)

    strengths = np.array([
        [row_steps[oy + 7::8].mean() + col_steps[ox + 7::8].mean() for ox in range(8)]
        for oy in range(8)
    ])
    dominance = (strengths.max() - strengths.mean()) / (strengths.mean() + 1e-6)
    return result(1.0 - min(dominance / 0.2, 1.0), grid_dominance=dominance)
