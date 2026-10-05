import numpy as np

from ._common import load_gray, not_applicable, result

_SIZE = 256


def frequency_analysis(image_path):
    """Share of spectral energy in the highest frequencies of a centre crop.

    The crop is not resized, so the measure is about pixel-level texture, not about
    how large the file is. Higher score means more high-frequency energy.
    """
    gray = load_gray(image_path).astype(np.float32)
    h, w = gray.shape
    if h < _SIZE or w < _SIZE:
        return not_applicable("Image smaller than 256 px")
    top, left = (h - _SIZE) // 2, (w - _SIZE) // 2
    crop = gray[top:top + _SIZE, left:left + _SIZE]
    crop = crop * np.outer(np.hanning(_SIZE), np.hanning(_SIZE))

    power = np.abs(np.fft.fftshift(np.fft.fft2(crop))) ** 2
    yy, xx = np.indices(power.shape)
    radius = np.hypot(yy - _SIZE / 2, xx - _SIZE / 2)
    total = power.sum() - power[_SIZE // 2, _SIZE // 2]
    high = power[radius > _SIZE * 0.25].sum()
    share = high / (total + 1e-9)
    # Typical photos keep well under 1% of their energy up here; map 0..2% to 0..1.
    return result(share / 0.02, high_freq_share=share)
