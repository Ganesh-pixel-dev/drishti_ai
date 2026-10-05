"""Unit tests for the image detectors on small synthetic fixtures.

The fixtures are generated here (smoothed random texture); no benchmark data is used.
"""
import math
import os
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_module.detectors.compression_anomaly import detect_compression_anomaly  # noqa: E402
from image_module.detectors.ela import ela_difference, perform_ela  # noqa: E402
from image_module.detectors.exif_analyzer import analyze_exif  # noqa: E402
from image_module.detectors.patch import detect_patch_repetition  # noqa: E402
from image_module.pipeline import DETECTORS, run_detectors  # noqa: E402


def texture(size=320, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 60, (size, size, 3)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    base = (base - base.min()) / (base.max() - base.min()) * 255
    return base.astype(np.uint8)  # RGB


@pytest.fixture
def jpeg_photo(tmp_path):
    path = tmp_path / "photo.jpg"
    Image.fromarray(texture()).save(path, "JPEG", quality=80)
    return str(path)


@pytest.fixture
def constant_image(tmp_path):
    path = tmp_path / "flat.png"
    Image.fromarray(np.full((128, 128, 3), 120, np.uint8)).save(path)
    return str(path)


@pytest.fixture
def tiny_image(tmp_path):
    path = tmp_path / "tiny.png"
    Image.fromarray(texture(8)).save(path)
    return str(path)


@pytest.fixture
def gray_png(tmp_path):
    path = tmp_path / "gray.png"
    Image.fromarray(texture()[:, :, 0]).save(path)
    return str(path)


@pytest.fixture
def rgba_png(tmp_path):
    path = tmp_path / "rgba.png"
    rgba = np.dstack([texture(), np.full((320, 320), 255, np.uint8)])
    Image.fromarray(rgba, "RGBA").save(path)
    return str(path)


def assert_valid(result):
    assert set(result) >= {"score", "applicable"}
    assert isinstance(result["score"], float)
    assert math.isfinite(result["score"])
    assert 0.0 <= result["score"] <= 1.0


@pytest.mark.parametrize("name", list(DETECTORS))
def test_each_detector_returns_valid_score(name, jpeg_photo, constant_image, tiny_image,
                                           gray_png, rgba_png):
    for path in (jpeg_photo, constant_image, tiny_image, gray_png, rgba_png):
        assert_valid(DETECTORS[name](path))


def test_run_detectors_survives_garbage_file(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    results = run_detectors(str(bad))
    assert set(results) == set(DETECTORS)
    for r in results.values():
        assert_valid(r)
        assert r["applicable"] is False


def test_non_ascii_path(tmp_path):
    path = tmp_path / "foto_é_日本.jpg"
    Image.fromarray(texture()).save(path, "JPEG", quality=80)
    assert_valid(DETECTORS["noise"](str(path)))


def test_ela_uses_rgb_channels_in_order():
    # A pure red image and a pure blue image must give different per-channel ELA.
    red = np.zeros((64, 64, 3), np.uint8)
    red[:, :, 2] = 255  # BGR red
    blue = np.zeros((64, 64, 3), np.uint8)
    blue[:, :, 0] = 255
    assert ela_difference(red).shape == (64, 64, 3)
    assert not np.array_equal(ela_difference(red), ela_difference(blue))


def test_ela_keeps_full_resolution_and_writes_heatmap(jpeg_photo, tmp_path):
    out = perform_ela(jpeg_photo, heatmap_dir=str(tmp_path / "maps"))
    assert os.path.exists(out["heatmap_path"])
    assert cv2.imread(out["heatmap_path"]).shape[:2] == (320, 320)
    assert not list(tmp_path.glob("temp_ela_*")), "ELA must not leave temp files"


def test_ela_flags_local_recompression_more_than_clean(tmp_path):
    clean = texture(384, seed=3)
    clean_path = tmp_path / "clean.jpg"
    Image.fromarray(clean).save(clean_path, "JPEG", quality=95)

    base = np.asarray(Image.open(clean_path).convert("RGB")).copy()
    patch = texture(128, seed=9)
    tmp = tmp_path / "patch.jpg"
    Image.fromarray(patch).save(tmp, "JPEG", quality=30)
    base[100:228, 100:228] = np.asarray(Image.open(tmp).convert("RGB"))
    spliced_path = tmp_path / "spliced.png"
    Image.fromarray(base).save(spliced_path)

    clean_score = perform_ela(str(clean_path))["score"]
    spliced_score = perform_ela(str(spliced_path))["score"]
    assert spliced_score > clean_score


def test_block_grid_is_weaker_after_resize(tmp_path):
    jpg = tmp_path / "grid.jpg"
    Image.fromarray(texture(400, seed=5)).save(jpg, "JPEG", quality=60)
    original = detect_compression_anomaly(str(jpg))

    resized = Image.open(jpg).convert("RGB").resize((357, 357), Image.BICUBIC)
    png = tmp_path / "resized.png"
    resized.save(png)
    after = detect_compression_anomaly(str(png))

    assert original["grid_dominance"] > after["grid_dominance"]
    assert original["score"] < after["score"]


def test_patch_detector_finds_cloned_region(tmp_path):
    img = texture(400, seed=11).copy()
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(0, 255, img.shape).astype(np.uint8), (0, 0), 1.2)
    plain = tmp_path / "plain.png"
    Image.fromarray(img).save(plain)

    cloned = img.copy()
    cloned[250:350, 250:350] = img[20:120, 20:120]
    cloned_path = tmp_path / "cloned.png"
    Image.fromarray(cloned).save(cloned_path)

    assert detect_patch_repetition(str(cloned_path))["matched_pairs"] > \
        detect_patch_repetition(str(plain))["matched_pairs"]


def test_exif_without_metadata_is_not_applicable(gray_png):
    result = analyze_exif(gray_png)
    assert result["applicable"] is False


def test_exif_editing_software_raises_score(tmp_path):
    path = tmp_path / "edited.jpg"
    img = Image.fromarray(texture())
    exif = Image.Exif()
    exif[0x0131] = "Adobe Photoshop 25.0"  # Software
    img.save(path, "JPEG", exif=exif)
    result = analyze_exif(str(path))
    assert result["applicable"] is True
    assert result["score"] >= 0.5


def test_constant_image_gives_no_nan_anywhere(constant_image):
    for name, result in run_detectors(constant_image).items():
        assert math.isfinite(result["score"]), name
