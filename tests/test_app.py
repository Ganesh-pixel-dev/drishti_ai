import io
import json
import os
import tempfile

import cv2
import numpy as np
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from PIL import Image

from image_module import combiner
from image_module.uploads import UploadError, validate_image
from text_module.analyzer import analyze_text
from video_module.analyzer import evaluate_video_final


def jpeg_bytes(size=(160, 120)):
    buf = io.BytesIO()
    Image.fromarray((np.random.default_rng(0).random((*size[::-1], 3)) * 255).astype("uint8")).save(buf, "JPEG")
    return buf.getvalue()


def test_validate_rejects_wrong_extension():
    with pytest.raises(UploadError):
        validate_image(SimpleUploadedFile("a.exe", b"x"))


def test_validate_rejects_fake_image():
    with pytest.raises(UploadError):
        validate_image(SimpleUploadedFile("a.jpg", b"not an image"))


def test_validate_rejects_oversize(settings):
    settings.MAX_IMAGE_UPLOAD_BYTES = 10
    with pytest.raises(UploadError):
        validate_image(SimpleUploadedFile("a.jpg", jpeg_bytes()))


def test_validate_accepts_jpeg():
    assert validate_image(SimpleUploadedFile("a.JPG", jpeg_bytes())) == ".jpg"


@pytest.mark.django_db
def test_upload_page_renders_report_and_leaves_no_files():
    before = set(os.listdir(tempfile.gettempdir()))
    resp = Client().post("/", {"image": SimpleUploadedFile("p.jpg", jpeg_bytes((400, 300)))})
    assert resp.status_code == 200
    assert resp.context.get("error") is None
    assert "report" in resp.context
    after = set(os.listdir(tempfile.gettempdir()))
    assert not [n for n in after - before if n.startswith("drishti_")]


@pytest.mark.django_db
def test_upload_bad_file_gives_error_not_500():
    resp = Client().post("/", {"image": SimpleUploadedFile("p.jpg", b"junk")})
    assert resp.status_code == 200
    assert resp.context["error"]


def test_chat_requires_post_and_message():
    c = Client()
    assert c.get("/api/chat/").status_code == 405
    assert c.post("/api/chat/", {"message": ""}).status_code == 400
    reply = c.post("/api/chat/", {"message": "what is ELA"}).json()["reply"]
    assert "JPEG" in reply


def test_combiner_contributions_sum_to_logit():
    model = {"features": ["a", "b", "c"], "mean": [0.5, 0.5, 0.5], "scale": [0.1, 0.2, 0.1],
             "coef": [1.0, -0.5, 0.0], "intercept": -0.2}
    pred = combiner.predict(model, {"a": 0.7, "b": 0.9, "c": 0.1})
    assert pred["logit"] == pytest.approx(-0.2 + 1.0 * 2 - 0.5 * 2)
    assert sum(c["contribution"] for c in pred["contributions"]) + model["intercept"] == pytest.approx(pred["logit"])
    assert [c["name"] for c in pred["contributions"]] == ["a", "b"]  # zero weight omitted
    assert 0 < pred["probability"] < 1


def test_combiner_ignores_missing_features():
    model = {"features": ["a"], "mean": [0], "scale": [1], "coef": [2.0], "intercept": 0.0}
    assert combiner.predict(model, {})["probability"] == 0.5


def test_verdict_text_never_says_proof():
    for task in ("edit", "ai"):
        for p in (0.1, 0.9):
            assert "proof" not in combiner.verdict_text(task, p, 0.5).lower().replace("not proof", "")


def test_text_analyzer_runs_without_search_key():
    out = analyze_text("It is important to note that this is a test. " * 10)
    assert "is_ai" in out and 0 <= out["ai_confidence"] <= 100
    assert "SERPER" in out["search_message"]


def test_text_analyzer_short_text():
    assert analyze_text("hi")["ai_confidence"] >= 0


def test_video_unreadable_file_returns_error(tmp_path):
    p = tmp_path / "bad.mp4"
    p.write_bytes(b"nope")
    assert "error" in evaluate_video_final(str(p))


def test_video_synthetic_clip_runs(tmp_path):
    p = str(tmp_path / "noise.mp4")
    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
    rng = np.random.default_rng(0)
    for _ in range(100):
        w.write((rng.random((240, 320, 3)) * 255).astype("uint8"))
    w.release()
    out = evaluate_video_final(p)
    assert "error" not in out
    assert out["frames_analyzed"] >= 1
    assert "unvalidated" in out["verdict"]
