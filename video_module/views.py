import json
import logging
import os

from django.conf import settings
from django.shortcuts import render

from image_module.uploads import saved_temp

from .analyzer import evaluate_video_final

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def home(request):
    context = {}
    if request.method == "POST":
        video = request.FILES.get("video")
        if video is None:
            context["error"] = "Choose a video first."
        else:
            ext = os.path.splitext(video.name)[1].lower()
            if ext not in VIDEO_EXTENSIONS:
                context["error"] = "Unsupported file type. Use MP4, AVI, MOV, MKV or WebM."
            elif video.size > settings.MAX_VIDEO_UPLOAD_BYTES:
                context["error"] = f"File is larger than {settings.MAX_VIDEO_UPLOAD_BYTES // (1024 * 1024)} MB."
            else:
                try:
                    with saved_temp(video, ext) as (_, path):
                        result = evaluate_video_final(path)
                    if "error" in result:
                        context["error"] = result["error"]
                    else:
                        context["result"] = result
                        request.session["last_context"] = json.dumps({"tasks": {"video": {
                            "verdict": result["verdict"], "probability": result["flagged_frame_percent"] / 100,
                            "evidence": [result["strongest_check"]]}}})
                except Exception:
                    logger.exception("Video analysis failed")
                    context["error"] = "Analysis failed on this video."
    return render(request, "video_module/video.html", context)
