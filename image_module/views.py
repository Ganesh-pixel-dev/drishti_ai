import json
import logging
import os

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .analysis import analyze_image
from .chat_engine import generate_chat_response
from .uploads import UploadError, data_uri, saved_temp, validate_image

logger = logging.getLogger(__name__)


def home(request):
    context = {}
    if request.method == "POST":
        upload = request.FILES.get("image")
        if upload is None:
            context["error"] = "Choose an image first."
            return render(request, "image_module/home.html", context)
        try:
            ext = validate_image(upload)
            with saved_temp(upload, ext) as (directory, path):
                report = analyze_image(path, heatmap_dir=os.path.join(directory, "maps"))
                context["image_uri"] = data_uri(path)
                if report["heatmap_path"]:
                    context["ela_uri"] = data_uri(report["heatmap_path"])
            context["report"] = report
            context["filename"] = os.path.basename(upload.name)
            request.session["last_context"] = json.dumps({
                "tasks": {k: {"verdict": t["verdict"], "probability": round(t["probability"], 3),
                              "evidence": [r["label"] for r in t["rows"][:4]]}
                          for k, t in report["tasks"].items()},
            })
        except UploadError as exc:
            context["error"] = str(exc)
        except Exception:
            logger.exception("Image analysis failed")
            context["error"] = "Analysis failed on this image. Try another file."
    return render(request, "image_module/home.html", context)


@require_POST
def chat_api(request):
    message = request.POST.get("message", "")
    if not message and request.content_type == "application/json":
        try:
            message = json.loads(request.body).get("message", "")
        except (json.JSONDecodeError, AttributeError):
            message = ""
    message = (message or "").strip()[:500]
    if not message:
        return JsonResponse({"error": "Empty message."}, status=400)
    try:
        reply = generate_chat_response(message, request.session.get("last_context", ""))
    except Exception:
        logger.exception("Chat failed")
        return JsonResponse({"error": "Server error while processing chat."}, status=500)
    return JsonResponse({"reply": reply})
