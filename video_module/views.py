from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .analyzer import evaluate_video_final
import os
import uuid
import logging

logger = logging.getLogger(__name__)

def home(request):
    context = {}
    
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        
        # Security: validate extension
        ext = os.path.splitext(video.name)[1].lower()
        if ext not in ['.mp4', '.avi', '.mov', '.mkv']:
            context['error'] = "Invalid file type. Only MP4, AVI, MOV, MKV are allowed."
            return render(request, 'video_module/video.html', context)

        # Unique filename
        safe_filename = f"vid_{uuid.uuid4().hex}{ext}"
        fs = FileSystemStorage()
        filename = fs.save(safe_filename, video)
        file_url = fs.url(filename)
        video_path = os.path.join(fs.location, filename)

        try:
            # Analyze
            result = evaluate_video_final(video_path, max_duration=60)
            
            if "error" in result:
                context['error'] = result['error']
            else:
                context['result'] = result
                import json
                request.session['last_image_path'] = None
                request.session['last_context'] = json.dumps({
                    "video_duration_seconds": result.get('duration'),
                    "verdict": result.get('verdict'),
                    "ai_ratio_percent": result.get('ai_ratio_percent'),
                    "jitter_score": result.get('jitter_score'),
                    "frames_analyzed": result.get('frames_analyzed')
                })
                
        except Exception as e:
            logger.error(f"Video analysis failed: {str(e)}", exc_info=True)
            context['error'] = "Server error during video deepfake breakdown."
            
        context['video_url'] = file_url

    return render(request, 'video_module/video.html', context)
