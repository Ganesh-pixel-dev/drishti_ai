from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from .utils import run_full_analysis
from .chat_engine import generate_chat_response
import os
import uuid
import logging
import json

logger = logging.getLogger(__name__)

def home(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        
        # Security: validate extension
        ext = os.path.splitext(image.name)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png']:
            context['error'] = "Invalid file type. Only JPG and PNG are allowed."
            return render(request, 'image_module/home.html', context)

        # Security: Generate unique filename
        safe_filename = f"{uuid.uuid4().hex}{ext}"

        # Save file
        fs = FileSystemStorage()
        filename = fs.save(safe_filename, image)
        file_url = fs.url(filename)

        image_path = os.path.join(fs.location, filename)

        try:
            # 🔥 FULL MULTI-DETECTOR ANALYSIS
            result = run_full_analysis(image_path)

            # Core Output
            context['verdict'] = result.get("verdict")
            context['confidence'] = result.get("confidence")
            context['explanation'] = result.get("explanation")
            context['edited_region'] = result.get("edited_region")
            
            # Additional Context for Template
            context['ai_result'] = result.get("ai_verdict", "N/A")
            context['ai_conf'] = f"{result.get('ai_confidence', 0):.2f}%"

            # Individual Scores (for debugging/UI)
            ela_data = result.get("details", {}).get("ela", {})
            context['ela_score'] = round(ela_data.get("score", 0), 3)
            context['ela_url'] = ela_data.get("heatmap_url", "")
            
            context['noise_score'] = round(result.get("details", {}).get("noise", {}).get("score", 0), 3)
            context['frequency_score'] = round(result.get("details", {}).get("frequency", {}).get("score", 0), 3)
            context['patch_score'] = round(result.get("details", {}).get("patch", {}).get("score", 0), 3)

            # Store in session for the Chat Bot
            request.session['last_image_path'] = image_path
            request.session['last_context'] = json.dumps({
                "verdict": context['verdict'],
                "ai_verdict": context['ai_result'],
                "confidence": context['confidence'],
                "edited_region": context['edited_region'],
                "explanations": context['explanation'],
                "ela_url": context['ela_url']
            })

        except Exception as e:
            logger.error(f"Analysis failed for {filename}: {str(e)}", exc_info=True)
            context['error'] = "Analysis failed. Please try another image. Check server logs."

        # Image Preview
        context['image_url'] = file_url

    return render(request, 'image_module/home.html', context)

def chat_api(request):
    if request.method == 'POST':
        try:
            # Handle both JSON and FormData depending on how frontend sends it
            user_message = request.POST.get('message')
            if not user_message and request.body:
                try:
                    data = json.loads(request.body)
                    user_message = data.get('message', '')
                except json.JSONDecodeError:
                    pass
            
            image_path = request.session.get('last_image_path')
            context_data = request.session.get('last_context', "No previous context.")
            
            if not user_message:
                return JsonResponse({"error": "Empty message."}, status=400)
                
            reply = generate_chat_response(user_message, image_path, context_data)
            return JsonResponse({"reply": reply})
            
        except Exception as e:
            logger.error(f"Chat API error: {e}", exc_info=True)
            return JsonResponse({"error": "Server error while processing chat."}, status=500)
            
    return JsonResponse({"error": "POST required."}, status=405)