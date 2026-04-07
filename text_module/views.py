from django.shortcuts import render
from .analyzer import analyze_text
import logging

logger = logging.getLogger(__name__)

def home(request):
    context = {}
    
    if request.method == 'POST':
        text_content = request.POST.get('text_content', '').strip()
        context['submitted_text'] = text_content
        
        if not text_content:
            context['error'] = "Please paste some text to analyze."
        else:
            try:
                result = analyze_text(text_content)
                if "error" in result:
                    context['error'] = result['error']
                else:
                    context['result'] = result
                    import json
                    request.session['last_image_path'] = None
                    request.session['last_context'] = json.dumps({
                        "text_analyzed": text_content[:500] + ("..." if len(text_content) > 500 else ""),
                        "verdict": "AI-WRITTEN" if result.get('is_ai') else "HUMAN-WRITTEN",
                        "explanation": result.get('explanation'),
                        "plagiarism": result.get('plagiarism_message')
                    })
            except Exception as e:
                logger.error(f"Text UI Error: {e}")
                context['error'] = "An unexpected error occurred during analysis."

    return render(request, 'text_module/text.html', context)
