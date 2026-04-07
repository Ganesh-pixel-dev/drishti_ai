import logging
import os
from PIL import Image
from django.conf import settings

logger = logging.getLogger(__name__)

def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key: return key
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip().startswith('GEMINI_API_KEY='):
                    return line.split('=', 1)[1].strip()
    return None

def generate_chat_response(prompt, image_path, context_data):
    """
    Calls Google Gemini to act as a forensics analyst.
    Requires GEMINI_API_KEY environment variable.
    """
    try:
        from google import genai
        # Pull key robustly
        api_key = get_api_key()
        
        if not api_key:
            return "Setup Required: I cannot analyze this until you set the GEMINI_API_KEY environment variable."

        client = genai.Client(api_key=api_key)
        
        # Load image for multimodal context
        try:
            img = Image.open(image_path)
        except Exception:
            img = None
            
        system_instructions = f"""You are Detective Drishti, the highly skilled digital forensic investigator working for the Drishti AI Forensic Agency.
The user has uploaded a file and our backend has mathematically analyzed it.
Backend Analysis Results: {context_data}

IMPORTANT RULES FOR YOUR PERSONA (Detective Drishti):
1. Be professional, direct, and straightforward. Do not act overly goofy and do not use slang.
2. Be extremely concise. Break things down simply in 1-2 short sentences.
3. If the user asks why something was flagged, answer directly: "Because [reason]".
4. Only rely on the Context Data provided above. Do not invent new reasons."""

        full_prompt = f"{system_instructions}\n\nUser Question: {prompt}"
        
        contents = [full_prompt]
        if img:
            contents.insert(0, img)
            
        # We use gemini-2.5-flash
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents
        )
        
        return response.text
        
    except Exception as e:
        logger.error(f"Chat engine error: {e}", exc_info=True)
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            return "⚠️ I'm currently over my daily API quota. The analysis is still saved — please try asking again in a few minutes!"
        elif "401" in err or "API_KEY" in err or "invalid" in err.lower():
            return "🔑 Authentication error: The AI API key appears to be invalid. Please contact the administrator."
        else:
            return "😔 Something went wrong on my end. Please try again in a moment."
