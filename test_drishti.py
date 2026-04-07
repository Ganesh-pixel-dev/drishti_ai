import sys
import logging
from PIL import Image

logging.basicConfig(level=logging.DEBUG)

image_path = 'c:\\Users\\LENOVO\\projects\\drishti_ai\\media\\fcd90ae1f56a4371bf79e26124ff3101.png'

# Manually recreate ai_model logic
import os

def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key: return key
    env_path = os.path.join('c:\\Users\\LENOVO\\projects\\drishti_ai', '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip().startswith('GEMINI_API_KEY='):
                    return line.split('=', 1)[1].strip()
    return None

try:
    api_key = get_api_key()
    print(f"DEBUG: Found API Key: {api_key is not None}")
    
    if api_key:
        from google import genai
        client = genai.Client(api_key=api_key)
        img = Image.open(image_path).convert('RGB')
        
        prompt = "Analyze this image. Is it AI-generated or a real photograph? Answer precisely with the word 'AI-GENERATED' or 'REAL'. Provide no other text."
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[img, prompt]
        )
        
        text_ans = response.text.upper()
        print(f"DEBUG: Gemini raw text response: '{text_ans}'")
        
    else:
        print("DEBUG: API key was missing, falling back to local model.")
        
except Exception as e:
    import traceback
    traceback.print_exc()
