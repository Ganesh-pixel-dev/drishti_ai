import sys
import os
from PIL import Image

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
    from google import genai
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)
    img = Image.open('c:\\Users\\LENOVO\\projects\\drishti_ai\\media\\fcd90ae1f56a4371bf79e26124ff3101.png').convert('RGB')
    
    prompt = "Analyze this image. Is it AI-generated or a real photograph? Answer precisely with the word 'AI-GENERATED' or 'REAL'. Provide no other text."
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[img, prompt]
    )
    
    with open("c:\\Users\\LENOVO\\projects\\drishti_ai\\debug_output.txt", "w") as f:
        f.write("OUTPUT:\n")
        f.write(response.text)
except Exception as e:
    with open("c:\\Users\\LENOVO\\projects\\drishti_ai\\debug_output.txt", "w") as f:
        f.write("ERROR:\n")
        f.write(str(e))
