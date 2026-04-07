import os
import json
from google import genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    env_path = os.path.join('c:\\Users\\LENOVO\\projects\\drishti_ai', '.env')
    with open(env_path, 'r') as f:
        for line in f:
            if line.strip().startswith('GEMINI_API_KEY='):
                api_key = line.split('=', 1)[1].strip()

try:
    client = genai.Client(api_key=api_key)
    res = client.models.generate_content(
        model='gemini-2.5-flash',
        contents="Say hello"
    )
    print("GEMINI WORKS: " + res.text)
except Exception as e:
    print("GEMINI FAILED: " + str(e))
