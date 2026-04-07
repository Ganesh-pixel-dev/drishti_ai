import os
import json
from PIL import Image
from google import genai
import numpy as np

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # 2. Check the .env file directly (bypasses Windows shell issues)
    env_path = os.path.join(r'C:\Users\LENOVO\projects\drishti_ai', '.env')
    with open(env_path, 'r') as f:
        for line in f:
            if line.strip().startswith('GEMINI_API_KEY='):
                api_key = line.split('=', 1)[1].strip()

client = genai.Client(api_key=api_key)

# Create 5 dummy random noisy images
contents = []
for i in range(5):
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    contents.append(img)
    contents.append(f"[Frame {i+1} at {i}.0s]")

prompt = """You are an elite video forensics expert specializing in deepfake detection.
I have extracted 5 keyframes from a video at evenly spaced intervals.
For EACH frame, determine if it looks AI-generated or real.
Then provide an OVERALL verdict for the entire video.

CRITICAL RULE: High-end AI models like Sora, Runway Gen-3, Kling, or Midjourney produce perfectly smooth video with ZERO temporal jitter or flickering. Do NOT assume smooth video = REAL. Look for:
1. Dream-like surrealism, objects morphing, or physically impossible scene geometry.
2. Hyper-polished, plastic-looking lighting, or completely noise-free digital textures.
3. Over-saturated, cinematically "too-perfect" aesthetics.
If you see these signs of purely AI-generated video (Sora, Runway, Pika, etc.), you MUST flag individual frames as "AI-GENERATED" and output "DEEPFAKE" for the overall verdict.

Respond EXCLUSIVELY with valid JSON in this exact format:
{
  "overall_verdict": "DEEPFAKE" or "REAL",
  "overall_confidence": 0.95,
  "frames": [
    {"frame": 1, "verdict": "AI-GENERATED", "confidence": 0.9},
    {"frame": 2, "verdict": "REAL", "confidence": 0.8}
  ]
}
Output ONLY the JSON, no extra text."""

contents.append(prompt)

try:
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=contents
    )
    print("RAW RESPONSE:")
    print(response.text)
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)
    print("LOADS OK:")
    print(json.dumps(data, indent=2))
except Exception as e:
    print(f"FAILED: {e}")
