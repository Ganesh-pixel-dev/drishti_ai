import logging
import os
from PIL import Image

logger = logging.getLogger(__name__)

def get_api_key():
    # 1. Check environment normally
    key = os.getenv("GEMINI_API_KEY")
    if key: return key
    # 2. Check the .env file directly (bypasses Windows shell issues)
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip().startswith('GEMINI_API_KEY='):
                    return line.split('=', 1)[1].strip()
    return None

# Initialize HF Pipeline — using a BETTER model
try:
    from transformers import pipeline
    logger.info("Loading AI Forgery Detection Model (Organika/sdxl-detector)...")
    classifier = pipeline("image-classification", model="Organika/sdxl-detector")
    MODEL_LOADED = True
    logger.info("AI Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load HF model: {e}")
    MODEL_LOADED = False

def predict_image(image_path):
    """
    Analyzes an image using Gemini Vision (if available), then falls back to local HF model.
    Returns: Tuple(verdict_string, confidence_float_0_to_1, edited_region_string)
    """
    # 1. Try Google Gemini Vision first
    try:
        api_key = get_api_key()
        if api_key:
            from google import genai
            import json
            client = genai.Client(api_key=api_key)
            img = Image.open(image_path).convert('RGB')
            
            prompt = """You are an elite Digital Forensics Expert. Analyze this image to determine two things:
1. Is it purely AI-generated (Midjourney, DALL-E) or a real photograph?
CRITICAL RULE: If the image contains physically impossible geometry, dream-like surrealism, bizarre visual concepts (like a 6-headed driver in a G-Wagon), plastic-looking lighting, or perfectly smooth digital textures, DO NOT call it a "real photograph with photoshop". It is an AI-GENERATED image. 
2. Only if the base image looks like a completely normal, boring, everyday photograph but has a single awkwardly spliced object, call it a real photograph with Photoshop manipulation.

Respond EXCLUSIVELY with valid JSON.
Format exactly like this:
{"verdict": "AI-GENERATED", "confidence": 0.98, "edited_region": "None"}
or if it's a real photo that was photoshopped/face-swapped:
{"verdict": "REAL", "confidence": 0.95, "edited_region": "Face"}
or if it's completely authentic with no edits:
{"verdict": "REAL", "confidence": 0.99, "edited_region": "None"}"""
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[img, prompt]
            )
            
            raw_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            try:
                data = json.loads(raw_text)
                v = data.get("verdict", "").upper()
                c = float(data.get("confidence", 0.0))
                region = data.get("edited_region", "None")
                if str(region).lower() == "none" or not region:
                    region = None
                
                if "AI" in v or "GENERATED" in v:
                    return "AI-generated", c, region
                else:
                    return "Likely Real", c, region
            except json.JSONDecodeError:
                if "AI-GENERATED" in raw_text.upper():
                    return "AI-generated", 0.85, None
                elif "REAL" in raw_text.upper():
                    return "Likely Real", 0.85, None
            
    except Exception as e:
        logger.warning(f"Gemini failed, falling back to local HF model. Error: {e}")

    # 2. Fallback to Local Hugging Face Pipeline
    try:
        if not MODEL_LOADED:
            return "Suspicious", 0.65, None
            
        image = Image.open(image_path).convert('RGB')
        results = classifier(image)
        
        top_result = results[0]
        label = top_result['label'].upper()
        confidence = top_result['score']

        # capcheck model uses "Fake" / "Real" labels
        if "FAKE" in label or "AI" in label or "SYNTHETIC" in label or "ARTIFICIAL" in label or "GENERATED" in label:
            verdict = "AI-generated"
        elif "HUMAN" in label or "REAL" in label:
            verdict = "Likely Real"
        else:
            verdict = "Likely Real"
            
        return verdict, confidence, None

    except Exception as e:
        logger.error(f"Error in predict_image during local inference: {e}", exc_info=True)
        return "Unknown", 0.0, None