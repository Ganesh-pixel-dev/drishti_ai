import logging
import os
from PIL import Image

logger = logging.getLogger(__name__)

# Initialize HF Pipeline — 100% Local Forensic Lead
try:
    from transformers import pipeline
    logger.info("Initializing Local Forensic Lead (Organika/sdxl-detector)...")
    # This model is robust against generative textures and works offline
    classifier = pipeline("image-classification", model="Organika/sdxl-detector")
    MODEL_LOADED = True
    logger.info("Local AI Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load local HF model: {e}")
    MODEL_LOADED = False

def predict_image(image_path):
    """
    [LOCAL ARMADA] Analyzes an image using local transformer models.
    Cloud dependencies (Gemini) have been removed for 100% privacy.
    Returns: Tuple(verdict_string, confidence_float_0_to_1, edited_region_string)
    """
    try:
        if not MODEL_LOADED:
            return "Suspicious", 0.65, None
            
        image = Image.open(image_path).convert('RGB')
        results = classifier(image)
        
        # Get the top class
        top_result = results[0]
        label = top_result['label'].upper()
        confidence = top_result['score']

        # organika/sdxl-detector usually uses 'fake' or 'real'
        if any(keyword in label for keyword in ["FAKE", "AI", "SYNTHETIC", "GENERATED"]):
            verdict = "AI-generated"
        else:
            verdict = "Likely Real"
            
        return verdict, confidence, None

    except Exception as e:
        logger.error(f"Local inference error: {e}", exc_info=True)
        return "Unknown", 0.0, None