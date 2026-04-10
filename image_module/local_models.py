import logging
import torch
from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)

# Global cache for the pipelines to avoid re-loading
_IMAGE_CLASSIFIER_VIT = None
_IMAGE_CLASSIFIER_CNN = None

def get_vit_classifier():
    global _IMAGE_CLASSIFIER_VIT
    if _IMAGE_CLASSIFIER_VIT is None:
        try:
            logger.info("Initializing ViT Deepfake Classifier (dima806)...")
            _IMAGE_CLASSIFIER_VIT = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
        except Exception as e:
            logger.error(f"ViT load failed: {e}")
    return _IMAGE_CLASSIFIER_VIT

def get_cnn_classifier():
    global _IMAGE_CLASSIFIER_CNN
    if _IMAGE_CLASSIFIER_CNN is None:
        try:
            logger.info("Initializing CNN Deepfake Classifier (organika/sdxl-detector)...")
            # This model is robust against generative textures
            _IMAGE_CLASSIFIER_CNN = pipeline("image-classification", model="organika/sdxl-detector")
        except Exception as e:
            logger.error(f"CNN load failed: {e}")
    return _IMAGE_CLASSIFIER_CNN

def predict_local_ml(image_path, metadata="Generic"):
    """
    Ensemble Inference: Runs both ViT and CNN models to prevent adversarial misses.
    """
    vit_clf = get_vit_classifier()
    cnn_clf = get_cnn_classifier()
    
    votes = []
    
    try:
        img = Image.open(image_path).convert("RGB")
        
        # Model 1: ViT
        if vit_clf:
            res_vit = vit_clf(img)
            label = res_vit[0]['label'].lower()
            score = res_vit[0]['score']
            is_ai = "fake" in label or "ai" in label or "generated" in label
            votes.append((is_ai, score))

        # Model 2: CNN
        if cnn_clf:
            res_cnn = cnn_clf(img)
            label = res_cnn[0]['label'].lower()
            score = res_cnn[0]['score']
            is_ai = "fake" in label or "ai" in label or "generated" in label
            votes.append((is_ai, score))

        if not votes: return "Unknown", 0.0

        # ENSEMBLE LOGIC:
        final_verdict = "Likely Real"
        final_score = 0.0

        ai_votes = [v for v in votes if v[0]]
        if len(ai_votes) == 0:
            final_verdict = "Likely Real"
            final_score = 1.0 - max(v[1] for v in votes)
        elif len(ai_votes) == len(votes):
            final_verdict = "AI-generated"
            final_score = max(v[1] for v in votes)
        else:
            final_verdict = "AI-generated (Ensemble Conflict)"
            final_score = max(v[1] for v in votes)

        # [ARMADA v3] WEBCAM DAMPING:
        # If it's a generic webcam, we strictly limit the 'certainty' of texture models.
        # This forces the consensus engine to look for geometric/temporal proof.
        if metadata == "Generic/Webcam" and final_verdict != "Likely Real":
            final_score = min(final_score, 0.85) # Raised from 0.55 to allow deepfakes to be flagged
            
        return final_verdict, final_score

    except Exception as e:
        logger.error(f"Ensemble inference error: {e}")
        return "Error", 0.0
