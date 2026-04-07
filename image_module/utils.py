import logging
from .detectors.ela import perform_ela
from .detectors.noise import detect_noise
from .detectors.frequency import frequency_analysis
from .detectors.patch import detect_patch_repetition
from .ai_model import predict_image

from .decision_engine import analyze_results

logger = logging.getLogger(__name__)

def run_full_analysis(image_path):
    results = {}

    # 🔒 Safe execution wrapper
    def safe_run(detector_func, name):
        try:
            return detector_func(image_path)
        except Exception as e:
            logger.error(f"{name} failed to process {image_path}: {e}", exc_info=True)
            return {
                "score": 0.0,
                "map": None
            }

    # 🧪 Run all detectors safely
    results["ela"] = safe_run(perform_ela, "ELA")
    results["noise"] = safe_run(detect_noise, "Noise")
    results["frequency"] = safe_run(frequency_analysis, "Frequency")
    results["patch"] = safe_run(detect_patch_repetition, "Patch")

    # 🤖 Run AI Core Model
    try:
        ai_verdict, ai_confidence, edited_region = predict_image(image_path)
        results["ai_core"] = {
            "verdict": ai_verdict,
            "confidence": ai_confidence,
            "edited_region": edited_region
        }
    except Exception as e:
        logger.error(f"AI Core model failed: {e}", exc_info=True)
        results["ai_core"] = {
            "verdict": "Unknown",
            "confidence": 0.0,
            "edited_region": None
        }

    # 🧠 Final decision
    final_result = analyze_results(results)

    return final_result