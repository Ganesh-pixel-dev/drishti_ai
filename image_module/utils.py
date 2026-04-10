import logging
import os
from .detectors.ela import perform_ela
from .detectors.noise import detect_noise
from .detectors.frequency import frequency_analysis
from .detectors.patch import detect_patch_repetition
from .detectors.luminance_gradient import detect_luminance_gradient
from .detectors.median_filter import detect_median_filtering
from .detectors.histogram_kurtosis import detect_histogram_anomaly
from .detectors.cfa_artifacts import detect_cfa_artifacts
from .detectors.wavelet_stats import detect_wavelet_anomalies
from .detectors.compression_anomaly import detect_compression_anomaly
from .detectors.exif_analyzer import analyze_exif
from .detectors.reflection_integrity import detect_reflection_inconsistency
from .detectors.adversarial_noise import detect_adversarial_noise
from .detectors.geometric_symmetry import detect_geometric_warping
from .detectors.face_boundary_blend import analyze_face_boundary
from .ai_model import predict_image
from .local_models import predict_local_ml

from .decision_engine import analyze_results

logger = logging.getLogger(__name__)

def run_full_analysis(image_path, metadata="Generic", jitter=1.0):
    """
    Executes the complete suite of 20+ local detectors.
    """
    results = {}

    def safe_run(detector_func, name):
        try:
            return detector_func(image_path)
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            return {"score": 0.0}

    # 🧪 PHASE 1: PHYSICS & SIGNAL PROCESSING
    results["ela"] = safe_run(perform_ela, "ELA")
    results["noise"] = safe_run(detect_noise, "Noise")
    results["frequency"] = safe_run(frequency_analysis, "Frequency")
    results["patch"] = safe_run(detect_patch_repetition, "Patch")
    results["luminance"] = safe_run(detect_luminance_gradient, "Luminance")
    results["median"] = safe_run(detect_median_filtering, "Median")
    results["histogram"] = safe_run(detect_histogram_anomaly, "Histogram")
    results["cfa"] = safe_run(detect_cfa_artifacts, "CFA")
    results["wavelet"] = safe_run(detect_wavelet_anomalies, "Wavelets")
    results["compression"] = safe_run(detect_compression_anomaly, "Compression")
    results["exif"] = safe_run(analyze_exif, "EXIF")
    results["reflection"] = safe_run(detect_reflection_inconsistency, "Reflection")
    results["adversarial"] = safe_run(detect_adversarial_noise, "Adversarial")
    results["geometric"] = safe_run(detect_geometric_warping, "Geometric")
    results["boundary"] = safe_run(analyze_face_boundary, "FaceBoundary")

    # 🤖 PHASE 2: AI CORE (GEMINI + LOCAL ML)
    try:
        # 1. Try Gemini (Vision)
        gemini_verdict, gemini_conf, gemini_region = predict_image(image_path)
        
        # 2. Try Local ML (Texture) - especially if Gemini is 'cooked'
        local_verdict, local_conf = predict_local_ml(image_path, metadata=metadata)
        
        # Consensus between AI models
        if "AI" in local_verdict and gemini_conf < 0.2:
            # Local ML caught it while Gemini was unsure
            ai_verdict = local_verdict
            ai_conf = local_conf
        else:
            ai_verdict = gemini_verdict
            ai_conf = gemini_conf

        results["ai_core"] = {
            "verdict": ai_verdict,
            "confidence": ai_conf,
            "edited_region": gemini_region,
            "local_ml_score": local_conf
        }
    except Exception as e:
        logger.error(f"AI layers failed: {e}")
        results["ai_core"] = {"verdict": "Unknown", "confidence": 0.0}

    # 🧠 PHASE 3: FINAL DECISION FUSION
    return analyze_results(results, metadata=metadata, jitter=jitter)