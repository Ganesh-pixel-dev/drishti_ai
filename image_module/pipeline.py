import logging

from .detectors.adversarial_noise import detect_adversarial_noise
from .detectors.cfa_artifacts import detect_cfa_artifacts
from .detectors.compression_anomaly import detect_compression_anomaly
from .detectors.ela import perform_ela
from .detectors.exif_analyzer import analyze_exif
from .detectors.face_boundary_blend import analyze_face_boundary
from .detectors.frequency import frequency_analysis
from .detectors.geometric_symmetry import detect_geometric_warping
from .detectors.histogram_kurtosis import detect_histogram_anomaly
from .detectors.luminance_gradient import detect_luminance_gradient
from .detectors.median_filter import detect_median_filtering
from .detectors.noise import detect_noise
from .detectors.patch import detect_patch_repetition
from .detectors.reflection_integrity import detect_reflection_inconsistency
from .detectors.wavelet_stats import detect_wavelet_anomalies

logger = logging.getLogger(__name__)

DETECTORS = {
    "ela": perform_ela,
    "cfa": detect_cfa_artifacts,
    "compression": detect_compression_anomaly,
    "noise": detect_noise,
    "median": detect_median_filtering,
    "wavelet": detect_wavelet_anomalies,
    "luminance": detect_luminance_gradient,
    "histogram": detect_histogram_anomaly,
    "patch": detect_patch_repetition,
    "frequency": frequency_analysis,
    "reflection": detect_reflection_inconsistency,
    "adversarial": detect_adversarial_noise,
    "geometric": detect_geometric_warping,
    "boundary": analyze_face_boundary,
    "exif": analyze_exif,
}

LABELS = {
    "ela": "Error level analysis",
    "cfa": "Camera sensor pattern (CFA)",
    "compression": "JPEG 8x8 block grid",
    "noise": "Noise level and uniformity",
    "median": "Median-filter trace",
    "wavelet": "Wavelet sub-band statistics",
    "luminance": "Gradient consistency",
    "histogram": "Histogram kurtosis and gaps",
    "patch": "Copy-move (ORB matches)",
    "frequency": "High-frequency energy",
    "reflection": "Eye highlights",
    "adversarial": "High-pass residual statistics",
    "geometric": "Face symmetry",
    "boundary": "Face boundary colour shift",
    "exif": "EXIF metadata",
}


def run_detectors(image_path, **options):
    """Run every detector. A detector that raises is recorded, never fatal."""
    out = {}
    for name, func in DETECTORS.items():
        try:
            kwargs = options if name == "ela" else {}
            out[name] = func(image_path, **kwargs)
        except Exception as exc:
            logger.warning("%s failed on %s: %s", name, image_path, exc)
            out[name] = {"score": 0.0, "applicable": False, "error": str(exc)}
    return out


def scores_only(results):
    return {name: float(r["score"]) for name, r in results.items()}
