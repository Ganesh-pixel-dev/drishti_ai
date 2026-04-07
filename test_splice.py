import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'drishti_ai.settings'
import django
django.setup()

from image_module.detectors.ela import perform_ela
from image_module.detectors.noise import detect_noise
from image_module.detectors.frequency import frequency_analysis
from image_module.detectors.patch import detect_patch_repetition
from image_module.ai_model import predict_image
from image_module.decision_engine import analyze_results
import glob

# Find face swap image (most recent jpeg upload)
jpgs = sorted(glob.glob('media/*.jpeg'), key=os.path.getmtime, reverse=True)
if not jpgs:
    jpgs = sorted(glob.glob('media/*.jpg'), key=os.path.getmtime, reverse=True)

img = jpgs[0]
print(f"Testing: {img}")
print("="*50)

ela = perform_ela(img)
noise = detect_noise(img)
freq = frequency_analysis(img)
patch = detect_patch_repetition(img)

print(f"ELA Score:       {ela['score']:.4f}")
print(f"Noise Score:     {noise['score']:.4f}")
print(f"Frequency Score: {freq['score']:.4f}")
print(f"Patch Score:     {patch['score']:.6f}")

heuristic = (ela['score'] * 0.35) + (noise['score'] * 0.30) + (freq['score'] * 0.20) + (patch['score'] * 0.15)
print(f"TOTAL Heuristic: {heuristic:.4f}")
print("="*50)

ai_v, ai_c = predict_image(img)
print(f"AI Verdict: {ai_v}, Confidence: {ai_c:.4f}")
print("="*50)

results = {
    "ela": ela, "noise": noise, "frequency": freq, "patch": patch,
    "ai_core": {"verdict": ai_v, "confidence": ai_c}
}
final = analyze_results(results)
print(f"FINAL VERDICT: {final['verdict']}")
print(f"CONFIDENCE: {final['confidence']}%")
print(f"EXPLANATION: {final['explanation']}")
