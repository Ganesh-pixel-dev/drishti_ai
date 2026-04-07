import sys
import logging
from PIL import Image
from transformers import pipeline

logging.basicConfig(level=logging.INFO)

image_path = "c:\\Users\\LENOVO\\projects\\drishti_ai\\temp.jpg"
image = Image.open(image_path).convert('RGB')

models_to_test = [
    "umm-maybe/AI-image-detector",
    "prithivMLmods/Deep-Fake-Detector-Model",
    "Organika/sdxl-detector"
]

for m in models_to_test:
    try:
        print(f"\n--- Testing {m} ---")
        clf = pipeline("image-classification", model=m)
        res = clf(image)
        print(f"Result for {m}: {res}")
    except Exception as e:
        print(f"Failed {m}: {e}")
