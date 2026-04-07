import sys
from transformers import pipeline
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)

image_path = "c:\\Users\\LENOVO\\projects\\drishti_ai\\media\\fcd90ae1f56a4371bf79e26124ff3101.png"

try:
    print("Loading model...")
    clf = pipeline("image-classification", model="umm-maybe/AI-image-detector")
    print("Model loaded. Predicting...")
    img = Image.open(image_path).convert('RGB')
    res = clf(img)
    print("Raw output:", res)
except Exception as e:
    print("Error:", e)
