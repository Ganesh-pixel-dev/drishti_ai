import sys
import logging
logging.basicConfig(level=logging.INFO)
sys.path.append('c:\\Users\\LENOVO\\projects\\drishti_ai')

try:
    from image_module.ai_model import predict_image, MODEL_LOADED
    print(f"Model loaded: {MODEL_LOADED}")
    
    res = predict_image("temp.jpg")
    print(f"Prediction logic output: {res}")
except Exception as e:
    print("Error:", e)
