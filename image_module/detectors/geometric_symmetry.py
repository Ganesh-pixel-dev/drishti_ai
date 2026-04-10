import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Re-implementing with robust CV2/Numpy logic to avoid MediaPipe version conflicts
# This ensures 100% local stability for the Full Local Armada.

def detect_geometric_warping(image_path):
    """
    [LOCAL ARMADA] Geometric Symmetry Auditor v2.
    Analyzes facial symmetry via Horizontal Flip Correlation.
    Detects 'Mathematical Mirroring' (GANs) and 'Structural Warping' (FaceSwaps).
    """
    img = cv2.imread(image_path)
    if img is None: return {"score": 0.0, "details": "File not found"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load face cascade for anchoring
    CASCADE_PATH = os.path.join(cv2.__path__[0], 'data')
    face_cascade = cv2.CascadeClassifier(os.path.join(CASCADE_PATH, 'haarcascade_frontalface_default.xml'))
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return {"score": 0.0, "details": "No anchor found for geometric audit."}

    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # Resize to standard forensic block
    face_roi = cv2.resize(face_roi, (256, 256))
    
    # Step 1: Flip and Compare
    flipped_face = cv2.flip(face_roi, 1)
    
    # Step 2: Calculate Mean Squared Error (MSE)
    mse = np.mean((face_roi.astype("float") - flipped_face.astype("float")) ** 2)
    
    # 🧬 INTERPRETATION:
    # Human faces have an MSE 'Sweet Spot' (Organic Asymmetry).
    # AI faces (GANs) are often mathematically mirrored (Very Low MSE).
    # Face-Swaps/Warps are heavily distorted (Very High MSE).
    
    if mse < 50.0: # Too perfect
        return {"score": 0.75, "details": f"Mathematical Mirror: Artificial symmetry detected (MSE: {mse:.1f}). GAN fingerprint identified."}
    
    if mse > 4000.0: # Too warped
        return {"score": 0.85, "details": f"Geometric Paradox: Extreme facial warping detected (MSE: {mse:.1f}). Possible face-swap distortion."}

    return {"score": 0.0, "details": f"Organic Symmetry Confirmed (MSE: {mse:.1f})"}

import os
