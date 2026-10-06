import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def analyze_landmark_jitter(video_path, max_frames=150):
    """
    Armada True Sight: Analyzes Facial Volume and Landmark Jitter.
    High-quality deepfakes (like LivePortrait) often look stable but their 
    facial landmarks (eyes, nose, mouth) subtly morph, breathe, or jitter
    relative to the actual skull/head bounding box.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"score": 0.0, "details": "Could not open video"}
            
        distances = []
        
        frames_processed = 0
        while frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret: break
            
            frames_processed += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                face_center_x = fx + fw / 2.0
                
                # Look for eyes
                roi_gray = gray[fy:int(fy+fh*0.6), fx:fx+fw]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                
                # We need exactly 2 eyes detected this frame to measure the facial span
                if len(eyes) == 2:
                    ex1, ey1, ew1, eh1 = eyes[0]
                    ex2, ey2, ew2, eh2 = eyes[1]
                    
                    ecx1 = fx + ex1 + ew1 / 2.0
                    ecx2 = fx + ex2 + ew2 / 2.0
                    
                    # Normalize distance by face width so camera zoom doesn't break it
                    inter_eye_dist = abs(ecx1 - ecx2) / fw
                    distances.append(inter_eye_dist)
                    
        cap.release()
        
        if len(distances) < 15:
            return {"score": 0.0, "details": "Not enough consistent facial landmarks found for volume analysis."}
            
        # Calculate the temporal jitter (standard deviation) of the relative inter-eye distance
        # In a real human, the relative eye distance (normalized by head width) shouldn't change
        # by more than a tiny fraction (unless turning profile, which drops eyes to 1 anyway).
        variance = np.std(distances)
        
        score = 0.0
        details = f"Facial volume rigid and structurally consistent across frames."
        
        # Deepfakes often have 'breathing' faces where the eyes slide around relative to the jaw
        if variance > 0.035: 
            score = 0.80
            details = f"3D Volume Anomaly: Severe facial morphing/landmark sliding detected (Variance: {variance:.4f}). Characteristic of fluid AI mesh generation."
        elif variance > 0.025:
            score = 0.50
            details = f"Suspicious facial structure instability (Variance: {variance:.4f})."
            
        return {
            "score": score,
            "details": details,
            "variance": float(variance)
        }
        
    except Exception as e:
        logger.error(f"Landmark jitter error: {str(e)}")
        return {"score": 0.0, "details": "Volume analysis failed."}
