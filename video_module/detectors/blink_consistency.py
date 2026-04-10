import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

# Initialize OpenCV Cascades
CASCADE_PATH = os.path.join(cv2.__path__[0], 'data')
face_cascade = cv2.CascadeClassifier(os.path.join(CASCADE_PATH, 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(CASCADE_PATH, 'haarcascade_eye.xml'))

def analyze_blink_consistency(video_path, max_frames=150):
    """
    Armada True Sight: Analyzes eye blink frequency using cascade approximations.
    Deepfakes (HeyGen, D-ID, Roop) often have 'Reptilian Gaze' (0 blinks) or 
    erratic micro-flickering lacking natural cadence.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"score": 0.0, "details": "Could not open video"}
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 1: fps = 30
        
        frames_processed = 0
        eyes_visible_history = []
        
        while frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret: break
            
            frames_processed += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            eyes_found_this_frame = 0
            if len(faces) > 0:
                x, y, w, h = faces[0]
                # Focus on the upper half of the face to reduce false positive eyes (like nostrils/mouth)
                roi_gray = gray[y:int(y+h*0.55), x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                eyes_found_this_frame = len(eyes)
            else:
                # If no face, we can't judge blinking
                eyes_found_this_frame = -1
                
            eyes_visible_history.append(eyes_found_this_frame)
            
        cap.release()
        
        if len(eyes_visible_history) < 30:
            return {"score": 0.0, "details": "Insufficient frames for blink analysis."}
            
        # Analyze Blink Cadence
        blinks = 0
        in_blink = False
        valid_frames = 0
        
        for state in eyes_visible_history:
            if state == -1: continue # No face
            valid_frames += 1
            
            if state == 0 and not in_blink:
                in_blink = True
                blinks += 1
            elif state > 0 and in_blink:
                in_blink = False
                
        if valid_frames < 30:
            return {"score": 0.0, "details": "Face barely visible for blink tracking."}
            
        # Humans blink approx 15-20 times per minute (1 blink every 3-4 seconds)
        duration_sec = valid_frames / fps
        expected_blinks = duration_sec / 4.0
        
        score = 0.0
        details = f"Natural blink cadence detected ({blinks} blinks over {duration_sec:.1f}s)."
        
        if duration_sec > 4.0 and blinks == 0:
            score = 0.85
            details = f"Reptilian Gaze Anomaly: 0 blinks detected over {duration_sec:.1f}s. Highly indicative of static AI generation."
        elif blinks > (expected_blinks * 4): # e.g. 10 blinks in 3 seconds
            score = 0.70
            details = f"Erratic Eyelid Flicker: Abnormal blink frequency ({blinks} blinks in {duration_sec:.1f}s)."
            
        return {
            "score": score,
            "details": details,
            "blinks": blinks,
            "duration": float(duration_sec)
        }
        
    except Exception as e:
        logger.error(f"Blink consistency error: {str(e)}")
        return {"score": 0.0, "details": "Blink processing failed."}
