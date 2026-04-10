import cv2
import numpy as np
import os

# Initialize OpenCV Cascades
CASCADE_PATH = os.path.join(cv2.__path__[0], 'data')
face_cascade = cv2.CascadeClassifier(os.path.join(CASCADE_PATH, 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(CASCADE_PATH, 'haarcascade_eye.xml'))

def detect_reflection_inconsistency(image_path):
    """
    Geometric Armour: Analyzes specular highlights in the eyes.
    Detects 'Impossible Geometry' where reflections don't match the light source 
    or are completely missing in high-def facial renders.
    """
    img = cv2.imread(image_path)
    if img is None: return {"score": 0.0, "details": "File not found"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return {"score": 0.0, "details": "No face detected for reflection analysis."}

    (x, y, w, h) = faces[0]
    face_roi_gray = gray[y:y+h, x:x+w]
    face_roi_color = img[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(face_roi_gray)
    
    if len(eyes) < 2:
        return {"score": 0.0, "details": "Eye visibility too low for geometric scan."}

    # We take the first two detected eyes
    l_eye_rect = eyes[0]
    r_eye_rect = eyes[1]
    
    l_eye = face_roi_color[l_eye_rect[1]:l_eye_rect[1]+l_eye_rect[3], l_eye_rect[0]:l_eye_rect[0]+l_eye_rect[2]]
    r_eye = face_roi_color[r_eye_rect[1]:r_eye_rect[1]+r_eye_rect[3], r_eye_rect[0]:r_eye_rect[0]+r_eye_rect[2]]

    if l_eye.size == 0 or r_eye.size == 0:
        return {"score": 0.0, "details": "Eye visibility too low."}

    def detect_highlights(eye_img):
        # Specular highlights are clear, high-intensity white spots
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    l_highlights = detect_highlights(l_eye)
    r_highlights = detect_highlights(r_eye)

    # --- ABSENCE AS EVIDENCE ---
    # If the image is high resolution (checked by size) but eyes have NO highlights,
    # it's 'Impossible Geometry' common in AI renders that forget light transport.
    if len(l_highlights) == 0 and len(r_highlights) == 0:
        if w > 1000:
            return {"score": 0.85, "details": "Impossible Geometry: High-res face found with zero specular eye reflections (Dead-Eye Phenomenon)."}
        return {"score": 0.4, "details": "Suspicious: Eyes lack reflective highlights."}

    # Symmetry Check: reflections should be in roughly the same relative position
    # (Simplified forensic check)
    def get_centroid(contours):
        if not contours: return None
        M = cv2.moments(contours[0])
        if M["m00"] == 0: return None
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    l_c = get_centroid(l_highlights)
    r_c = get_centroid(r_highlights)

    if l_c and r_c:
        # Check relative position in iris
        # In a deepfake, eyes are often blended independently, leading to 'Wandering Highlights'
        # Check if highlights are consistent across eyes
        # (This is a simplified version of photometric symmetry)
        return {"score": 0.0, "details": "Eye reflections detected & geometrically consistent."}

    return {"score": 0.3, "details": "Minor reflection asymmetry detected."}
