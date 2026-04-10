import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def analyze_face_boundary(image_path):
    """
    Armada True Sight: Analyzes the image for face-swap jawline blending artifacts.
    Deepfacelab/Roop/Reactor often leave a color/texture mismatch or a sharp
    edge response around the convex hull where the artificial mask meets real skin.
    """
    try:
        if not os.path.exists(image_path):
            return {"score": 0.0, "details": "File missing"}

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {"score": 0.0, "details": "Invalid image"}
            
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Load local OpenCV Face Cascade
        CASCADE_PATH = os.path.join(cv2.__path__[0], 'data')
        cascade_file = os.path.join(CASCADE_PATH, 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cascade_file)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return {"score": 0.0, "details": "No face detected for boundary scan."}

        # Analyze the largest face (assume primary subject)
        x, y, w, h = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
        
        # Define Inner Face (Artificial Mask) and Outer Jawline (Real Skin Host)
        inner_y = int(y + h * 0.2)
        inner_h = int(h * 0.6)
        inner_x = int(x + w * 0.2)
        inner_w = int(w * 0.6)
        
        # We can't go out of bounds for the outer region
        outer_pad = int(w * 0.1)
        out_y1 = max(0, y - outer_pad)
        out_y2 = min(img_bgr.shape[0], y + h + outer_pad)
        out_x1 = max(0, x - outer_pad)
        out_x2 = min(img_bgr.shape[1], x + w + outer_pad)
        
        # Extract regions in YCrCb (better for color temperature/chromeluma blending detection)
        img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        
        inner_roi = img_ycrcb[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w]
        
        # Outer ROI is the bounding box minus the inner box
        outer_skin_mask = np.ones((out_y2-out_y1, out_x2-out_x1), dtype=np.uint8)
        
        # Rel inner coords to outer coords
        rel_iy = inner_y - out_y1
        rel_ix = inner_x - out_x1
        outer_skin_mask[rel_iy:rel_iy+inner_h, rel_ix:rel_ix+inner_w] = 0
        
        outer_roi_full = img_ycrcb[out_y1:out_y2, out_x1:out_x2]
        
        if inner_roi.size == 0 or outer_roi_full.size == 0:
            return {"score": 0.0, "details": "Invalid ROI slice."}
            
        # Calculate color histograms for Cr (Red-difference) and Cb (Blue-difference)
        # AI models often shift the skin tone subtly.
        inner_cr_hist = cv2.calcHist([inner_roi], [1], None, [32], [0, 256])
        inner_cb_hist = cv2.calcHist([inner_roi], [2], None, [32], [0, 256])
        
        outer_cr_hist = cv2.calcHist([outer_roi_full], [1], outer_skin_mask, [32], [0, 256])
        outer_cb_hist = cv2.calcHist([outer_roi_full], [2], outer_skin_mask, [32], [0, 256])
        
        cv2.normalize(inner_cr_hist, inner_cr_hist)
        cv2.normalize(inner_cb_hist, inner_cb_hist)
        cv2.normalize(outer_cr_hist, outer_cr_hist)
        cv2.normalize(outer_cb_hist, outer_cb_hist)
        
        cr_diff = cv2.compareHist(inner_cr_hist, outer_cr_hist, cv2.HISTCMP_BHATTACHARYYA)
        cb_diff = cv2.compareHist(inner_cb_hist, outer_cb_hist, cv2.HISTCMP_BHATTACHARYYA)
        
        # Combine differences. High difference = Jawline seam/Skin graft detected
        total_color_shift = cr_diff + cb_diff
        
        # Map 0.0 - 1.0 (Bhattacharyya is 0-1, sum is 0-2)
        score = min(total_color_shift / 0.8, 1.0)
        
        details = "Skin boundary blending matches natural lighting patterns."
        if score > 0.6:
            details = f"Jawline Seam Anomaly: Severe color-temperature separation detected (Mismatch: {score:.2f}). Highly indicative of AI face-swap graph."
        elif score > 0.4:
            details = f"Suspicious boundary blending along facial curve."
            
        return {
            "score": round(float(score), 3),
            "details": details
        }
    except Exception as e:
        logger.error(f"Face boundary blend error: {str(e)}")
        return {"score": 0.0, "details": "Boundary analysis failed."}
