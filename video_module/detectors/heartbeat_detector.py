import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

# Initialize OpenCV Cascades
CASCADE_PATH = os.path.join(cv2.__path__[0], 'data')
face_cascade = cv2.CascadeClassifier(os.path.join(CASCADE_PATH, 'haarcascade_frontalface_default.xml'))

def detect_heartbeat(video_path, samples=30):
    """
    Biological Armour: Detects human heart rate from sub-pixel skin color changes (r-PPG).
    AI videos (Sora/Runway) often lack this biological signal.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"score": 0.0, "details": "Could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1: fps = 30 # Fallback
    
    # We need a face for this. 
    # If a face exists but provides 0 heartbeat, it's a massive AI flag.
    
    green_signals = []
    face_found = False
    
    frames_processed = 0
    while frames_processed < samples:
        ret, frame = cap.read()
        if not ret: break
        
        frames_processed += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            face_found = True
            (x, y, w, h) = faces[0]
            
            # Use forehead region (upper middle of face box)
            roi_y = int(y + h * 0.1)
            roi_h = int(h * 0.2)
            roi_x = int(x + w * 0.25)
            roi_w = int(w * 0.5)
            
            if roi_h > 0 and roi_w > 0:
                roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                green_signals.append(np.mean(roi[:, :, 1]))
    
    cap.release()
    
    if not face_found:
        # User requested: no N/A. But if no face is detected, we can't do pulse analysis.
        # We return a 0 score but indicate face absence.
        return {"score": 0.0, "details": "No face detected for biological scan."}

    if len(green_signals) < samples // 2:
        return {"score": 0.0, "details": "Signal too short/unstable."}

    # Signal Processing: Find Heart Rate (60-100 BPM range)
    # 1. Detrend and Normalize
    signal = np.array(green_signals)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
    
    # 2. FFT to find frequency
    f = np.fft.fftfreq(len(signal), d=1.0/fps)
    fft = np.abs(np.fft.fft(signal))
    
    # Human pulse is typically 0.8 - 3.0 Hz (48 - 180 BPM)
    freq_range = (f >= 0.8) & (f <= 3.0)
    if not np.any(freq_range):
        return {"score": 0.95, "details": "Biological Void: Face detected with effectively zero heartbeat pulses."}
        
    f_sub = f[freq_range]
    fft_sub = fft[freq_range]
    
    max_power = np.max(fft_sub)
    avg_power = np.mean(fft)
    
    # Confidence in heartbeat (SNR)
    # If the peak frequency is not significantly stronger than noise, it's 'Flatline'
    snr = max_power / (avg_power + 1e-6)
    
    if snr < 1.2: # Lowered from 1.5 for better webcam stability
        # ABSENCE AS EVIDENCE (Conditioned on SNR)
        # We found a face but the blood-flow signal is missing or purely noise.
        # We lower the score for lower SNR to avoid 'binary' failure.
        score = 0.85 if snr < 1.0 else 0.4
        return {
            "score": score,
            "details": f"Biological Void: Flatline detected (SNR: {snr:.2f}). No living cardiovascular signature found.",
            "snr": float(snr)
        }
    else:
        bpm = f_sub[np.argmax(fft_sub)] * 60
        return {
            "score": 0.0,
            "details": f"Living Signature: R-PPG heart rate detected at {bpm:.1f} BPM.",
            "bpm": float(bpm),
            "snr": float(snr)
        }
