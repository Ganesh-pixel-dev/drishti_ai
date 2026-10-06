import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def analyze_spectral_fingerprint(video_path, max_frames=60, roi_size=32):
    """
    Global Parity Module: Spectral Temporal Fingerprinting.
    Analyzes the 'invisible pulse' of pixels across time.
    AI-generated content (Stable Video Diffusion, Sora) often exhibits
    periodic mathematical artifacts in the frequency domain due to temporal 
    upsampling or latent-space sampling rates.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"score": 0.0, "details": "Could not open video"}
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 1: fps = 30
        
        # We look for a face to anchor the ROI
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        pixel_history = []
        
        frames_processed = 0
        roi_rect = None
        
        while frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if roi_rect is None:
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    # Anchor a small ROI in the center of the forehead/cheek
                    roi_x = x + int(w * 0.4)
                    roi_y = y + int(h * 0.4)
                    roi_rect = (roi_x, roi_y, roi_size, roi_size)
                else:
                    # If no face, use center of frame as fallback
                    ch, cw = frame.shape[:2]
                    roi_rect = (cw//2, ch//2, roi_size, roi_size)
            
            rx, ry, rw, rh = roi_rect
            roi = gray[ry:ry+rh, rx:rx+rw]
            
            if roi.shape[0] == rh and roi.shape[1] == rw:
                pixel_history.append(roi.flatten())
                frames_processed += 1
                
        cap.release()
        
        if len(pixel_history) < 30:
            return {"score": 0.0, "details": "Insufficient temporal signal for spectral audit."}
            
        # 3D Signal Volume (N_Pixels x T)
        signal_volume = np.array(pixel_history).T # Shape: (roi_size*roi_size, T)
        
        # 1D FFT along the temporal axis for every pixel
        # We look for periodic spectral spikes
        ffts = np.abs(np.fft.fft(signal_volume, axis=1))
        
        # Calculate mean spectrum across the ROI block
        avg_spectrum = np.mean(ffts, axis=0)
        
        # Exclude DC component (index 0)
        freqs = np.fft.fftfreq(len(avg_spectrum), d=1.0/fps)
        valid_indices = (freqs > 1.0) # Look above 1Hz to avoid slow motion
        
        spec_sub = avg_spectrum[valid_indices]
        if len(spec_sub) == 0:
             return {"score": 0.0, "details": "Spectral domain too narrow."}
             
        # Calculate Spectral Signal-to-Noise Ratio (SSNR)
        # Periodic AI artifacts create 'spikes'. Natural noise is 'flat'.
        max_spike = np.max(spec_sub)
        mean_noise = np.mean(spec_sub)
        
        ssnr = max_spike / (mean_noise + 1e-6)
        
        score = 0.0
        details = "Temporal spectral noise is stochastic (Natural)."
        
        # SOTA Threshold: Periodic pulses (> 2.5 SNR) are characteristic of AI generation
        if ssnr > 4.5:
            score = 0.95
            details = f"Generative Spectral Spike detected (SSNR: {ssnr:.2f}). Periodic pixel oscillation found."
        elif ssnr > 3.0:
            score = 0.60
            details = f"Suspicious temporal spectral rhythm (SSNR: {ssnr:.2f})."
            
        return {
            "score": score,
            "details": details,
            "ssnr": float(ssnr)
        }
        
    except Exception as e:
        logger.error(f"Spectral fingerprint error: {str(e)}")
        return {"score": 0.0, "details": "Spectral audit failed."}
