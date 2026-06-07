"""
emotion_engine.py
Rule-based Emotion Classification Engine
Bachelor Thesis - Ivayla Markova
"""

import cv2
import numpy as np

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

EMOTION_COLORS = {
    "Happy":     (0, 220, 0),
    "Neutral":   (200, 200, 200),
    "Surprised": (0, 200, 255),
    "Sad":       (200, 100, 50),
    "Fear":      (130, 0, 180),
    "Angry":     (0, 0, 220),
    "Disgust":   (0, 180, 120),
    "No Face":   (80, 80, 80),
}


def _count_eyes(roi):
    """Return the number of eyes detected in the face ROI"""
    eyes = eye_cascade.detectMultiScale(
        roi, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10)
    )
    return len(eyes)


def _has_smile(roi, h):
    """Return true if a smile is detected in the lower half of the face ROI"""
    lower_half = roi[h // 2:, :]
    smiles = smile_cascade.detectMultiScale(
        lower_half, scaleFactor=1.5, minNeighbors=10
    )
    return len(smiles) > 0


def _mouth_open_ratio(roi, h, w):
    """
    Proportion of dark pixels in the mouth zone.
    A high ratio means the mouth is open.
    """
    mouth = roi[int(h * 0.65):h, int(w * 0.25):int(w * 0.75)]
    if mouth.size == 0:
        return 0.0
    return float((mouth < 80).sum()) / mouth.size


def _brow_darkness(roi, h, w):
    """
    Mean darkness of the brow region (upper 30% of face).
    Furrowed brows create darker shadow → lower mean value.
    Works on both large and tiny upscaled images.
    """
    brow = roi[:int(h * 0.30), int(w * 0.10):int(w * 0.90)]
    if brow.size == 0:
        return 128.0
    return float(np.mean(brow))


def _brow_tension(roi, h, w):
    """
    Pixel variance in the brow region — furrowed brows produce more
    local contrast (light + dark transitions) than relaxed brows.
    """
    brow = roi[:int(h * 0.35), int(w * 0.15):int(w * 0.85)]
    if brow.size == 0:
        return 0.0
    return float(np.std(brow))


def _mouth_corner_drop(roi, h, w):
    """
    Compare brightness of left vs right mouth corners in the lower face.
    A symmetric downward curl (sadness) creates slightly darker outer corners.
    Used as a supplementary cue.
    """
    lower = roi[int(h * 0.70):h, :]
    if lower.size == 0:
        return 0.0
    lw = lower.shape[1]
    left  = float(np.mean(lower[:, :lw // 4]))
    right = float(np.mean(lower[:, 3 * lw // 4:]))
    center = float(np.mean(lower[:, lw // 4:3 * lw // 4]))
    # If corners are darker than center → corners are turned down
    corner_darkness = center - (left + right) / 2
    return corner_darkness


def detect_emotion(gray, x, y, w, h):
    """
    Classify the emotion of a face region using rule-based heuristics.

    Parameters
    ----------
    gray : np.ndarray  — grayscale full frame (already equalised)
    x, y, w, h : int  — bounding box of detected face

    Returns
    -------
    emotion : str
    color   : tuple (B, G, R)
    """
    roi = gray[y:y + h, x:x + w]

    if roi.size == 0:
        return "No Face", EMOTION_COLORS["No Face"]

    eye_count    = _count_eyes(roi)
    smile        = _has_smile(roi, h)
    mouth_open   = _mouth_open_ratio(roi, h, w)
    brow_tension = _brow_tension(roi, h, w)
    brow_dark    = _brow_darkness(roi, h, w)
    corner_drop  = _mouth_corner_drop(roi, h, w)

    # ── Decision tree ────────────────────────────────────────────────
    #
    # We use ≥1 eye as the "face found clearly" threshold.
    # On tiny upscaled images eye detection is unreliable, so we also
    # allow the fallback branch (eyes=0) to reach Happy/Surprised.
    #
    # Threshold values are tuned for contrast-equalised, upscaled
    # images.  Brow tension is proportional to image brightness
    # variance so thresholds scale naturally with image size.

    if eye_count >= 1:
        
        if smile:
            emotion = "Happy"
        
        elif mouth_open > 0.22:
            emotion = "Surprised"
            
        elif brow_dark < 90 and brow_tension > 28:
            emotion = "Angry"
            
        elif brow_tension > 24 and mouth_open > 0.12:
            emotion = "Fear"
            
        elif corner_drop > 12:
            emotion = "Sad"
            
        elif brow_tension > 20:
            emotion = "Disgust"
            
        else:
            emotion = "Neutral"

    else:
        # Eyes not detected (small/blurry/profile face)
        if smile:
            emotion = "Happy"
        elif mouth_open > 0.25:
            emotion = "Surprised"
        elif brow_dark < 90 and brow_tension > 30:
            emotion = "Angry"
        elif corner_drop > 12:
            emotion = "Sad"
        else:
            emotion = "Neutral"

    return emotion, EMOTION_COLORS[emotion]
