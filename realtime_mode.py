"""
realtime_mode.py
Real-Time Webcam Emotion Recognition with Wellbeing Monitoring
Bachelor Thesis - Ivayla Markova
"""

import cv2
import numpy as np
import os
from collections import deque

from emotion_engine import detect_emotion, EMOTION_COLORS

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Constants ─────────────────────────────────────────────────────────
SMOOTHING_WINDOW        = 10
SIDEBAR_WIDTH           = 250
WEBCAM_WIDTH            = 640
WEBCAM_HEIGHT           = 480
SCREENSHOT_DIR          = "screenshots"
MAX_FAILED_FRAMES = 60   # ~2 seconds of failed reads before giving up


# Wellbeing monitoring: how many consecutive frames count as "prolonged"
NEGATIVE_ALERT_FRAMES   = 30   # ~1 seconds at 30fps
NO_FACE_ALERT_FRAMES    = 20   # ~0.7 seconds → possible fatigue/looking away

# How many frames the alert stays visible after the condition clears
ALERT_HOLD_FRAMES = 90 #~3 seconds

#How many consecutive "good" frames before showing a positive message
POSITIVE_FRAMES = 120 #~4 seconds of Happy or Neutral

# Which emotions are considered negative/positive for monitoring purposes
NEGATIVE_EMOTIONS       = {"Sad", "Fear", "Angry", "Disgust"}
POSITIVE_EMOTIONS = {"Happy", "Neutral"}

ALERT_COLOR             = (0, 100, 255)   # orange-ish in BGR
POSITIVE_COLOR = (80, 200, 80)
HINT_COLOR              = (150, 150, 150)
WHITE                   = (255, 255, 255)


def _majority_emotion(history):
    """Return the most frequent emotion in the history deque."""
    if not history:
        return "No Face"
    counts = {}
    for e in history:
        counts[e] = counts.get(e, 0) + 1
    return max(counts, key=counts.get)


def _select_message(negative_streak, no_face_streak,
                    alert_hold, positive_streak):
    """
    Decide which message to display in the sidebar.
 
    Priority: alert (negative/fatigue) > hold > positive > nothing.
 
    Returns (message_line1, message_line2, color) or ("", "", None).
    """
    # Active alert condition
    if no_face_streak >= NO_FACE_ALERT_FRAMES:
        return (
            "Possible fatigue",
            "Look away & rest :)",
            ALERT_COLOR,
        )
    if negative_streak >= NEGATIVE_ALERT_FRAMES:
        return (
            "Take a short break!",
            "Step away for 5 min",
            ALERT_COLOR,
        )
 
    # Alert hold — condition just cleared but we keep showing the message
    if alert_hold > 0:
        return (
            "Remember to rest",
            "You are doing great!",
            ALERT_COLOR,
        )
 
    # Positive reinforcement — sustained good emotional state
    if positive_streak >= POSITIVE_FRAMES:
        return (
            "You seem focused!",
            "Keep up the good work",
            POSITIVE_COLOR,
        )
 
    return ("", "", None)


def create_text_panel(emotion, width, height, msg_line1, msg_line2, msg_color):
    """Create the sidebar panel showing emotion label and any wellbeing message."""
    panel = np.full((height, width, 3), 30, dtype=np.uint8)

    cv2.putText(
        panel, "Emotion:",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        WHITE, 2,
    )

    color = EMOTION_COLORS.get(emotion, WHITE)
    cv2.putText(
        panel, emotion,
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        color, 3,
    )

    # Wellbeing alert section
    if msg_line1 and msg_color:
        #Thin separator line
        cv2.line(panel, (10,140), (width - 10,140), HINT_COLOR, 1)

        cv2.putText(
            panel, "! Wellbeing",
            (10, 165),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            HINT_COLOR, 1,
        )
        cv2.putText(
            panel, msg_line1,
            (10, 190),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            msg_color, 1,
        )
        cv2.putText(
            panel, msg_line2,
            (10, 212),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            msg_color, 1,
        )

    cv2.putText(
        panel, "Q: quit",
        (10, height - 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        HINT_COLOR, 1,
    )
    cv2.putText(
        panel, "S: screenshot",
        (10, height - 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        HINT_COLOR, 1,
    )

    return panel


def run_realtime():
    """Start the webcam loop for real-time emotion recognition and wellbeing monitoring."""
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

    frame_count     = 0
    emotion_history = deque(maxlen=SMOOTHING_WINDOW)

    # Wellbeing counters
    negative_streak = 0
    no_face_streak  = 0
    positive_streak = 0
    alert_hold = 0 #counts down after alert clears
    failed_frames = 0

    print("[Info] Real-time mode started. Press Q to quit, S to screenshot.")

    while True:
        ret, frame = cap.read()
        #Retry limit if loosing the access to webcam temporarily due to Windows Warning

        if not ret:
            failed_frames += 1
            if failed_frames >= MAX_FAILED_FRAMES:
                print("[Warning] Webcam unavailable. Closing.")
                break
            continue

        failed_frames = 0

        frame_count += 1
        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        raw_emotion = "No Face"

        for (x, y, w, h) in faces:
            emotion, color = detect_emotion(gray, x, y, w, h)
            raw_emotion = emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                frame, emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color, 2,
            )

        # ── Wellbeing monitoring ──────────────────────────────────────
        was_alerting =(
            negative_streak >= NEGATIVE_ALERT_FRAMES
            or no_face_streak >= NO_FACE_ALERT_FRAMES
        )
        if raw_emotion in NEGATIVE_EMOTIONS:
            negative_streak += 1
            positive_streak = 0
        else:
            negative_streak = 0

        if raw_emotion == "No Face":
            no_face_streak += 1
            positive_streak = 0
        else:
            no_face_streak = 0

        if raw_emotion in POSITIVE_EMOTIONS:
            positive_streak += 1
        elif raw_emotion != "No Face":
            positive_streak = 0

 
        # Start or continue the hold timer when alert just cleared
        is_alerting = (
            negative_streak >= NEGATIVE_ALERT_FRAMES
            or no_face_streak >= NO_FACE_ALERT_FRAMES
        )
        if was_alerting and not is_alerting:
            alert_hold = ALERT_HOLD_FRAMES
        elif is_alerting:
            alert_hold = 0
        elif alert_hold > 0:
            alert_hold -= 1
 
        # ── Build display ─────────────────────────────────────────────
        msg_line1, msg_line2, msg_color = _select_message(
            negative_streak, no_face_streak, alert_hold, positive_streak
        )
 
        emotion_history.append(raw_emotion)
        smooth_emotion = _majority_emotion(emotion_history)
 
        panel = create_text_panel(
            smooth_emotion, SIDEBAR_WIDTH, frame.shape[0],
            msg_line1, msg_line2, msg_color,
        )
        combined = np.hstack((frame, panel))
 
        cv2.imshow("Emotion Recognition — Real-Time", combined)
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = os.path.join(
                SCREENSHOT_DIR, f"realtime_{frame_count:05d}.jpg"
            )
            cv2.imwrite(path, combined)
            print(f"[Info] Screenshot saved: {path}")
 
    cap.release()
    cv2.destroyAllWindows()
    print("[Info] Real-time mode ended.")
 