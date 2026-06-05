"""
realtime_mode.py
Real-Time Webcam Emotion Recognition
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

# Number of frames used for temporal majority-vote smoothing
SMOOTHING_WINDOW = 10


def _majority_emotion(history):
    """Return the most frequent emotion in the history deque."""
    if not history:
        return "No Face"
    counts = {}
    for e in history:
        counts[e] = counts.get(e, 0) + 1
    return max(counts, key=counts.get)


def create_text_panel(emotion, width, height):
    """Create a sidebar panel showing the current emotion label."""
    panel = np.full((height, width, 3), 30, dtype=np.uint8)

    # Title
    cv2.putText(
        panel, "Emotion:",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 255), 2,
    )

    # Emotion label in its own color
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    cv2.putText(
        panel, emotion,
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        color, 3,
    )

    # Controls hint
    cv2.putText(
        panel, "Q: quit",
        (10, height - 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (150, 150, 150), 1,
    )
    cv2.putText(
        panel, "S: screenshot",
        (10, height - 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (150, 150, 150), 1,
    )

    return panel


def run_realtime():
    """Start the webcam loop for real-time emotion recognition."""

    screenshots_dir = "screenshots"
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] Cannot open webcam.")
        return

    # Optionally set a higher resolution if the camera supports it
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    emotion_history = deque(maxlen=SMOOTHING_WINDOW)

    print("[Info] Real-time mode started. Press Q to quit, S to screenshot.")

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        frame_count += 1
        frame = cv2.flip(frame, 1)  # mirror for natural interaction

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

        # Temporal smoothing
        emotion_history.append(raw_emotion)
        smooth_emotion = _majority_emotion(emotion_history)

        panel = create_text_panel(smooth_emotion, 250, frame.shape[0])
        combined = np.hstack((frame, panel))

        cv2.imshow("Emotion Recognition — Real-Time", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            path = os.path.join(
                screenshots_dir,
                f"realtime_{frame_count:05d}.jpg",
            )
            cv2.imwrite(path, combined)
            print(f"[Info] Screenshot saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[Info] Real-time mode ended.")
