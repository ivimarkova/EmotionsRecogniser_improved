
"""
image_mode.py
Image-Based Emotion Analysis
Bachelor Thesis - Ivayla Markova
"""

import cv2
import numpy as np
import os

from tkinter import filedialog, messagebox
from emotion_engine import detect_emotion, EMOTION_COLORS

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Panel width for the sidebar (same as real-time mode)
PANEL_WIDTH = 250


def load_image_safe(image_path):
    """Load an image in a Unicode-safe way. Returns BGR array or None."""
    try:
        raw = np.fromfile(image_path, dtype=np.uint8)
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception:
        return None


def detect_faces_robust(gray):
    """
    Try progressively more lenient Haar detection passes.
    For tiny images (like FER2013 48x48) the face IS the whole image,
    so we fall back to treating the entire frame as a face.
    """
    h, w = gray.shape

    # Pass 1 — standard parameters
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
    )
    if len(faces) > 0:
        return faces

    # Pass 2 — very lenient, handles upscaled small images
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=1, minSize=(10, 10)
    )
    if len(faces) > 0:
        return faces

    # Pass 3 — for tiny images where the entire frame is a face,
    # treat the whole image as one face region.
    # Only do this when the image is small (≤ 64×64 original-ish size).
    if max(h, w) <= 200:
        return np.array([[0, 0, w, h]])

    return np.array([])


def upscale_if_tiny(frame, min_dim=200):
    """
    Upscale very small images so Haar detection has something to work with.
    FER2013 images are 48x48 — we scale them up to ~200px while keeping
    the aspect ratio.
    """
    h, w = frame.shape[:2]
    if max(h, w) < min_dim:
        scale = min_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return frame


def create_sidebar_panel(emotion, height):
    """Create the right-hand info panel — same design as real-time mode."""
    panel = np.full((height, PANEL_WIDTH, 3), 30, dtype=np.uint8)

    cv2.putText(
        panel, "Emotion:",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 255), 2,
    )

    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    cv2.putText(
        panel, emotion,
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        color, 3,
    )

    cv2.putText(
        panel, "Press any key",
        (10, height - 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (150, 150, 150), 1,
    )
    cv2.putText(
        panel, "to close",
        (10, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (150, 150, 150), 1,
    )

    return panel


def analyze_image(image_path):
    """
    Detect faces and classify emotions in a single image file.

    Works robustly on all image sizes, including tiny 48x48 FER2013 images.

    Returns
    -------
    annotated_frame : np.ndarray | None
    detected_emotion : str | None
    """
    frame = load_image_safe(image_path)
    if frame is None:
        return None, None

    # Upscale tiny images so Haar has pixels to work with
    frame = upscale_if_tiny(frame, min_dim=200)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve contrast on low-quality / grayscale-converted images
    gray = cv2.equalizeHist(gray)

    faces = detect_faces_robust(gray)
    detected_emotion = "No Face"

    for (x, y, w, h) in faces:
        emotion, color = detect_emotion(gray, x, y, w, h)
        detected_emotion = emotion

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label with background for readability
        label = emotion
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            frame,
            (x, max(0, y - th - 10)),
            (x + tw + 6, y),
            color, -1,
        )
        cv2.putText(
            frame, label,
            (x + 3, max(th, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 0), 2,
        )

    return frame, detected_emotion


def run_image_mode():
    """Open a file-dialog, analyse the image, display it with the sidebar."""
    filename = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")],
    )
    if not filename:
        return

    image, emotion = analyze_image(filename)

    if image is None:
        messagebox.showerror("Error", "Could not read the selected image.")
        return

    # Ensure the display image is large enough to be comfortable
    image = upscale_if_tiny(image, min_dim=300)

    # Build combined layout: image | sidebar  (same as real-time mode)
    sidebar = create_sidebar_panel(emotion, image.shape[0])
    combined = np.hstack((image, sidebar))

    cv2.imshow("Emotion Recognition — Image Analysis", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
