"""
================================================
  image_mode.py - Image-Based Analysis
  Emotion Recognition System
  Abschlussarbeit - Ivayla Markova
================================================

Approach 2: Static Image Emotion Analysis
- User can select a single image file to analyze
- OR browse images from the FER2013 database folder
- Detects faces and classifies emotions using
  the same Haar Cascade + heuristic rules
- Results are displayed on screen
- Press S to save the result, Q to go back

DATABASE: FER2013
  Download from: https://www.kaggle.com/datasets/msambare/fer2013
  Expected folder structure after download:
    database/
      train/
        happy/
        sad/
        angry/
        surprise/
        neutral/
        fear/
        disgust/
      test/
        happy/
        ...

  Set DATABASE_PATH below to your local folder path.
"""

import cv2
import numpy as np
import os

# ── DATABASE PATH ─────────────────────────────────────────────────
# Change this to where you saved the FER2013 dataset on your computer
# Example Windows: r"C:\Users\Admin\Downloads\fer2013"
# Example relative: "database"
DATABASE_PATH = "database"

# ── Load Haar Cascades ────────────────────────────────────────────
face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# ── Emotion colors (BGR) ──────────────────────────────────────────
EMOTION_COLORS = {
    'Happy':     (0,   220,   0),
    'Neutral':   (200, 200, 200),
    'Surprised': (0,   200, 255),
    'Sad':       (200, 100,  50),
    'Angry':     (0,     0, 220),
    'Fear':      (130,   0, 180),
    'Disgust':   (0,   140, 255),
}

# Supported image file extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.pgm')


def detect_emotion(gray_frame, x, y, w, h):
    """
    Rule-based emotion classification — same logic as realtime_mode.py.
    Kept here so image_mode.py works independently.
    """
    roi_gray = gray_frame[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(
        roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )

    lower_half = roi_gray[h // 2:, :]
    smiles = smile_cascade.detectMultiScale(
        lower_half, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25)
    )

    mouth_region = roi_gray[int(h * 0.65):h, int(w * 0.25):int(w * 0.75)]
    mouth_dark_ratio = 0.0
    if mouth_region.size > 0:
        dark_pixels = (mouth_region < 80).sum()
        mouth_dark_ratio = dark_pixels / mouth_region.size

    eye_count   = len(eyes)
    smile_found = len(smiles) > 0

    if eye_count >= 2:
        if smile_found:
            emotion = 'Happy'
        elif mouth_dark_ratio > 0.15:
            emotion = 'Surprised'
        else:
            emotion = 'Neutral'
    elif eye_count == 1:
        emotion = 'Neutral'
    else:
        emotion = 'Fear' if mouth_dark_ratio > 0.15 else 'Sad'

    color = EMOTION_COLORS.get(emotion, (200, 200, 200))
    return emotion, color


def load_image_safe(image_path):
    """
    Load an image safely even if the path contains Cyrillic or
    special characters (which cv2.imread cannot handle on Windows).

    Solution: read the raw bytes with numpy first, then decode with
    cv2.imdecode — this bypasses the Windows path encoding issue.
    """
    try:
        raw = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def analyze_image(image_path):
    """
    Load one image, detect face, classify emotion, draw result.

    Returns the annotated image ready for display,
    and the detected emotion string.
    """
    # Use safe loader to handle Cyrillic/special characters in path
    frame = load_image_safe(image_path)

    if frame is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # FER2013 images are 48x48 pixels — minSize must be small enough
    # For regular photos a larger minSize is fine, but we use (20,20)
    # so the function works for both the database AND personal photos
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )

    detected_emotion = "No Face"

    if len(faces) == 0:
        cv2.putText(frame, "No face detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            emotion, color = detect_emotion(gray, x, y, w, h)
            detected_emotion = emotion

            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

            # Draw emotion label above the face
            label_size, _ = cv2.getTextSize(
                emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
            )
            cv2.rectangle(frame,
                          (x, y - 38),
                          (x + label_size[0] + 10, y),
                          color, -1)
            cv2.putText(frame, emotion,
                        (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 0), 2)

    # Show filename at the bottom of the image
    h_img, w_img = frame.shape[:2]
    cv2.rectangle(frame, (0, h_img - 35), (w_img, h_img), (0, 0, 0), -1)
    cv2.putText(frame,
                f"File: {os.path.basename(image_path)}  |  Result: {detected_emotion}",
                (8, h_img - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    return frame, detected_emotion


def browse_database():
    """
    Browse images from the FER2013 database folder.

    Shows a menu of available emotion subfolders.
    User picks a folder, then navigates images with
    arrow keys or N (next) / P (previous).
    """
    # Check if database folder exists
    if not os.path.exists(DATABASE_PATH):
        print(f"\n[ERROR] Database folder not found: '{DATABASE_PATH}'")
        print("  Please download FER2013 from:")
        print("  https://www.kaggle.com/datasets/msambare/fer2013")
        print(f"  And place it in: {os.path.abspath(DATABASE_PATH)}\n")
        return

    # Find subfolders (train/test or direct emotion folders)
    subfolders = [
        d for d in os.listdir(DATABASE_PATH)
        if os.path.isdir(os.path.join(DATABASE_PATH, d))
    ]

    if not subfolders:
        print(f"[ERROR] No subfolders found in {DATABASE_PATH}")
        return

    # Show available subfolders to user
    print("\n" + "="*50)
    print(f"  Database: {DATABASE_PATH}")
    print("="*50)
    print("  Available folders:")
    for i, folder in enumerate(subfolders):
        folder_path = os.path.join(DATABASE_PATH, folder)
        # Count images inside
        count = sum(
            1 for f in os.listdir(folder_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        # Also count images in subfolders (FER2013 has train/test -> emotion)
        for sub in os.listdir(folder_path):
            sub_full = os.path.join(folder_path, sub)
            if os.path.isdir(sub_full):
                count += sum(
                    1 for f in os.listdir(sub_full)
                    if f.lower().endswith(IMAGE_EXTENSIONS)
                )
        print(f"    [{i+1}] {folder}  ({count} images)")

    print("    [Q] Back to menu")
    print("="*50)

    choice = input("Select folder number: ").strip().lower()

    if choice == 'q':
        return

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(subfolders):
            print("[!] Invalid selection.")
            return
    except ValueError:
        print("[!] Please enter a number.")
        return

    selected_folder = os.path.join(DATABASE_PATH, subfolders[idx])

    # Collect all images from selected folder and its subfolders
    image_list = []

    for root, dirs, files in os.walk(selected_folder):
        for filename in sorted(files):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_list.append(os.path.join(root, filename))

    if not image_list:
        print(f"[INFO] No images found in {selected_folder}")
        return

    print(f"\n[INFO] Found {len(image_list)} images.")
    print("[INFO] Controls in viewer: [N] Next  [P] Previous  [S] Save  [Q] Back\n")

    # ── Image viewer loop ─────────────────────────────────────────
    current_index = 0

    while True:
        img_path = image_list[current_index]
        annotated, emotion = analyze_image(img_path)

        if annotated is None:
            current_index = (current_index + 1) % len(image_list)
            continue

        print(f"  [{current_index+1}/{len(image_list)}] {os.path.basename(img_path)} -> {emotion}")

        cv2.imshow("Image Analysis - Database Browser", annotated)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('n') or key == 83:    # N or right arrow
            current_index = (current_index + 1) % len(image_list)

        elif key == ord('p') or key == 81:  # P or left arrow
            current_index = (current_index - 1) % len(image_list)

        elif key == ord('s'):
            if not os.path.exists('screenshots'):
                os.makedirs('screenshots')
            fname = f"screenshots/image_{current_index:04d}_{emotion}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"  [INFO] Saved: {fname}")

        elif key == ord('q'):
            print("\n[INFO] Closing image viewer...")
            break

    cv2.destroyAllWindows()


def select_own_image():
    """
    Let the user type a path to their own image file.
    Analyzes it and shows the result.
    """
    print("\n  Enter the full path to your image file.")
    print("  Example: C:\\Users\\Admin\\Pictures\\photo.jpg")
    image_path = input("  Path: ").strip().strip('"').strip("'")

    if not os.path.isfile(image_path):
        print(f"[ERROR] File not found: {image_path}\n")
        return

    annotated, emotion = analyze_image(image_path)

    if annotated is None:
        return

    print(f"\n[INFO] Result: {emotion}")
    print("[INFO] Press [S] to save, any other key to close.")

    cv2.imshow("Image Analysis - Single Image", annotated)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('s'):
        if not os.path.exists('screenshots'):
            os.makedirs('screenshots')
        fname = f"screenshots/single_{os.path.splitext(os.path.basename(image_path))[0]}_result.jpg"
        cv2.imwrite(fname, annotated)
        print(f"[INFO] Saved: {fname}")

    cv2.destroyAllWindows()


def run_image_mode():
    """
    Approach 2 — Image-Based Analysis Mode.
    User chooses between their own image or the database.
    """
    while True:
        print("\n" + "="*50)
        print("  Approach 2: Image-Based Analysis")
        print("="*50)
        print("  [1] Analyze my own image")
        print("  [2] Browse FER2013 database")
        print("  [Q] Back to main menu")
        print("="*50)

        choice = input("  Enter choice: ").strip().lower()

        if choice == '1':
            select_own_image()
        elif choice == '2':
            browse_database()
        elif choice == 'q':
            break
        else:
            print("[!] Invalid choice.")