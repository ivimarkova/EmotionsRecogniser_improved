"""
evaluation.py
Dataset Evaluation with Per-Class Statistics
Bachelor Thesis - Ivayla Markova
"""

import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tkinter import messagebox

from image_mode import analyze_image

# Map folder names → canonical emotion labels used by the engine
FOLDER_TO_LABEL = {
    "happy":    "Happy",
    "neutral":  "Neutral",
    "surprise": "Surprised",
    "sad":      "Sad",
    "fear":     "Fear",
    "angry":    "Angry",
    "disgust":  "Disgust",
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_WORKERS = 4


def _classify_single_image(args):
    """Classify one image. Runs inside a worker process."""
    image_path, true_label = args
    _, predicted = analyze_image(image_path)
    return true_label, predicted


def _collect_tasks(database):
    """Return list of (image_path, true_label) for all supported images."""
    tasks = []
    for folder_name in sorted(os.listdir(database)):
        folder_path = os.path.join(database, folder_name)
        if not os.path.isdir(folder_path):
            continue
        true_label = FOLDER_TO_LABEL.get(
            folder_name.lower(), folder_name.capitalize()
        )
        for img_name in os.listdir(folder_path):
            ext = os.path.splitext(img_name)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                tasks.append((os.path.join(folder_path, img_name), true_label))
    return tasks


def run_evaluation():
    """
    Iterate over the test/ dataset, classify each image and report:
      - Overall accuracy
      - Per-class recall table
      - Top misclassification per class
    """
    start_time = time.time()

    database = os.path.join(
        os.path.dirname(__file__), "database", "test"
    )

    if not os.path.exists(database):
        messagebox.showerror(
            "Error", f"Test dataset not found:\n{database}"
        )
        return

    tasks = _collect_tasks(database)

    class_correct      = {}
    class_total        = {}
    class_predicted_as = {}

    # Run classifications in parallel — the executor is used as a context
    # manager so all workers finish cleanly before we continue.
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_classify_single_image, task) for task in tasks]
        for future in as_completed(futures):
            true_label, predicted = future.result()
            if predicted is None:
                continue
            class_correct.setdefault(true_label, 0)
            class_total.setdefault(true_label, 0)
            class_predicted_as.setdefault(true_label, {})

            class_total[true_label] += 1
            class_predicted_as[true_label][predicted] = (
                class_predicted_as[true_label].get(predicted, 0) + 1
            )
            if predicted == true_label:
                class_correct[true_label] += 1

    total   = sum(class_total.values())
    correct = sum(class_correct.values())

    if total == 0:
        messagebox.showwarning("Evaluation", "No images were processed.")
        return

    overall_acc = correct / total * 100
    elapsed = time.time() - start_time

    lines = [
        "─" * 50,
        f"  Overall Accuracy: {correct}/{total}  ({overall_acc:.1f}%)",
        "─" * 50,
        f"{'Class':<12} {'Correct':>8} {'Total':>8} {'Recall':>8}",
        "─" * 50,
    ]

    for label in sorted(class_total):
        tot  = class_total[label]
        corr = class_correct[label]
        recall = (corr / tot * 100) if tot else 0.0
        lines.append(
            f"{label:<12} {corr:>8} {tot:>8} {recall:>7.1f}%"
        )

    lines.append("─" * 50)
    lines.append("\nTop misclassifications:")

    for label in sorted(class_total):
        preds = class_predicted_as[label]
        wrong = {k: v for k, v in preds.items() if k != label}
        if wrong:
            worst = max(wrong, key=wrong.get)
            lines.append(f"  {label} → {worst}  ({wrong[worst]}x)")

    lines.append("")
    lines.append(f"Evaluation time: {elapsed:.2f} seconds")

    report = "\n".join(lines)
    print(report)
    messagebox.showinfo("Evaluation Complete", report)