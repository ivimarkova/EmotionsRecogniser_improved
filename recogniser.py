"""
recogniser.py
Main GUI Entry Point
Bachelor Thesis - Ivayla Markova
Emotion Recognition System — Local, real-time & image-based
"""

import tkinter as tk
from tkinter import ttk

import threading

from realtime_mode import run_realtime
from image_mode import run_image_mode
from evaluation import run_evaluation

def start_evaluation():
    threading.Thread(
        target=run_evaluation,
        daemon=True
    ).start()

# ── Constants ─────────────────────────────────────────────────────────
APP_TITLE   = "Emotion Recognition System"
WINDOW_SIZE = "500x420"
BTN_WIDTH   = 32
BTN_HEIGHT  = 2
FONT_TITLE  = ("Times New Roman", 18, "bold")
FONT_SUB    = ("Times New Roman", 9)


def main():

    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry(WINDOW_SIZE)
    root.resizable(False, False)

    # Title
    tk.Label(
        root,
        text=APP_TITLE,
        font=FONT_TITLE,
    ).pack(pady=(20, 4))

    tk.Label(
        root,
        text="Bachelor Thesis  ·  Ivayla Markova",
        font=FONT_SUB,
        fg="grey",
    ).pack(pady=(0, 16))

    ttk.Separator(root, orient="horizontal").pack(
        fill="x", padx=20, pady=4
    )

    # Buttons
    button_defs = [
        ("🎥  Real-Time Webcam Analysis", run_realtime),
        ("🖼   Image Analysis",            run_image_mode),
        ("📊  Evaluate on Test Dataset",  start_evaluation),
    ]

    for label, command in button_defs:
        tk.Button(
            root,
            text=label,
            command=command,
            width=BTN_WIDTH,
            height=BTN_HEIGHT,
        ).pack(pady=6)

    ttk.Separator(root, orient="horizontal").pack(
        fill="x", padx=20, pady=8
    )

    tk.Button(
        root,
        text="Exit",
        command=root.destroy,
        width=BTN_WIDTH,
        height=1,
        fg="red",
    ).pack(pady=4)

    root.mainloop()


if __name__ == "__main__":
    main()
