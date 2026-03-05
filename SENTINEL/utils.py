"""
SENTINEL — Utility Functions
Image processing helpers and logging setup.
"""
import cv2
import numpy as np
import logging
from datetime import datetime

import config


def setup_logger(name="sentinel"):
    """Configure application logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = setup_logger()


def load_image(filepath):
    """Load an image from disk, return BGR numpy array or None."""
    img = cv2.imread(filepath)
    if img is None:
        logger.error(f"Failed to load image: {filepath}")
    return img


def preprocess_image(image, size=None):
    """Resize and normalize an image for model input."""
    size = size or config.IMAGE_SIZE
    img = cv2.resize(image, size)
    img = img.astype("float32") / 255.0
    return img


def detect_green_regions(image):
    """Detect green vegetation regions and return contours."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([36, 25, 25])
    upper = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if len(c) > 10]


def draw_classification_overlay(image, label, contours=None):
    """Draw classification result and contour boxes on image."""
    img = image.copy()
    if contours:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 136), 2)

    # Add label banner
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 40), (10, 14, 23), -1)
    cv2.putText(
        img, f"CLASSIFIED: {label.upper()}",
        (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 136), 2,
    )
    return img


def format_timestamp():
    """Return formatted current timestamp."""
    return datetime.now().strftime("%H:%M:%S")
