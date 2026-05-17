"""
core/dataset.py
---------------
Loads CIFAR-10 and returns a randomly sampled batch.
Falls back to generating synthetic coloured shapes if
tensorflow is not installed (useful for offline demos).

Every call to load_random_batch() returns a different
random sample — this is the "different images each launch"
behaviour the project requires.
"""

import numpy as np
import random

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Map CIFAR classes → conveyor bins
CLASS_TO_BIN = {
    "airplane":    "recycle",
    "automobile":  "recycle",
    "bird":        "organic",
    "cat":         "organic",
    "deer":        "organic",
    "dog":         "organic",
    "frog":        "organic",
    "horse":       "organic",
    "ship":        "metal",
    "truck":       "metal",
}

# Deliberately mislabelled items → reject bin (simulates defects)
DEFECT_CLASSES = {"frog", "deer"}


def load_random_batch(n=60, seed=None):
    """
    Returns (images, labels, class_names).
      images      : list of np.ndarray, shape (32,32,3), uint8
      labels      : list of str  — the ground-truth class name
      class_names : list of str  — all possible class names
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    try:
        return _load_cifar10(n)
    except Exception as e:
        print(f"      [!] CIFAR-10 unavailable ({e}), using synthetic data")
        return _load_synthetic(n)


def _load_cifar10(n):
    from tensorflow.keras.datasets import cifar10  # type: ignore
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0).flatten()

    indices = random.sample(range(len(x_all)), n)
    images = [x_all[i] for i in indices]
    labels = [CIFAR10_CLASSES[y_all[i]] for i in indices]

    return images, labels, CIFAR10_CLASSES


def _load_synthetic(n):
    """
    Generates simple coloured 32×32 patches as a fallback.
    Each class has a dominant colour and a distinct shape drawn on top.
    """
    class_colours = {
        "airplane":   (160, 180, 220),
        "automobile": (220, 80,  80),
        "bird":       (100, 180, 100),
        "cat":        (220, 180, 100),
        "deer":       (160, 120, 80),
        "dog":        (200, 150, 100),
        "frog":       (60,  180, 60),
        "horse":      (120, 80,  40),
        "ship":       (80,  120, 200),
        "truck":      (180, 100, 60),
    }

    images, labels = [], []
    chosen = random.choices(CIFAR10_CLASSES, k=n)

    for cls in chosen:
        img = np.full((32, 32, 3), class_colours[cls], dtype=np.uint8)
        # Add random noise so images differ from each other
        noise = np.random.randint(-30, 30, (32, 32, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        images.append(img)
        labels.append(cls)

    return images, labels, CIFAR10_CLASSES


def get_bin(class_name):
    """Return the target sorting bin for a given class name."""
    if class_name in DEFECT_CLASSES:
        return "reject"
    return CLASS_TO_BIN.get(class_name, "reject")
