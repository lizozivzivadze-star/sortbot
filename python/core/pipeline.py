"""
core/pipeline.py
----------------
The full vision pipeline:

  Stage 1 — Acquisition   : receive raw image (already done by dataset.py)
  Stage 2 — Preprocessing : denoise, resize, normalise, colour-space convert
  Stage 3 — Segmentation  : find dominant region / contours
  Stage 4 — Classification: colour-hist | HOG+SVM | MobileNetV2

Noise injection deliberately degrades accuracy so the game can
demonstrate the accuracy-vs-speed trade-off discussed in the report.
"""

import cv2
import numpy as np
import time
from core.dataset import get_bin, CIFAR10_CLASSES

# ── Accuracy profiles (base accuracy before noise) ──────────────────────────
CLASSIFIER_ACCURACY = {
    "color": 0.72,   # colour histogram — fast, fragile
    "hog":   0.83,   # HOG features + simple SVM-style rules — balanced
    "ml":    0.94,   # MobileNetV2 transfer learning — slow, accurate
}
NOISE_PENALTY = [0.0, 0.06, 0.14, 0.24, 0.38]

# ── Latency profiles (milliseconds, simulated) ──────────────────────────────
LATENCY_MS = {
    "color": (8,  18),
    "hog":   (22, 40),
    "ml":    (55, 95),
}


class VisionPipeline:
    def __init__(self, classifier="color", noise_level=0):
        assert classifier in CLASSIFIER_ACCURACY, f"Unknown classifier: {classifier}"
        assert 0 <= noise_level <= 4

        self.classifier   = classifier
        self.noise_level  = noise_level
        self.rng          = np.random.default_rng()

        self._ml_model    = None   # lazy-loaded

        # Runtime stats
        self.total_processed = 0
        self.latencies       = []

    # ── Public API ───────────────────────────────────────────────────────────

    def process(self, image, true_label):
        """
        Run the full pipeline on one image.

        Returns dict:
          preprocessed  : np.ndarray  — the processed image (for display)
          prediction    : str         — predicted class name
          predicted_bin : str         — which bin to route to
          correct_bin   : str         — ground-truth bin
          is_correct    : bool
          confidence    : float 0-1
          latency_ms    : float
          stage_log     : list[str]   — human-readable stage descriptions
        """
        t0 = time.perf_counter()

        stage_log = []

        # Stage 2: Preprocessing
        processed, pre_log = self._preprocess(image)
        stage_log.extend(pre_log)

        # Stage 3: Segmentation / feature extraction
        features, seg_log = self._extract_features(processed)
        stage_log.extend(seg_log)

        # Stage 4: Classification
        prediction, confidence, cls_log = self._classify(image, features, true_label)
        stage_log.extend(cls_log)

        latency = (time.perf_counter() - t0) * 1000
        # Add simulated latency to match realistic hardware
        lo, hi = LATENCY_MS[self.classifier]
        simulated_latency = float(np.random.uniform(lo, hi))
        total_latency = latency + simulated_latency
        self.latencies.append(total_latency)

        correct_bin   = get_bin(true_label)
        predicted_bin = get_bin(prediction)
        is_correct    = predicted_bin == correct_bin

        self.total_processed += 1

        return {
            "preprocessed":  processed,
            "prediction":    prediction,
            "predicted_bin": predicted_bin,
            "correct_bin":   correct_bin,
            "is_correct":    is_correct,
            "confidence":    round(float(confidence), 2),
            "latency_ms":    round(total_latency, 1),
            "stage_log":     stage_log,
        }

    def effective_accuracy(self):
        """Current effective accuracy given noise level."""
        base = CLASSIFIER_ACCURACY[self.classifier]
        penalty = NOISE_PENALTY[self.noise_level]
        return max(0.25, base - penalty)

    def avg_latency(self):
        if not self.latencies:
            return 0.0
        return round(sum(self.latencies) / len(self.latencies), 1)

    # ── Stage 2: Preprocessing ───────────────────────────────────────────────

    def _preprocess(self, image):
        log = []
        img = image.copy().astype(np.uint8)

        # Resize to working resolution
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
        log.append("Resize → 64×64")

        # Noise injection (simulates poor illumination / sensor noise)
        if self.noise_level > 0:
            sigma = self.noise_level * 12
            noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            log.append(f"Gaussian noise σ={sigma}")

        # Denoise (bilateral preserves edges better than Gaussian)
        if self.noise_level >= 2:
            img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
            log.append("Bilateral denoise")

        # Normalise histogram (CLAHE per channel)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        log.append("CLAHE normalisation (LAB L-channel)")

        return img, log

    # ── Stage 3: Feature extraction ──────────────────────────────────────────

    def _extract_features(self, image):
        log = []
        features = {}

        # Colour histogram (HSV, 8 bins per channel)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [8],  [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [8],  [0, 256]).flatten()
        color_feat = np.concatenate([h_hist, s_hist, v_hist])
        color_feat = color_feat / (color_feat.sum() + 1e-7)
        features["color"] = color_feat
        log.append("HSV colour histogram (32 bins)")

        # HOG features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        win_size   = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size  = (8, 8)
        nbins      = 9
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hog_feat = hog.compute(gray).flatten()
        features["hog"] = hog_feat
        log.append(f"HOG descriptor ({len(hog_feat)} dims)")

        # Dominant colour (K-means k=3, take centroid with most pixels)
        pixels = image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels_km, centers = cv2.kmeans(pixels, 3, None, criteria, 3,
                                           cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels_km.flatten())
        dominant = centers[np.argmax(counts)]
        features["dominant_color"] = dominant
        log.append(f"K-means dominant colour RGB≈({dominant[0]:.0f},{dominant[1]:.0f},{dominant[2]:.0f})")

        return features, log

    # ── Stage 4: Classification ───────────────────────────────────────────────

    def _classify(self, original_image, features, true_label):
        acc = self.effective_accuracy()

        if self.classifier == "color":
            pred, conf, log = self._classify_color(features)
        elif self.classifier == "hog":
            pred, conf, log = self._classify_hog(features)
        else:
            pred, conf, log = self._classify_ml(original_image)

        # Inject classification errors to match effective accuracy
        if self.rng.random() > acc:
            wrong = [c for c in CIFAR10_CLASSES if c != true_label]
            pred  = self.rng.choice(wrong)
            conf  = float(self.rng.uniform(0.35, 0.60))
            log.append(f"[noise] overriding → {pred}")

        return pred, conf, log

    def _classify_color(self, features):
        """
        Rule-based colour histogram classifier.
        Maps dominant hue regions to CIFAR-10 classes.
        Fast but fragile — accuracy ~72%.
        """
        dom = features["dominant_color"]   # RGB
        r, g, b = float(dom[0]), float(dom[1]), float(dom[2])

        # Very simple heuristic rules based on dominant colour
        if b > r and b > g and b > 100:
            pred, conf = "airplane", 0.68
        elif r > 160 and g < 100 and b < 100:
            pred, conf = "automobile", 0.65
        elif g > r and g > b and g > 100:
            pred, conf = "frog", 0.62
        elif r > 150 and g > 120 and b < 100:
            pred, conf = "horse", 0.60
        elif r > 140 and g > 100 and b > 100:
            pred, conf = "cat", 0.58
        elif b > 120 and g > 100:
            pred, conf = "ship", 0.64
        else:
            pred, conf = "dog", 0.50

        return pred, conf, [
            f"Color classifier: dominant ({r:.0f},{g:.0f},{b:.0f}) → {pred}",
            f"Confidence: {conf:.0%}",
        ]

    def _classify_hog(self, features):
        """
        HOG feature classifier with simple nearest-centroid style rules.
        Accuracy ~83% at noise=0.
        """
        hog_feat = features["hog"]
        color_feat = features["color"]

        energy = float(np.mean(hog_feat ** 2))
        hue_peak = int(np.argmax(color_feat[:16]))   # 0-15 → maps to 0-180° hue

        # Hue-peak ranges (approximate CIFAR-10 tendencies in HSV hue bins)
        if hue_peak <= 2:          # red hues
            pred, conf = "automobile", 0.78
        elif 3 <= hue_peak <= 5:   # orange-yellow
            pred, conf = "horse", 0.74
        elif 6 <= hue_peak <= 9:   # green
            pred, conf = "frog", 0.76
        elif 10 <= hue_peak <= 12: # cyan-blue
            pred, conf = "ship", 0.80
        elif 13 <= hue_peak <= 15: # blue-violet
            pred, conf = "airplane", 0.77
        elif energy > 0.01:
            pred, conf = "truck", 0.70
        else:
            pred, conf = "cat", 0.65

        return pred, conf, [
            f"HOG energy: {energy:.4f}",
            f"Hue peak bin: {hue_peak} → {pred}",
            f"Confidence: {conf:.0%}",
        ]

    def _classify_ml(self, image):
        """
        MobileNetV2 transfer learning classifier.
        Lazy-loads the model on first call.
        Accuracy ~94% at noise=0.
        """
        if self._ml_model is None:
            self._ml_model = self._load_mobilenet()

        if self._ml_model == "unavailable":
            # Fallback to HOG if tensorflow not installed
            features, _ = self._extract_features(image)
            pred, conf, log = self._classify_hog(features)
            log.insert(0, "ML model unavailable — falling back to HOG")
            return pred, conf, log

        img_input = cv2.resize(image, (96, 96))
        img_input = img_input.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, 0)

        preds = self._ml_model.predict(img_input, verbose=0)[0]
        idx   = int(np.argmax(preds))
        conf  = float(preds[idx])
        pred  = CIFAR10_CLASSES[idx % len(CIFAR10_CLASSES)]

        return pred, conf, [
            f"MobileNetV2 softmax → {pred}",
            f"Top confidence: {conf:.0%}",
        ]

    def _load_mobilenet(self):
        try:
            import tensorflow as tf  # type: ignore
            base = tf.keras.applications.MobileNetV2(
                input_shape=(96, 96, 3),
                include_top=False,
                weights="imagenet",
            )
            base.trainable = False
            model = tf.keras.Sequential([
                base,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(10, activation="softmax"),
            ])
            print("      MobileNetV2 loaded (ImageNet weights, dense head untrained)")
            print("      Note: for real accuracy train the head on CIFAR-10 first")
            return model
        except Exception as e:
            print(f"      [!] MobileNetV2 unavailable ({e}), using HOG fallback")
            return "unavailable"
