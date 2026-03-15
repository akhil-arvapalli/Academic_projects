"""
SENTINEL — Model Engine v2
Enhanced CNN with BatchNorm, Data Augmentation, Validation, Early Stopping,
Confusion Matrix, and Per-Class Metrics.
"""
import os
import pickle
import numpy as np
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    Input, BatchNormalization
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import config


class TrainingProgressCallback(Callback):
    """Keras callback that reports training progress to the UI."""

    def __init__(self, on_epoch_end=None, on_train_end=None):
        super().__init__()
        self._on_epoch_end = on_epoch_end
        self._on_train_end = on_train_end

    def on_epoch_end(self, epoch, logs=None):
        if self._on_epoch_end:
            self._on_epoch_end(epoch, logs or {})

    def on_train_end(self, logs=None):
        if self._on_train_end:
            self._on_train_end(logs or {})


class SatelliteClassifier:
    """Enhanced CNN classifier for satellite image land-use classification."""

    def __init__(self, model_dir=None):
        self.model_dir = model_dir or config.MODEL_DIR
        self.model = None
        self.history = None
        self.is_loaded = False
        self.accuracy = 0.0
        self.val_accuracy = 0.0
        self.X = None
        self.Y = None
        self._confusion_matrix = None
        self._class_report = None
        self._model_version = "v2"  # Track model version

    # ── Model Architecture ──────────────────────────────────

    def build_model(self):
        """
        Build enhanced CNN architecture.
        3 Conv blocks with increasing filters + BatchNorm + Dropout.
        """
        model = Sequential([
            Input(shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)),

            # Block 1: 32 filters
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            # Block 2: 64 filters
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            # Block 3: 128 filters
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            # Classifier head
            Flatten(),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(config.NUM_CLASSES, activation="softmax"),
        ])
        self.model = model
        return model

    def build_legacy_model(self):
        """Build the original simple CNN (for loading old weights)."""
        model = Sequential([
            Input(shape=(64, 64, 3)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(config.NUM_CLASSES, activation="softmax"),
        ])
        self.model = model
        return model

    # ── Data Loading ────────────────────────────────────────

    def load_features(self):
        """Load pre-extracted features from .npy files."""
        x_path = os.path.join(self.model_dir, "X.txt.npy")
        y_path = os.path.join(self.model_dir, "Y.txt.npy")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError("Feature files not found in model directory.")

        self.X = np.load(x_path)
        self.Y = np.load(y_path)

        # Normalize if not already
        if self.X.max() > 1.0:
            self.X = self.X.astype("float32") / 255.0

        return self.X, self.Y

    # ── Data Augmentation ───────────────────────────────────

    def _get_augmentor(self):
        """Create image data augmentation generator."""
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.15,
            brightness_range=[0.8, 1.2],
            fill_mode="nearest",
        )

    # ── Weight Management ───────────────────────────────────

    def load_weights(self):
        """Build model and load pre-trained weights (tries v2 first, then legacy)."""
        v2_path = os.path.join(self.model_dir, "model_weights_v2.h5")
        legacy_path = os.path.join(self.model_dir, "model_weights.h5")

        # Try v2 weights first
        if os.path.exists(v2_path):
            if self.model is None:
                self.build_model()
            self.model.load_weights(v2_path)
            self._model_version = "v2"
            self.is_loaded = True
            self._load_history()
            self._load_metrics()
            return True

        # Fall back to legacy weights
        if os.path.exists(legacy_path):
            self.build_legacy_model()
            self.model.load_weights(legacy_path)
            self._model_version = "legacy"
            self.is_loaded = True
            self._load_history()
            return True

        return False

    def _load_history(self):
        """Load training history from pickle file."""
        # Try v2 history first
        v2_hist = os.path.join(self.model_dir, "history_v2.pckl")
        legacy_hist = os.path.join(self.model_dir, "history.pckl")

        path = v2_hist if os.path.exists(v2_hist) else legacy_hist
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.history = pickle.load(f)
            if "accuracy" in self.history:
                self.accuracy = self.history["accuracy"][-1] * 100
            if "val_accuracy" in self.history:
                self.val_accuracy = self.history["val_accuracy"][-1] * 100

    def _load_metrics(self):
        """Load confusion matrix and classification report."""
        metrics_path = os.path.join(self.model_dir, "metrics_v2.pckl")
        if os.path.exists(metrics_path):
            with open(metrics_path, "rb") as f:
                data = pickle.load(f)
                self._confusion_matrix = data.get("confusion_matrix")
                self._class_report = data.get("classification_report")

    # ── Training ────────────────────────────────────────────

    def train(self, epochs=None, batch_size=None, use_augmentation=True,
              validation_split=0.2, progress_callback=None):
        """
        Train the enhanced CNN model.

        Args:
            epochs: Number of training epochs (default from config)
            batch_size: Batch size (default from config)
            use_augmentation: Enable data augmentation
            validation_split: Fraction for validation set (0 to disable)
            progress_callback: Keras callback for UI progress
        """
        if self.X is None or self.Y is None:
            raise ValueError("Features not loaded. Call load_features() first.")

        if self.model is None:
            self.build_model()

        epochs = epochs or config.EPOCHS
        batch_size = batch_size or config.BATCH_SIZE
        Y_cat = to_categorical(self.Y, num_classes=config.NUM_CLASSES)

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # ── Callbacks ──
        callbacks = []
        if progress_callback:
            callbacks.append(progress_callback)

        # Early stopping: stop if val_loss doesn't improve for 5 epochs
        if validation_split > 0:
            callbacks.append(EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=0,
            ))
            callbacks.append(ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=0,
            ))

        # ── Split data ──
        if validation_split > 0:
            X_train, X_val, Y_train, Y_val = train_test_split(
                self.X, Y_cat,
                test_size=validation_split,
                random_state=42,
                stratify=self.Y,
            )
            validation_data = (X_val, Y_val)
        else:
            X_train, Y_train = self.X, Y_cat
            validation_data = None

        # ── Train ──
        if use_augmentation:
            augmentor = self._get_augmentor()
            hist = self.model.fit(
                augmentor.flow(X_train, Y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=0,
            )
        else:
            hist = self.model.fit(
                X_train, Y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                shuffle=True,
                callbacks=callbacks,
                verbose=0,
            )

        # ── Save weights and history ──
        weights_path = os.path.join(self.model_dir, "model_weights_v2.h5")
        self.model.save_weights(weights_path)

        with open(os.path.join(self.model_dir, "history_v2.pckl"), "wb") as f:
            pickle.dump(hist.history, f)

        self.history = hist.history
        self.accuracy = hist.history["accuracy"][-1] * 100
        if "val_accuracy" in hist.history:
            self.val_accuracy = hist.history["val_accuracy"][-1] * 100
        self.is_loaded = True
        self._model_version = "v2"

        # ── Compute confusion matrix on validation set ──
        if validation_data is not None:
            self._compute_metrics(X_val, Y_val)

        return hist.history

    # ── Metrics ─────────────────────────────────────────────

    def _compute_metrics(self, X_val, Y_val):
        """Compute confusion matrix and per-class metrics on validation data."""
        preds = self.model.predict(X_val, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = np.argmax(Y_val, axis=1)

        # Confusion matrix
        present_classes = sorted(set(y_true) | set(y_pred))
        self._confusion_matrix = confusion_matrix(
            y_true, y_pred, labels=list(range(config.NUM_CLASSES))
        ).tolist()

        # Per-class report
        report = classification_report(
            y_true, y_pred,
            labels=list(range(config.NUM_CLASSES)),
            target_names=config.LABELS,
            output_dict=True,
            zero_division=0,
        )
        self._class_report = report

        # Save metrics
        metrics_path = os.path.join(self.model_dir, "metrics_v2.pckl")
        with open(metrics_path, "wb") as f:
            pickle.dump({
                "confusion_matrix": self._confusion_matrix,
                "classification_report": self._class_report,
            }, f)

    def get_metrics(self):
        """Return confusion matrix and per-class metrics."""
        return {
            "confusion_matrix": self._confusion_matrix,
            "classification_report": self._class_report,
            "labels": config.LABELS,
        }

    # ── Prediction ──────────────────────────────────────────

    def predict(self, image: np.ndarray):
        """
        Classify a satellite image.

        Args:
            image: Raw BGR image (any size).

        Returns:
            dict with label, label_index, confidence, probabilities, icon
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_weights() or train() first.")

        import cv2
        img = cv2.resize(image, config.IMAGE_SIZE)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        preds = self.model.predict(img, verbose=0)
        probs = preds[0]
        idx = int(np.argmax(probs))

        return {
            "label": config.LABELS[idx],
            "label_index": idx,
            "icon": config.LABEL_ICONS[idx],
            "confidence": float(probs[idx]) * 100,
            "probabilities": {
                config.LABELS[i]: float(probs[i]) * 100
                for i in range(len(config.LABELS))
            },
        }

    def predict_batch(self, images: list):
        """
        Classify multiple images at once.

        Args:
            images: List of BGR numpy arrays.

        Returns:
            List of prediction dicts.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        import cv2
        processed = []
        for img in images:
            resized = cv2.resize(img, config.IMAGE_SIZE)
            resized = resized.astype("float32") / 255.0
            processed.append(resized)

        batch = np.array(processed)
        preds = self.model.predict(batch, verbose=0)

        results = []
        for probs in preds:
            idx = int(np.argmax(probs))
            results.append({
                "label": config.LABELS[idx],
                "label_index": idx,
                "icon": config.LABEL_ICONS[idx],
                "confidence": float(probs[idx]) * 100,
                "probabilities": {
                    config.LABELS[i]: float(probs[i]) * 100
                    for i in range(len(config.LABELS))
                },
            })
        return results

    # ── Info ────────────────────────────────────────────────

    def get_dataset_stats(self):
        """Return dataset statistics."""
        if self.X is None or self.Y is None:
            return None
        unique, counts = np.unique(self.Y, return_counts=True)
        dist = {}
        for u, c in zip(unique, counts):
            if u < len(config.LABELS):
                dist[config.LABELS[int(u)]] = int(c)
        return {
            "total_images": int(len(self.X)),
            "image_shape": [int(s) for s in self.X.shape[1:]],
            "class_distribution": dist,
            "num_classes_present": int(len(unique)),
        }

    def get_model_info(self):
        """Return model architecture summary."""
        if self.model is None:
            return None
        return {
            "version": self._model_version,
            "total_params": int(self.model.count_params()),
            "num_layers": len(self.model.layers),
            "is_loaded": self.is_loaded,
            "accuracy": round(self.accuracy, 2),
            "val_accuracy": round(self.val_accuracy, 2),
        }
