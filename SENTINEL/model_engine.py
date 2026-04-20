"""
SENTINEL — Model Engine v2
Enhanced CNN with BatchNorm, Data Augmentation, Validation, Early Stopping,
Confusion Matrix, and Per-Class Metrics.
"""
import os
import pickle
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    Input, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.optimizers import Adam
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
        self.model_type = "cnn"
        self.history = None
        self.is_loaded = False
        self.accuracy = 0.0
        self.val_accuracy = 0.0
        self.X = None
        self.Y = None
        self._loaded_input_size = None
        self._confusion_matrix = None
        self._class_report = None
        self._model_version = "v2"  # Track model version
        self._backbone = None

    # ── Model Architecture ──────────────────────────────────

    def _get_input_size(self, model_type=None):
        """Return input image size based on selected model type."""
        mtype = model_type or self.model_type or "cnn"
        if mtype == "efficientnet":
            return (224, 224)
        return config.IMAGE_SIZE

    def build_model(self, model_type="cnn"):
        """Build model architecture based on selected model type."""
        if model_type == "cnn":
            self.model_type = "cnn"
            return self._build_cnn()
        if model_type == "efficientnet":
            self.model_type = "efficientnet"
            return self.build_efficientnet()
        raise ValueError(f"Unsupported model_type: {model_type}")

    def _build_cnn(self):
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

    def build_efficientnet(self):
        """Build EfficientNetB0 transfer-learning classifier."""
        base_model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
        )
        # Freeze backbone for a stable default; can be fine-tuned later.
        base_model.trainable = False
        self._backbone = base_model

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        output = Dense(config.NUM_CLASSES, activation="softmax")(x)

        model = Model(inputs=base_model.input, outputs=output)
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
        self.model_type = "cnn"
        return model

    # ── Data Loading ────────────────────────────────────────

    def load_features(self, model_type=None):
        """Load pre-extracted features from .npy files."""
        x_path = os.path.join(self.model_dir, config.FEATURES_X_FILE)
        y_path = os.path.join(self.model_dir, config.FEATURES_Y_FILE)

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError("Feature files not found in model directory.")

        self.X = np.load(x_path)
        self.Y = np.load(y_path)

        if model_type:
            self.model_type = model_type

        # Normalize if not already
        if self.X.max() > 1.0:
            self.X = self.X.astype("float32") / 255.0

        target_size = self._get_input_size(model_type)
        current_size = tuple(self.X.shape[1:3]) if self.X.ndim == 4 else None
        if current_size != target_size:
            import cv2
            self.X = np.array(
                [cv2.resize(img, target_size) for img in self.X],
                dtype="float32",
            )

        self._loaded_input_size = tuple(self.X.shape[1:3])

        return self.X, self.Y

    # ── Data Augmentation ───────────────────────────────────

    def _get_augmentor(self, preprocessing_function=None):
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
            preprocessing_function=preprocessing_function,
        )

    # ── Weight Management ───────────────────────────────────

    def load_weights(self, preferred_model_type=None):
        """Build model and load pre-trained weights.

        Args:
            preferred_model_type: Optional explicit model choice ("cnn" or "efficientnet").
                If not provided, will try CNN/legacy first, then EfficientNet as a fallback.
        """
        preferred = (preferred_model_type or "").strip().lower() or None
        if preferred and preferred not in {"cnn", "efficientnet"}:
            raise ValueError(f"Unsupported model_type: {preferred_model_type}")

        efficientnet_candidates = [
            ("model_weights_efficientnet_v2.weights.h5", "efficientnet_v2", "efficientnet"),
            ("model_weights_efficientnet.weights.h5", "efficientnet", "efficientnet"),
            ("model_weights_efficientnet.h5", "efficientnet", "efficientnet"),
        ]

        cnn_candidates = [
            (config.WEIGHTS_V2_FILE, "v2", "cnn"),
            ("model_weights_v3.weights.h5", "v3", "cnn"),
            ("model_weights_v2.weights.h5", "v2", "cnn"),
        ]

        legacy_candidates = [
            (config.WEIGHTS_FILE, "legacy", "legacy"),
            ("model_weights.weights.h5", "legacy", "legacy"),
            ("model_weights.h5", "legacy", "legacy"),
        ]

        if preferred == "efficientnet":
            candidates = efficientnet_candidates
        elif preferred == "cnn":
            candidates = cnn_candidates + legacy_candidates
        else:
            candidates = cnn_candidates + efficientnet_candidates + legacy_candidates

        seen = set()
        for file_name, version, arch in candidates:
            if file_name in seen:
                continue
            seen.add(file_name)

            path = os.path.join(self.model_dir, file_name)
            if not os.path.exists(path):
                continue

            if arch == "efficientnet":
                self.build_model("efficientnet")
                self.model.load_weights(path)
                self._model_version = version
                self.is_loaded = True
                self._load_history()
                self._load_metrics()
                return True

            if arch == "cnn":
                if self.model is None or self.model_type != "cnn" or self._model_version == "legacy":
                    self.build_model("cnn")
                self.model.load_weights(path)
                self._model_version = version
                self.is_loaded = True
                self._load_history()
                self._load_metrics()
                return True

            # legacy
            self.build_legacy_model()
            self.model.load_weights(path)
            self._model_version = version
            self.is_loaded = True
            self._load_history()
            return True

        return False

    def _load_history(self):
        """Load training history from pickle file."""
        if self.model_type == "efficientnet" or self._model_version in {"efficientnet", "efficientnet_v2"}:
            history_files = [
                "history_efficientnet_v2.pckl",
                "history_efficientnet.pckl",
                "history_v3.pckl",
                config.HISTORY_V2_FILE,
                config.HISTORY_FILE,
            ]
        elif self._model_version == "v3":
            history_files = ["history_v3.pckl", config.HISTORY_V2_FILE, config.HISTORY_FILE]
        elif self._model_version == "v2":
            history_files = [config.HISTORY_V2_FILE, "history_v3.pckl", config.HISTORY_FILE]
        else:
            history_files = [config.HISTORY_FILE, config.HISTORY_V2_FILE, "history_v3.pckl", "history_efficientnet.pckl"]

        path = None
        for file_name in history_files:
            candidate = os.path.join(self.model_dir, file_name)
            if os.path.exists(candidate):
                path = candidate
                break

        if path:
            with open(path, "rb") as f:
                self.history = pickle.load(f)
            if "accuracy" in self.history:
                self.accuracy = max(self.history["accuracy"]) * 100
            if "val_accuracy" in self.history:
                self.val_accuracy = max(self.history["val_accuracy"]) * 100

    def _load_metrics(self):
        """Load confusion matrix and classification report."""
        if self.model_type == "efficientnet" or self._model_version in {"efficientnet", "efficientnet_v2"}:
            metrics_files = [
                "metrics_efficientnet_v2.pckl",
                "metrics_efficientnet.pckl",
                "metrics_v3.pckl",
                config.METRICS_V2_FILE,
            ]
        else:
            metrics_files = ["metrics_v3.pckl", config.METRICS_V2_FILE, "metrics_efficientnet_v2.pckl", "metrics_efficientnet.pckl"]
        metrics_path = None
        for file_name in metrics_files:
            candidate = os.path.join(self.model_dir, file_name)
            if os.path.exists(candidate):
                metrics_path = candidate
                break

        if metrics_path:
            with open(metrics_path, "rb") as f:
                data = pickle.load(f)
                self._confusion_matrix = data.get("confusion_matrix")
                self._class_report = data.get("classification_report")

    # ── Training ────────────────────────────────────────────

    def train(self, epochs=None, batch_size=None, use_augmentation=True,
              validation_split=0.2, progress_callback=None, model_type="cnn"):
        """
        Train the enhanced CNN model.

        Args:
            epochs: Number of training epochs (default from config)
            batch_size: Batch size (default from config)
            use_augmentation: Enable data augmentation
            validation_split: Fraction for validation set (0 to disable)
            progress_callback: Keras callback for UI progress
            model_type: "cnn" (default) or "efficientnet"
        """
        self.model_type = model_type or "cnn"
        target_size = self._get_input_size(self.model_type)

        if (
            self.X is None
            or self.Y is None
            or self._loaded_input_size != target_size
        ):
            self.load_features(model_type=self.model_type)

        # Always build from the selected architecture before training.
        self.build_model(self.model_type)

        epochs = epochs or config.EPOCHS
        batch_size = batch_size or config.BATCH_SIZE
        y_int = self.Y.astype(int)

        is_efficientnet = self.model_type == "efficientnet"
        # EfficientNet v2 training: apply correct preprocessing and fine-tune top layers.
        efficientnet_v2 = is_efficientnet

        loss_fn = (
            tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
            if efficientnet_v2
            else "categorical_crossentropy"
        )

        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=loss_fn,
            metrics=["accuracy"],
        )

        # ── Callbacks ──
        callbacks = []
        if progress_callback:
            callbacks.append(progress_callback)

        # Early stopping: EfficientNet tends to benefit from optimizing for best val accuracy.
        if validation_split > 0:
            if efficientnet_v2:
                callbacks.append(EarlyStopping(
                    monitor="val_accuracy",
                    mode="max",
                    patience=8,
                    restore_best_weights=True,
                    verbose=0,
                ))
            else:
                callbacks.append(EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True,
                    verbose=0,
                ))
            callbacks.append(ReduceLROnPlateau(
                monitor="val_accuracy" if efficientnet_v2 else "val_loss",
                mode="max" if efficientnet_v2 else "auto",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=0,
            ))

        # ── Split data (keep integer labels for class balancing) ──
        if validation_split > 0:
            X_train, X_val, y_train_int, y_val_int = train_test_split(
                self.X, y_int,
                test_size=validation_split,
                random_state=42,
                stratify=y_int,
            )
            Y_train = to_categorical(y_train_int, num_classes=config.NUM_CLASSES)
            Y_val = to_categorical(y_val_int, num_classes=config.NUM_CLASSES)
            validation_data = (X_val, Y_val)
        else:
            X_train, y_train_int = self.X, y_int
            Y_train = to_categorical(y_train_int, num_classes=config.NUM_CLASSES)
            X_val, Y_val, y_val_int = None, None, None
            validation_data = None

        # ── Class balancing (EfficientNet only) ──
        sample_weight_train = None
        if efficientnet_v2:
            class_counts = np.bincount(y_train_int, minlength=config.NUM_CLASSES).astype("float32")
            class_counts[class_counts == 0] = 1.0
            class_weights = float(len(y_train_int)) / (config.NUM_CLASSES * class_counts)
            sample_weight_train = class_weights[y_train_int]

        # ── Train ──
        def _effnet_preprocess_from_0_1(x):
            # Our pipeline stores images in [0,1]; EfficientNet preprocess expects [0,255].
            return efficientnet_preprocess_input(x * 255.0)

        preprocessing_fn = _effnet_preprocess_from_0_1 if efficientnet_v2 else None

        def _fit(epochs_to_run, initial_epoch=0):
            if epochs_to_run <= 0:
                return None

            # Keep the classic CNN path unchanged.
            if use_augmentation and not efficientnet_v2:
                augmentor = self._get_augmentor()
                steps = max(1, math.ceil(len(X_train) / batch_size))
                return self.model.fit(
                    augmentor.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=steps,
                    epochs=initial_epoch + epochs_to_run,
                    initial_epoch=initial_epoch,
                    validation_data=validation_data,
                    callbacks=callbacks,
                    verbose=0,
                )

            if use_augmentation:
                # Keras 3 can treat python iterators as finite and "run out of data".
                # Use a repeating tf.data pipeline so training always has enough batches.
                aug = tf.keras.Sequential(
                    [
                        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                        tf.keras.layers.RandomRotation(0.08),
                        tf.keras.layers.RandomZoom(0.12),
                        tf.keras.layers.RandomTranslation(0.12, 0.12),
                        tf.keras.layers.RandomContrast(0.12),
                    ],
                    name="augmentation",
                )

                def _mixup(images, labels, sample_weights, alpha=0.2):
                    # MixUp regularization helps small datasets generalize better.
                    images = tf.cast(images, tf.float32)
                    labels = tf.cast(labels, tf.float32)
                    sample_weights = tf.cast(sample_weights, tf.float32)

                    batch_n = tf.shape(images)[0]
                    shuffle_idx = tf.random.shuffle(tf.range(batch_n))

                    images_b = tf.gather(images, shuffle_idx)
                    labels_b = tf.gather(labels, shuffle_idx)
                    sw_b = tf.gather(sample_weights, shuffle_idx)

                    gamma_1 = tf.random.gamma(shape=[batch_n], alpha=alpha)
                    gamma_2 = tf.random.gamma(shape=[batch_n], alpha=alpha)
                    lam = gamma_1 / (gamma_1 + gamma_2 + 1e-8)
                    lam = tf.cast(lam, tf.float32)

                    lam_x = tf.reshape(lam, (-1, 1, 1, 1))
                    lam_y = tf.reshape(lam, (-1, 1))

                    mixed_images = lam_x * images + (1.0 - lam_x) * images_b
                    mixed_labels = lam_y * labels + (1.0 - lam_y) * labels_b
                    mixed_sw = lam * sample_weights + (1.0 - lam) * sw_b
                    return mixed_images, mixed_labels, mixed_sw

                def _prep_train(x, y, sw):
                    x = tf.cast(x, tf.float32)
                    y = tf.cast(y, tf.float32)
                    x = aug(x, training=True)
                    x, y, sw = _mixup(x, y, tf.cast(sw, tf.float32), alpha=0.2)
                    if preprocessing_fn:
                        # preprocessing_fn expects [0,1] and converts to EfficientNet space
                        x = tf.cast(x, tf.float32)
                        x = efficientnet_preprocess_input(x * 255.0)
                    return x, y, sw

                def _prep_val(x, y):
                    x = tf.cast(x, tf.float32)
                    y = tf.cast(y, tf.float32)
                    if preprocessing_fn:
                        x = efficientnet_preprocess_input(x * 255.0)
                    return x, y

                train_ds = (
                    tf.data.Dataset.from_tensor_slices((X_train, Y_train, sample_weight_train))
                    .shuffle(buffer_size=max(1, len(X_train)), seed=42, reshuffle_each_iteration=True)
                    .batch(batch_size)
                    .map(_prep_train, num_parallel_calls=tf.data.AUTOTUNE)
                    .prefetch(tf.data.AUTOTUNE)
                    .repeat()
                )

                if validation_data is not None:
                    val_ds = (
                        tf.data.Dataset.from_tensor_slices((X_val, Y_val))
                        .batch(batch_size)
                        .map(_prep_val, num_parallel_calls=tf.data.AUTOTUNE)
                        .prefetch(tf.data.AUTOTUNE)
                    )
                else:
                    val_ds = None

                steps = max(1, math.ceil(len(X_train) / batch_size))
                return self.model.fit(
                    train_ds,
                    steps_per_epoch=steps,
                    epochs=initial_epoch + epochs_to_run,
                    initial_epoch=initial_epoch,
                    validation_data=val_ds,
                    callbacks=callbacks,
                    verbose=0,
                )

            # No augmentation: preprocess in-memory for EfficientNet v2.
            X_train_in = preprocessing_fn(X_train) if preprocessing_fn else X_train
            if validation_data is not None:
                X_val_in = preprocessing_fn(X_val) if preprocessing_fn else X_val
                vdata = (X_val_in, Y_val)
            else:
                vdata = None

            return self.model.fit(
                X_train_in, Y_train,
                batch_size=batch_size,
                epochs=initial_epoch + epochs_to_run,
                initial_epoch=initial_epoch,
                validation_data=vdata,
                sample_weight=sample_weight_train if efficientnet_v2 else None,
                shuffle=True,
                callbacks=callbacks,
                verbose=0,
            )

        # Phase 1: warm-up head (backbone frozen)
        warmup_epochs = min(5, max(1, epochs // 3)) if efficientnet_v2 else epochs
        hist1 = _fit(warmup_epochs, initial_epoch=0)

        # Phase 2: fine-tune last layers (EfficientNet only)
        hist2 = None
        if efficientnet_v2 and epochs > warmup_epochs and self._backbone is not None:
            fine_tune_epochs = epochs - warmup_epochs

            # Unfreeze a subset of layers for gentle fine-tuning
            self._backbone.trainable = True
            unfreeze_last = 30
            if unfreeze_last < len(self._backbone.layers):
                for layer in self._backbone.layers[:-unfreeze_last]:
                    layer.trainable = False

            self.model.compile(
                optimizer=Adam(learning_rate=1e-5),
                loss=loss_fn,
                metrics=["accuracy"],
            )

            hist2 = _fit(fine_tune_epochs, initial_epoch=warmup_epochs)

        # Merge histories
        history = {}
        for key in (hist1.history.keys() if hist1 else []):
            history[key] = list(hist1.history.get(key, []))
        if hist2:
            for key, values in hist2.history.items():
                history.setdefault(key, [])
                history[key].extend(values)

        # ── Save weights and history ──
        weights_file = (
            "model_weights_efficientnet_v2.weights.h5"
            if self.model_type == "efficientnet"
            else config.WEIGHTS_V2_FILE
        )
        history_file = (
            "history_efficientnet_v2.pckl"
            if self.model_type == "efficientnet"
            else config.HISTORY_V2_FILE
        )

        weights_path = os.path.join(self.model_dir, weights_file)
        self.model.save_weights(weights_path)

        with open(os.path.join(self.model_dir, history_file), "wb") as f:
            pickle.dump(history, f)

        self.history = history
        self.accuracy = max(history.get("accuracy", [0.0])) * 100
        if "val_accuracy" in history:
            self.val_accuracy = max(history["val_accuracy"]) * 100
        else:
            self.val_accuracy = 0.0
        self.is_loaded = True
        self._model_version = "efficientnet_v2" if self.model_type == "efficientnet" else "v2"

        # ── Compute confusion matrix on validation set ──
        if validation_data is not None:
            self._compute_metrics(X_val, Y_val)

        return history

    # ── Metrics ─────────────────────────────────────────────

    def _compute_metrics(self, X_val, Y_val):
        """Compute confusion matrix and per-class metrics on validation data."""
        X_val_in = X_val
        if self.model_type == "efficientnet" and self._model_version == "efficientnet_v2":
            X_val_in = efficientnet_preprocess_input(X_val * 255.0)
        preds = self.model.predict(X_val_in, verbose=0)
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
        metrics_file = (
            "metrics_efficientnet_v2.pckl"
            if self.model_type == "efficientnet"
            else config.METRICS_V2_FILE
        )
        metrics_path = os.path.join(self.model_dir, metrics_file)
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
        img = cv2.resize(image, self._get_input_size())
        img = img.astype("float32") / 255.0

        if self.model_type == "efficientnet" and self._model_version == "efficientnet_v2":
            img = efficientnet_preprocess_input(img * 255.0)
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
            resized = cv2.resize(img, self._get_input_size())
            resized = resized.astype("float32") / 255.0
            if self.model_type == "efficientnet" and self._model_version == "efficientnet_v2":
                resized = efficientnet_preprocess_input(resized * 255.0)
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
            "model_type": self.model_type,
            "total_params": int(self.model.count_params()),
            "num_layers": len(self.model.layers),
            "is_loaded": self.is_loaded,
            "accuracy": round(self.accuracy, 2),
            "val_accuracy": round(self.val_accuracy, 2),
        }

    def get_model_type(self):
        """Return currently selected model type."""
        return self.model_type
