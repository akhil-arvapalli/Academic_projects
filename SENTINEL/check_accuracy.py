"""
Quick script to evaluate the external model weights.
Writes results to check_results.txt to avoid console encoding issues.
"""
import os
import pickle
import numpy as np
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

EXTERNAL_MODEL_DIR = r"D:\C M R\sem 6\Academic_projects\model"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENTINEL_MODEL_DIR = os.path.join(BASE_DIR, "model")
LABELS_4 = ["Urban Land", "Agricultural Land", "Range Land", "Forest Land"]

output = []
def log(msg=""):
    output.append(msg)
    print(msg)

log("SENTINEL - Model Accuracy Comparison")
log("=" * 60)

# 1. Load dataset
log("\n[1] Loading dataset...")
x_path = os.path.join(EXTERNAL_MODEL_DIR, "X.txt.npy")
y_path = os.path.join(EXTERNAL_MODEL_DIR, "Y.txt.npy")
X = np.load(x_path)
Y = np.load(y_path)
log(f"   Loaded from: {EXTERNAL_MODEL_DIR}")

if X.max() > 1.0:
    X = X.astype("float32") / 255.0

log(f"   Shape: X={X.shape}, Y={Y.shape}")

unique, counts = np.unique(Y, return_counts=True)
for u, c in zip(unique, counts):
    label = LABELS_4[int(u)] if int(u) < len(LABELS_4) else f"Class {u}"
    log(f"   {label}: {c} samples ({c/len(Y)*100:.1f}%)")

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)
Y_val_cat = to_categorical(Y_val, num_classes=4)
log(f"   Val set: {len(X_val)} images")

# 2. External model
log("\n[2] Evaluating EXTERNAL model (model_weights.h5)...")
ext_weights = os.path.join(EXTERNAL_MODEL_DIR, "model_weights.h5")
if os.path.exists(ext_weights):
    ext_model = Sequential([
        Input(shape=(64, 64, 3)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(4, activation="softmax"),
    ])
    ext_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    try:
        ext_model.load_weights(ext_weights)
        loss, acc = ext_model.evaluate(X_val, Y_val_cat, verbose=0)
        log(f"   EXTERNAL val accuracy: {acc*100:.2f}%")
        log(f"   EXTERNAL val loss:     {loss:.4f}")
        
        preds = ext_model.predict(X_val, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        present = sorted(set(Y_val.astype(int)) | set(y_pred))
        target_names = [LABELS_4[i] for i in present]
        report = classification_report(Y_val, y_pred, labels=present, target_names=target_names, zero_division=0)
        log("\n   Per-class report (EXTERNAL):")
        for line in report.split("\n"):
            log(f"   {line}")
    except Exception as e:
        log(f"   Failed to load: {e}")

# Check external history
ext_hist = os.path.join(EXTERNAL_MODEL_DIR, "history.pckl")
if os.path.exists(ext_hist):
    with open(ext_hist, "rb") as f:
        hist = pickle.load(f)
    acc_key = "accuracy" if "accuracy" in hist else "acc"
    if acc_key in hist:
        log(f"\n   Training history: {len(hist[acc_key])} epochs")
        log(f"   Final train acc:  {hist[acc_key][-1]*100:.2f}%")
        val_key = "val_accuracy" if "val_accuracy" in hist else "val_acc"
        if val_key in hist:
            log(f"   Final val acc:    {hist[val_key][-1]*100:.2f}%")

# 3. SENTINEL v1
log("\n[3] Evaluating SENTINEL v1 (model_weights.h5)...")
v1_weights = os.path.join(SENTINEL_MODEL_DIR, "model_weights.h5")
if os.path.exists(v1_weights):
    v1_model = Sequential([
        Input(shape=(64, 64, 3)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(4, activation="softmax"),
    ])
    v1_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    try:
        v1_model.load_weights(v1_weights)
        loss, acc = v1_model.evaluate(X_val, Y_val_cat, verbose=0)
        log(f"   V1 val accuracy: {acc*100:.2f}%")
        log(f"   V1 val loss:     {loss:.4f}")
    except Exception as e:
        log(f"   Failed: {e}")

# 4. SENTINEL v2
log("\n[4] Evaluating SENTINEL enhanced model (v3/v2 if available)...")
v2_weights = None
for candidate in ["model_weights_v3.weights.h5", "model_weights_v2.weights.h5"]:
    candidate_path = os.path.join(SENTINEL_MODEL_DIR, candidate)
    if os.path.exists(candidate_path):
        v2_weights = candidate_path
        break

if v2_weights and os.path.exists(v2_weights):
    v2_model = Sequential([
        Input(shape=(64, 64, 3)),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(4, activation="softmax"),
    ])
    v2_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    try:
        v2_model.load_weights(v2_weights)
        loss, acc = v2_model.evaluate(X_val, Y_val_cat, verbose=0)
        log(f"   Loaded weights file: {os.path.basename(v2_weights)}")
        log(f"   V2 val accuracy: {acc*100:.2f}%")
        log(f"   V2 val loss:     {loss:.4f}")
    except Exception as e:
        log(f"   Failed: {e}")
else:
    log("   Enhanced weights not found (checked v3/v2 names)")

# 5. Compare datasets
log("\n[5] Comparing datasets...")
sentinel_x = os.path.join(SENTINEL_MODEL_DIR, "X.txt.npy")
external_x = os.path.join(EXTERNAL_MODEL_DIR, "X.txt.npy")
if os.path.exists(sentinel_x) and os.path.exists(external_x):
    sx = np.load(sentinel_x)
    ex = np.load(external_x)
    if sx.shape == ex.shape and np.array_equal(sx, ex):
        log("   Datasets are IDENTICAL (same X.txt.npy)")
    else:
        log(f"   Datasets DIFFER: sentinel={sx.shape}, external={ex.shape}")

# 6. Compare weights
log("\n[6] Comparing weight files...")
ext_size = os.path.getsize(ext_weights) if os.path.exists(ext_weights) else 0
v1_size = os.path.getsize(v1_weights) if os.path.exists(v1_weights) else 0
log(f"   External model_weights.h5: {ext_size:>12,} bytes")
log(f"   SENTINEL model_weights.h5: {v1_size:>12,} bytes")
if ext_size == v1_size:
    log("   -> Same file size -- likely identical weights")
else:
    log("   -> Different file sizes -- different models")

log("\n" + "=" * 60)

# Write to file
with open("check_results.txt", "w") as f:
    f.write("\n".join(output))
log("Results written to check_results.txt")
