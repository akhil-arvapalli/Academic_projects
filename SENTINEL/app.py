"""
SENTINEL — Flask API Server
Satellite ENvironmental Terrain INtelligence EvaLuation
"""
import os
import io
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

import config
from model_engine import SatelliteClassifier
from utils import logger, detect_green_regions, draw_classification_overlay

# ── App Setup ────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend/dist", static_url_path="")
CORS(app)

classifier = SatelliteClassifier()

# ── API Routes ───────────────────────────────────────────

@app.route("/api/status")
def status():
    """Get system status: model state, accuracy, dataset info."""
    stats = classifier.get_dataset_stats()
    model_info = classifier.get_model_info()
    return jsonify({
        "model_loaded": classifier.is_loaded,
        "accuracy": round(classifier.accuracy, 2),
        "val_accuracy": round(classifier.val_accuracy, 2),
        "features_loaded": classifier.X is not None,
        "dataset_stats": stats,
        "model_info": model_info,
        "labels": config.LABELS,
        "label_icons": config.LABEL_ICONS,
        "app_name": config.APP_NAME,
        "app_version": config.APP_VERSION,
    })


@app.route("/api/features/load", methods=["POST"])
def load_features():
    """Load pre-extracted features from .npy files."""
    try:
        X, Y = classifier.load_features()
        stats = classifier.get_dataset_stats()
        logger.info(f"Features loaded: {len(X)} images")
        return jsonify({
            "success": True,
            "message": f"Loaded {len(X)} images",
            "stats": stats,
        })
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/model/load", methods=["POST"])
def load_model():
    """Build model architecture and load pre-trained weights."""
    try:
        success = classifier.load_weights()
        if success:
            logger.info(f"Model loaded — accuracy: {classifier.accuracy:.2f}%")
            return jsonify({
                "success": True,
                "accuracy": round(classifier.accuracy, 2),
                "message": f"Model loaded with {classifier.accuracy:.2f}% accuracy",
            })
        else:
            return jsonify({
                "success": False,
                "error": "No pre-trained weights found. Train the model first.",
            }), 404
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/model/train", methods=["POST"])
def train_model():
    """Train the CNN model with augmentation and validation."""
    try:
        if classifier.X is None:
            classifier.load_features()

        data = request.get_json() or {}
        epochs = data.get("epochs", config.EPOCHS)
        use_augmentation = data.get("augmentation", True)

        logger.info(f"Starting training — {epochs} epochs, augmentation={use_augmentation}")
        history = classifier.train(
            epochs=epochs,
            use_augmentation=use_augmentation,
            validation_split=0.2,
        )

        result = {
            "success": True,
            "accuracy": round(classifier.accuracy, 2),
            "val_accuracy": round(classifier.val_accuracy, 2),
            "history": {
                "accuracy": [round(a, 4) for a in history.get("accuracy", [])],
                "loss": [round(l, 4) for l in history.get("loss", [])],
            },
            "message": f"Training complete — {classifier.accuracy:.2f}% (val: {classifier.val_accuracy:.2f}%)",
        }
        # Add val metrics if available
        if "val_accuracy" in history:
            result["history"]["val_accuracy"] = [round(a, 4) for a in history["val_accuracy"]]
            result["history"]["val_loss"] = [round(l, 4) for l in history["val_loss"]]

        return jsonify(result)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """Classify an uploaded satellite image."""
    try:
        if not classifier.is_loaded:
            # Auto-load if weights exist
            if not classifier.load_weights():
                return jsonify({
                    "success": False,
                    "error": "Model not loaded. Load or train the model first.",
                }), 400

        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image uploaded"}), 400

        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"success": False, "error": "Invalid image file"}), 400

        # Predict
        result = classifier.predict(image)

        # Create overlay image
        display_img = cv2.resize(image, (800, 400))
        contours = detect_green_regions(display_img)
        overlay = draw_classification_overlay(display_img, result["label"], contours)

        # Encode overlay as base64 for frontend
        _, buffer = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        logger.info(f"Prediction: {result['label']} ({result['confidence']:.1f}%)")

        return jsonify({
            "success": True,
            "result": result,
            "overlay_image": f"data:image/jpeg;base64,{img_base64}",
        })
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/history")
def get_history():
    """Get training history (accuracy/loss per epoch)."""
    import pickle
    history = classifier.history

    if history is None:
        # Try loading from file (v2 first, then legacy)
        for fname in ["history_v2.pckl", "history.pckl"]:
            path = os.path.join(config.MODEL_DIR, fname)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    history = pickle.load(f)
                break

    if history is None:
        return jsonify({"success": False, "error": "No training history available"}), 404

    result = {
        "success": True,
        "history": {
            "accuracy": [round(a, 4) for a in history.get("accuracy", [])],
            "loss": [round(l, 4) for l in history.get("loss", [])],
        },
    }
    if "val_accuracy" in history:
        result["history"]["val_accuracy"] = [round(a, 4) for a in history["val_accuracy"]]
        result["history"]["val_loss"] = [round(l, 4) for l in history["val_loss"]]
    return jsonify(result)


@app.route("/api/model/info")
def model_info():
    """Get model architecture info."""
    info = classifier.get_model_info()
    if info is None:
        return jsonify({"success": False, "error": "Model not built"}), 404
    return jsonify({"success": True, "info": info})


@app.route("/api/metrics")
def get_metrics():
    """Get confusion matrix and per-class metrics."""
    metrics = classifier.get_metrics()
    if metrics["confusion_matrix"] is None:
        return jsonify({"success": False, "error": "No metrics available. Train the model first."}), 404
    return jsonify({"success": True, **metrics})


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """Classify multiple uploaded images."""
    try:
        if not classifier.is_loaded:
            if not classifier.load_weights():
                return jsonify({"success": False, "error": "Model not loaded"}), 400

        files = request.files.getlist("images")
        if not files:
            return jsonify({"success": False, "error": "No images uploaded"}), 400

        images = []
        filenames = []
        for f in files:
            file_bytes = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
                filenames.append(f.filename)

        results = classifier.predict_batch(images)
        for i, r in enumerate(results):
            r["filename"] = filenames[i]

        return jsonify({"success": True, "results": results, "count": len(results)})
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sample-images")
def list_sample_images():
    """List available sample images."""
    sample_dir = config.SAMPLE_DIR
    if not os.path.exists(sample_dir):
        return jsonify({"images": []})

    images = [
        f for f in os.listdir(sample_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    return jsonify({"images": sorted(images)})


@app.route("/api/sample-images/<filename>")
def serve_sample_image(filename):
    """Serve a sample image file."""
    return send_from_directory(config.SAMPLE_DIR, filename)


# ── Serve React Frontend ─────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serve React app for all non-API routes."""
    dist_dir = os.path.join(config.BASE_DIR, "frontend", "dist")
    if path and os.path.exists(os.path.join(dist_dir, path)):
        return send_from_directory(dist_dir, path)
    return send_from_directory(dist_dir, "index.html")


# ── Main ─────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"🛰️  {config.APP_NAME} v{config.APP_VERSION} starting...")
    logger.info(f"   Model dir: {config.MODEL_DIR}")

    # Auto-load features and model on startup
    try:
        classifier.load_features()
        logger.info(f"   Features loaded: {len(classifier.X)} images")
    except Exception:
        logger.warning("   Features not found — load manually via API")

    try:
        if classifier.load_weights():
            logger.info(f"   Model loaded — accuracy: {classifier.accuracy:.2f}%")
        else:
            logger.warning("   No pre-trained weights — train via API")
    except Exception:
        logger.warning("   Model loading failed — train via API")

    app.run(host="0.0.0.0", port=5000, debug=True)
