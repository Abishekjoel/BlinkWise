import time

import cv2
import numpy as np
import tensorflow as tf
from django.conf import settings

# Alphabetical order matches sklearn LabelEncoder's fit_transform used in train_model.py
CLASS_NAMES = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]

# Clinical severity order (mild -> severe), short descriptions, and standard
# clinical naming, keyed by label. Kept separate from CLASS_NAMES (model output
# order) so the UI can always show stages in clinical order regardless of how
# confident the model is in each.
SEVERITY_INFO = {
    "No_DR": {
        "rank": 0,
        "clinical_name": "No Diabetic Retinopathy",
        "description": "No visible signs of diabetic retinopathy.",
    },
    "Mild": {
        "rank": 1,
        "clinical_name": "Mild Non-Proliferative DR (NPDR)",
        "description": "Early stage: a few microaneurysms present.",
    },
    "Moderate": {
        "rank": 2,
        "clinical_name": "Moderate Non-Proliferative DR (NPDR)",
        "description": "Blood vessel changes are more widespread.",
    },
    "Severe": {
        "rank": 3,
        "clinical_name": "Severe Non-Proliferative DR (NPDR)",
        "description": "Significant vessel damage; high risk of progression.",
    },
    "Proliferate_DR": {
        "rank": 4,
        "clinical_name": "Proliferative DR (PDR)",
        "description": "Advanced stage: abnormal new vessel growth.",
    },
}

_model = None
_grad_model = None


def get_model():
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model
        _model = load_model(settings.MODEL_PATH)
    return _model


def get_grad_model():
    """Builds a sub-model returning (last conv feature map, final prediction)
    for Grad-CAM. The trained model wraps a convolutional backbone as a
    single nested layer, so the backbone's own output (its last conv
    feature map) is exposed and the remaining head layers are replayed on top.
    """
    global _grad_model
    if _grad_model is None:
        model = get_model()
        inputs = tf.keras.Input(shape=(settings.IMG_SIZE, settings.IMG_SIZE, 3))
        x = inputs
        conv_output = None
        for layer in model.layers[1:]:
            if isinstance(layer, tf.keras.Model):
                x = layer(x)
                conv_output = x
            else:
                x = layer(x)
        if conv_output is None:
            raise ValueError("No convolutional backbone found for Grad-CAM")
        _grad_model = tf.keras.Model(inputs, [conv_output, x])
    return _grad_model


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (settings.IMG_SIZE, settings.IMG_SIZE))
    img = (img / 255.0).astype(np.float32)
    return np.expand_dims(img, axis=0)


def build_result(predictions):
    best_index = int(np.argmax(predictions))
    best_label = CLASS_NAMES[best_index]

    return {
        "label": best_label,
        "confidence": float(predictions[best_index]) * 100,
        "rank": SEVERITY_INFO[best_label]["rank"],
        "clinical_name": SEVERITY_INFO[best_label]["clinical_name"],
        "description": SEVERITY_INFO[best_label]["description"],
        "probabilities": [
            {
                "label": name,
                "clinical_name": SEVERITY_INFO[name]["clinical_name"],
                "confidence": float(prob) * 100,
                "rank": SEVERITY_INFO[name]["rank"],
            }
            for name, prob in sorted(
                zip(CLASS_NAMES, predictions), key=lambda x: x[1], reverse=True
            )
        ],
    }, best_index


def predict_image(image_path):
    model = get_model()
    batch = preprocess_image(image_path)
    predictions = model.predict(batch)[0]
    result, _ = build_result(predictions)
    return result


def make_gradcam_heatmap(batch, pred_index):
    grad_model = get_grad_model()
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch)
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def build_heatmap_overlay_png(image_path, heatmap):
    """Overlays a Grad-CAM heatmap on the original image and returns PNG bytes."""
    display_size = settings.IMG_SIZE * 2
    original = cv2.imread(image_path)
    original = cv2.resize(original, (display_size, display_size))

    heatmap_resized = cv2.resize(heatmap, (display_size, display_size))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(heatmap_color, 0.4, original, 0.6, 0)
    success, buffer = cv2.imencode(".png", overlay)
    return buffer.tobytes()


def predict_with_heatmap(image_path):
    """Runs prediction and Grad-CAM in one pass over the same preprocessed
    batch, returning (result_dict, heatmap_png_bytes, processing_time_ms)."""
    start = time.perf_counter()

    model = get_model()
    batch = preprocess_image(image_path)
    predictions = model.predict(batch)[0]
    result, best_index = build_result(predictions)

    heatmap = make_gradcam_heatmap(batch, best_index)
    heatmap_png = build_heatmap_overlay_png(image_path, heatmap)

    processing_time_ms = (time.perf_counter() - start) * 1000
    return result, heatmap_png, processing_time_ms
