from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import json, os, time
from PIL import Image
from gtts import gTTS

# ======================
# CONFIG
# ======================
MODEL_PATH = "model/uang_mobilenet_lite.h5"
CLASS_PATH = "model/class_indices.json"
UPLOAD_FOLDER = "static/upload"
AUDIO_FOLDER = "static/audio"
IMG_SIZE = (160, 160)

CONFIDENCE_THRESHOLD = 0.50  # ambang aman (60%)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH) as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# ======================
# APP
# ======================
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image uploaded"
        })

    file = request.files["image"]

    # === SAVE IMAGE ===
    filename = f"{int(time.time())}_{file.filename}"
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)

    # === PREPROCESS ===
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        return jsonify({
            "status": "error",
            "message": "Invalid image file"
        })

    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # === PREDICT ===
    preds = model.predict(img)
    confidence = float(np.max(preds))
    class_id = int(np.argmax(preds))

    # ============================
    # ❌ BUKAN RUPIAH
    # ============================
    if confidence < CONFIDENCE_THRESHOLD:
        audio_text = "Uang tidak terdeteksi"
        audio_filename = f"audio_invalid_{int(time.time())}.mp3"
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)

        tts = gTTS(audio_text, lang="id")
        tts.save(audio_path)

        return jsonify({
            "status": "invalid",
            "confidence": round(confidence * 100, 2),
            "audio": f"/{audio_path}"
        })

    # ============================
    # ✅ RUPIAH TERDETEKSI
    # ============================
    label = labels.get(class_id, "Tidak diketahui")

    audio_text = f"{label} rupiah"
    audio_filename = f"audio_{int(time.time())}.mp3"
    audio_path = os.path.join(AUDIO_FOLDER, audio_filename)

    tts = gTTS(audio_text, lang="id")
    tts.save(audio_path)

    return jsonify({
        "status": "success",
        "label": label,
        "confidence": round(confidence * 100, 2),
        "audio": f"/{audio_path}"
    })

if __name__ == "__main__":
    app.run(debug=True)
