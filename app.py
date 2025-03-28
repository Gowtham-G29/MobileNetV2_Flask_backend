import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import traceback  # For detailed error logs

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

MODEL_PATH = os.path.abspath("mobilenetv2_model2.h5")

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = load_model(MODEL_PATH, custom_objects={"LeakyReLU": LeakyReLU})
    input_shape = model.input_shape  

    print(f"✅ Model loaded successfully: {MODEL_PATH}")
except Exception as e:
    print("❌ Error loading model:", str(e))
    model = None  

class_names = ['Lumpy Skin', 'Flea Allergy', 'Hotspot', 'Mange', 'Ringworm'] + [f"Class_{i}" for i in range(5, 181)]

def load_and_preprocess_image(image):
    try:
        img = image.resize((input_shape[1], input_shape[2]))  # Resize dynamically
        img_array = np.array(img).astype(np.float32) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        raise Exception(f"Image preprocessing error: {str(e)}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model failed to load"}), 500

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_data = load_and_preprocess_image(image)

        # Run inference
        output_data = model.predict(input_data)
        predicted_class_index = int(np.argmax(output_data))
        predicted_class_name = class_names[predicted_class_index]

        print(f"🔹 Prediction: {predicted_class_name} (Confidence: {np.max(output_data):.4f})")

        return jsonify({
            "prediction": predicted_class_name,
            "confidence": float(np.max(output_data)),
            "status": "Success"
        })

    except Exception as e:
        error_trace = traceback.format_exc()
        return jsonify({
            "error": "Prediction failed",
            "details": str(e),
            "traceback": error_trace
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
