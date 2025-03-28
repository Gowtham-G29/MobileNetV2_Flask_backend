import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import traceback  # For detailed error logs

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://67e677368fb4bbe7dea02ef4--cool-kitten-d086b6.netlify.app"])

# Path to TensorFlow Lite model
TFLITE_MODEL_PATH = "mobilenetv2_model2.tflite"

# Class names for prediction results
class_names = ['Lumpy Skin', 'Flea Allergy', 'Hotspot', 'Mange', 'Ringworm'] + [f"Class_{i}" for i in range(5, 181)]

# Ensure model exists
if not os.path.exists(TFLITE_MODEL_PATH):
    print(f"❌ Model file not found: {TFLITE_MODEL_PATH}")
    exit(1)

# Load TensorFlow Lite model
try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']  # [1, height, width, 3]
    print(f"✅ TFLite model loaded with input shape: {input_shape}")

except Exception as e:
    print("❌ Error loading TFLite model:", str(e))
    exit(1)

# Image Preprocessing Function
def load_and_preprocess_image(image):
    try:
        img = image.resize((input_shape[1], input_shape[2]))  # Resize dynamically
        img_array = np.array(img).astype(np.float32) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        raise Exception(f"Image preprocessing error: {str(e)}")

# Flask Route for Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_data = load_and_preprocess_image(image)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

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

# Run Flask Server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True, threaded=True)
