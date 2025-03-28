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
CORS(app, origins=["http://localhost:5173", "https://67e677368fb4bbe7dea02ef4--cool-kitten-d086b6.netlify.app"])

# Paths for models
H5_MODEL_PATH = "mobilenetv2_model2.h5"
TFLITE_MODEL_PATH = "mobilenetv2_model2.tflite"

# Class names for prediction results
class_names = ['Lumpy Skin', 'Flea Allergy', 'Hotspot', 'Mange', 'Ringworm'] + [f"Class_{i}" for i in range(5, 181)]


# ✅ Step 1: Convert .h5 to .tflite if not already converted
if not os.path.exists(TFLITE_MODEL_PATH):
    try:
        if not os.path.exists(H5_MODEL_PATH):
            raise FileNotFoundError(f"❌ Model file not found: {H5_MODEL_PATH}")

        print("🔄 Converting .h5 model to TensorFlow Lite format...")
        model = load_model(H5_MODEL_PATH, custom_objects={"LeakyReLU": LeakyReLU})
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        with open(TFLITE_MODEL_PATH, "wb") as f:
            f.write(tflite_model)

        print(f"✅ Converted and saved as {TFLITE_MODEL_PATH}")

    except Exception as e:
        print("❌ Error during model conversion:", str(e))
        exit(1)

# ✅ Step 2: Load TensorFlow Lite model
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


# ✅ Step 3: Image Preprocessing Function
def load_and_preprocess_image(image):
    try:
        img = image.resize((input_shape[1], input_shape[2]))  # Resize dynamically
        img_array = np.array(img).astype(np.float32) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        raise Exception(f"Image preprocessing error: {str(e)}")


# ✅ Step 4: Flask Route for Prediction
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


# ✅ Step 5: Run Flask Server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True, threaded=True)
