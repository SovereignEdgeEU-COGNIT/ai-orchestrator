
import sys
import base64
import cv2
import object_detector
from flask import Flask, request, jsonify
import time
import logging
import numpy as np
import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the TensorFlow Lite model ONCE at startup
MODEL_PATH = 'model/cube_bottom_modelv1.tflite'
LABELS_FILE = 'metadata/labelmap.txt'

#print("Loading model...")
#interpreter = tf.lite.Interpreter(MODEL_PATH)
#interpreter.allocate_tensors()
#print("Model loaded successfully!")

#interpreter = None
#input_details = None
#output_details = None

# Load labels once
with open(LABELS_FILE, 'r') as f:
    LABELS = f.read().splitlines()

app = Flask(__name__)

def get_object_interest(detected_objects, target_label):
    """Find center of the detected object."""
    #print(f"Detected objects: {detected_objects}")
    #available_labels = [obj['obj'] for obj in detected_objects]
    #print(f"Available object labels: {available_labels}")
    for obj in detected_objects:
        if obj['obj'] == target_label:
            center_point = ((obj['x_min'] + obj['x_max']) // 2, (obj['y_min'] + obj['y_max']) // 2)
            bounding_box = {
                "x_min": obj["x_min"],
                "x_max": obj["x_max"],
                "y_min": obj["y_min"],
                "y_max": obj["y_max"],
                #"angle": obj["angle"]
            }
            return center_point, bounding_box

    #print(f"Object '{target_label}' not found in image!")
    return None

# OpenFaaS health check endpoint
@app.route("/_/health", methods=["GET"])
def health():
    return "OK", 200
def decode(img_b64):
    """Efficient image decoding using OpenCV (cv2) instead of TensorFlow."""
    try:
        img_bytes = base64.b64decode(img_b64)  # Decode Base64 to raw bytes
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)  # Convert bytes to NumPy array
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Decode image using OpenCV
        if img_cv is None:
            raise ValueError("Failed to decode image using OpenCV.")
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        logger.info("Image decoded successfully")
        return img_rgb
    except Exception as e:
        logger.error(f"Error decoding image: {str(e)}", exc_info=True)
        return None

@app.route("/init", methods=["POST"])
def init():
    """Initialize action (required for OpenWhisk blackbox runtime)"""
    #global  interpreter, input_details, output_details

    #try:
    #    logger.info("Initializing TFLite model")
    #    interpreter = tf.lite.Interpreter(MODEL_PATH)
    #    interpreter.allocate_tensors()
    #    input_details = interpreter.get_input_details()
    #    output_details = interpreter.get_output_details()
    #    logger.info("Model loaded and allocated successfully")
    #    return jsonify({"ok": True})
    #except Exception as e:
    #    logger.error(f"Model loading failed: {str(e)}", exc_info=True)
    #    return jsonify({"error": f"Model loading failed: {str(e)}"}), 500

    logger.info("init() called — model will be loaded per request.")
    return jsonify({"ok": True})

@app.route("/run", methods=["POST"])
def handle_detection():

    """Handle image detection requests."""
    try:
        logger.info("Starting object detection function")
        print("Received request...")
        start_time = time.time()  # Measure execution time
        #print(interpreter._get_delegate_status())

        data = request.get_json()
        if "value" in data:
            data = data["value"] #Extract the actual payload from OpenWhisk wrapper
            #print(f"Input Data: {data}")

     #exit if this is a prewarm activation
        if data.get("forcePrewarm"):
            logger.info("Prewarm request received — skipping detection.")
            #time.sleep(5)
            return jsonify({"status": "container prewarmed"}), 200

        object_of_interest = data.get('object_of_interest')
        if not object_of_interest:
            print("Missing object_of_interest", file=sys.stderr)
            return jsonify({"error": "Missing required parameters"}), 400

        img_b64 = data.get('frame')
        if not img_b64:
            print("No image provided", file=sys.stderr)
            return jsonify({"error": "No image provided"}), 400
        decode_start = time.time()
        try:
            img_cv = decode(img_b64)
        except Exception as e:
            return jsonify({"error": f"Image decoding failed: {str(e)}"}), 400
        # Fast image decoding using OpenCV
        #decode_start = time.time()
        #img_bytes = base64.b64decode(img_b64)
        #img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        #img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        decode_end = time.time()

        if img_cv is None:
            print("Failed to decode image", file=sys.stderr)
            return jsonify({"error": "Invalid image format"}), 400

        # Create and prepare TFLite interpreter safely per request
        interpreter = tf.lite.Interpreter(MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        inference_start = time.time()
        try:
            detected_objects = object_detector.detect_objects_from_image(interpreter, input_details, LABELS, img_cv)
            print(f"Detected Objects: {detected_objects}")
        except Exception as e:
            return jsonify({"error": f"Object detection failed: {str(e)}"}), 500
        inference_end = time.time()
        center_point, bounding_box = get_object_interest(detected_objects, object_of_interest)

        response_data = {
            "center_point": center_point,
            "bounding_box": bounding_box
        }
        end_time = time.time()  # Measure execution time
        # Print timing breakdown
        print(f"Image decoding time: {decode_end - decode_start:.4f} seconds")
        print(f"Inference time: {inference_end - inference_start:.4f} seconds")
        print(f"Total request processing time: {end_time - start_time:.4f} seconds")
        sys.stdout.flush()  # Force immediate output
        return jsonify(response_data)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return jsonify({"error": f"Unhandled exception: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
