import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import sys
import io
from ultralytics import YOLO
import easyocr

# Set the default encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
app = Flask(__name__)

# Load your pre-trained vegetable classification model
vegetables = [
    "banana", "beans broad", "beans cluster", "beans haricot", "beetroot",
    "bitter guard", "bottle guard", "brinjal long", "brinjal[purple]", "cabbage",
    "capsicum green", "carrot", "cauliflower", "chilli green", "colocasia arvi",
    "corn", "cucumber", "drumstick", "garlic", "ginger", "ladies finger",
    "lemons", "Onion red", "potato", "sweet potato", "tomato", "Zuchini"
]
model = YOLO('fresh_model.pt')
# Load the YOLO model
yolo_model = YOLO('another_model.pt')  # Adjust the path as needed
CONFIDENCE_THRESHOLD = 0.1

# Initialize EasyOCR reader (for English language)
reader = easyocr.Reader(['en'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and decode image data
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,

        # Decode the base64 string into a NumPy array
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)

        # Convert the NumPy array into an OpenCV image (BGR format)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform inference using the YOLO model
        results = model(image)  # Pass the image to the YOLO model
        detections = results[0].boxes
        detected_vegetables = []
        for box in detections:
            if box.conf > CONFIDENCE_THRESHOLD:
                class_index = int(box.cls[0].item())  # Get class index as an integer
                
                # Use model's class names dynamically
                vegetable_name = model.names[class_index]  # Get the name directly from the model
                detected_vegetables.append(vegetable_name)
        detected_classes = detected_vegetables

        return jsonify({'predictions': detected_classes})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Extract and decode image data
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,

        # Decode the base64 string into a NumPy array
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)

        # Convert the NumPy array into an OpenCV image (BGR format)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform inference using the YOLO model
        results = yolo_model(image)  # Pass the image to the YOLO model

        # Extract the detected classes
        class_indices = results[0].boxes.cls.cpu().numpy().astype(int)  # Get the class indices
        unique_classes = np.unique(class_indices)  # Get unique classes detected

        # Map class indices to labels if needed
        # Assuming you have a mapping for YOLO classes
        detected_classes = [str(cls) for cls in unique_classes]

        return jsonify({'predictions': detected_classes})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        # Extract and decode the base64 image data
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove "data:image/jpeg;base64,"
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert the image to RGB (easyocr works better with RGB images)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use easyocr to read text from the image
        results = reader.readtext(rgb_image)

        # Extract detected text from the results
        extracted_text = ' '.join([res[1] for res in results])  # Combine all detected text

        # Check if any text is detected (non-empty string)
        text_present = True if extracted_text.strip() else False

        return jsonify({'text_present': text_present, 'extracted_text': extracted_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
