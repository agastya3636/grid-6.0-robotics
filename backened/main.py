from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
import json
import io
from io import BytesIO
import requests
import os

# Initialize the Flask app
app = Flask(__name__)

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the OCR model and tokenizer
ocr_save_directory = "./saved_model_GOT_OCR2_0"

ocr_tokenizer = AutoTokenizer.from_pretrained(ocr_save_directory, trust_remote_code=True, use_auth_token=True, verify=False)
ocr_model = AutoModel.from_pretrained(ocr_save_directory, trust_remote_code=True)

# Set the model to evaluation mode and move it to the appropriate device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ocr_model = ocr_model.to(device).eval()


# Load the classification model (VGG16 in this case)
classification_model = models.vgg16(weights=None)
num_classes = 14  # Change this to the actual number of classes used during training
classification_model.classifier[6] = torch.nn.Linear(classification_model.classifier[6].in_features, num_classes)

# Load the saved weights for the classification model
classification_checkpoint_path = 'C:/Users/agast/Downloads/p/vgg16_fruit_freshness_model_50.pth'
classification_model.load_state_dict(torch.load(classification_checkpoint_path, map_location=device))
classification_model = classification_model.to(device)
classification_model.eval()  # Set the model to evaluation mode

# Load class labels from the JSON file for classification
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Preprocess the image (same as training transforms)
classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to process the input image for classification
def process_classification_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return classification_transform(image).unsqueeze(0).to(device)


# Route to handle OCR requests

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Get the image from the request
    image_file = request.files['image']

    # Save the image temporarily
    image_path = "./temp_image.webp"
    image_file.save(image_path)

    # Perform OCR using the model
    try:
        # Pass the path of the saved image to the model
        res = ocr_model.chat(ocr_tokenizer, image_path, ocr_type='ocr')

        # Return the result as a JSON response
        return jsonify({"result": res}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the temporary image file
        if os.path.exists(image_path):
            os.remove(image_path)
        print("Cleanup completed.")

# Route to handle image classification requests
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        file = request.files['file']
        img_bytes = file.read()  # Read the image file as bytes
        image_tensor = process_classification_image(img_bytes)  # Preprocess the image
        
        # Make prediction
        with torch.no_grad():
            outputs = classification_model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # Map predicted index to the class label
        class_idx = predicted.item()
        class_name = class_labels[class_idx]
        
        return jsonify({'class_id': class_idx, 'class_name': class_name})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
