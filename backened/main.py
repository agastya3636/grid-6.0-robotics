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
from transformers import BlipProcessor, BlipForConditionalGeneration

from flask import Flask
from flask_cors import CORS


from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/ocr": {"origins": "http://localhost:5173"}})
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


# Load the saved model and processor
object_detection_processor = BlipProcessor.from_pretrained("./blip-image-captioning-large-processor")
object_detection_model = BlipForConditionalGeneration.from_pretrained("./blip-image-captioning-large-model").to("cuda")


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

def gen_ans(query):
    # API URL and API key (replace with your actual key)
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyDQz5dgbtryYd3MG_sDxfdFBPtL9JeBIPU"
    
    # Data to be sent (input for the language model)
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": query
                    }
                ]
            }
        ]
    }

    try:
        # Send the POST request
        response = requests.post(url, json=data)
        response.raise_for_status()  # Check for HTTP errors

        # Extract and print the generated response
        generated_text = response.json()['candidates'][0]['content']
        print(generated_text)
        return  generated_text

    except requests.exceptions.RequestException as e:
        print("Error making request:", e) 

@app.route('/caption', methods=['POST'])
def caption_image():
    # Check if an image file is provided in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    img_bytes = file.read()  # Read the image file as bytes
    
    # Open the image using PIL
    raw_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    text = request.form.get('text', '')  # Optional text for conditional captioning

    if text:
        # Conditional captioning
        inputs = object_detection_processor(raw_image, text, return_tensors="pt").to("cuda")
    else:
        # Unconditional captioning
        inputs = object_detection_processor(raw_image, return_tensors="pt").to("cuda")

    # Generate caption
    out = object_detection_model.generate(**inputs)
    caption = object_detection_processor.decode(out[0], skip_special_tokens=True)
    res = gen_ans(query=caption+" conert it in json also add count i.e no of perticular item")
    return jsonify({"caption": caption,"json":res})

# Route to handle OCR requests
@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Get the image from the request
    image_file = request.files['image']
    

    # Save the image temporarily
    image_path = "./temp_image.webp"
    print(image_path)
    image_file.save(image_path)

    # Perform OCR using the model
    try:
        # Pass the path of the saved image to the model
        res = ocr_model.chat(ocr_tokenizer, image_path, ocr_type='ocr')
        final = gen_ans(query=res+"  convert this in json format")
        print(final)
        # Return the result as a JSON response
        return jsonify({"result": res,"json":final}), 200
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
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        file = request.files['image']
        img_bytes = file.read()  # Read the image file as bytes
        image_tensor = process_classification_image(img_bytes)  # Preprocess the image
        
        # Make prediction
        with torch.no_grad():
            outputs = classification_model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # Map predicted index to the class label
        class_idx = predicted.item()
        class_name = class_labels[class_idx]
        response = {
            'json': {
                'parts': [
                    {
                        'text': f"""{{
                            'class_id': {class_idx},
                            'class_name': '{class_name}'
                        }}"""  # Use f-strings for formatting
                    }
                ]
            }
        }

        
        return (response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test successful!"})

# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
