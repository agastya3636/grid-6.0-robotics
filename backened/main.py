from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel
import os
import requests

# Initialize the Flask app
app = Flask(__name__)

# Load the model and tokenizer
save_directory = "./saved_model_GOT_OCR2_0"
tokenizer = AutoTokenizer.from_pretrained(save_directory, trust_remote_code=True, use_auth_token=True, verify=False)

model = AutoModel.from_pretrained(save_directory, trust_remote_code=True)

# Set the model to evaluation mode and move it to the appropriate device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()




# Route to handle OCR requests
@app.route('/ocr', methods=['GET'])
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
        # Plain texts OCR
        res = model.chat(tokenizer, image_file, ocr_type='ocr')

        # Return the result as a JSON response
        return jsonify({"result": res}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the temporary image file
        if os.path.exists(image_path):
            os.remove(image_path)
        print(1)

# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug =True)


