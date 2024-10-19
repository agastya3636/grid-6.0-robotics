from flask import Flask, request, jsonify
from PIL import Image
import torch
import io
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the saved model and processor
object_detection_processor = BlipProcessor.from_pretrained("./blip-image-captioning-large-processor")
object_detection_model = BlipForConditionalGeneration.from_pretrained("./blip-image-captioning-large-model").to("cuda")

app = Flask(__name__)

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

    return jsonify({"caption": caption})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug =True)
