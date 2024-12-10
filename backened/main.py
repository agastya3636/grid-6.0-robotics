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
from pymongo import MongoClient
from pydantic import BaseModel, validator
from typing import List
from datetime import datetime
from flask import Flask
from flask_cors import CORS
import re
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

gemini_password=os.getenv('GeminiPassword')
app = Flask(__name__)
CORS(app, resources={r"/ocr": {"origins": "http://localhost:5173"}})

#setup database
client = MongoClient('localhost', 27017)
db = client['test-database']
collection = db['test-collection']
expiy_and_des = db['test-expiy']
count=db['test-count']
#setup schema 
class Freshness(BaseModel):
    timestamp: datetime
    produce: str
    freshness: int
    expected_life_span: str
    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.utcnow()



def extract_details(class_name):
    try:
        match = re.match(r"([a-zA-Z]+)\((\d+)-(\d+)\)", class_name)
        
        if match:
            produce = match.group(1)  # Extract the produce (e.g., "Banana")
            lower_value = int(match.group(2))  # Extract the lower value of the shelf life (e.g., 5)
            upper_value = int(match.group(3))  # Extract the upper value of the shelf life (e.g., 10)
            
            # Calculate the freshness as the lower value divided by 5
            freshness = lower_value / 5
            
            # Create a Freshness Pydantic model instance
            freshness_data = Freshness(
                produce=produce,
                expected_life_span=f"{lower_value}-{upper_value}",
                freshness=freshness,
                timestamp=datetime.utcnow()  # Set the current timestamp
            )

            # Print for debugging purposes
            print(f"Produce: {produce}, Shelf Life: {lower_value}-{upper_value}, Freshness: {freshness}")

            # Insert the Freshness data into MongoDB
            collection.insert_one(freshness_data.dict())  # Use .dict() to convert Pydantic model to a dictionary

            return freshness_data.dict()  # Return the dictionary form of the model for further processing
        else:
            raise ValueError(f"Invalid class_name format: {class_name}")
    
    except Exception as e:
        print(f"Error in extract_details: {str(e)}")
        raise  # Re-raise the error to propagate it to the calling function

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
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={gemini_password}"
    
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

# @app.route('/caption', methods=['POST'])
# def caption_image():
#     # Check if an image file is provided in the request
#     if 'image' not in request.files:
#         return jsonify({"error": "No image provided"}), 400

#     file = request.files['image']
#     img_bytes = file.read()  # Read the image file as bytes
    
#     # Open the image using PIL
#     raw_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

#     text = request.form.get('text', '')  # Optional text for conditional captioning

#     if text:
#         # Conditional captioning
#         inputs = object_detection_processor(raw_image, text, return_tensors="pt").to("cuda")
#     else:
#         # Unconditional captioning
#         inputs = object_detection_processor(raw_image, return_tensors="pt").to("cuda")

#     # Generate caption
#     out = object_detection_model.generate(**inputs)
#     caption = object_detection_processor.decode(out[0], skip_special_tokens=True)
#     res = gen_ans(query=caption+" conert it in json also add count i.e no of perticular item")
#     #store in database
#     if res:
#         try:
#             # Insert the Freshness data into MongoDB
#             count.insert_one(res.parts[0])  # Use .dict() to convert Pydantic model to a dictionary
#             return jsonify({"caption": caption,"json":res})
#         except Exception as e:
#             print(f"Error in extract_details: {str(e)}")
#             return jsonify({'error': f"Error in extract_details: {str(e)}"}), 500
    
@app.route('/caption', methods=['POST'])
def caption_image():
    # Check if an image file is provided in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    img_bytes = file.read()  # Read the image file as bytes

    try:
        # Open the image using PIL
        raw_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Optional text for conditional captioning
        text = request.form.get('text', '')

        if text:
            # Conditional captioning
            inputs = object_detection_processor(raw_image, text, return_tensors="pt").to("cuda")
        else:
            # Unconditional captioning
            inputs = object_detection_processor(raw_image, return_tensors="pt").to("cuda")

        # Generate caption
        out = object_detection_model.generate(**inputs)
        caption = object_detection_processor.decode(out[0], skip_special_tokens=True)

        # Call gen_ans to process the caption
        res = gen_ans(query=caption + " convert it in json also add count i.e no of particular item and dont give me comment only the json data")

        if res:
            try:
                # Convert `res` to a dictionary if it's not already
                json_data = json.loads(res) if isinstance(res, str) else res
                g={"caption": caption, "json": json_data}
                # Insert the JSON data into MongoDB
                count.insert_one(g)

                return jsonify({"caption": caption, "json": json_data}), 200
            except Exception as e:
                print(f"Error in MongoDB insertion: {str(e)}")
                return jsonify({'error': f"Error in MongoDB insertion: {str(e)}"}), 500
        else:
            return jsonify({"caption": caption, "json": None}), 200
    except Exception as e:
        print(f"Error in image captioning: {str(e)}")
        return jsonify({"error": f"Error in image captioning: {str(e)}"}), 500


# Route to handle OCR requests
# @app.route('/ocr', methods=['POST'])
# def ocr():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image provided"}), 400

#     # Get the image from the request
#     image_file = request.files['image']
    

#     # Save the image temporarily
#     image_path = "./temp_image.webp"
#     print(image_path)
#     image_file.save(image_path)

#     # Perform OCR using the model
#     try:
#         # Pass the path of the saved image to the model
#         res = ocr_model.chat(ocr_tokenizer, image_path, ocr_type='ocr')
#         print(res)
#         final = gen_ans(query=res+" convert this in json format if expiry date is present the give following also Brand , Expiry date , Expired(yes/no) ,Expected life span (Days) dont add json as heading give only json")
#         print(final)
#         if(final):
#             try:
#                 # Insert the Freshness data into MongoDB
#                 g=jsonify({"result": res,"json":final})
#                 expiy_and_des.insert_one(g)  # Use .dict() to convert Pydantic model to a dictionary
#                 return jsonify({"result": res,"json":final}),200
#             except Exception as e:
#                 print(f"Error in extract_details: {str(e)}")
#                 return jsonify({'error': f"Error in extract_details: {str(e)}"}), 500
#         else:
#             return jsonify({"result": res}),200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#     finally:
#         # Clean up the temporary image file
#         if os.path.exists(image_path):
#             os.remove(image_path)
#         print("Cleanup completed.")

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
        print(res)
        final = gen_ans(query=res+" convert this in json format if expiry date is present the give following also Brand , Expiry date , Expired(yes/no) ,Expected life span (Days) dont add json as heading give only json")
        # print(final)
        if final:
            try:
                # Convert `final` to a dictionary if it is not already
                final_json = json.loads(final) if isinstance(final, str) else final
                
                # Insert the Freshness data into MongoDB
                document = {"result": res, "json": final_json}
                expiy_and_des.insert_one(document)  # Insert dictionary into MongoDB
                return jsonify({"result": res, "json": final_json}), 200
            except Exception as e:
                print(f"Error in extract_details: {str(e)}")
                return jsonify({'error': f"Error in extract_details: {str(e)}"}), 500
        else:
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
        try:
            demore = extract_details(class_name)  # Assuming extract_details is defined
            
            
        except Exception as e:
            print(f"Error in extract_details: {str(e)}")
            return jsonify({'error': f"Error in extract_details: {str(e)}"}), 500
        
        response = {
            'json': {
                'parts': [
                    {
                        'text': f"""{demore}""" 
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
