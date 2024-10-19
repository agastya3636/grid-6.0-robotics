from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the processor and model from the pre-trained checkpoint
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

# Save the processor and model to a directory for later use
processor.save_pretrained("./blip-image-captioning-large-processor")
model.save_pretrained("./blip-image-captioning-large-model")