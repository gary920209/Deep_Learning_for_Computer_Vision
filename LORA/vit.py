import os
import json
from PIL import Image
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

# Initialize the processor and model
model_id = "llava-hf/llava-1.5-7b-hf"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model and move to the correct device
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

# Load the processor
processor = AutoProcessor.from_pretrained(model_id)

# Path to the images folder
image_path = 'hw3_data/p1_data/images/val'

# Dictionary to store the filename and predicted captions
captions = {}

# Iterate over each image in the folder
for filename in os.listdir(image_path):
    if filename.endswith(".jpg"):  # Only process .jpg files
        image_path_full = os.path.join(image_path, filename)
        raw_image = Image.open(image_path_full).convert("RGB")  # Convert to RGB format

        # Define the conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image in a detailed caption."},
                    {"type": "image"},
                ],
            },
        ]

        # Apply the chat template to format the prompt
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Process the image and prompt together
        inputs = processor(
            images=raw_image,
            text=prompt,
            return_tensors='pt'
        ).to(device, torch.float16)  # Move to the correct device and dtype

        # Generate output from the model
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

        # Decode the output and store it in the dictionary
        caption = processor.decode(output[0][2:], skip_special_tokens=True)
        if "ASSISTANT:" in caption:
            caption = caption.split("ASSISTANT:")[1].strip()
        filename_without_extension = os.path.splitext(filename)[0]
        captions[filename_without_extension] = caption

# Save the captions to a JSON file
with open("image_captions.json", "w") as json_file:
    json.dump(captions, json_file, indent=4)

print("Captions saved successfully in image_captions.json!")
