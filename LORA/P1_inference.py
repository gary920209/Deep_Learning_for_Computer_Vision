import os
import sys
import json

import torch
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

 # ------- parameters ------- #
image_folder = sys.argv[1]
output_dir = sys.argv[2]
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

 # -------    model   ------- #

model_id = "llava-hf/llava-1.5-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float32,  
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    quantization_config=bnb_config,  
    trust_remote_code=True
).to("cuda")
processor = LlavaProcessor.from_pretrained(model_id)
torch.cuda.empty_cache()

# Set patch size and vision feature select strategy if they're not already in the processor
if not hasattr(processor, "patch_size"):
    processor.patch_size = 32  
if not hasattr(processor, "vision_feature_select_strategy"):
    processor.vision_feature_select_strategy = "mean"  


# Set up the conversation prompt and generation configuration
conversation = "USER: <image> \n  Give me a desciption of the image in one short sentence. ASSISTANT:"
generation_config = {
    "max_new_tokens": 30,
    "num_beams": 2,
    "do_sample": False
}


# Initialize a dictionary to store captions
captions = {}
print("Generating captions for images...")
for filename in sorted(os.listdir(image_folder)):
    if filename.endswith(".jpg"):
        image_path = os.path.join(image_folder, filename)
        image_id = os.path.splitext(filename)[0]  
        image = Image.open(image_path).convert("RGB")  
        inputs = processor(text=conversation, images=image, return_tensors="pt").to("cuda", torch.float32)

        # Check if pixel values are processed
        if "pixel_values" not in inputs or inputs["pixel_values"] is None:
            continue

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                **generation_config
            )

        caption = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract only the assistant's response if needed
        if "ASSISTANT:" in caption:
            caption = caption.split("ASSISTANT:")[-1].strip()  
        captions[image_id] = caption

# Save the captions dictionary to a JSON file
if not os.path.exists(output_dir):
    with open(output_dir, "w") as f:
        json.dump({}, f)
with open(output_dir, "w") as f:
    json.dump(captions, f, indent=4)
