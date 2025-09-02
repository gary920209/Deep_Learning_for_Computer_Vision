import os
import sys
import json

import torch
import numpy as np
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import matplotlib.pyplot as plt

 # ------- parameters ------- #
image_folder = sys.argv[1]
output_dir = sys.argv[2]
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


 # -------    model   ------- #

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to("cuda:2")
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
    "num_beams": 1,
    "output_attentions": True,
    "do_sample": False,
    "return_dict_in_generate": True
}

layer = int(sys.argv[3])
head = int(sys.argv[4])

if not os.path.exists(output_dir):
    with open(output_dir, "w") as f:
        json.dump({}, f)
    print(f"Created an empty JSON file at {output_dir}")

# Visualize attention maps
def visualize_attention_map(attentions, image_path, output_path, tokens):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(15, 10))
    vision_token_start = len(processor.tokenizer(conversation.split("<image>")[0], return_tensors='pt')["input_ids"][0]) + 1
    vision_token_end = vision_token_start + 576  # Adjust for your model's grid size
    # print("vision_token_start: ", vision_token_start)
    # print("vision_token_end: ", vision_token_end)
    joint_attentions = None
    for i, token in enumerate(tokens):
        # if i >= 10:
        #     break
        per_token_attention = attentions[i]
        last_layer_attention = per_token_attention[layer]
        print("last_layer_attention shape: ", last_layer_attention.size())
        attention = last_layer_attention[0, :, -1, vision_token_start:vision_token_end]  # Get attention for the <image> token
        # att_mat = torch.mean(attention, dim=0).squeeze(0)
        att_mat = attention[head].squeeze(0)
        att_mat = att_mat / att_mat.sum(dim=-1)
        # matrix multiplication to get the attention of the image tokens
        # joint_attentions = joint_attentions.to(dtype=torch.float32)
        # att_mat = att_mat.to(dtype=torch.float32)
        attention_vec = att_mat#joint_attentions @ att_mat
        attn_map = attention_vec.reshape(24, 24)# Reshape to [24, 24]
        attn_map = attn_map.cpu().numpy()
        attn_map -= attn_map.min()
        attn_map /= attn_map.max()

        # Resize to original image size
        attn_map_resized = np.array(Image.fromarray(np.uint8(attn_map * 255)).resize(image.size, resample=Image.BICUBIC))
        
        plt.subplot(4, 5, i + 1)
        plt.imshow(image)
        plt.imshow(attn_map_resized, cmap='jet', alpha=0.5)
        plt.title(token)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

captions = {}
print("Generating captions for images...")
for filename in sorted(os.listdir(image_folder)):
    if filename.endswith(".jpg"):
        image_path = os.path.join(image_folder, filename)
        image_id = os.path.splitext(filename)[0]

        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=conversation, images=image, return_tensors="pt").to("cuda:2", torch.float32)

        if "pixel_values" not in inputs or inputs["pixel_values"] is None:
            print(f"Error: Pixel values not generated correctly for {filename}. Skipping...")
            continue

        # Generate caption with attention
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            print("inputs: ", inputs["input_ids"].size(), inputs["pixel_values"].size())
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                **generation_config
            )
            # print("outputs: ", outputs.sequences[0].size())
            #print the attributes of the outputs
            print(outputs.__doc__)
            for attr in dir(outputs):
                if attr == "__doc__":
                    print(outputs.__doc__)

        caption = processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        print("caption: ", caption)
        attentions = outputs.attentions
        print("attentions length", len(outputs.attentions))
        if attentions is not None:
            tokens = processor.tokenizer.convert_ids_to_tokens(outputs.sequences[0][inputs["input_ids"].size(1):])
            output_path = os.path.join('output', f"{image_id}_attention.png")
            visualize_attention_map(attentions, image_path, output_path, tokens)

        if "ASSISTANT:" in caption:
            caption = caption.split("ASSISTANT:")[-1].strip()

        captions[image_id] = caption

