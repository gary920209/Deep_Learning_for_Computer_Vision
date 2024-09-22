import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the SAM model
model_type = "vit_h"  # Choose from "vit_base", "vit_large", "vit_h"
checkpoint_path = "sam_vit_h_4b8939.pth"  
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.eval()

if torch.cuda.is_available():
    sam = sam.cuda()

# Create the automatic mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# Load and process images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

# Convert PIL images to numpy arrays for SAM
image_arrays = [np.array(img) for img in images]

# Use SAM's automatic mask generator to segment the images
for idx, image_np in enumerate(image_arrays):
    masks = mask_generator.generate(image_np)

    # Visualize the result by overlaying the mask on the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    
    # Visualize the largest mask (by area) as an example
    if len(masks) > 0:
        best_mask = max(masks, key=lambda x: x['area'])['segmentation']
        plt.imshow(best_mask, alpha=0.5)  # Overlay the mask

    # Save the resulting image
    plt.axis('off')
    plt.savefig(f"segmented_result_{idx}.png")
    plt.close()
