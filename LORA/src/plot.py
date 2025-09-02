import argparse
import os
import pathlib
import random
from PIL import Image

import torch
import timm
import timm.data
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt
import numpy as np

from model import ImageCaptionModel
from tokenizer import BPETokenizer
from config import DEVICE, SEED, DATA_CFG


# === Configuration Constants ===
IMAGE_PROMPT_PATH = DATA_CFG["image_prompt_path"]
ENCODER_FILE = DATA_CFG["encoder_file"]
VOCAB_FILE = DATA_CFG["vocab_file"]


# === Utility Functions ===
def set_random_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


class ImageDataset(Dataset):
    """Dataset class for loading images and applying transformations."""
    
    def __init__(self, root_dir, transform):
        self.transform = transform
        self.image_paths = list(root_dir.glob("*"))

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        image = self.transform(image)
        image_name = os.path.splitext(self.image_paths[index].name)[0]
        return image, image_name

    def __len__(self):
        return len(self.image_paths)


def attach_attention_hook(model, attention_features, feature_shapes):
    """Attach a forward hook to extract attention weights from the model."""
    
    def extract_attention_weights(module, inputs, outputs):
        batch_size, seq_len, _ = inputs[0].size()
        attention_weights = module.att
        avg_attention = attention_weights.mean(dim=1)  # Average across heads
        attention_features.append(avg_attention.detach().cpu())
        feature_shapes.append((seq_len, seq_len))

    attention_layer = model.decoder.transformer.h[-1].attn
    hook_handle = attention_layer.register_forward_hook(extract_attention_weights)
    return [hook_handle]


def visualize_attention_map(attention_matrix, token_ids, token_offset, patch_size, output_file, image_file, tokenizer):
    """Visualize attention map overlayed on the image."""
    seq_len, _ = attention_matrix.size()
    fig, axes = plt.subplots(
        nrows=(len(token_ids) + 4) // 5, ncols=5, figsize=(16, 8)
    )
    axes = axes.flatten()
    
    for idx, token_id in enumerate(token_ids):
        if idx + token_offset >= seq_len:
            continue

        attention_vector = attention_matrix[idx + token_offset, 1:257]
        attention_map = attention_vector.view(*patch_size)
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        original_image = Image.open(image_file)
        mask = resize(attention_map.unsqueeze(0), original_image.size[::-1]).squeeze(0)
        mask = (mask.numpy() * 255).astype(np.uint8)

        axes[idx].imshow(original_image)
        axes[idx].imshow(mask, alpha=0.7, cmap='jet')
        axes[idx].set_title(
            "<SOS>" if idx == 0 else "<EOS>" if idx == len(token_ids) - 1 else tokenizer.decode([token_id])
        )
        axes[idx].axis("off")
    
    for ax in axes[len(token_ids):]:
        ax.axis("off")

    plt.savefig(output_file)
    plt.close(fig)


# === Main Program ===
def main(args):
    set_random_seed(SEED)

    # Initialize tokenizer and encoder
    tokenizer = BPETokenizer(encoder_file=ENCODER_FILE, vocab_file=VOCAB_FILE)
    encoder_model = timm.create_model("vit_large_patch14_clip_224.laion2b", pretrained=True)
    preprocess = timm.data.create_transform(
        **timm.data.resolve_model_data_config(encoder_model), is_training=False
    )

    # Create image dataset
    dataset = ImageDataset(root_dir=args.image_dir, transform=preprocess)

    # Load the captioning model
    caption_model = ImageCaptionModel(
        decoder_model_path=args.decoder_path,
        encoder=encoder_model,
        tokenizer=tokenizer,
        image_prompt_path=IMAGE_PROMPT_PATH
    ).to(DEVICE)
    caption_model.load_state_dict(torch.load(args.checkpoint_path, map_location=DEVICE), strict=False)
    caption_model.eval()

    # Process images in the dataset
    for image, image_name in dataset:
        attention_features, feature_shapes = [], []

        # Attach attention hook and process image
        hooks = attach_attention_hook(caption_model, attention_features, feature_shapes)
        image_tensor = image.unsqueeze(0).to(DEVICE)
        output_ids, offset = caption_model.generate_for_viz(image_tensor, max_new_tokens=30)

        attention_matrix = attention_features[-1].squeeze(0)
        visualize_attention_map(
            attention_matrix=attention_matrix,
            token_ids=output_ids[0],
            token_offset=offset,
            patch_size=(16, 16),
            output_file=args.output_dir / f"{image_name}.png",
            image_file=args.image_dir / f"{image_name}.jpg",
            tokenizer=tokenizer,
        )

        for hook in hooks:
            hook.remove()


# === Argument Parsing ===
def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize attention maps for image captioning.")
    parser.add_argument("--image_dir", type=pathlib.Path, required=True, help="Path to the directory containing images.")
    parser.add_argument("--decoder_path", type=pathlib.Path, required=True, help="Path to the decoder model.")
    parser.add_argument("--checkpoint_path", type=pathlib.Path, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--output_dir", type=pathlib.Path, required=True, help="Directory to save the output visualizations.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
