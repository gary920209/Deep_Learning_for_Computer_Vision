import json
import os
import torch
from torch import Tensor, nn
from PIL import Image
from tqdm.auto import tqdm
import logging
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tokenizer import BPETokenizer
import timm
import torchvision.transforms as trns
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
import loralib as lora
import math
import collections


# ------------------------------------ #
#           Global variable            #
# ------------------------------------ #

device = "cuda:2" if torch.cuda.is_available() else "cpu"
PAD_TOKEN = 50256
UNK_TOKEN = 1
BOS_TOKEN = 50256
EOS_TOKEN = 50256
rank = 32

# Define start and end token IDs based on the tokenizer
tokenizer = BPETokenizer(encoder_file="encoder.json",  vocab_file="vocab.bpe")
endoftext_token = "<|endoftext|>"
endoftext_token_id = 50256
padding_token_id = 0  # Assuming padding token ID is 0

def custom_collate_fn(batch):
    images, captions = zip(*batch)
    
    # Add start and end tokens to each caption
    encoded_captions = [
        torch.cat([torch.tensor([endoftext_token_id]), caption, torch.tensor([endoftext_token_id])], dim=0)
        for caption in captions
    ]
    
    # Pad captions
    padded_captions = pad_sequence(encoded_captions, batch_first=True, padding_value=padding_token_id)
    print(padded_captions)
    # Stack images
    images = torch.stack(images)
    
    return images, padded_captions

image_transforms = create_transform(
    **resolve_data_config({}, model="vit_large_patch14_clip_224"), is_training=True
)

# ------------------------------------ #
#               Dataset                #
# ------------------------------------ #


class ImageCaptionDataset(Dataset):
    def __init__(self, image_folder, annotations_file, tokenizer, transform=None):
        """
        Args:
            image_folder (str): Path to the image directory.
            annotations_file (str): Path to the JSON annotations file.
            tokenizer (BPETokenizer): The tokenizer for encoding captions.
            transform (callable, optional): Transform to be applied on the images.
        """
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        # Map image IDs to file names
        self.image_id_to_file = {img['id']: img['file_name'] for img in data['images']}
        # Create a list of (image_id, caption) pairs
        self.captions = [(ann['image_id'], ann['caption']) for ann in data['annotations']]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_id, caption = self.captions[idx]
        image_file = self.image_id_to_file[image_id]
        image_path = os.path.join(self.image_folder, image_file)
        
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Tokenize caption
        caption_tokens = torch.tensor(self.tokenizer.encode(caption), dtype=torch.long)
        
        return image, caption_tokens

# ------------------------------------ #
#                 Model                #
# ------------------------------------ #



class Config:

    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=rank)
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=rank)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=rank)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=rank))
        ]))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = lora.Embedding(cfg.vocab_size, cfg.n_embd, r=rank),
            wpe = lora.Embedding(cfg.block_size, cfg.n_embd, r=rank),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, bias=False, r=rank)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, input_embeddings, caption_tokens=None):
        # Use position embeddings
        B, T, _ = input_embeddings.size()
        pos = torch.arange(T, dtype=torch.long, device=input_embeddings.device).unsqueeze(0)
        pos_embeddings = self.transformer.wpe(pos).expand(B, -1, -1)
        
        # Combine input_embeddings with position embeddings
        x = input_embeddings + pos_embeddings

        if caption_tokens is not None:
            # Get token embeddings for the caption tokens (teacher forcing)
            token_embeddings = self.transformer.wte(caption_tokens)
            x = torch.cat((x, token_embeddings), dim=1)

        x = self.transformer.ln_f(self.transformer.h(x))
        return self.lm_head(x)


class ImageCaptioningModel(nn.Module):
    def __init__(self, cfg, tokenizer, vit_model_name='vit_large_patch14_clip_224'):
        super().__init__()
        # Initialize Vision Transformer Encoder (ViT)
        self.vit_encoder = timm.create_model(
            vit_model_name, pretrained=True, num_classes=0
        ).to(device)
        # Ensure the ViT encoder does not include the classification head
        self.vit_encoder.reset_classifier(0)  # Remove classifier if present
        self.decoder = Decoder(cfg).to(device)
        self.tokenizer = tokenizer

        # Linear layer to map ViT output to match decoder's embedding size
        self.proj = nn.Linear(self.vit_encoder.embed_dim, cfg.n_embd)

    def top_p_sampling(self, logits, top_p=0.9, temperature=0.6, repetition_penalty=1.2, past_tokens=None):
        """
        Apply Top-p (nucleus) sampling with temperature and repetition penalty.
        
        Args:
            logits (Tensor): Logits of shape [batch_size, vocab_size] for the current token.
            top_p (float): Cumulative probability threshold for nucleus sampling.
            temperature (float): Temperature scaling factor for logits.
            repetition_penalty (float): Penalty factor for repeated tokens.
            past_tokens (Tensor): List of previously generated tokens for repetition penalty.

        Returns:
            Tensor: Selected token IDs of shape [batch_size].
        """
        # Apply repetition penalty
        if past_tokens is not None:
            for batch_idx, tokens in enumerate(past_tokens):
                for token in tokens:
                    # Penalize logits of previously generated tokens
                    if logits[batch_idx, token] < 0:
                        logits[batch_idx, token] *= repetition_penalty
                    else:
                        logits[batch_idx, token] /= repetition_penalty

        # Apply temperature scaling
        logits = logits / temperature

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Initialize list for selected tokens
        next_tokens = []

        # Process each item in the batch independently
        for i in range(probs.size(0)):
            sorted_probs, sorted_indices = torch.sort(probs[i], descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)

            # Mask out indices where cumulative probability exceeds top_p
            sorted_indices_to_keep = cumulative_probs < top_p
            sorted_indices_to_keep[0] = True  # Always keep at least one token

            # Get the filtered probabilities and renormalize
            filtered_indices = sorted_indices[sorted_indices_to_keep]
            filtered_probs = probs[i][filtered_indices]
            filtered_probs = filtered_probs / filtered_probs.sum()

            # Sample from the filtered probabilities
            next_token = filtered_indices[torch.multinomial(filtered_probs, 1)]
            next_tokens.append(next_token)

        return torch.stack(next_tokens)


    def forward(self, image, prompt, caption_tokens=None, mode='train', max_length=30, p=0.9, temperature=0.3, repetition_penalty=1.2):
        # Step 1: Pass the image through the ViT encoder
        vit_outputs = self.vit_encoder.forward_features(image)
        image_embedding = self.proj(vit_outputs)  # Take the [CLS] token representation

        if prompt is not None:
            # Step 2: Tokenize the prompt and get embeddings
            prompt_tokens = torch.tensor(self.tokenizer.encode(prompt), device=image.device).unsqueeze(0)
            prompt_embeddings = self.decoder.transformer.wte(prompt_tokens)
            # Duplicate the prompt embeddings for each item in the batch
            prompt_embeddings = prompt_embeddings.repeat(image.size(0), 1, 1)  # Repeat along batch dimension


        # Step 3: Concatenate prompt embeddings with the image embedding
        if prompt is not None:
            input_embeddings = torch.cat((prompt_embeddings, image_embedding), dim=1)
        else:
            input_embeddings = image_embedding

        # Training Mode (Teacher Forcing)
        if mode == 'train' and caption_tokens is not None:
            output = self.decoder(input_embeddings, caption_tokens)[:, 256:, :]
            return output  # Return logits for loss calculation

        # Inference Mode (Autoregressive Decoding)
        elif mode == 'inference':
            generated_tokens = []
            current_input = input_embeddings
            # Store past tokens to apply repetition penalty
            past_tokens = [[] for _ in range(image.size(0))]

            for _ in range(max_length):
                output = self.decoder(current_input)  # Pass the sequence so far to the decoder
                logits = output[:, -1, :]  # Get logits for the last token in the sequence
                
                # Use Top-p sampling with temperature and repetition penalty
                next_token = self.top_p_sampling(logits, top_p=p, temperature=temperature, repetition_penalty=repetition_penalty, past_tokens=past_tokens)
                generated_tokens.append(next_token.unsqueeze(1))  # Shape [batch_size, 1]
                for i, token in enumerate(next_token):
                    past_tokens[i].append(token.item())

                next_token_embedding = self.decoder.transformer.wte(next_token)#.unsqueeze(1)
                current_input = torch.cat((current_input, next_token_embedding), dim=1)

            # Concatenate all generated tokens along the sequence dimension
            return torch.cat(generated_tokens, dim=1)  # Shape: [batch_size, sequence_length]

def norm_long(x):
    x /= x.norm(dim=-1, keepdim=True)
    return x.long()

# ------------------------------------ #
#           Training stage             #
# ------------------------------------ #


def train_model(model, dataloader, optimizer, scheduler, num_epochs, device, prompt=None, save_folder='p2_outputs_austin'):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=padding_token_id)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
            images, captions = batch
            images, captions = images.to(device), captions.to(device)

            # Tokenize the prompt once per batch
            input_prompt = prompt

            # Shift target captions for teacher forcing
            input_captions = captions[:, :-1]
            target_captions = captions[:, :]

            # Forward pass with teacher forcing
            outputs = model(image=images, prompt=input_prompt, caption_tokens=input_captions, mode='train')
            outputs = outputs.reshape(-1, outputs.size(-1))  # Use .reshape() instead of .view()
            # print("target captions shape: ", target_captions.size())
            target_captions = target_captions.contiguous().view(-1)
            

            # Compute loss and backpropagate
            loss = criterion(outputs, target_captions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Visualization: Decode and print the last batch's output and target captions
            if batch_idx == len(dataloader) - 1:  # Last batch of the epoch
                with torch.no_grad():  # Disable gradient tracking for visualization
                    # Reshape to original batch format for decoding
                    output_tokens = model(image=images, prompt=input_prompt, mode='inference').view(images.size(0), -1)
                    target_tokens = target_captions.view(images.size(0), -1)
                    
        scheduler.step()
        trainable_weights = [
            name for name, param in model.named_parameters() if param.requires_grad == True
        ]
        save_weights = {
            k: v for k, v in model.state_dict().items() if k in trainable_weights
        }
        torch.save(save_weights, os.path.join(save_folder, f"model_lora_r16_{epoch}_1110_austin.pt"))



def main():
    # args parameters
    EPOCHS = 20
    # Paths to data
    image_folder = 'hw3_data/p2_data/images/train/'
    annotations_file = 'hw3_data/p2_data/train.json'
    save_folder = 'p2'
    os.makedirs(save_folder, exist_ok=True)

    # Create the dataset and dataloader
    dataset = ImageCaptionDataset(
        image_folder=image_folder,
        annotations_file=annotations_file,
        tokenizer=tokenizer,
        transform=image_transforms
    )

    # Create DataLoader
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

    # Model
    cfg = Config("hw3_data/p2_data/decoder_model.bin")
    model = ImageCaptioningModel(cfg=cfg, tokenizer=tokenizer, vit_model_name='vit_large_patch14_clip_224').to(device)
    # model.load_state_dict(lora_cfg, strict=False)
    lora.mark_only_lora_as_trainable(model)
    # Freeze encoder
    for param in model.proj.parameters():
        param.requires_grad = True

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = lr_scheduler.LinearLR(
        optimizer=optimizer,  # Your optimizer here
        start_factor=1.0,  # Start with the full learning rate
        end_factor=0.0,    # Linearly reduce to zero
        total_iters=EPOCHS  # Number of epochs for linear decay
    )

    # Run the training loop
    train_model(model, train_dataloader, optimizer, scheduler, num_epochs=EPOCHS, device=device, save_folder=save_folder)# prompt right now is useless

if __name__ == "__main__":
    main()

