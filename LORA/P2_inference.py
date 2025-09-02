'''
inferencing script for image captioning
Ref: https://github.com/Lucashien/
'''
import os
import sys
import json
import math
import collections

import timm
import torch
import torch.nn.functional as F
import random
import loralib as lora
from tqdm import tqdm
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

from tokenizer import BPETokenizer

# --- Constants ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rank = 32

# ---- Path -----
image_folder = sys.argv[1]
output_file = sys.argv[2]
decoder_model_path = sys.argv[3]
# image_folder = 'hw3_data/p2_data/images/val'
# output_file = 'output.json'
# decoder_model_path = 'hw3_data/p2_data/decoder_model.bin'
prompt = 'prompt.txt'
encoder_file = 'encoder.json'
vocab_file = 'vocab.bpe'
checkpoint_path = 'best_model.pt'

# ---- Dataset ----

class ImageDataset(Dataset):
    def __init__(self, image_dir, preprocess=None):
        self.images_dir = image_dir
        self.preprocess = preprocess
        self.images = []
        self._load_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_file = self.images[idx]
        image_path = os.path.join(self.images_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        if self.preprocess:
            image = self.preprocess(image)
        return image_file, image

    def _load_data(self):
        for file in sorted(os.listdir(self.images_dir)):
            self.images.append(file)

# --- Model ---
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
        self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=rank,lora_alpha=0.5*rank)
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=rank,lora_alpha=0.5*rank)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))
        self.att = None  # Initialize attribute to store attention weights


    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        self.att = att  # Store attention weights (for visualization)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=rank,lora_alpha=0.5*rank)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=rank,lora_alpha=0.5*rank))
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
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, r=rank, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor, image_features: Tensor = None, max_len: int = None):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        if image_features is not None:
            x = torch.cat([image_features, x], dim=1)
        x = self.transformer.h(x)
        x = self.transformer.ln_f(x)
        
        if image_features is not None:
            x = x[:, -max_len:, :]
        x = self.lm_head(x)
        return x

    def generate(self, x: Tensor, image_features: Tensor = None):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        if image_features is not None:
            x = torch.cat([image_features, x], dim=1)
        x = self.transformer.h(x)
        x = self.transformer.ln_f(x)
        
        x = x[:, -1, :]  
        x = self.lm_head(x)
        return x

class ImageCaptionModel(nn.Module):
    def __init__(self, decoder_path, encoder, tokenizer, prompt):
        super().__init__()
        self.decoder = self._load_decoder(decoder_path)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.projector = nn.Linear(1024, 768)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss = 0
        self.image_prompt = open(prompt, "r").read()
        self.BOS_TOKEN_ID = self.tokenizer.encode("<|endoftext|>", allowed_special="<|endoftext|>")[0]

    def forward(self, images, captions):
        image_features = self.encoder.forward_features(images)
        image_features = self.projector(image_features)
        image_prompt_tokens = self.tokenizer.encode(self.image_prompt)

        text_tokens = []
        target_tokens = []
        for caption in captions:
            encoded_cap = self.tokenizer.encode(caption)
            if encoded_cap[0] != self.BOS_TOKEN_ID:
                encoded_cap = [self.BOS_TOKEN_ID] + encoded_cap
            if encoded_cap[-1] != self.BOS_TOKEN_ID:
                encoded_cap += [self.BOS_TOKEN_ID]
            text_tokens.append(encoded_cap[:-1])  # Remove last token
            target_tokens.append(encoded_cap[1:])  # Remove first token

        max_len = max(len(t) for t in text_tokens)
        text_tokens = [image_prompt_tokens + self._pad_text_tokens(t, max_len, self.BOS_TOKEN_ID) for t in text_tokens]
        target_tokens = [self._pad_text_tokens(t, max_len, -100) for t in target_tokens]

        text_tokens = torch.tensor(text_tokens).to(device)
        target_tokens = torch.tensor(target_tokens).to(device)

        outputs = self.decoder(text_tokens, image_features, max_len)

        return outputs, target_tokens

    def generate(self, images, max_new_tokens=30):
        batch_size = len(images)

        # Get image features
        image_features = self.encoder.forward_features(images)
        image_features = self.projector(image_features)

        image_prompt_tokens = self.tokenizer.encode(self.image_prompt)
        sequences = torch.tensor(image_prompt_tokens, device=device)
        sequences = sequences.repeat(batch_size, 1)
        bos_tokens = torch.full(
            (batch_size, 1), self.BOS_TOKEN_ID, dtype=torch.long, device=device
        )
        sequences = torch.cat([sequences, bos_tokens], dim=1)
        for _ in range(max_new_tokens):
            next_token_logits = self.decoder.generate(
                sequences, image_features
            ).log_softmax(-1)
            next_tokens = next_token_logits.argmax(-1, keepdim=True)
            sequences = torch.cat([sequences, next_tokens], dim=-1)
        sequences = sequences.tolist()

        # get the tokens between the BOS and EOS tokens
        first_eos_idx = sequences[0].index(self.BOS_TOKEN_ID)
        for i in range(batch_size):
            try:
                sequences[i] = sequences[i][first_eos_idx+1:sequences[i].index(self.BOS_TOKEN_ID, first_eos_idx+1)]
            except:
                sequences[i] = sequences[i][first_eos_idx+1:]
        return sequences
    
    def _load_decoder(self, decoder_path):
        cfg = Config(checkpoint=decoder_path)
        decoder = Decoder(cfg)
        decoder = decoder.to(device)
        return decoder

    def _pad_text_tokens(self, text_token, max_text_seq_len, pad_token):
        text_token = text_token + [pad_token] * (max_text_seq_len - len(text_token))
        return text_token

# --- Inference ---

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


if __name__ == "__main__":
    # seed
    seed_everything(42)
    tokenizer = BPETokenizer(encoder_file=encoder_file, vocab_file=vocab_file)
    encoder = timm.create_model("vit_large_patch14_clip_224.laion2b", pretrained=True)
    data_config = timm.data.resolve_model_data_config(encoder)
    validation_preprocess = timm.data.create_transform(**data_config, is_training=False)

    model = ImageCaptionModel(
        decoder_model_path, encoder, tokenizer, prompt
    ).to(device)

    model.load_state_dict(
        torch.load(checkpoint_path),
        strict=False,
    )
    checkpoint_params = sum(p.numel() for p in torch.load(checkpoint_path).values())
    print(f"Total parameters in checkpoint: {checkpoint_params}")

    val_dataset = ImageDataset(image_folder, validation_preprocess)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    outputs = {}
    model.eval()
    
    # Generate captions for validation images
    with torch.no_grad():
        for image_filenames, images in tqdm(
            val_dataloader, desc="Generating captions", unit="batch"
        ):
            images = images.to(device)
            predicted_tokens = model.generate(images, max_new_tokens=30)
            decoded_captions = tokenizer.batch_decode(predicted_tokens)
            assert len(image_filenames) == len(decoded_captions)
            for image_filename, decoded_caption in zip(
                image_filenames, decoded_captions
            ):
                image_filename = image_filename.split(".")[0]
                outputs[image_filename] = decoded_caption

    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            json.dump({}, f)
        print(f"Created an empty JSON file at {output_file}")
    with open(output_file, "w") as f:
        json.dump(outputs, f, indent=4)

