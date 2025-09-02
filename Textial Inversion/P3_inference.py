import argparse, os, json

import torch
import numpy as np
from PIL import Image
from contextlib import nullcontext
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

def load_model_from_config(config, ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    _, _ = model.load_state_dict(sd, strict=False)
    model.cuda().eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_json', help='Path to input JSON file', default='hw2_data/textual_inversion/input.json')
    # parser.add_argument('--outdir', help='Directory to save generated images',default='p3')
    # parser.add_argument('--ckpt', help='Path to the model checkpoint file',default='stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt')
    parser.add_argument("input_json", type=str, help="Path to the input JSON file containing prompts.")
    parser.add_argument("outdir", type=str, help="Directory to write results to")
    parser.add_argument("ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt", help="Model checkpoint path")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of ddim sampling steps")
    parser.add_argument("--n_samples", type=int, default=4, help="Samples per batch (set to 1 to avoid CUDA OOM)")
    parser.add_argument("--n_images", type=int, default=25, help="Total images to generate per prompt")
    parser.add_argument("--H", type=int, default=512, help="Image height in pixels")
    parser.add_argument("--W", type=int, default=512, help="Image width in pixels")
    parser.add_argument("--scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--config", type=str, default="stable-diffusion/configs/stable-diffusion/v1-inference.yaml", help="Model config path")
    parser.add_argument('--precision', type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
    opt = parser.parse_args()
    seed_everything(148) # Seed for reproducibility 
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sampler = DPMSolverSampler(model)
    embedding_path_1 = 'stable-diffusion/emb_0.pt'
    special_tokens_1 = ['<new1>-0', '<new1>-1', '<new1>-2', '<new1>-3']
    multi_token_1 = "<new1>-0 <new1>-1 <new1>-2 <new1>-3"
    saved_embeddings_1 = torch.load(embedding_path_1).to(device)  
    assert saved_embeddings_1.shape[0] == len(special_tokens_1), "Number of embeddings and tokens must match."
    clip_embedder = model.cond_stage_model
    existing_vocab = clip_embedder.tokenizer.get_vocab()
    for special_token in special_tokens_1:
        if special_token not in existing_vocab:
            clip_embedder.tokenizer.add_tokens(special_token)
    clip_embedder.transformer.resize_token_embeddings(len(clip_embedder.tokenizer))

    for idx, special_token in enumerate(special_tokens_1):
        new_token_id = clip_embedder.tokenizer.convert_tokens_to_ids([special_token])[0]
        with torch.no_grad():
            clip_embedder.transformer.get_input_embeddings().weight[new_token_id] = saved_embeddings_1[idx]

    embedding_path_2 = 'stable-diffusion/emb_1.pt' 
    special_token_2 = '<new2>'
    saved_embedding_2 = torch.load(embedding_path_2).to(device)  # Load and move to GPU
    clip_embedder = model.cond_stage_model

    if special_token_2 not in clip_embedder.tokenizer.get_vocab():
        clip_embedder.tokenizer.add_tokens([special_token_2])
        clip_embedder.transformer.resize_token_embeddings(len(clip_embedder.tokenizer))
    new_token_id_2 = clip_embedder.tokenizer.convert_tokens_to_ids([special_token_2])[0]
    
    with torch.no_grad():
        clip_embedder.transformer.get_input_embeddings().weight[new_token_id_2] = saved_embedding_2
    with open(opt.input_json, "r") as f:
        data = json.load(f)

    os.makedirs(opt.outdir, exist_ok=True)

    for item_key, item_value in data.items():
        prompts = item_value["prompt"]
        prompts = [prompt_str.replace("<new1>", multi_token_1) for prompt_str in prompts]
        item_dir = os.path.join(opt.outdir, item_key)
        os.makedirs(item_dir, exist_ok=True)
        for prompt_idx, prompt_text in enumerate(prompts):
            prompt_dir = os.path.join(item_dir, str(prompt_idx))
            os.makedirs(prompt_dir, exist_ok=True)
            total_images_generated = 0  # Track total images generated for each prompt
            precision_scope = autocast if opt.precision=="autocast" else nullcontext
            with torch.no_grad():
                with precision_scope("cuda"):
                    while total_images_generated < opt.n_images:
                        # Conditioning and sampling
                        c = model.get_learned_conditioning([prompt_text] * opt.n_samples)
                        uc = model.get_learned_conditioning(opt.n_samples * [""]) if opt.scale != 1.0 else None
                        shape = [4, opt.H // 8, opt.W // 8]
                        
                        # Generate a batch of images
                        samples_ddim, _ = sampler.sample(
                            S=opt.ddim_steps,
                            conditioning=c,
                            batch_size=opt.n_samples,
                            shape=shape,
                            eta=0.0,
                            verbose=False,
                            unconditional_guidance_scale=opt.scale,
                            unconditional_conditioning=uc
                        )
                        
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, 0.0, 1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        for idx in range(opt.n_samples):
                            if total_images_generated >= opt.n_images:
                                break
                            img = Image.fromarray((x_samples_ddim[idx] * 255).astype(np.uint8))
                            img_path = os.path.join(prompt_dir, f"{total_images_generated}.png")
                            img.save(img_path)
                            total_images_generated += 1

if __name__ == "__main__":
    main()
