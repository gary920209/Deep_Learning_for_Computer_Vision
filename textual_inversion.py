import os
import numpy as np
from packaging import version
import matplotlib.pyplot as plt

import PIL
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.autograd import Variable

from einops import rearrange
# from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from pathlib import Path


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Load model from config and checkpoint."""
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    model.cond_stage_model.device = device
    return model


class PreprocessImage(Dataset):
    def __init__(self, data_root, size=512, repeats=100, interpolation="bicubic", flip_p=0.5, set="train", center_crop=False):
        """Initialize dataset for image preprocessing."""
        self.data_root = data_root
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.image_paths = [os.path.join(data_root, file) for file in os.listdir(data_root)]
        self.num_images = len(self.image_paths)
        self._length = self.num_images * (repeats if set == "train" else 1)

        self.interpolation = PIL_INTERPOLATION[interpolation]
        self.flip_transform = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        """Preprocess a single image."""
        image = Image.open(self.image_paths[i % self.num_images]).convert("RGB")
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[:2])
            h, w = img.shape[:2]
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img).resize((self.size, self.size), resample=self.interpolation)
        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8) / 127.5 - 1.0
        return {"pixel_values": torch.from_numpy(image).permute(2, 0, 1)}


@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1, t_start=-1, log_every_t=None, till_T=None, verbose=True):
    """Generate samples from the model."""
    uc = model.get_learned_conditioning([""] * start_code.shape[0]) if scale != 1.0 else None
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(
        S=ddim_steps, conditioning=c, batch_size=n_samples, shape=shape, x_T=start_code, 
        unconditional_guidance_scale=scale, unconditional_conditioning=uc, eta=ddim_eta, 
        verbose_iter=verbose, t_start=t_start, log_every_t=log_every_t or 100, till_T=till_T
    )
    return (samples_ddim, inters) if log_every_t is not None else samples_ddim


def train_inverse(model, sampler, train_data_dir, devices, concept, lr, resolution, repeats, center_crop, train_batch_size, dataloader_num_workers, epochs, noise_scale, ddim_steps, start_guidance, ddim_eta, image_size, models_path):
    """Train token embedding directly using the tokenizer."""
    clip_embedder = model.cond_stage_model
    clip_embedder.tokenizer.add_tokens([concept])
    clip_embedder.transformer.resize_token_embeddings(len(clip_embedder.tokenizer))

    new_token_id = clip_embedder.tokenizer.convert_tokens_to_ids([concept])[0]
    corgi_token_id = clip_embedder.tokenizer.convert_tokens_to_ids(['dog'])[0]
    corgi_emb = clip_embedder.transformer.get_input_embeddings().weight[corgi_token_id].clone()

    original_embedding_matrix = clip_embedder.transformer.get_input_embeddings().weight.clone()
    
    with torch.no_grad():
        clip_embedder.transformer.get_input_embeddings().weight[new_token_id] = corgi_emb.clone()

    embedding_layer = clip_embedder.transformer.get_input_embeddings()
    opt = torch.optim.Adam(embedding_layer.parameters(), lr=lr)

    train_dataset = PreprocessImage(
        data_root=train_data_dir, size=resolution, repeats=repeats, 
        center_crop=center_crop, set="train"
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=dataloader_num_workers
    )    
    
    fixed_start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

    quick_sample_till_t = lambda cond, s, code, t: sample_model(
        model, sampler, cond, image_size, image_size, ddim_steps, s, ddim_eta,
        start_code=code, till_T=t, verbose=False
    )
    
    def decode_and_save_image(model, z, path):
        """Decode latent vector and save image."""
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = rearrange(x, 'b c h w -> b h w c')
        image = Image.fromarray((x[0].cpu().numpy() * 255).astype(np.uint8))
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
        plt.savefig(path)
        plt.close()

    os.makedirs(f'evaluation_folder/textual_inversion/{concept}', exist_ok=True)
    os.makedirs(f'{models_path}/embedding_textual_inversion', exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        for i, batch in enumerate(train_dataloader):
            opt.zero_grad()
            model.train()

            batch_images = batch['pixel_values'].to(device=devices[0])
            encoder_posterior = model.encode_first_stage(batch_images)
            batch_z = model.get_first_stage_encoding(encoder_posterior).detach()

            emb_prompt = model.get_learned_conditioning([f"{concept}"])
            cond = torch.repeat_interleave(emb_prompt, batch_z.shape[0], dim=0)

            t_enc = torch.randint(0, ddim_steps, (1,), device=devices[0]).long()
            t_enc_ddpm = torch.randint(0, 1000, (batch_z.shape[0],), device=devices[0])

            noise = torch.randn_like(batch_z) * noise_scale
            x_noisy = model.q_sample(x_start=batch_z, t=t_enc_ddpm, noise=noise)
            model_output = model.apply_model(x_noisy, t_enc_ddpm, cond)
            
            loss = torch.nn.functional.mse_loss(model_output, noise)
            loss.backward()
            opt.step()

            if i % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')

        with torch.no_grad():
            clip_embedder.transformer.get_input_embeddings().weight[:-1] = original_embedding_matrix[:-1].clone()
            model.eval()

            emb_val = model.get_learned_conditioning([f"A photo of a {concept} perched on a park bench with the Colosseum."])
            z_r_till_T = quick_sample_till_t(emb_val.to(devices[0]), start_guidance, fixed_start_code, int(ddim_steps))
            decode_and_save_image(model, z_r_till_T, path=f'evaluation_folder/textual_inversion/{concept}/gen_{epoch}.png')

            torch.save(clip_embedder.transformer.get_input_embeddings().weight[new_token_id].cpu(), 
                       f'{models_path}/embedding_textual_inversion/emb_{concept}_{epoch}.pt')
    return clip_embedder.transformer.get_input_embeddings().weight[new_token_id].detach()


def load_and_generate_image(model, sampler, devices, concept, image_size, ddim_steps, start_guidance, ddim_eta, embedding_path, prompt, output_path):
    """Load token embedding, generate an image based on the prompt, and save it."""
    saved_embedding = torch.load(embedding_path).to(devices[0])
    clip_embedder = model.cond_stage_model

    if concept not in clip_embedder.tokenizer.get_vocab():
        clip_embedder.tokenizer.add_tokens([concept])
        clip_embedder.transformer.resize_token_embeddings(len(clip_embedder.tokenizer))

    new_token_id = clip_embedder.tokenizer.convert_tokens_to_ids([concept])[0]

    with torch.no_grad():
        clip_embedder.transformer.get_input_embeddings().weight[new_token_id] = saved_embedding

    prompt_embedding = model.get_learned_conditioning([prompt])
    fixed_start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

    z_r_till_T = sample_model(model, sampler, prompt_embedding, image_size, image_size, 
                              ddim_steps, start_guidance, ddim_eta, start_code=fixed_start_code, till_T=int(ddim_steps))

    def decode_and_save_image(model, z, path):
        """Decode latent vector and save image."""
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = rearrange(x, 'b c h w -> b h w c')
        image = Image.fromarray((x[0].cpu().numpy() * 255).astype(np.uint8))
        image.save(path)

    decode_and_save_image(model, z_r_till_T, output_path)
    print(f"Image saved at {output_path}")


def personalized_inference(model, sampler, devices, concept, image_size, ddim_steps, start_guidance, ddim_eta, models_path, prompt, output_dir, ind=0):
    """Generate an image using personalized embedding."""
    emb_path = f'{models_path}/embedding_textual_inversion/emb_{concept}_99.pt'
    learned_emb = torch.load(emb_path).to(devices[0])

    emb = model.get_learned_conditioning([prompt])
    token_name = concept

    if token_name in prompt:
        token_idx = prompt.split().index(token_name)
        emb[0, 4:6, :] = learned_emb[0, 1:3, :]

    fixed_start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

    with torch.no_grad():
        z_r_till_T = sample_model(model, sampler, emb, image_size, image_size, ddim_steps, 
                                  start_guidance, ddim_eta, start_code=fixed_start_code, verbose=False)

        x = model.decode_first_stage(z_r_till_T)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = rearrange(x, 'b c h w -> b h w c')

        image = Image.fromarray((x[0].cpu().numpy() * 255).astype(np.uint8))
        os.makedirs(output_dir, exist_ok=True)
        image.save(f'{output_dir}/personalized_image_{ind}.png')
        print(f"Personalized image saved at: {output_dir}/personalized_image_{ind}.png")


if __name__ == '__main__':
    # Parameters
    concept = "new_concept"
    start_guidance = 3.0
    lr = 1e-5
    config_path = 'configs/stable-diffusion/v1-inference.yaml'
    ckpt_path = 'models/ldm/stable-diffusion-v1/model.ckpt'
    devices = ['cuda:0']
    image_size = 512
    ddim_steps = 50
    ddim_eta = 0.0
    models_path = 'models'
    train_data_dir = 'path/to/train/data'
    resolution = 512
    repeats = 100
    center_crop = False
    train_batch_size = 4
    dataloader_num_workers = 4
    epochs = 100
    noise_scale = 1.0

    # Load the model
    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DPMSolverSampler(model)

    # Train the inverse model
    embedding = train_inverse(model, sampler, train_data_dir, devices, concept, lr, resolution, repeats, center_crop, train_batch_size, dataloader_num_workers, epochs, noise_scale, ddim_steps, start_guidance, ddim_eta, image_size, models_path)

    # Save the learned embedding
    torch.save(embedding, f'{models_path}/embedding_textual_inversion/emb_{concept}.pt')

    # Generate an image
    personalized_inference(model, sampler, devices, concept, image_size, ddim_steps, start_guidance, ddim_eta, models_path, f"A photo of a {concept} perched on a park bench with the Colosseum looming behind.", "generated_images")
