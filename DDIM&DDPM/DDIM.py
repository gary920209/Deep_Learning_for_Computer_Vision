'''
This script does DDIM.
'''
import os
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import MNIST, SVHN
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import csv
from UNet import UNet
from utils import beta_scheduler

class DDIM(nn.Module):
    def __init__(self, nn_model, timesteps=1000, beta_schedule=beta_scheduler()):
        super(DDIM, self).__init__()        
        self.model = nn_model
        self.timesteps = timesteps
        self.betas = beta_schedule

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))

    # get the param of given timestep t
    def get_params(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    # use ddim to sample
    @torch.no_grad()
    def sample(self, batch_size=10, ddim_timesteps=50, ddim_eta=0.0, clip_denoised=True,):
        
        # data preparation
        filenames = [f"{i:02d}.pt" for i in range(0, batch_size)]
        tensors = [
            torch.load(os.path.join("b10901091/dlcv-fall-2024-hw2-gary920209/hw2_data/face/noise", filename))
            for filename in filenames
        ]

        # same interval for all timesteps
        c = self.timesteps // ddim_timesteps
        # a sequence of timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        ddim_timestep_seq += 1
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_img = torch.cat(tensors, dim=0)


        for i in tqdm (reversed(range(ddim_timesteps)), desc="DDIM sampling", total = ddim_timesteps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,),ddim_timestep_prev_seq[i],device=device,dtype=torch.long,)

            # get the params of timestep t
            alpha_cumprod_t = self.get_params(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self.get_params(self.alphas_cumprod, prev_t, sample_img.shape)

            # predict the noise
            pred_noise = self.model(sample_img, t)

            # get the predicted image
            pred_x0 = (sample_img - torch.sqrt((1.0 - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1.0, max=1.0)
            # compute the posterior mean and variance
            sigmas_t = ddim_eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            pred_dir_xt = (torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise)
            x_prev = (torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img) )

            sample_img = x_prev

        return sample_img.cpu()


def out_img(img_num=10, eta=0):
    n_T = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Unet_pt = 'b10901091/dlcv-fall-2024-hw2-gary920209/hw2_data/face/UNet.pt'
    save_dir = 'b10901091/dlcv-fall-2024-hw2-gary920209/p2/output/'
    model = UNet().to(device)
    model.load_state_dict(torch.load(Unet_pt))
    ddim = DDIM(model, timesteps=1000, beta_schedule=beta_scheduler())

    # sample images
    with torch.no_grad():
        x_gen = ddim.sample(batch_size=img_num, ddim_eta=eta)
        for i in range(len(x_gen)):
            img = x_gen[i]
            min_val = torch.min(img)
            max_val = torch.max(img)

            # Min-Max Normalization
            normalized_x_gen = (img - min_val) / (max_val - min_val)
            save_image(normalized_x_gen, save_dir + f"{i:02d}.png")


if __name__ == "__main__":
    out_img(img_num=10, eta=0)
