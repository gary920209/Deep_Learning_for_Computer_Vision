'''
This script does conditional image generation on MNIST + SVHN, using a diffusion model.
Reference: https://github.com/TeaPearce/Conditional_Diffusion_MNIST
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
from PIL import Image
import csv

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):        
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers =[            
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=10, n_datasets=2):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.n_datasets = n_datasets
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes + n_datasets, 2*n_feat)  # Embed both digit and dataset labels
        self.contextembed2 = EmbedFC(n_classes + n_datasets, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, context, dataset_label, time, context_mask):
        print(f"Context shape: {context.shape}")

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # Convert context (digit) and dataset_label to one hot embeddings and combine them
        context = nn.functional.one_hot(context, num_classes=self.n_classes).type(torch.float)
        dataset_label = nn.functional.one_hot(dataset_label, num_classes=self.n_datasets).type(torch.float)
        combined_context = torch.cat((context, dataset_label), dim=-1)  # Combine both conditions

        # Mask out context if context_mask == 1
        context_mask = context_mask[:, None].repeat(1, combined_context.shape[-1])
        context_mask = -1 * (1 - context_mask)  # need to flip 0 <-> 1
        combined_context = combined_context * context_mask  # Apply masking

        # Embed the combined context
        cemb1 = self.contextembed1(combined_context).view(-1, 2*self.n_feat, 1, 1)
        cemb2 = self.contextembed2(combined_context).view(-1, 1*self.n_feat, 1, 1)
        temb1 = self.timeembed1(time).view(-1, 2*self.n_feat, 1, 1)
        temb2 = self.timeembed2(time).view(-1, 1*self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c, dataset_label):
        """
        Forward pass for DDPM. This method is used during training, where 
        random time steps and noise are added to the input image.
        
        x: Input images (batch of real images).
        c: Digit labels (0-9).
        dataset_label: Dataset labels (0 for MNIST-M, 1 for SVHN).
        """
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)
        
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(self.device)

        Unet_predict = self.nn_model(x_t, c, dataset_label, _ts / self.n_T, context_mask)
        
        return self.loss_mse(noise, Unet_predict)

    def sample(self, n_sample, size, device, guide_w=0.0):
        """
        Sample images using DDPM, with optional guidance.
        
        n_sample: Number of samples to generate.
        size: Size of the generated images.
        guide_w: Guidance strength for classifier-free guidance.
        """
        # x_T ~ N(0, 1), sample initial noise
        x_i = torch.randn(n_sample, *size).to(device)

        c_i = torch.arange(0, 10).to(device)
        c_i = c_i.repeat(int(n_sample / c_i.shape[0]))

        dataset_labels = torch.randint(0, 2, (n_sample,)).to(device)  # 0 for MNIST-M, 1 for SVHN

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch for classifier-free guidance
        c_i = c_i.repeat(2)
        dataset_labels = dataset_labels.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.0  

        x_i_store = []
        for i in range(self.n_T, 0, -1):
            print(f"sampling timestep {i}", end="\r")
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            # double batch for guidance
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, dataset_labels, t_is, context_mask)
            eps1 = eps[:n_sample]  # condition
            eps2 = eps[n_sample:]  # no condition
            eps = (1 + guide_w) * eps1 - guide_w * eps2

            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

class ImageDataset(Dataset):
    def __init__(self, mnistm_path, mnistm_csv, svhn_path, svhn_csv, transform):
        self.transform = transform
        self.files = []
        self.labels = []
        self.dataset_labels = []  # to distinguish MNIST-M (0) from SVHN (1)
        
        # Load MNIST-M
        with open(mnistm_csv, "r", newline="") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader)
            for row in reader:
                img_name, label = row
                self.files.append(os.path.join(mnistm_path, img_name))
                self.labels.append(torch.tensor(int(label)))
                self.dataset_labels.append(torch.tensor(0))  # 0 for MNIST-M
        
        # Load SVHN
        with open(svhn_csv, "r", newline="") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader)
            for row in reader:
                img_name, label = row
                self.files.append(os.path.join(svhn_path, img_name))
                self.labels.append(torch.tensor(int(label)))
                self.dataset_labels.append(torch.tensor(1))  # 1 for SVHN

    def __getitem__(self, idx):
        data = Image.open(self.files[idx])
        data = self.transform(data)
        return data, self.labels[idx], self.dataset_labels[idx]  # return both digit and dataset label

    def __len__(self):
        return len(self.files)



if __name__ == "__main__":
    # hardcoding these here
    n_epoch = 100
    batch_size = 256
    n_T = 400  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = 10
    n_feat = 128  
    lrate = 1e-4
    save_model = True
    save_dir = "b10901091/dlcv-fall-2024-hw2-gary920209/p1/output/"
    ws_test = [2.0]  # strength of generative guidance

    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.1,
    )
    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor()]
    ) 
    train_mnistm_dir = "b10901091/dlcv-fall-2024-hw2-gary920209/hw2_data/digits/mnistm/data"
    train_mnistm_dir_csv = "b10901091/dlcv-fall-2024-hw2-gary920209/hw2_data/digits/mnistm/train.csv"
    train_svhn_dir = "b10901091/dlcv-fall-2024-hw2-gary920209/hw2_data/digits/svhn/data"
    train_svhn_dir_csv = "b10901091/dlcv-fall-2024-hw2-gary920209/hw2_data/digits/svhn/train.csv"
    
    dataset = ImageDataset(
        mnistm_path=train_mnistm_dir,
        mnistm_csv=train_mnistm_dir_csv,
        svhn_path=train_svhn_dir,
        svhn_csv=train_svhn_dir_csv,
        transform=tf,
    )


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for epoch in range(n_epoch):
        print(f"epoch {epoch}")
        ddpm.train()

        optim.param_groups[0]["lr"] = lrate * (1 - epoch / n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c, dataset_label in pbar:  # Include dataset_label in the loop
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            dataset_label = dataset_label.to(device)
            loss = ddpm(x, c, dataset_label)  # Pass dataset_label to the model
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        # ddpm.eval()  # Switch to evaluation mode
        # with torch.no_grad():  # Disable gradient computation for sampling
        #     n_sample = 4 * n_classes  # Number of samples to generate
        #     dataset_labels = torch.tensor([0, 1])  # 0 for MNIST-M, 1 for SVHN (you can adjust this depending on the use case)
            
        #     for w_i, w in enumerate(ws_test):
        #         for dataset_label in dataset_labels:  # Loop over the datasets (MNIST-M and SVHN)
        #             print(f"Generating for dataset: {dataset_label}")
                    
        #             # Generate n_sample images for the current dataset label
        #             # Create dataset label tensor (repeated for all samples)
        #             dataset_label_tensor = torch.full((n_sample,), dataset_label).to(device)

        #             # Sample from the model with conditional labels for digits (0-9) and dataset label
        #             x_gen, x_gen_store = ddpm.sample(
        #                 n_sample, (3, 28, 28), device, guide_w=w
        #             )
                    
        #             # Generate corresponding real images from the dataset for comparison
        #             # You can modify this part based on your dataset
        #             x_real = torch.Tensor(x_gen.shape).to(device)
        #             for k in range(n_classes):
        #                 for j in range(int(n_sample / n_classes)):
        #                     try:
        #                         idx = torch.squeeze((c == k).nonzero())[j]
        #                     except:
        #                         idx = 0
        #                     x_real[k + (j * n_classes)] = x[idx]

        #             # Concatenate generated and real images for comparison
        #             x_all = torch.cat([x_gen, x_real])

        #             # Create a grid and save the images
        #             grid = make_grid(x_all * -1 + 1, nrow=10)
        #             save_image(grid, save_dir + f"image_epoch{epoch}_w{w}_dataset{dataset_label}.png")
        #             print(f"Saved image at {save_dir + f'image_epoch{epoch}_w{w}_dataset{dataset_label}.png'}")
        if save_model and epoch == int(n_epoch - 1) or epoch % 10 == 0:
            torch.save(ddpm.state_dict(), save_dir + f"model_{epoch}.pth")
            print("saved model at " + save_dir + f"model_{epoch}.pth")

