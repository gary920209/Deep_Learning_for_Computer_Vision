'''
This file contains the code to load the OfficeHome dataset using the PyTorch Dataset class.
'''
import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import models
from torchvision.transforms import ToPILImage
import torchvision.io as tvio
import os


# load the office dataset
class OfficeHomeDatasetCV2(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get image name and path
        img_name = self.data_frame.iloc[idx, 1]  
        img_path = os.path.join(self.root_dir, img_name)
        # Load image using torchvision.io.read_image (returns a tensor)
        image = tvio.read_image(img_path)
        # Get label for the image
        label = self.data_frame.iloc[idx, 2]


        # Apply transformations if provided
        if self.transform:
            to_pil = ToPILImage()
            img_pil = to_pil(image)
            image = self.transform(img_pil)

        return image, label
class OfficeHomeClassifier(nn.Module):
    def __init__(self):
        super(OfficeHomeClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=False)  
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 65)  

    def forward(self, x):
        return self.resnet(x)
