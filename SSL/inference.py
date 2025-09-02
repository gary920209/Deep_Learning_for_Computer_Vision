import os
import csv
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.io as tvio
import torchvision.transforms as trns
from torch.optim import Adam, SGD
from torchvision import models
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader

# Dataset class to load images based on the CSV file
class FinetuneDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = trns.Compose([
            trns.Resize((128, 128)),      # Resize to 128x128
            trns.ToTensor(),
            trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx, 1]  # filename column
        img_path = os.path.join(self.img_dir, img_name)
        image = tvio.read_image(img_path)
        image_id = self.data_frame.iloc[idx, 0]  # id column

        if self.transform:
            to_pil = ToPILImage()
            img_pil = to_pil(image)
            image = self.transform(img_pil)

        return image, img_name, image_id

class OfficeHomeClassifier(nn.Module):
    def __init__(self):
        super(OfficeHomeClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        # Modify the fully connected layer to match the output size (65 classes in your case)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 65),
        )

    def forward(self, x):
        return self.resnet(x)


# Command line arguments
csv_file = sys.argv[1]  # Path to CSV file
img_dir = sys.argv[2]   # Path to the image folder
output_csv = sys.argv[3]  # Path to output CSV file

# Now create the model and load the saved weights
model = OfficeHomeClassifier()
model.load_state_dict(torch.load('p1_best_model.pth'))

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Load the dataset
dataset = FinetuneDataset(csv_file, img_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Perform inference
model.eval()
preds = []
image_infos = []

for img, img_name, img_id in dataloader:
    img = img.to(device)
    with torch.no_grad():
        pred = model(img)
    preds.append(pred)
    for name, id in zip(img_name, img_id):
        image_infos.append((id, name))

# Convert predictions and image information to proper format
np_image_infos = np.array(image_infos, dtype=object)
np_preds = torch.cat(preds, dim=0).cpu().numpy()

# Write predictions to CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'filename', 'label'])
    for image_info, pred in zip(np_image_infos, np_preds):
        # Convert image_info[0] (id) to an integer
        image_id = image_info[0].item() if isinstance(image_info[0], torch.Tensor) else image_info[0]
        writer.writerow([image_id, image_info[1], np.argmax(pred)])
