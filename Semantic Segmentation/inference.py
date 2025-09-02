import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as trns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import imageio.v2 as imageio
from torchvision.models.segmentation import FCN_ResNet101_Weights
from torchvision import models


class SegmentationDataset(Dataset):
    def __init__(self, file_path, img_transform=None):
        self.path = file_path
        self.data = []
        self.imgfile = sorted(
            [img for img in os.listdir(self.path) if img.endswith("sat.jpg")]
        )

        for img in self.imgfile:
            self.data.append(Image.open(os.path.join(self.path, img)).copy())
        self.transform = img_transform

    def __getitem__(self, idx):
        data = self.data[idx]

        if self.transform:
            data = self.transform(data)

        return data, self.imgfile[idx]

    def __len__(self):
        return len(self.imgfile)

class FCN_ResNet101_Model(nn.Module):
    def __init__(self):
        super(FCN_ResNet101_Model, self).__init__()
        self.model = models.segmentation.fcn_resnet101(weights=FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        # Modify classifier[4] as per the training script
        self.model.classifier[4] = nn.Conv2d(512, 7, kernel_size=1, stride=1)  # Match with training
    


    def forward(self, x):
        output = self.model(x)
        return output['out']


def pred2image(batch_preds, batch_names, out_path):
    for pred, name in zip(batch_preds, batch_names):
        pred = pred.detach().cpu().numpy()
        pred_img = np.zeros((512, 512, 3), dtype=np.uint8)
        pred_img[np.where(pred == 0)] = [0, 255, 255]
        pred_img[np.where(pred == 1)] = [255, 255, 0]
        pred_img[np.where(pred == 2)] = [255, 0, 255]
        pred_img[np.where(pred == 3)] = [0, 255, 0]
        pred_img[np.where(pred == 4)] = [0, 0, 255]
        pred_img[np.where(pred == 5)] = [255, 255, 255]
        pred_img[np.where(pred == 6)] = [0, 0, 0]
        imageio.imwrite(
            os.path.join(out_path, name.replace("sat.jpg", "mask.png")), pred_img
        )

img_dir_val = sys.argv[1]
output_folder = sys.argv[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_val = trns.Compose(
    [
        trns.Resize([512, 512]),
        trns.ToTensor(),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_data = SegmentationDataset(img_dir_val, img_transform=transform_val)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=16)

# Define Model
model = FCN_ResNet101_Model()
model = model.to(device)
model.load_state_dict(torch.load("p2_bestmodel.pt"))

val_loss, val_pred_list, val_mask_list = [], [], []

model.eval()
with torch.no_grad():
    for i, (imgs, filenames) in enumerate(val_loader):
        imgs = imgs.to(device)
        output = model(imgs)
        pred = output.cpu().argmax(dim=1)
        val_pred_list.append(pred)
        pred2image(pred, filenames, output_folder)

