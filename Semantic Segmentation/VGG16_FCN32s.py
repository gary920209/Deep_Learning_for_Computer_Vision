'''
DLCV @ NTUEE 2024 Fall
HW1-2: Semantic Segmentation
Author  : Gary
Date    : 2024/9/21
'''
import os
import sys
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
import torchvision as tv
import torchvision.models as models
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torchvision.models import VGG16_Weights
import imageio
from mean_iou_evaluate import read_masks, mean_iou_score

# VCG16 + FCN32s model
class VGG16_FCN32s(nn.Module):
    def __init__(self, num_classes):
        super(VGG16_FCN32s, self).__init__()
        self.features = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )
        self.upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False)
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        
        return x
    
# Load dataset
class SegmentationDataset(Dataset):
    """Preprocessed Image Dataset."""
    def __init__(
            self, 
            data_dir,
            data_list,
            data_mask,
            data_transform=None,
            data_mask_transform=None
                 ): 
            self.data_dir = data_dir
            self.data_list = data_list
            self.data_mask = data_mask
            self.data_transform = data_transform
            self.data_mask_transform = data_mask_transform
            self.random = 0
            self.isprocessed = []
    def __len__(self):
        self.datalen = len(self.data_list)
        return len(self.data_list)
    def __getitem__(self, idx):
        img_name = self.data_list[idx]
        img_path = os.path.join(self.data_dir, img_name)
        img = tv.io.read_image(img_path)
        mask = self.data_mask[idx]
        if self.data_transform:
            self.random += 10
            random.seed(self.random)
            to_pil = ToPILImage()
            img_pil = to_pil(img)
            img = self.data_transform(img_pil)
        return img, mask





# Train model
def train_model(
    model,
    loss_func,
    optimizer,
    num_epochs,
    train_loader,
    val_loader,
):
    """Main function to train and validate the model with early stopping."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model = model.to(device)
    model.train()

    training_loss, training_iou_list = [], []
    val_loss_list, val_iou_list = [], []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")                
        # Training 
        train_iou, val_iou = 0, 0
        train_predicted_list, train_mask_list = [], []

        for i, (inputs, masks) in enumerate(train_loader):
            print(f"Batch {i+1}/{len(train_loader)} processed", end="\r")

            inputs = inputs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()  

            outputs = model(inputs)  # Forward propagation
            loss = loss_func(outputs, masks.long())  # Compute loss  
            loss.backward()  
            optimizer.step()  

            _, predicted = torch.max(outputs, 1)
            train_predicted_list.append(predicted.cpu().numpy())
            train_mask_list.append(masks.cpu().numpy())
        

        # Stack the predictions and labels along the batch dimension
        train_predicted_list = np.concatenate(train_predicted_list, axis=0)  # Shape: (total_samples, 512, 512)
        train_mask_list = np.concatenate(train_mask_list, axis=0)  # Shape: (total_samples, 512, 512)

        # Compute mean IoU score
        train_iou = mean_iou_score(
            np.array(train_predicted_list), np.array(train_mask_list)
        )
        training_loss.append(loss.item())
        training_iou_list.append(train_iou)
        scheduler.step() 

        # Clear memory
        torch.cuda.empty_cache()


        # Validation 
        model.eval()  
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_predicted_list, val_mask_list = [], []
        with torch.no_grad(): 
            for i, (inputs, masks) in enumerate(val_loader):
                inputs = inputs.to(device)
                masks = masks.to(device)

                outputs = model(inputs)
                loss = loss_func(outputs, masks.long())
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_predicted_list.append(predicted.cpu().numpy())
                val_mask_list.append(masks.cpu().numpy())

        # Stack the predictions and labels along the batch dimension
        val_predicted_list = np.concatenate(val_predicted_list, axis=0)  # Shape: (total_samples, 512, 512)
        val_mask_list = np.concatenate(val_mask_list, axis=0)  # Shape: (total_samples, 512, 512)

        # Compute mean IoU score
        val_iou = mean_iou_score(
            np.array(val_predicted_list), np.array(val_mask_list)
        )
        val_loss_list.append(val_loss)
        val_iou_list.append(val_iou)

        # Save the best model
        if val_iou > best_val_acc:
            best_val_acc = val_iou
            torch.save(model.state_dict(), "b10901091/dlcv-fall-2024-hw1-gary920209/p2_best_model.pt")
        best_val_acc = max(val_iou, best_val_acc)

        model.train()  
        print(
            "Train Epoch: {}/{} Traing_Loss: {:.3f} Traing_iou {:.3f} ,Test_Loss: {:.3f},Test_iou:{:.3f}".format(
                epoch + 1,
                num_epochs,
                loss.item(),
                train_iou,
                val_loss,
                val_iou,
            )
        )
        
    return training_loss, training_iou_list, val_loss_list, val_iou_list

if __name__ == '__main__':

    # parameters
    num_classes = 7
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.0001
    train_dir = "b10901091/dlcv-fall-2024-hw1-gary920209/hw1_data/p2_data/train"
    val_dir = "b10901091/dlcv-fall-2024-hw1-gary920209/hw1_data/p2_data/validation"

    # load dataset, mask end in .png, image end in .jpg
    train_list = [file for file in os.listdir(train_dir) if file.endswith(".jpg")]
    train_list.sort()
    
    val_list = [file for file in os.listdir(val_dir) if file.endswith(".jpg")]
    val_list.sort()

    # data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # no data augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load dataloader
    train_dataset = SegmentationDataset(train_dir, train_list, read_masks(train_dir), train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = SegmentationDataset(val_dir, val_list, read_masks(val_dir), val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # train model and save the best model
    model = VGG16_FCN32s(num_classes)
    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    training_loss, training_acc_list, val_loss_list, val_acc_list = train_model(
        model, loss_func, optimizer, num_epochs, train_loader, val_loader
    )
