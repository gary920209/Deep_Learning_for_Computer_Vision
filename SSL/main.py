'''
DLCV @ NTUEE 2024 Fall
HW1-1: Self-Supervised Pre-training for Image Classification 
Author         : Gary
Date           : 2024/9/18
Description    : This code is Train full model (backbone + classifier) on office dataset 
'''
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.transforms import ToPILImage
import torchvision.io as tvio
import os
from train import train_model, validate_model
from dataset import OfficeHomeDatasetCV2, OfficeHomeClassifier
    
if __name__ == '__main__':

    # Load the office dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = OfficeHomeDatasetCV2(csv_file='b10901091/dlcv-fall-2024-hw1-gary920209/hw1_data/p1_data/office/train.csv',
                                    root_dir='b10901091/dlcv-fall-2024-hw1-gary920209/hw1_data/p1_data/office/train',
                                    transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = OfficeHomeDatasetCV2(csv_file='b10901091/dlcv-fall-2024-hw1-gary920209/hw1_data/p1_data/office/val.csv',
                                    root_dir='b10901091/dlcv-fall-2024-hw1-gary920209/hw1_data/p1_data/office/val',
                                    transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # Load the pre-trained model
    model = OfficeHomeClassifier()
    pretrained_model_path = 'b10901091/dlcv-fall-2024-hw1-gary920209/hw1_data/p1_data/improved-net.pt'
    pretrained_dict = torch.load(pretrained_model_path)
    model_dict = model.state_dict()
    
    # Filter out unnecessary keys related to the last layer (fc layer)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('fc')}
    
    # Update the model state with the pre-trained weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    num_epochs = 150
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs) 
    validate_model(model, val_loader)
