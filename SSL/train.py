'''
This file contains the main function to train and validate the model.
'''
import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import OfficeHomeDatasetCV2


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50):
    """Main function to train and validate the model with early stopping."""
    model.train()
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%')
        scheduler.step()

        # Validate and save the model if it achieves better accuracy
        val_acc = validate_model(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')  
            print(f'Saved best model with accuracy: {best_val_acc:.2f}%')

def validate_model(model, val_loader):
    """Evaluates the model on the validation set and returns the accuracy."""
    
    model.eval()  
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()  

    with torch.no_grad(): 
        for images, labels in val_loader:  
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f'Validation Accuracy: {val_acc:.2f}%')
    return val_acc




