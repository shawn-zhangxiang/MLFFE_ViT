import torch
import torchvision
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]  

img_size= 224
data_transforms =transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
test_data = datasets.ImageFolder(test_dir, transform=data_transforms)
val_data = datasets.ImageFolder(val_dir, transform=data_transforms)

NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32
  
train_dataloader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
test_dataloader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
val_dataloader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

train_dataloader, test_dataloader, val_dataloader 
