import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os

def get_loader(ds_path='./data', batch_size=128, image_size=32):

    transform = transforms.Compose([transforms.Resize([image_size, image_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])

    ds = datasets.ImageFolder(ds_path, transform=transform)

    ds_loader = torch.utils.data.DataLoader(dataset=ds,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              drop_last=True)
    return ds_loader
