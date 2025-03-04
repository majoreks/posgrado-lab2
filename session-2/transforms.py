import torch.nn as nn
import torchvision.transforms as transforms

data_transforms = transforms.Compose([
    transforms.ToTensor()
])

__all__ = ['data_transforms']