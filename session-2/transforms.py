import torch.nn as nn
import torchvision.transforms as transforms

data_transforms = transforms.Compose([
    transforms.PILToTensor()
])

__all__ = ['data_transforms']