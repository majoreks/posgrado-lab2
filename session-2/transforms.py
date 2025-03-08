import torchvision.transforms as transforms

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

__all__ = ['data_transforms']