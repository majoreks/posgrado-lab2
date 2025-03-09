import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MyDataset(Dataset):
    def __init__(self, images_dir, info_df, transform=None):
        self.images_dir = images_dir
        self.info_df = info_df
        self.transform = transform

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        suite_id, sample_id, code, _, _ = self.info_df.loc[idx, :]
        img = Image.open(self.__bulild_image_path(suite_id, sample_id, code))
        if self.transform:
            img = self.transform(img)
        return img, code-1

    def __bulild_image_path(self, suite_id, sample_id, code):
        return os.path.join('/home/szymon/code/posgrado/lab2/', self.images_dir, "data", f"input_{suite_id}_{sample_id}_{code}.jpg")

