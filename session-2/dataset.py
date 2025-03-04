import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, images_dir, labels_path, transform=None):
        self.images_dir = images_dir
        self.info_df = pd.read_csv(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        suite_id, sample_id, code, _, _ = self.info_df[idx, :-1]
        img = Image.open(self.__bulild_image_path(suite_id, sample_id, code))
        if self.transform:
            img = self.transform(img)
        
        return img, code-1

    def __bulild_image_path(self, suite_id, sample_id, code):
        return os.path.join(self.images_dir + "input_" + suite_id + "_" + sample_id + "_" + code + ".jpg")

