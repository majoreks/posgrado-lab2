import torch

from torch.utils.data import DataLoader
from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model
from transforms import data_transforms
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", help="path to data dir", type=str, default='data/')
parser.add_argument("--info-fname", help="file name of csv with data info", type=str, default='chinese_mnist.csv')
args = parser.parse_args()

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# def train_single_epoch(...):
#     pass


# def eval_single_epoch(...):
#     pass


def train_model(config):
    my_dataset = MyDataset(args.data_dir, os.path.join(args.data_dir, args.info_fname), transform=data_transforms)
    my_model = MyModel().to(device)

    dataloader = DataLoader(my_dataset)
    
    x, y = next(iter(dataloader))
    print(x, y)
    # for epoch in range(config["epochs"]):
    #     train_single_epoch(...)
    #     eval_single_epoch(...)

    return my_model


if __name__ == "__main__":

    config = {
        "hyperparam_1": 1,
        "hyperparam_2": 2,
    }
    train_model(config)
