import pandas as pd
import sklearn.model_selection
import torch

from torch.utils.data import DataLoader
from dataset import MyDataset
from transforms import data_transforms
import os
import argparse
import sklearn
import numpy as np
import random
from training import train_model

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", help="path to data dir", type=str, default='data/')
parser.add_argument("--info-fname", help="file name of csv with data info", type=str, default='chinese_mnist.csv')
parser.add_argument("--n-epochs", help="number of epochs to train", type=int, default=15)
parser.add_argument("--batch-size", help="batch size", type=int, default=128)
parser.add_argument("--lr", help="learning rate", type=float, default=0.0005)
parser.add_argument("--mlp-width", help="width of mlp", type=int, default=256)
parser.add_argument("--weight-decay", help="weight decay value", type=float, default=1e-4)
args = parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

if __name__ == "__main__":

    config = {
        "epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "mlp_width": args.mlp_width,
        "weight_decay": args.weight_decay
    }

    info = pd.read_csv(os.path.join(args.data_dir, args.info_fname))
    train_valid_df, test_df = sklearn.model_selection.train_test_split(info, test_size=0.3, stratify=info['code'])
    train_df, valid_df = sklearn.model_selection.train_test_split(train_valid_df, test_size=0.3, stratify=train_valid_df['code'])

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_dataset = MyDataset(args.data_dir, train_df, transform=data_transforms)
    valid_dataset = MyDataset(args.data_dir, valid_df, transform=data_transforms)

    test_dataset = MyDataset(args.data_dir, test_df, transform=data_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'])

    train_model(config, train_dataset, valid_dataset)
