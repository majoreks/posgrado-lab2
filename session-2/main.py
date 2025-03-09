import pandas as pd
import sklearn.model_selection
import torch

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataset import MyDataset
from model import MyModel
from transforms import data_transforms
import os
import argparse
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import random
from training import train_single_epoch, eval_single_epoch
from device import device

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

def train_model(config, train_dataset, val_dataset):
    my_model = MyModel(config['mlp_width']).to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8) 
    valid_dataloader = DataLoader(val_dataset, batch_size=config['batch_size']) 

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(my_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []

    for _ in tqdm(range(config["epochs"]), 'epoch'):
        train_epoch_loss, train_epoch_acc = train_single_epoch(my_model, criterion, optimizer, train_dataloader)
        
        train_losses.append(np.asarray(train_epoch_loss).mean())
        train_accuracies.append(np.asarray(train_epoch_acc).mean())
        
        val_epoch_loss, val_epoch_acc = eval_single_epoch(my_model, criterion, valid_dataloader)
        val_losses.append(np.asarray(val_epoch_loss).mean())
        val_accuracies.append(np.asarray(val_epoch_acc).mean())
   
    stats_len = range(len(train_accuracies))

    plt.plot(stats_len, train_accuracies, 'b', label='Training acc')
    plt.plot(stats_len, val_accuracies, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'acc.png'), dpi=400)       

    plt.figure()
    plt.plot(stats_len, train_losses, 'b', label='Training loss')
    plt.plot(stats_len, val_losses, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'loss.png'), dpi=400)

    return my_model


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
