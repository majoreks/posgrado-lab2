import pandas as pd
import sklearn.model_selection
import torch

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from dataset import MyDataset
from model import MyModel
from utils import AverageMeter, accuracy, save_model
from transforms import data_transforms
import os
import argparse
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import sklearn
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", help="path to data dir", type=str, default='data/')
parser.add_argument("--info-fname", help="file name of csv with data info", type=str, default='chinese_mnist.csv')
parser.add_argument("--n-epochs", help="number of epochs to train", type=int, default=10)
parser.add_argument("--batch-size", help="batch size", type=int, default=100)
parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
parser.add_argument("--mlp-width", help="width of mlp", type=int, default=512)
args = parser.parse_args()

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_single_epoch(model, criterion, optimizer, dataloader):
    epoch_loss = []
    epoch_acc = []

    model.train()
    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_ = model(x)
        # print(y_)
        # print(y)
        # raise Exception("XD")
        loss = criterion(y_, y)
        # print(y_)
        # print(f'loss | train | {loss}')
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        epoch_acc.append(accuracy(y, y_))

    return epoch_loss, epoch_acc

def eval_single_epoch(model, criterion, dataloader):
    epoch_loss = []
    epoch_acc = []

    model.eval()
    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        y_ = model(x)
        loss = criterion(y_, y)
        # print(f'loss | valid | {loss}')

        epoch_loss.append(loss.item())
        epoch_acc.append(accuracy(y, y_))
    
    return epoch_loss, epoch_acc


def train_model(config):
    info = pd.read_csv(os.path.join(args.data_dir, args.info_fname))
    train_valid_df, test_df = sklearn.model_selection.train_test_split(info, test_size=0.3, stratify=info['code'])
    train_df, valid_df = sklearn.model_selection.train_test_split(train_valid_df, test_size=0.3, stratify=train_valid_df['code'])

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_dataset = MyDataset(args.data_dir, train_df, transform=data_transforms)
    valid_dataset = MyDataset(args.data_dir, valid_df, transform=data_transforms)
    test_dataset = MyDataset(args.data_dir, test_df, transform=data_transforms)

    my_model = MyModel(config['mlp_width']).to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2) 
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size']) 
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'])

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(my_model.parameters(), lr=config['lr'])
    
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
        "mlp_width": args.mlp_width
    }
    train_model(config)
