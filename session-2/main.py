import pandas as pd
import sklearn.model_selection
import torch

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from dataset import MyDataset
from model import MyModel
from utils import AverageMeter, accuracy, save_model
from transforms import data_transforms
import os
import argparse
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import sklearn

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", help="path to data dir", type=str, default='data/')
parser.add_argument("--info-fname", help="file name of csv with data info", type=str, default='chinese_mnist.csv')
parser.add_argument("--n-epochs", help="number of epochs to train", type=int, default=5)
parser.add_argument("--batch-size", help="batch size", type=int, default=100)
parser.add_argument("--lr", help="learning rate", type=float, default=0.1)
args = parser.parse_args()

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_single_epoch(model, criterion, optimizer, dataloader, train_loss, train_acc):
    train_loss.reset()
    train_acc.reset()
    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_ = model(x)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), n=len(y))
        train_acc.update(accuracy(y, y_), n=len(y))

def eval_single_epoch(model, criterion, dataloader, val_loss, val_acc):
    val_loss.reset()
    val_acc.reset()
    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        y_ = model(x)
        loss = criterion(y_, y)
        loss.backward()

        val_loss.update(loss.item(), n=len(y))
        val_acc.update(accuracy(y, y_), n=len(y))


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
    print(train_df.head())
    print(valid_df.head())

    my_model = MyModel().to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size']) 
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size']) 
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'])

    criterion = F.nll_loss
    optimizer = optim.Adam(my_model.parameters(), lr=config['lr'])
    
    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()

    for _ in tqdm(range(config["epochs"]), 'epoch'):
        my_model.train()
        train_single_epoch(my_model, criterion, optimizer, train_dataloader, train_loss, train_accuracy)
        train_losses.append(train_loss.avg)
        train_accuracies.append(train_accuracy.avg)
        
        my_model.eval()
        eval_single_epoch(my_model, criterion, valid_dataloader, val_loss, val_accuracy)
        val_losses.append(val_loss.avg)
        val_accuracies.append(val_accuracy.avg)
   
    epochs = range(len(train_accuracies))

    plt.plot(epochs, train_accuracies, 'b', label='Training acc')
    plt.plot(epochs, val_accuracies, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'acc.png'), dpi=400)       

    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'loss.png'), dpi=400)

    return my_model


if __name__ == "__main__":

    config = {
        "epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    train_model(config)
