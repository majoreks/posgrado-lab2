import torch

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model
from transforms import data_transforms
import os
import argparse
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", help="path to data dir", type=str, default='data/')
parser.add_argument("--info-fname", help="file name of csv with data info", type=str, default='chinese_mnist.csv')
parser.add_argument("--n-epochs", help="number of epochs to train", type=int, default=10)
parser.add_argument("--batch-size", help="batch size", type=int, default=100)
parser.add_argument("--lr", help="learning rate", type=float, default=0.1)
args = parser.parse_args()

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_single_epoch(model, criterion, optimizer, dataloader):
    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# def eval_single_epoch(...):
#     pass


def train_model(config):
    my_dataset = MyDataset(args.data_dir, os.path.join(args.data_dir, args.info_fname), transform=data_transforms)
    my_model = MyModel().to(device)

    dataloader = DataLoader(my_dataset, batch_size=config['batch_size'])

    criterion = F.nll_loss
    optimizer = optim.Adam(my_model.parameters(), lr=config['lr'])
    
    x, y = next(iter(dataloader))
    for epoch in tqdm(range(config["epochs"]), 'epoch'):
        train_single_epoch(my_model, criterion, optimizer, dataloader)
    #     train_single_epoch(...)
    #     eval_single_epoch(...)

    return my_model


if __name__ == "__main__":

    config = {
        "epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    train_model(config)
