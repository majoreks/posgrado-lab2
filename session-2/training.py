import os
from matplotlib import pyplot as plt
import numpy as np
from torch import device
from tqdm import tqdm
from utils import accuracy, config_to_string
from device import device
from model import MyModel
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

def train_single_epoch(model, criterion, optimizer, dataloader):
    epoch_loss = []
    epoch_acc = []

    model.train()
    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_ = model(x)
        loss = criterion(y_, y)
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

        epoch_loss.append(loss.item())
        epoch_acc.append(accuracy(y, y_))
    
    return epoch_loss, epoch_acc

def train_model(config, train_dataset, val_dataset):
    train_run_path = os.path.join(os.path.dirname(__file__), 'train_runs', config_to_string(config))
    os.makedirs(train_run_path)

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
    plt.savefig(os.path.join(train_run_path, 'acc.png'), dpi=400)       

    plt.figure()
    plt.plot(stats_len, train_losses, 'b', label='Training loss')
    plt.plot(stats_len, val_losses, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(train_run_path, 'loss.png'), dpi=400)

    return my_model