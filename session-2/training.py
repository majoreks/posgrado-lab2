from torch import device
from tqdm import tqdm
from utils import accuracy
from device import device

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