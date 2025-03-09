import os
import torch
from device import device
import numpy as np
import random

proj_path = '/home/szymon/code/posgrado/lab2/'

def accuracy(labels, outputs):
    preds = outputs.argmax(-1)
    acc = (preds == labels.view_as(preds)).cpu().float().detach().numpy().mean()
    return acc

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()

    return model

def config_to_string(config):
    return f"epochs_{config['epochs']}_batch_{config['batch_size']}_lr_{config['lr']:.5f}_mlp_{config['mlp_width']}_wd_{config['weight_decay']:.8f}"

def create_best_txt(directory):
    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, "best.txt")

    with open(file_path, "w") as file:
        file.write("This is the best model.\n") 

    print(f"File 'best.txt' created at: {file_path}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)