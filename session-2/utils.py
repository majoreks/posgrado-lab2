import torch

def accuracy(labels, outputs):
    preds = outputs.argmax(-1)
    acc = (preds == labels.view_as(preds)).cpu().float().detach().numpy().mean()
    return acc

def save_model(model, path):
    torch.save(model.state_dict(), path)

def config_to_string(config):
    return f"epochs_{config['epochs']}_batch_{config['batch_size']}_lr_{config['lr']:.5f}_mlp_{config['mlp_width']}_wd_{config['weight_decay']:.8f}"