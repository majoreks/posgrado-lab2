import argparse
import os
import pandas as pd
import sklearn
from ray import tune
import ray
from dataset import MyDataset
from model import MyModel
from transforms import data_transforms
from torch.utils.data import DataLoader
from training import train_model

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", help="path to data dir", type=str, default='data/')
parser.add_argument("--info-fname", help="file name of csv with data info", type=str, default='chinese_mnist.csv')
args = parser.parse_args()


def test_model(config, model, test_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'])

if __name__ == "__main__":
    info = pd.read_csv(os.path.join(args.data_dir, args.info_fname))
    train_valid_df, test_df = sklearn.model_selection.train_test_split(info, test_size=0.3, stratify=info['code'])
    train_df, valid_df = sklearn.model_selection.train_test_split(train_valid_df, test_size=0.3, stratify=train_valid_df['code'])

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_dataset = MyDataset(args.data_dir, train_df, transform=data_transforms)
    valid_dataset = MyDataset(args.data_dir, valid_df, transform=data_transforms)

    test_dataset = MyDataset(args.data_dir, test_df, transform=data_transforms)

    ray.init(configure_logging=False)
    tuner = tune.with_parameters(train_model, train_dataset=train_dataset, val_dataset=valid_dataset, print_progress=False)
    analysis = tune.run(
        tuner,
        metric="val_loss",
        mode="min",
        num_samples=1,
        resources_per_trial={"gpu": 1, "cpu": 4},
        config={
            "epochs": tune.choice([10, 20, 30, 40]),
            "lr": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([16, 32, 64]),
            "mlp_width": tune.choice([128, 256, 512]),
            "weight_decay": tune.loguniform(1e-6, 1e-2),
        })

    print("Best hyperparameters found were: ", analysis.best_config)
    # print(test_model(...))
