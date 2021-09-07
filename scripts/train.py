import os
import random
import string
import yaml
import fire
from rich import print, pretty, inspect, traceback
pretty.install()
traceback.install()

import torch
import numpy as np
import pytorch_lightning as pl

import wandb
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from project_x.utils import utils

from project_x.pl import LitDataset, LitModel

def train(config_path: str = "./config/experiments.yml", job: str = "main"):
    # seed
    pl.seed_everything(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(device)
    # args
    print(config_path)
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    data_path = config["DATA_PATH"]
    output_path = config["OUTPUT_PATH"]
    project = config["PROJECT"]
    hyperparameters = config[job]

    # hypterparamters
    batch_size = int(hyperparameters["BATCH_SIZE"])
    learning_rate = float(hyperparameters["LEARNING_RATE"])
    max_epoch = int(hyperparameters["MAX_EPOCH"])
    num_workers = int(hyperparameters["NUM_WORKERS"])
    gpus = int(hyperparameters["GPUS"])
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")
    # https://github.com/tchaton/lightning-geometric/blob/master/examples/utils/loggers.py
    # output directory
    random_str = ''.join(random.choices(
        string.ascii_uppercase + string.digits, k=5))
    experiment = f"{random_str}_{job}_{batch_size}_{learning_rate}_{max_epoch}"
    output_dir = f"{output_path}/checkpoints/{experiment}"
    logger = f"{output_path}/logs/{experiment}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # loggers
    wandb.login()
    tb_logger = loggers.TensorBoardLogger(save_dir=logger)
    wandb_logger = loggers.WandbLogger(
        project=project, log_model="all", name=experiment)
    # data
    dataset = LitDataset(data_path, batch_size, num_workers)
    dataset.prepare_data()
    dataset.setup()
    val_samples = next(iter(dataset.val_dataloader()))
    val_imgs, val_labels = val_samples[0], val_samples[1]
    val_imgs.shape, val_labels.shape
    # model
    litmodel = LitModel(dataset.size(), dataset.num_classes, len(dataset.train_dataloader()), learning_rate)
    wandb_logger.watch(litmodel)
    _loggers = [tb_logger, wandb_logger]
    _callbacks = [ModelCheckpoint(dirpath=output_dir), utils.ImagePredictionLogger(val_samples),LearningRateMonitor(logging_interval='step')]
    # trainer
    trainer = pl.Trainer(
        logger=_loggers,
        callbacks=_callbacks,
        gpus=gpus,
        max_epochs=max_epoch,
        progress_bar_refresh_rate=20
    )
    trainer.fit(litmodel, dataset)
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")


if __name__ == '__main__':
    fire.Fire(train)
