from typing import Optional
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

from {{cookiecutter.project_name}}.pl import LitDataset, LitModel

def test(config_path: str = "./config/experiments.yml", job: str = "main",checkpoint_path:str = "outputs/checkpoints/E41JR_main_256_0.001_10/epoch=9-step=1759-v9.ckpt"):
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
    logger = f"{output_path}/logs/{experiment}"

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
    litmodel = LitModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=None,
    )
    wandb_logger.watch(litmodel)
    _loggers = [tb_logger, wandb_logger]
    # trainer
    trainer = pl.Trainer(
        logger=_loggers,
        gpus=gpus,
        max_epochs=max_epoch,
        progress_bar_refresh_rate=20
    )
    trainer.test(litmodel)


if __name__ == '__main__':
    fire.Fire(test)
