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
import torchmetrics

import wandb
from pytorch_lightning import loggers
import torch.nn.functional as F

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from project_x.utils import utils
from project_x.models.simple_classifier import Model

class LitDataset(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (3, 32, 32)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.num_classes = 10

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset_full = CIFAR10(self.data_dir, train=True,
                                   transform=self.transform)
            self.dataset_train, self.dataset_val = random_split(
                dataset_full, [45000, 5000])

        if stage == "test" or stage is None:
            self.dataset_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)


class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, data_len, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.model = Model(input_shape, num_classes)
        self.learning_rate = learning_rate
        self.loss = F.nll_loss
        self.metric = torchmetrics.Accuracy()
        self.data_len = data_len
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{
                'params': [p for p in self.parameters()],
                'name': 'my_parameter_group_name'
            }],
            lr=self.learning_rate
        )
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, self.data_len - 1)
        def fun(iter_num: int) -> float:
            if iter_num >= warmup_iters:
                return 1
            alpha = float(iter_num) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, fun),
            'name': 'learning_rate'
        }
        return [optimizer], [lr_scheduler]
    def forward(self,x):
        logits = self.model(x)
        return logits
    def _step(self, batch, stage=None):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.metric(preds, y)
        return {'loss': loss, 'acc': acc}
    def _epoch_end(self, outputs, stage=None):
        if stage:
            acc = np.mean([float(x['acc']) for x in outputs])
            loss = np.mean([float(x['loss']) for x in outputs])
            self.logger.log_metrics(
                {f"{stage}/acc": acc}, self.current_epoch + 1)
            self.logger.log_metrics(
                {f"{stage}/loss": loss}, self.current_epoch + 1)
    
    # on_train_start()
    # for epoch in epochs:
    #   on_train_epoch_start()
    #   for batch in train_dataloader():
    #       on_train_batch_start()
    #       training_step()
    #       ...
    #       on_train_batch_end()
    #       on_validation_epoch_start()
    #
    #       for batch in val_dataloader():
    #           on_validation_batch_start()
    #           validation_step()
    #           on_validation_batch_end()
    #       on_validation_epoch_end()
    #
    #   on_train_epoch_end()
    # on_train_end
    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch)

    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs)


def train(config_path: str = "./config/experiments.yml", job: str = "main", test: bool=False):
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
