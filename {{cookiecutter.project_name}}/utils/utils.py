from typing import List, Tuple
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, 1)
        trainer.logger[1].experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                            for x, pred, y in zip(val_imgs, preds, self.val_labels)],
            "global_step": trainer.global_step
            })

def collate_fn(batch: List[torch.Tensor]) -> Tuple[Tuple[torch.Tensor]]:
    """[summary]
    Args:
        batch (List[torch.Tensor]): [description]
    Returns:
        Tuple[Tuple[torch.Tensor]]: [description]
    """
    return tuple(zip(*batch))

def warmup_lr_scheduler(optimizer: torch.optim, warmup_iters: int,
                        warmup_factor: float) -> torch.optim:
    """[summary]
    Args:
        optimizer (torch.optim): [description]
        warmup_iters (int): [description]
        warmup_factor (float): [description]
    Returns:
        torch.optim: [description]
    """
    def fun(iter_num: int) -> float:
        if iter_num >= warmup_iters:
            return 1
        alpha = float(iter_num) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fun)

def mkdir(path: str):
    """[summary]
    Args:
        path (str): [description]
    """
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise