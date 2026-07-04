import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping


def torch_device() -> str:
    """Automatically figure out pytorch device.

    Returns
    -------
    str
        'cuda' or 'cpu'
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def device(model: nn.Module):
    """Returns the device a given model sits in.

    Parameters
    ----------
    model : nn.Module
        [description]

    Returns
    -------
    [type]
        Compute device the model sits in.
    """
    return next(model.parameters()).device


def pl_train(
    model: pl.LightningModule,
    gpus=None,
    max_epochs: int = 5,
    log_dir="/home/zwl/tmp/tb",
    early_stop_callback=False,
) -> pl.LightningModule:
    """ pytorch lightning training helper methods.

    Parameters
    ----------
    model : pl.LightningModule
        [description]
    gpus : int or list, optional
        number of gpus (int) or specific gpu ids (list), by default None
    max_epochs : int, optional
        [description], by default 5
    log_dir : str, optional
        directory for tensorboard logs, by default "/home/zwl/tmp/tb"
    early_stop_callback : bool, optional
        if True, add an EarlyStopping callback monitoring val_loss,
        by default False

    Returns
    -------
    pl.LightningModule
        [description]
    """
    # map the legacy `gpus` argument onto the modern accelerator/devices API.
    backend = None
    if gpus is None:
        accelerator, devices = "cpu", None
    else:
        accelerator = "gpu"
        if isinstance(gpus, list):
            devices = len(gpus)
        elif isinstance(gpus, int):
            devices = gpus
        else:
            devices = 1
        # multi-gpu training uses ddp
        if isinstance(devices, int) and devices > 1:
            backend = "ddp"

    print(f"GPUs = {gpus}, accelerator = {accelerator}, devices = {devices}")
    print(f"Tensorboard cmd: tensorboard --logdir {log_dir}")
    print(f"strategy = {backend}")

    # only create a logger when a log directory is provided
    if log_dir is not None:
        logger = TensorBoardLogger(save_dir=log_dir, name="default")
    else:
        logger = False

    callbacks = []
    if early_stop_callback:
        callbacks.append(EarlyStopping(monitor="val_loss"))

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=backend,
        logger=logger,
        default_root_dir=log_dir,
        callbacks=callbacks,
    )
    trainer.fit(model)

    return model
