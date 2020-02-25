import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# from test_tube import Experiment
from pytorch_lightning.logging import TestTubeLogger


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
    gpus : [type], optional
        [description], by default None
    max_epochs : int, optional
        [description], by default 5
    log_dir : str, optional
        [description], by default "/home/zwl/tmp/tb"
    early_stop_callback : bool, optional
        [description], by default False

    Returns
    -------
    pl.LightningModule
        [description]
    """
    # most basic trainer, uses good defaults
    # tensorboard --logdir /Users/zwl/tmp/tb

    # put model on the right device
    # device = torch_device()
    # print(f"Move model to device: {device}, gpus = {gpus}")
    # model = model.to(torch_device())

    # see pl docs for details here. latest version as of 2019.09 needs to use
    # backend for multi-gpu training
    if (isinstance(gpus, list) and len(gpus) > 1) or (
        isinstance(gpus, int) and gpus > 1
    ):
        backend = "ddp"
    else:
        backend = None

    print(f"GPUs = {gpus}, backend = {backend}")
    print(f"Tensorbord cmd: tensorboard --logdir {log_dir}")
    print(f"distributed_backend = {backend}")

    # exp = Experiment(save_dir=log_dir)
    if log_dir is None:
        tt_logger = TestTubeLogger(
            save_dir=log_dir, name="default", debug=False, create_git_tag=False
        )
    else:
        tt_logger = None
    trainer = Trainer(
        max_epochs=max_epochs,
        max_nb_epochs=max_epochs,
        show_progress_bar=False,
        logger=tt_logger,
        default_save_path=log_dir,
        # experiment=exp,
        gpus=gpus,
        distributed_backend=backend,
        early_stop_callback=early_stop_callback,
    )
    trainer.fit(model)

    return model
