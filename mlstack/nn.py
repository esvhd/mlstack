import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


def save_model(model: nn.Module, filepath: str):
    # save state dict for future inference
    torch.save({"model_state_dict": model.state_dict()}, filepath)


def load_model(model: nn.Module, filepath: str) -> nn.Module:
    # Load state dict into a model for inference
    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict.get("model_state_dict"))
    model.eval()
    return model


class FCNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        n_layers: int,
        activation: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.bias = bias

        self.layers = []
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(input_dim, input_dim, bias=bias))
            # activation
            self.layers.append(nn.ReLU())
        # last layer
        self.layers.append(nn.Linear(input_dim, out_dim, bias=bias))

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)


class Dataset2D(Dataset):
    def __init__(self, x, y):
        """Make dataset from 2D tabular data.

        Parameters
        ----------
        x : [type]
            [description]
        y : [type]
            [description]
        """
        super().__init__()
        assert x.shape[0] == y.shape[0] or x.shape[0] == len(y)
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


class PLNet(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        train_set: Dataset,
        valid_set: Dataset,
        # test_set: Dataset = None,
        lr=3e-4,
        batch_size: int = 32,
    ):
        super().__init__()

        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.train_set = train_set
        self.valid_set = valid_set

    def forward(self, x):
        return self.model(x)

    # TODO: how to use Transformer for regression problems.
    # or convert y into a classification problem.
    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        # don't reduce loss here, this is so that in multi-gpu training
        # we don't get a scaler loss warning
        # for multi-gpu training, better to run reduction outside
        # https://discuss.pytorch.org/t/mse-loss-gpu-warning/24782/2
        # https://discuss.pytorch.org/t/how-to-fix-gathering-dim-0-warning-in-multi-gpu-dataparallel-setting/41733
        # reduction is then handled in Trainer
        # loss = F.l1_loss(y_hat, y.float(), reduction="none")

        # single GPU
        loss = F.l1_loss(y_hat, y.float(), reduction="mean")
        # print(f"y_hat.shape = {y_hat.shape}, y.shape = {y.shape}, loss={loss}")
        return {"loss": loss}
        # return {'loss': F.l1_loss(y_hat, y.float())}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        # loss = F.l1_loss(y_hat, y.float(), reduction="none")
        # singla GPU
        loss = F.l1_loss(y_hat, y.float(), reduction="mean")
        # print(f"Valid step info: y_hat.shape = {y_hat.shape}, y.shape = {y.shape}, loss={loss: .6f}")
        return {"val_loss": loss}
        # return {'val_loss': F.l1_loss(y_hat, y.float()).mean(axis=1)}

    def validation_end(self, outputs):
        # OPTIONAL
        # single GPU
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        print(f"Valid end info: Avg Loss: {avg_loss.item():.6f}")

        # avg_loss = torch.cat(
        #     [x["val_loss"].mean(axis=1, keepdim=True) for x in outputs], axis=0
        # ).mean()

        # below did not work for tensorflow
        # return {"avg_val_loss": avg_loss}

        # below worked for tensorflow 2.0..
        # val_loss = {"avg_val_loss": avg_loss.item()}
        tensorbord_log = {"val_loss": avg_loss.item()}
        return {"log": tensorbord_log, "avg_val_loss": avg_loss.item()}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=False
        )

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(
            self.valid_set, batch_size=self.batch_size, shuffle=False
        )
