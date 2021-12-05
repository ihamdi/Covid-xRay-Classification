from typing import Any, List, Tuple

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.modules.models import Models


class CovidModel(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: str = "Densenet121",
        pretrained: bool = False,
        num_classes: int = 2,
        drop_rate: float = 0,
        optimizer: str = "Adam",
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        momentum: float = 0,
        eps: float = 1e-08,
        alpha: float = 0.99,
        dampening: float = 0,
        betas: Tuple[float, float] = (0.9, 0.999),
        centered: bool = False,
        nesterov: bool = False,
        amsgrad: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        self.model = Models(hparams=self.hparams)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass
    
    def configure_optimizers(self):
        if self.hparams.optimizer == "RMSprop":
            optimizer = torch.optim.RMSprop(params=self.parameters(),
                                            lr=self.hparams.lr,
                                            weight_decay=self.hparams.weight_decay,
                                            momentum=self.hparams.momentum, 
                                            eps=self.hparams.eps, 
                                            alpha=self.hparams.alpha, 
                                            centered=self.hparams.centered)
        elif self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(params=self.parameters(), 
                                        lr=self.hparams.lr, 
                                        weight_decay=self.hparams.weight_decay, 
                                        momentum=self.hparams.momentum, 
                                        dampening=self.hparams.dampening, 
                                        nesterov=self.hparams.nesterov)
        else:
            optimizer = torch.optim.Adam(params=self.parameters(), 
                                        lr=self.hparams.lr, 
                                        weight_decay=self.hparams.weight_decay, 
                                        betas=self.hparams.betas, 
                                        eps=self.hparams.eps,  
                                        amsgrad=self.hparams.amsgrad)  
        return optimizer

