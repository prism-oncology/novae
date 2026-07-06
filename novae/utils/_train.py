from collections.abc import Iterator

import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import optim
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def train(
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    accelerator: str,
    max_epochs: int = 50,
    patience: int = 3,
    min_delta: float = 0,
    callbacks: list[Callback] | None = None,
    logger: Logger | list[Logger] | bool = False,
    **trainer_kwargs: int,
):
    """Internal function to train a LightningModule with early stopping."""

    early_stopping = EarlyStopping(
        monitor="train/loss_epoch",
        min_delta=min_delta,
        patience=patience,
        check_on_train_epoch_end=True,
    )
    callbacks = [early_stopping] + (callbacks or [])
    enable_checkpointing = any(isinstance(c, ModelCheckpoint) for c in callbacks)

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=enable_checkpointing,
        **trainer_kwargs,
    )
    trainer.fit(model, datamodule=datamodule)


def configure_optimizers(
    parameters: Iterator[Parameter],
    lr: float,
    use_scheduler: bool,
    max_epochs: int,
    warmup_epochs: int = 2,
    start_factor: float = 0.1,
    end_factor: float = 0.01,
) -> OptimizerLRScheduler:
    optimizer = optim.Adam(parameters, lr=lr)

    if not use_scheduler:
        return optimizer

    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=lr * end_factor)

    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1}}
