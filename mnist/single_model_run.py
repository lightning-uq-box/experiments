import torch
from torch import Tensor
import os
import torch.nn as nn
from functools import partial
from datamodule import MNISTDataModule
from lightning_uq_box.uq_methods import (
    DeterministicClassification,
    MCDropoutClassification,
)
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from datetime import datetime


# https://github.com/runame/laplace-redux/blob/main/baselines/vanilla/models/lenet.py
class LeNet(nn.Module):
    """LeNet model for MNIST."""

    def __init__(self, num_classes=10, drop_rate=0.0):
        """Initialize LeNet.

        Args:
            num_classes: Number of classes. Default is 10 for MNIST.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.net(x)


def create_experiment_dir(experiment_dir: str) -> str:
    """Create experiment directory.

    Args:
        experiment_dir: experiment directory

    Returns:
        experiment directory
    """
    os.makedirs(experiment_dir, exist_ok=True)
    exp_dir_name = f"mnist" f"_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S-%f')}"
    exp_dir_path = os.path.join(experiment_dir, exp_dir_name)
    os.makedirs(exp_dir_path)
    return exp_dir_path


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    dataset_path = "/p/project/hai_uqmethodbox/nils/projects/experiments/mnist/data"

    experiment_dir = (
        "/p/project/hai_uqmethodbox/nils/projects/experiments/mnist/experiments"
    )

    # create a random dir with time in it
    experiment_dir = create_experiment_dir(experiment_dir)

    datamodule = MNISTDataModule(
        root=dataset_path,
        batch_size=128,
        num_workers=4,
    )
    datamodule.setup("fit")

    # decides which "best" model checkpoint to save
    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_dir,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        every_n_epochs=1,
    )

    model = DeterministicClassification(
        model=LeNet(),
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=partial(torch.optim.Adam, lr=1e-3),
    )

    trainer = Trainer(
        max_epochs=50,
        default_root_dir=experiment_dir,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule)

    # evaluate on the in distribution dataset under this csv path in the experiment_dir
    model.pred_file_name = "preds_in_dist.csv"
    trainer.test(ckpt_path="best", dataloaders=datamodule.test_dataloader())

    # evaluate on FMNIST, EMNIST, KMNIST for which numbers are reported in the laplace paper
    for ood_dataset in ["FMNIST", "EMNIST", "KMNIST"]:
        model.pred_file_name = f"preds_ood_{ood_dataset}.csv"
        ood_loader = datamodule.ood_dataloader(ood_dataset)
        ood_loader.shuffle = False
        trainer.test(ckpt_path="best", dataloaders=ood_loader)

    # also commonly evaluated on, but usually no hard numbers reported
    # evaluate on rotation angles for rotated MNIST
    # for angle in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]:
    #     model.pred_file_name = f"preds_ood_{angle}.csv"
    #     ood_loader = datamodule.rotated_dataloader(angle=angle)
    #     ood_loader.shuffle = False
    #     trainer.test(ckpt_path="best", dataloaders=ood_loader)

    print("FINISHED EXPERIMENT", flush=True)
