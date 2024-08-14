import os
from datetime import datetime
from typing import Any
import torch

from hydra.utils import instantiate
from hydra.errors import InstantiationException
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torchvision.transforms import Resize
from omegaconf.errors import ConfigAttributeError
from lightning.pytorch.loggers import CSVLogger, WandbLogger  # noqa: F401
from omegaconf import OmegaConf
from datamodule import MNISTDataModule
from lenet import LeNet


def create_experiment_dir(config: dict[str, Any]) -> str:
    """Create experiment directory.

    Args:
        config: config file

    Returns:
        config with updated save_dir
    """
    os.makedirs(config["experiment"]["exp_dir"], exist_ok=True)
    exp_dir_name = (
        f"{config['experiment']['experiment_name']}"
        f"_{config['uq_method']['_target_'].split('.')[-1]}"
        f"_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S-%f')}"
    )
    config["experiment"]["experiment_name"] = exp_dir_name
    exp_dir_path = os.path.join(config["experiment"]["exp_dir"], exp_dir_name)
    os.makedirs(exp_dir_path)
    config["experiment"]["save_dir"] = exp_dir_path
    config["trainer"]["default_root_dir"] = exp_dir_path
    return config


def generate_trainer(config: dict[str, Any]) -> Trainer:
    """Generate a pytorch lightning trainer."""
    loggers = [
        CSVLogger(config["experiment"]["save_dir"], name="csv_logs"),
        WandbLogger(
            name=config["experiment"]["experiment_name"],
            save_dir=config["experiment"]["save_dir"],
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            resume="allow",
            mode="offline",
        ),
    ]
    mode = "min"

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["experiment"]["save_dir"],
        save_top_k=1,
        monitor="val_loss",
        mode=mode,
        every_n_epochs=1,
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    if "SWAG" in config.uq_method["_target_"]:
        callbacks = None
    else:
        callbacks = [checkpoint_callback, lr_monitor_callback]

    return instantiate(
        config.trainer,
        default_root_dir=config["experiment"]["save_dir"],
        callbacks=callbacks,
        logger=loggers,
    )


post_hoc_methods = [
    "SWAG",
    "Laplace",
    "ConformalQR",
    "CARD",
    "DeepEnsemble",
    "TempScaling",
    "RAPS",
    "TTA",
]


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    command_line_conf = OmegaConf.from_cli()

    model_conf = OmegaConf.load(command_line_conf.model_config)
    data_conf = OmegaConf.load(command_line_conf.data_config)
    trainer_conf = OmegaConf.load(command_line_conf.trainer_config)

    full_config = OmegaConf.merge(data_conf, trainer_conf, model_conf)
    full_config = create_experiment_dir(full_config)

    datamodule = instantiate(full_config.datamodule)

    trainer = generate_trainer(full_config)

    if any(method in full_config.uq_method._target_ for method in post_hoc_methods):
        # post hoc methods just load a checkpoint
        if (
            "SWAG" in full_config.uq_method["_target_"]
            or "CARD" in full_config.uq_method["_target_"]
        ):
            model = instantiate(full_config.uq_method)
            trainer.fit(model, datamodule=datamodule)
        elif "Laplace" in full_config.uq_method["_target_"]:
            model = instantiate(full_config.uq_method)
            trainer.test(model, datamodule)
        elif "DeepEnsemble" in full_config.uq_method["_target_"]:
            ensemble_members = [
                {
                    "base_model": instantiate(full_config.ensemble_members),
                    "ckpt_path": path,
                }
                for path in full_config.uq_method.ensemble_members
            ]
            model = instantiate(
                full_config.uq_method, ensemble_members=ensemble_members
            )
        elif (
            "ConformalQR" in full_config.uq_method["_target_"]
            or "RAPS" in full_config.uq_method["_target_"]
        ):
            datamodule.setup("fit")
            model = instantiate(full_config.uq_method)
            trainer.validate(model, dataloaders=calib_loader)
        else:
            model = instantiate(full_config.uq_method)
            trainer.validate(model, datamodule=datamodule)

    else:
        model = instantiate(full_config.uq_method)
        trainer.fit(model, datamodule)

    # train dataset results
    model.pred_file_name = "preds_train.csv"
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    train_loader.shuffle = False

    try:
        trainer.test(ckpt_path="best", dataloaders=train_loader)
    except:
        trainer.test(model, dataloaders=train_loader)

    # evaluate on rotation angles
    for angle in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]:
        model.pred_file_name = f"preds_ood_{angle}.csv"
        ood_loader = datamodule.rotated_dataloader(angle=angle)
        ood_loader.shuffle = False
        try:
            trainer.test(ckpt_path="best", dataloaders=ood_loader)
        except:
            trainer.test(model, dataloaders=ood_loader)

    # evaluate on FMNIST, EMNIST, KMNIST
    for ood_dataset in ["FMNIST", "EMNIST", "KMNIST"]:
        model.pred_file_name = f"preds_ood_{ood_dataset}.csv"
        ood_loader = datamodule.ood_dataloader(ood_dataset)
        ood_loader.shuffle = False
        try:
            trainer.test(ckpt_path="best", dataloaders=ood_loader)
        except:
            trainer.test(model, dataloaders=ood_loader)

    # save configuration file
    with open(
        os.path.join(full_config["experiment"]["save_dir"], "config.yaml"), "w"
    ) as f:
        OmegaConf.save(config=full_config, f=f)

    print("FINISHED EXPERIMENT", flush=True)
