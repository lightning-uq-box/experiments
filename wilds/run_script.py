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
from datamodule import WILDSPovertyDataModule
from resnet import ResNet18
from lightning.pytorch.profilers import SimpleProfiler

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
        monitor="train_loss",  # https://github.com/Feuermagier/Beyond_Deep_Ensembles/blob/b805d6f9de0bd2e6139237827497a2cb387de11c/experiments/poverty/poverty.py#L152 model is just saved 
        mode=mode,
        every_n_epochs=1,
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    profiler = SimpleProfiler(dirpath=config["experiment"]["save_dir"], filename="profiler")

    if "SWAG" in config.uq_method["_target_"]:
        callbacks = None
    else:
        callbacks = [checkpoint_callback, lr_monitor_callback]

    return instantiate(
        config.trainer,
        default_root_dir=config["experiment"]["save_dir"],
        callbacks=callbacks,
        logger=loggers,
        profiler=profiler,
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

        model.pred_file_name = "preds_ood_test.csv"
        trainer.test(model, datamodule=datamodule)
    else:
        model = instantiate(full_config.uq_method)
        trainer.fit(model, datamodule)
        model.pred_file_name = "preds_ood_test.csv"
        trainer.test(ckpt_path="best", datamodule=datamodule)


    # val dataset results
    model.pred_file_name = "preds_val.csv"
    val_loader = datamodule.val_dataloader()
    val_loader.shuffle = False

    try:
        trainer.test(ckpt_path="best", dataloaders=val_loader)
    except:
        trainer.test(model, dataloaders=val_loader)

    # iid test dataset results
    model.pred_file_name = "preds_iid_test.csv"
    iid_test_loader = datamodule.iid_test_dataloader()
    iid_test_loader.shuffle = False

    try:
        trainer.test(ckpt_path="best", dataloaders=iid_test_loader)
    except:
        trainer.test(model, dataloaders=iid_test_loader)

    # save configuration file
    with open(
        os.path.join(full_config["experiment"]["save_dir"], "config.yaml"), "w"
    ) as f:
        OmegaConf.save(config=full_config, f=f)

    print("FINISHED EXPERIMENT", flush=True)
