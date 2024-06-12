"""Tropical Cyclone Datamodule."""

"""Tropical Cyclone Wind Speed Estimation."""

from typing import Any, Dict

import kornia.augmentation as K
import torch
import pandas as pd
from torch import Tensor
import numpy as np
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datamodules.utils import group_shuffle_split
from sklearn.model_selection import train_test_split
from torchgeo.transforms import AugmentationSequential
from torchvision.transforms import Resize

from tropical_cyclone_ds import TropicalCycloneSequence


def combined_collate_fn(batch):
    """Combined collate fn."""
    # resize = Resize(224, antialias=False)
    # combine things from "input" and "image" keys
    inputs = [x["input"] for x in batch]

    # combine things from "target" and "label" keys
    targets = [x["target"] for x in batch]

    return {
        "input": torch.stack(inputs),
        "target": torch.stack(targets),
        "storm_ids": [x.get("storm_id") for x in batch],
        "indices": [x.get("index") for x in batch],
        "wind_speeds": [int(x["wind_speed"]) for x in batch],
    }


class CombinedTCDataModule(NonGeoDataModule):
    """Data Module combining the NASA Cyclone and Digital Typhoon datasets."""

    valid_tasks = ["regression", "classification"]

    def __init__(
        self,
        task: str,
        batch_size: int,
        num_workers: int,
        tc_args: Any,
        dgtl_typhoon_args: Any,
    ) -> None:
        super().__init__(TropicalCycloneSequence, batch_size, num_workers, **tc_args)

        assert (
            task in self.valid_tasks
        ), f"invalid task '{task}', please choose one of {self.valid_tasks}"
        self.task = task

        self.tc_args = tc_args
        self.tc_args["task"] = self.task
        self.dgtl_typhoon_args = dgtl_typhoon_args
        self.dgtl_typhoon_args["task"] = self.task

        self.aug = AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=(90, 91), p=0.5),
            K.RandomRotation(degrees=(270, 271), p=0.5),
            data_keys=["input"],
        )

        self.tc_dataset = TropicalCycloneSequence(split="train", **self.tc_args)
        self.tc_targets = self.tc_dataset.sequence_df["wind_speed"].astype(float).values
        self.dgtl_typhoon_dataset = MyDigitalTyphoonAnalysis(**self.dgtl_typhoon_args)

        sequences = list(enumerate(self.dgtl_typhoon_dataset.sample_sequences))
        train_indices, test_indices = group_shuffle_split(
            [x[1]["id"] for x in sequences], train_size=0.8, random_state=0
        )

        # based on train_indices select target values
        self.dgtl_typhoon_targets = self.dgtl_typhoon_dataset.aux_df.iloc[train_indices]

        # Create a DataFrame from sequences
        train_sequences = [sequences[i] for i in train_indices]
        train_sequence_df = pd.DataFrame(
            [
                (x[1]["id"], seq_id)
                for x in train_sequences
                for seq_id in x[1]["seq_id"]
            ],
            columns=["id", "seq_id"],
        )
        train_sequence_df = pd.merge(
            train_sequence_df, self.dgtl_typhoon_dataset.aux_df, on=["id", "seq_id"]
        )

        all_targets = np.concatenate(
            [self.tc_targets, train_sequence_df["wind"].values]
        )

        self.target_mean = all_targets.mean()
        self.target_std = all_targets.std()

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """

        dgtl_sequences = list(enumerate(self.dgtl_typhoon_dataset.sample_sequences))
        dgtl_train_indices, test_indices = group_shuffle_split(
            [x[1]["id"] for x in dgtl_sequences], train_size=0.8, random_state=0
        )

        if stage in ["fit", "validate"]:
            # setup Tropical cyclone dataset
            train_indices, val_indices = group_shuffle_split(
                self.tc_dataset.sequence_df.storm_id, test_size=0.20, random_state=0
            )
            validation_indices, calibration_indices = train_test_split(
                val_indices, test_size=0.20, random_state=0
            )

            self.tc_train_dataset = Subset(self.tc_dataset, train_indices)
            self.tc_val_dataset = Subset(self.tc_dataset, validation_indices)
            self.tc_calibration_dataset = Subset(self.tc_dataset, calibration_indices)

            # setup Digital Typhoon dataset

            # first train and validation

            index_mapping_train = {
                new_index: original_index
                for new_index, original_index in enumerate(dgtl_train_indices)
            }
            train_sequences = [
                self.dgtl_typhoon_dataset.sample_sequences[i]
                for i in dgtl_train_indices
            ]
            train_sequences = list(enumerate(train_sequences))
            train_indices, val_indices = group_shuffle_split(
                [x[1]["id"] for x in train_sequences], train_size=0.8, random_state=0
            )
            train_indices = [index_mapping_train[i] for i in train_indices]
            val_indices = [index_mapping_train[i] for i in val_indices]

            # then validation and calibration split
            index_mapping_val = {
                new_index: original_index
                for new_index, original_index in enumerate(val_indices)
            }
            val_sequences = [
                self.dgtl_typhoon_dataset.sample_sequences[i] for i in val_indices
            ]
            val_sequences = list(enumerate(val_sequences))
            val_indices, calibration_indices = group_shuffle_split(
                [x[1]["id"] for x in val_sequences], train_size=0.8, random_state=0
            )
            val_indices = [index_mapping_val[i] for i in val_indices]
            calibration_indices = [index_mapping_val[i] for i in calibration_indices]

            # Create train val subset dataset
            self.dgtl_train_dataset = Subset(self.dgtl_typhoon_dataset, train_indices)
            self.dgtl_val_dataset = Subset(self.dgtl_typhoon_dataset, val_indices)
            self.dgtl_calibration_dataset = Subset(
                self.dgtl_typhoon_dataset, calibration_indices
            )

            # concat datasetssequences
            self.train_dataset = ConcatDataset(
                [self.tc_train_dataset, self.dgtl_train_dataset]
            )
            self.val_dataset = ConcatDataset(
                [self.tc_val_dataset, self.dgtl_val_dataset]
            )
            self.calibration_dataset = ConcatDataset(
                [self.tc_calibration_dataset, self.dgtl_calibration_dataset]
            )

        if stage in ["test"]:
            self.tc_test_dataset = TropicalCycloneSequence(split="test", **self.tc_args)
            self.dgtl_test_dataset = Subset(self.dgtl_typhoon_dataset, test_indices)

            self.test_dataset = ConcatDataset(
                [self.tc_test_dataset, self.dgtl_test_dataset]
            )

    def calibration_dataloader(self) -> torch.utils.data.DataLoader:
        """Return a dataloader for the calibration dataset."""
        return DataLoader(
            self.calibration_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=combined_collate_fn,
        )

    def on_after_batch_transfer(
        self, batch: Dict[str, Tensor], dataloader_idx: int
    ) -> Dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.trainer.training:
            input = self.aug({"input": batch["input"].float()})["input"]
        else:
            input = batch["input"].float()

        if self.task == "regression":
            new_batch = {
                "input": input,
                "target": (batch["target"].float() - self.target_mean)
                / self.target_std,
            }
        else:
            new_batch = {
                "input": input,
                "target": batch["target"].long(),
            }

        # add back all other keys
        for key, value in batch.items():
            if key not in ["input", "target"]:
                new_batch[key] = value
        return new_batch


class TropicalCycloneSequenceDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the NASA Cyclone dataset.

    Implements 80/20 train/val splits based on hurricane storm ids.
    See :func:`setup` for more details.
    """

    input_mean = torch.Tensor([0.28154722, 0.28071895, 0.27990073])
    input_std = torch.Tensor([0.23435517, 0.23392765, 0.23351675])
    # target_mean = torch.Tensor([50.54925])
    # target_std = torch.Tensor([26.836512])

    valid_tasks = ["regression", "classification"]

    def __init__(
        self,
        task: str = "regression",
        batch_size: int = 64,
        num_workers: int = 0,
        img_size: int = 224,
        **kwargs: Any,
    ) -> None:
        """Initialize a new TropicalCycloneDataModule instance.

        Args:
            task: One of "regression" or "classification"
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~tropical_cyclone_uq.datasets.TropicalCyclone`.
        """
        super().__init__(TropicalCycloneSequence, batch_size, num_workers, **kwargs)

        assert (
            task in self.valid_tasks
        ), f"invalid task '{task}', please choose one of {self.valid_tasks}"
        self.task = task
        self.img_size = img_size

        self.dataset = TropicalCycloneSequence(
            split="train", img_size=img_size, **self.kwargs
        )
        # mean and std can change based on setup because min wind speed is a variable

        self.target_mean = self.dataset.target_mean
        self.target_std = self.dataset.target_std

        self.train_aug = AugmentationSequential(
            K.Resize(img_size),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=(90, 91), p=0.5),
            K.RandomRotation(degrees=(270, 271), p=0.5),
            data_keys=["input"],
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            self.dataset = TropicalCycloneSequence(
                split="train", img_size=self.img_size, task=self.task, **self.kwargs
            )
            train_indices, val_indices = group_shuffle_split(
                self.dataset.sequence_df.storm_id, test_size=0.20, random_state=0
            )
            validation_indices, calibration_indices = train_test_split(
                val_indices, test_size=0.20, random_state=0
            )

            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, validation_indices)
            self.calibration_dataset = Subset(self.dataset, calibration_indices)

        if stage in ["test"]:
            self.test_dataset = TropicalCycloneSequence(
                split="test", img_size=self.img_size, task=self.task, **self.kwargs
            )

    def calibration_dataloader(self) -> torch.utils.data.DataLoader:
        """Return a dataloader for the calibration dataset."""
        return DataLoader(
            self.calibration_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def on_after_batch_transfer(
        self, batch: Dict[str, Tensor], dataloader_idx: int
    ) -> Dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.trainer.training:
            input = self.train_aug({"input": batch["input"].float()})["input"]
        else:
            input = batch["input"].float()

        if self.task == "regression":
            new_batch = {
                "input": input,
                "target": (batch["target"].float() - self.target_mean)
                / self.target_std,
            }
        else:
            new_batch = {
                "input": input,
                "target": batch["target"].long(),
            }

        # add back all other keys
        for key, value in batch.items():
            if key not in ["input", "target"]:
                new_batch[key] = value
        return new_batch
