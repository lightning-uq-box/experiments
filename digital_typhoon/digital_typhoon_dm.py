"""Digital Typhoon DataModule."""

from typing import Any
from torchgeo.datamodules import NonGeoDataModule
from digital_typhoon_ds import MyDigitalTyphoonAnalysis

from torchgeo.transforms import AugmentationSequential
from torchgeo.datamodules.utils import group_shuffle_split
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
import kornia.augmentation as K
from torch.utils.data import DataLoader, Subset


class MyDigitalTyphoonAnalysisDataModule(NonGeoDataModule):
    valid_split_types = ["time", "typhoon_id"]

    min_input_clamp = 170.0
    max_input_clamp = 300.0

    def __init__(
        self,
        split_by: str = "time",
        batch_size: int = 64,
        num_workers: int = 0,
        img_size: int = 224,
        **kwargs: Any,
    ) -> None:
        """Initialize a new DigitalTyphoonAnalysisDataModule instance.

        Args:
            split_by: Either 'time' or 'typhoon_id', which decides how to split
                the dataset for train, val, test
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.DigitalTyphoonAnalysis`.

        """
        super().__init__(MyDigitalTyphoonAnalysis, batch_size, num_workers, **kwargs)

        assert (
            split_by in self.valid_split_types
        ), f"Please choose from {self.valid_split_types}"
        self.split_by = split_by

        self.train_aug = AugmentationSequential(
            K.Resize(img_size),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=(90, 91), p=0.5),
            K.RandomRotation(degrees=(270, 271), p=0.5),
            data_keys=["input"],
        )

        self.dataset = MyDigitalTyphoonAnalysis(img_size=img_size, **kwargs)

        sequences = list(enumerate(self.dataset.sample_sequences))
        train_indices, test_indices = group_shuffle_split(
            [x[1]["id"] for x in sequences], train_size=0.8, random_state=0
        )

        # based on train_indices select target values
        self.dgtl_typhoon_targets = self.dataset.aux_df.iloc[train_indices][
            "wind"
        ].values
        self.target_mean = self.dgtl_typhoon_targets.mean()
        self.target_std = self.dgtl_typhoon_targets.std()

    def split_dataset(self, dataset: MyDigitalTyphoonAnalysis):
        """Split dataset into two parts.

        Args:
            dataset: Dataset to be split into train/test or train/val subsets

        Returns:
            a tuple of the subset datasets
        """
        if self.split_by == "time":
            sequences = list(enumerate(dataset.sample_sequences))

            sorted_sequences = sorted(sequences, key=lambda x: x[1]["seq_id"])
            selected_indices = [x[0] for x in sorted_sequences]

            split_idx = int(len(sorted_sequences) * 0.8)
            train_indices = selected_indices[:split_idx]
            val_indices = selected_indices[split_idx:]

        else:
            sequences = list(enumerate(dataset.sample_sequences))
            train_indices, val_indices = group_shuffle_split(
                [x[1]["id"] for x in sequences], train_size=0.8, random_state=0
            )

        # select train and val sequences and remove enumeration
        train_sequences = [sequences[i][1] for i in train_indices]
        val_sequences = [sequences[i][1] for i in val_indices]

        return train_sequences, val_sequences

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """

        self.task = self.dataset.task

        dgtl_sequences = list(enumerate(self.dataset.sample_sequences))
        dgtl_train_indices, test_indices = group_shuffle_split(
            [x[1]["id"] for x in dgtl_sequences], train_size=0.8, random_state=0
        )

        if stage in ["fit", "validate"]:
            index_mapping_train = {
                new_index: original_index
                for new_index, original_index in enumerate(dgtl_train_indices)
            }
            train_sequences = [
                self.dataset.sample_sequences[i] for i in dgtl_train_indices
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
            val_sequences = [self.dataset.sample_sequences[i] for i in val_indices]
            val_sequences = list(enumerate(val_sequences))
            val_indices, calibration_indices = group_shuffle_split(
                [x[1]["id"] for x in val_sequences], train_size=0.8, random_state=0
            )
            val_indices = [index_mapping_val[i] for i in val_indices]
            calibration_indices = [index_mapping_val[i] for i in calibration_indices]

            # Create train val subset dataset
            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, val_indices)
            self.calibration_dataset = Subset(self.dataset, calibration_indices)

        if stage in ["test"]:
            self.test_dataset = Subset(self.dataset, test_indices)

    def calibration_dataloader(self) -> torch.utils.data.DataLoader:
        """Return a dataloader for the calibration dataset."""
        return DataLoader(
            self.calibration_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
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
