from lightning import LightningDataModule
from torchvision.datasets import CIFAR10
from torchvision import transforms
from dataset import CIFAR10C
import torch

from pathlib import Path

from torch.utils.data import DataLoader


# Values taken from Wilson et al.
MEAN = torch.tensor([0.49, 0.48, 0.44])
STD = torch.tensor([0.2, 0.2, 0.2])
normalize = transforms.Normalize(MEAN, STD)

# From https://github.com/izmailovpavel/understandingbdl/blob/5d1004896ea4eb674cff1c2088dc49017a667e9e/swag/models/preresnet.py
transform_train = transforms.Compose(
    [
        transforms.Resize(32),  # For STL10
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize(32),  # For STL10
        transforms.ToTensor(),
        normalize,
    ]
)

transform_test_corrupted = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.ToTensor(),
        normalize,
    ]
)

class ToTensorTransform:
    """Convert integer label to tensor."""

    def __call__(self, label):
        return torch.tensor(label, dtype=torch.long)


def collate_to_dict(batch):
    """Collate function to return a dictionary."""
    image, target = [b[0] for b in batch], [b[1] for b in batch]
    return {
        "input": torch.stack(image, dim=0),
        "target": torch.stack(target, dim=0).squeeze(-1).long(),
    }


class CIFAR10DataModule(LightningDataModule):
    """CIFAR10 DataModule."""

    def __init__(self, root: str, batch_size: int, num_workers: int):
        super().__init__()
        self.root = root
        self.iid_root = Path(root) / "CIAR-10-batches-py"
        self.ood_root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = CIFAR10(
            root=self.root,
            train=True,
            transform=transform_train,
            target_transform=ToTensorTransform(),
        )

        self.val_dataset = None

        self.iid_test_dataset = CIFAR10(
            root=self.root,
            train=False,
            transform=transform_test,
            target_transform=ToTensorTransform(),
        )

    def train_dataloader(self) -> DataLoader:
        """Get Train Dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_to_dict,
        )

    def val_dataloader(self) -> DataLoader:
        pass

    def test_dataloader(self) -> DataLoader:
        """Get Test Dataloader."""
        return DataLoader(
            self.iid_test_dataset,
            batch_size=self.batch_size * 4,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_to_dict,
        )

    def ood_test_dataloader(self, severity: int) -> DataLoader:
        ds = CIFAR10C(
            root=self.ood_root,
            subset="all",
            severity=severity,
            transform=transform_test_corrupted,
            target_transform=ToTensorTransform(),
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size * 4,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_to_dict,
        )

