import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, EMNIST, KMNIST
import torchvision.transforms as transforms
from dataset import RotatedMNIST, ToTensorTransform


def collate_fn(batch) -> dict[str, torch.Tensor]:
    """Colate function for dataloader as dictionary."""
    images, targets = zip(*batch)
    images = torch.stack(images)
    targets = torch.stack(targets)
    return {"input": images, "target": targets}


def val_test_split(
    dataset, val_size=5000, batch_size=512, num_workers=5, pin_memory=False
):
    # Split into val and test sets
    test_size = len(dataset) - val_size
    dataset_val, dataset_test = torch.utils.data.random_split(
        dataset, (val_size, test_size), generator=torch.Generator().manual_seed(42)
    )
    return dataset_val, dataset_test


class MNISTDataModule(LightningDataModule):
    def __init__(self, root: str, batch_size: int = 64, num_workers=0) -> None:
        """Initialize MNIST datamodule.

        Args:
            root: Path to data.
            batch_size: Batch size.
            num_workers: Number of workers.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root

    def setup(self, stage: str) -> None:
        """Setup data loader."""
        # laplace paper https://github.com/runame/laplace-redux/blob/f40526ac9366a6eb0ecca660b7f407387d771f0c/utils/data_utils.py#L297
        # setup data loader for validation from the test set
        self.mnist_train = RotatedMNIST(
            self.root,
            train=True,
            download=True,
            angle=0,
        )

        mnist_val_test = RotatedMNIST(
            self.root,
            train=False,
            download=True,
            angle=0,
        )

        self.mnist_val, self.mnist_test = val_test_split(mnist_val_test, val_size=2000)

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size * 10,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader of in distribution data."""
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size * 10,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )

    def rotated_dataloader(self, angle: int) -> DataLoader:
        """Return dataloader for rotated MNIST."""
        rot_mnist = RotatedMNIST(
            self.root,
            train=False,
            download=True,
            angle=angle,
        )
        _, rot_mnist_test = val_test_split(rot_mnist, val_size=2000)
        return DataLoader(
            rot_mnist_test,
            batch_size=self.batch_size * 10,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )

    def ood_dataloader(self, ood_type: str) -> DataLoader:
        """Return dataloader for out of distribution data."""
        # https://github.com/runame/laplace-redux/blob/f40526ac9366a6eb0ecca660b7f407387d771f0c/utils/data_utils.py#L422
        target_transform = ToTensorTransform()
        input_transform = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
        if ood_type == "FMNIST":
            fmnist_val_test_set = FashionMNIST(
                self.root,
                train=False,
                transform=input_transform,
                target_transform=target_transform,
                download=True,
            )
            val_loader, test_loader = val_test_split(fmnist_val_test_set, val_size=0)
        elif ood_type == "EMNIST":
            emnist_val_test_set = EMNIST(
                self.root,
                split="digits",
                train=False,
                transform=input_transform,
                target_transform=target_transform,
                download=True,
            )
            val_loader, test_loader = val_test_split(emnist_val_test_set, val_size=0)
        elif ood_type == "KMNIST":
            kmnist_val_test_set = KMNIST(
                self.root,
                train=False,
                transform=input_transform,
                target_transform=target_transform,
                download=True,
            )
            val_loader, test_loader = val_test_split(kmnist_val_test_set, val_size=0)

        return DataLoader(
            test_loader,
            batch_size=self.batch_size * 10,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )


dm = MNISTDataModule(root="data")

dm.setup("fit")

train_loader = dm.train_dataloader()

batch = next(iter(train_loader))

for ood_type in ["FMNIST", "EMNIST", "KMNIST"]:
    ood_loader = dm.ood_dataloader(ood_type)
