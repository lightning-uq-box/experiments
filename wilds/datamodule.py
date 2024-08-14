from lightning import LightningDataModule
from wilds import get_dataset

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch


def collate_to_dict(batch):
    """Collate function to return a dictionary."""
    # Unpack list of tuples
    x, y, metadata = zip(*batch)

    # Stack the lists into tensors
    x = torch.stack(x)
    y = torch.stack(y)
    metadata = torch.stack(metadata)

    # Split metadata into single columns
    meta_1, meta_2, meta_3, meta_4 = torch.unbind(metadata, dim=1)

    new_batch = {
        "input": x,
        "target": y,
        "meta_1": meta_1,
        "meta_2": meta_2,
        "meta_3": meta_3,
        "meta_4": meta_4,
    }
    return new_batch


class WILDSPovertyDataModule(LightningDataModule):
    """Wilds Poverty DataModule."""

    def __init__(self, root: str, batch_size: int, num_workers: int):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # https://github.com/Feuermagier/Beyond_Deep_Ensembles/blob/b805d6f9de0bd2e6139237827497a2cb387de11c/experiments/base/wilds1.py#L17
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                # Images are already normalized tensors
            ]
        )

        dataset = get_dataset(dataset="poverty", root_dir=self.root)

        self.train_dataset = dataset.get_subset("train")
        self.val_dataset = dataset.get_subset("val")
        self.ood_test_dataset = dataset.get_subset("test")
        self.iid_test_dataset = dataset.get_subset("id_test")

    def train_dataloader(self) -> DataLoader:
        """Get Train Dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_to_dict,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get Validation Dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_to_dict,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Get Test Dataloader."""
        return DataLoader(
            self.ood_test_dataset,
            batch_size=self.batch_size * 4,
            num_workers=self.num_workers,
            collate_fn=collate_to_dict,
            shuffle=False,
        )

    def iid_test_dataloader(self) -> DataLoader:
        """Get IID Test Dataloader."""
        return DataLoader(
            self.iid_test_dataset,
            batch_size=self.batch_size * 4,
            num_workers=self.num_workers,
            collate_fn=collate_to_dict,
            shuffle=False,
        )


# dm = WILDSPovertyDataModule(root="/p/project/hai_uqmethodbox/nils/projects/experiments/wilds/data/data", batch_size=32, num_workers=0)
# dm.setup()

# train_loader = dm.train_dataloader()
# val_loader = dm.val_dataloader()
# test_loader = dm.test_dataloader()
# iid_test_loader = dm.iid_test_dataloader()

# print(len(train_loader))
# print(len(val_loader))
# print(len(test_loader))

# batch = next(iter(train_loader))

# import pdb
# pdb.set_trace()

# print(0)
