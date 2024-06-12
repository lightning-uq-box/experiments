"""SKIPPD Dataset Implementation."""

from torch import Tensor
from torchgeo.datasets import SKIPPD


class MySKIPPDDataset(SKIPPD):
    def __getitem__(self, index) -> dict[str, str | Tensor]:
        """Return the data and label at the given index.

        Args:
            index: Index of the element to return.

        Returns:
            A dictionary containing the data and label.
        """
        sample = super().__getitem__(index)
        sample["input"] = sample["image"].float() / 255.0
        if self.task == "forecast":
            sample["target"] = sample["label"][-1].unsqueeze(-1)
        else:
            sample["target"] = sample["label"].unsqueeze(-1)
        del sample["image"]
        del sample["label"]
        return sample
