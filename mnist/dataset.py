import torch

from torchvision.datasets import MNIST

from torchvision import transforms


# see https://github.com/runame/laplace-redux/blob/f40526ac9366a6eb0ecca660b7f407387d771f0c/utils/data_utils.py#L297


class RotationTransform:
    """Rotate the given angle."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return transforms.functional.rotate(x, self.angle)


class ToTensorTransform:
    """Convert integer label to tensor."""

    def __call__(self, label):
        return torch.tensor(label, dtype=torch.long)


class RotatedMNIST(MNIST):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        angle=0,
    ):
        self.angle = angle
        transform = transforms.Compose(
            [
                RotationTransform(angle),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        target_transform = ToTensorTransform()
        super().__init__(root, train, transform, target_transform, download)
