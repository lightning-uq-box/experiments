import torch
import torch.nn as nn
import torch.nn.functional as F

# Code is adapted from https://github.com/Feuermagier/Beyond_Deep_Ensembles/blob/b805d6f9de0bd2e6139237827497a2cb387de11c/src/architectures/resnet.py


class FixableDropout(nn.Module):
    """
    A special version of PyTorchs torch.nn.Dropout that applies dropout when the model is in evaluation mode.
    If freeze_on_eval is True, the same dropout mask will be used for the entire minibatch when in evaluation mode (not in train model!)
    """

    def __init__(self, p, freeze_on_eval=True):
        super().__init__()
        self.p = torch.tensor(p)
        self.freeze_on_eval = freeze_on_eval

    def forward(self, x):
        if not self.training and self.freeze_on_eval:
            mask = (1 - self.p).expand(x.shape[1:])
            mask = torch.bernoulli(mask).to(x.device)
            return x * mask
        else:
            return F.dropout(x, self.p)

    def __repr__(self) -> str:
        return f"FixableDropout({self.p:0.3})"


def patch_dropout(module, freeze_on_eval=False, override_p=None, patch_fixable=False):
    """
    Replaces all torch.nn.Dropout layers by FixableDropout layers.
    If override_p is None, the original dropout rate is being conserved.
    Otherwise, the rate is set to dropout_p.
    If patch_fixable is True, FixableDropout layers get also replace (useful for changing the dropout rates)
    """
    patched = 0
    for name, m in list(module._modules.items()):
        if m._modules:
            patched += patch_dropout(
                m,
                freeze_on_eval=freeze_on_eval,
                override_p=override_p,
                patch_fixable=patch_fixable,
            )
        elif m.__class__.__name__ == "Dropout" or (
            patch_fixable and m.__class__.__name__ == "FixableDropout"
        ):
            patched += 1
            if override_p is not None:
                setattr(module, name, FixableDropout(override_p, freeze_on_eval))
            else:
                setattr(module, name, FixableDropout(m.p, freeze_on_eval))
    return patched


class FilterResponseNorm(nn.Module):
    def __init__(self, num_filters, eps=1e-6):
        super().__init__()
        self.eps = eps
        par_shape = (1, num_filters, 1, 1)  # [1,C,1,1]
        self.tau = torch.nn.Parameter(torch.zeros(par_shape))
        self.beta = torch.nn.Parameter(torch.zeros(par_shape))
        self.gamma = torch.nn.Parameter(torch.ones(par_shape))

    def forward(self, x):
        nu2 = torch.mean(torch.square(x), dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps)
        y = self.gamma * x + self.beta
        z = torch.max(y, self.tau)
        return z


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "swish":
        return nn.SiLU()
    else:
        raise ValueError("Unknown activation function " + activation)


def get_norm_layer(norm, out_channels, prior=None):
    if norm == "batch_static":
        return nn.BatchNorm2d(out_channels, track_running_stats=False)
    elif norm == "frn":
        if prior is None or isinstance(
            prior, tuple
        ):  # check for tuple to use plain frn on rank1
            return FilterResponseNorm(out_channels)
    else:
        raise ValueError("Unknown renormalization layer " + norm)


def get_conv_layer(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    bias=True,
    variational=False,
    prior=None,
    rank1=False,
    components=1,
):
    layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    nn.init.kaiming_normal_(layer.weight.data)
    return layer


def get_linear_layer(
    in_features, out_features, variational, prior, rank1=False, components=1
):
    return nn.Linear(in_features, out_features)


########################### ResNet ###########################


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        activation="relu",
        norm="batch_static",
        dropout_p=None,
        variational=False,
        rank1=False,
        prior=None,
        components=1,
    ):
        super().__init__()

        self.main_path = nn.Sequential(
            get_conv_layer(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True,
                variational=variational,
                prior=prior,
                rank1=rank1,
                components=components,
            ),
            FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
            get_norm_layer(norm, out_channels, prior=prior),
            get_activation(activation),
            get_conv_layer(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                variational=variational,
                prior=prior,
                rank1=rank1,
                components=components,
            ),
            FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
            get_norm_layer(norm, out_channels, prior=prior),
            # get_activation(activation),
        )

        if stride != 1:
            self.skip_path = nn.Sequential(
                get_conv_layer(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                    variational=variational,
                    prior=prior,
                    rank1=rank1,
                    components=components,
                ),
                FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
                # get_norm_layer(norm, out_channels, prior=prior)
            )
        else:
            self.skip_path = nn.Identity()

        self.out_activation = get_activation(activation)

    def forward(self, input):
        return self.out_activation(self.main_path(input) + self.skip_path(input))


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        activation="relu",
        norm="batch_static",
        dropout_p=None,
        variational=False,
        rank1=False,
        prior=None,
        components=1,
    ):
        super().__init__()

        self.main_path = nn.Sequential(
            get_conv_layer(
                in_channels,
                in_channels,
                kernel_size=1,
                stride=stride,
                padding=1,
                bias=True,
                variational=variational,
                prior=prior,
                rank1=rank1,
                components=components,
            ),
            FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
            get_norm_layer(norm, in_channels, prior=prior),
            get_activation(activation),
            get_conv_layer(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True,
                variational=variational,
                prior=prior,
                rank1=rank1,
                components=components,
            ),
            FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
            get_norm_layer(norm, in_channels, prior=prior),
            get_activation(activation),
            get_conv_layer(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=1,
                bias=True,
                variational=variational,
                prior=prior,
                rank1=rank1,
                components=components,
            ),
            FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
            get_norm_layer(norm, out_channels, prior=prior),
            # get_activation(activation),
        )

        if stride != 1:
            self.skip_path = nn.Sequential(
                get_conv_layer(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                    variational=variational,
                    prior=prior,
                    rank1=rank1,
                    components=components,
                ),
                FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
                # get_norm_layer(norm, out_channels, prior=prior)
            )
        else:
            self.skip_path = nn.Identity()

        self.out_activation = get_activation(activation)

    def forward(self, input):
        return self.out_activation(self.main_path(input) + self.skip_path(input))


class ResNet20(nn.Module):
    def __init__(
        self,
        in_size=32,
        in_channels=3,
        classes=10,
        activation="swish",
        norm="frn",
        dropout_p=None,
        variational=False,
        prior=None,
        rank1=False,
        components=1,
    ):
        super().__init__()

        self.model = nn.Sequential(
            get_conv_layer(
                in_channels,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                variational=variational,
                prior=prior,
                rank1=rank1,
                components=components,
            ),
            FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
            BasicBlock(
                16,
                16,
                1,
                activation,
                norm,
                dropout_p,
                variational,
                rank1,
                prior,
                components=components,
            ),
            BasicBlock(
                16,
                16,
                1,
                activation,
                norm,
                dropout_p,
                variational,
                rank1,
                prior,
                components=components,
            ),
            BasicBlock(
                16,
                16,
                1,
                activation,
                norm,
                dropout_p,
                variational,
                rank1,
                prior,
                components=components,
            ),
            BasicBlock(
                16,
                32,
                2,
                activation,
                norm,
                dropout_p,
                variational,
                rank1,
                prior,
                components=components,
            ),
            BasicBlock(
                32,
                32,
                1,
                activation,
                norm,
                dropout_p,
                variational,
                rank1,
                prior,
                components=components,
            ),
            BasicBlock(
                32,
                32,
                1,
                activation,
                norm,
                dropout_p,
                variational,
                rank1,
                prior,
                components=components,
            ),
            BasicBlock(
                32,
                64,
                2,
                activation,
                norm,
                dropout_p,
                variational,
                rank1,
                prior,
                components=components,
            ),
            BasicBlock(
                64,
                64,
                1,
                activation,
                norm,
                dropout_p,
                variational,
                rank1,
                prior,
                components=components,
            ),
            BasicBlock(
                64,
                64,
                1,
                activation,
                norm,
                dropout_p,
                variational,
                rank1,
                prior,
                components=components,
            ),
            nn.AvgPool2d(8) if in_size >= 32 else nn.Identity(),
            nn.Flatten(),
            get_linear_layer(
                64 * (in_size // (32 if in_size >= 32 else 4)) ** 2,
                classes,
                variational,
                prior,
                rank1=rank1,
                components=components,
            ),
        )

    def forward(self, input):
        return self.model(input)


# model = ResNet20(32, 3, 10, activation="swish", norm="frn", dropout_p=0.1)
# print(model)
