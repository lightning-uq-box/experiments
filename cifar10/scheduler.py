import torch

def wilson_scheduler(optimizer, pretrain_epochs, lr_init, swag_lr=None):
    def wilson_schedule(epoch):
        t = (epoch) / pretrain_epochs
        lr_ratio = swag_lr / lr_init if swag_lr is not None else 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return factor
    return torch.optim.lr_scheduler.LambdaLR(optimizer, wilson_schedule)