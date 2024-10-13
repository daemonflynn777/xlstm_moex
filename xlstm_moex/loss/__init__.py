from torch import nn


LOSSES_REGISTRY = {
    'MSE': nn.MSELoss(),
    'MAE': nn.L1Loss()
}