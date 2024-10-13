from torch import nn


LOSSES_REGISTRY = {
    'MSE': nn.MSELoss()
}