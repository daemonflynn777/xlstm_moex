import torch.optim as optim


OPTIMIZERS_REGISTRY = {
    'Adam': optim.Adam,
    'AdamW': optim.AdamW
}