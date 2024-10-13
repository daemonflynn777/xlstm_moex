from dacite import from_dict
from dacite import Config as DaciteConfig
import numpy as np
from omegaconf import OmegaConf
from pprint import pprint
import torch
from torch import nn
import torch.optim as optim
from typing import Dict, Any

from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig

from xlstm_moex.loss import LOSSES_REGISTRY
from xlstm_moex.models.base import BaseModel
from xlstm_moex.optimizer import OPTIMIZERS_REGISTRY
from xlstm_moex.scheduler import SCHEDULERS_REGISTRY
from xlstm_moex.utils.logging import init_logger

logger = init_logger(__name__)


class xLSTMtime(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            xlstm_config: Dict[str, Any],
            **kwargs
    ):
        """Init xLSTM model for time series prediction.

        Args:
            input_size: number of features in each element of input sequence.
            output_size: number of features to output for each element of input sequence.
        """
        super(xLSTMtime, self).__init__()

        block_stack_config = from_dict(
            data_class=xLSTMBlockStackConfig,
            data=OmegaConf.to_container(xlstm_config),
            config=DaciteConfig(strict=True)
        )

        self.fc_in = nn.Linear(in_features=input_size, out_features=xlstm_config.embedding_dim)
        self.xlstm = xLSTMBlockStack(block_stack_config)
        self.fc_out = nn.Linear(in_features=xlstm_config.embedding_dim, out_features=output_size)
    
    def forward(self, x):
        x = self.fc_in(x)
        x = self.xlstm(x)
        x = self.fc_out(x)
        return x
    

class xLSTMmodel(BaseModel):
    def __init__(self, device: str, model_params: Dict[str, Any]):
        """Init interface for xLSTMtime model.
        """
        super(xLSTMmodel).__init__()

        self.device = device
        self.model_params = model_params
        self.model = xLSTMtime(
            input_size=model_params['input_features'],
            output_size=model_params['output_features'],
            xlstm_config=model_params['xlstm_config']
        )

        self.model = self.model.to(device=self.device)


    def train(self, train_params: Dict[str, Any], data: Dict[str, Dict[str, np.array]]):
        num_epochs = train_params['num_epochs']
        criterion = LOSSES_REGISTRY[train_params['criterion']]
        optimizer = OPTIMIZERS_REGISTRY[train_params['optimizer']['type']]
        optimizer = optim.Adam(
            params=self.model.parameters(),
            **train_params['optimizer']['optimizer_params']
        )
        if 'scheduler' in train_params:
            scheduler = SCHEDULERS_REGISTRY[train_params['scheduler']['type']]
            scheduler = scheduler(
                optimizer=optimizer,
                **train_params['scheduler']['scheduler_params']
            )

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(input_seq)
                loss = criterion(output, target_seq)
                loss.backward()
                optimizer.step()

                if (batch_idx + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # Validation
            self.model.eval()
            with torch.no_grad():
                total_loss = 0
                for input_seq, target_seq in val_loader:
                    output = self.model(input_seq)
                    loss = criterion(output, target_seq)
                    total_loss += loss.item()
                avg_loss = total_loss / len(val_loader)
                print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_loss:.4f}")