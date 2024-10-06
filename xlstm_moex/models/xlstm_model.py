from torch import nn
from typing import Dict, Any

from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig

from xlstm_moex.models.base import BaseModel
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
        self.fc_in = nn.Linear(in_features=input_size, out_features=xlstm_config.embedding_dim)
        self.xlstm = xLSTMBlockStack(xlstm_config)
        self.fc_out = nn.Linear(in_features=xlstm_config.embedding_dim, out_features=output_size)
    
    def forward(self, x):
        x = self.fc_in(x)
        x = self.xlstm(x)
        x = self.fc_out(x)
        return x
    

class xLSTMmodel(BaseModel):
    def __init__(self, **kwargs):
        """Init interface for xLSTMtime model.
        """
        super(xLSTMmodel).__init__(**kwargs)

    def train(self, train_params: Dict[str, Any]):
        pass