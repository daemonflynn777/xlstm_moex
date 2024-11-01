from dacite import from_dict
from dacite import Config as DaciteConfig
import numpy as np
from omegaconf import OmegaConf
from pprint import pprint
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any

from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig

from xlstm_moex.data.dataset import TimeSeriesDataset
from xlstm_moex.loss import LOSSES_REGISTRY
from xlstm_moex.models.base import BaseModel
from xlstm_moex.optimizer import OPTIMIZERS_REGISTRY
from xlstm_moex.scheduler import SCHEDULERS_REGISTRY
from xlstm_moex.utils.logging import init_logger

logger = init_logger(__name__)


class LSTMtime(nn.Module):
    def __init__(
            self,
            lstm_config: Dict[str, Any],
            **kwargs
    ):
        """Init LSTM model for time series prediction.

        Args:
            input_size: number of features in each element of input sequence.
            output_size: number of features to output for each element of input sequence.
        """
        super(LSTMtime, self).__init__()
        self.lstm = nn.LSTM(**lstm_config)
        self.fc_out = nn.Linear(
            in_features=lstm_config['hidden_size'],
            out_features=lstm_config['input_size']
        )
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc_out(x)
        return x
    

class LSTMmodel(BaseModel):
    def __init__(self, device: str, model_params: Dict[str, Any]):
        """Init interface for LSTMtime model.
        """
        super(LSTMmodel).__init__()

        self.device = device
        self.model_params = model_params
        self.model = LSTMtime(
            lstm_config=model_params
        )

        self.model = self.model.to(device=self.device)


    def train(
            self,
            train_params: Dict[str, Any],
            data: Dict[str, Dict[str, np.array]],
            device: str
        ):
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

        train_dataloader = DataLoader(
            dataset=TimeSeriesDataset(
                X=data['train']['X'],
                y=data['train']['y'],
                device=device
            ),
            batch_size=train_params['batch_size'],
            shuffle=train_params.get('shuffle_data', True)
        )
        val_dataloader = DataLoader(
            dataset=TimeSeriesDataset(
                X=data['val']['X'],
                y=data['val']['y'],
                device=device
            ),
            batch_size=train_params.get('val_batch_size', train_params['batch_size']),
            shuffle=train_params.get('shuffle_data', True)
        )

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            total_loss_train = 0
            for batch_idx, (input_seq, target_seq) in enumerate(train_dataloader):
                # optimizer.zero_grad()
                output = self.model(input_seq)
                # output = output[:,-1,:]
                loss = criterion(output, target_seq)
                total_loss_train += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if 'scheduler' in train_params:
                    scheduler.step()

                if (batch_idx + 1) % 200 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
            avg_loss_train = total_loss_train / len(train_dataloader)
            # print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}")

            # Validation
            self.model.eval()
            with torch.no_grad():
                total_loss = 0
                for input_seq, target_seq in val_dataloader:
                    output = self.model(input_seq)
                    # output = output[:,-1,:]
                    loss = criterion(output, target_seq)
                    total_loss += loss.item()
                avg_loss = total_loss / len(val_dataloader)
                # print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_loss:.4f}")
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss_train:.4f}, "
                  f"Validation Loss: {avg_loss:.4f}")

    def forecast(
            self,
            forecast_params: Dict[str, Any],
            data: Dict[str, Dict[str, np.array]],
            device: str
        ):
        start_state = (
            torch
            .from_numpy(data['test']['X'][0, :].astype('float32'))
            .to(device=device)
            .reshape(-1, data['test']['X'].shape[1], 1)
        )
        true_labels = data['test']['y'][:, -1]
        # true_labels = data['test']['y']
        predicted_labels = []

        batch_size, seq_len, in_features = start_state.shape

        self.model.eval()
        for i in range(true_labels.shape[0]):
            next_el = self.model.forward(start_state)
            next_el = next_el[:,-1,:].view(batch_size, 1, in_features)
            predicted_labels.append(next_el)
            start_state = torch.cat((start_state, next_el), dim=1)[:, 1:, :] # concatenate along `seq_len` axis
        
        predicted_labels = torch.cat(predicted_labels, dim=1).view(true_labels.shape[0], in_features)
        predicted_labels = predicted_labels.cpu().detach().numpy()

        criterion = LOSSES_REGISTRY[forecast_params['criterion']]

        top_k_losses = {}
        for top_k in forecast_params['criterion_top_k']:
            top_k_loss = criterion(
                torch.from_numpy(predicted_labels[:top_k, :]),
                torch.from_numpy(true_labels[:top_k].reshape(-1, 1))
            ).item()
            top_k_losses[top_k] = {
                'loss': top_k_loss,
                'true': true_labels[:top_k].reshape(-1,),
                'predicted': predicted_labels[:top_k, :].reshape(-1,)
            }

        all_preds = self.model.forward(
            torch
            .from_numpy(data['test']['X'].astype('float32'))
            .to(device=device)
            .reshape(-1, data['test']['X'].shape[1], 1)
        )[:,-1,:].view(true_labels.shape[0], in_features)
        top_k_losses['all'] = {
            'loss': criterion(
                torch.from_numpy(all_preds.cpu().detach().numpy()),
                torch.from_numpy(true_labels.reshape(-1, 1))
            ).item(),
            'true': true_labels.reshape(-1),
            'predicted': all_preds.cpu().detach().numpy().reshape(-1)
        }

        return top_k_losses