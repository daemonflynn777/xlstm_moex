import numpy as np
from sktime.forecasting.arima import AutoARIMA
import torch
from typing import Any, Dict

from xlstm_moex.loss import LOSSES_REGISTRY
from xlstm_moex.models.base import BaseModel


class ARIMAmodel(BaseModel):
    def __init__(self, model_params: Dict[str, Any], **kwargs):
        super(ARIMAmodel).__init__()

        self.model_params = model_params
        self.model = AutoARIMA(**model_params)
    
    def train(
            self,
            train_params: Dict[str, Any],
            data: Dict[str, Dict[str, np.array]],
            device: str
        ):
        self.model.fit(data['train']['X'])

        print(self.model.get_fitted_params())
    
    def forecast(
            self,
            forecast_params: Dict[str, Any],
            data: Dict[str, Dict[str, np.array]],
            device: str
        ):
        forecast_horizon = np.arange(1, len(data['test']['X']) + 1)

        predicted_labels = self.model.predict(fh=forecast_horizon)
        true_labels = data['test']['X']

        criterion = LOSSES_REGISTRY[forecast_params['criterion']]

        top_k_losses = {}
        for top_k in forecast_params['criterion_top_k']:
            top_k_loss = criterion(
                torch.from_numpy(predicted_labels[:top_k].reshape(-1, 1)),
                torch.from_numpy(true_labels[:top_k].reshape(-1, 1))
            ).item()
            top_k_losses[top_k] = {
                'loss': top_k_loss,
                'true': true_labels[:top_k],
                'predicted': predicted_labels[:top_k]
            }
        all_loss = criterion(
            torch.from_numpy(predicted_labels.reshape(-1, 1)),
            torch.from_numpy(true_labels.reshape(-1, 1))
        ).item()
        top_k_losses['all'] = {
            'loss': all_loss,
            'true': true_labels,
            'predicted': predicted_labels

        }
        return top_k_losses