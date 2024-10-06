import torch

from typing import Any, Dict

from xlstm_moex.data.process import PROCESSORS_REGISTRY
from xlstm_moex.utils.misc import load_experiment_config


class ExperimentPipeline:
    def __init__(
            self,
            pipeline_cfg_dict: Dict[str, Any] = None,
            pipeline_cfg_filename: str = None
    ):
        if pipeline_cfg_dict:
            self.pipeline_cfg = pipeline_cfg_dict
        elif pipeline_cfg_filename:
            self.pipeline_cfg = load_experiment_config(pipeline_cfg_filename)
        else:
            raise ValueError(
                'You must provide either `pipeline_cfg_dict` of `pipeline_cfg_filename`'
            )
        
        self.device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.train_val_test_data: Dict[str, Any] = {}
        self.model = None

    def data_preprocess(self):
        data_processor = PROCESSORS_REGISTRY[self.pipeline_cfg['data_preproces']['type']]
        return data_processor(**self.pipeline_cfg['data_preproces']['data_process_params'])
    
    def train_val_test_split(self, processed_data):
        return {}

    def run(self):
        """Run experiment."""
        processed_data = None

        if 'data_preproces' in self.pipeline_cfg:
            processed_data = self.data_preprocess

        if 'train_val_test_split' in self.pipeline_cfg:
            self.train_val_test_data = self.train_val_test_split(processed_data)