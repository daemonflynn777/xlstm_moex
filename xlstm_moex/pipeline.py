import torch

from typing import Any, Dict

from xlstm_moex.data import (
    PROCESSORS_REGISTRY,
    SPLITTERS_REGISTRY
)
from xlstm_moex.models import MODELS_REGISTRY
from xlstm_moex.utils.misc import load_experiment_config
from xlstm_moex.utils.logging import init_logger

logger = init_logger(__name__)


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
        self.processed_data = None
        self.train_val_test_data: Dict[str, Any] = {}
        self.model = None
        self.predictions = None

    def data_preprocess(self):
        data_processor = PROCESSORS_REGISTRY[self.pipeline_cfg['data_preproces']['type']]
        self.processed_data = data_processor(
            **self.pipeline_cfg['data_preproces']['data_process_params']
        )
    
    def train_val_test_split(self):
        data_splitter = SPLITTERS_REGISTRY[self.pipeline_cfg['train_val_test_split']['type']]
        self.train_val_test_data = data_splitter(
            X=self.processed_data[0],
            y=self.processed_data[1],
            **self.pipeline_cfg['train_val_test_split']['train_val_test_split_params']
        )

    def init_model(self):
        model = MODELS_REGISTRY[self.pipeline_cfg['model']['type']]
        self.model = model(
            device=self.device,
            model_params=self.pipeline_cfg['model']['model_params']
        )

    def train_model(self):
        self.model.train(
            train_params=self.pipeline_cfg['train'],
            data=self.train_val_test_data,
            device=self.device
        )
    
    def test(self):
        self.predictions = self.model.forecast(
            forecast_params=self.pipeline_cfg['test'],
            data=self.train_val_test_data,
            device=self.device
        )

    def run(self):
        """Run experiment."""

        if 'data_preproces' in self.pipeline_cfg:
            logger.info('Start `data_preproces` stage')
            self.data_preprocess()

        if 'train_val_test_split' in self.pipeline_cfg:
            logger.info('Start `train_val_test_split` stage')
            self.train_val_test_split()
        
        if 'model' in self.pipeline_cfg:
            logger.info('Start `init_model` stage')
            self.init_model()

        if 'train' in self.pipeline_cfg:
            logger.info('Start `train` stage')
            self.train_model()
        
        if 'test' in self.pipeline_cfg:
            logger.info('Start `test` stage')
            self.test()