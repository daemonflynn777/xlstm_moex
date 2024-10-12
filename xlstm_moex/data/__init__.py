from xlstm_moex.data.process import arima_processor, nn_processor
from xlstm_moex.data.split import arima_splitter, nn_splitter


PROCESSORS_REGISTRY = {
    'arima': arima_processor,
    'nn': nn_processor
} 

SPLITTERS_REGISTRY = {
    'arima': arima_splitter,
    'nn': nn_splitter
}