# sktime.forecasting.arima AutoARIMA
# https://github.com/styalai/xLSTM-pytorch
# https://github.com/muslehal/xLSTMTime/tree/main
# https://github.com/myscience/x-lstm
# https://coderzcolumn.com/tutorials/artificial-intelligence/pytorch-lstm-networks-for-time-series-regression-tasks

from xlstm_moex.models.xlstm_model import xLSTMmodel


MODELS_REGISTRY = {
    'xlstm': xLSTMmodel
}