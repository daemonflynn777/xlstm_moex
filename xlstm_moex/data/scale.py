import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Iterable, List, Union

from xlstm_moex.utils.logging import init_logger

logger = init_logger(__name__)


def std_scaler(input_seq: Iterable[Union[int, float]]) -> List[Union[int, float]]:
    """Apply standard scaling to input sequence."""
    input_seq_mean = np.mean(input_seq)
    input_seq_std = np.std(input_seq)

    logger.info(f'Apply `std` scaling to input data, mean={input_seq_mean}, std={input_seq_std}')

    return [
        (el - input_seq_mean)/input_seq_std for el in input_seq
    ]

def min_max_scaler(input_seq: Iterable[Union[int, float]]) -> List[Union[int, float]]:
    scaler = MinMaxScaler()
    fitted_scaler = scaler.fit([[el] for el in input_seq])
    scaled_data = fitted_scaler.transform([[el] for el in input_seq]).reshape(-1,)
    print(type(scaled_data))
    print(scaled_data)
    return scaled_data

SCALERS_REGISTRY = {
    'std': std_scaler,
    'min_max': min_max_scaler
}
