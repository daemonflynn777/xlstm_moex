import numpy as np
from typing import Iterable, List, Union

from xlstm_moex.utils.logging import init_logger

logger = init_logger(__name__)


def std_scaler(input_seq: Iterable[Union[int, float]]) -> List[Union[int, float]]:
    input_seq_mean = np.mean(input_seq)
    input_seq_std = np.std(input_seq)

    logger.info(f'Apply `std` scaling to input data, mean={input_seq_mean}, std={input_seq_std}')

    return [
        (el - input_seq_mean)/input_seq_std for el in input_seq
    ]


SCALERS_REGISTRY = {
    'std': std_scaler,
}
