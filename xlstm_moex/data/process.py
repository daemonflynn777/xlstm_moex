import numpy as np
import pandas as pd
from typing import Tuple, Iterable, List, Union

from xlstm_moex.data.scale import SCALERS_REGISTRY
from xlstm_moex.utils.logging import init_logger

logger = init_logger(__name__)


def first_diff(input_seq: Iterable[Union[int, float]]) -> List[Union[int, float]]:
    return [
        input_seq[i+1] - input_seq[i] for i in range(len(input_seq) - 1)
    ]


def first_diff_rel(input_seq: Iterable[Union[int, float]]) -> List[Union[int, float]]:
    return [
        (input_seq[i+1] - input_seq[i])/input_seq[i]
        for i in range(len(input_seq) - 1)
    ]


def arima_processor(data_filename: str, **kwargs) -> pd.DataFrame:
    pass


def nn_processor(
        data_filename: str,
        value_column: str,
        date_column: str,
        sequence_length: int,
        scaler_type: str,
        diff_data: str = None,
        **kwargs
) -> Tuple[np.array, np.array]:
    """Function to process downloaded data for use with neural networks.
    Returns tuple of arrays, e.g.:
    ([[1,2], [3,4]], [4,6])
    where tuple[0][i] is input sequence and tuple[1][i] is target for input sequence.

    Args:
        data_filename: path to file with data.
        sequence_length: desired length of input sequence.
        scaler_type: type of scaling to apply. Only `std` is supported for now.
        diff_data: whether to use raw data or first differrences or relative differences.
            Default is None, allowed values are `first_diff` and `first_diff_rel`.

    Returns:
        tuple of numpy arrays.
    """
    logger.info(
        'Start applying `nn` processing to input data, '
        f'data_filename={data_filename}, '
        f'sequence_length={sequence_length}, scaler_type={scaler_type}'
    )
    data = (
        pd
        .read_csv(data_filename)
        .sort_values(by=[date_column], ascending=[True])
        [value_column]
        .to_list()
    )
    if diff_data is not None:
        logger.info(f'Will convert data to `{diff_data}`')
    if diff_data == 'first_diff':
        data = first_diff(data)
    if diff_data == 'first_diff_rel':
        data = first_diff_rel(data)

    data_scaled = SCALERS_REGISTRY[scaler_type](data)

    X_examples = []
    y_examples = []
    for i in range(len(data_scaled) - sequence_length - 1):
        X_examples.append(data_scaled[i:i+sequence_length])
        y_examples.append(data_scaled[i+sequence_length])
    X_examples = np.array(X_examples)
    y_examples = np.array(y_examples)
    logger.info(f'Number of examples is {X_examples.shape[0]}')

    return X_examples, y_examples


    
    
