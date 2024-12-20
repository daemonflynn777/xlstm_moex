import math
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict

from xlstm_moex.utils.logging import init_logger

logger = init_logger(__name__)


def arima_splitter(
        X: np.array,
        test_size: int = 0,
        **kwargs,
    ) -> Dict[str, Dict[str, np.array]]:
    """Split data into train, validation and test sets for use with arima"""
    num_examples = X.shape[0]
    X_test = X[num_examples-test_size:num_examples]
    X_train = X[:num_examples-test_size]

    print(X_test[-5:])

    return {
        'train': {
            'X': X_train, 'y': X_train
        },
        'test': {
            'X': X_test, 'y': X_test
        }
    }


def nn_splitter(
        X: np.array,
        y: np.array,
        val_part: float = 0.0,
        test_size: int = 0,
        random_state: int = 666,
        **kwargs,
    ) -> Dict[str, Dict[str, np.array]]:
    """Split data into train, validation and test sets."""
    num_examples = X.shape[0]
    X_test = X[num_examples-test_size:num_examples,:]
    y_test = y[num_examples-test_size:num_examples,:]
    # y_test = y[num_examples-test_size:num_examples]

    # print(y_test[-5:, -1])

    X_rest = X[:num_examples-test_size,:]
    y_rest = y[:num_examples-test_size,:]
    # y_rest = y[:num_examples-test_size]

    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        train_size=1 - val_part,
        random_state=random_state,
        shuffle=True
    )

    logger.info(f'Shape of train is {X_train.shape}')
    logger.info(f'Shape of val is {X_val.shape}')
    logger.info(f'Shape of test is {X_test.shape}')

    return {
        'train': {
            'X': X_train, 'y': y_train
        },
        'val': {
            'X': X_val, 'y': y_val
        },
        'test': {
            'X': X_test, 'y': y_test
        }
    }
       

    # val_data_X, val_data_y = None, None
    # test_data_X, test_data_y = None, None

    # train_part = 1.0 - val_part - test_part
    # num_train_examples = math.ceil(X.shape[0] * train_part)
    # train_data_X, train_data_y = X[:num_train_examples, :], y[:num_train_examples]

    # if num_train_examples == X.shape[0]:
    #     logger.info(
    #         'All data will be used for trainig as num_train_examples '
    #         'is equal to the number of all examples'
    #     )
    #     return {
    #         'train': {
    #             'X': train_data_X, 'y': train_data_y
    #         },
    #         'val': {
    #             'X': val_data_X, 'y': val_data_y
    #         },
    #         'test': {
    #             'X': test_data_X, 'y': test_data_y
    #         }
    #     }

    # if val_part == 0.0:
    #     logger.info('Warning, validation data size is zero!')
    # num_val_examples = math.ceil((X.shape[0] - num_train_examples) * val_part / (val_part + ))
    # val_data_X = X[num_train_examples:num_train_examples + num_val_examples, :]
    # val_data_y = y[num_train_examples:num_train_examples + num_val_examples]

    # if test_part == 0.0:
    #     logger.info('Warning, test data size is zero!')
    # num_test_examples = X.shape[0] - num_train_examples - num_val_examples
