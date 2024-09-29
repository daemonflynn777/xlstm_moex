import asyncio
import aiohttp
import aiomoex
import pandas as pd

from typing import List
from xlstm_moex.utils.logging import init_logger

logger = init_logger(__name__)


async def download_data(
        stock: str,
        columns: List[str] = None,
        start_date: str = None,
        end_date: str = None
) -> pd.DataFrame:
    """Async function to download raw data from MOEX.

    Args:
        stock: stock identifier.
        columns: list of columns to select.
            Most useful value is ['BOARDID', 'TRADEDATE', 'CLOSE', 'VOLUME', 'VALUE', 'OPEN', 'LOW', 'HIGH']
        start_date: start date in YYYY-MM-DD format.
        end_date: end date in YYYY-MM-DD format.
    
    Returns:
        dataframe with historical stock data
    """
    async with aiohttp.ClientSession() as session:
        data = await aiomoex.get_board_history(
            session=session,
            security=stock,
            columns=columns,
            start=start_date,
            end=end_date
        )
        df = pd.DataFrame(data)
    return df


async def get_historical_data(
        output_filename: str,
        stock: str,
        columns: List[str] = None,
        start_date: str = None,
        end_date: str = None
):
    """Async function to download and save historical data for specified stock
    
    Args:
        output_filename: name of the output .csv file.
        stock: stock identifier.
        columns: list of columns to select.
            Most useful value is ['BOARDID', 'TRADEDATE', 'CLOSE', 'VOLUME', 'VALUE', 'OPEN', 'LOW', 'HIGH']
        start_date: start date in YYYY-MM-DD format.
        end_date: end date in YYYY-MM-DD format.
    
    Returns:
        dataframe with historical stock data
    """
    df = await download_data(
        stock=stock,
        columns=columns,
        start_date=start_date,
        end_date=end_date
    )
    logger.info(f'Finished downloading data, {df.shape[0]} rows and {df.shape[1]} columns')

    df.to_csv(output_filename, index=False)
    logger.info(f'Saved to {output_filename}')
