import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the participant selection dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    logger.info(f"Loading data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"No data found in file: {filepath}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the data: {e}")
        raise


