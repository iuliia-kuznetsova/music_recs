'''
    Raw data loading

    This module provides functionality to load raw data from the environment variables.

    Usage examples:
    python -m src.raw_data_loading # load raw data from the environment variables
'''

# ---------- Imports ---------- #
import os
import logging
import gc
import requests

from dotenv import load_dotenv

from src.logging_set_up import setup_logging

# ---------- Logging setup ---------- #
logger = setup_logging('raw_data_loading')

# ---------- Load environment variables ---------- #
def load_env_with_logging():
    '''
        Load .env file from config/ directory and log status info

        Args:
        - None

        Returns:
        - True if the environment variables were loaded successfully, False otherwise
    '''
    # Load from config/.env (relative to project root)
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
    env_path = os.path.join(config_dir, '.env')
    required_vars = [
        'RAW_DATA_DIR',
        'RAW_URL_TRACKS',
        'RAW_URL_CATALOG_NAMES',
        'RAW_URL_INTERACTIONS'
    ]
    
    logger.info(f'Loading .env: {required_vars} from {env_path}')

    # Check if .env exists
    if not os.path.exists(env_path):
        logger.error(f'Failed to load environment variables: .env file not found at {env_path}')
        return False

    load_dotenv(env_path)

    # Check required variables
    missing_var = [var for var in required_vars if not os.getenv(var)]
    if missing_var:
        logger.error(f'Failed to load environment variable {missing_var}')
        return False

    logger.info('DONE: Environment variables loaded successfully')
    return True

# ---------- Dwnload datasets ---------- #
def download_file(url: str, save_path: str) -> bool:
    '''
        Download a file from URL and save it locally.

        Args:
        - url - URL of the file to download
        - save_path - path to save the file

        Returns:
        - True if the file was downloaded successfully, False otherwise
    '''
    
    logger.info(f'Downloading: {url}')
    # Make directory if not yet exists 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Download dataset
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        logger.error(f'Failed to download {url} with exception {e}')
        return False

    # Save dataset locally
    try:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        logger.info(f'DONE: Dataframe saved to {save_path}')
        return True
    except Exception as e:
        logger.error(f'Failed to save {save_path}: {e}')
        return False

def download_all_raw() -> bool:
    '''
        Download all raw datasets from the environment variables.

        Args:
        - None

        Returns:
        - True if all raw datasets were downloaded successfully, False otherwise
    '''

    logger.info('Downloading all raw datasets')
    raw_dir = os.getenv('RAW_DATA_DIR', './data/raw')

    datasets = {
        'tracks.parquet': os.getenv('RAW_URL_TRACKS'),
        'catalog_names.parquet': os.getenv('RAW_URL_CATALOG_NAMES'),
        'interactions.parquet': os.getenv('RAW_URL_INTERACTIONS'),
    }

    failed = False
    for filename, url in datasets.items():
        save_path = os.path.join(raw_dir, filename)
        success = download_file(url, save_path)
        if not success:
            failed = True
    
    # Free memory after downloading
    gc.collect()
    
    if failed:
        logger.error('Failed to download one or more raw datasets')
        return False
    else:
        logger.info('DONE: All raw datasets downloaded successfully')
        return True

# ---------- Main entry point ---------- #
if __name__ == '__main__':

    logger.info('Running raw data loading pipeline')

    load_env_with_logging()
    download_all_raw()

    logger.info('DONE: Raw data loading completed successfully')

__all__ = ['load_env_with_logging', 'download_file', 'download_all_raw']