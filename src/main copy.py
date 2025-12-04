'''
    CLI entry point for music recommendation system.

    Usage:
        python3 -m src.main
        python3 -m src.main --raw-dir /path/to/raw --preprocessed-dir /path/to/output
        python3 -m src.main --skip-download  # Skip data download if already present
'''

import os
import sys
import argparse
from dotenv import load_dotenv
from scipy.sparse import load_npz
import logging
from pythonjsonlogger import jsonlogger
import traceback

import polars as pl

from src.raw_data_loading import load_env_with_logging, download_all_raw
from src.data_preprocessing import run_preprocessing
from src.train_test_split import run_train_test_split
from src.popularity_based_rec import find_top_popular_tracks
from src.collaborative_rec import train_als_model
from src.similar_based_als import get_similar_tracks

# ---------- Logging setup ---------- #
# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure root logger for console output
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
)

# Get logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Terminal output
terminal_handler = logging.StreamHandler()
terminal_handler.setLevel(logging.INFO)
logger.addHandler(terminal_handler)

# JSON file output
log_file = 'logs/app_log.json'
json_handler = logging.FileHandler(log_file, mode='a')
json_handler.setLevel(logging.INFO)

json_formatter = jsonlogger.JsonFormatter(
    fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
    rename_fields={'asctime': 'timestamp', 'levelname': 'level', 'name': 'module'}
)
json_handler.setFormatter(json_formatter)
logger.addHandler(json_handler)

# Initial logging
logger.info('Music recommendation system starting', extra={'version': '1.0', 'component': 'main'})

# ---------- Main pipeline---------- #
def main():
    '''
        Main entry point: 
        1. Load environment variables
        2. Download raw data (if needed)
        3. Preprocess data
        4. Split data into train/test sets
        5. Train models
        6. Evaluate models
        7. Generate final recommendations
    '''

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Music Recommendation System - Data Loading & Preprocessing Pipeline',
    )
    
    parser.add_argument(
        '--raw-dir',
        help='Directory with raw parquet files (tracks, catalog_names, interactions)',
        metavar='PATH'
    )
    parser.add_argument(
        '--preprocessed-dir',
        help='Output directory for preprocessed data files',
        metavar='PATH'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip downloading raw data if files already exist'
    )
    
    args = parser.parse_args()
    
    # ---------- Step 1: Load environment variables ---------- #
    print('\n' + '='*60)
    logger.info('STEP 1: Loading environment variables')
    print('='*60)
    
    if not load_env_with_logging():
        logger.error('ERROR: Failed to load environment variables')
        sys.exit(1)

    raw_dir = args.raw_dir or os.getenv('RAW_DATA_DIR', './data/raw')
    preprocessed_dir = args.preprocessed_dir or os.getenv('PREPROCESSED_DATA_DIR', './data/preprocessed')

    logger.info('DONE: loading environment variables completed')
    logger.info(f'INFO: Raw data directory: {raw_dir}')
    logger.info(f'INFO: Preprocessed data directory: {preprocessed_dir}')
    
    # ---------- Step 2: Download raw data (if not skipped) ---------- #
    if not args.skip_download:
        print('\n' + '='*60)
        logger.info('STEP 2: Downloading raw data')
        print('='*60)
        
        try:
            download_all_raw()
            logger.info('DONE: Raw data download completed')
        except Exception as e:
            logger.error(f'ERROR: Failed to download raw data: {e}')
            sys.exit(1)
    else:
        print('\n' + '='*60)
        logger.info('STEP 2: Skipping raw data download (--skip-download flag)')
        print('='*60)
        
        # Verify that the required raw data files exist
        raw_files = ['tracks.parquet', 'catalog_names.parquet', 'interactions.parquet']
        missing_raw_files = [f for f in raw_files if not os.path.exists(os.path.join(raw_dir, f))]
        
        if missing_raw_files:
            logger.error(f'ERROR: Missing files: {missing_raw_files}')
            logger.info(f'INFO: Run without --skip-download to download raw data')
            sys.exit(1)
        
        logger.info('DONE: Raw data download completed')
        logger.info(f'INFO: All raw files present in {raw_dir}')
    
    # ---------- Step 3: Run preprocessing pipeline ---------- #
    print('\n' + '='*60)
    print('STEP 3: Preprocessing data')
    print('='*60)
    
    try:
        run_preprocessing(raw_dir, preprocessed_dir)
        
        print('\n' + '='*60)
        logger.info('DONE: Preprocessing pipeline completed successfully')
        print('='*60)

        # Verify that the required preprocessed data files exist
        preprocessed_files = [
            'items.parquet', 'tracks_catalog_clean.parquet', 
            'events.parquet', 'label_encoders.pkl'
        ]
        missing_prerprocessed_files = [f for f in preprocessed_files 
            if not os.path.exists(os.path.join(preprocessed_dir, f))]
        
        if missing_prerprocessed_files:
            logger.error(f'ERROR: Missing files: {missing_prerprocessed_files}')
            logger.info(f'INFO: Check preprocessing pipeline to generate preprocessed data')
            sys.exit(1)
        
        # Log shapes
        items_shape = pl.read_parquet(f'{preprocessed_dir}/items.parquet').shape
        catalog_shape = pl.read_parquet(f'{preprocessed_dir}/tracks_catalog_clean.parquet').shape
        events_shape = pl.read_parquet(f'{preprocessed_dir}/events.parquet').shape

        logger.info(f'INFO: Items dataframe shape:          {items_shape}')
        logger.info(f'INFO: Tracks catalog dataframe shape: {catalog_shape}')
        logger.info(f'INFO: Events dataframe shape:         {events_shape}')
        print('='*60 + '\n')
        
    except Exception as e:
        logger.error(f'ERROR: Preprocessing failed: {e}')
        traceback.print_exc()
        sys.exit(1)
    
    # ---------- Step 4: Split data into train/test sets ---------- #
    print('\n' + '='*60)
    logger.info('STEP 4: Splitting data into train/test sets')
    print('='*60)
    
    try:
        run_train_test_split(preprocessed_dir)
    except Exception as e:
        logger.error(f'ERROR: Splitting data into train/test sets failed: {e}')
        traceback.print_exc()
        sys.exit(1)

    # Verify that the required preprocessed data files exist
    train_test_split_files = ['train_events.parquet', 'test_events.parquet', 'train_matrix.npz', 'test_matrix.npz']
    missing_train_test_split_files = [f for f in train_test_split_files if not os.path.exists(os.path.join(preprocessed_dir, f))]
    
    if missing_train_test_split_files:
        logger.error(f'ERROR: Missing files: {missing_train_test_split_files}')
        logger.info(f'INFO: Check train/test split pipeline to generate train/test split data')
        sys.exit(1)
        
    # Log shapes
    import polars as pl
    train_events_shape = pl.read_parquet(f'{preprocessed_dir}/train_events.parquet').shape
    test_events_shape = pl.read_parquet(f'{preprocessed_dir}/test_events.parquet').shape
    train_matrix_shape = load_npz(f'{preprocessed_dir}/train_matrix.npz').shape
    test_matrix_shape = load_npz(f'{preprocessed_dir}/test_matrix.npz').shape
    
    logger.info(f'INFO: Train events dataframe shape: {train_events_shape}')
    logger.info(f'INFO: Test events dataframe shape: {test_events_shape}')
    logger.info(f'INFO: Train matrix shape: {train_matrix_shape}')
    logger.info(f'INFO: Test matrix shape: {test_matrix_shape}')
     
    print('\n' + '='*60)
    logger.info('STEP 5: Finding popularity-based recommendations')
    print('='*60)
    
    try:
        find_top_popular_tracks(preprocessed_dir)
    except Exception as e:
        logger.error(f'ERROR: Finding popular tracks failed: {e}')
        traceback.print_exc()
        sys.exit(1)

    print('\n' + '='*60)
    logger.info('STEP 6: Training collaborative filtering model')
    print('='*60)
    
    try:
        train_als_model(preprocessed_dir)
    except Exception as e:
        logger.error(f'ERROR: Training collaborative filtering model failed: {e}')
        traceback.print_exc()
        sys.exit(1)

    print('\n' + '='*60)
    logger.info('STEP 7: Loading similar tracks finder')
    print('='*60)
    
    try:
        similar_finder = get_similar_tracks(preprocessed_dir)
        similar_finder.build_full_index()
        logger.info('Similar tracks index built for all tracks')
    except Exception as e:
        logger.error(f'ERROR: Building similar tracks index failed: {e}')
        traceback.print_exc()
        sys.exit(1)

    print('\n' + '='*60)
    logger.info('DONE: Recommendation system completed successfully')
    print('='*60)

if __name__ == '__main__':
    main()




