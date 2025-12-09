'''
    CLI entry point for music recommendation system.

    Usage examples:
        python3 -m src.main # run full pipeline
        python3 -m src.main --skip-download  # skip raw data download if already present
'''

# ---------- Imports ---------- #
import os
import sys
import gc
import argparse
from scipy.sparse import load_npz
import logging
from pythonjsonlogger import jsonlogger
import traceback
from dotenv import load_dotenv
import polars as pl

from src.logging_set_up import setup_logging
from src.raw_data_loading import load_env_with_logging, download_all_raw
from src.data_preprocessing import run_preprocessing
from src.train_test_split import run_train_test_split
from src.popularity_based_model import generate_popularity_recommendations
from src.als_model import als_recommendations
from src.similarity_based_model import similarity_based_recommendations
from src.rec_ranking import run_ranking_pipeline
from src.rec_evaluation import evaluate_model

# ---------- Logging setup ---------- #
logger = setup_logging('main')

# ---------- Main pipeline---------- #
def main():
    '''
        Main entry point: 
        1. Load environment variables
        2. Download raw data (if needed)
        3. Preprocess data
        4. Split data into train/test sets
        5. Train models: popularity based, als, ranking CatBoost
        6. Evaluate models
        7. Generate final recommendations
    '''

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Music Recommendation System')
    parser.add_argument('--skip-download', action='store_true', default=False)
    args = parser.parse_args()
    
    # ---------- Step 1: Load environment variables ---------- #
    print('\n' + '='*80)
    logger.info('STEP 1: Loading environment variables')
    print('='*80)
    
    load_dotenv()

    raw_dir = os.getenv('RAW_DATA_DIR', './data/raw')
    preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', './data/preprocessed')

    logger.info('DONE: loading environment variables completed')
    logger.info(f'INFO: Raw data directory: {raw_dir}')
    logger.info(f'INFO: Preprocessed data directory: {preprocessed_dir}')
    
    # ---------- Step 2: Download raw data (if not skipped) ---------- #
    if not args.skip_download:
        print('\n' + '='*80)
        logger.info('STEP 2: Downloading raw data')
        print('='*80)
        
        try:
            download_all_raw()
            logger.info('DONE: Raw data download completed')
        except Exception as e:
            logger.error(f'ERROR: Failed to download raw data: {e}')
            sys.exit(1)
    else:
        print('\n' + '='*80)
        logger.info('STEP 2: Skipping raw data download (--skip-download flag)')
        print('='*80)
        
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
    print('\n' + '='*80)
    print('STEP 3: Preprocessing data')
    print('='*80)
    
    try:
        run_preprocessing()
        
        print('\n' + '='*80)
        logger.info('DONE: Preprocessing pipeline completed successfully')
        print('='*80)

        # Verify that the required preprocessed data files exist
        preprocessed_files = [
            'items.parquet', 'tracks_catalog_clean.parquet', 
            'events.parquet'
        ]
        missing_prerprocessed_files = [f for f in preprocessed_files 
            if not os.path.exists(os.path.join(preprocessed_dir, f))]
        
        # Check label_encoders.pkl in models directory
        models_dir = os.getenv('MODELS_DIR', './models')
        if not os.path.exists(os.path.join(models_dir, 'label_encoders.pkl')):
            missing_prerprocessed_files.append('label_encoders.pkl (in models/)')
        
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
        print('='*80 + '\n')
        
    except Exception as e:
        logger.error(f'ERROR: Preprocessing failed: {e}')
        traceback.print_exc()
        sys.exit(1)
    
    # ---------- Step 4: Split data into train/test sets ---------- #
    print('\n' + '='*80)
    logger.info('STEP 4: Splitting data into train/test sets')
    print('='*80)
    
    try:
        run_train_test_split()
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
    train_events_shape = pl.read_parquet(f'{preprocessed_dir}/train_events.parquet').shape
    test_events_shape = pl.read_parquet(f'{preprocessed_dir}/test_events.parquet').shape
    train_matrix_shape = load_npz(f'{preprocessed_dir}/train_matrix.npz').shape
    test_matrix_shape = load_npz(f'{preprocessed_dir}/test_matrix.npz').shape
    
    logger.info(f'INFO: Train events dataframe shape: {train_events_shape}')
    logger.info(f'INFO: Test events dataframe shape: {test_events_shape}')
    logger.info(f'INFO: Train matrix shape: {train_matrix_shape}')
    logger.info(f'INFO: Test matrix shape: {test_matrix_shape}')

    # ---------- Step 5: Find popularity_based recommendations ---------- #     
    print('\n' + '='*80)
    logger.info('STEP 5: Finding popularity-based recommendations')
    print('='*80)
    
    try:
        generate_popularity_recommendations()
    except Exception as e:
        logger.error(f'ERROR: Finding popular tracks failed: {e}')
        traceback.print_exc()
        sys.exit(1)

    # ---------- Step 6: Find ALS recommendations ---------- #
    print('\n' + '='*80)
    logger.info('STEP 6: Finding ALS recommendations')
    print('='*80)
    
    try:
        als_recommendations()
        # Free memory after ALS training
        gc.collect()
    except Exception as e:
        logger.error(f'ERROR: Finding ALS recommendations failed: {e}')
        traceback.print_exc()
        sys.exit(1)

    # ---------- Step 7: Find similarity-based recommendations ---------- #
    print('\n' + '='*80)
    logger.info('STEP 7: Finding similarity-based recommendations')
    print('='*80)
    
    try:
        similarity_based_recommendations()
        # Free memory after similar tracks computation
        gc.collect()
    except Exception as e:
        logger.error(f'ERROR: Generating similarity-based recommendations failed: {e}')
        traceback.print_exc()
        sys.exit(1)

    # ---------- Step 8: Rank recommendations ---------- #
    print('\n' + '='*80)
    logger.info('STEP 8: Ranking recommendations')
    print('='*80)
    
    try:
        run_ranking_pipeline()
        gc.collect()
    except Exception as e:
        logger.error(f'ERROR: Ranking recommendations failed: {e}')
        traceback.print_exc()
        sys.exit(1)

    # ---------- Step 9: Evaluate models ---------- #
    print('\n' + '='*80)
    logger.info('STEP 9: Evaluating models')
    print('='*80)
    
    try:
        evaluate_model()
        gc.collect()
    except Exception as e:
        logger.error(f'ERROR: Models evaluation failed: {e}')
        traceback.print_exc()
        sys.exit(1)

    print('\n' + '='*80)
    logger.info('DONE: Recommendation system completed successfully')
    print('='*80)

if __name__ == '__main__':
    main()




