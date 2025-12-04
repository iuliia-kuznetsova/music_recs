'''
    Train/Test Data Split

    This module provides functionality to split user-item interactions into
    train and test sets based on a chronological date threshold.

    Strategy:
    - Calculate date threshold as quantile of last_listen dates if not set in .env file.
    - Train set: all interactions BEFORE the threshold date
    - Test set: all interactions AFTER the threshold date

    Input:
    - events.parquet - User-track interaction events
    - ./models/label_encoders.pkl - User and track ID to index mappings (for model training)

    Output:
    - train_events.parquet - Training interactions
    - test_events.parquet - Test interactions
    - train_matrix.npz - Training sparse matrix
    - test_matrix.npz - Test sparse matrix
    - split_info.pkl - Split information

    Usage:
    python -m src.train_test_split --calculate-date-threshold
    python -m src.train_test_split --run-train-test-split
'''

# ---------- Imports ---------- #
import os
import gc
import pickle
import logging
from datetime import date
from typing import Tuple

import polars as pl
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from dotenv import load_dotenv

# ---------- Load environment variables ---------- #
# Load from config/.env (relative to project root)
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))

# ---------- Logging setup ---------- #
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ---------- Calculate date threshold ---------- #
def calculate_date_threshold(preprocessed_dir: str, train_ratio: float = None) -> date:
    '''
        Calculate the date threshold for train/test split.
        
        First checks DATE_THRESHOLD env variable.
        If not set, calculates threshold as quantile of last_listen dates.
    '''
    
    # Check if DATE_THRESHOLD is set in environment
    date_threshold_str = os.getenv('DATE_THRESHOLD')
    if date_threshold_str:
        try:
            # Parse date in DD.MM.YYYY format
            day, month, year = date_threshold_str.split('.')
            date_threshold = date(int(year), int(month), int(day))
            logger.info('Using DATE_THRESHOLD from environment: %s', date_threshold)
            return date_threshold
        except (ValueError, AttributeError) as e:
            logger.warning(f'Invalid DATE_THRESHOLD format: {date_threshold_str}. Expected DD.MM.YYYY. Using quantile method.')
    
    # Fall back to quantile-based calculation
    if train_ratio is None:
        train_ratio = float(os.getenv('TRAIN_RATIO', 0.8))

    logger.info('Loading events from %s', preprocessed_dir)
    events = pl.scan_parquet(f'{preprocessed_dir}/events.parquet')
    
    logger.info('Calculating date threshold using %.0f%% quantile', train_ratio * 100)
    # Convert date to days since epoch, find quantile, convert back
    # Polars cannot directly compute quantile on dates
    date_threshold_days = (
        events
            .select(
                pl.col('last_listen')
                .cast(pl.Date)
                .to_physical()
                .quantile(train_ratio)
                .cast(pl.Int32)
            )
            .collect()
            .item()
    )
    # Convert back to date
    date_threshold = pl.Series([date_threshold_days]).cast(pl.Date).item()
    logger.info('Date threshold calculated: %s', date_threshold)
    
    # Free up memory
    del (events, date_threshold_days)
    gc.collect()

    return date_threshold

# ---------- Split by date threshold ---------- #
def split_by_date_threshold(preprocessed_dir: str, date_threshold: date) -> Tuple[pl.DataFrame, pl.DataFrame]:
    '''
        Split events into train/test sets based on date threshold.
    '''

    logger.info('Loading events from %s', preprocessed_dir)
    events = pl.scan_parquet(f'{preprocessed_dir}/events.parquet')

    logger.info('Splitting events by date threshold')  
    # Split based on date (last_listen column)
    train_events = (
        events
        .filter(pl.col('last_listen').cast(pl.Date) <= date_threshold)
    )
    test_events = (
        events
        .filter(pl.col('last_listen').cast(pl.Date) > date_threshold)
    )
    logger.info('Successfully done with train/test events split')

    # Save train/test events
    train_events.sink_parquet(f'{preprocessed_dir}/train_events.parquet')
    test_events.sink_parquet(f'{preprocessed_dir}/test_events.parquet')
    logger.info('Train/test events split saved to %s and %s', f'{preprocessed_dir}/train_events.parquet', f'{preprocessed_dir}/test_events.parquet')

    # Save split metadata (need to collect to get stats)
    train_collected = pl.read_parquet(f'{preprocessed_dir}/train_events.parquet')
    test_collected = pl.read_parquet(f'{preprocessed_dir}/test_events.parquet')
    
    split_info = {
        'date_threshold': str(date_threshold),
        'train_interactions': train_collected.height,
        'test_interactions': test_collected.height,
        'train_users': train_collected['user_id'].n_unique(),
        'test_users': test_collected['user_id'].n_unique(),
        'train_tracks': train_collected['track_id'].n_unique(),
        'test_tracks': test_collected['track_id'].n_unique(),
    }
    with open(f'{preprocessed_dir}/train_test_split_info.pkl', 'wb') as f:
        pickle.dump(split_info, f)
    logger.info('Train/test split metadata saved to %s', f'{preprocessed_dir}/train_test_split_info.pkl')
    
    del train_collected, test_collected

    # Free up memory
    del (events, train_events, test_events, split_info)
    gc.collect()

    return None

# ---------- Create sparse matrix ---------- #
def create_sparse_matrix(preprocessed_dir: str) -> csr_matrix:
    '''
        Create sparse interaction matrix from dataframe with columns [user_id, track_id, listen_count].
    '''

    logger.info('Loading label encoders from %s', preprocessed_dir)
    models_dir = os.getenv('MODELS_DIR', './models')
    with open(f'{models_dir}/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)

    user_encoder = encoders['user_encoder']
    track_encoder = encoders['track_encoder']

    n_users = len(user_encoder)
    n_tracks = len(track_encoder)

    logger.info('Loading events from %s', preprocessed_dir)
    train_events = pl.read_parquet(f'{preprocessed_dir}/train_events.parquet')
    test_events = pl.read_parquet(f'{preprocessed_dir}/test_events.parquet')

    # Build sparse matrixes (users * tracks) for train and test sets
    # Use listen_count as the interaction strength
    
    # Build train sparse matrix
    logger.info('Building train sparse matrix')
    train_events_encoded = (
        train_events
            .with_columns([
                pl.col('user_id').replace(user_encoder).alias('user_idx'),
                pl.col('track_id').replace(track_encoder).alias('track_idx'),
            ])
            .drop_nulls(['user_idx', 'track_idx'])
    )

    train_row_indices = train_events_encoded['user_idx'].to_numpy()
    train_col_indices = train_events_encoded['track_idx'].to_numpy()
    train_data = train_events_encoded['listen_count'].to_numpy().astype(np.float32)
       
    train_matrix = csr_matrix(
        (train_data, (train_row_indices, train_col_indices)),
        shape=(n_users, n_tracks),
        dtype=np.float32
    )
    
    logger.info(f'Train sparse matrix created: {train_matrix.shape} with {train_matrix.nnz:,} non-zero entries')
    logger.info(f'Train sparsity: {100 * (1 - train_matrix.nnz / (n_users * n_tracks)):.4f}%')
    
    # Save train sparse matrix
    train_sparse_matrix_path = f'{preprocessed_dir}/train_matrix.npz'
    save_npz(train_sparse_matrix_path, train_matrix)
    logger.info(f'Train sparse matrix saved to {train_sparse_matrix_path}')

    # Build test sparse matrix
    logger.info('Building test sparse matrix')
    test_events_encoded = (
        test_events
            .with_columns([
                pl.col('user_id').replace(user_encoder).alias('user_idx'),
                pl.col('track_id').replace(track_encoder).alias('track_idx'),
            ])
            .drop_nulls(['user_idx', 'track_idx'])
    )

    test_row_indices = test_events_encoded['user_idx'].to_numpy()
    test_col_indices = test_events_encoded['track_idx'].to_numpy()
    test_data = test_events_encoded['listen_count'].to_numpy().astype(np.float32)
    
    test_matrix = csr_matrix(
        (test_data, (test_row_indices, test_col_indices)),
        shape=(n_users, n_tracks),
        dtype=np.float32
    )

    logger.info(f'Test sparse matrix created: {test_matrix.shape} with {test_matrix.nnz:,} non-zero entries')
    logger.info(f'Test sparsity: {100 * (1 - test_matrix.nnz / (n_users * n_tracks)):.4f}%')
    
    # Save test sparse matrix
    test_sparse_matrix_path = f'{preprocessed_dir}/test_matrix.npz'
    save_npz(test_sparse_matrix_path, test_matrix)
    logger.info(f'Test sparse matrix saved to {test_sparse_matrix_path}')

    logger.info('Successfully done with train and test sparse matrix creation')

    # Free up memory
    del (
        n_users, n_tracks,
        train_events, test_events, 
        train_events_encoded, test_events_encoded, 
        train_row_indices, train_col_indices, train_data, 
        test_row_indices, test_col_indices, test_data,
        user_encoder, track_encoder,
        train_matrix, test_matrix
    )
    gc.collect()

    return None
    
# ---------- Main entry point ---------- #
def run_train_test_split(preprocessed_dir: str):
    '''
        Main entry point for the train/test split pipeline.
        Loads events and runs the train/test split pipeline.
    '''
    logger.info('Starting train/test split')
    
    date_threshold = calculate_date_threshold(preprocessed_dir)
    split_by_date_threshold(preprocessed_dir, date_threshold)
    create_sparse_matrix(preprocessed_dir)

    logger.info('Successfully done with train/test split')
    
    return None

# ---------- All exports ---------- #
__all__ = ['run_train_test_split']
