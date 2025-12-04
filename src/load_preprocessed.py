'''
Helper module to load preprocessed data for model training.

Usage:
    from src.load_preprocessed import load_interaction_data, load_encoders
    
    # Load sparse matrix and encoders
    matrix, encoders = load_interaction_data()
    
    # Or load separately
    encoders = load_encoders()
    user_encoder = encoders['user_encoder']
    track_encoder = encoders['track_encoder']
'''

import os
import pickle
from typing import Dict, Tuple, Any

import polars as pl
from scipy.sparse import load_npz, csr_matrix
from dotenv import load_dotenv

# Load environment variables from config/.env
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))


def load_encoders(models_dir: str = None) -> Dict[str, Dict[int, int]]:
    '''
    Load label encoders for users and tracks.
    
    Args:
        models_dir: Directory containing model files (default: from MODELS_DIR env or ./models)
        
    Returns:
        Dictionary with keys:
        - user_encoder: {user_id -> user_idx}
        - track_encoder: {track_id -> track_idx}
        - user_decoder: {user_idx -> user_id}
        - track_decoder: {track_idx -> track_id}
    '''
    if models_dir is None:
        models_dir = os.getenv('MODELS_DIR', './models')
    
    encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
    
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(
            f"Label encoders not found at {encoders_path}. "
            "Run preprocessing first: python3 -m src.main"
        )
    
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    
    return encoders


def load_interaction_matrix(preprocessed_dir: str = 'data/preprocessed') -> csr_matrix:
    '''
    Load sparse interaction matrix (users x tracks).
    
    Args:
        preprocessed_dir: Directory containing preprocessed data
        
    Returns:
        Sparse CSR matrix of shape (n_users, n_tracks)
    '''
    matrix_path = os.path.join(preprocessed_dir, 'interaction_matrix.npz')
    
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(
            f"Interaction matrix not found at {matrix_path}. "
            "Run preprocessing first: python3 -m src.main"
        )
    
    matrix = load_npz(matrix_path)
    return matrix


def load_interaction_data(preprocessed_dir: str = 'data/preprocessed') -> Tuple[csr_matrix, Dict[str, Dict[int, int]]]:
    '''
    Load both interaction matrix and encoders.
    
    Args:
        preprocessed_dir: Directory containing preprocessed data
        
    Returns:
        Tuple of (interaction_matrix, encoders)
    '''
    matrix = load_interaction_matrix(preprocessed_dir)
    encoders = load_encoders(preprocessed_dir)
    
    return matrix, encoders


def load_events(preprocessed_dir: str = 'data/preprocessed') -> pl.DataFrame:
    '''
    Load aggregated events dataframe.
    
    Args:
        preprocessed_dir: Directory containing preprocessed data
        
    Returns:
        Polars DataFrame with columns: user_id, track_id, listen_count, last_listen
    '''
    events_path = os.path.join(preprocessed_dir, 'events.parquet')
    
    if not os.path.exists(events_path):
        raise FileNotFoundError(
            f"Events not found at {events_path}. "
            "Run preprocessing first: python3 -m src.main"
        )
    
    return pl.read_parquet(events_path)


def load_catalog(preprocessed_dir: str = 'data/preprocessed') -> pl.DataFrame:
    '''
    Load tracks catalog with metadata.
    
    Args:
        preprocessed_dir: Directory containing preprocessed data
        
    Returns:
        Polars DataFrame with track metadata
    '''
    catalog_path = os.path.join(preprocessed_dir, 'tracks_catalog_clean.parquet')
    
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(
            f"Catalog not found at {catalog_path}. "
            "Run preprocessing first: python3 -m src.main"
        )
    
    return pl.read_parquet(catalog_path)


def load_items(preprocessed_dir: str = 'data/preprocessed') -> pl.DataFrame:
    '''
    Load items catalog (full denormalized catalog).
    
    Args:
        preprocessed_dir: Directory containing preprocessed data
        
    Returns:
        Polars DataFrame with all catalog information
    '''
    items_path = os.path.join(preprocessed_dir, 'items.parquet')
    
    if not os.path.exists(items_path):
        raise FileNotFoundError(
            f"Items not found at {items_path}. "
            "Run preprocessing first: python3 -m src.main"
        )
    
    return pl.read_parquet(items_path)


def get_data_summary(preprocessed_dir: str = 'data/preprocessed') -> Dict[str, Any]:
    '''
    Get summary statistics of preprocessed data.
    
    Args:
        preprocessed_dir: Directory containing preprocessed data
        
    Returns:
        Dictionary with summary information
    '''
    encoders = load_encoders(preprocessed_dir)
    matrix = load_interaction_matrix(preprocessed_dir)
    events = load_events(preprocessed_dir)
    
    summary = {
        'n_users': len(encoders['user_encoder']),
        'n_tracks': len(encoders['track_encoder']),
        'n_interactions': matrix.nnz,
        'sparsity': 100 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        'min_listens': matrix.data.min(),
        'max_listens': matrix.data.max(),
        'avg_listens': matrix.data.mean(),
        'total_listens': events['listen_count'].sum(),
        'matrix_shape': matrix.shape,
        'matrix_dtype': str(matrix.dtype),
    }
    
    return summary


__all__ = [
    'load_encoders',
    'load_interaction_matrix',
    'load_interaction_data',
    'load_events',
    'load_catalog',
    'load_items',
    'get_data_summary',
]

