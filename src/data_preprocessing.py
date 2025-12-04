'''
    Data Preprocessing

    This module provides functionality to preprocess raw data 
    into a format suitable for model training.

    Strategy:
    - Explode tracks to get all artists, albums, and genres for each track,
    - Clean names of entities,
    - Build entity catalogs (standardize names, deduplicate entities, create canonical IDs),
    - Create ID mappings (raw_id → canonical_id dictionaries),
    - Build items table (join all metadata),
    - Build tracks catalog (unique tracks with metadata),
    - Build events dataframe (filter outliers, map IDs, aggregate),
    - Create label encoders (user_id and track_id to sequential indices),
    - Not implemented, but left in case needed in the project part 2: 
    Create sparse interaction matrix (users × tracks).

    Input:
    - raw_dir - Directory with raw parquet files (tracks, catalog_names, interactions),
    - preprocessed_dir - Output directory for processed parquet files.

    Output:
    - items.parquet - Canonical music catalog with track_group_id for diversity,
    - tracks_catalog_clean.parquet - Track lookup table,
    - events.parquet - User-track interaction events,
    - ./models/label_encoders.pkl - User and track ID to index mappings (for model training),
    - Not implemented, but left in case needed in the project part 2: 
    interaction_matrix.npz - Sparse CSR matrix of user-track interactions.

    Usage:
    python -m src.data_preprocessing --build-items-df
    python -m src.data_preprocessing --build-tracks-catalog
    python -m src.data_preprocessing --build-events-df
    python -m src.data_preprocessing --create-label-encoders
    python -m src.data_preprocessing --create-sparse-interaction-matrix
'''

# ---------- Imports ---------- #
import os
import gc
import logging
import pickle
import argparse

import polars as pl
from scipy.sparse import csr_matrix, save_npz
from dotenv import load_dotenv

from src.s3_utils import upload_data_to_s3

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

# ---------- Data transformation helpers ---------- #
def standardize_text(col: str) -> pl.Expr:
    '''
        Basic text cleaning: lowercase, remove punctuation, trim.
    '''
    return (
        pl.col(col)
            .cast(pl.Utf8).fill_null('')
            .str.normalize('NFKD').str.replace_all(r'[\p{M}]', '')
            .str.to_lowercase()
            .str.replace_all(r'[^\w\s]', ' ')
            .str.replace_all(r'\s+', ' ')
            .str.strip_chars()
    )

def clean_name(col: str) -> pl.Expr:
    '''
        Clean entity names: remove brackets, feat., ft., featuring, versions.
    '''
    return (
        pl.col(col)
            .fill_null('')
            .str.replace_all(r'[\(\[][^\)\]]*[\)\]]', ' ')  # Remove brackets
            .str.replace_all(r'\b(feat\.?|ft\.?|featuring)\b.*', ' ')  # Remove feat
            .str.replace_all(r'\b(live|remix|extended|radio edit|acoustic|remastered(?:\s+\d{4})?)\b', ' ')
            .str.replace_all(r'[^\w\s]', ' ')
            .str.replace_all(r'\s+', ' ')
            .str.strip_chars()
            .str.to_titlecase()
    )

def normalize_title(col: str) -> pl.Expr:
    '''
        Track name normalization for grouping purposes
    '''
    return (
        pl.col(col)
            .fill_null('')
            .str.to_lowercase()
            .str.replace_all(r'[\(\[][^\)\]]*[\)\]]', ' ')
            .str.replace_all(r'\b(feat\.?|ft\.?|featuring)\b.*', ' ')
            .str.replace_all(r'\b(live|remix|extended|radio edit|acoustic|remastered(?:\s+\d{4})?)\b', ' ')
            .str.replace_all(r'\bcover(ed)?\b.*', ' ')
            .str.replace_all(r'[^\w\s]', ' ')
            .str.replace_all(r'\s+', ' ')
            .str.strip_chars()
    )

# ---------- Load raw data ---------- #
def load_raw_data(raw_dir: str):
    '''
        Load raw data.
    '''
    logger.info('Loading raw data from %s', raw_dir)
    
    tracks = pl.scan_parquet(f'{raw_dir}/tracks.parquet')
    catalog = pl.scan_parquet(f'{raw_dir}/catalog_names.parquet')
    interactions = pl.scan_parquet(f'{raw_dir}/interactions.parquet')

    return (
        tracks,
        catalog,
        interactions,
    )

# ---------- Main preprocessing pipeline ---------- #
def build_items_df(raw_dir: str, preprocessed_dir: str):
    '''
        Build items dataframe.
    '''

    # Load raw data
    logger.info('Loading tracks and catalog raw data from %s', raw_dir)
    tracks = pl.scan_parquet(f'{raw_dir}/tracks.parquet')
    catalog = pl.scan_parquet(f'{raw_dir}/catalog_names.parquet')

    # Create output directory
    logger.info('Creating output directory %s', preprocessed_dir)
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Explode tracks
    logger.info('Exploding tracks')
    tracks_exploded = (
        tracks
            .select(['track_id', 'albums', 'artists', 'genres'])
            .explode('albums')
            .explode('artists')
            .explode('genres')
            .rename({'albums': 'album_id', 'artists': 'artist_id', 'genres': 'genre_id'})
    )

    # Build entity catalogs
    logger.info('Building intermediate catalogs')
    def make_catalog(entity_type: str, id_col: str, name_col: str):
        '''
            Build entity catalog.
        '''
        return (
            catalog
                .filter(pl.col('type') == entity_type)
                .select([
                    pl.col('id').alias(id_col),
                    pl.col('name').alias(name_col),
                    standardize_text('name').alias(f'{name_col}_std')
                ])
        )
    
    tracks_cat = make_catalog('track', 'track_id', 'track_name')
    artists_cat = make_catalog('artist', 'artist_id', 'artist_name')
    albums_cat = make_catalog('album', 'album_id', 'album_name')
    genres_cat = make_catalog('genre', 'genre_id', 'genre_name')

    # Deduplicate entities
    logger.info('Deduplicating intermediate catalogs')
    
    # Artists
    artists_dedup = (
        artists_cat
            .collect()
            .group_by('artist_name_std')
            .agg([
                pl.min('artist_id').alias('canonical_id'),
                clean_name('artist_name_std').first().alias('clean_name')
            ])
    )
    artist_map = (
        artists_cat
            .collect()
            .join(artists_dedup.select(['artist_name_std', 'canonical_id']), on='artist_name_std')
            .select(['artist_id', pl.col('canonical_id').alias('artist_id_canonical')])
    )
    
    # Albums (per artist context)
    album_artists = (
        tracks_exploded
            .select(['album_id', 'artist_id'])
            .unique()
            .join(artist_map.lazy(), on='artist_id')
            .select(['album_id', 'artist_id_canonical'])
            .collect()
    )
    albums_dedup = (
        albums_cat
            .collect()
            .join(album_artists, on='album_id', how='left')
            .with_columns(pl.col('artist_id_canonical').fill_null(0))
            .group_by(['album_name_std', 'artist_id_canonical'])
            .agg([
                pl.min('album_id').alias('canonical_id'),
                clean_name('album_name_std').first().alias('clean_name'),
            ])
    )
    album_map = (
        albums_cat
            .collect()
            .join(album_artists, on='album_id', how='left')
            .with_columns(pl.col('artist_id_canonical').fill_null(0))
            .join(albums_dedup.select(['album_name_std', 'artist_id_canonical', 'canonical_id']), 
                on=['album_name_std', 'artist_id_canonical'])
            .select(['album_id', pl.col('canonical_id').alias('album_id_canonical')])
    )
    
    # Tracks (per artist context) + track grouping
    track_artists = (
        tracks_exploded
            .select(['track_id', 'artist_id'])
            .unique()
            .join(artist_map.lazy(), on='artist_id')
            .select(['track_id', 'artist_id_canonical'])
            .collect()
    )
    tracks_dedup = (
        tracks_cat
            .collect()
            .join(track_artists, on='track_id', how='left')
            .with_columns(pl.col('artist_id_canonical').fill_null(0))
            .group_by(['track_name_std', 'artist_id_canonical'])
            .agg([
                pl.min('track_id').alias('canonical_id'),
                clean_name('track_name_std').first().alias('clean_name'),
            ])
            .with_columns(normalize_title('track_name_std').alias('title_normalized')) # Add normalized title for grouping
    )
    
    # Create track groups
    track_groups = (
        tracks_dedup
            .group_by(['title_normalized', 'artist_id_canonical'])
            .agg(pl.min('canonical_id').alias('track_group_id'))
    )
    tracks_dedup = tracks_dedup.join(
        track_groups, 
        on=['title_normalized', 'artist_id_canonical'], 
        how='left'
    )
    
    track_map = (
        tracks_cat
            .collect()
            .join(track_artists, on='track_id', how='left')
            .with_columns(pl.col('artist_id_canonical').fill_null(0))
            .join(tracks_dedup.select(['track_name_std', 'artist_id_canonical', 'canonical_id']), 
                on=['track_name_std', 'artist_id_canonical'])
            .select(['track_id', pl.col('canonical_id').alias('track_id_canonical')])
    )
    
    # Genres
    genres_dedup = (
        genres_cat
            .collect()
            .group_by('genre_name_std')
            .agg([
                pl.min('genre_id').alias('canonical_id'),
                clean_name('genre_name_std').first().alias('clean_name')
            ])
    )
    genre_map = (
        genres_cat
            .collect()
            .join(genres_dedup.select(['genre_name_std', 'canonical_id']), on='genre_name_std')
            .select(['genre_id', pl.col('canonical_id').alias('genre_id_canonical')])
    )
   
    # Create mapping dicts
    # Helps to avoid memory issues when dealing with large datasets: 
    # creates small dicts instead of loading the whole exploded table into memory
    logger.info('Creating mapping dicts')
    id_mappings = {
        'track': dict(zip(track_map['track_id'], track_map['track_id_canonical'])),
        'artist': dict(zip(artist_map['artist_id'], artist_map['artist_id_canonical'])),
        'album': dict(zip(album_map['album_id'], album_map['album_id_canonical'])),
        'genre': dict(zip(genre_map['genre_id'], genre_map['genre_id_canonical'])),
    }
    
    # Build items table
    logger.info('Building items dataframe')
    # Map raw IDs to canonical IDs and join clean names and track_group_id
    items = (
        tracks_exploded
            .select(['track_id', 'artist_id', 'album_id', 'genre_id'])
            .with_columns([
                pl.col('track_id').replace(id_mappings['track'], default=None).cast(pl.Int64),
                pl.col('artist_id').replace(id_mappings['artist'], default=None).cast(pl.Int64),
                pl.col('album_id').replace(id_mappings['album'], default=None).cast(pl.Int64),
                pl.col('genre_id').replace(id_mappings['genre'], default=None).cast(pl.Int64)
            ])
            .drop_nulls()
            .join(
                tracks_dedup.lazy().select([
                    pl.col('canonical_id').alias('track_id'),
                    pl.col('clean_name').alias('track_clean'),
                    'track_group_id'
                ]),
                on='track_id', how='left'
            )
            .join(
                artists_dedup.lazy().select([
                    pl.col('canonical_id').alias('artist_id'),
                    pl.col('clean_name').alias('artist_clean')
                ]),
                on='artist_id', how='left'
            )
            .join(
                albums_dedup.lazy().select([
                    pl.col('canonical_id').alias('album_id'),
                    pl.col('clean_name').alias('album_clean')
                ]),
                on='album_id', how='left'
            )
            .join(
                genres_dedup.lazy().select([
                    pl.col('canonical_id').alias('genre_id'),
                    pl.col('clean_name').alias('genre_clean')
                ]),
                on='genre_id', how='left'
            )
            .with_columns([
                pl.col('track_clean').fill_null('Unknown'),
                pl.col('artist_clean').fill_null('Unknown'),
                pl.col('album_clean').fill_null('Unknown'),
                pl.col('genre_clean').fill_null('Unknown'),
                pl.when(pl.col('track_group_id').is_null()) # Fallback: if no group assigned, use track_id itself
                .then(pl.col('track_id'))
                .otherwise(pl.col('track_group_id'))
                .alias('track_group_id')
            ])
            .unique(subset=['track_id', 'artist_id', 'album_id', 'genre_id'])
    )

    # Save items dataframe
    items.sink_parquet(f'{preprocessed_dir}/items.parquet')
    logger.info('Succesfully done with items.parquet')

    # Remove all dataframes from memory
    del (tracks, catalog, 
        tracks_exploded, artists_cat, 
        albums_cat, genres_cat, 
        artists_dedup, album_artists, 
        albums_dedup, track_artists, 
        tracks_dedup, track_groups, 
        track_map, genres_dedup, 
        genre_map, id_mappings,
        items
    )
    # Collect garbage
    gc.collect()

    return None

# ---------- Build tracks catalog ---------- #
def build_tracks_catalog(preprocessed_dir: str):
    '''
        Build tracks catalog.
    '''

    logger.info('Building catalog')

    # Build tracks catalog
    tracks_catalog = (
        pl.scan_parquet(f'{preprocessed_dir}/items.parquet')
            .group_by('track_id')
            .agg([
                pl.first('track_clean'),
                pl.first('track_group_id'),
                pl.first('artist_id'),
                pl.first('album_id'),
                pl.first('genre_id')
            ])
    )

    # Save tracks catalog
    tracks_catalog.sink_parquet(f'{preprocessed_dir}/tracks_catalog_clean.parquet')
    logger.info('Succesfully done with tracks_catalog_clean.parquet')

    # Remove all dataframes from memory
    del (tracks_catalog)
    # Collect garbage
    gc.collect()

    return None

# ---------- Build events dataframe ---------- #
def build_events_df(raw_dir: str, preprocessed_dir: str):
    '''
        Build events dataframe.
    '''

    logger.info('Building events dataframe')

    # Load interactions
    logger.info('Loading interactions from %s', raw_dir)
    interactions = pl.scan_parquet(f'{raw_dir}/interactions.parquet')
    
    quantile = float(os.getenv('INTERACTION_QUANTILE', 0.95))
    threshold = interactions.select(pl.col('track_seq').quantile(quantile)).collect().item()
    logger.info(f'Filtering interactions at {quantile*100:.0f}th percentile: track_seq <= {threshold}')
    
    # Load valid track ids from already-created catalog
    valid_tracks = pl.scan_parquet(f'{preprocessed_dir}/tracks_catalog_clean.parquet').select('track_id')
    logger.info('Loaded valid track IDs from catalog')
    
    # Build events with semi-join to ensure all tracks exist in catalog
    # Note: No ID mapping needed here - tracks_catalog already has canonical IDs
    events = (
        interactions
            .select([
                'user_id',
                'track_id',
                'track_seq',
                pl.col('started_at').dt.date().alias('started_at'),
            ])
            .filter(pl.col('track_seq') <= threshold)
            # Semi-join: keep only tracks that exist in catalog
            .join(valid_tracks, on='track_id', how='semi')
            .group_by(['user_id', 'track_id'])
            .agg([
                pl.len().alias('listen_count'),
                pl.max('started_at').alias('last_listen')
            ])
    )

    # Save events dataframe
    events.sink_parquet(f'{preprocessed_dir}/events.parquet')
    logger.info('Succesfully done with events.parquet')

    # Remove all dataframes from memory
    del (events)
    # Collect garbage
    gc.collect()

    return None

# ---------- Create label encoders and sparse matrix ---------- #
def create_label_encoders(preprocessed_dir: str):
    '''
        Create label encoders and sparse interaction matrix.
    '''
    logger.info('Creating label encoders')
    
    # Load events
    logger.info('Loading events from %s', preprocessed_dir)
    events = pl.read_parquet(f'{preprocessed_dir}/events.parquet')

    # Create label encoders (user_id and track_id to sequential indices)
    unique_users = sorted(events['user_id'].unique().to_list())
    unique_tracks = sorted(events['track_id'].unique().to_list())
    
    user_encoder = {user_id: idx for idx, user_id in enumerate(unique_users)}
    track_encoder = {track_id: idx for idx, track_id in enumerate(unique_tracks)}
    
    # Create reverse mappings for decoding
    user_decoder = {idx: user_id for user_id, idx in user_encoder.items()}
    track_decoder = {idx: track_id for track_id, idx in track_encoder.items()}
    
    logger.info(f'Encoded {len(user_encoder):,} users and {len(track_encoder):,} tracks')
    
    # Save encoders
    encoders = {
        'user_encoder': user_encoder,
        'track_encoder': track_encoder,
        'user_decoder': user_decoder,
        'track_decoder': track_decoder,
    }
    
    # Save encoders to models directory
    models_dir = os.getenv('MODELS_DIR', './models')
    os.makedirs(models_dir, exist_ok=True)
    encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    logger.info(f'Saved label encoders to {encoders_path}')

    # Free up memory
    del (events, encoders, 
        user_encoder, track_encoder, 
        user_decoder, track_decoder
    )
    gc.collect()

    return None
    
#    Not implemented, but left in case needed in the project part 2
#    # ---------- Create sparse interaction matrix ---------- #
#    def create_sparse_interaction_matrix(preprocessed_dir: str):
#    '''
#        Create sparse interaction matrix.
#    '''
#    logger.info('Creating sparse interaction matrix')
#    
#    # Load events
#    logger.info('Loading events from %s', preprocessed_dir)
#    events = pl.read_parquet(f'{preprocessed_dir}/events.parquet')
#    encoders = load_encoders(preprocessed_dir)
#    
#    user_encoder = encoders['user_encoder']
#    track_encoder = encoders['track_encoder']
#    
#    n_users = len(user_encoder)
#    n_tracks = len(track_encoder)
#    
#    # Build sparse interaction matrix (users * tracks)
#    # Use listen_count as the interaction strength
#    events_encoded = (
#        events
#            .with_columns([
#                pl.col('user_id').replace(user_encoder).alias('user_idx'),
#                pl.col('track_id').replace(track_encoder).alias('track_idx'),
#            ])
#    #        .drop_nulls(['user_idx', 'track_idx'])
#    )
#    
#    row_indices = events_encoded['user_idx'].to_numpy()
#    col_indices = events_encoded['track_idx'].to_numpy()
#    data = events_encoded['listen_count'].to_numpy().astype(np.float32)
#    
#    n_users = len(user_encoder)
#    n_tracks = len(track_encoder)
#    
#    events_matrix = csr_matrix(
#        (data, (row_indices, col_indices)),
#        shape=(n_users, n_tracks),
#        dtype=np.float32
#    )
#    
#    logger.info(f'Created sparse matrix: {events_matrix.shape} with {events_matrix.nnz:,} non-zero entries')
#    logger.info(f'Sparsity: {100 * (1 - events_matrix.nnz / (n_users * n_tracks)):.4f}%')
#    
#    # Save sparse matrix
#    sparse_matrix_path = f'{preprocessed_dir}/events_matrix.npz'
#    save_npz(sparse_matrix_path, events_matrix)
#    logger.info(f'Saved events sparse matrix to {sparse_matrix_path}')
#    
#    # Free up memory
#    del (events_encoded, events_matrix, sparse_matrix_path)
#    gc.collect()
#
#    return None

# ---------- Data preprocessing pipeline ---------- #
def run_preprocessing(raw_dir: str, preprocessed_dir: str):
    '''
        Data preprocessing pipeline.
        Loads raw data and runs the full preprocessing pipeline.
    '''
    logger.info('Starting raw data preprocessing')
    
    build_items_df(raw_dir, preprocessed_dir)
    build_tracks_catalog(preprocessed_dir)
    build_events_df(raw_dir, preprocessed_dir)
    create_label_encoders(preprocessed_dir)
    
    # Upload to S3
    logger.info('Uploading data files to S3')
    upload_data_to_s3(f'{preprocessed_dir}/items.parquet', 'items.parquet')
    upload_data_to_s3(f'{preprocessed_dir}/events.parquet', 'events.parquet')
    
    logger.info('Succesfully done with data preprocessing')
    
    return None

# ---------- Main entry point ---------- #
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Data Preprocessing Pipeline')
    parser.add_argument('--build-items-df', action='store_true', help='Build items dataframe only')
    parser.add_argument('--build-tracks-catalog', action='store_true', help='Build tracks catalog only')
    parser.add_argument('--build-events-df', action='store_true', help='Build events dataframe only')
    parser.add_argument('--create-label-encoders', action='store_true', help='Create label encoders only')
    
    args = parser.parse_args()
    
    raw_dir = os.getenv('RAW_DATA_DIR', './data/raw')
    preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', './data/preprocessed')
    
    # Check if any individual step flag is set
    individual_steps = (
        args.build_items_df or 
        args.build_tracks_catalog or 
        args.build_events_df or 
        args.create_label_encoders
    )
    
    if individual_steps:
        # Run only specified steps
        if args.build_items_df:
            logger.info('Running: build_items_df')
            build_items_df(raw_dir, preprocessed_dir)
            
        if args.build_tracks_catalog:
            logger.info('Running: build_tracks_catalog')
            build_tracks_catalog(preprocessed_dir)
            
        if args.build_events_df:
            logger.info('Running: build_events_df')
            build_events_df(raw_dir, preprocessed_dir)
            
        if args.create_label_encoders:
            logger.info('Running: create_label_encoders')
            create_label_encoders(preprocessed_dir)
    else:
        # Run full pipeline
        logger.info('Running full preprocessing pipeline')
        run_preprocessing(raw_dir, preprocessed_dir)

# ---------- All exports ---------- #
__all__ = ['run_preprocessing']