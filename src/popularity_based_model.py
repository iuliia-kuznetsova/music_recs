'''
    Populararity-based Recommender

    This module provides functionality to find and recommend the most popular tracks.

    Popularity can be measured by:
    - Total listen count
    - Number of unique users
    - Average listens per user

    Input:
    - events.parquet - user-track interaction events
    - tracks_catalog_clean.parquet - tracks catalog with clean metadata

    Output:
    - top_popular_tracks.parquet - top N popular tracks
    - popularity_track_scores.parquet - popularity scores for all tracks

    Usage examples:
    python3 -m src.popularity_based_rec # generate top N popularity-based recommendations
    python3 -m src.popularity_based_rec --user-id 1234567890 # get recommendations for a specific user ID
    python3 -m src.popularity_based_rec --n 100 # number of top popular tracks to return (default: 100)
    python3 -m src.popularity_based_rec--n-recs 10 # number of recommendations to return for each user (default: 10)
    python3 -m src.popularity_based_rec--method listen_count # method to compute track popularity (listen_count, user_count, avg_listens)
    python3 -m src.popularity_based_rec--with-metadata True # whether to add metadata to the top popular tracks (default: True)
    python3 -m src.popularity_based_rec--filter-listened True # whether to filter out tracks the user has already listened to (default: True)
'''

# ---------- Imports ---------- #
import os
import gc
import logging
import argparse
from typing import List

import polars as pl
from dotenv import load_dotenv

from src.logging_set_up import setup_logging
from src.s3_loading import upload_recommendations_to_s3

# ---------- Load environment variables ---------- #
# Load from config/.env (relative to project root)
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))

# ---------- Logging setup ---------- #
logger = setup_logging('popularity_based_model')

# ---------- Popularity-based Recommender ---------- #
class PopularityRecommender:
    '''
        Simple popularity-based recommender.
        
        Recommends the most popular tracks that the user hasn't interacted with yet.
        Compute track popularity from events based on the method provided.
            
        Args:
        - method - method to compute track popularity (listen_count, user_count, avg_listens)

        Attributes:
        - top_tracks - DataFrame with columns [track_id, popularity_score]

        Methods:
        - fit - compute track popularity from events based on the method provided
        - recommend - recommend top N popular tracks that the user hasn't listened to
        - generate_popularity_recommendations - generate top N popularity-based recommendations
    '''

    def __init__(self):
        self.top_tracks = None
        
    def fit(
        self, 
        preprocessed_dir: str=None, 
        method: str=None, 
        with_metadata: bool=None, 
        n: int=None
    ) -> None:
        '''
            Compute track popularity from events based on the method provided.
            
            Args:
            - preprocessed_dir - path to preprocessed directory,
            - method - method to compute track popularity (listen_count, user_count, avg_listens),
            - with_metadata - whether to add metadata to the top tracks.
            - n - number of top popular tracks to return

            Returns:
            - None
        '''

        # Load defaults from environment if not provided
        if preprocessed_dir is None:
            preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
        if method is None:
            method = os.getenv('POPULARITY_METHOD', 'listen_count')
        if with_metadata is None:
            with_metadata = os.getenv('POPULARITY_WITH_METADATA', True)
        if n is None:
            n = int(os.getenv('POPULARITY_TOP_N', 100))

        # Load events
        try:
            logger.info('Loading events from %s', preprocessed_dir)
            events = pl.scan_parquet(f'{preprocessed_dir}/events.parquet')
        except FileNotFoundError:
            raise FileNotFoundError(f'Events not found at {preprocessed_dir}/events.parquet')

        # Compute track popularity using the method provided
        logger.info('Computing track popularity using method: %s', method)        
        if method == 'listen_count':
            # Total listens per track
            popularity = (
                events
                    .group_by('track_id')
                    .agg(pl.sum('listen_count').alias('popularity_score'))
            )
        elif method == 'user_count':
            # Number of unique users per track
            popularity = (
                events
                    .group_by('track_id')
                    .agg(pl.col('user_id').n_unique().alias('popularity_score'))
            )
        elif method == 'avg_listens':
            # Average listens per user (for users who listened)
            popularity = (
                events
                    .group_by('track_id')
                    .agg([
                        pl.sum('listen_count').alias('total_listens'),
                        pl.col('user_id').n_unique().alias('n_users')
                    ])
                    .with_columns(
                        (pl.col('total_listens') / pl.col('n_users')).alias('popularity_score')
                    )
                    .select(['track_id', 'popularity_score'])
            )
        else:
            raise ValueError(f'Unknown method: {method}')
        
        # Sort by popularity and collect
        popularity_track_scores = (
            popularity
                .sort('popularity_score', descending=True)
                .collect()
                .select(['track_id', 'popularity_score'])
        )
        
        logger.info(f'DONE: Computed popularity for {popularity_track_scores.height:,} tracks')
        logger.info(f'Top track score: {popularity_track_scores["popularity_score"][0]:.2f}')

        # Save locally
        results_dir = os.getenv('RESULTS_DIR', './results')
        os.makedirs(results_dir, exist_ok=True)
        popularity_track_scores_path = os.path.join(results_dir, 'popularity_track_scores.parquet')
        popularity_track_scores.write_parquet(popularity_track_scores_path)
        logger.info('Results of popularity-based recommendations saved to %s', popularity_track_scores_path)

        # Free up memory
        del (events, popularity)
        gc.collect()
       
       # Add metadata to top tracks
        if with_metadata:
            logger.info('Loading catalog from %s', preprocessed_dir)
            catalog = pl.read_parquet(f'{preprocessed_dir}/tracks_catalog_clean.parquet')
            self.top_tracks = (
                popularity_track_scores
                    .head(n)
                    .join(
                        catalog.select(['track_id', 'track_clean', 'artist_id', 'track_group_id']),
                        on='track_id',
                        how='left'
                    )
                    .select(['track_id', 'popularity_score', 'track_clean', 'artist_id', 'track_group_id'])
            )
            logger.info(f'Added metadata to {self.top_tracks.height:,} tracks')
            # Free catalog memory
            del catalog
        else:
            self.top_tracks = popularity_track_scores.head(n)
            logger.info(f'No metadata added, {self.top_tracks.height:,} tracks returned')

        # Save locally
        top_popular_path = os.path.join(results_dir, 'top_popular.parquet')
        self.top_tracks.write_parquet(top_popular_path)
        logger.info('Results of popularity-based recommendations saved to %s', top_popular_path)

        # Upload to S3
        upload_recommendations_to_s3(top_popular_path, 'top_popular.parquet')
        logger.info(f'Uploaded top_popular.parquet to S3')

        # Free up memory
        del popularity_track_scores
        gc.collect()

        return self.top_tracks
    
    # ---------- Recommend top popular tracks for a user ---------- #
    def recommend(
        self, 
        user_id: int, 
        n_recs: int = None,
        user_listened: set = None,
        preprocessed_dir: str = None
    ) -> List[int]:
        '''
            Recommend top popular tracks for a single given user_id 
            that the user hasn't listened to.
            
            Args:
            - user_id - user ID to generate recommendations for.
            - n_recs - number of recommendations to return.
            - user_listened - set of track_ids the user has listened to (optional, for efficiency).
            - preprocessed_dir - path to load events from if user_listened not provided.

            Returns:
            - list of recommended track_ids
        '''

        # Load defaults from environment if not provided
        if preprocessed_dir is None:
            preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
        if n_recs is None:
            n_recs = int(os.getenv('POPULARITY_N_RECS', 10))
        if self.top_tracks is None:
            raise ValueError('Model not fitted.')

        # Get user's listened tracks
        if user_listened is not None:
            # Use provided listening history (efficient for batch processing)
            listened_tracks = user_listened
        elif preprocessed_dir is not None:
            # Load from file (fallback for single user)
            events = pl.scan_parquet(f'{preprocessed_dir}/events.parquet')
            listened_tracks = set(
                events
                    .filter(pl.col('user_id') == user_id)
                    .select('track_id')
                    .collect()['track_id'].to_list()
            )
        else:
            listened_tracks = set()
        
        # Get top popular tracks that the user hasn't listened to
        recommendations = (
            self.top_tracks
                .filter(~pl.col('track_id').is_in(list(listened_tracks)))
                .head(n_recs)
        )
        
        return recommendations['track_id'].to_list()

# ---------- Wrapper function for popularity-based recommendations---------- #
def popularity_based_recommendations(
    preprocessed_dir: str = None, 
    n: int = None,
    method: str = None,
    with_metadata: bool = None
) -> None:
    '''
        Generate popularity-based recommendations for all users.
        
        Args:
        - preprocessed_dir - path to preprocessed data directory (optional, uses env var if not provided)
        - n - number of top popular tracks to return
            
        Returns:
        - list of track_ids sorted by popularity
    '''

    # Load defaults from environment if not provided
    if preprocessed_dir is None:
        preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
    if n is None:
        n = int(os.getenv('POPULARITY_TOP_N', 100))
    if method is None:
        method = os.getenv('POPULARITY_METHOD', 'listen_count')
    if with_metadata is None:
        with_metadata = os.getenv('POPULARITY_WITH_METADATA', True)
    
    logger.info('Starting popularity-based model training and recommendations')
    logger.info(f'Computing top {n} popularity recommendations')

    recommender = PopularityRecommender()
    recommender.fit(preprocessed_dir=preprocessed_dir, method=method, with_metadata=with_metadata, n=n)
    
    # Free memory
    del recommender
    gc.collect()

    logger.info('DONE: Popularity-based model training and recommendations completed successfully')
    return None

# ---------- Main entry point ---------- #
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Popularity-based Recommender')
    parser.add_argument('--user-id', type=int, help='Get recommendations for a specific user ID')
    parser.add_argument('--n', type=int, default=100, help='Number of top popular tracks to return (default: 100)')
    parser.add_argument('--n-recs', type=int, default=10, help='Number of recommendations (default: 10)')
    parser.add_argument('--method', type=str, default=None, 
                        help='Popularity method: listen_count, user_count, avg_listens')
    parser.add_argument('--filter-listened', action='store_true', default=False, help='Filter out tracks the user has already listened to')
    parser.add_argument('--with-metadata', action='store_true', default=False, help='Add metadata to the top popular tracks')
    args = parser.parse_args()

    logger.info('Running popularity-based model training and recommendations')

    # Check required environment variables
    required_env_vars = [
        'PREPROCESSED_DATA_DIR', 
        'RESULTS_DIR', 
        'POPULARITY_METHOD', 
        'POPULARITY_TOP_N', 
        'POPULARITY_N_RECS', 
    ]

    missing_vars = [var for var in required_env_vars if os.getenv(var) is None]
    if missing_vars:
        logger.error(f'Missing required environment variables: {", ".join(missing_vars)}')
        raise EnvironmentError(f'Missing required environment variables: {", ".join(missing_vars)}')

    # Load config from environment
    results_dir = os.getenv('RESULTS_DIR', './results')
    preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
    user_id = args.user_id or int(os.getenv('USER_ID', None))
    n = args.n or int(os.getenv('POPULARITY_TOP_N', 100))
    n_recs = args.n_recs or int(os.getenv('POPULARITY_N_RECS', 10))
    method = args.method or str(os.getenv('POPULARITY_METHOD', 'listen_count'))
    filter_listened = args.filter_listened or bool(os.getenv('POPULARITY_FILTER_LISTENED', True))
    with_metadata = args.with_metadata or bool(os.getenv('POPULARITY_WITH_METADATA', True))

    if user_id:
        logger.info(f'Getting recommendations for user {user_id}')
        recommender = PopularityRecommender()
        recommender.fit(
            preprocessed_dir=preprocessed_dir, 
            method=method, 
            with_metadata=with_metadata, 
            n=n
        )
        recommendations = recommender.recommend(
            user_id=user_id,
            n_recs=n_recs,
            preprocessed_dir=preprocessed_dir if filter_listened else None
        )
        print(f'Top {n_recs} popular tracks for user {user_id}:')
        for i, track_id in enumerate(recommendations, 1):
            print(f'  {i}. Track {track_id}')
    else:
        logger.info(f'Generating popularity top {n} tracks recommendations')
        recommender = PopularityRecommender()
        recommender.fit(
            preprocessed_dir=preprocessed_dir, 
            method=method, 
            with_metadata=with_metadata, 
            n=n
        )
        logger.info(f'Saved top {n} popular tracks')

    logger.info('DONE: Popularity-based model training and recommendations completed successfully')

# ---------- All exports ---------- #
__all__ = ['PopularityRecommender', 'popularity_based_recommendations']

