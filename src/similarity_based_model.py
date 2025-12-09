'''
    Similar Tracks Recommender

    This module provides functionality to find similar tracks using 
    ALS model's built-in similar_items method.

    Input:
    - als_model.pkl - trained ALS model
    - track_id - track id to find similar tracks for
    - n - number of similar tracks to return

    Output:
    - similar.parquet - similar tracks for all tracks
    - similar_tracks_index.pkl - similar tracks index for all tracks

    Usage:
    python -m src.similar_based_als --all-tracks # build full index for all tracks
    python -m src.similar_based_als --track-id 1234567890 # find similar tracks for a specific track
'''

# ---------- Imports ---------- #
import os
import gc
import logging
import argparse
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl
from dotenv import load_dotenv

from src.logging_set_up import setup_logging
from src.s3_loading import upload_recommendations_to_s3
from src.als_model import load_als_model

# ---------- Load environment variables ---------- #
# Load from config/.env (relative to project root)
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))

# ---------- Logging setup ---------- #
logger = setup_logging('similarity_based_model')

# ---------- Similar Tracks Finder using ALS model's built-in similar_items method ---------- #
class ALSSimilarTracks:
    '''
        Find similar tracks using ALS model's built-in similar_items.

        Args:
        - als_model - ALS model instance
        - track_id - track id to find similar tracks for
        - n - number of similar tracks to return

        Attributes:
        - model - ALS model instance
        - track_encoder - track id to index mapping
        - track_decoder - index to track id mapping
        - is_fitted - whether the model is fitted

        Methods:
        - fit - load from trained ALS model
        - find_similar_to_one - find n most similar tracks to a single given track_id
        - find_similar_to_all - find n most similar tracks to all tracks at a time
    '''
    
    def __init__(self):
        self.model = None
        self.track_encoder = None
        self.track_decoder = None
        
    def fit(self, als_model):
        '''
            Load from trained ALS model.

            Args:
            - als_model - ALS model instance

            Returns:
            - None
        '''

        self.model = als_model.model
        self.track_encoder = als_model.track_encoder
        self.track_decoder = als_model.track_decoder
        logger.info(f'Loaded {len(self.track_decoder):,} tracks')
        
    def recommend(self, track_id: int, n_recs: int=None) -> List[Tuple[int, float]]:
        '''
            Recommend n most similar tracks to a single given track_id.

            Args:
            - track_id - track id to recommend similar tracks for
            - n_recs - number of similar tracks to return

            Returns:
            - list of similar track_ids and scores
        '''

        # Load defaults from environment if not provided
        if n_recs is None:
            n_recs = int(os.getenv('SIMILARITY_N_RECS', 10))

        logger.info(f'Finding similar tracks to {track_id}')

        # Check if track exists
        if track_id not in self.track_encoder:
            logger.warning(f'Track {track_id} not found')
            return []
        
        # Get track index
        track_idx = self.track_encoder[track_id]

        # Get similar items
        indices, scores = self.model.similar_items(track_idx, N=n_recs+1)
        
        # Decode track indices to ids
        # Skip first (self) and decode
        return [
            (self.track_decoder.get(int(idx)), float(score))
            for idx, score in zip(indices[1:], scores[1:])
            if int(idx) in self.track_decoder
        ]
    
    def generate_similarity_recommendations(
        self, 
        top_k: int = None, 
        batch_size: int = None, 
        results_dir: str = None
        ) -> None:
        '''
            Generate similarity recommendations for all tracks and save to parquet.
            Uses batch processing for better performance.

            Args:
            - top_k - number of similar tracks per item
            - batch_size - number of tracks to process per batch
            - results_dir - path to results directory

            Returns:
            - None
        '''
        # Load defaults from environment if not provided
        if top_k is None:
            top_k = int(os.getenv('SIMILARITY_TOP_K', 10))
        if batch_size is None:
            batch_size = int(os.getenv('SIMILARITY_BATCH_SIZE', 50000))
        if results_dir is None:
            results_dir = os.getenv('RESULTS_DIR', './results')
        
        logger.info(f'Building similarity indexes: top {top_k} per track (batch_size={batch_size:,})')
        
        # Get all indices
        all_indices = np.array(list(self.track_decoder.keys()))
        n_tracks = len(all_indices)
        n_batches = (n_tracks + batch_size - 1) // batch_size
        
        logger.info(f'Computing similar items for {n_tracks:,} tracks in {n_batches} batches')
        
        # Setup checkpoint directory for batch persistence
        checkpoint_dir = Path(results_dir) / 'similar_batches_checkpoint'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing batches to resume from
        completed_batches = set()
        for batch_file in checkpoint_dir.glob('batch_*.pkl'):
            batch_num = int(batch_file.stem.split('_')[1])
            completed_batches.add(batch_num)
        
        if completed_batches:
            logger.info(f'Found {len(completed_batches)} completed batches, resuming from last checkpoint')
        
        # Finding similar tracks for all tracks
        # Process in batches with progress logging in order to reduce memory usage
        for batch_idx in range(n_batches):
            # Skip already completed batches
            if batch_idx in completed_batches:
                logger.info(f'Skipping batch {batch_idx + 1}/{n_batches} (already completed)')
                continue
                
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_tracks)
            batch_indices = all_indices[start_idx:end_idx]
            
            progress_pct = (batch_idx + 1) / n_batches * 100
            logger.info(f'Processing batch {batch_idx + 1}/{n_batches} '
                       f'(tracks {start_idx:,}-{end_idx:,}, {progress_pct:.1f}%)')
            
            # Get similar items for this batch
            indices_batch, scores_batch = self.model.similar_items(batch_indices, N=top_k+1)
            
            # Build index dict for this batch
            batch_similar = {}
            for i, track_idx in enumerate(batch_indices):
                track_id = self.track_decoder[track_idx]
                batch_similar[str(track_id)] = [
                    (self.track_decoder.get(int(idx)), float(score))
                    for idx, score in zip(indices_batch[i, 1:], scores_batch[i, 1:])
                    if int(idx) in self.track_decoder
                ]
            
            # Save batch results to checkpoint
            batch_file = checkpoint_dir / f'batch_{batch_idx}.pkl'
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_similar, f)
            logger.info(f'Saved batch {batch_idx + 1} to checkpoint ({len(batch_similar):,} tracks)')
            
            # Free batch memory
            del indices_batch, scores_batch, batch_similar
            gc.collect()
        
        logger.info(f'Completed processing all {n_batches} batches')
        
        # Merge all batches into final result
        logger.info('Merging all batches')
        similar = {}
        for batch_idx in range(n_batches):
            batch_file = checkpoint_dir / f'batch_{batch_idx}.pkl'
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
                similar.update(batch_data)
                del batch_data
        
        logger.info(f'Merged {len(similar):,} tracks from {n_batches} batches')
        
        # Convert nested dictionary to flat table format for Parquet
        logger.info('Converting to flat table format')
        rows = []
        for track_id, similar_tracks in similar.items():
            for rank, (similar_track_id, score) in enumerate(similar_tracks, start=1):
                rows.append({
                    'track_id': int(track_id),
                    'similar_track_id': int(similar_track_id),
                    'similarity_score': float(score),
                    'rank': rank
                })
        
        # Create DataFrame and save
        output_path = os.path.join(results_dir, 'similar.parquet')
        os.makedirs(results_dir, exist_ok=True)
        
        similar_df = pl.DataFrame(rows)
        similar_df.write_parquet(output_path)
        
        logger.info(f'Saved flat index with {len(rows):,} similarity pairs to {output_path}')
        
        index_path = os.path.join(results_dir, 'similar_tracks_index.pkl')
        with open(index_path, 'wb') as f:
            pickle.dump(similar, f)
        logger.info(f'Saved dictionary index for {len(similar):,} tracks to {index_path}')

        # Upload to S3
        upload_recommendations_to_s3(output_path, 'similar.parquet')
        upload_recommendations_to_s3(index_path, 'similar_tracks_index.pkl')
        
        # Clean up checkpoint directory
        logger.info('Cleaning up checkpoint files')
        for batch_file in checkpoint_dir.glob('batch_*.pkl'):
            batch_file.unlink()
        checkpoint_dir.rmdir()
        logger.info('Checkpoint files cleaned up')

        # Free memory
        del similar
        gc.collect()

        return None

# ---------- Wrapper function for similarity recommendations---------- #
def similarity_based_recommendations() -> None:
    '''
        Generate similarity recommendations for all tracks.

        Args:
        - models_dir - path to models directory

        Returns:
        - None
    '''

    als_model = load_als_model()
    finder = ALSSimilarTracks()
    finder.fit(als_model)
    finder.generate_similarity_recommendations()

    return None

# ---------- Main entry point ---------- #
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Find similar tracks using ALS')
    parser.add_argument('--track-id', type=int, help='Find similar tracks for this track')
    parser.add_argument('--all-tracks', action='store_true', help='Build full index for all tracks')
    args = parser.parse_args()

    logger.info('Running similarity-based model pipeline')

    # Check required environment variables
    required_env_vars = [
        'PREPROCESSED_DATA_DIR', 
        'RESULTS_DIR', 
        'MODELS_DIR', 
        'SIMILARITY_TOP_K', 
        'SIMILARITY_BATCH_SIZE'
    ]

    missing_vars = [var for var in required_env_vars if os.getenv(var) is None]
    if missing_vars:
        logger.error(f'Missing required environment variables: {", ".join(missing_vars)}')
        raise EnvironmentError(f'Missing required environment variables: {", ".join(missing_vars)}')

    # Load config from environment
    models_dir = os.getenv('MODELS_DIR', './models')
    top_k = int(os.getenv('SIMILARITY_TOP_K', 10))
    batch_size = int(os.getenv('SIMILARITY_BATCH_SIZE', 50000))
    n_recs = int(os.getenv('SIMILARITY_N_RECS', 10))

    similarity_based_recommendations()
    
    logger.info('Similarity-based model pipeline completed')

# ---------- All exports ---------- #
__all__ = ['ALSSimilarTracks', 'similarity_based_recommendations']