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
    python -m src.similar_based_als --all-tracks
    python -m src.similar_based_als --track-id 1234567890
'''

# ---------- Imports ---------- #
import os
import gc
import logging
import argparse
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv

from src.collaborative_rec import load_als_model
from src.s3_utils import upload_recommendations_to_s3

# ---------- Load environment variables ---------- #
# Load from config/.env (relative to project root)
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))

# ---------- Logging setup ---------- #
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

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
        '''

        self.model = als_model.model
        self.track_encoder = als_model.track_encoder
        self.track_decoder = als_model.track_decoder
        logger.info(f'Loaded {len(self.track_decoder):,} tracks')
        
    def find_similar_to_one (self, track_id, n=10):
        '''
            Find n most similar tracks to a singlegiven track_id.
        '''

        logger.info(f'Finding similar tracks to {track_id}')

        # Check if track exists
        if track_id not in self.track_encoder:
            logger.warning(f'Track {track_id} not found')
            return []
        
        # Get track index
        track_idx = self.track_encoder[track_id]

        # Get similar items
        indices, scores = self.model.similar_items(track_idx, N=n+1)
        
        # Decode track indices to ids
        # Skip first (self) and decode
        return [
            (self.track_decoder.get(int(idx)), float(score))
            for idx, score in zip(indices[1:], scores[1:])
            if int(idx) in self.track_decoder
        ]
    
    def find_similar_to_all(self, top_k=None, batch_size=50000):
        '''
            Build full similar tracks index for all tracks using batched processing.
            
            Args:
                top_k: Number of similar tracks per item (default from env SIMILAR_TRACKS_TOP_K)
                batch_size: Number of tracks to process per batch (default 50K)
        '''

        # Get top k
        if top_k is None:
            top_k = int(os.getenv('SIMILAR_TRACKS_TOP_K', 10))
        
        logger.info(f'Building full index: top {top_k} per track (batch_size={batch_size:,})')
        
        # Get all indices
        all_indices = np.array(list(self.track_decoder.keys()))
        n_tracks = len(all_indices)
        n_batches = (n_tracks + batch_size - 1) // batch_size
        
        logger.info(f'Computing similar items for {n_tracks:,} tracks in {n_batches} batches')
        
        # Setup checkpoint directory for batch persistence
        results_dir = os.getenv('RESULTS_DIR', './results')
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
        logger.info('Merging all batches into final result...')
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

def get_similar_tracks(models_dir=None):
    '''
        Load ALS model and return ALSSimilarTracks instance.
    '''

    if models_dir is None:
        models_dir = os.getenv('MODELS_DIR', './models')
    
    als_model = load_als_model(f'{models_dir}/als_model.pkl')
    
    finder = ALSSimilarTracks()
    finder.fit(als_model)
    
    return finder

# ---------- Main entry point ---------- #
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Find similar tracks using ALS')
    parser.add_argument('--track-id', type=int, help='Find similar tracks for this track')
    parser.add_argument('--all-tracks', action='store_true', help='Build full index for all tracks')
    args = parser.parse_args()

    logger.info('Starting similar tracks finder')
    finder = get_similar_tracks()
    
    if args.all_tracks:
        finder.find_similar_to_all()
    elif args.track_id:
        similar = finder.find_similar_to_one(args.track_id)
        print(f'Similar tracks to {args.track_id}:')
        for track_id, score in similar:
            print(f'{track_id}: {score:.4f}')
    else:
        parser.print_help()
        print('\nExamples:')
        print('  python -m src.similar_based_als --all-tracks')
        print('  python -m src.similar_based_als --track-id 1234567890')
    
    logger.info('Similar tracks finder complete')

# ---------- All exports ---------- #
__all__ = ['ALSSimilarTracks', 'get_similar_tracks']