'''
    ALS Collaborative Filtering Recommender

    This module provides functionality to train and evaluate 
    ALS collaborative filtering model and generate recommendations.

    Input:
    - train_matrix.npz - Sparse user-track interaction matrix for training
    - ./models/label_encoders.pkl - User and track ID to index mappings

    Output:
    - als_model.pkl - Trained ALS model with encoders
    - personal_als.parquet - Personal recommendations for all users

    Usage:
    python3 -m src.als_model --train # train ALS model
    python3 -m src.als_model --recommend # generate recommendations for all users
    python3 -m src.als_model --user-id 12345 # generate recommendations for a specific user
'''

# ---------- Imports ---------- #
import os
import gc
import pickle
import argparse
from typing import List, Tuple

import numpy as np
import polars as pl
from scipy.sparse import load_npz
from implicit.als import AlternatingLeastSquares
from dotenv import load_dotenv

from src.logging_set_up import setup_logging
from src.s3_loading import upload_recommendations_to_s3

# ---------- Load environment variables ---------- #
# Load from config/.env (relative to project root)
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))

# ---------- Logging setup ---------- #
logger = setup_logging('als_model')

# ---------- ALS Recommender ---------- #
class ALSRecommender:
    '''
        ALS-based collaborative filtering recommender.

        Args:
        - factors - number of latent factors
        - regularization - regularization strength
        - iterations - number of iterations
        - alpha - confidence scaling factor
        - num_threads - number of threads to use

        Attributes:
        - model - trained ALS model
        - user_encoder - user ID to index mapping
        - track_encoder - track ID to index mapping
        - user_decoder - index to user ID mapping
        - track_decoder - index to track ID mapping
        - is_fitted - whether the model is fitted

        Methods:
        - fit - train ALS model on user-track interactions
        - recommend - get top-N recommendations for a single given user_id
        - generate_als_recommendations - generate recommendations for all users and save to parquet
        - save - save ALS model to file
    '''
    
    def __init__(self, factors=64, regularization=0.01, iterations=15, alpha=1.0, num_threads=0, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.num_threads = num_threads  # 0 means use all available cores
        self.seed = seed
        
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            num_threads=num_threads,
            random_state=seed
        )
        
        self.user_encoder = None
        self.track_encoder = None
        self.user_decoder = None
        self.track_decoder = None
        self.is_fitted = False
        
    def fit(self, train_matrix, user_encoder, track_encoder):
        '''
            Train ALS model on user-track interaction events.

            Args:
            - train_matrix - sparse user-track interaction matrix for training
            - user_encoder - user id to index mapping
            - track_encoder - track id to index mapping

            Returns:
            - None
        '''
        logger.info(f'Fitting ALS model: {train_matrix.shape}, {train_matrix.nnz:,} interaction events')
        logger.info(f'Fitting params: factors={self.factors}, reg={self.regularization}, iter={self.iterations}')
        
        # Store encoders
        self.user_encoder = user_encoder
        self.track_encoder = track_encoder
        self.user_decoder = {idx: uid for uid, idx in user_encoder.items()}
        self.track_decoder = {idx: tid for tid, idx in track_encoder.items()}
        
        # Apply confidence scaling: C = 1 + alpha * listen_count
        train_confidence = train_matrix.copy()
        train_confidence.data = 1 + self.alpha * train_confidence.data
        
        # Fit model (user × item matrix: rows=users, columns=items)
        self.model.fit(train_confidence, show_progress=True)
        self.is_fitted = True
        
        logger.info('DONE: ALS model training completed successfully')
        
    def recommend(
        self, 
        user_id: int=None, 
        user_items=None, 
        n_recs: int=None, 
        filter_already_liked: bool = True) -> List[Tuple[int, float]]:
        '''
            Get top-N recommendations for a single given user_id.

            Args:
            - user_id - user ID to get recommendations for
            - user_items - user-track interaction
            - n_recs - number of recommendations to return
            - filter_already_liked - whether to filter out already liked items

            Returns:
            - list of recommended track_ids and scores
        '''

        # Load defaults from environment if not provided
        if user_id is None:
            user_id = os.getenv('USER_ID', 1)
        if user_items is None:
            user_items = os.getenv('USER_ITEMS', None)
        logger.info(f'Getting top {n_recs} recommendations for user {user_id}')

        # Check if user exists
        if user_id not in self.user_encoder:
            logger.warning(f'User {user_id} not found in training data')
            return []
        
        user_idx = self.user_encoder[user_id]
        
        # Check if user index is within model bounds
        # Needed for cold start users, that may exist in test data but not be in the train data
        if user_idx >= self.model.user_factors.shape[0]:
            logger.warning(f'User {user_id} (idx={user_idx}) out of bounds for model')
            return []
        
        # Get recommendations
        track_indices, scores = self.model.recommend(
            user_idx,
            user_items[user_idx],
            N=n_recs,
            filter_already_liked_items=filter_already_liked
        )
        
        # Decode track indices to ids
        recommendations = [
            (self.track_decoder[idx], float(score))
            for idx, score in zip(track_indices, scores)
            if idx in self.track_decoder
        ]
        
        logger.info(f'DONE: Found {len(recommendations):,} recommendations for user {user_id}')
        
        return recommendations
    
    def generate_als_recommendations(
        self, 
        train_matrix, 
        results_dir: str=None, 
        n_recs: int=None,
        batch_size: int=50000,
        checkpoint_dir: str=None,
        resume: bool=True) -> None:
        '''
            Generate recommendations for all users and save to parquet.
            Uses batch processing with checkpointing for resumability.

            Args:
            - train_matrix - sparse user-track interaction matrix for training
            - results_dir - path to results directory
            - n_recs - number of recommendations to return
            - batch_size - number of users to process per batch (default 50,000)
            - checkpoint_dir - directory to store batch checkpoints (default: results_dir/als_checkpoints)
            - resume - whether to resume from existing checkpoints (default True)

            Returns:
            - None
        '''

        # Load defaults from environment if not provided
        if results_dir is None:
            results_dir = os.getenv('RESULTS_DIR', './results')
        if n_recs is None:
            n_recs = int(os.getenv('ALS_N_RECS', 10))
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(results_dir, 'als_checkpoints')

        output_path = os.path.join(results_dir, 'personal_als.parquet')
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Get all user indices as numpy array for batch processing
        all_user_indices = np.array(list(self.user_decoder.keys()))
        
        # Filter to valid indices within model bounds
        n_model_users = self.model.user_factors.shape[0]
        valid_mask = all_user_indices < n_model_users
        all_user_indices = all_user_indices[valid_mask]
        
        total_users = len(all_user_indices)
        n_batches = (total_users + batch_size - 1) // batch_size
        
        logger.info(f'Generating top {n_recs} recommendations for {total_users:,} users')
        logger.info(f'Processing in {n_batches} batches of {batch_size:,} users each')
        
        # Check for existing checkpoints if resuming
        completed_batches = set()
        if resume:
            for fname in os.listdir(checkpoint_dir):
                if fname.startswith('batch_') and fname.endswith('.parquet'):
                    try:
                        batch_num = int(fname.replace('batch_', '').replace('.parquet', ''))
                        completed_batches.add(batch_num)
                    except ValueError:
                        pass
            if completed_batches:
                logger.info(f'Found {len(completed_batches)} completed batches, resuming from batch {max(completed_batches) + 1}')
        
        # Process each batch
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, total_users)
            batch_user_indices = all_user_indices[batch_start:batch_end]
            
            checkpoint_path = os.path.join(checkpoint_dir, f'batch_{batch_idx}.parquet')
            
            # Skip if batch already completed
            if resume and batch_idx in completed_batches:
                logger.info(f'Skipping batch {batch_idx + 1}/{n_batches} (already completed)')
                continue
            
            # Progress logging
            progress_pct = (batch_start / total_users) * 100
            logger.info(f'Processing batch {batch_idx + 1}/{n_batches} (users {batch_start:,}-{batch_end:,}, {progress_pct:.1f}%)')
            
            # Generate recommendations for this batch
            track_indices_batch, scores_batch = self.model.recommend(
                batch_user_indices, 
                train_matrix[batch_user_indices], 
                N=n_recs, 
                filter_already_liked_items=True
            )
            
            # Build dataframe for this batch
            n_batch_users = len(batch_user_indices)
            n_recs_actual = track_indices_batch.shape[1]
            
            user_ids = np.array([self.user_decoder[idx] for idx in batch_user_indices])
            user_ids_repeated = np.repeat(user_ids, n_recs_actual)
            
            track_indices_flat = track_indices_batch.flatten()
            scores_flat = scores_batch.flatten()
            ranks = np.tile(np.arange(1, n_recs + 1), n_batch_users)
            
            track_decoder_arr = np.array([self.track_decoder.get(int(idx), -1) for idx in track_indices_flat])
            
            batch_df = pl.DataFrame({
                'user_id': user_ids_repeated,
                'track_id': track_decoder_arr,
                'score': scores_flat.astype(np.float32),
                'rank': ranks.astype(np.int8)
            })
            
            # Filter out invalid track_ids
            batch_df = batch_df.filter(pl.col('track_id') != -1)
            
            # Save batch checkpoint
            batch_df.write_parquet(checkpoint_path)
            
            logger.info(f'Batch {batch_idx + 1}/{n_batches} complete: {batch_df.height:,} recommendations saved')
            
            # Free batch memory
            del track_indices_batch, scores_batch, user_ids, user_ids_repeated
            del track_indices_flat, scores_flat, ranks, track_decoder_arr, batch_df
            gc.collect()
        
        # Merge all batch checkpoints into final output
        logger.info(f'Merging {n_batches} batch checkpoints into final output')
        
        batch_dfs = []
        for batch_idx in range(n_batches):
            checkpoint_path = os.path.join(checkpoint_dir, f'batch_{batch_idx}.parquet')
            if os.path.exists(checkpoint_path):
                batch_dfs.append(pl.read_parquet(checkpoint_path))
        
        if batch_dfs:
            als_df = pl.concat(batch_dfs)
            logger.info('DONE: ALS recommendations generated successfully')
            
            # Save final results
            als_df.write_parquet(output_path)
            logger.info(f'Saved final recommendations to {output_path}')
            
            # Upload to S3
            upload_recommendations_to_s3(output_path, 'personal_als.parquet')
            logger.info('Uploaded personal_als.parquet to S3')
            
            # Clean up checkpoint files after successful merge
            for batch_idx in range(n_batches):
                checkpoint_path = os.path.join(checkpoint_dir, f'batch_{batch_idx}.parquet')
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
            
            # Remove checkpoint directory if empty
            try:
                os.rmdir(checkpoint_dir)
                logger.info('Cleaned up checkpoint directory')
            except OSError:
                pass  # Directory not empty or other issue
            
            del als_df, batch_dfs
            gc.collect()
        else:
            logger.error('No batch checkpoints found to merge')

        return None
    
    def save(self, models_dir: str=None):
        '''
            Save ALS model to file.

            Args:
            - models_dir - path to models directory

            Returns:
            - None
        '''

        if models_dir is None:
            models_dir = os.getenv('MODELS_DIR', './models')
        als_model_path = os.path.join(models_dir, 'als_model.pkl')
        logger.info(f'ALS model path: {als_model_path}')

        # Save model to file
        with open(als_model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'user_encoder': self.user_encoder,
                'track_encoder': self.track_encoder,
                'user_decoder': self.user_decoder,
                'track_decoder': self.track_decoder,
                'factors': self.factors,
                'regularization': self.regularization,
                'iterations': self.iterations,
                'alpha': self.alpha,
                'num_threads': self.num_threads,
                'seed': self.seed,
            }, f)
        logger.info('DONE: ALS model saved successfully')

        return None

def load_als_model(models_dir: str=None):
    '''
        Load trained ALS model from file.

        Args:
        - models_dir - path to models directory

        Returns:
        - ALSRecommender instance
    '''

    if models_dir is None:
        models_dir = os.getenv('MODELS_DIR', './models')
    als_model_path = os.path.join(models_dir, 'als_model.pkl')
    logger.info(f'Loading ALS model from {als_model_path}')

    if not os.path.exists(als_model_path):
        logger.error(f'ALS model not found at {als_model_path}')
        raise FileNotFoundError(f'ALS model not found at {als_model_path}')

    with open(als_model_path, 'rb') as f:
        data = pickle.load(f)
    
    recommender = ALSRecommender(
        factors=data['factors'],
        regularization=data['regularization'],
        iterations=data['iterations'],
        alpha=data['alpha'],
        num_threads=data.get('num_threads', 0),
        seed=data.get('seed', 42)
    )
    
    recommender.model = data['model']
    recommender.user_encoder = data['user_encoder']
    recommender.track_encoder = data['track_encoder']
    recommender.user_decoder = data['user_decoder']
    recommender.track_decoder = data['track_decoder']
    recommender.is_fitted = True
    
    logger.info(f'Model loaded from {als_model_path}')

    return recommender


# ---------- Wrapper function for ALS recommendations---------- #
def als_recommendations(
    preprocessed_dir: str = None, 
    models_dir: str = None, 
    factors: int = None, 
    regularization: float = None, 
    iterations: int = None, 
    alpha: float = None, 
    num_threads: int = None, 
    n_recs: int = None,
    batch_size: int = None,
    resume: bool = True,
    seed: int = None
) -> None:
    '''
        Generate ALS recommendations for all users.

        Args:
        - preprocessed_dir - path to preprocessed data
        - factors - number of latent factors
        - regularization - regularization strength
        - iterations - number of iterations    
        - alpha - confidence scaling factor
        - num_threads - number of threads to use
        - n_recs - number of recommendations to generate
        - batch_size - number of users per batch for recommendation generation
        - resume - whether to resume from existing checkpoints
        - seed - random seed for reproducibility

        Returns:
        - None
    '''
    # Load defaults from environment if not provided
    if preprocessed_dir is None:
        preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
    if models_dir is None:
        models_dir = os.getenv('MODELS_DIR', './models')
    if factors is None:
        factors = int(os.getenv('ALS_FACTORS', 64))
    if regularization is None:
        regularization = float(os.getenv('ALS_REGULARIZATION', 0.01))
    if iterations is None:
        iterations = int(os.getenv('ALS_ITERATIONS', 15))
    if alpha is None:
        alpha = float(os.getenv('ALS_ALPHA', 1.0))
    if num_threads is None:
        num_threads = int(os.getenv('ALS_NUM_THREADS', 0))
    if n_recs is None:
        n_recs = int(os.getenv('ALS_N_RECS', 10))
    if batch_size is None:
        batch_size = int(os.getenv('ALS_BATCH_SIZE', 50000))
    if seed is None:
        seed = int(os.getenv('SEED', 42))
   
    logger.info('Starting ALS model training and recommendations')

    logger.info('Loading train matrix')
    train_path = f'{preprocessed_dir}/train_matrix.npz'
    train_matrix = load_npz(train_path)

    logger.info('Loading encoders')
    with open(f'{models_dir}/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    logger.info('Fitting ALS model')
    recommender = ALSRecommender(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
        num_threads=num_threads,
        seed=seed
    )
    recommender.fit(
        train_matrix,
        encoders['user_encoder'],
        encoders['track_encoder']
    )
    
    logger.info(f'Generating personal recommendations (n={n_recs}, batch_size={batch_size:,})')
    recommender.generate_als_recommendations(
        train_matrix, 
        n_recs=n_recs, 
        batch_size=batch_size,
        resume=resume
    )

    logger.info('Saving ALS model')
    recommender.save(models_dir)

    # Free memory
    del train_matrix, encoders, recommender
    gc.collect()

    logger.info('DONE: ALS model training and recommendations completed successfully')

    return None

# ---------- Main entry point ---------- #
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ALS Collaborative Filtering Recommender')
    parser.add_argument('--train', action='store_true', help='Train ALS model (includes evaluation and saving)')
    parser.add_argument('--recommend', action='store_true', help='Generate recommendations for all users')
    parser.add_argument('--user-id', type=int, help='Get recommendations for a specific user ID')
    parser.add_argument('--n-recs', type=int, default=10, help='Number of recommendations to return')
    parser.add_argument('--batch-size', type=int, default=50000, help='Number of users per batch (default: 50000)')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh instead of resuming from checkpoints')
    args = parser.parse_args()

    logger.info('Running ALS model training pipeline')

    # Check required environment variables
    required_env_vars = [
        'PREPROCESSED_DATA_DIR', 
        'RESULTS_DIR', 
        'MODELS_DIR', 
        'SEED',
        'ALS_FACTORS', 
        'ALS_REGULARIZATION', 
        'ALS_ITERATIONS', 
        'ALS_ALPHA', 
        'ALS_NUM_THREADS', 
        'ALS_N_RECS'
    ]

    missing_vars = [var for var in required_env_vars if os.getenv(var) is None]
    if missing_vars:
        logger.error(f'Missing required environment variables: {", ".join(missing_vars)}')
        raise EnvironmentError(f'Missing required environment variables: {", ".join(missing_vars)}')

    # Load config from environment
    preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
    models_dir = os.getenv('MODELS_DIR', './models')
    factors = int(os.getenv('ALS_FACTORS', 64))
    regularization = float(os.getenv('ALS_REGULARIZATION', 0.01))
    iterations = int(os.getenv('ALS_ITERATIONS', 15))
    alpha = float(os.getenv('ALS_ALPHA', 1.0))
    num_threads = int(os.getenv('ALS_NUM_THREADS', 0))  # 0 = use all cores
    n_recs = int(os.getenv('ALS_N_RECS', 10))

    if args.train:
        logger.info('Starting ALS model training and recommendations')
        als_recommendations(
            batch_size=args.batch_size,
            resume=not args.no_resume
        )
    
    elif args.recommend or args.user_id:
        # Load model and data        
        recommender = load_als_model(models_dir)
        train_matrix = load_npz(f'{preprocessed_dir}/train_matrix.npz')
        
        if args.user_id:
            # Get recommendations for a specific user
            recommendations = recommender.recommend(args.user_id, train_matrix, n_recs=args.n_recs)
            print(f'Top {args.n_recs} recommendations for user {args.user_id}:')
            for track_id, score in recommendations:
                print(f'Track {track_id}: {score:.4f}')
        else:
            # Generate recommendations for all users
            logger.info('Generating recommendations for all users')
            recommender.generate_als_recommendations(
                train_matrix, 
                n_recs=args.n_recs,
                batch_size=args.batch_size,
                resume=not args.no_resume
            )
    
    else:
        logger.error('Invalid command. Use --train, --recommend, or --user-id <user_id>')
        parser.print_help()
        print('Invalid command. Use --train, --recommend, or --user-id <user_id>')
    
    logger.info('DONE: ALS model training and recommendations completed successfully')

# ---------- All exports ---------- #
__all__ = ['ALSRecommender', 'load_als_model', 'als_recommendations']

