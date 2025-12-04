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
    python -m src.collaborative_rec --train
    python -m src.collaborative_rec --recommend
    python -m src.collaborative_rec --user-id 12345 --n-recs 20
'''

# ---------- Imports ---------- #
import os
import gc
import pickle
import logging
import argparse

import numpy as np
import polars as pl
from scipy.sparse import load_npz
from implicit.als import AlternatingLeastSquares
from dotenv import load_dotenv

from src.s3_utils import upload_recommendations_to_s3

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
        - evaluate - evaluate precision@k on test set
        - generate_recommendations - generate recommendations for all users and save to parquet
        - save - save ALS model to file
    '''
    
    def __init__(self, factors=64, regularization=0.01, iterations=15, alpha=1.0, num_threads=0):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.num_threads = num_threads  # 0 means use all available cores
        
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            num_threads=num_threads,
            random_state=42
        )
        
        self.user_encoder = None
        self.track_encoder = None
        self.user_decoder = None
        self.track_decoder = None
        self.is_fitted = False
        
    def fit(self, train_matrix, user_encoder, track_encoder):
        '''
            Train ALS model on user-track interactions.
        '''
        logger.info(f'Training ALS model: {train_matrix.shape}, {train_matrix.nnz:,} interactions')
        logger.info(f'Params: factors={self.factors}, reg={self.regularization}, iter={self.iterations}')
        
        # Store encoders
        self.user_encoder = user_encoder
        self.track_encoder = track_encoder
        self.user_decoder = {idx: uid for uid, idx in user_encoder.items()}
        self.track_decoder = {idx: tid for tid, idx in track_encoder.items()}
        
        # Apply confidence scaling: C = 1 + alpha * listen_count
        train_confidence = train_matrix.copy()
        train_confidence.data = 1 + self.alpha * train_confidence.data
        
        # Fit model (implicit expects items × users, so transpose)
        self.model.fit(train_confidence.T, show_progress=True)
        self.is_fitted = True
        
        logger.info('Model training complete')
        
    def recommend(self, user_id, user_items, n=10, filter_already_liked=True):
        '''
            Get top-N recommendations for a single given user_id.
        '''

        logger.info(f'Getting top {n} recommendations for user {user_id}')

        # Check if user exists
        if user_id not in self.user_encoder:
            logger.warning(f'User {user_id} not found in training data')
            return []
        
        user_idx = self.user_encoder[user_id]
        
        # Get recommendations
        track_indices, scores = self.model.recommend(
            user_idx,
            user_items[user_idx],
            N=n,
            filter_already_liked_items=filter_already_liked
        )
        
        # Decode track indices to ids
        recommendations = [
            (self.track_decoder[idx], float(score))
            for idx, score in zip(track_indices, scores)
            if idx in self.track_decoder
        ]
        
        logger.info(f'Found {len(recommendations):,} recommendations for user {user_id}')
        
        return recommendations
    
    def evaluate(self, train_matrix, test_matrix, k=10, sample_users=10000):
        '''
        Evaluate precision@k on test set.
        '''
        logger.info(f'Evaluating precision@{k} on {sample_users} users')
        
        # Get users with test interactions
        test_user_indices = np.unique(test_matrix.tocoo().row)
        
        # Filter to users within model bounds (users that were in training)
        n_users = self.model.user_factors.shape[0]
        test_user_indices = test_user_indices[test_user_indices < n_users]
        logger.info(f'Users with test interactions within model bounds: {len(test_user_indices)}')
        
        # Sample users for speed
        if sample_users and sample_users < len(test_user_indices):
            np.random.seed(42)
            test_user_indices = np.random.choice(test_user_indices, sample_users, replace=False)
        
        hits = 0
        total = 0
        
        for user_idx in test_user_indices:
            try:
                # Get recommendations
                rec_indices, _ = self.model.recommend(
                    user_idx, train_matrix[user_idx], N=k, filter_already_liked_items=True
                )
                
                # Get test items
                test_items = set(test_matrix[user_idx].tocoo().col)
                
                if test_items:
                    hits += len(set(rec_indices) & test_items)
                    total += k
            except IndexError:
                continue
        
        precision = hits / total if total > 0 else 0.0
        logger.info(f'Precision@{k}: {precision:.4f}')
        
        return {'precision': precision, 'users_evaluated': len(test_user_indices)}
    
    def generate_recommendations(self, train_matrix, n=10):
        '''
            Generate recommendations for all users and save to parquet.
            Uses batch processing for better performance.
        '''

        results_dir = os.getenv('RESULTS_DIR', './results')
        output_path = os.path.join(results_dir, 'personal_als.parquet')
        
        # Get all user indices as numpy array for batch processing
        all_user_indices = np.array(list(self.user_decoder.keys()))
        
        # Filter to valid indices within model bounds
        n_model_users = self.model.user_factors.shape[0]
        valid_mask = all_user_indices < n_model_users
        all_user_indices = all_user_indices[valid_mask]
        
        logger.info(f'Generating top {n} recommendations for {len(all_user_indices):,} users (batch mode)')
        
        # Use batch recommend for much faster processing
        track_indices_batch, scores_batch = self.model.recommend(
            all_user_indices, 
            train_matrix[all_user_indices], 
            N=n, 
            filter_already_liked_items=True
        )
        
        logger.info('Building recommendations dataframe (vectorized)')
        
        n_users = len(all_user_indices)
        n_recs = track_indices_batch.shape[1]
        
        # Create user_id array (repeat each user_id n times)
        user_ids = np.array([self.user_decoder[idx] for idx in all_user_indices])
        user_ids_repeated = np.repeat(user_ids, n_recs)
        
        # Flatten track indices and scores
        track_indices_flat = track_indices_batch.flatten()
        scores_flat = scores_batch.flatten()
        
        # Create rank array (1, 2, ..., n repeated for each user)
        ranks = np.tile(np.arange(1, n_recs + 1), n_users)
        
        # Decode track indices to track_ids using vectorized lookup
        track_decoder_arr = np.array([self.track_decoder.get(int(idx), -1) for idx in track_indices_flat])
        
        # Create DataFrame directly from arrays
        als_df = pl.DataFrame({
            'user_id': user_ids_repeated,
            'track_id': track_decoder_arr,
            'score': scores_flat.astype(np.float32),
            'rank': ranks.astype(np.int8)
        })
        
        # Filter out invalid track_ids (where decoder returned -1)
        als_df = als_df.filter(pl.col('track_id') != -1)
        
        logger.info(f'Generated {als_df.height:,} recommendations')
        
        # Save results
        os.makedirs(results_dir, exist_ok=True)
        als_df.write_parquet(output_path)
        
        logger.info(f'Saved {als_df.height:,} recommendations to {output_path}')

        # Upload to S3
        upload_recommendations_to_s3(output_path, 'personal_als.parquet')

        # Free memory
        del als_df, track_indices_batch, scores_batch, user_ids, user_ids_repeated
        del track_indices_flat, scores_flat, ranks, track_decoder_arr
        gc.collect()

        return None
    
    def save(self, filepath):
        '''
            Save ALS model to file.
        '''

        logger.info(f'Saving ALS model to {filepath}')
        with open(filepath, 'wb') as f:
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
            }, f)
        logger.info(f'Model saved to {filepath}')


def load_als_model(filepath):
    '''
        Load trained ALS model from file.
    '''

    logger.info(f'Loading ALS model from {filepath}')
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    recommender = ALSRecommender(
        factors=data['factors'],
        regularization=data['regularization'],
        iterations=data['iterations'],
        alpha=data['alpha'],
        num_threads=data.get('num_threads', 0)
    )
    
    recommender.model = data['model']
    recommender.user_encoder = data['user_encoder']
    recommender.track_encoder = data['track_encoder']
    recommender.user_decoder = data['user_decoder']
    recommender.track_decoder = data['track_decoder']
    recommender.is_fitted = True
    
    logger.info(f'Model loaded from {filepath}')

    return recommender

# ---------- Main training function ---------- #
def train_als_model(preprocessed_dir=None):
    '''
        Train ALS model on preprocessed data.
    '''

    # Load config from environment
    preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
    factors = int(os.getenv('ALS_FACTORS', 64))
    regularization = float(os.getenv('ALS_REGULARIZATION', 0.01))
    iterations = int(os.getenv('ALS_ITERATIONS', 15))
    alpha = float(os.getenv('ALS_ALPHA', 1.0))
    num_threads = int(os.getenv('ALS_NUM_THREADS', 0))  # 0 = use all cores
    n_recommendations = int(os.getenv('ALS_NUMBER_REC', 10))
    
    models_dir = os.getenv('MODELS_DIR', './models')
    os.makedirs(models_dir, exist_ok=True)
    
    logger.info('ALS model training')

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
        num_threads=num_threads
    )
    recommender.fit(
        train_matrix,
        encoders['user_encoder'],
        encoders['track_encoder']
    )
    
    logger.info('Evaluating ALS model')
    test_path = f'{preprocessed_dir}/test_matrix.npz'
    test_matrix = load_npz(test_path)
    recommender.evaluate(train_matrix, test_matrix, k=10, sample_users=10000)
    
    logger.info(f'Generating personal recommendations (n={n_recommendations})')
    recommender.generate_recommendations(train_matrix, n=n_recommendations)

    logger.info('Saving ALS model')
    model_path = os.path.join(models_dir, 'als_model.pkl')
    recommender.save(model_path)

    logger.info('ALS model training complete')

    return None

# ---------- All exports ---------- #
__all__ = ['ALSRecommender', 'load_als_model', 'train_als_model']

# ---------- Main entry point ---------- #
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ALS Collaborative Filtering Recommender')
    parser.add_argument('--train', action='store_true', help='Train ALS model (includes evaluation and saving)')
    parser.add_argument('--recommend', action='store_true', help='Generate recommendations for all users')
    parser.add_argument('--user-id', type=int, help='Get recommendations for a specific user ID')
    args = parser.parse_args()

    if args.train:
        logger.info('Starting ALS model training pipeline')
        train_als_model()
    
    elif args.recommend or args.user_id:
        # Load model and data
        models_dir = os.getenv('MODELS_DIR', './models')
        preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
        
        model_path = os.path.join(models_dir, 'als_model.pkl')
        if not os.path.exists(model_path):
            logger.error(f'Model not found at {model_path}. Run with --train first.')
            exit(1)
        
        recommender = load_als_model(model_path)
        train_matrix = load_npz(f'{preprocessed_dir}/train_matrix.npz')
        
        if args.user_id:
            # Get recommendations for a specific user
            recommendations = recommender.recommend(args.user_id, train_matrix, n=args.n_recs)
            print(f'\nTop {args.n_recs} recommendations for user {args.user_id}:')
            for track_id, score in recommendations:
                print(f'Track {track_id}: {score:.4f}')
        else:
            # Generate recommendations for all users
            logger.info('Generating recommendations for all users')
            recommender.generate_recommendations(train_matrix, n=args.n_recs)
    
    else:
        parser.print_help()
        print('\nExamples:')
        print('  python -m src.collaborative_rec --train')
        print('  python -m src.collaborative_rec --recommend')
        print('  python -m src.collaborative_rec --user-id 12345 --n-recs 20')

# ---------- All exports ---------- #
__all__ = ['ALSRecommender', 'load_als_model', 'train_als_model']
