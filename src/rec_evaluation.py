'''
    Recommendation Model Evaluation

    This module provides functionality to evaluate recommendation models using metrics:
    - Precision@K - fraction of recommended items that are relevant
    - Recall@K - fraction of relevant items that are recommended
    - NDCG@K - normalized discounted cumulative gain (ranking quality)
    - Novelty - measures how novel/unpopular the recommendations are
    - Diversity - measures variety in recommendations using track_group_id

    Models: popularity, collaborative (ALS), ranked (CatBoost)

    Results are saved to ./results as JSON

    Usage:
        python -m src.rec_evaluation # evaluate all models
        python -m src.rec_evaluation --model popularity # model to evaluate: popularity, als, ranked, all
'''

# ---------- Imports ---------- #
import os
import gc
import json
import logging
from datetime import datetime
from typing import List, Dict, Set
from collections import defaultdict
import argparse

import numpy as np
import polars as pl
from scipy.sparse import load_npz
from dotenv import load_dotenv

from src.popularity_based_model import PopularityRecommender
from src.als_model import ALSRecommender, load_als_model
from src.rec_ranking import generate_ranked_recommendations
from src.logging_set_up import setup_logging

# ---------- Load environment variables ---------- #
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))

# ---------- Logging setup ---------- #
logger = setup_logging('rec_evaluation')

# ---------- Recommendation Evaluator ---------- #
class RecommendationEvaluator:
    '''
        Compute Precision@K, Recall@K, NDCG@K, Novelty, Diversity.
        
        1. Initialize the evaluator with the catalog and events data
        2. Compute the track popularity and track groups for diversity
        3. Compute the precision@k, recall@k, ndcg@k, novelty, diversity metrics

        Args:
        - catalog_df - catalog dataframe
        - events_df - events dataframe
        - results_dir - path to results directory

        Attributes:
        - catalog_df - catalog dataframe
        - events_df - events dataframe
        - results_dir - path to results directory
        - track_pop - dictionary of track ids and popularity scores
        - max_pop - maximum popularity score
        - track_to_group - dictionary of track ids and track group ids

        Methods:
        - precision_at_k - compute precision@k
        - recall_at_k - compute recall@k
        - ndcg_at_k - compute ndcg@k
        - novelty - compute novelty
        - diversity - compute diversity
        - all_metrics - compute all metrics
    '''
    
    def __init__(self, catalog_df: pl.DataFrame, events_df: pl.DataFrame, results_dir: str = None):
        
        # Track popularity for novelty
        if results_dir is None:
            results_dir = os.getenv('RESULTS_DIR', './results')
        
        popularity_track_scores_path = os.path.join(results_dir, 'popularity_track_scores.parquet')
        if os.path.exists(popularity_track_scores_path):
            popularity_track_scores = pl.read_parquet(popularity_track_scores_path)
            self.track_pop = dict(zip(popularity_track_scores['track_id'].to_list(), popularity_track_scores['popularity_score'].to_list()))
        else:
            # Compute from events if file doesn't exist
            track_pop = events_df.group_by('track_id').agg(pl.sum('listen_count').alias('pop'))
            self.track_pop = dict(zip(track_pop['track_id'].to_list(), track_pop['pop'].to_list()))
        
        self.max_pop = max(self.track_pop.values()) if self.track_pop else 1
        
        # Track groups for diversity
        self.track_to_group = dict(zip(
            catalog_df['track_id'].to_list(),
            catalog_df['track_group_id'].to_list()
        ))
    
    def precision_at_k(self, recs: Dict[int, List[int]], test: Dict[int, Set[int]], k: int) -> float:
        '''
            Precision@K - fraction of recommended items that are relevant.            
            - Compute the precision@k for each user
            - Return the average precision@k

            Args:
            - recs - dictionary of recommendations
            - test - dictionary of test items
            - k - number of recommendations to return

            Returns:
            - average precision@k
        '''
        
        scores = [len(set(recs[u][:k]) & test[u]) / k for u in recs if u in test and recs[u]]
        
        return float(np.mean(scores)) if scores else 0.0
    
    def recall_at_k(self, recs: Dict[int, List[int]], test: Dict[int, Set[int]], k: int) -> float:
        '''
            Recall@K - fraction of relevant items that are recommended.

            Args:
            - recs - dictionary of recommendations
            - test - dictionary of test items
            - k - number of recommendations to return

            Returns:
            - average recall@k
        '''
        
        scores = [len(set(recs[u][:k]) & test[u]) / len(test[u]) for u in recs if u in test and test[u]]
        
        return float(np.mean(scores)) if scores else 0.0
    
    def ndcg_at_k(self, recs: Dict[int, List[int]], test: Dict[int, Set[int]], k: int) -> float:
        '''
            NDCG@K - normalized discounted cumulative gain.            
            - Compute the NDCG@k for each user
            - Return the average NDCG@k

            Args:
            - recs - dictionary of recommendations
            - test - dictionary of test items
            - k - number of recommendations to return

            Returns:
            - average NDCG@k
        '''
        
        ndcgs = []
        for u in recs:
            if u not in test:
                continue
            rel = test[u]
            dcg = sum(1.0 / np.log2(i + 2) for i, t in enumerate(recs[u][:k]) if t in rel)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
            if idcg > 0:
                ndcgs.append(dcg / idcg)
        
        return float(np.mean(ndcgs)) if ndcgs else 0.0
    
    def novelty(self, recs: Dict[int, List[int]], k: int) -> float:
        '''
            Novelty - measures how novel/unpopular the recommendations are.            
            - Compute the novelty score for each user
            - Return the average novelty score

            Args:
            - recs - dictionary of recommendations
            - k - number of recommendations to return

            Returns:
            - average novelty score
        '''
        
        scores = [-np.log2(self.track_pop.get(t, 1) / self.max_pop + 1e-10)
                  for r in recs.values() for t in r[:k]]
        
        return float(np.mean(scores)) if scores else 0.0
    
    def diversity(self, recs: Dict[int, List[int]], k: int) -> float:
        '''
            Diversity - measures variety in recommendations using track_group_id.            
            - Compute the diversity score for each user
            - Return the average diversity score

            Args:
            - recs - dictionary of recommendations
            - k - number of recommendations to return

            Returns:
            - average diversity score
        '''
        
        scores = [len({self.track_to_group.get(t, t) for t in r[:k]}) / len(r[:k])
                  for r in recs.values() if r[:k]]
        
        return float(np.mean(scores)) if scores else 0.0
    
    def all_metrics(self, recs: Dict[int, List[int]], test: Dict[int, Set[int]], k: int) -> Dict:
        '''
            Compute all metrics and return them as a dictionary.            
            - Compute the precision@k, recall@k, ndcg@k, novelty, diversity metrics
            - Return the metrics as a dictionary

            Args:
            - recs - dictionary of recommendations
            - test - dictionary of test items
            - k - number of recommendations to return

            Returns:
            - dictionary of metrics
        '''
        
        return {
            'precision@k': self.precision_at_k(recs, test, k),
            'recall@k': self.recall_at_k(recs, test, k),
            'ndcg@k': self.ndcg_at_k(recs, test, k),
            'novelty@k': self.novelty(recs, k),
            'diversity@k': self.diversity(recs, k),
        }

# ---------- Popularity-based recommendations ---------- #
def get_popular_recommendations(
    preprocessed_dir: str,
    results_dir: str,
    train_events: pl.DataFrame,
    test_events: pl.DataFrame,
    catalog: pl.DataFrame,
    n_popular: int,
    n_recs_popular: int,
    method: str = 'listen_count'
) -> Dict[int, List[int]]:
    '''
        Generate popularity-based recommendations for each test user using PopularityRecommender.
        
        1. Check if top_popular.parquet exists, load it, otherwise compute from train_events
        2. For each test user, filter out tracks they've listened to
        3. Return top N recommendations per user

        Args:
        - preprocessed_dir - path to preprocessed data directory
        - results_dir - path to results directory
        - train_events - training events dataframe
        - test_events - test events dataframe
        - catalog - catalog dataframe
        - n - number of recommendations to return
        - n_recs - number of recommendations per user
        - method - method to use for popularity-based recommendations

        Returns:
        - dictionary of recommendations
    '''
    logger.info(f'Generating popular recommendations using method={method}')
    
    # Check if popular tracks exist
    popularity_path = os.path.join(results_dir, 'top_popular.parquet')
    
    # Initialize recommender and load/compute popularity
    recommender = PopularityRecommender()
    
    if os.path.exists(popularity_path):
        logger.info(f'Loading popular tracks from {popularity_path}')
        recommender.top_tracks = pl.read_parquet(popularity_path)
        logger.info(f'Loaded {recommender.top_tracks.height:,} popular tracks')
    else:
        # Compute from preprocessed_dir (uses full events.parquet)
        logger.info(f'No file found, computing popularity using PopularityRecommender')
        recommender.fit(
            preprocessed_dir=preprocessed_dir,
            method=method,
            with_metadata=False,
            n=n_popular
        )
    
    # Build user listening history from train events (computed once)
    user_listened = defaultdict(set)
    for row in train_events.iter_rows(named=True):
        user_listened[row['user_id']].add(row['track_id'])
    
    # Get test users
    test_users = test_events['user_id'].unique().to_list()
    logger.info(f'Generating recommendations for {len(test_users):,} test users')
    
    # Generate recommendations for each user using recommender.recommend()
    popularity_based_rec = {}
    for i, user_id in enumerate(test_users):
        if i % 10000 == 0 and i > 0:
            logger.info(f'Generated {i:,} / {len(test_users):,}')
        
        # Use pre-computed user_listened for efficiency (no file loading)
        recs = recommender.recommend(
            user_id=user_id,
            n_recs=n_recs_popular,
            user_listened=user_listened.get(user_id, set())
        )
        popularity_based_rec[user_id] = recs
    
    gc.collect()
    logger.info(f'Generated {len(popularity_based_rec):,} recommendation lists')
    return popularity_based_rec

# ---------- Collaborative-based recommendations ---------- #
def get_als_recommendations(
    model_path: str,
    train_matrix,
    test_events: pl.DataFrame,
    n_als: int
) -> Dict[int, List[int]]:
    '''
        Generate ALS-based recommendations using batch processing for speed.
        
        1. Load the ALS model
        2. Get the test users and filter to valid ones
        3. Use batch recommend for all users at once (much faster than one-by-one)
        4. Return the recommendations

        Args:
        - model_path - path to ALS model
        - train_matrix - training matrix
        - test_events - test events dataframe
        - n - number of recommendations to return

        Returns:
        - dictionary of recommendations
    '''
    logger.info(f'Generating ALS recommendations from {model_path}')
    
    # Load model
    als_model = load_als_model(model_path)
    
    # Get test users
    test_users = test_events['user_id'].unique().to_list()
    logger.info(f'Generating for {len(test_users):,} test users')
    
    # Filter to users that exist in training data and are within model bounds
    n_model_users = als_model.model.user_factors.shape[0]
    valid_users = []
    valid_indices = []
    
    for user_id in test_users:
        if user_id in als_model.user_encoder:
            user_idx = als_model.user_encoder[user_id]
            if user_idx < n_model_users:
                valid_users.append(user_id)
                valid_indices.append(user_idx)
    
    logger.info(f'Valid users for batch recommendation: {len(valid_users):,}')
    
    if not valid_users:
        logger.warning('No valid users found for ALS recommendations')
        return {}
    
    # Convert to numpy array for batch processing
    valid_indices = np.array(valid_indices)
    
    # Use batch recommend for much faster processing (all users at once)
    logger.info('Running batch recommendation with ALS model')
    track_indices_batch, scores_batch = als_model.model.recommend(
        valid_indices,
        train_matrix[valid_indices],
        N=n_als,
        filter_already_liked_items=True
    )
    logger.info('Batch recommendation complete')
    
    # Build recommendations dict
    als_based_rec = {}
    for i, user_id in enumerate(valid_users):
        # Decode track indices to track_ids
        track_ids = [
            als_model.track_decoder[int(idx)]
            for idx in track_indices_batch[i]
            if int(idx) in als_model.track_decoder
        ]
        als_based_rec[user_id] = track_ids
    
    logger.info(f'Generated {len(als_based_rec):,} recommendation lists')
    return als_based_rec

# ---------- Ranked recommendations ---------- #
def get_ranked_recommendations(
    preprocessed_dir: str,
    results_dir: str,
    models_dir: str,
    n_ranked: int,
    sample_users: int
) -> Dict[int, List[int]]:
    '''
        Load or generate ranked recommendations.

        Args:
        - preprocessed_dir - path to preprocessed data directory
        - results_dir - path to results directory
        - models_dir - path to models directory
        - n_ranked - number of recommendations to return
        - sample_users - number of users to sample

        Returns:
        - dictionary of recommendations
    '''
    logger.info('Loading ranked recommendations')
    # Delegate to generate_ranked_recommendations which handles caching internally
    # and always returns Dict[int, List[int]]
    ranked_recs = generate_ranked_recommendations(preprocessed_dir, results_dir, models_dir, n_ranked, sample_users)
    logger.info(f'Loaded/generated recommendations for {len(ranked_recs):,} users')
    return ranked_recs

# ---------- Evaluation ---------- #
def evaluate_model(
    model_name: str = None, 
    preprocessed_dir: str = None, 
    models_dir: str = None,
    k_values: List[int] = None, 
    output_dir: str = None, 
    n_popular: int = None,
    n_recs_popular: int = None,
    n_als: int = None,
    n_ranked: int = None,
    sample_users: int = None
) -> Dict:
    '''
        Evaluate a model and save results to JSON.

        Args:
        - model_name - name of the model
        - preprocessed_dir - path to preprocessed data directory
        - models_dir - path to models directory
        - k_values - list of k values to evaluate
        - output_dir - path to output directory
        Evaluate a model and save results to JSON.

        Returns:
        - dictionary of results
    '''
    # Load defaults from environment if not provided
    if model_name is None:
        model_name = 'all'
    if preprocessed_dir is None:
        preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
    if models_dir is None:
        models_dir = os.getenv('MODELS_DIR', './models')
    if k_values is None:
        k_values = [int(k.strip()) for k in os.getenv('EVALUATION_K_VALUES', '5,10').split(',')]
    if output_dir is None:
        output_dir = os.getenv('RESULTS_DIR', './results')
    if n_popular is None:
        n_popular = int(os.getenv('POPULARITY_TOP_N', 100))
    if n_recs_popular is None:
        n_recs_popular = int(os.getenv('POPULARITY_N_RECS', 10))
    if n_als is None:
        n_als = int(os.getenv('ALS_N_RECS', 10))
    if n_ranked is None:
        n_ranked = int(os.getenv('RANKED_N_RECS', 10))
    if sample_users is None:
        sample_users = int(os.getenv('EVALUATION_SAMPLE_USERS', 10000))
    
    # Handle 'all' model evaluation
    if model_name == 'all':
        results = {}
        for m in ['popularity', 'als', 'ranked']:
            results[m] = evaluate_model(m, preprocessed_dir, models_dir, k_values, output_dir, n_popular, n_recs_popular, n_als, n_ranked, sample_users)
        compare_models(output_dir)
        return results
    
    logger.info(f'Evaluating {model_name.upper()}')
    
    # Load data
    catalog = pl.read_parquet(f'{preprocessed_dir}/tracks_catalog_clean.parquet')
    events = pl.read_parquet(f'{preprocessed_dir}/events.parquet')
    train_events = pl.read_parquet(f'{preprocessed_dir}/train_events.parquet')
    test_events = pl.read_parquet(f'{preprocessed_dir}/test_events.parquet')
    train_matrix = load_npz(f'{preprocessed_dir}/train_matrix.npz')
    
    # Get recommendations based on model type
    if model_name == 'popularity':
        recs = get_popular_recommendations(preprocessed_dir, output_dir, train_events, test_events, catalog, n_popular, n_recs_popular, method='listen_count')
    elif model_name == 'als':
        als_model_path = os.path.join(models_dir, 'als_model.pkl')
        recs = get_als_recommendations(als_model_path, train_matrix, test_events, n_als)
    elif model_name == 'ranked':
        recs = get_ranked_recommendations(preprocessed_dir, output_dir, models_dir, n_ranked, sample_users)
    else:
        logger.error(f'Unknown model: {model_name}')
        return {}
    
    # Free train_matrix early as it's no longer needed
    del train_matrix
    gc.collect()
    
    if not recs:
        logger.error(f'No recommendations found for {model_name}')
        return {}
    
    # Build test set
    test_items = defaultdict(set)
    for row in test_events.iter_rows(named=True):
        test_items[row['user_id']].add(row['track_id'])
    
    # Free train_events early as it's no longer needed
    del train_events
    
    # Evaluate
    evaluator = RecommendationEvaluator(catalog, events, output_dir)
    total_users = events['user_id'].n_unique()
    
    # Free catalog and events after evaluator is initialized
    del catalog, events, test_events
    gc.collect()
    
    results = {
        'model_name': model_name,
        'evaluation_date': datetime.now().isoformat(),
        'total_users': total_users,
        'evaluated_users': len(recs),
        'metrics': {}
    }
    
    for k in k_values:
        metrics = evaluator.all_metrics(recs, test_items, k)
        results['metrics'][f'k={k}'] = metrics
        logger.info(f'K={k}: P={metrics["precision@k"]:.4f} R={metrics["recall@k"]:.4f} NDCG={metrics["ndcg@k"]:.4f}')
    
    # Free remaining large objects
    del evaluator, recs, test_items
    gc.collect()
    
    # Save locally
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/evaluation_{model_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Saved to {output_dir}/evaluation_{model_name}.json')
    
    return results

def compare_models(results_dir: str) -> pl.DataFrame:
    '''
        Compare multiple evaluated models from the results directory and save the comparison to a parquet file.

        Args:
        - results_dir - path to results directory

        Returns:
        - dataframe of results
    '''
    
    logger.info('Comparing models')
    
    result_files = [
        os.path.join(results_dir, 'evaluation_popularity.json'),
        os.path.join(results_dir, 'evaluation_collaborative.json'),
        os.path.join(results_dir, 'evaluation_ranked.json'),
    ]

    rows = []
    for path in result_files:
        if not os.path.exists(path):
            logger.warning(f'Result file not found: {path}')
            continue
        
        with open(path) as f:
            data = json.load(f)
        
        model_name = data.get('model_name', os.path.basename(path))
        
        for k_key, metrics in data.get('metrics', {}).items():
            row = {'model': model_name, 'k': k_key}
            row.update(metrics)
            rows.append(row)
        
        logger.info(f'{model_name}: {data.get("metrics", {})}')
    
    if not rows:
        logger.warning('No results to compare')
        return pl.DataFrame()
    
    models_comparison = pl.DataFrame(rows)
    
    # Save to file if requested
    models_comparison_path = os.path.join(results_dir, 'models_comparison.parquet')
    os.makedirs(os.path.dirname(models_comparison_path), exist_ok=True)
    models_comparison.write_parquet(models_comparison_path)
    logger.info(f'Saved comparison to {models_comparison_path}')
    
    return models_comparison   

# ---------- Main entry point ---------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate recommendation models')
    parser.add_argument('--model', choices=['popularity', 'als', 'ranked', 'all'], default='all')
    args = parser.parse_args()
    
    logger.info('Running recommendation models evaluation pipeline')

    # Check required environment variables
    required_env_vars = [
        'PREPROCESSED_DATA_DIR', 
        'RESULTS_DIR', 
        'MODELS_DIR',
        'POPULARITY_METHOD',
        'POPULARITY_TOP_N',
        'POPULARITY_N_RECS',
        'POPULARITY_WITH_METADATA',
        'POPULARITY_FILTER_LISTENED',
        'POPULARITY_USER_ID',
        'ALS_N_RECS',
        'ALS_NUM_THREADS',
        'ALS_FACTORS',
        'ALS_REGULARIZATION',
        'ALS_ITERATIONS',
        'ALS_ALPHA',
        'RANKED_N_RECS',
        'EVALUATION_K_VALUES',
        'EVALUATION_SAMPLE_USERS',
        'EVALUATION_N_RECS',
        'EVALUATION_MODELS'
    ]
    missing_vars = [var for var in required_env_vars if os.getenv(var) is None]
    if missing_vars:
        raise EnvironmentError(f'Missing required environment variables: {", ".join(missing_vars)}')

    # Load from env
    preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR')
    results_dir = os.getenv('RESULTS_DIR')
    models_dir = os.getenv('MODELS_DIR')
    method = os.getenv('POPULARITY_METHOD')
    top_n_popular = int(os.getenv('POPULARITY_TOP_N'))
    n_recs_popular = int(os.getenv('POPULARITY_N_RECS'))
    with_metadata_popular = bool(os.getenv('POPULARITY_WITH_METADATA'))
    filter_listened_popular = bool(os.getenv('POPULARITY_FILTER_LISTENED'))
    user_id_popular = int(os.getenv('POPULARITY_USER_ID'))
    factors = int(os.getenv('ALS_FACTORS'))
    regularization = float(os.getenv('ALS_REGULARIZATION'))
    iterations = int(os.getenv('ALS_ITERATIONS'))
    alpha = float(os.getenv('ALS_ALPHA'))
    num_threads = int(os.getenv('ALS_NUM_THREADS'))
    n_als = int(os.getenv('ALS_N_RECS'))
    k_values = [int(k.strip()) for k in os.getenv('EVALUATION_K_VALUES').split(',')]
    sample_users = int(os.getenv('EVALUATION_SAMPLE_USERS'))   
    n_ranked = int(os.getenv('RANKED_N_RECS'))
    model_selection = args.model or os.getenv('EVALUATION_MODELS', 'all')
    models = ['popularity', 'als', 'ranked'] if model_selection == 'all' else [model_selection]
    
    for m in models:
        evaluate_model(m, preprocessed_dir, models_dir, k_values, results_dir, top_n_popular, n_recs_popular, n_als, n_ranked, sample_users)
    
    if model_selection == 'all':
        compare_models(results_dir)
    
    gc.collect()
    logger.info('Recommendation models evaluation pipeline completed')