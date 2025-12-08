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
        python -m src.rec_evaluation --model popularity # model to evaluate: popularity, collaborative, ranked, all
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

from src.popularity_based_rec import PopularityRecommender
from src.collaborative_rec import ALSRecommender, load_als_model
from src.rec_ranking import generate_ranked_recommendations

# ---------- Load environment variables ---------- #
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))

# ---------- Logging setup ---------- #
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

# ---------- Recommendation Evaluator ---------- #
class RecommendationEvaluator:
    '''
        Compute Precision@K, Recall@K, NDCG@K, Novelty, Diversity.
        
        1. Initialize the evaluator with the catalog and events data
        2. Compute the track popularity and track groups for diversity
        3. Compute the precision@k, recall@k, ndcg@k, novelty, diversity metrics
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
        '''
        
        scores = [len(set(recs[u][:k]) & test[u]) / k for u in recs if u in test and recs[u]]
        
        return float(np.mean(scores)) if scores else 0.0
    
    def recall_at_k(self, recs: Dict[int, List[int]], test: Dict[int, Set[int]], k: int) -> float:
        '''
            Recall@K - fraction of relevant items that are recommended.
        '''
        
        scores = [len(set(recs[u][:k]) & test[u]) / len(test[u]) for u in recs if u in test and test[u]]
        
        return float(np.mean(scores)) if scores else 0.0
    
    def ndcg_at_k(self, recs: Dict[int, List[int]], test: Dict[int, Set[int]], k: int) -> float:
        '''
            NDCG@K - normalized discounted cumulative gain.            
            - Compute the NDCG@k for each user
            - Return the average NDCG@k
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
        '''
        
        scores = [-np.log2(self.track_pop.get(t, 1) / self.max_pop + 1e-10)
                  for r in recs.values() for t in r[:k]]
        
        return float(np.mean(scores)) if scores else 0.0
    
    def diversity(self, recs: Dict[int, List[int]], k: int) -> float:
        '''
            Diversity - measures variety in recommendations using track_group_id.            
            - Compute the diversity score for each user
            - Return the average diversity score
        '''
        
        scores = [len({self.track_to_group.get(t, t) for t in r[:k]}) / len(r[:k])
                  for r in recs.values() if r[:k]]
        
        return float(np.mean(scores)) if scores else 0.0
    
    def all_metrics(self, recs: Dict[int, List[int]], test: Dict[int, Set[int]], k: int) -> Dict:
        '''
            Compute all metrics and return them as a dictionary.            
            - Compute the precision@k, recall@k, ndcg@k, novelty, diversity metrics
            - Return the metrics as a dictionary
        '''
        
        return {
            'precision@k': self.precision_at_k(recs, test, k),
            'recall@k': self.recall_at_k(recs, test, k),
            'ndcg@k': self.ndcg_at_k(recs, test, k),
            'novelty@k': self.novelty(recs, k),
            'diversity@k': self.diversity(recs, k),
        }

# ---------- Popularity-based recommendations ---------- #
def generate_popular_recommendations(
    results_dir: str,
    train_events: pl.DataFrame,
    test_events: pl.DataFrame,
    catalog: pl.DataFrame,
    n: int = 100,
    n_recs: int = 10,
    method: str = 'listen_count'
) -> Dict[int, List[int]]:
    '''
        Generate popularity-based recommendations for each test user using PopularityRecommender.
        
        1. Check if top_popular.parquet exists, load it, otherwise compute from train_events
        2. For each test user, filter out tracks they've listened to
        3. Return top N recommendations per user
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
        preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
        logger.info(f'No file found, computing popularity using PopularityRecommender')
        recommender.fit(
            preprocessed_dir=preprocessed_dir,
            method=method,
            with_metadata=False,
            n=n
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
            n_recs=n_recs,
            user_listened=user_listened.get(user_id, set())
        )
        popularity_based_rec[user_id] = recs
    
    gc.collect()
    logger.info(f'Generated {len(popularity_based_rec):,} recommendation lists')
    return popularity_based_rec

# ---------- Collaborative-based recommendations ---------- #
def generate_als_recommendations(
    model_path: str,
    train_matrix,
    test_events: pl.DataFrame,
    n: int = 10,
) -> Dict[int, List[int]]:
    '''
        Generate ALS-based recommendations.
        
        1. Load the ALS model
        2. Get the test users
        3. Generate recommendations for each test user
        4. Return the recommendations
    '''
    logger.info(f'Generating ALS recommendations from {model_path}')
    
    # Load model
    als_model = load_als_model(model_path)
    
    # Get test users
    test_users = test_events['user_id'].unique().to_list()
    logger.info(f'Generating for {len(test_users):,} test users')
    
    als_based_rec = {}
    
    for i, user_id in enumerate(test_users):
        if i % 10000 == 0 and i > 0:
            logger.info(f'  Generated {i:,} / {len(test_users):,}')    
        
        # Generate recommendations for the user
        recs = als_model.recommend(user_id, train_matrix, n=n, filter_already_liked=True)
        
        # Extract the track_ids
        als_based_rec[user_id] = [track_id for track_id, _ in recs]
    
    logger.info(f'Generated {len(als_based_rec):,} recommendation lists')
    return als_based_rec
    
def get_ranked_recommendations(
    preprocessed_dir: str,
    results_dir: str,
    models_dir: str,
    n: int = 10,
    sample_users: int = 5000
) -> Dict[int, List[int]]:
    '''
        Load or generate ranked recommendations.
    '''
    logger.info('Loading ranked recommendations')
    ranked_path = f'{results_dir}/ranked.parquet'
    
    if os.path.exists(ranked_path):
        ranked_recs = pl.read_parquet(ranked_path)['track_id'].to_list()
        logger.info(f'Loaded {len(ranked_recs):,} ranked recommendations')
    else:
        logger.info('Ranked recommendations not found, generating using ranked model...')
        ranked_recs = generate_ranked_recommendations(preprocessed_dir, models_dir, n=n)
        logger.info(f'Generated {len(ranked_recs):,} ranked recommendations')
    
    return ranked_recs

# ---------- Evaluation (implementation) ---------- #
def _evaluate_model_impl(model_name: str, preprocessed_dir: str, models_dir: str,
                         k_values: List[int], output_dir: str, n_recs: int = 10, sample_users: int = 5000) -> Dict:
    '''
        Evaluate a model and save results to JSON.
    '''
    logger.info(f'Evaluating {model_name.upper()}')
    
    # Load data
    catalog = pl.read_parquet(f'{preprocessed_dir}/tracks_catalog_clean.parquet')
    events = pl.read_parquet(f'{preprocessed_dir}/events.parquet')
    train_events = pl.read_parquet(f'{preprocessed_dir}/train_events.parquet')
    test_events = pl.read_parquet(f'{preprocessed_dir}/test_events.parquet')
    train_matrix = load_npz(f'{preprocessed_dir}/train_matrix.npz')
    
    # Get recommendations based on model type
    if model_name == 'popularity':
        recs = generate_popular_recommendations(output_dir, train_events, test_events, catalog, n_recs=n_recs)
    elif model_name == 'collaborative':
        als_model_path = os.path.join(models_dir, 'als_model.pkl')
        recs = generate_als_recommendations(als_model_path, train_matrix, test_events, n=n_recs)
    elif model_name == 'ranked':
        recs = get_ranked_recommendations(preprocessed_dir, output_dir, models_dir, n=n_recs, sample_users=sample_users)
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

# ---------- Wrapper for main.py (reads from env vars) ---------- #
def run_evaluation(model='all'):
    '''
        Run model evaluation.
    '''
    logger.info('Running recommendation models evaluation pipeline')

    # Check required environment variables
    required_env_vars = [
        'PREPROCESSED_DATA_DIR', 
        'RESULTS_DIR', 
        'MODELS_DIR',
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
    k_values = [int(k.strip()) for k in os.getenv('EVALUATION_K_VALUES').split(',')]
    sample_users = int(os.getenv('EVALUATION_SAMPLE_USERS'))
    n_recs = int(os.getenv('EVALUATION_N_RECS'))
    
    # Use arg or env for model selection
    model_selection = model or os.getenv('EVALUATION_MODELS', 'all')
    models = ['popularity', 'collaborative', 'ranked'] if model_selection == 'all' else [model_selection]
    
    for m in models:
        _evaluate_model_impl(m, preprocessed_dir, models_dir, k_values, results_dir, n_recs, sample_users)
    
    if model_selection == 'all':
        compare_models(results_dir)
    
    gc.collect()
    logger.info('Recommendation models evaluation pipeline completed')

# ---------- Main entry point ---------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate recommendation models')
    parser.add_argument('--model', choices=['popularity', 'collaborative', 'ranked', 'all'], default='all')
    args = parser.parse_args()
    
    run_evaluation(args.model)
