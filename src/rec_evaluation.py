'''
Recommendation Model Evaluation

This module provides metrics for evaluating recommendation quality:
- Precision@K - fraction of recommended items that are relevant
- Recall@K - fraction of relevant items that are recommended
- NDCG@K - normalized discounted cumulative gain (ranking quality)
- Novelty - measures how novel/unpopular the recommendations are
- Diversity - measures variety in recommendations using track_group_id

Models: popularity, collaborative (ALS), ranked (CatBoost)

Results are saved to ./results as JSON

Usage:
    python -m src.rec_evaluation --model popularity
    python -m src.rec_evaluation --model collaborative
    python -m src.rec_evaluation --model ranked
    python -m src.rec_evaluation --model all
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

from src.rec_ranking import generate_ranked_recommendations
from src.popularity_based_rec import generate_popularity_recommendations
from src.collaborative_rec import load_als_model, generate_als_recommendations

# ---------- Load environment variables ---------- #
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))

# ---------- Logging setup ---------- #
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

# ---------- Recommendation Evaluator ---------- #
class RecommendationEvaluator:
    '''Compute Precision@K, Recall@K, NDCG@K, Novelty, Diversity.'''
    
    def __init__(self, catalog_df: pl.DataFrame, events_df: pl.DataFrame):
        # Track popularity for novelty
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
        '''
        
        scores = [-np.log2(self.track_pop.get(t, 1) / self.max_pop + 1e-10)
                  for r in recs.values() for t in r[:k]]
        
        return float(np.mean(scores)) if scores else 0.0
    
    def diversity(self, recs: Dict[int, List[int]], k: int) -> float:
        '''
            Diversity - measures variety in recommendations using track_group_id.
        '''
        
        scores = [len({self.track_to_group.get(t, t) for t in r[:k]}) / len(r[:k])
                  for r in recs.values() if r[:k]]
        
        return float(np.mean(scores)) if scores else 0.0
    
    def evaluate(self, recs: Dict[int, List[int]], test: Dict[int, Set[int]], k: int) -> Dict:
        '''
            Evaluate recommendations and return all metrics.
        '''
        return {
            f'precision@{k}': self.precision_at_k(recs, test, k),
            f'recall@{k}': self.recall_at_k(recs, test, k),
            f'ndcg@{k}': self.ndcg_at_k(recs, test, k),
            f'novelty@{k}': self.novelty(recs, k),
            f'diversity@{k}': self.diversity(recs, k),
        }

# ---------- Recommendation Generators ---------- #
def get_popularity_recommendations(preprocessed_dir: str, n: int = 100) -> Dict[int, List[int]]:
    '''
        Popularity recommendations.
    '''

    logger.info('Loading popularity recommendations')
    results_dir = os.getenv('RESULTS_DIR', './results')
    popularity_path = f'{results_dir}/top_popular.parquet'
    
    if os.path.exists(popularity_path):
        popularity_recs = pl.read_parquet(popularity_path)['track_id'].to_list()
        logger.info(f'Loaded {len(popularity_recs):,} popularity recommendations')
    else:
        logger.info('Popularity recommendations not found, generating using popularity model...')
        popularity_recs = generate_popularity_recommendations(preprocessed_dir, n=n)
        logger.info(f'Generated {len(popularity_recs):,} popularity recommendations')
    
    return popularity_recs

def get_collaborative_recommendations(preprocessed_dir: str, models_dir: str, n: int = 10) -> Dict[int, List[int]]:
    '''
        ALS collaborative filtering recommendations.
    '''
    logger.info('Loading ALS recommendations')
    
    results_dir = os.getenv('RESULTS_DIR', './results')
    als_path = f'{results_dir}/personal_als.parquet'
    
    if os.path.exists(als_path):
        df = pl.read_parquet(als_path)
        als_recs = {u: df.filter(pl.col('user_id') == u).sort('rank')['track_id'].to_list()[:n]
                for u in df['user_id'].unique().to_list()}
        logger.info(f'Loaded {len(als_recs):,} ALS recommendations')
    else:
        logger.info('ALS recommendations not found, generating using ALS model...')
        model = load_als_model(f'{models_dir}/als_model.pkl')
        matrix = load_npz(f'{preprocessed_dir}/train_matrix.npz')
        users = pl.read_parquet(f'{preprocessed_dir}/test_events.parquet')['user_id'].unique().to_list()
        als_recs = generate_als_recommendations(model, matrix, users, n=n)
        logger.info(f'Generated {len(als_recs):,} ALS recommendations')
    
    return als_recs
    
def get_ranked_recommendations(preprocessed_dir: str, models_dir: str, n: int = 10, sample_users: int = 5000) -> Dict[int, List[int]]:
    '''
        Ranked recommendations.
    '''
    logger.info('Loading ranked recommendations')
    results_dir = os.getenv('RESULTS_DIR', './results')
    ranked_path = f'{results_dir}/ranked.parquet'
    
    if os.path.exists(ranked_path):
        ranked_recs = pl.read_parquet(ranked_path)['track_id'].to_list()
        logger.info(f'Loaded {len(ranked_recs):,} ranked recommendations')
    else:
        logger.info('Ranked recommendations not found, generating using ranked model...')
        ranked_recs = generate_ranked_recommendations(preprocessed_dir, models_dir, n=n)
        logger.info(f'Generated {len(ranked_recs):,} ranked recommendations')
    
    return ranked_recs

# ---------- Evaluation ---------- #
def evaluate_model(model_name: str, preprocessed_dir: str, models_dir: str,
                   k_values: List[int], output_dir: str, sample_users: int = 5000) -> Dict:
    '''
        Evaluate a model and save results to JSON.
    '''
    logger.info(f'Evaluating {model_name.upper()}')
    
    # Load data
    catalog = pl.read_parquet(f'{preprocessed_dir}/tracks_catalog_clean.parquet')
    events = pl.read_parquet(f'{preprocessed_dir}/events.parquet')
    test_events = pl.read_parquet(f'{preprocessed_dir}/test_events.parquet')
    
    # Get recommendations
    if model_name == 'popularity':
        recs = get_popularity_recommendations(preprocessed_dir)
    elif model_name == 'collaborative':
        recs = get_collaborative_recommendations(preprocessed_dir, models_dir)
    elif model_name == 'ranked':
        recs = get_ranked_recommendations(preprocessed_dir, sample_users=sample_users)
    else:
        logger.error(f'Unknown model: {model_name}')
        return {}
    
    if not recs:
        logger.error(f'No recommendations found for {model_name}')
        return {}
    
    # Build test set
    test_items = defaultdict(set)
    for row in test_events.iter_rows(named=True):
        test_items[row['user_id']].add(row['track_id'])
    
    # Evaluate
    evaluator = RecommendationEvaluator(catalog, events)
    results = {'model_name': model_name, 'evaluation_date': datetime.now().isoformat(), 'metrics': {}}
    
    for k in k_values:
        metrics = evaluator.evaluate(recs, test_items, k)
        results['metrics'][f'k={k}'] = metrics
        logger.info(f'K={k}: P={metrics[f"precision@{k}"]:.4f} R={metrics[f"recall@{k}"]:.4f} NDCG={metrics[f"ndcg@{k}"]:.4f}')
    
    # Save locally
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/evaluation_{model_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Saved to {output_dir}/evaluation_{model_name}.json')
    
    return results


def compare_models(output_dir: str) -> None:
    '''
        Compare evaluated models.
    '''

    logger.info('Comparing models')
    for name in ['popularity', 'collaborative', 'ranked']:
        path = f'{output_dir}/evaluation_{name}.json'
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            logger.info(f'{name}: {data["metrics"]}')


# ---------- Main entry point ---------- #
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description='Evaluate recommendation models')
    parser.add_argument('--model', choices=['popularity', 'collaborative', 'ranked', 'all'], default='all')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    
    preprocessed_dir = args.preprocessed_dir or os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
    models_dir = os.getenv('MODELS_DIR', './models')
    output_dir = os.getenv('RESULTS_DIR', './results')
    k_values = os.getenv('K_VALUES', 5)
    sample_users = os.getenv('SAMPLE_USERS', 5000)
    
    models = ['popularity', 'collaborative', 'ranked'] if args.model == 'all' else [args.model]
    
    for model in models:
        evaluate_model(model, preprocessed_dir, models_dir, k_values, sample_users)
    
    if args.compare or args.model == 'all':
        compare_models(output_dir)
