'''
    Model Evaluation

    This module provides functionality to evaluate recommendation models.

    Usage:
        python -m src.models_evaluation
'''

# ---------- Imports ---------- #
import os
import gc
import logging
from dotenv import load_dotenv
from typing import Dict, List
from collections import defaultdict

import polars as pl
from scipy.sparse import load_npz

from src.popularity_based_rec import PopularityRecommender
from src.collaborative_rec import ALSRecommender, load_als_model
from src.rec_ranking import RecommendationRanker
from src.rec_evaluation import evaluate_recommender, compare_models

# ---------- Load environment variables ---------- #
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))

# ---------- Logging setup ---------- #
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ---------- Popularity-based recommendations ---------- #
def generate_popular_recommendations(
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
    logger.info(f'Generating popular recommendations (method={method})')
    
    # Check if popular tracks exist
    results_dir = os.getenv('RESULTS_DIR', './results')
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

def apply_ranking(
    recommendations: Dict[int, List[int]],
    catalog: pl.DataFrame,
    train_events: pl.DataFrame,
    n: int = 10,
    diversity_weight: float = 0.3
) -> Dict[int, List[int]]:
    '''
    Apply re-ranking to recommendations.
    '''
    logger.info(f'Applying re-ranking with diversity_weight={diversity_weight}')
    
    ranker = RecommendationRanker(catalog)
    
    # Get popularity scores
    popularity_scores = (
        train_events
        .group_by('track_id')
        .agg(pl.sum('listen_count').alias('popularity'))
        .to_dict()
    )
    pop_dict = dict(zip(popularity_scores['track_id'], popularity_scores['popularity']))
    
    # Get user histories
    user_histories = defaultdict(set)
    for row in train_events.iter_rows(named=True):
        user_histories[row['user_id']].add(row['track_id'])
    
    # Re-rank for each user
    reranked = {}
    
    for user_id, rec_list in recommendations.items():
        # Convert to (track_id, score) format with dummy scores
        rec_tuples = [(tid, 1.0 - i*0.01) for i, tid in enumerate(rec_list)]
        
        # Apply multi-objective ranking
        ranked = ranker.rank_multi_objective(
            rec_tuples,
            popularity_scores=pop_dict,
            user_history=user_histories.get(user_id, set()),
            n=n,
            diversity_weight=diversity_weight,
            popularity_weight=0.1,
            novelty_weight=0.2
        )
        
        reranked[user_id] = [track_id for track_id, _ in ranked]
    
    logger.info(f'Re-ranked {len(reranked):,} recommendation lists')
    return reranked


def run_evaluation_pipeline(
    preprocessed_dir: str = 'data/preprocessed',
    results_dir: str = 'data/results',
    models_dir: str = 'models',
    n_recommendations: int = 20,
    k_values: List[int] = [5, 10, 20],
    sample_users: int = None
):
    '''
    Run complete evaluation pipeline.
    '''
    logger.info('Running models evaluation pipeline')
    
    # Load data
    logger.info('Loading data')
    train_events = pl.read_parquet(f'{preprocessed_dir}/train_events.parquet')
    test_events = pl.read_parquet(f'{preprocessed_dir}/test_events.parquet')
    catalog = pl.read_parquet(f'{preprocessed_dir}/tracks_catalog_clean.parquet')
    all_events = pl.read_parquet(f'{preprocessed_dir}/events.parquet')
    train_matrix = load_npz(f'{preprocessed_dir}/train_matrix.npz')
    
    total_users = all_events['user_id'].n_unique()
    
    logger.info(f'Data loaded')
    logger.info(f'Train: {train_events.height:,} interactions')
    logger.info(f'Test: {test_events.height:,} interactions')
    logger.info(f'Total users: {total_users:,}')
    
    # Sample users if requested
    if sample_users:
        logger.info(f'Sampling {sample_users:,} users for faster evaluation')
        test_users_sample = test_events['user_id'].unique().sample(n=sample_users, seed=42)
        test_events = test_events.filter(pl.col('user_id').is_in(test_users_sample))
        logger.info(f'Sampled test set: {test_events.height:,} interactions')
    
    # Create output directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Popular baseline
    logger.info('Generating popularity-based recommendations')
    
    popular_recs = generate_popular_recommendations(
        train_events, test_events, catalog, n=n_recommendations
    )
    
    popular_results = evaluate_recommender(
        model_name='PopularityBaseline',
        recommendations=popular_recs,
        test_events_df=test_events,
        catalog_df=catalog,
        all_events_df=all_events,
        total_users=total_users,
        k_values=k_values,
        output_file=os.path.join(results_dir, 'popular_baseline.json')
    )
    
    # ALS model
    logger.info('Generating ALS-based recommendations')
    
    als_model_path = os.path.join(models_dir, 'als_model.pkl')
    
    if os.path.exists(als_model_path):
        als_recs = generate_als_recommendations(
            als_model_path, train_matrix, test_events, n=n_recommendations
        )
        
        als_results = evaluate_recommender(
            model_name='ALS',
            recommendations=als_recs,
            test_events_df=test_events,
            catalog_df=catalog,
            all_events_df=all_events,
            total_users=total_users,
            k_values=k_values,
            output_file=os.path.join(results_dir, 'als_model.json')
        )
        
        # ALS with diversity re-ranking
        logger.info('Applying diversity re-ranking to ALS-based recommendations')
        
        als_reranked = apply_ranking(
            als_recs, catalog, train_events,
            n=max(k_values),
            diversity_weight=0.3
        )
        
        als_reranked_results = evaluate_recommender(
            model_name='ALS_Reranked',
            recommendations=als_reranked,
            test_events_df=test_events,
            catalog_df=catalog,
            all_events_df=all_events,
            total_users=total_users,
            k_values=k_values,
            output_file=os.path.join(results_dir, 'als_reranked.json')
        )
    else:
        logger.warning(f'ALS model not found at {als_model_path}')
        logger.warning('Skipping ALS evaluation')
    
    # Compare models
    logger.info('Comparing models')
    
    result_files = [
        os.path.join(results_dir, 'popular_baseline.json'),
    ]
    
    if os.path.exists(als_model_path):
        result_files.extend([
            os.path.join(results_dir, 'als_model.json'),
            os.path.join(results_dir, 'als_reranked.json'),
        ])
    
    comparison_df = compare_models(
        result_files,
        output_file=os.path.join(results_dir, 'model_comparison.csv')
    )
    
    logger.info('Models evaluation pipeline completed')

if __name__ == '__main__':

    logger.info('Running models evaluation pipeline')

    # Check if all required environment variables exist
    required_env_vars = [
        'PREPROCESSED_DATA_DIR',
        'RESULTS_DIR',
        'MODELS_DIR',
        'EVALUATION_N_RECOMMENDATIONS',
        'EVALUATION_K_VALUES',
        'EVALUATION_SAMPLE_USERS'
    ]
    
    missing_vars = [var for var in required_env_vars if os.getenv(var) is None]
    if missing_vars:
        logger.error(f'Missing required environment variables: {", ".join(missing_vars)}')
        raise EnvironmentError(f'Missing required environment variables: {", ".join(missing_vars)}')

    # Directory paths (strings)
    preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR')
    results_dir = os.getenv('RESULTS_DIR')
    models_dir = os.getenv('MODELS_DIR')
    
    n_recommendations = int(os.getenv('EVALUATION_N_RECOMMENDATIONS'))
    k_values = [int(k) for k in os.getenv('EVALUATION_K_VALUES').split(',')]
    sample_users = int(os.getenv('EVALUATION_SAMPLE_USERS'))
    
    run_evaluation_pipeline(
        preprocessed_dir=preprocessed_dir,
        results_dir=results_dir,
        models_dir=models_dir,
        n_recommendations=n_recommendations,
        k_values=k_values,
        sample_users=sample_users
    )

    logger.info('Models evaluation pipeline completed')

