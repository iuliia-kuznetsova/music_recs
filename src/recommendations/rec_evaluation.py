'''
    Recommendation Model Evaluation

    This module provides functionality to evaluate recommendation models using metrics:
    - Precision@K - fraction of recommended items that are relevant
    - Recall@K - fraction of relevant items that are recommended
    - NDCG@K - normalized discounted cumulative gain (ranking quality)
    - Novelty - measures how novel/unpopular the recommendations are
    - Diversity - measures variety in recommendations using track_group_id

    Models: popularity (top popular), collaborative (ALS), ranked (CatBoost)

    Results are saved to ./results as JSON

    Usage examples:
    python3 -m src.recommendations.rec_evaluation # evaluate all models
    python3 -m src.recommendations.rec_evaluation --model popularity # model to evaluate: popularity, als, ranked, all
    python3 -m src.recommendations.rec_evaluation --model all # evaluate all models
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

from src.recommendations.popularity_based_model import PopularityRecommender
from src.recommendations.als_model import ALSRecommender, load_als_model
from src.recommendations.rec_ranking import generate_ranked_recommendations
from src.logging_setup import setup_logging

# ---------- Load environment variables ---------- #
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, '.env'))

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
):
    '''
        Generate popularity-based recommendations for each test user using PopularityRecommender.
        
        1. Check if top_popular.parquet exists, load it, otherwise compute from train_events
        2. For each test user, filter out tracks they've listened to
        3. Return top N recommendations per user with actual popularity scores

        Args:
        - preprocessed_dir - path to preprocessed data directory
        - results_dir - path to results directory
        - train_events - training events dataframe
        - test_events - test events dataframe
        - catalog - catalog dataframe
        - n_popular - number of popular tracks to consider
        - n_recs_popular - number of recommendations per user
        - method - method to use for popularity-based recommendations

        Returns:
        - dictionary of user_id -> list of (track_id, popularity_score) tuples, ordered by score descending
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
    
    # Generate recommendations for each user with scores
    popularity_based_rec = {}
    for i, user_id in enumerate(test_users):
        if i % 10000 == 0 and i > 0:
            logger.info(f'Generated {i:,} / {len(test_users):,}')
        
        # Get filtered recommendations (track IDs only)
        listened = user_listened.get(user_id, set())
        
        # Filter top tracks not listened by user and get scores
        recs_with_scores = []
        for row in recommender.top_tracks.iter_rows(named=True):
            if row['track_id'] not in listened:
                recs_with_scores.append((row['track_id'], row['popularity_score']))
                if len(recs_with_scores) >= n_recs_popular:
                    break
        
        popularity_based_rec[user_id] = recs_with_scores
    
    gc.collect()
    logger.info(f'DONE: Popularity-based recommendations with scores for test users generated successfully')
    return popularity_based_rec

# ---------- ALS-based recommendations ---------- #
def get_als_recommendations(
    model_dir: str,
    train_matrix,
    test_events: pl.DataFrame,
    n_als: int
):
    '''
        Generate ALS-based recommendations using batch processing for speed.
        
        1. Load the ALS model
        2. Get the test users and filter to valid ones
        3. Use batch recommend for all users at once (much faster than one-by-one)
        4. Return the recommendations with scores

        Args:
        - model_dir - path to ALS model directory
        - train_matrix - training matrix
        - test_events - test events dataframe
        - n_als - number of recommendations to return

        Returns:
        - dictionary of user_id -> list of (track_id, als_score) tuples, ordered by score descending
    '''
    logger.info(f'Generating ALS recommendations from {model_dir}')
    
    # Load model
    als_model = load_als_model(model_dir)
    
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
    
    # Use batch recommend with progress tracking
    batch_size = 50_000
    n_users = len(valid_indices)
    n_batches = (n_users + batch_size - 1) // batch_size
    
    logger.info(f'Processing in {n_batches} batches of {batch_size:,} users each')
    
    all_track_indices = []
    all_scores = []
    total_recs_saved = 0
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_users)
        progress_pct = (start_idx / n_users) * 100
        
        logger.info(f'Processing batch {batch_idx + 1}/{n_batches} (users {start_idx:,}-{end_idx:,}, {progress_pct:.1f}%)')
        
        batch_indices = valid_indices[start_idx:end_idx]
        track_indices_batch, scores_batch = als_model.model.recommend(
            batch_indices,
            train_matrix[batch_indices],
            N=n_als,
            filter_already_liked_items=True
        )
        
        all_track_indices.append(track_indices_batch)
        all_scores.append(scores_batch)
        
        batch_recs = len(track_indices_batch) * n_als
        total_recs_saved += batch_recs
        logger.info(f'Batch {batch_idx + 1}/{n_batches} complete: {total_recs_saved:,} recommendations saved')
    
    # Combine all batches
    track_indices_batch = np.vstack(all_track_indices)
    scores_batch = np.vstack(all_scores)
    
    logger.info('Batch processing completed')
    
    # Build recommendations dict with scores: {user_id: [(track_id, score), ...]}
    als_based_rec = {}
    for i, user_id in enumerate(valid_users):
        recs_with_scores = []
        for j in range(len(track_indices_batch[i])):
            track_idx = int(track_indices_batch[i][j])
            if track_idx in als_model.track_decoder:
                track_id = als_model.track_decoder[track_idx]
                score = float(scores_batch[i][j])
                recs_with_scores.append((track_id, score))
        als_based_rec[user_id] = recs_with_scores
    
    logger.info(f'DONE: ALS recommendations with scores for test users generated successfully')
    return als_based_rec

# ---------- Ranked recommendations ---------- #
def get_ranked_recommendations(
    seed: int,
    preprocessed_dir: str,
    results_dir: str,
    models_dir: str,
    n_ranked: int,
    sample_users: int,
    als_recs: Dict[int, List[tuple]] = None,
    popularity_recs: Dict[int, List[tuple]] = None
) -> Dict[int, List[int]]:
    '''
        Generate ranked recommendations for test users using CatBoost model.
        
        Uses pre-computed ALS and popularity recommendations to avoid double computation.

        Args:
        - seed - seed for random number generator
        - preprocessed_dir - path to preprocessed data directory
        - results_dir - path to results directory
        - models_dir - path to models directory
        - n_ranked - number of final recommendations to return
        - sample_users - number of users to sample for evaluation
        - als_recs - pre-computed ALS recommendations dict (user_id -> list of (track_id, score) tuples)
        - popularity_recs - pre-computed popularity recommendations dict (user_id -> list of (track_id, score) tuples)

        Returns:
        - dictionary of recommendations
    '''
    
    logger.info(f'Generating ranked recommendations using pre-computed candidates')
    
    if als_recs is None or popularity_recs is None:
        logger.error('ALS and popularity recommendations must be provided')
        return {}
    
    # Get common users between ALS and popularity recommendations
    als_users = set(als_recs.keys())
    pop_users = set(popularity_recs.keys())
    valid_users = list(als_users & pop_users)
    
    if not valid_users:
        logger.warning('No valid users found for ranked recommendations')
        return {}
    
    # Sample users if needed (BEFORE converting to candidates to save memory)
    if sample_users and len(valid_users) > sample_users:
        np.random.seed(seed)
        valid_users = np.random.choice(valid_users, sample_users, replace=False).tolist()
        logger.info(f'Sampled {len(valid_users):,} users for ranked evaluation')
    else:
        logger.info(f'Using {len(valid_users):,} users for ranked evaluation')
    
    # Convert ALS recommendations dict to DataFrame with actual ALS scores
    # ALS recs format: {user_id: [(track_id, score), ...]}
    als_rows = []
    for user_id in valid_users:
        if user_id in als_recs:
            track_scores = als_recs[user_id]
            for track_id, score in track_scores:
                als_rows.append({
                    'user_id': user_id,
                    'track_id': track_id,
                    'als_score': score
                })
    
    als_candidates = pl.DataFrame(als_rows) if als_rows else pl.DataFrame({'user_id': [], 'track_id': [], 'als_score': []})
    
    # Normalize ALS scores to 0-1 range for consistency with other features
    if als_candidates.height > 0:
        max_als = als_candidates['als_score'].max()
        min_als = als_candidates['als_score'].min()
        if max_als > min_als:
            als_candidates = als_candidates.with_columns(
                ((pl.col('als_score') - min_als) / (max_als - min_als)).alias('als_score')
            )
    
    logger.info(f'Converted {als_candidates.height:,} ALS recommendations with actual scores for {len(valid_users):,} users')
    
    # Convert popularity recommendations dict to DataFrame with actual popularity scores
    # Popularity recs format: {user_id: [(track_id, score), ...]}
    pop_rows = []
    for user_id in valid_users:
        if user_id in popularity_recs:
            track_scores = popularity_recs[user_id]
            for track_id, score in track_scores:
                pop_rows.append({
                    'user_id': user_id,
                    'track_id': track_id,
                    'popularity_score': score
                })
    
    popularity_candidates = pl.DataFrame(pop_rows) if pop_rows else pl.DataFrame({'user_id': [], 'track_id': [], 'popularity_score': []})
    
    # Normalize popularity scores to 0-1 range for consistency with other features
    if popularity_candidates.height > 0:
        max_pop = popularity_candidates['popularity_score'].max()
        min_pop = popularity_candidates['popularity_score'].min()
        if max_pop > min_pop:
            popularity_candidates = popularity_candidates.with_columns(
                ((pl.col('popularity_score') - min_pop) / (max_pop - min_pop)).alias('popularity_score')
            )
    
    logger.info(f'Converted {popularity_candidates.height:,} popularity recommendations with actual scores for {len(valid_users):,} users')
    
    # Generate ranked recommendations
    ranked_recs = generate_ranked_recommendations(
        preprocessed_dir=preprocessed_dir,
        results_dir=results_dir,
        models_dir=models_dir,
        n=n_ranked,
        sample_users=None,  # Already sampled above
        test_user_ids=valid_users,
        als_candidates=als_candidates,
        popularity_candidates=popularity_candidates
    )
    
    logger.info(f'DONE: Ranked recommendations for {len(ranked_recs):,} users generated successfully')
    return ranked_recs

# ---------- Evaluation ---------- #
def evaluate_model(
    seed: int,
    model_name: str = None, 
    preprocessed_dir: str = None, 
    models_dir: str = None,
    k_value: int = None, 
    output_dir: str = None, 
    n_popular: int = None,
    n_recs_popular: int = None,
    n_als: int = None,
    n_ranked: int = None,
    sample_users: int = None,
    als_recs: Dict[int, List[tuple]] = None,
    popularity_recs: Dict[int, List[tuple]] = None
) -> Dict:
    '''
        Evaluate a model and save results to JSON.

        Args:
        - seed - seed for random number generator
        - model_name - name of the model
        - preprocessed_dir - path to preprocessed data directory
        - models_dir - path to models directory
        - k_value - number of recommendations to evaluate
        - output_dir - path to output directory
        - als_recs - pre-computed ALS recommendations with scores (user_id -> [(track_id, score), ...])
        - popularity_recs - pre-computed popularity recommendations with scores (user_id -> [(track_id, score), ...])

        Returns:
        - dictionary of results including 'recommendations' key with the computed recs
    '''
    # Load defaults from environment if not provided
    if model_name is None:
        model_name = 'all'
    if preprocessed_dir is None:
        preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
    if models_dir is None:
        models_dir = os.getenv('MODELS_DIR', './models')
    if k_value is None:
        k_value = int(os.getenv('EVALUATION_K_VALUE', 10))
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
        sample_users_env = os.getenv('EVALUATION_SAMPLE_USERS', '10000')
        sample_users = None if sample_users_env in (None, '', 'None') else int(sample_users_env)
    
    # Handle 'all' model evaluation - pass recommendations between models to avoid recomputation
    if model_name == 'all':
        results = {}
        
        # Evaluate popularity model first
        results['popularity'] = evaluate_model(
            seed=seed,
            model_name='popularity',
            preprocessed_dir=preprocessed_dir,
            models_dir=models_dir,
            k_value=k_value,
            output_dir=output_dir,
            n_popular=n_popular,
            n_recs_popular=n_recs_popular,
            n_als=n_als,
            n_ranked=n_ranked,
            sample_users=sample_users
        )
        popularity_recs = results['popularity'].get('recommendations', {})
        
        # Evaluate ALS model
        results['als'] = evaluate_model(
            seed=seed,
            model_name='als',
            preprocessed_dir=preprocessed_dir,
            models_dir=models_dir,
            k_value=k_value,
            output_dir=output_dir,
            n_popular=n_popular,
            n_recs_popular=n_recs_popular,
            n_als=n_als,
            n_ranked=n_ranked,
            sample_users=sample_users
        )
        als_recs = results['als'].get('recommendations', {})
        
        # Evaluate ranked model using pre-computed ALS and popularity recommendations
        results['ranked'] = evaluate_model(
            seed=seed,
            model_name='ranked',
            preprocessed_dir=preprocessed_dir,
            models_dir=models_dir,
            k_value=k_value,
            output_dir=output_dir,
            n_popular=n_popular,
            n_recs_popular=n_recs_popular,
            n_als=n_als,
            n_ranked=n_ranked,
            sample_users=sample_users,
            als_recs=als_recs,
            popularity_recs=popularity_recs
        )
        
        compare_models(output_dir)
        return results
    
    logger.info(f'Evaluating {model_name} model')
    
    # Load data
    catalog = pl.read_parquet(f'{preprocessed_dir}/tracks_catalog_clean.parquet')
    events = pl.read_parquet(f'{preprocessed_dir}/events.parquet')
    train_events = pl.read_parquet(f'{preprocessed_dir}/train_events.parquet')
    test_events = pl.read_parquet(f'{preprocessed_dir}/test_events.parquet')
    train_matrix = load_npz(f'{preprocessed_dir}/train_matrix.npz')
    
    # Get recommendations based on model type
    # recs_with_scores: dict for passing to ranked model (includes scores)
    # recs_for_eval: dict for metric evaluation (just track IDs)
    if model_name == 'popularity':
        # Popularity returns {user_id: [(track_id, score), ...]}
        recs_with_scores = get_popular_recommendations(preprocessed_dir, output_dir, train_events, test_events, catalog, n_popular, n_recs_popular, method='listen_count')
        # Extract just track IDs for evaluation
        recs_for_eval = {
            user_id: [track_id for track_id, score in track_scores]
            for user_id, track_scores in recs_with_scores.items()
        }
        recs = recs_for_eval
    elif model_name == 'als':
        # ALS returns {user_id: [(track_id, score), ...]}
        recs_with_scores = get_als_recommendations(models_dir, train_matrix, test_events, n_als)
        # Extract just track IDs for evaluation
        recs_for_eval = {
            user_id: [track_id for track_id, score in track_scores]
            for user_id, track_scores in recs_with_scores.items()
        }
        recs = recs_for_eval
    elif model_name == 'ranked':
        # Use pre-computed recommendations if provided, otherwise fail
        if als_recs is None or popularity_recs is None:
            logger.error('Ranked model requires pre-computed ALS and popularity recommendations when called individually')
            logger.info('Please evaluate all models together or provide als_recs and popularity_recs')
            return {}
        recs = get_ranked_recommendations(
            seed,
            preprocessed_dir, output_dir, models_dir, 
            n_ranked, sample_users,
            als_recs=als_recs, popularity_recs=popularity_recs
        )
        recs_with_scores = recs  # ranked returns {user_id: [track_id, ...]}
        recs_for_eval = recs
    else:
        logger.error(f'Unknown model: {model_name}')
        return {}
    
    # Free train_matrix early as it's no longer needed
    del train_matrix
    gc.collect()
    
    if not recs_for_eval:
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
        'evaluated_users': len(recs_for_eval),
        'metrics': {},
        'recommendations': recs_with_scores  # Include recommendations (with scores for ALS) for reuse by ranked model
    }
    
    metrics = evaluator.all_metrics(recs_for_eval, test_items, k_value)
    results['metrics'][f'k={k_value}'] = metrics
    logger.info(f'K={k_value}: P={metrics["precision@k"]:.4f} R={metrics["recall@k"]:.4f} NDCG={metrics["ndcg@k"]:.4f}')
    
    # Free remaining large objects (but keep recs in results for reuse)
    del evaluator, test_items
    gc.collect()
    
    # Save locally (without recommendations to keep file small)
    os.makedirs(output_dir, exist_ok=True)
    results_to_save = {k: v for k, v in results.items() if k != 'recommendations'}
    with open(f'{output_dir}/evaluation_{model_name}.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    logger.info(f'DONE: Saved evaluation_{model_name}.json to {output_dir}')

    logger.info('DONE: Evaluation completed successfully')
    
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
        os.path.join(results_dir, 'evaluation_als.json'),
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
    logger.info(f'DONE: Saved models_comparison.parquet to {models_comparison_path}')
    
    return models_comparison   

# ---------- Main entry point ---------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate recommendation models')
    parser.add_argument('--model', choices=['popularity', 'als', 'ranked', 'all'], default='all')
    args = parser.parse_args()
    
    logger.info('Running recommendation models evaluation')

    # Check required environment variables
    required_env_vars = [
        'SEED',
        'PREPROCESSED_DATA_DIR', 
        'RESULTS_DIR', 
        'MODELS_DIR',
        'POPULARITY_METHOD',
        'POPULARITY_TOP_N',
        'POPULARITY_N_RECS',
        'POPULARITY_WITH_METADATA',
        'POPULARITY_FILTER_LISTENED',
        'ALS_N_RECS',
        'ALS_NUM_THREADS',
        'ALS_FACTORS',
        'ALS_REGULARIZATION',
        'ALS_ITERATIONS',
        'ALS_ALPHA',
        'RANKED_N_RECS',
        'EVALUATION_K_VALUE',
        'EVALUATION_SAMPLE_USERS',
        'EVALUATION_N_RECS',
        'EVALUATION_MODELS'
    ]
    missing_vars = [var for var in required_env_vars if os.getenv(var) is None]
    if missing_vars:
        raise EnvironmentError(f'Missing required environment variables: {", ".join(missing_vars)}')

    # Load from env
    seed = int(os.getenv('SEED', 42))
    preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
    results_dir = os.getenv('RESULTS_DIR', './results')
    models_dir = os.getenv('MODELS_DIR', './models')
    method = os.getenv('POPULARITY_METHOD')
    k_value = int(os.getenv('EVALUATION_K_VALUE', 10))
    sample_users_env = os.getenv('EVALUATION_SAMPLE_USERS', '10000')
    sample_users = None if sample_users_env in (None, '', 'None') else int(sample_users_env)
    n_popular = int(os.getenv('POPULARITY_TOP_N', 100))
    n_recs_popular = int(os.getenv('POPULARITY_N_RECS', 10))
    n_als = int(os.getenv('ALS_N_RECS', 10))
    n_ranked = int(os.getenv('RANKED_N_RECS', 10))
    model_selection = args.model or os.getenv('EVALUATION_MODELS', 'all')
    
    if model_selection == 'all':
        # Evaluate all models together to allow reuse of recommendations
        evaluate_model(
            seed=seed,
            model_name='all',
            preprocessed_dir=preprocessed_dir,
            models_dir=models_dir,
            k_value=k_value,
            output_dir=results_dir,
            n_popular=n_popular,
            n_recs_popular=n_recs_popular,
            n_als=n_als,
            n_ranked=n_ranked,
            sample_users=sample_users
        )
    elif model_selection == 'ranked':
        # For ranked model alone, we need to first compute popularity and ALS
        logger.info('Ranked model requires pre-computed popularity and ALS recommendations')
        logger.info('Computing popularity recommendations')
        pop_results = evaluate_model(
            seed=seed,
            model_name='popularity',
            preprocessed_dir=preprocessed_dir,
            models_dir=models_dir,
            k_value=k_value,
            output_dir=results_dir,
            n_popular=n_popular,
            n_recs_popular=n_recs_popular,
            n_als=n_als,
            n_ranked=n_ranked,
            sample_users=sample_users
        )
        logger.info('Computing ALS recommendations')
        als_results = evaluate_model(
            seed=seed,
            model_name='als',
            preprocessed_dir=preprocessed_dir,
            models_dir=models_dir,
            k_value=k_value,
            output_dir=results_dir,
            n_popular=n_popular,
            n_recs_popular=n_recs_popular,
            n_als=n_als,
            n_ranked=n_ranked,
            sample_users=sample_users
        )
        logger.info('Computing ranked recommendations')
        evaluate_model(
            seed=seed,
            model_name='ranked',
            preprocessed_dir=preprocessed_dir,
            models_dir=models_dir,
            k_value=k_value,
            output_dir=results_dir,
            n_popular=n_popular,
            n_recs_popular=n_recs_popular,
            n_als=n_als,
            n_ranked=n_ranked,
            sample_users=sample_users,
            als_recs=als_results.get('recommendations', {}),
            popularity_recs=pop_results.get('recommendations', {})
        )
    else:
        # Evaluate single model (popularity or als)
        evaluate_model(
            seed=seed,
            model_name=model_selection,
            preprocessed_dir=preprocessed_dir,
            models_dir=models_dir,
            k_value=k_value,
            output_dir=results_dir,
            n_popular=n_popular,
            n_recs_popular=n_recs_popular,
            n_als=n_als,
            n_ranked=n_ranked,
            sample_users=sample_users
        )
    
    gc.collect()
    logger.info('DONE: Recommendation models evaluation completed successfully')

    # ---------- All exports ---------- #
__all__ = ['evaluate_model']