'''
    Recommendation Ranking

    This module provides functionality to train and evaluate a recommendation ranking model using CatBoost.

    Pipeline:
    1. Load candidates from ALS + similar-based sources
    2. Add target labels from events
    3. Filter users with positives and sample negatives
    4. Train CatBoostClassifier
    5. Predict and extract top-K recommendations

    Usage examples:
    python -m src.rec_ranking # run full pipeline
    python -m src.rec_ranking --sample_users 10000 # number of users to sample (default: None)
    python -m src.rec_ranking --top_k 10 # number of recommendations per user (default: 10)
    python -m src.rec_ranking --iterations 100 # number of iterations for CatBoost (default: 100)
    python -m src.rec_ranking --negatives_multiplier 4 # number of negative samples per user (default: 4)
'''

# ---------- Imports ---------- #
import os
import gc
import ast
import logging
import argparse
from typing import List

import numpy as np
import polars as pl
from dotenv import load_dotenv
from catboost import CatBoostClassifier, Pool

from src.s3_loading import upload_recommendations_to_s3
from src.logging_set_up import setup_logging

# ---------- Load environment variables ---------- #
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))

# ---------- Logging setup ---------- #
logger = setup_logging('rec_ranking')

# ---------- RecommendationRanker class ---------- #
class RecommendationRanker:
    '''
        Ranking class for applying multi-objective optimization to recommendations.

        Args:
        - catalog - catalog dataframe

        Attributes:
        - catalog - catalog dataframe
        - track_genres - dictionary of track ids and genres
        - track_artists - dictionary of track ids and artists

        Methods:
        - rank_multi_objective - rank recommendations using multi-objective optimization
    '''
    
    def __init__(self, catalog: pl.DataFrame):
        self.catalog = catalog
        self.track_genres = dict(zip(
            catalog['track_id'].to_list(),
            catalog['genre_id'].to_list()
        )) if 'genre_id' in catalog.columns else {}
        self.track_artists = dict(zip(
            catalog['track_id'].to_list(),
            catalog['artist_id'].to_list()
        )) if 'artist_id' in catalog.columns else {}
    
    def rank_multi_objective(
        self,
        recommendations: List[tuple],
        popularity_scores: dict,
        user_history: set,
        n: int = 10,
        diversity_weight: float = 0.3,
        popularity_weight: float = 0.1,
        novelty_weight: float = 0.2
    ) -> List[tuple]:
        '''
            Rank recommendations using multi-objective optimization:
            - Relevance (original recommendation score)
            - Popularity (normalized popularity score)
            - Novelty (1 if not in history, 0 otherwise)
            - Diversity (prefer different genres)
            - Combined score = (1 - diversity_weight - popularity_weight - novelty_weight) * relevance + diversity_weight * diversity_score + popularity_weight * pop_score + novelty_weight * novelty_score
            - Greedy selection to maximize diversity
            - Return the top n recommendations.

            Args:
            - recommendations - list of recommendations
            - popularity_scores - dictionary of popularity scores
            - user_history - set of user history
            - n - number of recommendations to return
            - diversity_weight - weight for diversity
            - popularity_weight - weight for popularity
            - novelty_weight - weight for novelty

            Returns:
            - list of recommendations
        '''
        if not recommendations:
            return []
        
        # Normalize popularity scores
        max_pop = max(popularity_scores.values()) if popularity_scores else 1
        
        # Calculate scores for each recommendation
        scored_recs = []
        selected_genres = set()
        
        for track_id, base_score in recommendations:
            # Skip if track is in user history
            if track_id in user_history:
                continue
            
            # Relevance score (original recommendation score)
            relevance = base_score
            
            # Popularity score (normalized)
            pop_score = popularity_scores.get(track_id, 0) / max_pop if max_pop > 0 else 0
            
            # Novelty score (1 if not in history, 0 otherwise)
            novelty_score = 1.0 if track_id not in user_history else 0.0
            
            # Diversity score (prefer different genres)
            genre = self.track_genres.get(track_id)
            diversity_score = 0.0 if genre in selected_genres else 1.0
            
            # Combined score
            combined_score = (
                (1 - diversity_weight - popularity_weight - novelty_weight) * relevance +
                diversity_weight * diversity_score +
                popularity_weight * pop_score +
                novelty_weight * novelty_score
            )
            
            scored_recs.append((track_id, combined_score, genre))
        
        # Greedy selection to maximize diversity
        final_recs = []
        remaining = scored_recs.copy()
        
        while len(final_recs) < n and remaining:
            # Re-score remaining items based on current selection
            for i, (track_id, score, genre) in enumerate(remaining):
                diversity_bonus = 0.0 if genre in selected_genres else diversity_weight
                remaining[i] = (track_id, score + diversity_bonus, genre)
            
            # Sort by score and pick best
            remaining.sort(key=lambda x: x[1], reverse=True)
            best = remaining.pop(0)
            
            final_recs.append((best[0], best[1]))
            if best[2] is not None:
                selected_genres.add(best[2])
        
        # Free up memory
        del remaining
        gc.collect()

        # Return final recommendations with combined score
        return final_recs

# ---------- Custom features ---------- #
def compute_track_features(catalog: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    '''
        Compute custom features for each track:
            - genre_popularity - normalized popularity of the genre (total listens)
            - artist_popularity - normalized popularity of the artist (total listens)
            - track_group_size - number of tracks in the same group (different versions of same song), 
            an indicator of track popularity and uniqueness

            Args:
            - catalog - catalog dataframe
            - events - events dataframe

            Returns:
            - dataframe with track features
    '''
    logger.info('Computing track custom features')

    logger.info('Computing genre popularity')
    # Genre popularity (normalized total listens per genre)
    genre_listens = (
        events
            .join(catalog.select(['track_id', 'genre_id']), on='track_id', how='left')
            .group_by('genre_id')
            .agg(pl.sum('listen_count').alias('genre_total'))
    )
    max_genre = genre_listens['genre_total'].max()
    genre_pop = (
        genre_listens
            .with_columns(
                (pl.col('genre_total') / max_genre).alias('genre_popularity')
            )
            .select(['genre_id', 'genre_popularity'])
    )
    logger.info(f'Genre popularity computed for {genre_pop.height:,} genres')
    
    logger.info('Computing artist popularity')
    # Artist popularity (normalized total listens per artist)
    artist_listens = (
        events
            .join(catalog.select(['track_id', 'artist_id']), on='track_id', how='left')
            .group_by('artist_id')
            .agg(pl.sum('listen_count').alias('artist_total'))
    )
    max_artist = artist_listens['artist_total'].max()
    artist_pop = (
        artist_listens
            .with_columns(
                (pl.col('artist_total') / max_artist).alias('artist_popularity')
            )
            .select(['artist_id', 'artist_popularity'])
    )
    logger.info(f'Artist popularity computed for {artist_pop.height:,} artists')
    
    logger.info('Computing track group size')
    # Track group size (number of versions of the same song)
    group_size = (
        catalog
            .group_by('track_group_id')
            .agg(pl.count('track_id').alias('track_group_size'))
    )
    # Normalize to 0-1 range
    max_group = group_size['track_group_size'].max()
    group_size = (
        group_size
        .with_columns(
            (pl.col('track_group_size') / max_group).alias('track_group_size')
        )
        .select(['track_group_id', 'track_group_size'])
    )
    logger.info(f'Track group size computed for {group_size.height:,} track groups')
    
    # Join all features to a dataframe
    track_features = (
        catalog
            .select(['track_id', 'genre_id', 'artist_id', 'track_group_id'])
            .join(genre_pop, on='genre_id', how='left')
            .join(artist_pop, on='artist_id', how='left')
            .join(group_size, on='track_group_id', how='left')
            .select(['track_id', 'genre_popularity', 'artist_popularity', 'track_group_size'])
            .with_columns([
                pl.col('genre_popularity').fill_null(0.0),
                pl.col('artist_popularity').fill_null(0.0),
                pl.col('track_group_size').fill_null(0.0),
            ])
    )

    # Free up memory
    del genre_listens, artist_listens, group_size
    gc.collect()

    logger.info(f'Computed track custom features for {track_features.height:,} tracks')

    # Return custom track features
    return track_features

# ---------- Load popularity-based recommendations ---------- #
def load_popular_candidates(results_dir: str, user_ids: List[int]) -> pl.DataFrame:
    '''
        Load popularity-based candidates by expanding top popular tracks to all target users.

        Args:
        - results_dir - path to results directory
        - user_ids - list of user ids

        Returns:
        - dataframe with popularity-based candidates
    '''
    logger.info('Loading popularity-based candidates')

    popular_path = os.path.join(results_dir, 'top_popular.parquet')
    popular = pl.read_parquet(popular_path).select(['track_id', 'popularity_score'])
    
    # Normalize popularity score to 0-1 range
    max_pop = popular['popularity_score'].max()
    popular = popular.with_columns(
        (pl.col('popularity_score') / max_pop).alias('popularity_score')
    )
    
    # Create user-track pairs for all target users
    users_df = pl.DataFrame({'user_id': user_ids})
    popular_candidates = users_df.join(popular, how='cross')
    
    # Free up memory
    del popular
    gc.collect()

    logger.info(f'Generated {popular_candidates.height:,} popularity candidates for {len(user_ids):,} users')
    # Return popularity-based candidates
    return popular_candidates


# ---------- Load ALS recommendations ---------- #
def load_als_candidates(results_dir: str) -> pl.DataFrame:
    '''
        Load ALS collaborative filtering candidates.

        Args:
        - results_dir - path to results directory

        Returns:
        - dataframe with ALS candidates
    '''
    logger.info(f'Loading ALS candidates')

    als_path = os.path.join(results_dir, 'personal_als.parquet')
    als_candidates = pl.read_parquet(als_path).select(['user_id', 'track_id', pl.col('score').alias('als_score')])   

    # Free up memory
    gc.collect()

    logger.info(f'Loaded {als_candidates.height:,} ALS candidates for {als_candidates["user_id"].n_unique():,} users')
    # Return ALS candidates 
    return als_candidates

# ---------- Load similar-based recommendations---------- #
def load_similar_candidates(
    results_dir: str, 
    events: pl.DataFrame, 
    max_similar: int
) -> pl.DataFrame:
    '''
        Load similar-based candidates by expanding user history with similar tracks.

        Args:
        - results_dir - path to results directory
        - events - events dataframe
        - max_similar - maximum number of similar tracks to load

        Returns:
        - dataframe with similar-based candidates
    '''

    similar_path = os.path.join(results_dir, 'similar.parquet')
    logger.info(f'Loading similar-based candidates from {similar_path}')
    similar = pl.read_parquet(similar_path).filter(pl.col('rank') <= max_similar)
    user_tracks = events.select(['user_id', 'track_id']).unique()
    
    # Join user tracks with similar tracks and aggregate similarity scores
    similar_candidates = (
        user_tracks
            .join(similar.select(['track_id', 'similar_track_id', 'similarity_score']), on='track_id', how='inner')
            .select(['user_id', pl.col('similar_track_id').alias('track_id'), 'similarity_score'])
            .group_by(['user_id', 'track_id'])
            .agg(pl.max('similarity_score').alias('similar_score'))
    )
    
    # Free up memory
    gc.collect()

    logger.info(f'Generated {similar_candidates.height:,} candidates for {similar_candidates["user_id"].n_unique():,} users')
    # Return similar_based candidates
    return similar_candidates

# ---------- Load and merge candidates ---------- #
def load_and_merge_candidates(
    results_dir: str, 
    events: pl.DataFrame, 
    user_ids: List[int], 
    max_similar: int
) -> pl.DataFrame:
    '''
        Load and merge candidates from ALS + similar-based + popularity-based sources.

        Args:
        - results_dir - path to results directory
        - events - events dataframe
        - user_ids - list of user ids
        - max_similar - maximum number of similar tracks to load

        Returns:
        - dataframe with merged candidates
    '''
    logger.info('Loading and merging candidates from ALS + similar-based + popularity-based sources')
    
    # Load ALS candidates
    als_candidates = load_als_candidates(results_dir)
    
    # Filter candidates to target users
    if user_ids is not None:
        als_candidates = als_candidates.filter(pl.col('user_id').is_in(user_ids))
        events = events.filter(pl.col('user_id').is_in(user_ids))
        logger.info(f'Filtered to {len(user_ids):,} users')
    
    # Load similar-based candidates
    similar_candidates = load_similar_candidates(results_dir, events, max_similar)
    
    # Load popularity-based candidates
    popular_candidates = load_popular_candidates(results_dir, user_ids)
    
    # Merge candidates: ALS + similar
    candidates = (
        als_candidates
            .join(similar_candidates, on=['user_id', 'track_id'], how='full', coalesce=True)
            .with_columns([pl.col('als_score').fill_null(0.0), pl.col('similar_score').fill_null(0.0)])
    )
    
    # Merge with popularity candidates
    candidates = (
        candidates
            .join(popular_candidates, on=['user_id', 'track_id'], how='full', coalesce=True)
            .with_columns([
                pl.col('als_score').fill_null(0.0), 
                pl.col('similar_score').fill_null(0.0),
                pl.col('popularity_score').fill_null(0.0)
            ])
    )
    
    # Free up memory
    del als_candidates, similar_candidates, popular_candidates
    gc.collect()

    logger.info(f'Merged: {candidates.height:,} candidates, {candidates["user_id"].n_unique():,} users')
    # Return merged candidates
    return candidates

def add_target_labels(candidates: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    '''
        Add target labels: 1 if user listened to track, 0 otherwise.

        Args:
        - candidates - dataframe with candidates
        - events - events dataframe

        Returns:
        - dataframe with target labels
    '''
    logger.info('Adding target labels to candidates')
    
    # Create label pairs (user_id, track_id) with target label 1
    label_pairs = events.select(['user_id', 'track_id']).unique().with_columns(pl.lit(1).alias('target'))
    
    # Join label pairs with candidates and add target label
    candidates = (
        candidates
            .join(label_pairs, on=['user_id', 'track_id'], how='left')
            .with_columns(pl.col('target').fill_null(0).cast(pl.Int8))
    )
    
    # Count number of positive and negative examples
    n_pos = candidates.filter(pl.col('target') == 1).height

    logger.info(f'Number of positive examples: {n_pos:,}, Number of negative examples: {candidates.height - n_pos:,}')
    return candidates

# ---------- Filter users with positive examples ---------- #
def filter_users_with_positive(candidates: pl.DataFrame) -> pl.DataFrame:
    '''
        Keep only users with at least one positive example.

        Args:
        - candidates - dataframe with candidates

        Returns:
        - dataframe with users with positive examples
    '''  
    logger.info('Filtering users with positive examples')

    # Filter users with at least one positive example
    users_with_pos = candidates.filter(pl.col('target') == 1).select('user_id').unique()
    candidates = candidates.join(users_with_pos, on='user_id', how='semi')

    # Count number of users with positive examples
    logger.info(f"Users with positives: {candidates['user_id'].n_unique():,}")
    return candidates

# ---------- Sample negatives ---------- #
def sample_negatives(
    candidates: pl.DataFrame, 
    multiplier: int, 
    seed: int
) -> pl.DataFrame:
    '''
        Sample negatives for each user_id: max multiplier * positives per user.

        Args:
        - candidates - dataframe with candidates
        - multiplier - multiplier for number of negatives to sample
        - seed - seed for random number generator

        Returns:
        - dataframe with sampled negatives
    '''

    logger.info(f'Sampling negatives (max {multiplier}x positives per user)')
    
    # Filter positives and negatives
    positives = candidates.filter(pl.col('target') == 1)
    negatives = candidates.filter(pl.col('target') == 0)
    
    # Count number of positives per user
    pos_counts = (
        positives
            .group_by('user_id')
            .agg(
                (pl.count('track_id') * multiplier).alias('max_neg')
            )
    )
    
    # Sample negatives per user
    np.random.seed(seed)
    sampled_negatives = (
        negatives
            .join(pos_counts, on='user_id', how='left')
            .with_columns([pl.col('max_neg').fill_null(0), pl.Series('_rand', np.random.rand(negatives.height))])
            .sort(['user_id', '_rand'])
            .with_columns(pl.col('track_id').cum_count().over('user_id').alias('_rank'))
            .filter(pl.col('_rank') <= pl.col('max_neg'))
            .drop(['_rand', '_rank', 'max_neg'])
    )
    
    # Combine positives and sampled negatives
    train_data = pl.concat([positives, sampled_negatives])

    logger.info(f'Training samples: {train_data.height:,} (pos={positives.height:,}, neg={sampled_negatives.height:,})')
    return train_data

# ---------- Train classifier ---------- #
def train_classifier(
    train_data: pl.DataFrame, 
    features: List[str], 
    iterations: int, 
    seed: int
) -> CatBoostClassifier:
    '''
        Train CatBoostClassifier.

        Args:
        - train_data - dataframe with training data
        - features - list of features to use
        - iterations - number of iterations for CatBoost
        - seed - seed for random number generator

        Returns:
        - CatBoostClassifier model
    '''
    logger.info('Training CatBoostClassifier')
    
    # Convert to pandas and create pool
    train_pd = train_data.select(features + ['target']).to_pandas()
    pool = Pool(data=train_pd[features], label=train_pd['target'], feature_names=features)
    
    # Initialize and fit model
    model = CatBoostClassifier(
        iterations=iterations, learning_rate=0.1, depth=6,
        loss_function='Logloss', random_seed=seed, verbose=100
    )
    model.fit(pool)
    
    # Print resulting feature importances
    for name, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
        logger.info(f'{name}: {imp:.2f}')
    
    logger.info('CatBoostClassifier trained')
    return model

def predict_and_rank(
    model: CatBoostClassifier, 
    candidates: pl.DataFrame, 
    features: List[str]
) -> pl.DataFrame:
    '''
        Predict scores and rank candidates per user.

        Args:
        - model - CatBoostClassifier model
        - candidates - dataframe with candidates
        - features - list of features to use

        Returns:
        - dataframe with ranked candidates
    '''
    logger.info('Predicting and ranking')
    
    # Predict scores and rank candidates per user
    predictions = model.predict_proba(candidates.select(features).to_pandas())[:, 1]
    
    # Rank candidates per user
    ranked = (
        candidates
            .with_columns(pl.Series('score', predictions))
            .sort(['user_id', 'score'], descending=[False, True])
            .with_columns(pl.col('track_id').cum_count().over('user_id').alias('rank'))
    )

    logger.info(f'Ranked {ranked.height:,} candidates for {ranked["user_id"].n_unique():,} users')
    return ranked

def top_k_per_user(ranked: pl.DataFrame, k: int) -> pl.DataFrame:
    '''
        Extract top-K recommendations per user.

        Args:
        - ranked - dataframe with ranked candidates
        - k - number of recommendations to return

        Returns:
        - dataframe with top-K recommendations per user
    '''
    logger.info('Extracting top-K recommendations per user')

    # Filter top-K recommendations per user
    top_k = ranked.filter(pl.col('rank') <= k)

    logger.info(f'Top-{k}: {top_k.height:,} recommendations for {top_k["user_id"].n_unique():,} users')
    return top_k

# ---------- Run ranking pipeline ---------- #
def run_ranking_pipeline(
    preprocessed_dir: str = None, 
    results_dir: str = None, 
    models_dir: str = None, 
    features: List[str] = None, 
    top_k: int = None, 
    sample_users: int = None, 
    iterations: int = None, 
    negatives_multiplier: int = None, 
    max_similar: int = None, 
    seed: int = None
) -> None:
    '''
        Run the complete ranking pipeline.

        Args:
        - preprocessed_dir - path to preprocessed data directory
        - results_dir - path to results directory
        - models_dir - path to models directory
        - features - list of features to use
        - top_k - number of recommendations to return
        - sample_users - number of users to sample
    '''
    # Load defaults from environment if not provided
    if preprocessed_dir is None:
        preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
    if results_dir is None:
        results_dir = os.getenv('RESULTS_DIR', './results')
    if models_dir is None:
        models_dir = os.getenv('MODELS_DIR', './models')
    if features is None:
        features = ast.literal_eval(os.getenv('RANKING_FEATURES', "['als_score', 'similar_score', 'popularity_score', 'genre_popularity', 'artist_popularity', 'track_group_size']"))
    if top_k is None:
        top_k = int(os.getenv('RANKING_TOP_K', 10))
    if sample_users is None:
        sample_users = int(os.getenv('RANKING_SAMPLE_USERS', 10000))
    if iterations is None:
        iterations = int(os.getenv('RANKING_CATBOOST_ITERATIONS', 100))
    if negatives_multiplier is None:
        negatives_multiplier = int(os.getenv('RANKING_NEGATIVES_MULTIPLIER', 4))
    if max_similar is None:
        max_similar = int(os.getenv('RANKING_MAX_SIMILAR', 10))
    if seed is None:
        seed = int(os.getenv('SEED', 42))
    
    logger.info('Running recommendation ranking pipeline')
    
    np.random.seed(seed)
    
    # Determine target users (only load user_id column to save memory)
    logger.info('Determining target users')
    als_path = os.path.join(results_dir, 'personal_als.parquet')
    als_users = pl.read_parquet(als_path, columns=['user_id'])['user_id'].unique().to_list()
    
    # Sample users if requested
    if sample_users and len(als_users) > sample_users:
        target_users = np.random.choice(als_users, sample_users, replace=False).tolist()
        logger.info(f'Sampling {sample_users:,} users')
    else:
        target_users = als_users
        logger.info(f'Using all {len(als_users):,} users')
    
    # Free up memory
    del als_users
    
    # Load catalog and train events
    logger.info('Loading catalog and train events for feature computation')
    catalog = pl.read_parquet(f'{preprocessed_dir}/tracks_catalog_clean.parquet')
    train_events = pl.read_parquet(f'{preprocessed_dir}/train_events.parquet')
    
    # Compute track features
    logger.info('Computing track features')
    track_features = compute_track_features(catalog, train_events)
    
    # Filter train events to target users for candidate generation
    train_events_filtered = train_events.filter(pl.col('user_id').is_in(target_users))
    
    # Free up memory
    del train_events
    gc.collect()
    
    # Load and merge candidates
    logger.info('Loading candidates')
    candidates = load_and_merge_candidates(results_dir, train_events_filtered, target_users, max_similar)
    
    # Free up memory
    del train_events_filtered
    gc.collect()
    
    # Add track features to candidates
    logger.info('Adding track features to candidates')
    candidates = (
        candidates
            .join(track_features, on='track_id', how='left')
            .with_columns([
                pl.col('genre_popularity').fill_null(0.0),
                pl.col('artist_popularity').fill_null(0.0),
                pl.col('track_group_size').fill_null(0.0),
            ])
    )
    
    # Free up memory
    del track_features, catalog
    gc.collect()
    
    # Load test events
    logger.info('Loading test events for labels')
    test_events = pl.read_parquet(f'{preprocessed_dir}/test_events.parquet')
    
    # Add labels and filter
    logger.info('Adding labels and filtering users with positive examples')
    candidates = add_target_labels(candidates, test_events)
    
    # Free up memory
    del test_events
    gc.collect()
    
    # Filter users with positive examples
    logger.info('Filtering users with positive examples')
    candidates = filter_users_with_positive(candidates)
    
    # Sample negatives
    logger.info('Sampling negatives')
    train_data = sample_negatives(candidates, negatives_multiplier, seed)
    
    # Train model
    logger.info('Training model')
    model = train_classifier(train_data, features, iterations, seed)
    
    # Free up memory
    del train_data
    gc.collect()

    # Save model
    os.makedirs(models_dir, exist_ok=True)
    model.save_model(f'{models_dir}/catboost_classifier.cbm')
    logger.info(f'Saved model to {models_dir}/catboost_classifier.cbm')
    
    # Predict and get top-K
    logger.info('Predicting and ranking')
    ranked = predict_and_rank(model, candidates, features)
    
    # Free up memory
    del candidates
    gc.collect()
    
    # Extract top-K recommendations
    final_recs = top_k_per_user(ranked, top_k)
    
    # Free up memory
    del ranked
    gc.collect()
    
    # Save recommendations
    output_path = os.path.join(results_dir, 'recommendations.parquet')
    final_recs.write_parquet(output_path)
    logger.info(f'Saved {final_recs.height:,} recommendations to {output_path}')

    # Upload recommendations to S3
    upload_recommendations_to_s3(output_path, 'recommendations.parquet')    
    logger.info(f'Uploaded {final_recs.height:,} recommendations to S3')

    # Free up memory
    del final_recs
    gc.collect()
    
    return None

# ---------- Generate ranked recommendations ---------- #
def generate_ranked_recommendations(
    preprocessed_dir: str, 
    results_dir: str,
    models_dir: str, 
    n: int = 10,
    sample_users: int = None
) -> dict:
    '''
    Generate ranked recommendations using pre-trained CatBoost model.
    
    Args:
        preprocessed_dir: Path to preprocessed data directory
        models_dir: Path to models directory
        n: Number of recommendations per user
        sample_users: Optional number of users to sample
        
    Returns:
        Dictionary mapping user_id to list of track_ids
    '''
    logger.info('Generating ranked recommendations')
       
    # Check if recommendations already exist
    recs_path = os.path.join(results_dir, 'recommendations.parquet')
    if os.path.exists(recs_path):
        logger.info(f'Loading existing recommendations from {recs_path}')
        recs_df = pl.read_parquet(recs_path)
        
        # Convert to dictionary format
        recs = {}
        for user_id in recs_df['user_id'].unique().to_list():
            user_recs = recs_df.filter(pl.col('user_id') == user_id).sort('rank')['track_id'].to_list()[:n]
            recs[user_id] = user_recs
        
        logger.info(f'Loaded recommendations for {len(recs):,} users')
        return recs
    
    # Load model
    model_path = os.path.join(models_dir, 'catboost_classifier.cbm')
    if not os.path.exists(model_path):
        logger.error(f'CatBoost model not found at {model_path}')
        logger.info('Please run the ranking pipeline first: python -m src.rec_ranking')
        return {}
    
    logger.info(f'Loading CatBoost model from {model_path}')
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    # Load ALS candidates
    als_path = os.path.join(results_dir, 'personal_als.parquet')
    if not os.path.exists(als_path):
        logger.error(f'ALS candidates not found at {als_path}')
        return {}
    
    candidates = pl.read_parquet(als_path)
    
    # Sample users if requested
    if sample_users:
        users = candidates['user_id'].unique().to_list()
        if len(users) > sample_users:
            np.random.seed(42)
            sampled = np.random.choice(users, sample_users, replace=False).tolist()
            candidates = candidates.filter(pl.col('user_id').is_in(sampled))
            logger.info(f'Sampled {sample_users:,} users')
    
    # Load track features
    catalog = pl.read_parquet(f'{preprocessed_dir}/tracks_catalog_clean.parquet')
    train_events = pl.read_parquet(f'{preprocessed_dir}/train_events.parquet')
    track_features = compute_track_features(catalog, train_events)
    
    # Free catalog and train_events after computing features
    del catalog, train_events
    gc.collect()
    
    # Add features to candidates
    candidates = candidates.join(track_features, on='track_id', how='left')
    candidates = candidates.with_columns([
        pl.col('genre_popularity').fill_null(0.0),
        pl.col('artist_popularity').fill_null(0.0),
        pl.col('track_group_size').fill_null(0.0),
    ])
    
    # Free track_features after joining
    del track_features
    gc.collect()
    
    # Rename score column if needed
    if 'score' in candidates.columns and 'als_score' not in candidates.columns:
        candidates = candidates.rename({'score': 'als_score'})
    
    # Add missing feature columns with defaults
    for col in ['similar_score', 'popularity_score']:
        if col not in candidates.columns:
            candidates = candidates.with_columns(pl.lit(0.0).alias(col))
    
    # Get features used by model
    features = model.feature_names_
    
    # Predict and rank
    logger.info('Predicting scores')
    predictions = model.predict_proba(candidates.select(features).to_pandas())[:, 1]
    
    ranked = (
        candidates
        .with_columns(pl.Series('pred_score', predictions))
        .sort(['user_id', 'pred_score'], descending=[False, True])
        .with_columns(pl.col('track_id').cum_count().over('user_id').alias('rank'))
        .filter(pl.col('rank') <= n)
    )
    
    # Convert to dictionary
    recs = {}
    for user_id in ranked['user_id'].unique().to_list():
        user_recs = ranked.filter(pl.col('user_id') == user_id).sort('rank')['track_id'].to_list()
        recs[user_id] = user_recs
    
    logger.info(f'Generated recommendations for {len(recs):,} users')
    return recs

# ---------- Main entry point ---------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run recommendation ranking pipeline')
    parser.add_argument('--sample_users', type=int, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--negatives_multiplier', type=int, default=None)
    args = parser.parse_args()
    
    logger.info('Running recommendation ranking pipeline')

    # Check required environment variables
    required_env_vars = [
        'PREPROCESSED_DATA_DIR', 
        'RESULTS_DIR', 
        'MODELS_DIR', 
        'SEED',
        'RANKING_FEATURES', 
        'RANKING_NEGATIVES_MULTIPLIER', 
        'RANKING_SAMPLE_USERS',
        'RANKING_TOP_K', 
        'RANKING_CATBOOST_ITERATIONS', 
        'RANKING_MAX_SIMILAR'
    ]
    missing_vars = [var for var in required_env_vars if os.getenv(var) is None]
    if missing_vars:
        raise EnvironmentError(f'Missing required environment variables: {", ".join(missing_vars)}')

    # Load from env
    preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR')
    results_dir = os.getenv('RESULTS_DIR')
    models_dir = os.getenv('MODELS_DIR')
    seed = int(os.getenv('SEED'))
    features = ast.literal_eval(os.getenv('RANKING_FEATURES'))
    top_k = args.top_k or int(os.getenv('RANKING_TOP_K'))
    sample_users = args.sample_users or int(os.getenv('RANKING_SAMPLE_USERS'))
    iterations = args.iterations or int(os.getenv('RANKING_CATBOOST_ITERATIONS'))
    negatives_multiplier = args.negatives_multiplier or int(os.getenv('RANKING_NEGATIVES_MULTIPLIER'))
    max_similar = int(os.getenv('RANKING_MAX_SIMILAR'))

    run_ranking_pipeline(
        preprocessed_dir, 
        results_dir, 
        models_dir, 
        features, 
        top_k, 
        sample_users, 
        iterations, 
        negatives_multiplier, 
        max_similar, 
        seed
    )

    logger.info('Recommendation ranking pipeline completed')
    