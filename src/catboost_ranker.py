"""
CatBoost Ranking Model for Recommendations (Pure Polars)

Approach:
1. Split test_events by date into train/eval parts
2. Generate candidates from ALS + Popular tracks
3. Merge candidates, add features and labels
4. Train CatBoost classifier
5. Rank and evaluate

Features:
- als_score: ALS collaborative filtering score
- popularity_score: Track popularity score  
- artist_popularity: Artist average popularity
- genre_popularity: Genre average popularity
- track_group_size: Number of versions of track

Usage:
    python3 -m src.catboost_ranker --preprocessed-dir data/preprocessed
"""

import logging
import numpy as np
import polars as pl
from scipy.sparse import load_npz
from catboost import CatBoostClassifier, Pool

from src.collaborative_rec import load_als_model

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

FEATURES = ['als_score', 'popularity_score', 'artist_popularity', 'genre_popularity', 'track_group_size']
NEGATIVES_PER_USER = 4


# ============================================================================
# DATA PREPARATION
# ============================================================================
def split_test_by_date(test_events: pl.DataFrame, train_ratio: float = 0.5) -> tuple:
    """Split test events by date into train/eval parts."""
    min_date = test_events['last_listen'].min()
    max_date = test_events['last_listen'].max()
    split_days = int((max_date - min_date).days * train_ratio)
    split_date = min_date + pl.duration(days=split_days)
    
    train_part = test_events.filter(pl.col('last_listen') < split_date)
    eval_part = test_events.filter(pl.col('last_listen') >= split_date)
    
    logger.info(f"Split test by date: train={train_part.height:,}, eval={eval_part.height:,}")
    return train_part, eval_part


def compute_track_features(catalog: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Compute features for each track using Polars."""
    logger.info("Computing track features...")
    
    # Artist popularity (normalized)
    artist_pop = (
        events.join(catalog.select(['track_id', 'artist_id']), on='track_id', how='left')
        .group_by('artist_id')
        .agg(pl.sum('listen_count').alias('artist_total'))
    )
    max_artist = artist_pop['artist_total'].max()
    artist_pop = artist_pop.with_columns(
        (pl.col('artist_total') / max_artist).alias('artist_popularity')
    ).select(['artist_id', 'artist_popularity'])
    
    # Genre popularity (normalized)
    genre_pop = (
        events.join(catalog.select(['track_id', 'genre_id']), on='track_id', how='left')
        .group_by('genre_id')
        .agg(pl.sum('listen_count').alias('genre_total'))
    )
    max_genre = genre_pop['genre_total'].max()
    genre_pop = genre_pop.with_columns(
        (pl.col('genre_total') / max_genre).alias('genre_popularity')
    ).select(['genre_id', 'genre_popularity'])
    
    # Track group size
    group_size = (
        catalog.group_by('track_group_id')
        .agg(pl.count('track_id').alias('track_group_size'))
    )
    
    # Join all to catalog
    track_features = (
        catalog.select(['track_id', 'artist_id', 'genre_id', 'track_group_id'])
        .join(artist_pop, on='artist_id', how='left')
        .join(genre_pop, on='genre_id', how='left')
        .join(group_size, on='track_group_id', how='left')
        .select(['track_id', 'artist_popularity', 'genre_popularity', 'track_group_size'])
        .fill_null(0)
    )
    
    logger.info(f"  Computed features for {track_features.height:,} tracks")
    return track_features


def get_popular_tracks(events: pl.DataFrame, n: int = 100) -> pl.DataFrame:
    """Get top N popular tracks with normalized scores."""
    pop = (
        events.group_by('track_id')
        .agg(pl.sum('listen_count').alias('pop'))
        .sort('pop', descending=True)
        .head(n)
    )
    max_pop = pop['pop'].max()
    return pop.with_columns(
        (pl.col('pop') / max_pop).alias('popularity_score')
    ).select(['track_id', 'popularity_score'])


def generate_als_candidates(als_model, train_matrix, user_ids: list, n: int = 50) -> pl.DataFrame:
    """Generate ALS recommendations for users."""
    logger.info(f"Generating ALS candidates for {len(user_ids):,} users...")
    
    records = []
    for i, user_id in enumerate(user_ids):
        if i % 5000 == 0 and i > 0:
            logger.info(f"  {i:,} users processed")
        
        if user_id not in als_model.user_encoder:
            continue
        
        try:
            user_idx = als_model.user_encoder[user_id]
            if user_idx >= train_matrix.shape[0]:
                continue
            
            track_indices, scores = als_model.model.recommend(
                user_idx, train_matrix[user_idx], N=n, filter_already_liked_items=True
            )
            
            for idx, score in zip(track_indices, scores):
                if idx in als_model.track_decoder:
                    records.append((user_id, als_model.track_decoder[idx], float(score)))
        except (IndexError, KeyError):
            continue
    
    df = pl.DataFrame(records, schema=['user_id', 'track_id', 'als_score'], orient='row')
    logger.info(f"  Generated {df.height:,} ALS candidates")
    return df


def generate_popular_candidates(popular_tracks: pl.DataFrame, user_ids: list) -> pl.DataFrame:
    """Generate popular track candidates for all users."""
    logger.info(f"Generating popular candidates for {len(user_ids):,} users...")
    
    # Cross join: all users × all popular tracks
    users_df = pl.DataFrame({'user_id': user_ids})
    candidates = users_df.join(popular_tracks, how='cross')
    
    logger.info(f"  Generated {candidates.height:,} popular candidates")
    return candidates


def prepare_candidates(
    als_candidates: pl.DataFrame,
    popular_candidates: pl.DataFrame,
    track_features: pl.DataFrame,
    label_events: pl.DataFrame
) -> pl.DataFrame:
    """Merge candidates from different sources and add labels."""
    logger.info("Merging candidates...")
    
    # Outer join ALS and Popular candidates
    candidates = als_candidates.join(
        popular_candidates,
        on=['user_id', 'track_id'],
        how='full',
        coalesce=True
    ).fill_null(0)
    
    logger.info(f"  After merge: {candidates.height:,} candidates")
    
    # Add track features
    candidates = candidates.join(track_features, on='track_id', how='left').fill_null(0)
    
    # Add target labels (1 = user listened, 0 = not)
    label_pairs = label_events.select(['user_id', 'track_id']).unique().with_columns(
        pl.lit(1).alias('target')
    )
    
    candidates = candidates.join(
        label_pairs,
        on=['user_id', 'track_id'],
        how='left'
    ).with_columns(
        pl.col('target').fill_null(0).cast(pl.Int32)
    )
    
    n_pos = candidates.filter(pl.col('target') == 1).height
    n_neg = candidates.filter(pl.col('target') == 0).height
    logger.info(f"  Positives: {n_pos:,}, Negatives: {n_neg:,}")
    
    return candidates


def sample_for_training(candidates: pl.DataFrame, negatives_per_user: int = 4) -> pl.DataFrame:
    """Sample training data: all positives + N negatives per user."""
    logger.info("Sampling training data...")
    
    # Users with at least one positive
    users_with_pos = candidates.filter(pl.col('target') == 1).select('user_id').unique()
    candidates = candidates.join(users_with_pos, on='user_id', how='semi')
    
    logger.info(f"  Users with positives: {users_with_pos.height:,}")
    
    # All positives
    positives = candidates.filter(pl.col('target') == 1)
    
    # Sample negatives per user
    negatives = (
        candidates.filter(pl.col('target') == 0)
        .with_columns(pl.lit(np.random.rand(candidates.filter(pl.col('target') == 0).height)).alias('_rand'))
        .sort(['user_id', '_rand'])
        .group_by('user_id')
        .head(negatives_per_user)
        .drop('_rand')
    )
    
    # Combine
    train_data = pl.concat([positives, negatives])
    
    logger.info(f"  Training samples: {train_data.height:,} (pos={positives.height:,}, neg={negatives.height:,})")
    return train_data


# ============================================================================
# MODEL TRAINING & INFERENCE
# ============================================================================
def train_model(train_data: pl.DataFrame, features: list) -> CatBoostClassifier:
    """Train CatBoost classifier."""
    logger.info("Training CatBoost model...")
    
    X = train_data.select(features).to_numpy()
    y = train_data['target'].to_numpy()
    
    train_pool = Pool(data=X, label=y, feature_names=features)
    
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        verbose=100,
        random_seed=42
    )
    model.fit(train_pool)
    
    logger.info("\nFeature Importances:")
    for name, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
        logger.info(f"  {name}: {imp:.2f}")
    
    return model


def rank_candidates(model: CatBoostClassifier, candidates: pl.DataFrame, 
                    features: list, max_per_user: int = 20) -> pl.DataFrame:
    """Predict scores and rank candidates."""
    logger.info("Ranking candidates...")
    
    X = candidates.select(features).to_numpy()
    predictions = model.predict_proba(X)[:, 1]
    
    # Add scores and rank
    ranked = (
        candidates.with_columns(pl.Series('cb_score', predictions))
        .sort(['user_id', 'cb_score'], descending=[False, True])
        .with_columns(
            pl.col('track_id').cum_count().over('user_id').alias('rank')
        )
        .filter(pl.col('rank') <= max_per_user)
    )
    
    logger.info(f"  Ranked {ranked.height:,} recommendations for {ranked['user_id'].n_unique():,} users")
    return ranked


def evaluate(recommendations: pl.DataFrame, eval_events: pl.DataFrame, k_values: list = [5, 10, 20]) -> dict:
    """Evaluate recommendations against ground truth."""
    logger.info("=" * 50)
    logger.info("EVALUATION")
    logger.info("=" * 50)
    
    # Ground truth as set per user
    ground_truth = (
        eval_events.select(['user_id', 'track_id'])
        .unique()
        .group_by('user_id')
        .agg(pl.col('track_id').alias('relevant_tracks'))
    )
    
    results = {}
    for k in k_values:
        # Get top-k recommendations per user
        top_k = recommendations.filter(pl.col('rank') <= k)
        
        # Group recommendations by user
        recs_grouped = top_k.group_by('user_id').agg(
            pl.col('track_id').alias('rec_tracks')
        )
        
        # Join with ground truth
        eval_df = recs_grouped.join(ground_truth, on='user_id', how='inner')
        
        # Calculate metrics
        metrics = eval_df.with_columns([
            pl.struct(['rec_tracks', 'relevant_tracks']).map_elements(
                lambda x: len(set(x['rec_tracks']) & set(x['relevant_tracks'])),
                return_dtype=pl.Int64
            ).alias('hits')
        ]).with_columns([
            (pl.col('hits') / k).alias('precision'),
            (pl.col('hits') > 0).cast(pl.Int32).alias('is_hit')
        ])
        
        avg_precision = metrics['precision'].mean()
        hit_rate = metrics['is_hit'].mean()
        
        results[f'precision@{k}'] = avg_precision
        results[f'hit_rate@{k}'] = hit_rate
        
        logger.info(f"  Precision@{k}: {avg_precision:.4f}, Hit Rate@{k}: {hit_rate:.4f}")
    
    return results


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_catboost_ranking_pipeline(preprocessed_dir: str = 'data/preprocessed',
                                   sample_users: int = 5000):
    """Run the complete CatBoost ranking pipeline."""
    logger.info("=" * 60)
    logger.info("CATBOOST RANKING PIPELINE (Pure Polars)")
    logger.info("=" * 60)
    
    # 1. Load data
    logger.info("\n1. Loading data...")
    catalog = pl.read_parquet(f'{preprocessed_dir}/tracks_catalog_clean.parquet')
    train_events = pl.read_parquet(f'{preprocessed_dir}/train_events.parquet')
    test_events = pl.read_parquet(f'{preprocessed_dir}/test_events.parquet')
    train_matrix = load_npz(f'{preprocessed_dir}/train_matrix.npz')
    als_model = load_als_model(f'{preprocessed_dir}/als_model.pkl')
    
    logger.info(f"  Train events: {train_events.height:,}")
    logger.info(f"  Test events: {test_events.height:,}")
    
    # 2. Split test by date
    logger.info("\n2. Splitting test data by date...")
    train_labels, eval_events = split_test_by_date(test_events, train_ratio=0.5)
    
    # 3. Sample users for training
    all_train_users = train_labels['user_id'].unique().to_list()
    np.random.seed(42)
    if sample_users < len(all_train_users):
        train_user_ids = list(np.random.choice(all_train_users, sample_users, replace=False))
    else:
        train_user_ids = all_train_users
    logger.info(f"  Training users: {len(train_user_ids):,}")
    
    # 4. Compute features
    logger.info("\n3. Computing track features...")
    track_features = compute_track_features(catalog, train_events)
    
    # 5. Get popular tracks
    logger.info("\n4. Getting popular tracks...")
    popular_tracks = get_popular_tracks(train_events, n=50)
    
    # 6. Generate candidates for training
    logger.info("\n5. Generating candidates for training...")
    als_candidates = generate_als_candidates(als_model, train_matrix, train_user_ids, n=50)
    popular_candidates = generate_popular_candidates(popular_tracks, train_user_ids)
    
    # 7. Prepare training data
    logger.info("\n6. Preparing training data...")
    candidates = prepare_candidates(als_candidates, popular_candidates, track_features, train_labels)
    train_data = sample_for_training(candidates, negatives_per_user=NEGATIVES_PER_USER)
    
    # 8. Train model
    logger.info("\n7. Training model...")
    model = train_model(train_data, FEATURES)
    
    # 9. Generate candidates for evaluation
    logger.info("\n8. Generating candidates for evaluation...")
    all_eval_users = eval_events['user_id'].unique().to_list()
    np.random.seed(42)
    if sample_users < len(all_eval_users):
        eval_user_ids = list(np.random.choice(all_eval_users, sample_users, replace=False))
    else:
        eval_user_ids = all_eval_users
    
    als_candidates_eval = generate_als_candidates(als_model, train_matrix, eval_user_ids, n=100)
    popular_candidates_eval = generate_popular_candidates(popular_tracks, eval_user_ids)
    
    # Merge eval candidates (no labels needed)
    eval_candidates = (
        als_candidates_eval.join(popular_candidates_eval, on=['user_id', 'track_id'], how='full', coalesce=True)
        .fill_null(0)
        .join(track_features, on='track_id', how='left')
        .fill_null(0)
    )
    
    # 10. Rank and evaluate
    logger.info("\n9. Ranking and evaluating...")
    ranked = rank_candidates(model, eval_candidates, FEATURES, max_per_user=20)
    results = evaluate(ranked, eval_events)
    
    # Save model
    model_path = f'{preprocessed_dir}/catboost_ranker.cbm'
    model.save_model(model_path)
    logger.info(f"\nModel saved to: {model_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    return model, results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CatBoost ranking model')
    parser.add_argument('--preprocessed-dir', type=str, default='data/preprocessed')
    parser.add_argument('--sample-users', type=int, default=5000)
    args = parser.parse_args()
    
    run_catboost_ranking_pipeline(args.preprocessed_dir, args.sample_users)
