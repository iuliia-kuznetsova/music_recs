'''
Recommendations Ranking and Re-ranking

This module provides functionality to rank and re-rank recommendations using:
1. Multiple scoring signals
2. Diversity constraints (using track_group_id)
3. Popularity boosting/dampening
4. Recency filtering
'''

# ---------- Imports ---------- #
from dotenv import load_dotenv
import os

import logging
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import polars as pl

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

# ---------- Recommendation ranker ---------- #
class RecommendationRanker:
    '''
    Rank recommendations.
    '''
    
    def __init__(self, catalog_df: pl.DataFrame):
        self.catalog = catalog_df
        
        # Build track group mapping
        self.track_to_group = {
            row['track_id']: row['track_group_id']
            for row in catalog_df.select(['track_id', 'track_group_id']).iter_rows(named=True)
        }
        
    def rank_with_diversity(
        self,
        recommendations: List[Tuple[int, float]],
        n: int = 10,
        diversity_weight: float = 0.3
    ) -> List[Tuple[int, float]]:
        '''
        Re-rank recommendations to promote diversity.
        
        Ensures different songs (not just different versions of same song).
        
        Args:
            recommendations: List of (track_id, score) tuples
            n: Number of final recommendations
            diversity_weight: Weight for diversity penalty (0-1)
            
        Returns:
            Re-ranked list of (track_id, score)
        '''
        if not recommendations:
            return []
        
        ranked = []
        used_groups: Set[int] = set()
        
        # Sort by score initially
        candidates = sorted(recommendations, key=lambda x: x[1], reverse=True)
        
        for track_id, score in candidates:
            if len(ranked) >= n:
                break
            
            track_group = self.track_to_group.get(track_id, track_id)
            
            # Apply diversity penalty if group already used
            if track_group in used_groups:
                adjusted_score = score * (1 - diversity_weight)
            else:
                adjusted_score = score
            
            # Add to ranked list
            ranked.append((track_id, adjusted_score))
            used_groups.add(track_group)
        
        # Re-sort by adjusted scores
        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)[:n]
        
        return ranked
    
    def rank_with_popularity(
        self,
        recommendations: List[Tuple[int, float]],
        popularity_scores: Dict[int, float],
        popularity_weight: float = 0.2
    ) -> List[Tuple[int, float]]:
        '''
        Re-rank recommendations considering popularity.
        
        Args:
            recommendations: List of (track_id, score) tuples
            popularity_scores: Dict mapping track_id to popularity score
            popularity_weight: Weight for popularity (0-1)
            
        Returns:
            Re-ranked list of (track_id, score)
        '''
        ranked = []
        
        for track_id, score in recommendations:
            pop_score = popularity_scores.get(track_id, 0.0)
            
            # Normalize popularity (assume max pop is 1.0)
            max_pop = max(popularity_scores.values()) if popularity_scores else 1.0
            norm_pop = pop_score / max_pop if max_pop > 0 else 0
            
            # Combine scores
            final_score = (1 - popularity_weight) * score + popularity_weight * norm_pop
            ranked.append((track_id, final_score))
        
        # Sort by combined score
        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def rank_with_novelty(
        self,
        recommendations: List[Tuple[int, float]],
        user_history: Set[int],
        novelty_weight: float = 0.3
    ) -> List[Tuple[int, float]]:
        '''
        Re-rank to promote novel recommendations.
        
        Boosts items from artists/genres user hasn't explored much.
        
        Args:
            recommendations: List of (track_id, score) tuples
            user_history: Set of track_ids user has already listened to
            novelty_weight: Weight for novelty boost (0-1)
            
        Returns:
            Re-ranked list of (track_id, score)
        '''
        # Get user's artist/genre history
        user_tracks = self.catalog.filter(pl.col('track_id').is_in(list(user_history)))
        
        if user_tracks.height == 0:
            return recommendations
        
        user_artists = set(user_tracks['artist_id'].to_list())
        user_genres = set(user_tracks['genre_id'].to_list())
        
        ranked = []
        
        for track_id, score in recommendations:
            track_info = self.catalog.filter(pl.col('track_id') == track_id)
            
            if track_info.height == 0:
                ranked.append((track_id, score))
                continue
            
            artist_id = track_info['artist_id'][0]
            genre_id = track_info['genre_id'][0]
            
            # Novelty boost if new artist or genre
            novelty_boost = 0.0
            if artist_id not in user_artists:
                novelty_boost += 0.5
            if genre_id not in user_genres:
                novelty_boost += 0.5
            
            # Normalize boost (max 1.0)
            novelty_boost = min(novelty_boost, 1.0)
            
            # Apply boost
            final_score = score * (1 + novelty_weight * novelty_boost)
            ranked.append((track_id, final_score))
        
        # Sort by final score
        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def rank_multi_objective(
        self,
        recommendations: List[Tuple[int, float]],
        popularity_scores: Optional[Dict[int, float]] = None,
        user_history: Optional[Set[int]] = None,
        n: int = 10,
        diversity_weight: float = 0.3,
        popularity_weight: float = 0.1,
        novelty_weight: float = 0.2
    ) -> List[Tuple[int, float]]:
        '''
        Rank recommendations using multiple objectives.
        
        Args:
            recommendations: List of (track_id, score) tuples
            popularity_scores: Optional popularity scores
            user_history: Optional user listening history
            n: Number of final recommendations
            diversity_weight: Weight for diversity
            popularity_weight: Weight for popularity
            novelty_weight: Weight for novelty
            
        Returns:
            Re-ranked list of (track_id, score)
        '''
        ranked = recommendations
        
        # Apply popularity if provided
        if popularity_scores and popularity_weight > 0:
            ranked = self.rank_with_popularity(ranked, popularity_scores, popularity_weight)
        
        # Apply novelty if user history provided
        if user_history and novelty_weight > 0:
            ranked = self.rank_with_novelty(ranked, user_history, novelty_weight)
        
        # Apply diversity last to get final N
        if diversity_weight > 0:
            ranked = self.rank_with_diversity(ranked, n, diversity_weight)
        else:
            ranked = sorted(ranked, key=lambda x: x[1], reverse=True)[:n]
        
        return ranked
    
    def explain_ranking(
        self,
        track_id: int,
        original_score: float,
        final_score: float
    ) -> str:
        '''
        Explain why a track was ranked at its position.
        
        Args:
            track_id: Track ID
            original_score: Original recommendation score
            final_score: Final score after re-ranking
            
        Returns:
            Explanation string
        '''
        track_info = self.catalog.filter(pl.col('track_id') == track_id)
        
        if track_info.height == 0:
            return f'Track {track_id}: Score {final_score:.4f}'
        
        track_name = track_info['track_clean'][0]
        
        explanation = f'{track_name}: '
        
        if final_score > original_score:
            boost_pct = ((final_score / original_score - 1) * 100)
            explanation += f'Boosted by {boost_pct:.1f}% '
        elif final_score < original_score:
            penalty_pct = ((1 - final_score / original_score) * 100)
            explanation += f'Penalized by {penalty_pct:.1f}% '
        else:
            explanation += 'No adjustment '
        
        explanation += f'(Final score: {final_score:.4f})'
        
        return explanation


def combine_recommendations(
    recommendation_lists: List[Tuple[str, List[Tuple[int, float]], float]],
    n: int = 10
) -> List[Tuple[int, float]]:
    '''
    Combine recommendations from multiple sources.
    
    Args:
        recommendation_lists: List of (name, recommendations, weight) tuples
        n: Number of final recommendations
        
    Returns:
        Combined and ranked recommendations
    '''
    logger.info(f'Combining {len(recommendation_lists)} recommendation sources')
    
    # Collect all track scores
    track_scores: Dict[int, float] = {}
    
    for name, recs, weight in recommendation_lists:
        logger.info(f'  - {name}: {len(recs)} recommendations, weight={weight}')
        
        for track_id, score in recs:
            # Normalize score to 0-1 range per source
            if track_id not in track_scores:
                track_scores[track_id] = 0.0
            track_scores[track_id] += score * weight
    
    # Sort and take top N
    combined = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    
    logger.info(f'✅ Combined into {len(combined)} recommendations')
    
    return combined


if __name__ == '__main__':
    # Example usage
    import polars as pl
    
    logger.info('Testing RecommendationRanker')
    
    # Load catalog
    catalog = pl.read_parquet('data/preprocessed/tracks_catalog_clean.parquet')
    
    # Create ranker
    ranker = RecommendationRanker(catalog)
    
    # Dummy recommendations
    recs = [
        (40330534, 0.95),  # In The End
        (53404, 0.90),     # Smells Like Teen Spirit
        (33311009, 0.85),  # Believer
        (251849, 0.80),    # The Show Must Go On
        (178477, 0.75),    # Numb
    ]
    
    logger.info(f'\nOriginal recommendations: {len(recs)}')
    
    # Apply diversity
    ranked = ranker.rank_with_diversity(recs, n=5, diversity_weight=0.3)
    
    logger.info(f'After diversity re-ranking: {len(ranked)}')
    for i, (tid, score) in enumerate(ranked, 1):
        track = catalog.filter(pl.col('track_id') == tid)
        if track.height > 0:
            logger.info(f'  {i}. {track["track_clean"][0]} (score: {score:.4f})')

