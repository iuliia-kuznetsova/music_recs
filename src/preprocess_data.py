"""
Music Recommendation System - Data Preprocessing Pipeline

Produces:
- items.parquet: Canonical music catalog with track_group_id for diversity
- events.parquet: User-track interaction events
- tracks_catalog_clean.parquet: Track lookup table (for recommendation display)

Usage:
    python3 -m src.main --raw-dir data/raw --preprocessed-dir data/preprocessed
"""

import os
import gc
import logging
from dataclasses import dataclass

import polars as pl
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration: input/output directories."""
    raw_dir: str
    preprocessed_dir: str

    @classmethod
    def from_env(cls, raw_dir: str = None, preprocessed_dir: str = None):
        load_dotenv()
        return cls(
            raw_dir=raw_dir or os.getenv('RAW_DATA_DIR', './data/raw'),
            preprocessed_dir=preprocessed_dir or os.getenv('PREPROCESSED_DATA_DIR', './data/preprocessed'),
        )


def standardize_text(col: str) -> pl.Expr:
    """Basic text cleaning: lowercase, remove punctuation, trim."""
    return (
        pl.col(col).cast(pl.Utf8).fill_null('')
        .str.normalize('NFKD').str.replace_all(r'[\p{M}]', '')  # Remove diacritics
        .str.to_lowercase()
        .str.replace_all(r'[^\w\s]', ' ').str.replace_all(r'\s+', ' ').str.strip_chars()
    )


def clean_name(col: str) -> pl.Expr:
    """Clean entity names: remove brackets, features, versions."""
    return (
        pl.col(col).fill_null('')
        .str.replace_all(r'[\(\[][^\)\]]*[\)\]]', ' ')  # Remove (brackets)
        .str.replace_all(r'\b(feat\.?|ft\.?|featuring)\b.*', ' ')  # Remove feat.
        .str.replace_all(r'\b(live|remix|extended|radio edit|acoustic|remastered(?:\s+\d{4})?)\b', ' ')
        .str.replace_all(r'[^\w\s]', ' ').str.replace_all(r'\s+', ' ')
        .str.strip_chars().str.to_titlecase()
    )


def normalize_title(col: str) -> pl.Expr:
    """Aggressive normalization for grouping versions: 'Song (Live)' → 'song'"""
    return (
        pl.col(col).fill_null('').str.to_lowercase()
        .str.replace_all(r'[\(\[][^\)\]]*[\)\]]', ' ')
        .str.replace_all(r'\b(feat\.?|ft\.?|featuring)\b.*', ' ')
        .str.replace_all(r'\b(live|remix|extended|radio edit|acoustic|remastered(?:\s+\d{4})?)\b', ' ')
        .str.replace_all(r'\bcover(ed)?\b.*', ' ')
        .str.replace_all(r'[^\w\s]', ' ').str.replace_all(r'\s+', ' ').str.strip_chars()
    )


def run_preprocessing(config: PreprocessConfig):
    """Main preprocessing pipeline."""
    logger.info('Starting preprocessing: %s → %s', config.raw_dir, config.preprocessed_dir)
    os.makedirs(config.preprocessed_dir, exist_ok=True)

    # 1. Load raw data (lazy)
    tracks = pl.scan_parquet(f'{config.raw_dir}/tracks.parquet')
    catalog = pl.scan_parquet(f'{config.raw_dir}/catalog_names.parquet')
    interactions = pl.scan_parquet(f'{config.raw_dir}/interactions.parquet')

    # 2. Explode tracks
    logger.info('Exploding tracks...')
    tracks_exploded = (
        tracks.select(['track_id', 'albums', 'artists', 'genres'])
        .explode('albums').explode('artists').explode('genres')
        .rename({'albums': 'album_id', 'artists': 'artist_id', 'genres': 'genre_id'})
    )

    # 3. Build entity catalogs
    logger.info('Building entity catalogs...')
    
    def make_catalog(entity_type: str, id_col: str, name_col: str):
        return (
            catalog.filter(pl.col('type') == entity_type)
            .select([
                pl.col('id').alias(id_col),
                pl.col('name').alias(name_col),
                standardize_text('name').alias(f'{name_col}_std'),
            ])
        )
    
    tracks_cat = make_catalog('track', 'track_id', 'track_name')
    artists_cat = make_catalog('artist', 'artist_id', 'artist_name')
    albums_cat = make_catalog('album', 'album_id', 'album_name')
    genres_cat = make_catalog('genre', 'genre_id', 'genre_name')

    # 4. Deduplicate entities
    logger.info('Deduplicating entities...')
    
    # Artists
    artists_dedup = (
        artists_cat.collect()
        .group_by('artist_name_std')
        .agg([
            pl.min('artist_id').alias('canonical_id'),
            clean_name('artist_name_std').first().alias('clean_name'),
        ])
    )
    artist_map = (
        artists_cat.collect()
        .join(artists_dedup.select(['artist_name_std', 'canonical_id']), on='artist_name_std')
        .select(['artist_id', pl.col('canonical_id').alias('artist_id_canonical')])
    )
    
    # Albums (per artist context)
    album_artists = (
        tracks_exploded.select(['album_id', 'artist_id']).unique()
        .join(artist_map.lazy(), on='artist_id')
        .select(['album_id', 'artist_id_canonical'])
        .collect()
    )
    albums_dedup = (
        albums_cat.collect()
        .join(album_artists, on='album_id', how='left')
        .with_columns(pl.col('artist_id_canonical').fill_null(0))
        .group_by(['album_name_std', 'artist_id_canonical'])
        .agg([
            pl.min('album_id').alias('canonical_id'),
            clean_name('album_name_std').first().alias('clean_name'),
        ])
    )
    album_map = (
        albums_cat.collect()
        .join(album_artists, on='album_id', how='left')
        .with_columns(pl.col('artist_id_canonical').fill_null(0))
        .join(albums_dedup.select(['album_name_std', 'artist_id_canonical', 'canonical_id']), 
              on=['album_name_std', 'artist_id_canonical'])
        .select(['album_id', pl.col('canonical_id').alias('album_id_canonical')])
    )
    
    # Tracks (per artist context) + track grouping
    track_artists = (
        tracks_exploded.select(['track_id', 'artist_id']).unique()
        .join(artist_map.lazy(), on='artist_id')
        .select(['track_id', 'artist_id_canonical'])
        .collect()
    )
    tracks_dedup = (
        tracks_cat.collect()
        .join(track_artists, on='track_id', how='left')
        .with_columns(pl.col('artist_id_canonical').fill_null(0))
        .group_by(['track_name_std', 'artist_id_canonical'])
        .agg([
            pl.min('track_id').alias('canonical_id'),
            clean_name('track_name_std').first().alias('clean_name'),
        ])
        # Add normalized title for grouping
        .with_columns(normalize_title('track_name_std').alias('title_normalized'))
    )
    
    # Create track groups: "Song (Live)" and "Song (Remix)" → same group
    track_groups = (
        tracks_dedup
        .group_by(['title_normalized', 'artist_id_canonical'])
        .agg(pl.min('canonical_id').alias('track_group_id'))
    )
    tracks_dedup = tracks_dedup.join(
        track_groups, 
        on=['title_normalized', 'artist_id_canonical'], 
        how='left'
    )
    
    track_map = (
        tracks_cat.collect()
        .join(track_artists, on='track_id', how='left')
        .with_columns(pl.col('artist_id_canonical').fill_null(0))
        .join(tracks_dedup.select(['track_name_std', 'artist_id_canonical', 'canonical_id']), 
              on=['track_name_std', 'artist_id_canonical'])
        .select(['track_id', pl.col('canonical_id').alias('track_id_canonical')])
    )
    
    # Genres
    genres_dedup = (
        genres_cat.collect()
        .group_by('genre_name_std')
        .agg([
            pl.min('genre_id').alias('canonical_id'),
            clean_name('genre_name_std').first().alias('clean_name'),
        ])
    )
    genre_map = (
        genres_cat.collect()
        .join(genres_dedup.select(['genre_name_std', 'canonical_id']), on='genre_name_std')
        .select(['genre_id', pl.col('canonical_id').alias('genre_id_canonical')])
    )

    # 5. Build items table
    logger.info('Building items table...')
    
    # Create ID mapping dicts
    track_dict = dict(zip(track_map['track_id'], track_map['track_id_canonical']))
    artist_dict = dict(zip(artist_map['artist_id'], artist_map['artist_id_canonical']))
    album_dict = dict(zip(album_map['album_id'], album_map['album_id_canonical']))
    genre_dict = dict(zip(genre_map['genre_id'], genre_map['genre_id_canonical']))
    
    # Map raw IDs → canonical IDs → join clean names + track_group_id
    items = (
        tracks_exploded
        .select(['track_id', 'artist_id', 'album_id', 'genre_id'])
        .with_columns([
            pl.col('track_id').replace(track_dict, default=None).cast(pl.Int64),
            pl.col('artist_id').replace(artist_dict, default=None).cast(pl.Int64),
            pl.col('album_id').replace(album_dict, default=None).cast(pl.Int64),
            pl.col('genre_id').replace(genre_dict, default=None).cast(pl.Int64),
        ])
        .drop_nulls()
        .join(
            tracks_dedup.lazy().select([
                pl.col('canonical_id').alias('track_id'),
                pl.col('clean_name').alias('track_clean'),
                'track_group_id',
            ]),
            on='track_id', how='left'
        )
        .join(
            artists_dedup.lazy().select([
                pl.col('canonical_id').alias('artist_id'),
                pl.col('clean_name').alias('artist_clean'),
            ]),
            on='artist_id', how='left'
        )
        .join(
            albums_dedup.lazy().select([
                pl.col('canonical_id').alias('album_id'),
                pl.col('clean_name').alias('album_clean'),
            ]),
            on='album_id', how='left'
        )
        .join(
            genres_dedup.lazy().select([
                pl.col('canonical_id').alias('genre_id'),
                pl.col('clean_name').alias('genre_clean'),
            ]),
            on='genre_id', how='left'
        )
        .with_columns([
            pl.col('track_clean').fill_null('Unknown'),
            pl.col('artist_clean').fill_null('Unknown'),
            pl.col('album_clean').fill_null('Unknown'),
            pl.col('genre_clean').fill_null('Unknown'),
            # Fallback: if no group assigned, use track_id itself
            pl.when(pl.col('track_group_id').is_null())
              .then(pl.col('track_id'))
              .otherwise(pl.col('track_group_id'))
              .alias('track_group_id'),
        ])
        .unique(subset=['track_id', 'artist_id', 'album_id', 'genre_id'])
    )
    
    items.sink_parquet(f'{config.preprocessed_dir}/items.parquet')
    logger.info('✓ items.parquet written')
    gc.collect()

    # 6. Build tracks catalog (for recommendation display)
    logger.info('Building tracks catalog...')
    tracks_catalog = (
        pl.scan_parquet(f'{config.preprocessed_dir}/items.parquet')
        .group_by('track_id')
        .agg([
            pl.first('track_clean'),
            pl.first('track_group_id'),
            pl.first('artist_id'),
            pl.first('album_id'),
            pl.first('genre_id'),
        ])
    )
    tracks_catalog.sink_parquet(f'{config.preprocessed_dir}/tracks_catalog_clean.parquet')
    logger.info('✓ tracks_catalog_clean.parquet written')

    # 7. Build events table
    logger.info('Building events table...')
    
    threshold = interactions.select(pl.col('track_seq').quantile(0.95)).collect().item()
    
    events = (
        interactions
        .select([
            'user_id',
            'track_id',
            'track_seq',
            pl.col('started_at').dt.date().alias('started_at'),
        ])
        .filter(pl.col('track_seq') <= threshold)
        .join(track_map.lazy(), on='track_id')
        .with_columns(pl.col('track_id_canonical').alias('track_id'))
        .drop_nulls('track_id')
        .select(['user_id', 'track_id', 'track_seq', 'started_at'])
        .group_by(['user_id', 'track_id'])
        .agg([
            pl.len().alias('listen_count'),
            pl.max('started_at').alias('last_listen'),
        ])
    )
    
    events.sink_parquet(f'{config.preprocessed_dir}/events.parquet')
    logger.info('✓ events.parquet written')
    
    logger.info('Preprocessing complete!')


__all__ = ['PreprocessConfig', 'run_preprocessing']
