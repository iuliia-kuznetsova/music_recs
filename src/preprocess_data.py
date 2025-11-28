import os
import gc
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import polars as pl
from dotenv import load_dotenv

# ---------- Logging ---------- #
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ---------- Text cleaning helpers ---------- #
UNKNOWN_TOKEN = 'Unknown'
BRACKETS_PATTERN = r'[\(\[][^\)\]]*[\)\]]'
FEATURE_PATTERN = r'\b(feat\.?|ft\.?|featuring)\b.*'
VERSION_TAG_PATTERN = r'\b(live|remix|extended|radio edit|acoustic|remastered(?:\s+\d{4})?)\b'
COVER_PATTERN = r'\bcover(ed)?\b.*'


@dataclass
class PreprocessConfig:
    raw_dir: str
    preprocessed_dir: str
    checkpoint_dir: str
    cache_fact_ids: bool = True
    cache_items: bool = True

    @classmethod
    def from_env(
        cls,
        raw_dir: str | None = None,
        preprocessed_dir: str | None = None,
        checkpoint_dir: str | None = None,
        cache_fact_ids: bool = True,
        cache_items: bool = True,
    ) -> 'PreprocessConfig':
        load_dotenv()
        raw_path = raw_dir or os.getenv('RAW_DATA_DIR', './data/raw')
        preprocessed_path = preprocessed_dir or os.getenv('PREPROCESSED_DATA_DIR', './data/preprocessed')
        checkpoint_path = checkpoint_dir or os.path.join(preprocessed_path, 'checkpoints')
        return cls(
            raw_dir=raw_path,
            preprocessed_dir=preprocessed_path,
            checkpoint_dir=checkpoint_path,
            cache_fact_ids=cache_fact_ids,
            cache_items=cache_items,
        )


def basic_standardize(colname: str, alias: str) -> pl.Expr:
    return (
        pl.col(colname)
        .cast(pl.Utf8)
        .fill_null('')
        .str.normalize('NFKD')
        .str.replace_all(r'[\p{M}]', '')
        .str.to_lowercase()
        .str.replace_all(r'[^\w\s]', ' ')
        .str.replace_all(r'\s+', ' ')
        .str.strip_chars()
        .alias(alias)
    )


def clean_entity_name(std_col: str, alias: str, strip_versions: bool = False, strip_features: bool = False) -> pl.Expr:
    expr = pl.col(std_col).fill_null('')
    expr = expr.str.replace_all(BRACKETS_PATTERN, ' ')
    if strip_features:
        expr = expr.str.replace_all(FEATURE_PATTERN, ' ')
    if strip_versions:
        expr = expr.str.replace_all(VERSION_TAG_PATTERN, ' ')
        expr = expr.str.replace_all(COVER_PATTERN, ' ')
    expr = expr.str.replace_all(r'[^\w\s]', ' ')
    expr = expr.str.replace_all(r'\s+', ' ')
    return expr.str.strip_chars().str.to_titlecase().alias(alias)


def normalize_track_title(source_col: str, alias: str = 'title_normalized') -> pl.Expr:
    expr = pl.col(source_col).cast(pl.Utf8).fill_null('')
    expr = expr.str.to_lowercase()
    expr = expr.str.replace_all(BRACKETS_PATTERN, ' ')
    expr = expr.str.replace_all(FEATURE_PATTERN, ' ')
    expr = expr.str.replace_all(VERSION_TAG_PATTERN, ' ')
    expr = expr.str.replace_all(COVER_PATTERN, ' ')
    expr = expr.str.replace_all(r'[^\w\s]', ' ')
    expr = expr.str.replace_all(r'\s+', ' ')
    return expr.str.strip_chars().alias(alias)


def ensure_token(col: str, token: str = UNKNOWN_TOKEN) -> pl.Expr:
    return (
        pl.when(pl.col(col).is_null() | (pl.col(col).str.len_chars() == 0))
        .then(pl.lit(token))
        .otherwise(pl.col(col))
        .alias(col)
    )


def read_raw_tables(config: PreprocessConfig) -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """All tables are now lazy scans to minimize memory."""
    tracks_path = os.path.join(config.raw_dir, 'tracks.parquet')
    catalog_path = os.path.join(config.raw_dir, 'catalog_names.parquet')
    interactions_path = os.path.join(config.raw_dir, 'interactions.parquet')

    logger.info('Opening lazy scans for raw tables')
    tracks_scan = pl.scan_parquet(tracks_path)
    catalog_scan = pl.scan_parquet(catalog_path)
    interactions_scan = pl.scan_parquet(interactions_path)
    return tracks_scan, catalog_scan, interactions_scan


def explode_tracks_lazy(tracks_lazy: pl.LazyFrame) -> pl.LazyFrame:
    logger.info('Building lazy exploded tracks view')
    return (
        tracks_lazy
        .select(['track_id', 'albums', 'artists', 'genres'])
        .explode('albums')
        .explode('artists')
        .explode('genres')
        .rename({'albums': 'album_id', 'artists': 'artist_id', 'genres': 'genre_id'})
    )


def build_standardized_catalogs(catalog_scan: pl.LazyFrame, checkpoint_dir: str) -> Dict[str, str]:
    """
    Materialize small catalog tables to parquet and return paths.
    These are small enough to collect but we checkpoint them for safety.
    """
    logger.info('Standardizing catalog tables (streaming to checkpoints)')
    os.makedirs(checkpoint_dir, exist_ok=True)

    paths = {}

    # Tracks catalog
    tracks_path = os.path.join(checkpoint_dir, 'tracks_catalog_std.parquet')
    (
        catalog_scan
        .filter(pl.col('type') == 'track')
        .select([pl.col('id').alias('track_id'), pl.col('name').alias('track_name')])
        .with_columns(basic_standardize('track_name', 'track_name_std'))
        .with_columns(ensure_token('track_name_std'))
        .sink_parquet(tracks_path)
    )
    paths['tracks'] = tracks_path

    # Artists catalog
    artists_path = os.path.join(checkpoint_dir, 'artists_catalog_std.parquet')
    (
        catalog_scan
        .filter(pl.col('type') == 'artist')
        .select([pl.col('id').alias('artist_id'), pl.col('name').alias('artist_name')])
        .with_columns(basic_standardize('artist_name', 'artist_name_std'))
        .with_columns(ensure_token('artist_name_std'))
        .sink_parquet(artists_path)
    )
    paths['artists'] = artists_path

    # Albums catalog
    albums_path = os.path.join(checkpoint_dir, 'albums_catalog_std.parquet')
    (
        catalog_scan
        .filter(pl.col('type') == 'album')
        .select([pl.col('id').alias('album_id'), pl.col('name').alias('album_name')])
        .with_columns(basic_standardize('album_name', 'album_name_std'))
        .with_columns(ensure_token('album_name_std'))
        .sink_parquet(albums_path)
    )
    paths['albums'] = albums_path

    # Genres catalog
    genres_path = os.path.join(checkpoint_dir, 'genres_catalog_std.parquet')
    (
        catalog_scan
        .filter(pl.col('type') == 'genre')
        .select([pl.col('id').alias('genre_id'), pl.col('name').alias('genre_name')])
        .with_columns(basic_standardize('genre_name', 'genre_name_std'))
        .with_columns(ensure_token('genre_name_std'))
        .sink_parquet(genres_path)
    )
    paths['genres'] = genres_path

    logger.info('Catalog checkpoints written')
    return paths


def deduplicate_entities(
    tracks_exploded_lazy: pl.LazyFrame,
    catalog_paths: Dict[str, str],
    checkpoint_dir: str,
) -> Dict[str, str]:
    """
    Deduplicate entities and write results to parquet checkpoints.
    Returns paths to the checkpoint files.
    """
    logger.info('Deduplicating catalog entities (streaming)')
    os.makedirs(checkpoint_dir, exist_ok=True)
    paths = {}

    # ----- Artists dedup (small, can collect) -----
    artists_catalog = pl.read_parquet(catalog_paths['artists'])
    artists_dedup = (
        artists_catalog
        .group_by('artist_name_std')
        .agg(pl.min('artist_id').alias('artist_id_canonical'))
        .with_columns(clean_entity_name('artist_name_std', 'artist_clean'))
        .with_columns(ensure_token('artist_clean'))
    )
    artists_dedup_path = os.path.join(checkpoint_dir, 'artists_dedup.parquet')
    artists_dedup.write_parquet(artists_dedup_path)
    paths['artists_dedup'] = artists_dedup_path

    artist_id_map = (
        artists_catalog
        .join(artists_dedup.select(['artist_name_std', 'artist_id_canonical']), on='artist_name_std', how='left')
        .with_columns(pl.coalesce([pl.col('artist_id_canonical'), pl.col('artist_id')]).alias('artist_id_canonical'))
        .select(['artist_id', 'artist_id_canonical'])
        .unique()
    )
    artist_id_map_path = os.path.join(checkpoint_dir, 'artist_id_map.parquet')
    artist_id_map.write_parquet(artist_id_map_path)
    paths['artist_id_map'] = artist_id_map_path

    del artists_catalog, artists_dedup, artist_id_map
    gc.collect()

    # ----- Album-artist bridge (sink to disk) -----
    album_artist_bridge_path = os.path.join(checkpoint_dir, 'album_artist_bridge.parquet')
    (
        tracks_exploded_lazy
        .select(['album_id', 'artist_id'])
        .drop_nulls()
        .unique()
        .join(pl.scan_parquet(artist_id_map_path), on='artist_id', how='left')
        .with_columns(pl.coalesce([pl.col('artist_id_canonical'), pl.col('artist_id')]).alias('artist_id_canonical'))
        .select(['album_id', 'artist_id_canonical'])
        .sink_parquet(album_artist_bridge_path)
    )

    # ----- Albums dedup -----
    albums_catalog = pl.read_parquet(catalog_paths['albums'])
    album_artist_bridge = pl.read_parquet(album_artist_bridge_path)
    albums_with_artist = (
        albums_catalog
        .join(album_artist_bridge, on='album_id', how='left')
        .with_columns(pl.coalesce([pl.col('artist_id_canonical'), pl.col('album_id')]).alias('artist_id_canonical'))
    )
    albums_dedup = (
        albums_with_artist
        .group_by(['album_name_std', 'artist_id_canonical'])
        .agg(pl.min('album_id').alias('album_id_canonical'))
        .with_columns(clean_entity_name('album_name_std', 'album_clean', strip_versions=True, strip_features=True))
        .with_columns(ensure_token('album_clean'))
    )
    albums_dedup_path = os.path.join(checkpoint_dir, 'albums_dedup.parquet')
    albums_dedup.write_parquet(albums_dedup_path)
    paths['albums_dedup'] = albums_dedup_path

    album_id_map = (
        albums_with_artist
        .join(
            albums_dedup.select(['album_name_std', 'artist_id_canonical', 'album_id_canonical']),
            on=['album_name_std', 'artist_id_canonical'],
            how='left',
        )
        .with_columns(pl.coalesce([pl.col('album_id_canonical'), pl.col('album_id')]).alias('album_id_canonical'))
        .select(['album_id', 'album_id_canonical'])
        .unique()
    )
    album_id_map_path = os.path.join(checkpoint_dir, 'album_id_map.parquet')
    album_id_map.write_parquet(album_id_map_path)
    paths['album_id_map'] = album_id_map_path

    del albums_catalog, album_artist_bridge, albums_with_artist, albums_dedup, album_id_map
    gc.collect()

    # ----- Track-artist bridge (sink to disk) -----
    track_artist_bridge_path = os.path.join(checkpoint_dir, 'track_artist_bridge.parquet')
    (
        tracks_exploded_lazy
        .select(['track_id', 'artist_id'])
        .drop_nulls()
        .unique()
        .join(pl.scan_parquet(artist_id_map_path), on='artist_id', how='left')
        .with_columns(pl.coalesce([pl.col('artist_id_canonical'), pl.col('artist_id')]).alias('artist_id_canonical'))
        .select(['track_id', 'artist_id_canonical'])
        .sink_parquet(track_artist_bridge_path)
    )

    # ----- Tracks dedup -----
    tracks_catalog = pl.read_parquet(catalog_paths['tracks'])
    track_artist_bridge = pl.read_parquet(track_artist_bridge_path)
    tracks_with_artist = (
        tracks_catalog
        .join(track_artist_bridge, on='track_id', how='left')
        .with_columns(pl.coalesce([pl.col('artist_id_canonical'), pl.col('track_id')]).alias('artist_id_canonical'))
    )
    tracks_dedup = (
        tracks_with_artist
        .group_by(['track_name_std', 'artist_id_canonical'])
        .agg(pl.min('track_id').alias('track_id_canonical'))
        .with_columns(clean_entity_name('track_name_std', 'track_clean', strip_versions=True, strip_features=True))
        .with_columns(normalize_track_title('track_clean', 'title_normalized'))
        .with_columns([ensure_token('track_clean'), ensure_token('title_normalized')])
    )
    track_group_lookup = (
        tracks_dedup
        .group_by(['title_normalized', 'artist_id_canonical'])
        .agg(pl.min('track_id_canonical').alias('track_group_id'))
    )
    tracks_dedup = tracks_dedup.join(track_group_lookup, on=['title_normalized', 'artist_id_canonical'], how='left')
    tracks_dedup_path = os.path.join(checkpoint_dir, 'tracks_dedup.parquet')
    tracks_dedup.write_parquet(tracks_dedup_path)
    paths['tracks_dedup'] = tracks_dedup_path

    track_id_map = (
        tracks_with_artist
        .join(
            tracks_dedup.select(['track_name_std', 'artist_id_canonical', 'track_id_canonical']),
            on=['track_name_std', 'artist_id_canonical'],
            how='left',
        )
        .with_columns(pl.coalesce([pl.col('track_id_canonical'), pl.col('track_id')]).alias('track_id_canonical'))
        .select(['track_id', 'track_id_canonical'])
        .unique()
    )
    track_id_map_path = os.path.join(checkpoint_dir, 'track_id_map.parquet')
    track_id_map.write_parquet(track_id_map_path)
    paths['track_id_map'] = track_id_map_path

    del tracks_catalog, track_artist_bridge, tracks_with_artist, tracks_dedup, track_group_lookup, track_id_map
    gc.collect()

    # ----- Genres dedup (small) -----
    genres_catalog = pl.read_parquet(catalog_paths['genres'])
    genres_dedup = (
        genres_catalog
        .group_by('genre_name_std')
        .agg(pl.min('genre_id').alias('genre_id_canonical'))
        .with_columns(clean_entity_name('genre_name_std', 'genre_clean'))
        .with_columns(ensure_token('genre_clean'))
    )
    genres_dedup_path = os.path.join(checkpoint_dir, 'genres_dedup.parquet')
    genres_dedup.write_parquet(genres_dedup_path)
    paths['genres_dedup'] = genres_dedup_path

    genre_id_map = (
        genres_catalog
        .join(genres_dedup.select(['genre_name_std', 'genre_id_canonical']), on='genre_name_std', how='left')
        .with_columns(pl.coalesce([pl.col('genre_id_canonical'), pl.col('genre_id')]).alias('genre_id_canonical'))
        .select(['genre_id', 'genre_id_canonical'])
        .unique()
    )
    genre_id_map_path = os.path.join(checkpoint_dir, 'genre_id_map.parquet')
    genre_id_map.write_parquet(genre_id_map_path)
    paths['genre_id_map'] = genre_id_map_path

    del genres_catalog, genres_dedup, genre_id_map
    gc.collect()

    logger.info('Entity deduplication checkpoints written')
    return paths


def build_fact_ids_lazy(
    tracks_exploded_lazy: pl.LazyFrame,
    dedup_paths: Dict[str, str],
) -> pl.LazyFrame:
    """Build canonical ID bridge using dictionary lookups (memory-efficient)."""
    logger.info('Building canonical ID bridge (lazy with dict lookups)')

    track_id_map = pl.read_parquet(dedup_paths['track_id_map'])
    artist_id_map = pl.read_parquet(dedup_paths['artist_id_map'])
    album_id_map = pl.read_parquet(dedup_paths['album_id_map'])
    genre_id_map = pl.read_parquet(dedup_paths['genre_id_map'])

    track_lookup = dict(zip(track_id_map['track_id'], track_id_map['track_id_canonical']))
    artist_lookup = dict(zip(artist_id_map['artist_id'], artist_id_map['artist_id_canonical']))
    album_lookup = dict(zip(album_id_map['album_id'], album_id_map['album_id_canonical']))
    genre_lookup = dict(zip(genre_id_map['genre_id'], genre_id_map['genre_id_canonical']))

    del track_id_map, artist_id_map, album_id_map, genre_id_map
    gc.collect()

    return (
        tracks_exploded_lazy
        .select(['track_id', 'artist_id', 'album_id', 'genre_id'])
        .with_columns([
            pl.col('track_id').replace(track_lookup, default=None).alias('track_id'),
            pl.col('artist_id').replace(artist_lookup, default=None).alias('artist_id'),
            pl.col('album_id').replace(album_lookup, default=None).alias('album_id'),
            pl.col('genre_id').replace(genre_lookup, default=None).alias('genre_id'),
        ])
        .with_columns([pl.col(col).cast(pl.Int64).alias(col) for col in ['track_id', 'artist_id', 'album_id', 'genre_id']])
        .drop_nulls(['track_id', 'artist_id', 'album_id', 'genre_id'])
    )


def build_items(
    fact_ids_lazy: pl.LazyFrame,
    dedup_paths: Dict[str, str],
    config: PreprocessConfig,
) -> None:
    """Sink items directly to parquet without collecting into memory."""
    logger.info('Constructing canonical items dataframe (sink to disk)')

    tracks_dedup = pl.scan_parquet(dedup_paths['tracks_dedup'])
    artists_dedup = pl.scan_parquet(dedup_paths['artists_dedup'])
    albums_dedup = pl.scan_parquet(dedup_paths['albums_dedup'])
    genres_dedup = pl.scan_parquet(dedup_paths['genres_dedup'])

    items_path = os.path.join(config.preprocessed_dir, 'items.parquet')
    os.makedirs(config.preprocessed_dir, exist_ok=True)

    (
        fact_ids_lazy
        .join(
            tracks_dedup.select(['track_id_canonical', 'track_clean', 'track_group_id']),
            left_on='track_id',
            right_on='track_id_canonical',
            how='left',
        )
        .join(
            artists_dedup.select(['artist_id_canonical', 'artist_clean']),
            left_on='artist_id',
            right_on='artist_id_canonical',
            how='left',
        )
        .join(
            albums_dedup.select(['album_id_canonical', 'album_clean']),
            left_on='album_id',
            right_on='album_id_canonical',
            how='left',
        )
        .join(
            genres_dedup.select(['genre_id_canonical', 'genre_clean']),
            left_on='genre_id',
            right_on='genre_id_canonical',
            how='left',
        )
        .select([
            pl.col('track_id').alias('track_id'),
            'track_clean',
            'track_group_id',
            pl.col('artist_id').alias('artist_id'),
            'artist_clean',
            pl.col('album_id').alias('album_id'),
            'album_clean',
            pl.col('genre_id').alias('genre_id'),
            'genre_clean',
        ])
        .with_columns([
            pl.when(pl.col('track_group_id').is_null())
            .then(pl.col('track_id'))
            .otherwise(pl.col('track_group_id'))
            .alias('track_group_id')
        ])
        .with_columns([ensure_token(col) for col in ['track_clean', 'artist_clean', 'album_clean', 'genre_clean']])
        .unique(subset=['track_id', 'artist_id', 'album_id', 'genre_id'])
        .sink_parquet(items_path)
    )

    logger.info('Items written to %s', items_path)


def clean_interactions_lazy(
    interactions_scan: pl.LazyFrame,
    track_id_map_path: str,
) -> pl.LazyFrame:
    """Clean interactions lazily."""
    logger.info('Cleaning interactions data (lazy)')

    # Get threshold from a quick streaming collect
    track_seq_threshold = (
        interactions_scan
        .select(pl.col('track_seq').quantile(0.95))
        .collect(engine='streaming')
        .item()
    )
    logger.info('Track sequence threshold: %s', track_seq_threshold)

    cols_to_keep = ['user_id', 'track_id', 'track_seq', 'started_at']
    select_cols = [c for c in cols_to_keep if c != 'started_at']

    return (
        interactions_scan
        .select([*select_cols, pl.col('started_at').dt.date().alias('started_at')])
        .filter(pl.col('track_seq') <= track_seq_threshold)
        .join(pl.scan_parquet(track_id_map_path), on='track_id', how='left')
        .with_columns(pl.col('track_id_canonical').alias('track_id'))
        .drop_nulls(['track_id'])
        .select(['user_id', 'track_id', 'track_seq', 'started_at'])
    )


def build_events(interactions_lazy: pl.LazyFrame, config: PreprocessConfig) -> None:
    """Aggregate and sink events directly to parquet."""
    logger.info('Aggregating events dataframe (sink to disk)')
    events_path = os.path.join(config.preprocessed_dir, 'events.parquet')

    (
        interactions_lazy
        .group_by(['user_id', 'track_id'])
        .agg([
            pl.len().alias('listen_count'),
            pl.max('started_at').alias('last_listen'),
        ])
        .sink_parquet(events_path)
    )
    logger.info('Events written to %s', events_path)


def build_catalog_outputs(
    dedup_paths: Dict[str, str],
    config: PreprocessConfig,
) -> None:
    """Build clean catalog tables from checkpoints and write to output."""
    logger.info('Preparing clean catalog tables')

    items_path = os.path.join(config.preprocessed_dir, 'items.parquet')

    # Tracks catalog from items
    tracks_catalog_path = os.path.join(config.preprocessed_dir, 'tracks_catalog_clean.parquet')
    (
        pl.scan_parquet(items_path)
        .group_by('track_id')
        .agg([
            pl.first('track_clean').alias('track_clean'),
            pl.first('track_group_id').alias('track_group_id'),
            pl.first('artist_id').alias('artist_id'),
            pl.first('album_id').alias('album_id'),
            pl.first('genre_id').alias('genre_id'),
        ])
        .sink_parquet(tracks_catalog_path)
    )
    logger.info('Written tracks_catalog_clean.parquet')

    # Artists catalog
    artists_catalog_path = os.path.join(config.preprocessed_dir, 'artists_catalog_clean.parquet')
    (
        pl.scan_parquet(dedup_paths['artists_dedup'])
        .select([
            pl.col('artist_id_canonical').alias('artist_id'),
            pl.col('artist_clean'),
        ])
        .unique('artist_id')
        .sink_parquet(artists_catalog_path)
    )
    logger.info('Written artists_catalog_clean.parquet')

    # Albums catalog
    albums_catalog_path = os.path.join(config.preprocessed_dir, 'albums_catalog_clean.parquet')
    (
        pl.scan_parquet(dedup_paths['albums_dedup'])
        .select([
            pl.col('album_id_canonical').alias('album_id'),
            pl.col('album_clean'),
            pl.col('artist_id_canonical').alias('artist_id'),
        ])
        .unique('album_id')
        .sink_parquet(albums_catalog_path)
    )
    logger.info('Written albums_catalog_clean.parquet')

    # Genres catalog
    genres_catalog_path = os.path.join(config.preprocessed_dir, 'genres_catalog_clean.parquet')
    (
        pl.scan_parquet(dedup_paths['genres_dedup'])
        .select([
            pl.col('genre_id_canonical').alias('genre_id'),
            pl.col('genre_clean'),
        ])
        .unique('genre_id')
        .sink_parquet(genres_catalog_path)
    )
    logger.info('Written genres_catalog_clean.parquet')


def run_preprocessing(config: PreprocessConfig) -> None:
    logger.info('Starting preprocessing with config: %s', config)
    os.makedirs(config.preprocessed_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Step 1: Read all tables lazily
    tracks_scan, catalog_scan, interactions_scan = read_raw_tables(config)
    tracks_exploded_lazy = explode_tracks_lazy(tracks_scan)

    # Step 2: Standardize catalogs (sink to checkpoints)
    catalog_paths = build_standardized_catalogs(catalog_scan, config.checkpoint_dir)
    gc.collect()

    # Step 3: Deduplicate entities (sink to checkpoints)
    dedup_paths = deduplicate_entities(tracks_exploded_lazy, catalog_paths, config.checkpoint_dir)
    gc.collect()

    # Step 4: Build fact IDs lazy plan
    fact_ids_lazy = build_fact_ids_lazy(tracks_exploded_lazy, dedup_paths)

    # Step 5: Optionally checkpoint fact IDs
    if config.cache_fact_ids:
        fact_ids_path = os.path.join(config.checkpoint_dir, 'fact_ids_checkpoint.parquet')
        logger.info('Sinking fact IDs checkpoint at %s', fact_ids_path)
        fact_ids_lazy.sink_parquet(fact_ids_path)
        gc.collect()
        fact_ids_lazy = pl.scan_parquet(fact_ids_path)

    # Step 6: Build items (sink directly)
    build_items(fact_ids_lazy, dedup_paths, config)
    gc.collect()

    # Step 7: Clean interactions and build events (sink directly)
    interactions_lazy = clean_interactions_lazy(interactions_scan, dedup_paths['track_id_map'])
    build_events(interactions_lazy, config)
    gc.collect()

    # Step 8: Build catalog outputs (sink directly)
    build_catalog_outputs(dedup_paths, config)

    logger.info('Preprocessing finished successfully')


__all__ = ['PreprocessConfig', 'run_preprocessing']
