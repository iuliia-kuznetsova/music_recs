"""
CLI entry point for music recommendation data preprocessing.

Usage:
    python3 -m src.main
    python3 -m src.main --raw-dir /path/to/raw --preprocessed-dir /path/to/output
"""

import argparse
from src.preprocess_data import PreprocessConfig, run_preprocessing


def main():
    parser = argparse.ArgumentParser(
        description='Music Recommendation System - Data Preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use environment variables from .env
  python3 -m src.main

  # Override with custom paths
  python3 -m src.main --raw-dir data/raw --preprocessed-dir data/preprocessed

Output files:
  - items.parquet: Canonical music catalog with track_group_id
  - events.parquet: User-track interaction events
  - tracks_catalog_clean.parquet: Track lookup table
        """
    )
    
    parser.add_argument(
        '--raw-dir',
        help='Directory with raw parquet files (tracks, catalog_names, interactions)',
        metavar='PATH'
    )
    parser.add_argument(
        '--preprocessed-dir',
        help='Output directory for processed parquet files',
        metavar='PATH'
    )
    
    args = parser.parse_args()
    
    # Create config from env + args
    config = PreprocessConfig.from_env(
        raw_dir=args.raw_dir,
        preprocessed_dir=args.preprocessed_dir,
    )
    
    # Run pipeline
    run_preprocessing(config)


if __name__ == '__main__':
    main()

