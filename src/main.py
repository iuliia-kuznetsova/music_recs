"""
CLI entry point for music recommendation system.

Usage:
    python3 -m src.main
    python3 -m src.main --raw-dir /path/to/raw --preprocessed-dir /path/to/output
    python3 -m src.main --skip-download  # Skip data download if already present
"""

import os
import sys
import argparse
from dotenv import load_dotenv

from src.load_data import load_env_with_logging, download_all_raw
from src.preprocess_data import run_preprocessing


def main():
    """
    Main entry point: 
    1. Load environment variables
    2. Download raw data (if needed)
    3. Preprocess data
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Music Recommendation System - Data Loading & Preprocessing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: download + preprocess (uses .env variables)
  python3 -m src.main

  # Skip download if data already exists
  python3 -m src.main --skip-download

  # Override paths
  python3 -m src.main --raw-dir data/raw --preprocessed-dir data/preprocessed

  # Just download data (no preprocessing)
  python3 -m src.load_data
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
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip downloading raw data (assumes files already exist)'
    )
    
    args = parser.parse_args()
    
    # Step 1: Load environment variables
    print('\n' + '='*60)
    print('STEP 1: Loading environment variables')
    print('='*60)
    
    if not load_env_with_logging():
        print('\n❌ Failed to load environment variables. Exiting.')
        sys.exit(1)
    
    load_dotenv()  # Also load for override support
    
    # Get directories from args or .env file (with defaults)
    raw_dir = args.raw_dir or os.getenv('RAW_DATA_DIR', './data/raw')
    preprocessed_dir = args.preprocessed_dir or os.getenv('PREPROCESSED_DATA_DIR', './data/preprocessed')
    
    print(f'✅ Raw data directory: {raw_dir}')
    print(f'✅ Preprocessed data directory: {preprocessed_dir}')
    
    # Step 2: Download raw data (if not skipped)
    if not args.skip_download:
        print('\n' + '='*60)
        print('STEP 2: Downloading raw data')
        print('='*60)
        
        try:
            download_all_raw()
            print('✅ Raw data download complete')
        except Exception as e:
            print(f'\n❌ Failed to download raw data: {e}')
            sys.exit(1)
    else:
        print('\n' + '='*60)
        print('STEP 2: Skipping download (--skip-download flag)')
        print('='*60)
        
        # Verify files exist
        required_files = ['tracks.parquet', 'catalog_names.parquet', 'interactions.parquet']
        missing = [f for f in required_files if not os.path.exists(os.path.join(raw_dir, f))]
        
        if missing:
            print(f'❌ Missing files: {missing}')
            print(f'   Run without --skip-download to download them')
            sys.exit(1)
        
        print(f'✅ All raw files present in {raw_dir}')
    
    # Step 3: Run preprocessing pipeline
    print('\n' + '='*60)
    print('STEP 3: Preprocessing data')
    print('='*60)
    
    try:
        items, tracks_catalog, events = run_preprocessing(raw_dir, preprocessed_dir)
        
        print('\n' + '='*60)
        print('✅ Pipeline completed successfully')
        print('='*60)
        
    except Exception as e:
        print(f'\n❌ Preprocessing failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()




