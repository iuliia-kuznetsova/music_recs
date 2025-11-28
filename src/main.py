import argparse
import logging

from src.preprocess_data import PreprocessConfig, run_preprocessing

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Canonical preprocessing pipeline')
    parser.add_argument('--raw-dir', help='Path to raw parquet files (default: RAW_DATA_DIR env)')
    parser.add_argument('--preprocessed-dir', help='Output directory for processed datasets', dest='pre_dir')
    parser.add_argument('--checkpoint-dir', help='Directory to store intermediate checkpoints')
    parser.add_argument('--no-cache-fact-ids', action='store_true', help='Disable fact_ids parquet checkpoint')
    parser.add_argument('--no-cache-items', action='store_true', help='Skip writing final items parquet (still returned)')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = PreprocessConfig.from_env(
        raw_dir=args.raw_dir,
        preprocessed_dir=args.pre_dir,
        checkpoint_dir=args.checkpoint_dir,
        cache_fact_ids=not args.no_cache_fact_ids,
        cache_items=not args.no_cache_items,
    )

    logger.info('Running preprocessing pipeline from CLI')
    run_preprocessing(config)


if __name__ == '__main__':
    main()

