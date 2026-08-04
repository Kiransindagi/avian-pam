import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.config.schema import AppConfig
from src.pipeline import BioAcousticPipeline
from src.utils.logging import setup_logger

logger = setup_logger("Script_GenerateFeatures")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone feature extraction and feature store exporter."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/development.yaml",
        help="Path to config file.",
    )
    parser.parse_args()

    config = AppConfig()
    pipeline = BioAcousticPipeline(config)
    logger.info("Starting feature extraction stage...")
    feats_df = pipeline.run_feature_extraction()
    store_file = pipeline.run_feature_store(feats_df)
    logger.info(f"Feature extraction completed successfully: '{store_file}'.")


if __name__ == "__main__":
    main()
