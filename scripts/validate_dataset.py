import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.config.schema import AppConfig
from src.data.validator import AudioValidator
from src.utils.logging import setup_logger

logger = setup_logger("Script_ValidateDataset")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone dataset integrity and hashing validator script."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/development.yaml",
        help="Path to config file.",
    )
    parser.parse_args()

    config = AppConfig()
    logger.info("Executing standalone dataset integrity validation...")
    validator = AudioValidator(config)
    report_df = validator.validate_dataset()
    logger.info(f"Validation finished. Total files scanned: {len(report_df)}.")


if __name__ == "__main__":
    main()
