import argparse
import sys
from pathlib import Path
from src.config.schema import AppConfig
from src.data.dummy_generator import generate_dummy_dataset
from src.pipeline import BioAcousticPipeline
from src.utils.logging import setup_logger

logger = setup_logger("CLI")


def parse_args():
    parser = argparse.ArgumentParser(
        description="BioDCASE 2026 Passive Acoustic Monitoring Pipeline CLI"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--generate-dummy",
        action="store_true",
        help="Generate synthetic bioacoustic dataset in data/raw for testing",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "validate", "preprocess", "eda", "features"],
        default="all",
        help="Pipeline stage to execute",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(
            f"Config file not found at '{config_path}'. Please specify a valid YAML file."
        )
        sys.exit(1)

    config = AppConfig.load_from_yaml(config_path)

    if args.generate_dummy:
        generate_dummy_dataset(config)
        sys.exit(0)

    pipeline = BioAcousticPipeline(config)

    if args.stage == "all":
        pipeline.run_full_pipeline()
    elif args.stage == "validate":
        pipeline.run_validation()
    elif args.stage == "preprocess":
        pipeline.run_preprocessing()
    elif args.stage == "eda":
        pipeline.run_eda()
    elif args.stage == "features":
        df = pipeline.run_feature_extraction()
        pipeline.run_feature_store(df)


if __name__ == "__main__":
    main()
