import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.config.schema import AppConfig
from src.data.dummy_generator import AudioDummyDatasetGenerator
from src.utils.logging import setup_logger

logger = setup_logger("Script_DownloadDataset")


def main():
    parser = argparse.ArgumentParser(
        description="Download or generate BioDCASE Avian Bioacoustics Dataset."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic developmental aviary dataset.",
    )
    parser.add_argument(
        "--num-aviaries", type=int, default=2, help="Number of aviaries to generate."
    )
    parser.add_argument(
        "--samples-per-aviary",
        type=int,
        default=5,
        help="Number of audio samples per aviary.",
    )
    args = parser.parse_args()

    config = AppConfig()
    logger.info(
        f"Generating synthetic BioDCASE dataset ({args.num_aviaries} aviaries, {args.samples_per_aviary} recordings/aviary)..."
    )
    gen = AudioDummyDatasetGenerator(
        output_dir=Path(config.paths.raw_data_dir),
        sample_rate=config.audio.target_sample_rate,
    )
    gen.generate_dataset(
        num_aviaries=args.num_aviaries, samples_per_aviary=args.samples_per_aviary
    )
    logger.info("Dataset initialization completed successfully.")


if __name__ == "__main__":
    main()
