import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.config.schema import AppConfig
from src.training.trainer import ModelTrainer
from src.utils.logging import setup_logger

logger = setup_logger("Script_TrainModels")


def main():
    parser = argparse.ArgumentParser(
        description="Master ML Training & Experimentation Platform CLI."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/development.yaml",
        help="Path to configuration file.",
    )
    parser.parse_args()

    config = AppConfig()
    logger.info("Executing Sprint 3 ML Training & Experiment Suite...")
    trainer = ModelTrainer(config)
    df_leaderboard = trainer.train_and_evaluate_all_models()
    logger.info(
        f"Training finished. Models ranked:\n{df_leaderboard[['model_name', 'cv_mae_mean', 'r2_score']]}"
    )


if __name__ == "__main__":
    main()
