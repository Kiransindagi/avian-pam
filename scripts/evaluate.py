import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.config.schema import AppConfig
from src.evaluation.reporter import MasterResearchEvaluator
from src.utils.logging import setup_logger

logger = setup_logger("Script_Evaluate")


def main():
    parser = argparse.ArgumentParser(
        description="Master Research Evaluation & Report Generator CLI."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/development.yaml",
        help="Path to config file.",
    )
    parser.parse_args()

    config = AppConfig()
    logger.info("Executing Sprint 4 Master Scientific Evaluation Suite...")
    evaluator = MasterResearchEvaluator(config)
    evaluator.run_full_scientific_evaluation()
    logger.info(
        "Sprint 4 Scientific Evaluation complete. All reports and figures generated."
    )


if __name__ == "__main__":
    main()
