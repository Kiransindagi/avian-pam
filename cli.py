import argparse
import sys
from pathlib import Path

from src.config.schema import AppConfig
from src.evaluation.reporter import MasterResearchEvaluator
from src.inference.engine import AvianInferenceEngine
from src.inference.submission import BioDCASESubmissionGenerator
from src.pipeline import BioAcousticPipeline
from src.training.trainer import ModelTrainer
from src.utils.logging import setup_logger

logger = setup_logger("Unified_CLI")


def main():
    parser = argparse.ArgumentParser(
        description="BioDCASE Avian Passive Acoustic Monitoring Master CLI Platform.",
        prog="avian-pam",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/development.yaml",
        help="Path to config file.",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available platform CLI commands"
    )

    # Command: train
    subparsers.add_parser("train", help="Train and benchmark all ML models.")

    # Command: evaluate
    subparsers.add_parser(
        "evaluate",
        help="Execute scientific evaluation, explainability & robustness suite.",
    )

    # Command: predict
    predict_parser = subparsers.add_parser(
        "predict", help="Run inference on single audio file or directory."
    )
    predict_parser.add_argument(
        "--input", type=str, required=True, help="Path to audio file or directory."
    )

    # Command: pipeline
    subparsers.add_parser(
        "pipeline", help="Run full end-to-end data, feature & model pipeline."
    )

    # Command: feature-extract
    subparsers.add_parser("feature-extract", help="Run feature extraction platform v2.")

    # Command: submit
    submit_parser = subparsers.add_parser(
        "submit", help="Generate official BioDCASE submission CSV."
    )
    submit_parser.add_argument(
        "--data-dir", type=str, default="data/raw", help="Path to raw audio folder."
    )

    # Command: validate
    subparsers.add_parser("validate", help="Run dataset audio contract validation.")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    config = AppConfig()

    if args.command == "train":
        logger.info("Executing ML Training Platform...")
        trainer = ModelTrainer(config)
        trainer.train_and_evaluate_all_models()

    elif args.command == "evaluate":
        logger.info("Executing Master Scientific Evaluation...")
        evaluator = MasterResearchEvaluator(config)
        evaluator.run_full_scientific_evaluation()

    elif args.command == "predict":
        engine = AvianInferenceEngine(config)
        p = Path(args.input)
        if p.is_file():
            res = engine.predict_audio_file(p)
            print("\n--- PREDICTION RESULT ---")
            for k, v in res.items():
                print(f"{k}: {v}")
        elif p.is_dir():
            df_preds = engine.predict_batch_dir(p)
            print(f"\n--- BATCH PREDICTIONS ({len(df_preds)} files) ---")
            print(
                df_preds[
                    [
                        "filename",
                        "predicted_bird_count",
                        "estimated_integer_count",
                        "inference_latency_ms",
                    ]
                ]
            )

    elif args.command == "pipeline":
        pipeline = BioAcousticPipeline(config)
        pipeline.run_full_pipeline()

    elif args.command == "feature-extract":
        pipeline = BioAcousticPipeline(config)
        pipeline.run_feature_extraction()

    elif args.command == "submit":
        gen = BioDCASESubmissionGenerator(config)
        sub_path = gen.generate_submission(args.data_dir)
        print(f"Generated BioDCASE submission at '{sub_path}'.")

    elif args.command == "validate":
        pipeline = BioAcousticPipeline(config)
        pipeline.run_validation()


if __name__ == "__main__":
    main()
