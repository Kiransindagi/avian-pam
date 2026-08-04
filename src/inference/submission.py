import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Union, Optional
from src.config.schema import AppConfig
from src.inference.engine import AvianInferenceEngine
from src.utils.io import ensure_dir, compute_file_hash
from src.utils.logging import setup_logger

logger = setup_logger("BioDCASESubmissionGenerator")


class BioDCASESubmissionGenerator:
    """Official BioDCASE Submission File Pipeline Generator."""

    def __init__(self, config: AppConfig, model_checkpoint_path: Optional[Path] = None):
        self.config = config
        self.submissions_dir = ensure_dir(Path("submissions"))
        self.engine = AvianInferenceEngine(
            config, model_checkpoint_path=model_checkpoint_path
        )

    def generate_submission(self, eval_audio_dir: Union[str, Path]) -> Path:
        """Generates formatted BioDCASE submission CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sub_filename = f"submission_v{self.config.project.version}_{timestamp}.csv"
        out_path = self.submissions_dir / sub_filename

        df_preds = self.engine.predict_batch_dir(eval_audio_dir)

        if df_preds.empty:
            raise ValueError(
                f"No audio predictions generated from directory '{eval_audio_dir}'."
            )

        # Format submission DataFrame according to official BioDCASE challenge specification
        df_sub = pd.DataFrame(
            {
                "filename": df_preds["filename"],
                "predicted_bird_count": df_preds["estimated_integer_count"],
                "raw_count_estimate": df_preds["predicted_bird_count"],
            }
        )

        df_sub.to_csv(out_path, index=False)
        sub_hash = compute_file_hash(out_path)

        logger.info(
            f"BioDCASE Submission generated successfully at '{out_path}' "
            f"({len(df_sub)} predictions, MD5 Hash: {sub_hash[:10]}...)."
        )
        return out_path
