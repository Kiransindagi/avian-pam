from pathlib import Path
from typing import Dict
import pandas as pd
import soundfile as sf
from src.config.schema import AppConfig
from src.utils.io import compute_file_hash, ensure_dir
from src.utils.logging import setup_logger


class AudioValidator:
    """Automated data validation system for passive acoustic recordings."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = setup_logger(
            "AudioValidator",
            log_file=config.validation.log_file,
        )

    def validate_single_file(
        self, file_path: Path
    ) -> Dict[str, str | float | int | bool]:
        """Validates individual audio file integrity, properties, and hash."""
        rel_path = str(file_path)
        res = {
            "file_path": rel_path,
            "filename": file_path.name,
            "status": "VALID",
            "error_msg": "",
            "sample_rate": 0,
            "channels": 0,
            "duration_sec": 0.0,
            "file_size_bytes": 0,
            "file_hash": "",
        }

        if not file_path.exists():
            res["status"] = "MISSING"
            res["error_msg"] = "File does not exist on disk"
            return res

        try:
            res["file_size_bytes"] = file_path.stat().st_size
            res["file_hash"] = compute_file_hash(
                file_path, algorithm=self.config.validation.hash_algorithm
            )

            # Attempt audio file read metadata
            info = sf.info(file_path)
            res["sample_rate"] = info.samplerate
            res["channels"] = info.channels
            res["duration_sec"] = round(info.duration, 3)

            # Check duration limits
            if info.duration < self.config.audio.min_duration_sec:
                res["status"] = "WARNING"
                res["error_msg"] += (
                    f"Duration ({info.duration:.2f}s) below minimum threshold; "
                )
            elif info.duration > self.config.audio.max_duration_sec:
                res["status"] = "WARNING"
                res["error_msg"] += (
                    f"Duration ({info.duration:.2f}s) above maximum threshold; "
                )

            # Check extension
            if file_path.suffix.lower() not in self.config.audio.valid_extensions:
                res["status"] = "CORRUPT"
                res["error_msg"] += f"Invalid extension '{file_path.suffix}'; "

        except Exception as e:
            res["status"] = "CORRUPT"
            res["error_msg"] = f"Corrupted audio or read failure: {str(e)}"

        return res

    def validate_dataset(self) -> pd.DataFrame:
        """Runs validation suite across entire dataset specified in raw_data_dir."""
        raw_dir = Path(self.config.paths.raw_data_dir)
        self.logger.info(f"Starting dataset validation scan in '{raw_dir}'...")

        if not raw_dir.exists():
            self.logger.warning(
                f"Raw directory '{raw_dir}' does not exist. Returning empty report."
            )
            return pd.DataFrame()

        audio_files = []
        for ext in self.config.audio.valid_extensions:
            audio_files.extend(list(raw_dir.rglob(f"*{ext}")))

        self.logger.info(f"Found {len(audio_files)} audio files for validation.")

        reports = []
        seen_hashes: Dict[str, str] = {}

        for file_path in audio_files:
            report_item = self.validate_single_file(file_path)

            # Duplicate file check via hash
            f_hash = str(report_item["file_hash"])
            if f_hash and f_hash in seen_hashes:
                report_item["status"] = "DUPLICATE"
                report_item["error_msg"] += f"Duplicate hash of {seen_hashes[f_hash]}; "
            elif f_hash:
                seen_hashes[f_hash] = str(file_path.name)

            reports.append(report_item)
            if report_item["status"] != "VALID":
                self.logger.warning(
                    f"[{report_item['status']}] File: {file_path.name} | Issue: {report_item['error_msg']}"
                )

        report_df = pd.DataFrame(reports)

        # Check metadata consistency if metadata CSV exists
        meta_path = raw_dir / self.config.paths.metadata_filename
        if meta_path.exists():
            self.logger.info(f"Cross-referencing with metadata file '{meta_path}'...")
            try:
                meta_df = pd.read_csv(meta_path)
                if "filename" in meta_df.columns:
                    meta_filenames = set(meta_df["filename"].astype(str))
                    scanned_filenames = (
                        set(report_df["filename"].astype(str))
                        if not report_df.empty
                        else set()
                    )

                    missing_on_disk = meta_filenames - scanned_filenames
                    for m_file in missing_on_disk:
                        reports.append(
                            {
                                "file_path": str(raw_dir / m_file),
                                "filename": m_file,
                                "status": "MISSING",
                                "error_msg": "Listed in metadata.csv but missing from disk",
                                "sample_rate": 0,
                                "channels": 0,
                                "duration_sec": 0.0,
                                "file_size_bytes": 0,
                                "file_hash": "",
                            }
                        )
                        self.logger.error(
                            f"[MISSING] File listed in metadata not found: {m_file}"
                        )
                    report_df = pd.DataFrame(reports)
            except Exception as e:
                self.logger.error(f"Error reading metadata.csv: {e}")

        # Save validation report
        out_report_path = Path(self.config.validation.report_file)
        ensure_dir(out_report_path.parent)
        report_df.to_csv(out_report_path, index=False)

        valid_count = (
            (report_df["status"] == "VALID").sum() if not report_df.empty else 0
        )
        self.logger.info(
            f"Validation scan complete. Total evaluated: {len(report_df)}. "
            f"Valid: {valid_count}, Warnings/Errors: {len(report_df) - valid_count}. "
            f"Report saved to '{out_report_path}'."
        )

        return report_df
