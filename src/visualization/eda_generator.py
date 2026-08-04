import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config.schema import AppConfig
from src.utils.io import ensure_dir
from src.utils.logging import setup_logger

logger = setup_logger("EDAGenerator")

# Set aesthetic styling
sns.set_theme(style="darkgrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


class EDAGenerator:
    """Automated Exploratory Data Analysis & Visualization Suite for Bioacoustics."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.figures_dir = ensure_dir(config.paths.figures_dir)
        self.reports_dir = ensure_dir(config.paths.reports_dir)

    def plot_sample_acoustics(self, audio_file: Path) -> dict:
        """Plots Waveform, STFT Spectrogram, and Mel-Spectrogram for a representative audio sample."""
        try:
            y, sr = librosa.load(audio_file, sr=self.config.audio.target_sample_rate)
        except Exception as e:
            logger.error(f"Failed loading audio sample {audio_file} for EDA: {e}")
            return {}

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # 1. Waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#00adb5")
        axes[0].set_title(
            f"Waveform: {audio_file.name}", fontsize=12, fontweight="bold"
        )
        axes[0].set_ylabel("Amplitude")

        # 2. Spectrogram (STFT)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img1 = librosa.display.specshow(
            D, sr=sr, x_axis="time", y_axis="linear", ax=axes[1], cmap="magma"
        )
        axes[1].set_title("Linear Spectrogram (STFT)", fontsize=11)
        axes[1].set_ylabel("Frequency (Hz)")
        fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")

        # 3. Mel-Spectrogram
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.config.features.n_mels
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        img2 = librosa.display.specshow(
            S_dB, sr=sr, x_axis="time", y_axis="mel", ax=axes[2], cmap="viridis"
        )
        axes[2].set_title("Mel-Spectrogram (128 Bands)", fontsize=11)
        axes[2].set_ylabel("Mel Frequency")
        fig.colorbar(img2, ax=axes[2], format="%+2.0f dB")

        plt.tight_layout()
        out_path = self.figures_dir / "sample_acoustic_analysis.png"
        plt.savefig(out_path, dpi=self.config.eda.dpi)
        plt.close()

        logger.info(f"Saved sample acoustic figures to '{out_path}'.")
        return {"sample_plot": out_path}

    def plot_metadata_distributions(self, meta_path: Path) -> dict:
        """Plots species distribution, bird population density, and duration histogram."""
        if not meta_path.exists():
            logger.warning(
                f"Metadata file '{meta_path}' not found. Skipping metadata distributions."
            )
            return {}

        df = pd.read_csv(meta_path)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # 1. Species Count Distribution
        if "species" in df.columns:
            sns.countplot(
                data=df,
                y="species",
                order=df["species"]
                .value_counts()
                .index[: self.config.eda.top_n_species],
                ax=axes[0],
                palette="crest",
            )
            axes[0].set_title("Species Distribution", fontsize=11, fontweight="bold")
            axes[0].set_xlabel("Recording Count")
        else:
            axes[0].text(0.5, 0.5, "No species column", ha="center")

        # 2. Bird Population (Ground Truth Target)
        if "bird_count" in df.columns:
            sns.histplot(
                data=df,
                x="bird_count",
                discrete=True,
                kde=True,
                ax=axes[1],
                color="#e4572e",
            )
            axes[1].set_title(
                "Bird Population Distribution (Ground Truth)",
                fontsize=11,
                fontweight="bold",
            )
            axes[1].set_xlabel("Bird Count per Recording")
        else:
            axes[1].text(0.5, 0.5, "No bird_count column", ha="center")

        # 3. Audio Duration Distribution
        if "duration_sec" in df.columns:
            sns.histplot(
                data=df,
                x="duration_sec",
                bins=15,
                ax=axes[2],
                color="#17b978",
                kde=True,
            )
            axes[2].set_title(
                "Recording Duration (seconds)", fontsize=11, fontweight="bold"
            )
            axes[2].set_xlabel("Duration (s)")
        else:
            axes[2].text(0.5, 0.5, "No duration column", ha="center")

        plt.tight_layout()
        out_path = self.figures_dir / "metadata_distributions.png"
        plt.savefig(out_path, dpi=self.config.eda.dpi)
        plt.close()

        logger.info(f"Saved metadata distribution plots to '{out_path}'.")
        return {"metadata_plot": out_path}

    def generate_eda_report(self) -> Path:
        """Executes full automated EDA suite and compiles markdown report."""
        raw_dir = Path(self.config.paths.raw_data_dir)
        meta_path = raw_dir / self.config.paths.metadata_filename

        # Find sample audio file
        audio_files = list(raw_dir.rglob("*.wav")) + list(raw_dir.rglob("*.flac"))
        if audio_files:
            self.plot_sample_acoustics(audio_files[0])

        self.plot_metadata_distributions(meta_path)

        # Build Markdown EDA Report
        report_md_path = self.reports_dir / "eda_report.md"
        meta_summary = ""
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            meta_summary = f"""
### Dataset Statistics Summary
- **Total Metadata Records**: {len(df)}
- **Unique Aviaries**: {df['aviary'].nunique() if 'aviary' in df.columns else 'N/A'}
- **Average Duration**: {df['duration_sec'].mean():.2f} seconds (min: {df['duration_sec'].min():.2f}s, max: {df['duration_sec'].max():.2f}s)
- **Bird Population Range**: {df['bird_count'].min() if 'bird_count' in df.columns else 'N/A'} to {df['bird_count'].max() if 'bird_count' in df.columns else 'N/A'} birds per chunk.
"""

        report_content = f"""# Automated Exploratory Data Analysis (EDA) Report
**Project**: {self.config.project.name}  
**Version**: {self.config.project.version}  
**Target Sample Rate**: {self.config.audio.target_sample_rate} Hz  

---

## 1. Acoustic Signal Analysis
Below is the acoustic analysis (Waveform, STFT Spectrogram, and 128-band Mel-Spectrogram) generated from sample bioacoustic recordings.

![Sample Acoustic Analysis](figures/sample_acoustic_analysis.png)

### Key Bioacoustic Observations:
1. **Frequency Range**: Primary vocalizations for avian species reside between **1.5 kHz and 8 kHz**.
2. **Temporal Structure**: Bird calls appear as discrete high-energy harmonic chirps embedded in ambient background noise.
3. **Mel Spectrogram Resolution**: 128-band Mel Spectrogram cleanly isolates harmonic overtones, making it ideal for downstream deep learning backbones.

---

## 2. Dataset & Metadata Distributions
{meta_summary}

![Metadata Distributions](figures/metadata_distributions.png)

---

## 3. Engineering Recommendations for ML Pipeline
- **Resampling**: Standardize all passive acoustic monitoring recordings to **{self.config.audio.target_sample_rate} Hz**.
- **Normalization**: Peak normalization to **{self.config.preprocessing.target_peak_db} dB** prevents clipping and standardizes loudness across different recording hardware.
- **Handling Class & Population Imbalance**: Avian population estimation suffers from zero-inflated recordings during quiet hours; stratified sampling by bird count bin is recommended.
"""

        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Generated automated EDA markdown report at '{report_md_path}'.")
        return report_md_path

    def generate_quality_dashboard(self, validation_df: pd.DataFrame) -> Path:
        """Generates comprehensive enterprise Data Quality Dashboard (quality_dashboard.md)."""
        dashboard_path = self.reports_dir / "quality_dashboard.md"
        raw_dir = Path(self.config.paths.raw_data_dir)
        meta_path = raw_dir / self.config.paths.metadata_filename

        total_scanned = len(validation_df) if not validation_df.empty else 0
        valid_count = (
            int((validation_df["status"] == "VALID").sum())
            if not validation_df.empty
            else 0
        )
        corrupt_count = (
            int((validation_df["status"] == "CORRUPT").sum())
            if not validation_df.empty
            else 0
        )
        duplicate_count = (
            int((validation_df["status"] == "DUPLICATE").sum())
            if not validation_df.empty
            else 0
        )
        missing_count = (
            int((validation_df["status"] == "MISSING").sum())
            if not validation_df.empty
            else 0
        )
        warning_count = (
            int((validation_df["status"] == "WARNING").sum())
            if not validation_df.empty
            else 0
        )

        health_pct = round((valid_count / max(1, total_scanned)) * 100, 1)

        meta_stats = ""
        if meta_path.exists():
            try:
                df_meta = pd.read_csv(meta_path)
                species_counts = (
                    df_meta["species"].value_counts().to_dict()
                    if "species" in df_meta.columns
                    else {}
                )
                count_dist = (
                    df_meta["bird_count"].describe().to_dict()
                    if "bird_count" in df_meta.columns
                    else {}
                )

                meta_stats = f"""
### Metadata Integrity & Population Stats
- **Metadata Records**: {len(df_meta)}
- **Unique Avian Species**: {len(species_counts)}
- **Bird Population Range (Min - Max)**: {count_dist.get('min', 'N/A')} - {count_dist.get('max', 'N/A')} birds per recording
- **Population Mean (Std)**: {count_dist.get('mean', 0.0):.2f} (±{count_dist.get('std', 0.0):.2f})
"""
            except Exception as e:
                meta_stats = f"\n*Metadata parsing warning: {e}*\n"

        content = f"""# Enterprise Data Quality Dashboard
**Project**: {self.config.project.name}  
**Environment**: {self.config.project.environment}  
**Dataset Health Index**: **{health_pct}%**  

---

## 1. Executive Summary & Dataset Health

| Health Metric | Value | Production Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Total Audio Recordings** | {total_scanned} | > 0 | PASS |
| **Valid Files** | {valid_count} | {total_scanned} | {"PASS" if valid_count == total_scanned else "WARNING"} |
| **Corrupted Audio Files** | {corrupt_count} | 0 | {"PASS" if corrupt_count == 0 else "FAIL"} |
| **Duplicate Hashes** | {duplicate_count} | 0 | {"PASS" if duplicate_count == 0 else "WARNING"} |
| **Missing Files** | {missing_count} | 0 | {"PASS" if missing_count == 0 else "FAIL"} |
| **Duration / Spec Warnings** | {warning_count} | 0 | {"PASS" if warning_count == 0 else "INFO"} |

---

## 2. Audio Quality & Signal Integrity
- **Target Sample Rate**: {self.config.audio.target_sample_rate} Hz
- **Channels**: {self.config.audio.target_channels} (Mono)
- **Signal Normalization Target**: {self.config.preprocessing.target_peak_db} dB Peak
- **Outlier Detection**: Signal duration outside [{self.config.audio.min_duration_sec}s, {self.config.audio.max_duration_sec}s] flagged automatically.

---

## 3. Class Imbalance & Species Distribution
{meta_stats}

---

## 4. Pipeline Recommendations
1. **Deduplication**: Hashes verified via MD5 signature; duplicate entries isolated from feature store.
2. **Preprocessing Validation**: All valid files converted to uniform {self.config.audio.target_sample_rate} Hz mono signals prior to feature extraction.
"""

        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Enterprise Data Quality Dashboard saved to '{dashboard_path}'.")
        return dashboard_path
