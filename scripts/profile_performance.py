import sys
import cProfile
import pstats
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.schema import AppConfig
from src.pipeline import BioAcousticPipeline
from src.utils.logging import setup_logger

logger = setup_logger("Script_Profile")


def main():
    logger.info("Starting Python memory & execution profiling...")
    pr = cProfile.Profile()
    pr.enable()

    config = AppConfig()
    pipeline = BioAcousticPipeline(config)
    pipeline.run_validation()

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(15)

    logger.info("Profiling Top 15 Operations:\n" + s.getvalue())


if __name__ == "__main__":
    main()
