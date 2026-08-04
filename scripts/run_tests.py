import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.utils.logging import setup_logger

logger = setup_logger("Script_RunTests")


def main():
    logger.info("Executing automated Pytest test suite...")
    ret_code = pytest.main(["tests/", "-v"])
    sys.exit(ret_code)


if __name__ == "__main__":
    main()
