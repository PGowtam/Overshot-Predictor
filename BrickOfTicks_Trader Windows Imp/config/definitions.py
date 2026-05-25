from pathlib import Path

# Base Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
INFERENCE_DIR = ROOT_DIR / "inference"
EXECUTION_DIR = ROOT_DIR / "execution"
UTILS_DIR = ROOT_DIR / "utils"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
TESTS_DIR = ROOT_DIR / "tests"

# File Paths
STATE_FILE = LOGS_DIR / "state.json"
TRADER_LOG = LOGS_DIR / "trader.log"
