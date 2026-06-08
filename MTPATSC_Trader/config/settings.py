import os
from pathlib import Path

# ── Symbol & Market ────────────────────────────────
SYMBOL = "XAUUSD"
K_MULTIPLIER = 0.00118     # MTPATSC training pipeline K (not 0.00295)
BRICK_SIZE = 5.3           # Default/Initial (will be overwritten by sync)
SPREAD_LIMIT = 0.5         # Max spread allowed for execution (absolute)

# ── Risk Management ────────────────────────────────
LOT_SIZE = 0.01            # Fixed lot size
DAILY_LOSS_LIMIT_R = 5.0   # Daily drawdown in R-units (5 × brick_size)
SL_MULT = 1.0              # SL = SL_MULT * BRICK_SIZE

# ── MTPATSC Model ──────────────────────────────────
# Thresholds are loaded from models/config.json at runtime.
# T2, T3, T4 are DISABLED (threshold=1.0) per user request.

# ── Tensor Features ────────────────────────────────
ANCS_FINE_SEGMENTS = 10    # 10 segments × 6 features = 60
ANCS_COARSE_SEGMENTS = 5   # 5 segments × 6 features = 30
CANDLE_FEATURES = 15
MOMENTUM_FEATURES = 19
SCALAR_DIM = 34            # candle(15) + momentum(19)
HISTORY_BRICKS = 5         # Rolling history of 5 bricks

# ── Paths ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
STATE_FILE = LOGS_DIR / "state.json"
MTPATSC_DIR = BASE_DIR.parent / "MTPATSC"
