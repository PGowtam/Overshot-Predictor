import os
from pathlib import Path

# ── Symbol & Market ────────────────────────────────
SYMBOL = "XAUUSD"
BRICK_SIZE_FACTOR = 0.0018 # Dynamic factor (size = factor * open)
BRICK_SIZE = 4.3           # Default/Initial (will be overwritten by sync)
SPREAD_LIMIT = 0.5         # Max spread allowed for execution

# ── Risk Management ────────────────────────────────
LOT_SIZE = 1.0           # Static lot size for now
DAILY_LOSS_LIMIT = 1000.0# Equity-based daily drawdown cap
SL_MULT = 1.0            # SL = SL_MULT * BRICK_SIZE
TP_MULT = 1.0            # TP = TP_MULT * BRICK_SIZE

# ── Ensemble Inference (Optimized 2024) ─────────────
ENSEMBLE_VOTE_THRESHOLD = 2   # Majority vote (2 out of 3)

# Standard Strategy thresholds
PROB_WIN_THRESHOLD = 0.7      # Sigmoid Head A
PRED_OS_THRESHOLD = 1.2       # ReLU Head B

# Baiting (Reversal) thresholds (Selection: 0.2 / 0.7)
BAIT_PROB_WIN_THRESHOLD = 0.2 # Head A < 0.2
BAIT_PRED_OS_THRESHOLD = 0.7  # Head B < 0.7

# ── Tensor Features ────────────────────────────────
WINDOW_SIZE = 10         # Lookback in bricks
MICRO_TICKS = 100        # Number of ticks inside micro-buffer
FEATURE_COUNT = 9        # Number of dimensions per tick
MACRO_COUNT = 3          # OFI, Volume, Volatility

# ── Feature Engine ─────────────────────────────────
W_ROLLING = 5000         # Z-score rolling window size
WARMUP_TICKS = 10000     # Ticks to process before first trade

# ── Paths ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
STATE_FILE = LOGS_DIR / "state.json"
