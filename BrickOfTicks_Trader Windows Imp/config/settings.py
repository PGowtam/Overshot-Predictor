import os
from pathlib import Path

# ── Symbol & Market ────────────────────────────────
SYMBOL = "XAUUSD"
BRICK_SIZE_FACTOR = 0.00118  # Dynamic factor (size = 0.00236/2 * open, matching training)
BRICK_SIZE = 4.3             # Default/Initial (will be overwritten by sync)
SPREAD_LIMIT = 0.5           # Max spread allowed for execution

# ── Path Optimizer (Phase 13) ─────────────────────
PATH_LOOKBACK_DAYS = 5       # Max trading days to fetch for path optimization
PATH_ANCHOR_LOOKBACK = 6     # Days before lookback window to find anchor day
PATH_STEP_FACTOR = 0.00236   # Step size base factor (matches training: open * 0.00236 * 0.01)
PATH_BE_TRIGGER = 0.3125     # Break-even trigger as fraction of brick size (matches training)

# ── Risk Management ────────────────────────────────
LOT_SIZE = 0.01             # User specified lot size
DAILY_LOSS_LIMIT = 1000.0   # Equity-based daily drawdown cap
SL_MULT = 1.0               # SL = SL_MULT * BRICK_SIZE
TP_MULT = 1.0               # TP = TP_MULT * BRICK_SIZE
MAGIC_NUMBER = 314159       # Unique ID for bot orders
DEVIATION = 20              # Max slippage in points
FILLING_MODE = "IOC"        # User specified filling mode

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
Z_SCORE_WINDOW = 1000    # Parity with training: 1000-tick window
Z_SCORE_WARMUP = 30      # Start returning values after 30 ticks
WARMUP_TICKS = 250000    # System warmup: replay 250k ticks at startup

# ── Paths ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
STATE_FILE = LOGS_DIR / "state.json"
