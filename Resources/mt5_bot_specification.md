# MT5 Trading Bot — Production Specification

> **Purpose**: Complete technical blueprint for building a production-ready MetaTrader 5 trading bot that deploys the BrickOfTicks CNN+LSTM ensemble. This document is self-contained — an engineer reading only this file should be able to build the entire bot from scratch.

---

## 1. System Overview

### 1.1 What the Bot Does
1. Connects to MT5 and streams real-time L1 ticks for XAUUSD.
2. Constructs Renko bricks from the tick stream (ATR-based dynamic sizing).
3. On every brick close, computes a 9-dimensional microstructure feature vector from the last 100 ticks and a 3-dimensional macro vector from the last 10 bricks.
4. Feeds the tensors through 3 pre-trained CNN+LSTM models (ensemble).
5. Applies majority voting for standard signals (ENTER if ≥ 2 of 3 models signal).
6. Applies "Baiting" logic for reversal signals (REVERSE if all models predict high-confidence loss).
7. Places orders with SL/TP = 1× brick_size and manages open positions.
8. Enforces daily risk limits and persists state for crash recovery.

### 1.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MT5 Terminal (Windows)                       │
│  ┌───────────────┐    ┌─────────────────┐    ┌──────────────────┐  │
│  │  Tick Stream   │    │  Order Gateway   │    │  Account Info    │  │
│  │ copy_ticks_from│    │  order_send()    │    │  account_info()  │  │
│  └───────┬───────┘    └────────▲────────┘    └────────▲─────────┘  │
└──────────┼─────────────────────┼─────────────────────┼──────────────┘
           │                     │                     │
    Python Bridge (MetaTrader5 pip package)
           │                     │                     │
┌──────────▼─────────────────────┼─────────────────────┼──────────────┐
│                     BrickOfTicks Bot (Python)                       │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │  TickStream   │──▶│ RenkoBuilder │──▶│ On Brick Close:      │    │
│  │  (gap-less)   │   │              │   │  1. Feature Engine   │    │
│  └──────────────┘   └──────────────┘   │  2. Buffer Snapshot  │    │
│                                         │  3. Tensor Assembly  │    │
│                                         │  4. Model Inference  │    │
│                                         │  5. Ensemble Vote    │    │
│                                         │  6. Order Execution  │    │
│                                         └──────────┬───────────┘    │
│                                                    │                │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────▼───────────┐    │
│  │ StateManager │   │ RiskManager  │   │   OrderExecutor      │    │
│  │ (JSON disk)  │   │ (daily cap)  │   │   (market/limit)     │    │
│  └──────────────┘   └──────────────┘   └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
BrickOfTicks_Trader/
├── main.py                     # Entry point
├── config/
│   ├── __init__.py
│   ├── settings.py             # Symbol, risk params, lot size
│   └── definitions.py          # Path constants
├── data/
│   ├── __init__.py
│   ├── connector.py            # MT5 connection management
│   ├── tick_stream.py          # Gap-less tick fetching
│   ├── renko.py                # Renko brick construction
│   └── feature_engine.py       # 9D tick vector + z-scores (LIVE version)
├── inference/
│   ├── __init__.py
│   ├── buffer.py               # Micro-buffer (deque) + macro-history
│   ├── tensor_assembler.py     # Stack snapshots into model input
│   └── ensemble.py             # Load 3 models + voting logic
├── execution/
│   ├── __init__.py
│   ├── orders.py               # Market/Limit order placement
│   └── risk.py                 # Daily drawdown limit
├── utils/
│   ├── __init__.py
│   ├── logger.py               # Rotating file + console logger
│   └── state.py                # JSON state persistence (crash recovery)
├── models/                     # Pre-trained model files (copied from training)
│   ├── fold_1/
│   │   ├── model.keras
│   │   └── config.json
│   ├── fold_2/
│   │   ├── model.keras
│   │   └── config.json
│   └── fold_3/
│       ├── model.keras
│       └── config.json
├── logs/
├── requirements.txt
└── README.md
```

---

## 3. Configuration (`config/settings.py`)

```python
# ── Symbol ─────────────────────────────────────────
SYMBOL = "XAUUSD"             # Broker symbol name (check: .m suffix?)
TIMEFRAME_M1 = 1              # For history fetch (optimization)

# ── Trading ────────────────────────────────────────
LOT_SIZE = 0.01               # Start conservative for paper trading
MAGIC_NUMBER = 314159          # Unique ID for our orders
DEVIATION = 20                # Max slippage in points

# ── Risk ───────────────────────────────────────────
MAX_DAILY_DRAWDOWN_PCT = 0.03 # 3% daily loss limit → halt all trading
MAX_CONCURRENT_TRADES = 1     # Only 1 open position at a time

# ── Renko ──────────────────────────────────────────
BRICK_SIZE_FACTOR = 0.00118   # ≈ ATR factor for XAUUSD (~$2.50 at $2100)
                               # brick_size = current_price × BRICK_SIZE_FACTOR

# ── Feature Engine ─────────────────────────────────
Z_SCORE_WINDOW = 1000         # Rolling z-score window size
Z_SCORE_WARMUP = 30           # Minimum ticks before z-scores activate
MICRO_BUFFER_SIZE = 100       # deque(maxlen=100) for tick snapshots
MACRO_HISTORY_SIZE = 10       # Last 10 bricks for macro tensor

# ── Ensemble ───────────────────────────────────────
ENSEMBLE_VOTE_THRESHOLD = 2   # Minimum models that must agree (out of 3)
PROB_WIN_THRESHOLD = 0.7      # Optimized for Return/Trade
PRED_OS_THRESHOLD = 1.2       # Optimized for Return/Trade

# ── Baiting (Reversal) ─────────────────────────────
BAIT_PROB_WIN_THRESHOLD = 0.2 # Below this, model is likely wrong
BAIT_PRED_OS_THRESHOLD = 0.7  # Below this, model is likely wrong (WR > 83%)
```

---

## 4. Component Specifications

### 4.1 MT5 Connector (`data/connector.py`)

**Responsibility**: Initialize and maintain the MT5 connection.

```python
import MetaTrader5 as mt5

class MT5Connector:
    def connect(self) -> bool:
        """Initialize MT5 and select symbol. Return True on success."""
        if not mt5.initialize():
            return False
        if not mt5.symbol_select(SYMBOL, True):
            return False
        return True

    def shutdown(self):
        mt5.shutdown()

    def check_connection(self) -> bool:
        """Re-connect if dropped."""
        if mt5.terminal_info() is None:
            return self.connect()
        return True
```

**Key points from reference bot:**
- Call `mt5.initialize()` once at startup.
- `symbol_select(SYMBOL, True)` enables the symbol if it's not in MarketWatch.
- Check connection health periodically and reconnect if terminal_info returns None.

---

### 4.2 Gap-Less Tick Stream (`data/tick_stream.py`)

**Responsibility**: Fetch ALL ticks since last processed timestamp with zero gaps.

```python
import MetaTrader5 as mt5

class TickStream:
    def __init__(self):
        tick = mt5.symbol_info_tick(SYMBOL)
        self.last_time_msc = tick.time_msc  # Millisecond precision

    def fetch(self) -> list:
        """Return new ticks since last fetch. Empty list if none."""
        # CRITICAL: Convert ms to seconds for copy_ticks_from
        date_from_sec = self.last_time_msc / 1000.0
        ticks = mt5.copy_ticks_from(SYMBOL, date_from_sec, 1000, mt5.COPY_TICKS_ALL)

        if ticks is None or len(ticks) == 0:
            return []

        # Filter strictly > last_time_msc to avoid duplicates
        new_ticks = [t for t in ticks if t['time_msc'] > self.last_time_msc]

        if new_ticks:
            self.last_time_msc = new_ticks[-1]['time_msc']

        return new_ticks
```

**Why gap-less matters:**
- The feature engine needs EVERY tick to maintain z-score rolling windows accurately.
- Missing even 1 tick corrupts OFI calculations (delta bid/ask depends on previous tick).
- The reference bot uses `copy_ticks_from` which is inclusive of the start time — hence the `>` filter.

---

### 4.3 Renko Builder (`data/renko.py`)

**Responsibility**: Convert a stream of prices into Renko bricks.

```python
from collections import namedtuple

BrickEvent = namedtuple('BrickEvent', [
    'open', 'close', 'high', 'low',
    'uptrend',      # True (UP) or False (DOWN)
    'timestamp',    # Milliseconds
    'brick_size',   # Dynamic ATR-based size
    'sequence'      # Binary string "10110..." for last N directions
])

class RenkoBuilder:
    def __init__(self, brick_size: float, start_price: float):
        self.brick_size = brick_size
        self.current_price = start_price
        self.uptrend = 0   # 0=neutral, 1=up, -1=down
        self.history = []
        self.sequence = ""

    def update_tick(self, price: float, timestamp_ms: int) -> list[BrickEvent]:
        """Process a single price. Returns list of new bricks (0, 1, or many)."""
        new_bricks = []

        # UP check
        up_threshold = self.current_price + self.brick_size
        if self.uptrend == -1:
            up_threshold = self.current_price + (2 * self.brick_size)  # Reversal needs 2×

        while price >= (self.current_price + self.brick_size
                        if self.uptrend != -1
                        else self.current_price + 2 * self.brick_size):
            if self.uptrend == -1:
                self.current_price += self.brick_size  # Ghost/pivot brick

            self.current_price += self.brick_size
            self.uptrend = 1
            self.sequence += "1"
            if len(self.sequence) > 100:
                self.sequence = self.sequence[-100:]

            brick = BrickEvent(
                open=self.current_price - self.brick_size,
                close=self.current_price,
                high=self.current_price,
                low=self.current_price - self.brick_size,
                uptrend=True,
                timestamp=timestamp_ms,
                brick_size=self.brick_size,
                sequence=self.sequence
            )
            self.history.append(brick)
            new_bricks.append(brick)

        # DOWN check (mirror logic)
        while price <= (self.current_price - self.brick_size
                        if self.uptrend != 1
                        else self.current_price - 2 * self.brick_size):
            if self.uptrend == 1:
                self.current_price -= self.brick_size

            self.current_price -= self.brick_size
            self.uptrend = -1
            self.sequence += "0"
            if len(self.sequence) > 100:
                self.sequence = self.sequence[-100:]

            brick = BrickEvent(
                open=self.current_price + self.brick_size,
                close=self.current_price,
                high=self.current_price + self.brick_size,
                low=self.current_price,
                uptrend=False,
                timestamp=timestamp_ms,
                brick_size=self.brick_size,
                sequence=self.sequence
            )
            self.history.append(brick)
            new_bricks.append(brick)

        return new_bricks
```

**Critical implementation details:**
- Reversals require **2× brick_size** movement (not 1×). This matches how Renko charts work — a reversal brick's open is 1 brick_size away from the last close, and the new brick needs to close 1 brick_size further.
- The `while` loop handles gap fills — if price jumps 5× brick_size in one tick, 5 bricks are emitted.
- Feed the **bid price** to `update_tick`. The reference bot uses bid for Renko construction.

**Brick size determination:**
```python
# At session start, compute brick_size from current market price:
tick = mt5.symbol_info_tick(SYMBOL)
brick_size = tick.ask * BRICK_SIZE_FACTOR  # ≈ $2.50 for XAUUSD at $2100

# Alternatively, optimize from recent history (7-day M1 bars) like reference bot
```

---

### 4.4 Live Feature Engine (`data/feature_engine.py`)

**Responsibility**: Compute the 9D feature vector for every incoming tick, maintaining rolling z-score state.

This is the **most critical component** — it must replicate the training pipeline's feature computation *exactly*.

```python
from collections import deque
from math import sqrt

class RollingZScore:
    """O(1) incremental z-score with sliding window of 1000 ticks."""
    def __init__(self, window=1000):
        self.window = window
        self.deque = deque(maxlen=window)
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x_new: float) -> float:
        N = len(self.deque)
        if N == self.window:
            x_old = self.deque[0]
            self.deque.append(x_new)
            mean_new = self.mean + (x_new - x_old) / N
            self.M2 += (x_new - x_old) * ((x_new - mean_new) + (x_old - self.mean))
            self.mean = mean_new
            if self.M2 < 0: self.M2 = 0.0
            sigma = sqrt(self.M2 / (N - 1)) if N > 1 else 0.0
            return (x_new - self.mean) / sigma if sigma > 1e-12 else 0.0
        else:
            self.deque.append(x_new)
            N = len(self.deque)
            if N < 30: return 0.0
            arr = list(self.deque)
            self.mean = sum(arr) / N
            self.M2 = sum((x - self.mean)**2 for x in arr)
            sigma = sqrt(self.M2 / (N - 1)) if N > 1 else 0.0
            return (x_new - self.mean) / sigma if sigma > 1e-12 else 0.0


class LiveFeatureEngine:
    def __init__(self):
        # 5 independent z-score trackers
        self.z_ofi = RollingZScore(1000)
        self.z_depth = RollingZScore(1000)
        self.z_susc = RollingZScore(1000)
        self.z_vel = RollingZScore(1000)
        self.z_spread = RollingZScore(1000)

        # Previous tick state (for deltas)
        self.prev_bid = None
        self.prev_ask = None
        self.prev_bid_vol = None
        self.prev_ask_vol = None
        self.prev_time_ms = None

        # Current brick context
        self.current_brick_open = 0.0
        self.current_brick_size = 1.0
        self.current_brick_id = 0
        self.prev_brick_open = 0.0
        self.prev_brick_size = 1.0

    def on_new_brick(self, brick):
        """Called when RenkoBuilder emits a new brick. Updates context."""
        self.prev_brick_open = self.current_brick_open
        self.prev_brick_size = self.current_brick_size
        self.current_brick_open = brick.close  # Next brick opens at this close
        self.current_brick_size = brick.brick_size
        self.current_brick_id += 1

    def compute_vector(self, bid, ask, bid_vol, ask_vol, time_ms) -> list:
        """
        Compute the 9D feature vector for a single tick.
        Returns [z_ofi, z_depth, z_susc, z_vel, z_spread, progress, flag_curr, flag_zone, decay]

        IMPORTANT: Call this for EVERY tick, not just brick-close ticks.
        The z-score windows need continuous updates.
        """
        mid = (bid + ask) / 2.0

        # ── Raw Features ─────────────────────────────────
        if self.prev_bid is None:
            # First tick — no deltas possible
            self.prev_bid = bid
            self.prev_ask = ask
            self.prev_bid_vol = bid_vol
            self.prev_ask_vol = ask_vol
            self.prev_time_ms = time_ms
            return [0.0] * 9

        # OFI (weak inequalities)
        dBid = bid - self.prev_bid
        dAsk = ask - self.prev_ask
        ofi_raw = (
            (1 if dBid >= 0 else 0) * bid_vol
          - (1 if dBid <= 0 else 0) * self.prev_bid_vol
          - (1 if dAsk <= 0 else 0) * ask_vol
          + (1 if dAsk >= 0 else 0) * self.prev_ask_vol
        )

        # Depth
        depth_raw = bid_vol + ask_vol

        # Susceptibility (RAW division FIRST, then z-score)
        susc_raw = ofi_raw / (depth_raw + 1e-8)

        # Velocity
        dt_ms = time_ms - self.prev_time_ms
        vel_raw = 1.0 / (dt_ms + 1e-3)

        # Spread
        spread_raw = ask - bid

        # ── Z-Score ──────────────────────────────────────
        z_ofi_val = self.z_ofi.update(ofi_raw)
        z_depth_val = self.z_depth.update(depth_raw)
        z_susc_val = self.z_susc.update(susc_raw)
        z_vel_val = self.z_vel.update(vel_raw)
        z_spread_val = self.z_spread.update(spread_raw)

        # ── Brick-Relative Features ──────────────────────
        progress = (mid - self.current_brick_open) / self.current_brick_size
        flag_curr = 1.0  # During live streaming, current tick is always in current brick
        flag_zone = 1.0 if abs(mid - self.prev_brick_open) >= self.prev_brick_size else 0.0
        decay = 0.0  # Current brick ticks have decay = 0

        # ── Update State ─────────────────────────────────
        self.prev_bid = bid
        self.prev_ask = ask
        self.prev_bid_vol = bid_vol
        self.prev_ask_vol = ask_vol
        self.prev_time_ms = time_ms

        return [z_ofi_val, z_depth_val, z_susc_val, z_vel_val, z_spread_val,
                progress, flag_curr, flag_zone, decay]
```

> [!CAUTION]
> **The feature engine must process EVERY tick, not just brick-close ticks.** The z-score rolling windows need continuous updates to maintain accuracy. If you skip ticks, the z-score statistics will drift from the training distribution.

---

### 4.5 Micro-Buffer & Tensor Assembly (`inference/buffer.py`)

**Responsibility**: Maintain the continuous tick buffer and assemble model inputs on brick close.

```python
import numpy as np
from collections import deque
from math import log

BUFFER_SIZE = 100
MACRO_HISTORY = 10

class InferenceBuffer:
    def __init__(self):
        # Micro-buffer: stores (9D_vector, brick_id) tuples
        # NEVER reset at brick boundaries
        self.micro_buffer = deque(maxlen=BUFFER_SIZE)
        self.current_brick_id = 0

        # Macro-history: stores 3D vectors
        self.macro_history = deque(maxlen=MACRO_HISTORY)

        # Brick size history for z_size calculation
        self.brick_size_history = []

        # Snapshot storage: last 10 brick snapshots
        self.snapshots = deque(maxlen=MACRO_HISTORY)

    def append_tick(self, feature_vector_9d: list, brick_id: int):
        """Append a tick's feature vector to the continuous buffer."""
        self.micro_buffer.append((np.array(feature_vector_9d, dtype=np.float32), brick_id))

    def on_brick_close(self, brick) -> tuple:
        """
        Called when a new brick closes.
        1. Compute macro vector for this brick.
        2. Snapshot the micro buffer.
        3. Rewrite Flag_Curr and Decay.
        4. Return (micro_tensor, macro_tensor) ready for model.

        Returns None if not enough history (< 10 bricks).
        """
        self.current_brick_id += 1

        # ── Macro Vector ─────────────────────────────────
        # Duration: time since previous brick
        prev_ts = self.snapshots[-1][1] if self.snapshots else brick.timestamp
        duration_s = max(0, (brick.timestamp - prev_ts) / 1000.0)
        log_dur = log(duration_s + 1)
        direction = 1.0 if brick.uptrend else -1.0

        self.brick_size_history.append(brick.brick_size)
        recent_sizes = self.brick_size_history[-50:]
        if len(recent_sizes) < 2:
            z_size = 0.0
        else:
            mu = np.mean(recent_sizes)
            sigma = np.std(recent_sizes, ddof=1)
            z_size = (brick.brick_size - mu) / sigma if sigma > 1e-12 else 0.0

        macro_vec = np.array([log_dur, direction, z_size], dtype=np.float32)
        self.macro_history.append(macro_vec)

        # ── Micro Snapshot ───────────────────────────────
        buf_len = len(self.micro_buffer)
        if buf_len == 0:
            snapshot = np.zeros((BUFFER_SIZE, 9), dtype=np.float32)
        else:
            vectors = []
            brick_ids = []
            for vec, bid in self.micro_buffer:
                vectors.append(vec.copy())
                brick_ids.append(bid)

            arr = np.stack(vectors)  # (buf_len, 9)

            # Rewrite Flag_Curr (index 6) and Decay (index 8)
            for i in range(len(arr)):
                arr[i, 6] = 1.0 if brick_ids[i] == self.current_brick_id else 0.0
                arr[i, 8] = min((self.current_brick_id - brick_ids[i]) / BUFFER_SIZE, 1.0)

            # Zero-pad at front if < BUFFER_SIZE ticks
            if buf_len < BUFFER_SIZE:
                pad = np.zeros((BUFFER_SIZE - buf_len, 9), dtype=np.float32)
                snapshot = np.vstack([pad, arr])
            else:
                snapshot = arr

        self.snapshots.append((snapshot, brick.timestamp))

        # ── Assemble Tensors ─────────────────────────────
        if len(self.snapshots) < MACRO_HISTORY:
            return None  # Not enough brick history yet

        # Micro: (10, 100, 9)
        micro_tensor = np.stack([s[0] for s in self.snapshots])  # (10, 100, 9)

        # Macro: (10, 3)
        macro_list = list(self.macro_history)
        if len(macro_list) < MACRO_HISTORY:
            pad_count = MACRO_HISTORY - len(macro_list)
            pad = [np.zeros(3, dtype=np.float32)] * pad_count
            macro_list = pad + macro_list
        macro_tensor = np.stack(macro_list)  # (10, 3)

        # Add batch dimension: (1, 10, 100, 9) and (1, 10, 3)
        return (
            micro_tensor[np.newaxis, ...],
            macro_tensor[np.newaxis, ...]
        )
```

> [!IMPORTANT]
> **The micro-buffer NEVER resets at brick boundaries.** Ticks from the previous brick persist in the buffer and naturally age out as new ticks arrive. This is critical — the model was trained on continuous buffers.

---

### 4.6 Ensemble Inference (`inference/ensemble.py`)

**Responsibility**: Load 3 fold models, run inference, apply majority voting.

```python
import json
import tensorflow as tf
import numpy as np
from pathlib import Path

class EnsemblePredictor:
    def __init__(self, models_dir: str):
        self.models = []
        self.configs = []
        self.models_dir = Path(models_dir)

    def load(self):
        """Load 3 fold models and their configs."""
        for fold in [1, 2, 3]:
            fold_dir = self.models_dir / f"fold_{fold}"
            model = tf.keras.models.load_model(fold_dir / "model.keras")
            with open(fold_dir / "config.json") as f:
                config = json.load(f)
            self.models.append(model)
            self.configs.append(config)
            print(f"Loaded Fold {fold}: Pred_OS threshold = {config['Pred_OS_threshold']}")

    def predict(self, micro_tensor, macro_tensor) -> dict:
        """
        Run all 3 models and apply majority voting.

        Args:
            micro_tensor: (1, 10, 100, 9)
            macro_tensor: (1, 10, 3)

        Returns:
            {
                "action": 1 (ENTER) or 0 (SKIP),
                "votes": int,       # How many models voted ENTER
                "details": [        # Per-model breakdown
                    {"prob_win": float, "pred_os": float, "signal": bool},
                    ...
                ]
            }
        """
        votes = 0
        details = []

        for i, (model, config) in enumerate(zip(self.models, self.configs)):
            preds = model([micro_tensor, macro_tensor], training=False)
            prob_win = float(preds[0].numpy().flatten()[0])
            pred_os = float(preds[1].numpy().flatten()[0])

            th_p = config["Prob_Win_threshold"]  # 0.5 for all folds
            th_o = config["Pred_OS_threshold"]   # 1.6, 1.7, 1.8

            signal = (prob_win >= th_p) and (pred_os >= th_o)
            if signal:
                votes += 1

            details.append({
                "prob_win": prob_win,
                "pred_os": pred_os,
                "signal": signal
            })

        # ── Baiting Logic ─────────────────────────────────
        # Criteria: ALL folds must have prob_win < 0.2 and pred_os < 0.7
        is_bait = all(d["prob_win"] < BAIT_PROB_WIN_THRESHOLD and
                      d["pred_os"] < BAIT_PRED_OS_THRESHOLD for d in details)

        return {
            "action": 1 if votes >= 2 else (-1 if is_bait else 0),  # 1: ENTER, -1: REVERSE, 0: SKIP
            "votes": votes,
            "details": details,
            "trade_type": "standard" if votes >= 2 else ("baiting" if is_bait else "none")
        }
```,StartLine:631,TargetContent:

**Calibrated thresholds (from cross-validation):**
| Fold | Prob_Win | Pred_OS |
|---|---|---|
| 1 | 0.50 | 1.60 |
| 2 | 0.50 | 1.70 |
| 3 | 0.50 | 1.80 |

---

### 4.7 Order Executor (`execution/orders.py`)

**Responsibility**: Place, modify, and close orders via MT5.

```python
import MetaTrader5 as mt5

class OrderExecutor:
    def send_market_order(self, direction: int, sl: float, tp: float) -> int:
        """
        Place a market order.
        direction: 1 (BUY) or -1 (SELL)
        Returns ticket number or None on failure.
        """
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(SYMBOL).ask if direction == 1 \
                else mt5.symbol_info_tick(SYMBOL).bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": float(LOT_SIZE),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": DEVIATION,
            "magic": MAGIC_NUMBER,
            "comment": "BrickOfTicks_v1",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,  # Check broker: IOC or FOK
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order Failed: {result.retcode} - {result.comment}")
            return None

        return result.order
```

**SL / TP Calculation:**
```python
# On brick close, after inference:
result = self.ensemble.predict(micro_tensor, macro_tensor)
action = result["action"]

if action == 1:       # Standard Trade
    entry = brick.close
    dist = brick.brick_size
    if brick.uptrend:
        sl, tp, direction = entry - dist, entry + dist, 1
    else:
        sl, tp, direction = entry + dist, entry - dist, -1
elif action == -1:    # Baiting (REVERSE) Trade
    entry = brick.close
    dist = brick.brick_size
    if brick.uptrend: # Signal is UP but we bet DOWN
        sl, tp, direction = entry + dist, entry - dist, -1
    else:             # Signal is DOWN but we bet UP
        sl, tp, direction = entry - dist, entry + dist, 1
```,StartLine:690,TargetContent:

> [!WARNING]
> **Filling type varies by broker.** Some brokers require `ORDER_FILLING_FOK` instead of `ORDER_FILLING_IOC`. Test with a demo account first.

---

### 4.8 Risk Manager (`execution/risk.py`)

**Responsibility**: Enforce daily drawdown limits.

```python
class RiskManager:
    def __init__(self, state_manager):
        self.state = state_manager

    def check_daily_limit(self) -> bool:
        """Returns False if daily loss limit exceeded."""
        account = mt5.account_info()
        if not account:
            return True  # Fail-open

        # Compare equity to balance (floating + realized)
        equity = account.equity
        balance = account.balance
        drawdown_pct = (balance - equity) / balance if balance > 0 else 0

        if drawdown_pct >= MAX_DAILY_DRAWDOWN_PCT:
            logger.warning(f"DAILY RISK LIMIT HIT: {drawdown_pct:.2%}")
            return False

        return True
```

---

### 4.9 State Manager (`utils/state.py`)

**Responsibility**: Persist bot state to disk for crash recovery.

```python
import json

class StateManager:
    def __init__(self, filepath="logs/state.json"):
        self.filepath = filepath
        self.state = {
            "last_tick_msc": 0,
            "daily_pnl": 0.0,
            "active_ticket": 0,
            "brick_count": 0
        }
        self.load()

    def load(self):
        try:
            with open(self.filepath) as f:
                self.state.update(json.load(f))
        except FileNotFoundError:
            pass

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.state, f, indent=2)

    def update(self, key, value):
        self.state[key] = value
        self.save()

    def get(self, key, default=None):
        return self.state.get(key, default)
```

---

## 5. The Main Loop (`main.py`)

```python
import time
from data.connector import MT5Connector
from data.tick_stream import TickStream
from data.renko import RenkoBuilder
from data.feature_engine import LiveFeatureEngine
from inference.buffer import InferenceBuffer
from inference.ensemble import EnsemblePredictor
from execution.orders import OrderExecutor
from execution.risk import RiskManager
from utils.state import StateManager
from utils.logger import logger
import MetaTrader5 as mt5

class OrbitEngine:
    def __init__(self):
        self.connector = MT5Connector()
        self.state = StateManager()
        self.risk = RiskManager(self.state)
        self.orders = OrderExecutor()

        # These are initialized after MT5 connects
        self.tick_stream = None
        self.renko = None
        self.features = LiveFeatureEngine()
        self.buffer = InferenceBuffer()
        self.ensemble = EnsemblePredictor("models/")

    def start(self):
        if not self.connector.connect():
            logger.error("MT5 connection failed")
            return False

        # Compute brick size from current price
        tick = mt5.symbol_info_tick(SYMBOL)
        brick_size = tick.ask * BRICK_SIZE_FACTOR
        logger.info(f"Brick Size: {brick_size:.4f}")

        # Initialize Renko at current price
        self.renko = RenkoBuilder(brick_size, tick.bid)
        self.features.current_brick_open = tick.bid
        self.features.current_brick_size = brick_size

        # Initialize tick stream
        self.tick_stream = TickStream()

        # Load ML models
        self.ensemble.load()

        # WARMUP: Replay recent history to fill z-score windows
        self._warmup()

        logger.info("Bot started. Waiting for signals...")
        return True

    def _warmup(self):
        """
        Replay recent ticks to fill z-score windows (1000 ticks)
        and build initial Renko state.
        """
        logger.info("Warming up z-score windows...")
        # Fetch last ~2000 ticks to ensure 1000-tick window is filled
        tick = mt5.symbol_info_tick(SYMBOL)
        ticks = mt5.copy_ticks_from(SYMBOL, tick.time_msc / 1000.0 - 300, 5000, mt5.COPY_TICKS_ALL)

        if ticks is not None and len(ticks) > 0:
            for t in ticks:
                # Update feature engine (fills z-score windows)
                self.features.compute_vector(
                    t['bid'], t['ask'], t['bid_vol'], t['ask_vol'], t['time_msc']
                )
                # Update Renko
                new_bricks = self.renko.update_tick(t['bid'], t['time_msc'])
                for brick in new_bricks:
                    self.features.on_new_brick(brick)
                    vec = self.features.compute_vector(
                        t['bid'], t['ask'], t['bid_vol'], t['ask_vol'], t['time_msc']
                    )
                    self.buffer.append_tick(vec, self.buffer.current_brick_id)
                    self.buffer.on_brick_close(brick)

            logger.info(f"Warmup complete: {len(ticks)} ticks, "
                        f"{len(self.renko.history)} bricks")

    def pulse(self):
        """Single heartbeat of the trading loop."""
        # 1. Risk Check
        if not self.risk.check_daily_limit():
            time.sleep(1)
            return True

        # 2. Fetch new ticks
        new_ticks = self.tick_stream.fetch()
        if not new_ticks:
            time.sleep(0.05)  # 50ms idle
            return True

        # 3. Process each tick
        for t in new_ticks:
            bid = t['bid']
            ask = t['ask']
            bid_vol = t['bid_vol']
            ask_vol = t['ask_vol']
            time_msc = t['time_msc']

            # A. Update feature engine (EVERY tick)
            vec = self.features.compute_vector(bid, ask, bid_vol, ask_vol, time_msc)
            self.buffer.append_tick(vec, self.buffer.current_brick_id)

            # B. Update Renko
            new_bricks = self.renko.update_tick(bid, time_msc)

            # C. Process new bricks
            for brick in new_bricks:
                self.features.on_new_brick(brick)
                self.process_signal(brick)

        return True

    def process_signal(self, brick):
        """Called on every new brick close. Runs inference and executes."""
        # 1. Snapshot buffer and assemble tensors
        tensors = self.buffer.on_brick_close(brick)
        if tensors is None:
            logger.info("Skipping inference: warming up buffer history")
            return

        micro_tensor, macro_tensor = tensors

        # 2. Skip if position already open
        active = self.state.get("active_ticket", 0)
        if active != 0:
            # Check if it's still open
            positions = mt5.positions_get(ticket=active)
            if positions:
                logger.info("Signal skipped: position already open")
                return
            else:
                self.state.update("active_ticket", 0)

        # 3. Run ensemble inference
        result = self.ensemble.predict(micro_tensor, macro_tensor)

        logger.info(f"Ensemble: {result['votes']}/3 votes. "
                    f"Action: {'ENTER' if result['action'] else 'SKIP'}")
        for i, d in enumerate(result['details']):
            logger.info(f"  Fold {i+1}: P(Win)={d['prob_win']:.3f}, "
                        f"Pred_OS={d['pred_os']:.3f}, Signal={d['signal']}")

        # 4. Execute if consensus
        if result['action'] == 1:
            entry = brick.close
            dist = brick.brick_size

            if brick.uptrend:
                sl, tp, direction = entry - dist, entry + dist, 1
            else:
                sl, tp, direction = entry + dist, entry - dist, -1

            ticket = self.orders.send_market_order(direction, sl, tp)
            if ticket:
                self.state.update("active_ticket", ticket)
                logger.info(f"TRADE OPENED: Ticket={ticket}, Dir={'BUY' if direction==1 else 'SELL'}, "
                            f"Entry={entry:.2f}, SL={sl:.2f}, TP={tp:.2f}")

    def run(self):
        if not self.start():
            return
        try:
            while True:
                if not self.pulse():
                    break
        except KeyboardInterrupt:
            logger.info("Shutdown requested.")
            self.connector.shutdown()
```

---

## 6. Startup Warmup Protocol

### 6.1 Why Warmup is Necessary
The z-score rolling windows need at least 1000 ticks before they produce meaningful values. The micro-buffer needs 100+ ticks. The macro-history needs 10 bricks. Without warmup, the first ~10 bricks would have garbage feature values.

### 6.2 Warmup Steps
1. Fetch the last 5,000–10,000 ticks from MT5 using `copy_ticks_from`.
2. Replay them through the FeatureEngine (fills z-score deques).
3. Replay them through the RenkoBuilder (establishes current Renko state).
4. For each brick that forms during replay, run `buffer.on_brick_close()` to fill the snapshot history.
5. After warmup, the system is in a steady state and inference is valid.

### 6.3 Daily Reset
At midnight UTC (or broker rollover):
1. Reset daily PnL counter.
2. Re-optimize brick size from last 7 days of M1 data (optional).
3. Re-initialize Renko if brick size changed.
4. Z-score windows persist — they don't need to reset.

---

## 7. Risk Management

### 7.1 Pre-Trade Guards
| Guard | Rule | Action |
|---|---|---|
| Daily Drawdown | Equity drawdown ≥ 3% of balance | Halt all new trades |
| Max Concurrent | 1 position at a time | Skip signal if active trade |
| Slippage | Ask/Bid ≠ expected entry by > 8% of brick_size | Use limit order instead |
| Duplicate TP | New TP ≈ existing open trade's TP | Skip signal |

### 7.2 Post-Trade Management
- **Break-Even**: When price moves 0.3125 × brick_size in favorable direction, move SL to entry price.
- **Exit**: Let TP or SL handle exit. No trailing stop in v1 (the model's edge is in entry selection, not exit optimization).

---

## 8. Model Files to Deploy

Copy these files from the training outputs to the bot's `models/` directory:

```
From: outputs/exec/cv/fold_1/model.keras  →  models/fold_1/model.keras
From: outputs/exec/cv/fold_1/config.json  →  models/fold_1/config.json
From: outputs/exec/cv/fold_2/model.keras  →  models/fold_2/model.keras
From: outputs/exec/cv/fold_2/config.json  →  models/fold_2/config.json
From: outputs/exec/cv/fold_3/model.keras  →  models/fold_3/model.keras
From: outputs/exec/cv/fold_3/config.json  →  models/fold_3/config.json
```

---

## 9. Requirements

```
MetaTrader5>=5.0.45
numpy>=1.24.0
tensorflow>=2.15.0
pandas>=2.0.0
```

> [!NOTE]
> The MetaTrader5 Python package only works on **Windows**. If developing on macOS/Linux, the bot must be deployed to a Windows machine (or Windows VPS) with MT5 installed.

---

## 10. Testing Checklist

### 10.1 Unit Tests
- [ ] `RollingZScore`: Verify output matches training's `feature_engine.py` on known tick sequences.
- [ ] `RenkoBuilder`: Feed a known price sequence and verify identical bricks to the training CSV.
- [ ] `LiveFeatureEngine`: Compare output against `feature_engine.py` on a shared tick stream.
- [ ] `InferenceBuffer`: Verify snapshot shape is `(100, 9)` with correct zero-padding.
- [ ] `EnsemblePredictor`: Load models and verify predictions (Standard AND Baiting trigger).

### 10.2 Integration Tests
- [ ] Connect to MT5 demo account.
- [ ] Stream 1000 ticks and verify RollingZScore is producing non-zero values.
- [ ] Wait for a brick close and verify tensor assembly shape.
- [ ] Run inference on a demo tick and verify action output.

### 10.3 Paper Trading
- [ ] Run for 1 full trading day on a demo account.
- [ ] Verify all trades have correct SL/TP distances.
- [ ] Verify daily reset occurs correctly.
- [ ] Verify crash recovery (kill process, restart, check state.json).

### 10.4 Live Validation
- [ ] Start with minimum lot size (0.01).
- [ ] Run for 1 week, compare win rate to backtest expectations.
- [ ] Monitor for slippage, rejected orders, or connection drops.
- [ ] Scale lot size only after confirming stable performance.

---

## 11. Common Pitfalls

| Pitfall | Description | Solution |
|---|---|---|
| Z-score drift | Skipping ticks breaks the rolling window | Process EVERY tick, even during catch-up |
| Buffer reset | Clearing micro-buffer between bricks | NEVER reset. Use continuous deque. |
| Wrong z-score formula | Dividing two z-scores for Susceptibility | Divide RAW OFI/Depth first, THEN z-score |
| Renko price source | Using ask or mid for Renko construction | Use BID price consistently |
| Filling type | Broker rejects IOC orders | Try FOK or RETURN filling modes |
| Model loading | TensorFlow version mismatch | Match TF version to training environment |
| Warmup gap | Bot starts during market close | Handle empty tick fetch gracefully |

---

## 12. Future Enhancements (v2.0)

1. **Dynamic brick sizing**: Recompute brick_size every N bricks based on recent ATR.
2. **Model retraining**: Monthly retrain with latest data, hot-swap models.
3. **Multi-symbol**: Extend to EURUSD, GBPUSD with symbol-specific models.
4. **Position sizing**: Kelly criterion based on ensemble confidence (vote count + Pred_OS magnitude).
5. **Trailing stop**: Post-entry management using Head B's predicted overshoot distance.
6. **Hardware acceleration**: ONNX runtime instead of TensorFlow for faster inference.
