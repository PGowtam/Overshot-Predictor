# BrickOfTicks MT5 Execution Engine — Task Breakdown

> Track progress by marking: `[ ]` todo, `[/]` in progress, `[x]` done.
> **Rule**: No phase begins until ALL verification tasks of the previous phase are marked `[x]`.

---

## Phase 0: Project Scaffolding & Model Deployment

### 0.1 Directory & Environment Setup
- [ ] Create `BrickOfTicks_Trader/` root directory with all subdirectories (`config/`, `data/`, `inference/`, `execution/`, `utils/`, `tests/`, `models/`, `logs/`)
- [ ] Create `__init__.py` in every Python package directory
- [ ] Create `requirements.txt` (MetaTrader5, numpy, tensorflow, pandas)
- [ ] Install dependencies via `pip install -r requirements.txt` on Windows target machine
- [ ] Verify `import MetaTrader5` succeeds
- [ ] Verify `import tensorflow` succeeds with version ≥ 2.15

### 0.2 Model Deployment
- [ ] Copy `outputs/exec/cv/fold_1/model.keras` → `models/fold_1/model.keras`
- [ ] Copy `outputs/exec/cv/fold_1/config.json` → `models/fold_1/config.json`
- [ ] Copy `outputs/exec/cv/fold_2/model.keras` → `models/fold_2/model.keras`
- [ ] Copy `outputs/exec/cv/fold_2/config.json` → `models/fold_2/config.json`
- [ ] Copy `outputs/exec/cv/fold_3/model.keras` → `models/fold_3/model.keras`
- [ ] Copy `outputs/exec/cv/fold_3/config.json` → `models/fold_3/config.json`
- [ ] Verify all 3 models load: `tf.keras.models.load_model("models/fold_N/model.keras")`
- [ ] Verify all 3 configs parse: confirm `Pred_OS_threshold` values are 1.60, 1.70, 1.80

### 0.3 Utility Modules
- [ ] Create `config/settings.py` with all constants (SYMBOL, LOT_SIZE, MAGIC_NUMBER, BRICK_SIZE_FACTOR, Z_SCORE_WINDOW, etc.)
- [ ] Create `config/definitions.py` with path constants (ROOT_DIR, MODELS_DIR, LOGS_DIR)
- [ ] Create `utils/logger.py` with RotatingFileHandler (10MB, 5 backups) + console handler
- [ ] Verify logger writes to `logs/trader.log` and console simultaneously
- [ ] Create `utils/state.py` with JSON read/write/update/get
- [ ] Verify StateManager: write state → kill process → re-read → all fields recovered

### 0.4 Test Data Export
- [ ] From training pipeline, export 500 consecutive ticks to `tests/saved_ticks/tick_sample_500.parquet`
- [ ] From training pipeline, export the 9D vectors produced for those 500 ticks to `tests/saved_ticks/vectors_reference_500.npy`
- [ ] From training pipeline, export a known micro/macro tensor pair (ENTER signal) to `tests/saved_tensors/micro_enter.npy`, `tests/saved_tensors/macro_enter.npy`
- [ ] From training pipeline, export a known micro/macro tensor pair (SKIP signal) to `tests/saved_tensors/micro_skip.npy`, `tests/saved_tensors/macro_skip.npy`
- [ ] Record the expected per-fold prob_win and pred_os for both tensor pairs

---

## Phase 1: RollingZScore Engine

### 1.1 Implementation
- [ ] Create `RollingZScore` class in `data/feature_engine.py`
- [ ] Window = 1000, warmup threshold = 30
- [ ] O(1) incremental Welford formula for full-window updates
- [ ] Recompute-from-scratch for filling phase (N < 1000, N ≥ 30)
- [ ] Float guard: clamp M2 to 0 if negative
- [ ] Sigma guard: return 0.0 if σ < 1e-12

### 1.2 Verification
- [ ] **Constant input**: Feed 2000 identical values → all outputs = 0.0
- [ ] **Simple sequence**: Feed [1, 2, ..., 100], manually verify last 10 z-scores match expected values
- [ ] **Warmup**: 29 values → 0.0, 30th → non-zero
- [ ] **Training parity**: Export 2000 raw OFI values from training pipeline, compare z-scored outputs (max error < 1e-6)
- [ ] **Numerical stability**: Alternating 1e6 / 1e-6 values → no NaN, no Inf

---

## Phase 2: Renko Builder

### 2.1 Implementation
- [ ] Create `BrickEvent` namedtuple (open, close, high, low, uptrend, timestamp, brick_size, sequence)
- [ ] Create `RenkoBuilder` class in `data/renko.py`
- [ ] UP detection: continuation (1×) vs reversal (2×)
- [ ] DOWN detection: mirror logic
- [ ] `while` loop for gap fills
- [ ] Ghost/pivot brick for reversals
- [ ] Sequence tracking (binary string, maxlen 100)

### 2.2 Verification
- [ ] **Manual bricks**: brick_size=10, start=100, feed [105, 110, 115, 95] → verify exact brick count and prices
- [ ] **Reversal threshold**: Confirm reversal requires 2× (95 from 110 does NOT trigger; 90 does)
- [ ] **Gap fill**: Feed [100, 150] → verify 5 UP bricks
- [ ] **Training CSV comparison**: First 200 bricks from training data → identical open/close/uptrend
- [ ] **Sequence**: 5 UP + 3 DOWN → `sequence[-8:] == "11111000"`

---

## Phase 3: Live Feature Engine

### 3.1 Implementation
- [ ] Create `LiveFeatureEngine` class in `data/feature_engine.py`
- [ ] 5 independent `RollingZScore` instances (OFI, Depth, Susc, Vel, Spread)
- [ ] Implementation: Volume Mitigation Fallback (`sign(diff(mid))` proxy)
- [ ] Previous tick state tracking (bid, ask, bid_vol, ask_vol, time_ms)
- [ ] OFI with weak inequalities (`>=`, `<=`)
- [ ] Susceptibility: raw division first (`ofi / (depth + 1e-8)`), then z-score
- [ ] Velocity: `1 / (dt_ms + 1e-3)`
- [ ] Progress, Flag_Curr, Flag_Zone, Decay computations
- [ ] `on_new_brick(brick)` context update
- [ ] First tick returns `[0.0] * 9`

### 3.2 Verification
- [ ] **Parity test**: Feed 500 saved ticks through `compute_vector`, compare to reference vectors (max error < 1e-6)
- [ ] **OFI formula**: Manually compute for 5 known tick pairs, verify match
- [ ] **Susceptibility safety**: Feed tick with `depth_raw = 0` → no crash, no NaN
- [ ] **Volume Fallback Test**: Inject ticks with `bid_vol = 0`, verify `raw_ofi` equals `sign(price_change)`
- [ ] **Z-Score Sustainability**: Feed 1000 zero-volume ticks, verify `z_depth/z_susc` stay `0.0`
- [ ] **First tick**: Assert returns `[0.0] * 9`
- [ ] **Brick transition**: Verify `on_new_brick` context update
- [ ] **Gate: Phase 3 Verification Passed**

---

## Phase 4: Micro-Buffer & Tensor Assembly

### 4.1 Implementation
- [ ] Create `InferenceBuffer` class in `inference/buffer.py`
- [ ] `micro_buffer = deque(maxlen=100)` storing `(9D_vector, brick_id)` tuples
- [ ] `on_brick_close(brick)`: snapshot → rewrite Flag_Curr, Decay → zero-pad → macro vector
- [ ] Macro vector: `[log(duration_s + 1), direction, z_size]`
- [ ] `z_size = (brick_size − mean_50) / std_50`
- [ ] 10-snapshot deque for micro tensor assembly
- [ ] 10-element macro history deque
- [ ] Return `None` if < 10 bricks in history
- [ ] Add batch dimension: `(1, 10, 100, 9)` and `(1, 10, 3)`

### 4.2 Verification
- [ ] **Shape test**: 15 brick closes → first 9 return None, 10th returns correct shapes
- [ ] **Zero-padding**: 30 ticks only → first 70 rows zeros, last 30 non-zero
- [ ] **Flag_Curr rewrite**: 50 ticks id=0 + 50 id=1 → correct 0/1 assignment
- [ ] **Decay rewrite**: Verify `(current_id − tick_id) / 100`
- [ ] **Continuity**: After brick close, verify old ticks still present in buffer
- [ ] **Macro vector**: Known durations/sizes → verify log_dur, direction, z_size

---

## Phase 5: Ensemble Inference + Baiting Logic

### 5.1 Standard Ensemble Implementation
- [ ] Create `EnsemblePredictor` class in `inference/ensemble.py`
- [ ] `load()`: Load 3 `.keras` models + 3 `config.json` files
- [ ] `predict()`: Run all 3 models with `training=False`
- [ ] Extract `prob_win` (Head A) and `pred_os` (Head B)
- [ ] Per-fold threshold comparison
- [ ] Majority voting: ≥ 2/3 → ENTER

### 5.2 Baiting (Reversal) Logic
- [ ] Add baiting thresholds to `config/settings.py`: `BAIT_PROB_WIN_THRESHOLD = 0.2`, `BAIT_PRED_OS_THRESHOLD = 0.7`
- [ ] In `predict()`, after standard vote check, evaluate baiting condition:
  - If ALL 3 folds have `prob_win < BAIT_PROB_WIN_THRESHOLD` AND `pred_os < BAIT_PRED_OS_THRESHOLD` → return `action = -1` (REVERSE trade)
- [ ] Return value includes `"trade_type": "standard" | "bait" | "skip"` field

### 5.3 Verification
- [ ] **Standard parity**: Load reference ENTER tensor → per-fold prob_win/pred_os match to 4 decimals
- [ ] **Standard SKIP parity**: Load reference SKIP tensor → action = 0
- [ ] **Voting logic**: Mock 3 models → 3/3=ENTER, 2/3=ENTER, 1/3=SKIP, 0/3=SKIP
- [ ] **Baiting trigger**: Mock 3 models with prob_win=0.12, pred_os=0.4 → action = -1 (REVERSE)
- [ ] **Baiting miss**: Mock with prob_win=0.25, pred_os=0.4 → action = 0 (SKIP, above bait threshold)
- [ ] **Threshold boundary**: Fold 2 with pred_os=1.69 → no signal. pred_os=1.70 → signal.

---

## Phase 6: Order Execution & Risk Management

### 6.1 Order Executor
- [ ] Create `OrderExecutor` class in `execution/orders.py`
- [ ] `send_market_order(direction, sl, tp)` → `TRADE_ACTION_DEAL`
- [ ] `send_limit_order(direction, price, sl, tp)` → `TRADE_ACTION_PENDING`
- [ ] `modify_sl(ticket, new_sl)` → `TRADE_ACTION_SLTP`
- [ ] `cancel_order(ticket)` → `TRADE_ACTION_REMOVE`
- [ ] `close_position(ticket)` → opposite market order
- [ ] All methods include proper MAGIC_NUMBER and error handling

### 6.2 Risk Manager
- [ ] Create `RiskManager` class in `execution/risk.py`
- [ ] `check_daily_limit()`: `(balance - equity) / balance ≥ 0.03` → return False
- [ ] Daily reset: clear PnL, reset limit flag

### 6.3 Baiting Order Logic
- [ ] When `action == -1` (bait): reverse the trade direction
  - Brick is UP → place SELL (not BUY)
  - Brick is DOWN → place BUY (not SELL)
  - SL/TP stay symmetric: 1× brick_size each direction from entry
- [ ] Log clearly: "BAIT TRADE: Reversing direction"

### 6.4 Verification (MT5 demo account required)
- [ ] **Market BUY**: Place on demo → `retcode == DONE`, position exists, SL/TP correct
- [ ] **Market SELL**: Same verification
- [ ] **SL modification**: Modify SL to entry → SL updated, TP unchanged
- [ ] **Close position**: Close via opposite order → position removed
- [ ] **Risk limit**: Set daily_pnl = -3.1 → `check_daily_limit()` returns False
- [ ] **Filling type**: Test IOC, FOK, RETURN → log which the broker accepts
- [ ] **Bait order**: Simulate brick.uptrend=True + action=-1 → SELL order placed (reversed)

---

## Phase 7: Warmup Protocol

### 7.1 Implementation
- [ ] Add `_warmup()` method to `OrbitEngine`
- [ ] Fetch 5,000–10,000 recent ticks from MT5
- [ ] Replay through FeatureEngine (fills z-score windows)
- [ ] Replay through RenkoBuilder (establishes Renko state)
- [ ] For each brick formed: call `buffer.on_brick_close()` (fills snapshot history)
- [ ] Do NOT run inference or place orders during warmup
- [ ] Log: ticks processed, bricks formed, z-score deque lengths, snapshot count

### 7.2 Verification
- [ ] **Z-score fill**: After warmup, all 5 deques have length ≥ 1000
- [ ] **Renko state**: `renko.current_price` is near current market price
- [ ] **Buffer state**: `len(buffer.snapshots) >= 10`
- [ ] **No-trade**: Assert zero orders placed during warmup
- [ ] **Timing**: Warmup completes in < 30 seconds

---

## Phase 8: Main Loop Integration

### 8.1 Implementation
- [ ] Create `OrbitEngine` class in `main.py`
- [ ] Wire: Connector → TickStream → Renko → FeatureEngine → Buffer → Ensemble → Orders
- [ ] `start()`: connect → compute brick_size → init components → load models → warmup
- [ ] `pulse()`: risk check → fetch ticks → process each tick → handle new bricks
- [ ] `process_signal(brick)`: tensor assembly → position check → ensemble inference → execution
- [ ] Handle standard ENTER (action=1) and bait REVERSE (action=-1) actions
- [ ] For bait trades: reverse direction before calling `send_market_order`
- [ ] Break-even logic: monitor price for 0.3125 × brick_size favorable move → move SL to entry
- [ ] Daily reset detection (new UTC day)
- [ ] Graceful shutdown on KeyboardInterrupt

### 8.2 Verification (MT5 demo account, market hours)
- [ ] **Dry run 2h**: Ticks streaming, bricks forming, z-scores non-zero, inference running
- [ ] **Standard signal**: ENTER trade → order placed, SL/TP correct, state updated
- [ ] **Bait signal**: REVERSE trade → direction reversed, order placed correctly
- [ ] **SKIP signal**: Verify no trade placed when < 2 votes and not bait
- [ ] **Position conflict**: Active trade → new signals skipped
- [ ] **Log quality**: Every brick close logged with per-fold prob_win, pred_os, vote count, trade type

---

## Phase 9: Paper Trading Validation

### 9.1 Execution
- [ ] Run bot unattended on demo account for 5 consecutive trading days
- [ ] Collect trade log (every signal and action)

### 9.2 Verification
- [ ] **Uptime**: Zero crashes, zero unhandled exceptions for 5 days
- [ ] **Standard trade quality**: All SL/TP distances = 1× brick_size
- [ ] **Bait trade quality**: All bait trades have reversed directions and correct SL/TP
- [ ] **Win rate (standard)**: Within 15% of backtest expectation (> 77% of 91%)
- [ ] **Win rate (bait)**: Within 15% of backtest expectation (> 75% of 88.75%)
- [ ] **Risk enforcement**: Manually trigger losing streak → daily limit halts trading
- [ ] **Crash recovery**: Kill process → restart → active trade managed, no duplicates
- [ ] **Daily reset**: Cross midnight UTC → PnL resets, Renko valid, z-scores persist
- [ ] **Log analysis**: Confirm no NaN/Inf in feature values, no shape mismatches

---

## Phase 10: Live Deployment

### 10.1 Deployment
- [ ] Deploy to Windows VPS with MT5 terminal
- [ ] Configure `LOT_SIZE = 0.01` (minimum risk)
- [ ] Set up process manager (e.g., Windows Task Scheduler) for auto-restart on crash

### 10.2 Week 1 (Minimum Risk)
- [ ] Run for 1 full week at 0.01 lots
- [ ] Compare live WR (standard) to paper WR
- [ ] Compare live WR (bait) to paper WR
- [ ] Verify slippage < 1 point average
- [ ] Verify zero rejected orders
- [ ] Verify < 3 reconnection events

### 10.3 Scale-Up
- [ ] If Week 1 stable: increase to 0.02 lots, run 1 week
- [ ] If Week 2 stable: increase to 0.05 lots, run 1 week
- [ ] Never increase by more than 2× per week
- [ ] If WR drops > 15% below expectation at any point: halt, investigate, fix
