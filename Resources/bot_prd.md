# BrickOfTicks MT5 Execution Engine — Product Requirements Document

> **Scope**: Production execution engine only. Consumes pre-trained model artifacts from the offline training pipeline. This document defines *what* must be built; see `bot_implementation.md` for *how*.

---

## 1. Objective

Build a production-ready Python trading bot that:
1. Connects to MetaTrader 5 and streams real-time L1 ticks for XAUUSD
2. Constructs ATR-based Renko bricks from the tick stream (identical algorithm to training)
3. Computes the 9-dimensional microstructure feature vector for every tick using rolling z-scores
4. On each brick close, assembles the hierarchical tensor inputs: Micro `(1, 10, 100, 9)` + Macro `(1, 10, 3)`
5. Runs inference through 3 pre-trained CNN+LSTM models (ensemble)
6. Applies majority voting for standard signals (ENTER if ≥ 2 of 3 models signal)
7. Applies "Baiting" logic for reversal signals (REVERSE if all models predict high-confidence loss)
8. Places orders with SL/TP = 1× brick_size and manages open positions
9. Enforces daily risk limits and persists state for crash recovery

**Success means**: A bot that reproduces the training pipeline's feature computation *exactly*, applies the ensemble/baiting inference correctly, and achieves a live win rate within 5% of optimized holdout expectations (**89.1% Standard**, **87.6% Baiting**).

---

## 2. Inputs (Pre-Trained Artifacts)

| Artifact | Source Path | Description |
|---|---|---|
| `fold_1/model.keras` | `outputs/exec/cv/fold_1/model.keras` | Fold 1 CNN+LSTM weights |
| `fold_1/config.json` | `outputs/exec/cv/fold_1/config.json` | `Pred_OS_threshold: 1.60` |
| `fold_2/model.keras` | `outputs/exec/cv/fold_2/model.keras` | Fold 2 CNN+LSTM weights |
| `fold_2/config.json` | `outputs/exec/cv/fold_2/config.json` | `Pred_OS_threshold: 1.70` |
| `fold_3/model.keras` | `outputs/exec/cv/fold_3/model.keras` | Fold 3 CNN+LSTM weights |
| `fold_3/config.json` | `outputs/exec/cv/fold_3/config.json` | `Pred_OS_threshold: 1.80` |

All models share `Prob_Win_threshold: 0.50`, `z_score_window: 1000`, `micro_buffer_size: 100`, `macro_history_size: 10`.

---

## 3. Output Artifacts

| Artifact | Description |
|---|---|
| Trade log (CSV/JSON) | Every signal: timestamp, ensemble votes, per-fold prob_win/pred_os, action taken |
| State file (`state.json`) | Crash-recoverable bot state: last tick msc, active ticket, daily PnL |
| Rotating log files | Structured log with timestamp, level, module, message |
| Performance report | Daily/weekly WR, drawdown, trade count (generated offline from trade log) |

---

## 4. Functional Requirements

### 4.1 MT5 Connection (FR-CONN)

**FR-CONN-01**: Initialize MT5 using the `MetaTrader5` Python package. Call `mt5.initialize()` once at startup and `mt5.symbol_select(SYMBOL, True)` to enable the symbol. Fail loudly if connection fails.

**FR-CONN-02**: Check connection health every pulse cycle. If `mt5.terminal_info()` returns `None`, attempt reconnection. Log all reconnection events with timestamps.

**FR-CONN-03**: On graceful shutdown (SIGINT / KeyboardInterrupt), call `mt5.shutdown()`.

### 4.2 Gap-Less Tick Stream (FR-TICK)

**FR-TICK-01**: Fetch ticks using `mt5.copy_ticks_from(SYMBOL, last_time_sec, 1000, COPY_TICKS_ALL)`. Convert `last_time_msc` to seconds for the API call (divide by 1000). Filter returned ticks strictly `> last_time_msc` to avoid duplicates.

**FR-TICK-02**: Every tick must be processed — no skipping, no sampling, no batching. The z-score rolling windows require continuous tick-by-tick updates to maintain accuracy. Even during catch-up after a disconnect, every historical tick must flow through the feature engine.

**FR-TICK-03**: When no new ticks are available, sleep for 50ms to avoid CPU spin. When ticks are available, process them with zero artificial delay.

### 4.3 Renko Construction (FR-RENKO)

**FR-RENKO-01**: Build Renko bricks using the **bid price** from each tick. This matches the training pipeline's convention.

**FR-RENKO-02**: Brick size is computed at session start: `brick_size = current_ask × BRICK_SIZE_FACTOR` where `BRICK_SIZE_FACTOR = 0.00118`. For XAUUSD at $2100, this yields ≈ $2.50.

**FR-RENKO-03**: Reversals require **2× brick_size** movement. The `while` loop must handle gap fills (multiple bricks from a single tick if price jumps).

**FR-RENKO-04**: Each brick emits a `BrickEvent` namedtuple with fields: `open`, `close`, `high`, `low`, `uptrend` (bool), `timestamp` (ms), `brick_size`, `sequence` (binary string of last 100 directions).

**FR-RENKO-05**: Maintain a history list of all bricks emitted during the session (needed for sequence tracking).

### 4.4 Feature Engineering (FR-FEAT)

**FR-FEAT-01**: For every incoming tick, compute a 9D feature vector identical to the training pipeline:

| Index | Feature | Formula |
|---|---|---|
| 0 | `z_OFI` | Weak-inequality OFI → rolling 1000-tick z-score |
| 1 | `z_Depth` | (bid_vol + ask_vol) → rolling z-score |
| 2 | `z_Susc` | raw_OFI / (raw_Depth + 1e-8) → rolling z-score |
| 3 | `z_Vel` | 1 / (Δt_ms + 1e-3) → rolling z-score |
| 4 | `z_Spread` | (ask − bid) → rolling z-score |
| 5 | `Progress` | (mid − brick_open) / brick_size |
| 6 | `Flag_Curr` | 1 (always, for live ticks) |
| 7 | `Flag_Zone` | 1 if |mid − prev_brick_open| ≥ prev_brick_size |
| 8 | `Decay` | 0 (current brick ticks) |

**FR-FEAT-02**: OFI Calculation & Volume Fallback (FR-FE-08):
- **Normal Mode**: Compute standard volume-weighted OFI if `bid_vol > 0` AND `ask_vol > 0`.
- **Mitigation Mode**: If any L1 volumes are missing (0.0), default to the **Tick Direction Proxy**:
  - `z_OFI = sign(mid_k - mid_{k-1})`
  - `z_Depth = 0.0`
  - `z_Susc = 0.0`
- **Justification**: Ablation testing (`ablation_report.json`) confirms that this price-proxy approach maintains **88.25% WR**, outperforming the volume-based baseline in the absence of L2 depth.

**FR-FEAT-03**: Susceptibility is computed as `raw_OFI / (raw_Depth + 1e-8)` FIRST, THEN z-scored. Never divide two z-scores.

**FR-FEAT-04**: Z-score uses O(1) incremental Welford formula with sliding window of 1000 ticks. Returns 0.0 when window has < 30 values. Returns 0.0 when σ < 1e-12.

**FR-FEAT-05**: On brick close, update `current_brick_open`, `current_brick_size`, and `current_brick_id` for the next brick's Progress/Flag_Zone calculations.

### 4.5 Micro-Buffer & Tensor Assembly (FR-BUF)

**FR-BUF-01**: Maintain a `deque(maxlen=100)` storing `(9D_vector, brick_id)` tuples. Append every tick's feature vector. **NEVER reset at brick boundaries.**

**FR-BUF-02**: On each brick close, snapshot the buffer → `(100, 9)` array:
- Copy all vectors from the deque.
- Rewrite `Flag_Curr` (index 6): 1 if tick belongs to current brick, 0 otherwise.
- Rewrite `Decay` (index 8): `(current_brick_id − tick_brick_id) / 100`.
- Zero-pad at the front if < 100 ticks in the buffer.

**FR-BUF-03**: Maintain a `deque(maxlen=10)` of brick snapshots. On each brick close, push the new snapshot.

**FR-BUF-04**: Compute the 3D macro-vector for each brick: `[log(duration_s + 1), direction(±1), z_size]`. `z_size = (brick_size − mean_50) / std_50` using the last 50 brick sizes.

**FR-BUF-05**: Assemble model inputs only when ≥ 10 brick snapshots exist:
- Micro tensor: `(1, 10, 100, 9)` from the snapshot deque.
- Macro tensor: `(1, 10, 3)` from the macro-history deque.

### 4.6 Ensemble & Baiting Inference (FR-INFER)

**FR-INFER-01**: Load 3 `.keras` models and their associated `config.json` at startup.

**FR-INFER-02**: On each brick close (after tensor assembly), run all 3 models:
- Extract `prob_win` (Head A, sigmoid) and `pred_os` (Head B, relu) from each model.
- **Standard Signal**: `prob_win >= 0.7` AND `pred_os >= 1.2`.
- **Baiting Signal**: `prob_win < 0.2` AND `pred_os < 0.7`.

**FR-INFER-03**: Apply majority voting for standard trades: enter trade if ≥ 2 of 3 models signal a win.

**FR-INFER-04**: Apply "Baiting" logic for reversal trades: if ALL 3 models meet the baiting criteria (`prob_win < 0.2` and `pred_os < 0.7`), trigger a REVERSE trade.

**FR-INFER-05**: Log the full detail of every inference: per-fold prob_win, pred_os, signal type (Standard/Baiting/None), vote count, final action.

### 4.7 Order Execution (FR-ORDER)

**FR-ORDER-01**: Determine trade direction:
- **Standard BUY**: Brick UP + Standard Signal.
- **Standard SELL**: Brick DOWN + Standard Signal.
- **Baiting BUY**: Brick DOWN + Baiting Signal (Reverse).
- **Baiting SELL**: Brick UP + Baiting Signal (Reverse).

**FR-ORDER-02**: SL/TP placement:
- For any trade, `dist = brick.brick_size`.
- `SL = entry - dist` (BUY) or `entry + dist` (SELL).
- `TP = entry + dist` (BUY) or `entry - dist` (SELL).

**FR-ORDER-03**: Use `TRADE_ACTION_DEAL` (market order) as the primary execution mode. Include `MAGIC_NUMBER` for identification.

**FR-ORDER-03**: If the current market price deviates from `brick.close` by more than 8% of brick_size, use a limit order at `brick.close` instead of a market order. If price has already passed the break-even level, skip the trade entirely.

**FR-ORDER-04**: Maximum 1 concurrent position. If an active ticket exists, skip new signals. Check position status via `mt5.positions_get(ticket=active_ticket)`.

**FR-ORDER-05**: After a successful fill, save `active_ticket`, `entry_price`, `direction`, `sl`, `tp` to the state file.

### 4.8 Risk Management (FR-RISK)

**FR-RISK-01**: Check daily drawdown before processing any signal. Drawdown = `(balance − equity) / balance`. If ≥ 3%, halt all new trades for the remainder of the day.

**FR-RISK-02**: On daily reset (UTC midnight or broker rollover): reset daily PnL counter, log the reset event.

**FR-RISK-03**: Break-even trigger: when price moves 0.3125 × brick_size in the favorable direction from entry, move SL to entry price using `TRADE_ACTION_SLTP`.

### 4.9 State Persistence (FR-STATE)

**FR-STATE-01**: Save bot state to `logs/state.json` after every state-changing event (trade opened, trade closed, daily reset).

**FR-STATE-02**: On startup, load state from disk. If `active_ticket` is set, verify the position still exists in MT5.

**FR-STATE-03**: Minimum state fields: `last_tick_msc`, `active_ticket`, `active_entry_price`, `active_direction`, `daily_pnl`, `brick_count`, `current_day`.

### 4.10 Warmup Protocol (FR-WARM)

**FR-WARM-01**: On startup, fetch the last 5,000–10,000 ticks from MT5 using `copy_ticks_from`. Replay them sequentially through the FeatureEngine (fills z-score windows), RenkoBuilder (establishes Renko state), and InferenceBuffer (fills snapshot history).

**FR-WARM-02**: During warmup, do NOT run inference or place trades. Only build internal state.

**FR-WARM-03**: Log the warmup result: number of ticks replayed, number of bricks formed, final Renko price.

---

## 5. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NF-01 | Feature parity: live feature values must be identical to training pipeline given identical tick input |
| NF-02 | Volume Robustness: the bot must switch to 'Mitigation Mode' automatically if broker lack tick volume |
| NF-03 | Warmup latency < 30 seconds for 10,000 ticks |
| NF-04 | Inference latency < 500ms per brick close (3 model forward passes) |
| NF-05 | Zero unhandled exceptions — all MT5 API calls wrapped in try/except with logging |
| NF-06 | State persistence guarantees no more than 1 missed trade after crash/restart |
| NF-07 | Log rotation: max 10MB per file, 5 backup files |
| NF-08 | Bot must run unattended on a Windows VPS for 24/5 (Mon–Fri market hours) |

---

## 6. Design Constraints

| ID | Constraint |
|---|---|
| DC-01 | MetaTrader5 Python package runs only on Windows |
| DC-02 | Feature engine must use IDENTICAL formulas to `src/feature_engine.py` from training pipeline |
| DC-03 | Renko construction must use bid price (matching training convention) |
| DC-04 | Z-score window = 1000, warmup = 30 ticks (matching training config) |
| DC-05 | Micro-buffer = deque(maxlen=100), NEVER reset at brick boundaries |
| DC-06 | Susceptibility = divide RAW, then z-score. NEVER divide two z-scores. |
| DC-07 | Model files are `.keras` format requiring TensorFlow ≥ 2.15 |
| DC-08 | All order operations use a unique MAGIC_NUMBER to avoid conflicts with manual trades |

---

## 7. Verification Requirements

Each component must be tested in isolation before integration. Testing uses saved tick data from the training pipeline for reproducibility.

### 7.1 Component Verification

| Component | Test | Pass Criteria |
|---|---|---|
| `RollingZScore` | Feed 2000 known tick values, compare output to training's `feature_engine.py` | Max absolute error < 1e-6 |
| `RenkoBuilder` | Feed the first 1000 bricks' worth of bid prices from training CSV | Brick count, opens, closes all match |
| `LiveFeatureEngine` | Process 500 ticks from a saved parquet, compare 9D outputs | Max absolute error < 1e-6 per feature |
| `InferenceBuffer` | Load a known snapshot, verify shape and zero-padding | Shape = `(100, 9)`, padding at front |
| `EnsemblePredictor` | Run all 3 models on a saved tensor pair, compare to `cv_evaluate.py` output | Predictions match to 4 decimal places |
| `OrderExecutor` | Send a market order on demo, verify ticket returned | `retcode == TRADE_RETCODE_DONE` |
| `RiskManager` | Simulate daily drawdown > 3% | Returns `False` correctly |
| `StateManager` | Write state, kill process, read state back | All fields recovered correctly |

### 7.2 Integration Verification

| Test | Pass Criteria |
|---|---|
| Warmup completes with 1000+ ticks | Z-score deques have len ≥ 1000, ≥ 10 bricks formed |
| Full pulse cycle (tick → renko → feature → buffer → inference) | No exceptions, valid ensemble output |
| Paper trade for 24 hours | ≥ 1 trade placed, SL/TP distances correct |
| Crash recovery | Kill bot, restart, verify active trade managed correctly |

---

## 8. Out of Scope (v1.0)

- Multi-symbol support (EURUSD, GBPUSD, etc.)
- Dynamic brick sizing (intra-session recalculation)
- Automated model retraining
- Position sizing (Kelly criterion, confidence-based)
- Trailing stop (post-entry management beyond break-even)
- ONNX inference acceleration
- Web dashboard / monitoring UI
- Trade journaling / analytics beyond CSV logging
