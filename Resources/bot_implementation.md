# BrickOfTicks MT5 Execution Engine — Implementation Plan

> **Scope**: Production execution engine. Each phase produces testable artifacts. No phase begins until the previous phase's verification tests pass.

---

## Project Structure

```
BrickOfTicks_Trader/
├── main.py                         # Entry point
├── config/
│   ├── __init__.py
│   ├── settings.py                 # Symbol, lot size, risk params, feature constants
│   └── definitions.py              # Path constants (ROOT_DIR, MODELS_DIR, LOGS_DIR)
├── data/
│   ├── __init__.py
│   ├── connector.py                # MT5 connection lifecycle
│   ├── tick_stream.py              # Gap-less tick fetching via copy_ticks_from
│   ├── renko.py                    # Renko brick construction (BrickEvent namedtuple)
│   └── feature_engine.py           # 9D feature vector + RollingZScore (LIVE version)
├── inference/
│   ├── __init__.py
│   ├── buffer.py                   # Micro-buffer (deque) + macro-history + tensor assembly
│   └── ensemble.py                 # Load 3 fold models + majority voting
├── execution/
│   ├── __init__.py
│   ├── orders.py                   # Market/Limit order placement + SL modification
│   └── risk.py                     # Daily drawdown limit enforcement
├── utils/
│   ├── __init__.py
│   ├── logger.py                   # Rotating file + console logger
│   └── state.py                    # JSON state persistence for crash recovery
├── tests/
│   ├── test_z_score.py             # Phase 1 verification
│   ├── test_renko.py               # Phase 2 verification
│   ├── test_feature_engine.py      # Phase 3 verification
│   ├── test_buffer.py              # Phase 4 verification
│   ├── test_ensemble.py            # Phase 5 verification
│   ├── test_orders.py              # Phase 6 verification
│   ├── saved_ticks/                # Exported tick sequences for offline testing
│   │   └── tick_sample_500.parquet # 500 ticks from training data for feature parity checks
│   └── saved_tensors/              # Known-good tensors for inference comparison
│       ├── micro_sample.npy        # (1, 10, 100, 9)
│       └── macro_sample.npy        # (1, 10, 3)
├── models/
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
│   ├── state.json
│   └── trader.log
├── requirements.txt
└── README.md
```

---

## Phase 0: Project Scaffolding & Model Deployment

### Objective
Set up the project directory, install dependencies, copy pre-trained model artifacts, and verify the environment is functional.

### Steps
1. Create the directory structure above.
2. Create `requirements.txt`:
   ```
   MetaTrader5>=5.0.45
   numpy>=1.24.0
   tensorflow>=2.15.0
   pandas>=2.0.0
   ```
3. Install dependencies: `pip install -r requirements.txt`.
4. Copy model files from training outputs:
   ```
   outputs/exec/cv/fold_1/model.keras → models/fold_1/model.keras
   outputs/exec/cv/fold_1/config.json → models/fold_1/config.json
   outputs/exec/cv/fold_2/model.keras → models/fold_2/model.keras
   outputs/exec/cv/fold_2/config.json → models/fold_2/config.json
   outputs/exec/cv/fold_3/model.keras → models/fold_3/model.keras
   outputs/exec/cv/fold_3/config.json → models/fold_3/config.json
   ```
5. Create `config/settings.py` with all constants.
6. Create `config/definitions.py` with path constants.
7. Create `utils/logger.py` (RotatingFileHandler + console).
8. Create `utils/state.py` (JSON read/write with crash safety).

### Verification (Phase 0 Gate)
- [ ] `python -c "import MetaTrader5"` runs without error on Windows target
- [ ] `python -c "import tensorflow; print(tensorflow.__version__)"` confirms ≥ 2.15
- [ ] All 3 model files load: `tf.keras.models.load_model("models/fold_1/model.keras")`
- [ ] All 3 config files parse: `json.load(open("models/fold_1/config.json"))`
- [ ] Logger writes to `logs/trader.log` and to console simultaneously
- [ ] StateManager writes and reads back all fields from `logs/state.json`

---

## Phase 1: RollingZScore Engine

### Objective
Implement the O(1) incremental z-score with sliding window, verified to produce output identical to the training pipeline's `feature_engine.py`.

### File: `data/feature_engine.py` (partial — `RollingZScore` class only)

### Algorithm
```
class RollingZScore:
    window = 1000
    deque(maxlen=window)
    mean = 0.0, M2 = 0.0

    update(x_new):
        if deque is full:
            x_old = deque[0]
            deque.append(x_new)
            mean_new = mean + (x_new - x_old) / N
            M2 = M2 + (x_new - x_old) * ((x_new - mean_new) + (x_old - mean))
            mean = mean_new
            if M2 < 0: M2 = 0  # float guard
            sigma = sqrt(M2 / (N-1))
            return (x_new - mean) / sigma if sigma > 1e-12 else 0.0
        else:
            deque.append(x_new)
            if len < 30: return 0.0
            recompute mean, M2 from scratch
            return z-scored value
```

### Key Implementation Rules
- Window = 1000, warmup = 30 (returns 0.0 below 30 values)
- Float guard: if M2 < 0 due to floating point errors, clamp to 0
- Sigma guard: if σ < 1e-12, return 0.0 (constant values)
- During filling phase (< 1000 values, ≥ 30): recompute from scratch (acceptable for small N)

### Verification (Phase 1 Gate)
**Test: `tests/test_z_score.py`**

1. **Constant input test**: Feed 2000 identical values. Assert all outputs = 0.0.
2. **Simple sequence test**: Feed `[1, 2, 3, ..., 100]`. Manually compute expected z-scores for the last 10 values. Assert match within 1e-6.
3. **Warmup test**: Feed 29 values → all return 0.0. Feed 30th value → returns non-zero.
4. **Training parity test**: Export 2000 raw OFI values from the training pipeline's feature computation for bricks 0–50. Feed them through `RollingZScore.update()` and compare z-scored outputs. Max absolute error < 1e-6.
5. **Numerical stability**: Feed alternating very large (1e6) and very small (1e-6) values. Assert no NaN, no Inf.

---

## Phase 2: Renko Builder

### Objective
Implement the Renko brick construction engine, verified to produce identical bricks to the training dataset.

### File: `data/renko.py`

### Algorithm
```
BrickEvent = namedtuple('BrickEvent', [open, close, high, low, uptrend, timestamp, brick_size, sequence])

class RenkoBuilder(brick_size, start_price):
    current_price = start_price
    uptrend = 0  (neutral)
    history = []
    sequence = ""

    update_tick(price, timestamp_ms) -> list[BrickEvent]:
        # UP: threshold = current + size (continuation) or current + 2*size (reversal)
        # DOWN: mirror
        # While loop handles multi-brick gap fills
        # Reversal adds ghost/pivot brick first
```

### Key Implementation Rules
- Reversals require 2× brick_size movement
- `while` loop for gap fills (single tick → multiple bricks)
- Sequence tracking: binary string of last 100 directions ("1" = UP, "0" = DOWN)
- Feed **bid price** to `update_tick` (not ask, not mid)

### Verification (Phase 2 Gate)
**Test: `tests/test_renko.py`**

1. **Manual brick test**: Create a RenkoBuilder with brick_size=10, start_price=100. Feed price sequence: `[100, 105, 110, 115, 95]`. Assert:
   - First UP brick at price 110 (open=100, close=110)
   - No brick at 105 (not enough movement)
   - Reversal at 95: requires 2×10=20 from 110 → triggers at 90.
   - Verify: 95 does NOT trigger reversal (needs to reach 90).

2. **Gap fill test**: Feed `[100, 150]` with brick_size=10. Assert 5 UP bricks emitted.

3. **Training data test**: Extract the first 200 bricks from the training Renko CSV. Compute the mid/bid price that generated each brick. Feed these prices through RenkoBuilder and compare:
   - Same number of bricks
   - Same open/close prices
   - Same uptrend flags

4. **Sequence tracking**: After 5 UP bricks and 3 DOWN bricks, assert `sequence[-8:] == "11111000"`.

---

## Phase 3: Live Feature Engine

### Objective
Implement the full 9D feature vector computation for live ticks, verified to produce output identical to the training pipeline.

### File: `data/feature_engine.py` (complete — adds `LiveFeatureEngine` class)

### The 9D Vector (per tick)
| Index | Feature | Raw Computation | Z-Score? |
|---|---|---|---|
| 0 | z_OFI | Weak-inequality OFI | Yes (1000) |
| 1 | z_Depth | bid_vol + ask_vol | Yes (1000) |
| 2 | z_Susc | raw_OFI / (raw_Depth + 1e-8) | Yes (1000) — raw division FIRST |
| 3 | z_Vel | 1 / (Δt_ms + 1e-3) | Yes (1000) |
| 4 | z_Spread | ask − bid | Yes (1000) |
| 5 | Progress | (mid − brick_open) / brick_size | No |
| 6 | Flag_Curr | 1 (always, live) | No |
| 7 | Flag_Zone | 1 if \|mid − prev_open\| ≥ prev_size | No |
| 8 | Decay | 0 (current brick) | No |

### Key Implementation Rules
- First tick returns `[0.0] * 9` (no previous tick for deltas)
- `on_new_brick(brick)` updates `current_brick_open`, `current_brick_size`, `prev_brick_open`, `prev_brick_size`, `current_brick_id`
- OFI weak inequalities: `dBid >= 0` and `dBid <= 0` (NOT strict `>` and `<`)
- Susceptibility: divide RAW OFI/Depth, then z-score. NEVER divide two z-scores.
- All 5 z-score trackers are independent `RollingZScore` instances.

### Verification (Phase 3 Gate)
**Test: `tests/test_feature_engine.py`**

1. **Export reference data**: From the training pipeline, export 500 consecutive ticks (bid, ask, bid_vol, ask_vol, timestamp_ms) and the corresponding 9D feature vectors that `feature_engine.py` produced for those ticks. Save as `tests/saved_ticks/tick_sample_500.parquet` and `tests/saved_ticks/vectors_reference_500.npy`.

2. **Parity test**: Feed the 500 ticks through `LiveFeatureEngine.compute_vector()`. Compare output to the reference vectors:
   - z_OFI: max absolute error < 1e-6
   - z_Depth: max absolute error < 1e-6
   - z_Susc: max absolute error < 1e-6
   - z_Vel: max absolute error < 1e-6
   - z_Spread: max absolute error < 1e-6
   - Progress: max absolute error < 1e-4 (depends on accumulated brick context)
   - Flag_Curr: exact match
   - Flag_Zone: exact match
   - Decay: exact match

3. **OFI formula test**: Manually compute OFI for 5 known tick pairs and verify against `compute_vector` output.

4. **Susceptibility safety test**: Feed tick where `depth_raw = 0`. Verify no division by zero (1e-8 guard).

5. **First tick test**: Assert first call returns `[0.0] * 9`.

---

## Phase 4: Micro-Buffer & Tensor Assembly

### Objective
Implement the continuous micro-buffer, macro-history, and tensor assembly, verified to produce correctly shaped model inputs.

### File: `inference/buffer.py`

### Key Implementation Rules
- `micro_buffer = deque(maxlen=100)` storing `(9D_vector, brick_id)` tuples
- **NEVER reset at brick boundaries** — continuous stream
- On brick close:
  1. Snapshot: copy all vectors, rewrite Flag_Curr (idx 6) and Decay (idx 8)
  2. Zero-pad at front if < 100 ticks
  3. Compute macro vector: `[log(duration_s + 1), direction, z_size]`
  4. Stack last 10 snapshots → `(1, 10, 100, 9)`
  5. Stack last 10 macro vectors → `(1, 10, 3)`
  6. Return None if < 10 bricks in history

### Verification (Phase 4 Gate)
**Test: `tests/test_buffer.py`**

1. **Shape test**: Create a buffer, simulate 15 brick closes with synthetic data, verify:
   - First 9 closes return `None`
   - 10th close returns `(micro, macro)` where `micro.shape == (1, 10, 100, 9)` and `macro.shape == (1, 10, 3)`

2. **Zero-padding test**: Create buffer with only 30 ticks, call `on_brick_close`. Verify:
   - First 70 rows of snapshot are all zeros
   - Last 30 rows contain actual feature vectors

3. **Flag_Curr rewrite test**: Append 50 ticks with brick_id=0 and 50 with brick_id=1. On close of brick 1:
   - First 50 should have Flag_Curr = 0
   - Last 50 should have Flag_Curr = 1

4. **Decay rewrite test**: With 100 ticks from brick_ids [0, 0, ..., 1, 1, ...]:
   - Verify Decay = `(current_id - tick_id) / 100`

5. **Continuity test**: After brick close, add more ticks, close another brick. Verify old ticks from previous brick are still in the buffer (shifted but present). This proves the buffer is continuous.

6. **Macro vector test**: Close 3 bricks with known durations and brick_sizes. Verify:
   - `log_dur` matches `log(duration_s + 1)` exactly
   - `direction` is ±1 matching uptrend
   - `z_size` matches `(size − mean) / std` of recent bricks

---

## Phase 5: Ensemble Inference

### Objective
Load the 3 fold models, run inference, and apply majority voting. Verify predictions match the training evaluation script.

### File: `inference/ensemble.py`

### Key Implementation Rules
- Load model via `tf.keras.models.load_model(path)` — requires matching TF version
- Call model with `training=False` to disable dropout
- Extract Head A (prob_win) and Head B (pred_os) from the list of predictions
- Per-fold thresholds: A unified PRED_OS_THRESHOLD = 1.2 and PROB_WIN_THRESHOLD = 0.7 are set after refined analysis.
- Majority vote: ≥ 2 of 3 models must signal for an ENTER action
- **Baiting Logic**: If ALL 3 models show `prob_win < 0.2` and `pred_os < 0.7`, signal a REVERSE action.

### Verification (Phase 5 Gate)
**Test: `tests/test_ensemble.py`**

1. **Export reference tensors**: From the training pipeline, save a known micro/macro tensor pair from the holdout set that resulted in a clear ENTER signal (3/3 vote) and another that resulted in a SKIP (0/3 vote). Save as `.npy` files in `tests/saved_tensors/`.

2. **Prediction parity test**: Load the reference tensors, run through `EnsemblePredictor.predict()`. Assert:
   - Each fold's prob_win matches to 4 decimal places
   - Each fold's pred_os matches to 4 decimal places
   - Final action (ENTER/SKIP) matches

3. **Voting logic test**: Mock 3 models with controlled outputs:
   - All 3 signal → action = 1
   - 2 signal → action = 1
   - 1 signal → action = 0
   - 0 signal → action = 0

4. **Threshold test**: Verify `prob_win = 0.7` and `pred_os = 1.2` → signal.
5. **Baiting test**: Mock 3 models with `prob_win=0.1` and `pred_os=0.5`. Assert `predict()` returns `action = -1` (REVERSE).

---

## Phase 6: Order Execution & Risk Management

### Objective
Implement order placement, position management, and daily risk limits. Test on MT5 demo account.

### Files: `execution/orders.py`, `execution/risk.py`

### Order Types
1. **Market Order**: `TRADE_ACTION_DEAL` with price from `symbol_info_tick`. SL and TP set at order time.
2. **Limit Order** (fallback): `TRADE_ACTION_PENDING` when slippage exceeds 8% of brick_size. Type = `ORDER_TYPE_BUY_LIMIT` or `SELL_LIMIT`.
3. **SL Modification**: `TRADE_ACTION_SLTP` for break-even move.
4. **Cancel Pending**: `TRADE_ACTION_REMOVE` if limit order invalidated.

### Risk Rules
- Daily drawdown ≥ 3% → halt trading (check `(balance - equity) / balance`)
- Max 1 concurrent position
- Slippage guard: market price vs brick.close > 8% of brick_size → limit order
- Break-even: SL → entry when price moves 0.3125 × brick_size favorably
- **Baiting Execution**: If `action == -1`, invert the trade direction (UP brick -> SELL, DOWN brick -> BUY).

### Verification (Phase 6 Gate)
**Test: `tests/test_orders.py` (MT5 demo account required)**

1. **Market order test**: Place a BUY at current price with SL/TP. Verify:
   - `result.retcode == TRADE_RETCODE_DONE`
   - Position appears in `positions_get(ticket=result.order)`
   - SL and TP match the requested values

2. **Sell order test**: Place a SELL. Same verifications.

3. **SL modification test**: Place a BUY, then modify SL to entry price. Verify:
   - SL updated correctly
   - TP unchanged

4. **Close position test**: Place a BUY, then close it using opposite market order. Verify:
   - Position no longer in `positions_get`

5. **Risk limit test**: Set `daily_pnl = -3.1` in state. Assert `check_daily_limit()` returns False.

6. **Filling type test**: Try IOC filling. If rejected, try FOK. Log which filling type the broker accepts. Update `settings.py` accordingly.

---

## Phase 7: Warmup Protocol

### Objective
Implement the startup warmup that replays historical ticks to fill z-score windows and establish Renko state before live trading begins.

### Part of: `main.py` → `OrbitEngine._warmup()`

### Steps
1. Fetch last 5,000–10,000 ticks from MT5 using `copy_ticks_from(SYMBOL, now - 300s, 10000, COPY_TICKS_ALL)`.
2. For each tick:
   - Feed through `LiveFeatureEngine.compute_vector()` → fills z-score windows
   - Feed bid price through `RenkoBuilder.update_tick()` → forms bricks
   - For each brick formed: call `buffer.on_brick_close(brick)` → builds snapshot history
3. Do NOT run inference or place trades during warmup.
4. After warmup, log: ticks processed, bricks formed, z-score deque length, buffer snapshot count.

### Verification (Phase 7 Gate)

1. **Z-score fill test**: After warmup with 5000+ ticks, verify all 5 z-score deques have length ≥ 1000.
2. **Renko state test**: After warmup, verify `renko.current_price` is reasonable (near current market price).
3. **Buffer state test**: After warmup, verify `len(buffer.snapshots) >= 10` (enough for inference).
4. **No-trade test**: Assert zero orders placed during warmup.
5. **Timing test**: Assert warmup completes in < 30 seconds.

---

## Phase 8: Main Loop & Integration

### Objective
Wire all components together into the `OrbitEngine` main loop. Verify end-to-end flow on a demo account.

### File: `main.py`

### Main Loop (`pulse()`)
```
1. Check connection health
2. Check daily risk limit → skip if exceeded
3. Fetch new ticks from TickStream
4. For each tick:
   a. Compute 9D feature vector
   b. Append to micro-buffer
   c. Update Renko
   d. For each new brick:
      - Update feature engine brick context
      - Call buffer.on_brick_close() → tensor
      - If tensor available AND no active position:
        - Run ensemble inference
        - If action == 1: place standard order
        - If action == -1: place REVERSED (bait) order
5. Check for break-even trigger on active position
6. Return True (keep running)
```

### Daily Reset
- Detect new UTC day (compare `datetime.utcnow().date()` to stored date)
- Reset daily PnL
- Optionally re-optimize brick size
- Log the reset

### Verification (Phase 8 Gate)

1. **Dry run test**: Run the full bot on a demo account for 2 hours during market hours. Verify:
   - Ticks are streaming (log shows tick counts)
   - Renko bricks are forming (log shows brick events)
   - Feature engine is producing non-zero z-scores
   - Ensemble inference is running on each brick close
   - Log shows per-fold prob_win, pred_os, and vote count

2. **Signal test**: Wait for a natural signals. Verify:
   - Order placed with correct SL/TP distances (= brick_size from entry)
   - Direction: Matches brick (Standard) or is Reversed (Baiting)
   - Position appears in MT5 terminal
   - State file updated with active_ticket

3. **SKIP test**: Verify that SKIP signals (< 2 votes and not baiting) do NOT place trades.
4. **Position conflict test**: If an active trade exists, verify new ENTER/BAIT signals are skipped with appropriate log message.

---

## Phase 9: Paper Trading Validation

### Objective
Run the bot unattended for 5+ consecutive trading days on a demo account. Collect performance data and compare to backtest expectations.

### Verification (Phase 9 Gate)

1. **Uptime test**: Bot runs continuously for 5 trading days without crashes or unhandled exceptions.

2. **Trade quality test**: Export all trades from the 5-day period. For each trade verify:
   - SL distance = 1× brick_size
   - TP distance = 1× brick_size
   - Entry direction matches brick uptrend (Standard) or is reversed (Baiting)

3. **Win rate check**: Compare observed WR to backtest expectation (91% ensemble holdout WR). Accept if within 15% (i.e., > 77%). If below, investigate:
   - Feature parity drift
   - Warmup insufficiency
   - Timing issues (latency between brick close and order fill)

4. **Risk enforcement test**: Manually trigger a losing streak on a second demo account. Verify daily limit halts trading.

5. **Crash recovery test**: Kill the bot process mid-session. Restart. Verify:
   - Active trade is detected and managed
   - No duplicate trades opened
   - State file integrity

6. **Daily reset test**: Let the bot run across a midnight UTC boundary. Verify:
   - Daily PnL resets
   - Renko state remains valid
   - Z-score windows persist (no reset)

---

## Phase 10: Live Deployment

### Objective
Deploy to a funded live account with minimum risk. Gradually scale up.

### Steps
1. Deploy to a Windows VPS with MT5 terminal and Python environment.
2. Configure `LOT_SIZE = 0.01` (minimum).
3. Run for 1 week, collecting every trade in the log.
4. Compare live WR to paper trading WR.
5. If stable (WR within 10% of paper, no rejected orders, no connection drops):
   - Increase to `LOT_SIZE = 0.02`
   - Monitor for another week
6. Scale incrementally, never increasing lot size by more than 2× per week.

### Verification (Phase 10 Gate)
1. **Execution quality**: Slippage < 1 point average across all trades.
2. **WR stability**: Live WR within 10% of paper WR.
3. **Zero rejected orders**: All `order_send` calls return `TRADE_RETCODE_DONE`.
4. **Connection stability**: < 3 reconnection events per week.
5. **Max drawdown**: Never exceeds 3% in a single day.

---

## Appendix: Critical Invariants

These rules must NEVER be violated. Any violation indicates a bug that must be fixed before proceeding.

| # | Invariant | How to Verify |
|---|---|---|
| 1 | Feature engine processes EVERY tick (no skipping) | Assert tick_count in z-score deque == total ticks received |
| 2 | Micro-buffer NEVER resets at brick boundaries | Assert buffer contains ticks from multiple brick_ids |
| 3 | Susceptibility = raw division first, then z-score | Code review: `susc_raw = ofi / (depth + 1e-8)` before z-score |
| 4 | OFI uses weak inequalities (>=, <=) | Code review: `dBid >= 0` not `dBid > 0` |
| 5 | Z-score window = 1000, warmup = 30 | Assert `self.window == 1000` and `if N < 30: return 0.0` |
| 6 | Renko consumes bid price | Assert `update_tick(t['bid'], ...)` not `t['ask']` or `t['mid']` |
| 7 | Reversals require 2× brick_size | Verify threshold calculation in while loop |
| 8 | Model called with `training=False` | Code review: `model(..., training=False)` |
| 9 | No trades during warmup | Assert order count == 0 after warmup |
| 10 | State persisted after every state change | Assert `state.save()` called in every `state.update()` |
