# BrickOfTicks — Local Socket Bridge
# Product Requirements Document (PRD)

> **Status**: v1.0 Final  
> **Target Platform**: macOS (Apple Silicon or Intel) — No Windows, No VPS  
> **Objective**: Live paper-trading of BrickOfTicks using the MQL5↔Python local socket bridge, replacing the `MetaTrader5` Python package (Windows-only) with a platform-native solution.

---

## 1. Background & Context

### 1.1 Why the Socket Bridge?

The `MetaTrader5` Python library (the original BrickOfTicks execution engine) is a Windows-only DLL wrapper. It cannot run natively on macOS. The Socket Bridge approach resolves this by splitting the system into two processes that communicate over a local TCP socket:

| Component | Runs In | Responsibility |
|---|---|---|
| `TickSender.mq5` | MT5 Terminal (Mac native app via Wine) | Streams real-time L1 ticks and receives trade commands |
| `BridgeEngine.py` | Native Python (macOS) | Processes ticks, runs models, generates signals |

MT5 for Mac (available from brokers like ICMarkets, Pepperstone) runs natively on macOS via a Crossover/Wine wrapper. The terminal itself works identically to Windows, including the MQL5 script environment. Only the `MetaTrader5` Python library does not work — which is exactly what the socket replaces.

### 1.2 Forensic Context (Why This Matters Now)

The BrickOfTicks system has been exhaustively forensically validated:

- **Predictive Edge**: 90.3% Win Rate (3-fold ensemble, `Pred_OS >= 1.4`) on 2024 holdout
- **Root Cause of Live Failure**: Spread-to-brick ratio at `K=0.00118` was 14.8%, consuming 30% of TP margin
- **Resolution**: Transition to **`K=0.00295`** (price-proportional multiplier), reducing spread burden to **5.9%** and delivering **+0.747 net expectancy** per trade
- **Model Validity**: Models are statistically valid. Ensemble is feature-parity-verified
- **Platform Blocker**: macOS cannot run the MT5 Python library, preventing any live validation

The Socket Bridge is the **only fully free, no-VPS solution** to enable live/paper trading on macOS.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MacOS Machine (Single Device)                       │
│                                                                         │
│  ┌──────────────────────────────────┐                                   │
│  │   MT5 Terminal (Mac/Wine App)     │                                  │
│  │                                   │                                  │
│  │  ┌───────────────────────────┐   │                                   │
│  │  │  TickSender.mq5 (EA)      │   │                                   │
│  │  │  - OnTick(): Sends L1 tick│   │                                   │
│  │  │  - OnTimer(): Heartbeat   │   │                                   │
│  │  │  - Socket Client (port    │   │                                   │
│  │  │    9000 for ticks)        │   │                                   │
│  │  │  - Socket Server (port    │   │                                   │
│  │  │    9001 for commands)     │   │                                   │
│  │  └──────────────┬────────────┘   │                                   │
│  └─────────────────┼────────────────┘                                   │
│                    │ TCP localhost:9000 (ticks)                         │
│                    │ TCP localhost:9001 (commands)                      │
│                    │                                                     │
│  ┌─────────────────▼────────────────────────────────────────────────┐   │
│  │                 BridgeEngine.py (Native Python)                   │   │
│  │                                                                   │   │
│  │  ┌──────────┐  ┌────────────┐  ┌──────────────┐  ┌───────────┐  │   │
│  │  │TickReader│→ │RenkoBuilder│→ │LiveFeature   │→ │Inference  │  │   │
│  │  │(port 9000│  │K=0.00295   │  │Engine        │  │Buffer     │  │   │
│  │  │listener) │  │            │  │(9D + z-score)│  │(10×100×9) │  │   │
│  │  └──────────┘  └────────────┘  └──────────────┘  └─────┬─────┘  │   │
│  │                                                         │         │   │
│  │  ┌──────────┐  ┌────────────┐  ┌──────────────────────┐│         │   │
│  │  │ Trade Log│  │State.json  │  │EnsemblePredictor     ││         │   │
│  │  │ (CSV)    │  │(Crash Rec.)│  │(3 fold models)       ││         │   │
│  │  └──────────┘  └────────────┘  └──────────┬───────────┘│         │   │
│  │                                            │            │         │   │
│  │  ┌─────────────────────────────────────────▼────────────┘         │   │
│  │  │              CommandSender (port 9001)                          │   │
│  │  │  Sends: BUY/SELL/CLOSE/MODIFYSL commands → MT5 EA              │   │
│  │  └─────────────────────────────────────────────────────           │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. New Architecture Findings (Updated vs. Original PRD)

The original `bot_prd.md` was written for the `K=0.00118` configuration and targeted a Windows VPS. The following critical findings **update and supersede** those specs:

### 3.1 Brick Size: K=0.00295 (MANDATORY UPDATE)

| Parameter | Original | **Updated (MANDATORY)** |
|---|---|---|
| Brick Size Factor | `0.0018` | **`0.00295`** |
| Spread Burden | 14.8% | **5.9%** |
| Net Expectancy | ❌ Negative | **+0.747** |
| Naive WR | ~45% | **50.22%** |

Formula: `brick_size = today_open_price × 0.00295`
At Gold $2400: `brick_size = 2400 × 0.00295 = 7.08 pts`

### 3.2 Tick Buffer: Scaling Requirement

At `K=0.00295`, the average brick duration increases from ~17 minutes to ~133 minutes. The current `deque(maxlen=100)` micro-buffer only captures the **last 1 minute** of price action — functionally blind to the trend-building order flow.

| Parameter | Original | **Updated (For This Release)** |
|---|---|---|
| Micro Buffer | `100 ticks` | **`100 ticks` (Phase 1)** |
| Target Buffer | — | **`5000 ticks` (Phase 2 — post-retraining)** |

**Decision**: Phase 1 uses 100 ticks to validate the bridge architecture. Phase 2 (post-retraining) will expand to 5000 ticks.

### 3.3 Platform: macOS Socket Bridge

| Parameter | Original | **Updated** |
|---|---|---|
| Platform | Windows VPS | **macOS (Local Socket Bridge)** |
| MT5 Integration | Python Library (`import MetaTrader5`) | **MQL5 EA ↔ TCP Socket** |
| Execution | `mt5.order_send()` | **Socket command `BUY/SELL/CLOSE`** |

### 3.4 Ensemble Configuration

The models (`BrickOfTicks_Trader/models/fold_1`, `fold_2`, `fold_3`) were trained on `K=0.00118` data. **They are being used cross-scale and still deliver 90.3% WR** on the `K=0.00295` holdout. They are valid for Phase 1 live testing.

| Threshold | Value |
|---|---|
| `Pred_OS_threshold` | `>= 1.4` (optimal from holdout audit) |
| `Prob_Win_threshold` | `>= 0.5` (standard) |
| Ensemble Vote | `>= 2 of 3` |

---

## 4. MQL5 Component Requirements

### 4.1 TickSender EA (`TickSender.mq5`)

**FR-MQL-01**: On every `OnTick()` event, format a tick message and send it over the TCP socket to Python.

**Tick Message Format** (pipe-delimited, one line):
```
TICK|<time_msc>|<bid>|<ask>|<bid_vol>|<ask_vol>\n
```
Example: `TICK|1714900800123|2400.12|2400.14|3.5|2.1\n`

**FR-MQL-02**: Use `SocketCreate()`, `SocketConnect()` (MQL5 native socket API, available in MT5 Build 2400+) to connect to Python's listening server on `localhost:9000`.

**FR-MQL-03**: On `OnTimer()` every 100ms, send a heartbeat if no ticks have been sent in the last 500ms:
```
HEARTBEAT|<time_msc>\n
```

**FR-MQL-04**: Listen on a second socket (`localhost:9001`) for incoming commands from Python. Parse command messages in `OnTimer()`:
```
BUY|<price>|<sl>|<tp>|<volume>
SELL|<price>|<sl>|<tp>|<volume>
CLOSE|<ticket>
MODIFYSL|<ticket>|<new_sl>
```

**FR-MQL-05**: Execute the received commands using `OrderSend()` with proper error handling. Log all executions to MT5's internal Experts log.

**FR-MQL-06**: On `OnDeinit()`, close all socket connections gracefully.

**FR-MQL-07**: Handle reconnection — if the Python server is not yet running, retry the socket connection every 2 seconds.

---

## 5. Python Component Requirements

### 5.1 Tick Receiver (`bridge/tick_receiver.py`)

**FR-PY-01**: Listen on `localhost:9000` using Python's `socket` module. Accept and parse incoming tick messages from the MQL5 EA.

**FR-PY-02**: Parse the pipe-delimited tick format into a Python dict: `{time_msc, bid, ask, bid_vol, ask_vol}`.

**FR-PY-03**: Push parsed ticks to a thread-safe queue for consumption by the main processing loop.

**FR-PY-04**: Handle `HEARTBEAT` messages — log them but do not process as trading ticks.

**FR-PY-05**: If no tick is received for 10 seconds, log a `WARNING: TICK STREAM STALLED`. If 60 seconds, log `ERROR: POSSIBLE DISCONNECTION`.

### 5.2 Command Sender (`bridge/command_sender.py`)

**FR-PY-06**: Connect to `localhost:9001` and send commands to the MQL5 EA when a trading signal is generated.

**FR-PY-07**: Command format mirrors FR-MQL-04. Add a unique `request_id` field for logging:
```
BUY|<price>|<sl>|<tp>|<volume>|<request_id>
```

**FR-PY-08**: Wait for a confirmation response from the EA before proceeding. Response format:
```
CONFIRM|<request_id>|<ticket>|OK
CONFIRM|<request_id>|0|ERROR|<code>
```

**FR-PY-09**: If no confirmation is received within 5 seconds, log `ERROR: COMMAND TIMEOUT` and do NOT update state (treat as unfilled).

### 5.3 Renko Builder (`bridge/renko.py`)

Identical to the existing `data/renko.py` spec in `bot_implementation.md`, with the following **critical update**:

**FR-PY-RENKO-01**: Brick size uses the **new multiplier**:
```python
brick_size = today_open_price * 0.00295  # Updated from 0.0018
```

**FR-PY-RENKO-02**: `today_open_price` is fetched once at session start from the first tick received that day. It is NOT fetched via the `MetaTrader5` Python library. Instead, the EA sends a `DAYOPEN|<price>` message at startup.

### 5.4 Live Feature Engine (`bridge/feature_engine.py`)

Identical to `src/feature_engine.py` from the training pipeline. No changes.

**Critical Invariants** (must NOT be violated):
- Susceptibility = `raw_ofi / (raw_depth + 1e-8)` FIRST, then z-score
- OFI uses weak inequalities (`>=`, `<=`)
- Z-score window = 1000, warmup threshold = 30
- Processes EVERY tick — no skipping

### 5.5 Inference Buffer (`bridge/buffer.py`)

Identical to the existing `inference/buffer.py` spec, with the following note:

**FR-PY-BUF-01**: Phase 1 uses `deque(maxlen=100)` for compatibility with current models trained on 100-tick buffers. Phase 2 (post-retraining) will expand to `deque(maxlen=5000)`.

### 5.6 Ensemble Predictor (`bridge/ensemble.py`)

Identical to the existing `inference/ensemble.py` spec, with updated thresholds:

**FR-PY-ENS-01**: Use `Pred_OS >= 1.4` as the primary threshold (validated on K=0.00295 holdout: 90.3% WR).

**FR-PY-ENS-02**: Ensemble vote requires `>= 2 of 3` models to signal.

### 5.7 Risk & State Manager

**FR-PY-RISK-01**: Daily drawdown limit: 3% of account balance. State loaded from `bridge/logs/state.json`.

**FR-PY-RISK-02**: Only 1 concurrent open position permitted.

**FR-PY-RISK-03**: Break-even trigger: when price moves `0.3125 × brick_size` favorably, send `MODIFYSL` command to EA.

**FR-PY-RISK-04**: State persisted to disk after every state-changing event.

### 5.8 Paper Trade Logger (`bridge/trade_logger.py`)

**FR-PY-LOG-01**: Log every brick close event with: timestamp, brick direction, ensemble votes, per-fold prob_win/pred_os, action taken.

**FR-PY-LOG-02**: Log every order event with: timestamp, direction, entry price, SL, TP, ticket number (from EA confirmation).

**FR-PY-LOG-03**: At end of session, generate a summary report: trades taken, WR, expectancy, comparison to backtest baseline.

---

## 6. Warmup Protocol

Since we cannot call `mt5.copy_ticks_from()`, the warmup must work differently:

**FR-WARM-01**: The MQL5 EA has a special `HISTORY_REQUEST` command channel. On startup, Python sends: `HISTORY_REQUEST|<count>`. The EA responds by sending the last N ticks as a batch before switching to live mode.

**FR-WARM-02**: Python replays the historical ticks through the feature engine and Renko builder to establish state.

**FR-WARM-03**: Minimum warmup requirement: 5,000 ticks AND at least 10 bricks formed. If this is not met from history, Python enters "Warmup Live Mode" — processing live ticks but suppressing trade execution until the gate is met.

**FR-WARM-04**: Log warmup result: ticks replayed, bricks formed, z-score window fill level.

---

## 7. Communication Protocol Specification

### 7.1 Socket Configuration

| Channel | Direction | Port | Protocol |
|---|---|---|---|
| Tick Stream | MT5 → Python | `localhost:9000` | TCP, persistent connection |
| Command Channel | Python → MT5 | `localhost:9001` | TCP, persistent connection |

### 7.2 Message Encoding

- All messages: **UTF-8**, newline-terminated (`\n`)
- All numbers: **dot-decimal** (not locale-dependent)
- All timestamps: **milliseconds since Unix epoch** (int64)

### 7.3 Full Message Catalog

**MT5 → Python (Port 9000)**
| Message | Format | Notes |
|---|---|---|
| Tick | `TICK\|<time_msc>\|<bid>\|<ask>\|<bid_vol>\|<ask_vol>` | Every OnTick() |
| Heartbeat | `HEARTBEAT\|<time_msc>` | If no tick for 500ms |
| Day Open | `DAYOPEN\|<time_msc>\|<price>` | At session start |
| History Tick | `HTICK\|<time_msc>\|<bid>\|<ask>\|<bid_vol>\|<ask_vol>` | During warmup |
| History Done | `HDONE\|<count>` | End of warmup batch |

**Python → MT5 (Port 9001)**
| Message | Format | Notes |
|---|---|---|
| Buy | `BUY\|<price>\|<sl>\|<tp>\|<volume>\|<req_id>` | Market order |
| Sell | `SELL\|<price>\|<sl>\|<tp>\|<volume>\|<req_id>` | Market order |
| Close | `CLOSE\|<ticket>\|<req_id>` | Close position |
| Modify SL | `MODIFYSL\|<ticket>\|<new_sl>\|<req_id>` | Break-even move |
| History Req | `HISTORY_REQUEST\|<count>` | Sent at warmup |

**MT5 → Python (Port 9001 — Confirmations)**
| Message | Format | Notes |
|---|---|---|
| Confirm OK | `CONFIRM\|<req_id>\|<ticket>\|OK` | Order filled |
| Confirm Error | `CONFIRM\|<req_id>\|0\|ERROR\|<code>` | Order rejected |

---

## 8. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NF-01 | Feature parity: live feature values identical to training pipeline given identical input |
| NF-02 | Socket latency: tick-to-feature-vector processing < 5ms per tick |
| NF-03 | Command latency: signal-to-EA-confirmation < 2 seconds |
| NF-04 | Bridge uptime: handle dropped socket connections with auto-reconnect |
| NF-05 | Zero data loss: ticks buffered during reconnection, replayed when connection restores |
| NF-06 | Crash recovery: state.json allows resume from exact point of crash |
| NF-07 | Log rotation: max 10MB per log file, 5 backups |
| NF-08 | Inference latency: < 500ms for 3 model forward passes (on Mac CPU) |
| NF-09 | No trades during warmup regardless of any signal generated |
| NF-10 | All float values passed over socket with 5 decimal precision |

---

## 9. Design Constraints

| ID | Constraint |
|---|---|
| DC-01 | MQL5 socket API requires MT5 Build ≥ 2400 |
| DC-02 | Brick size: `today_open * 0.00295` (never fixed, never ATR-based) |
| DC-03 | Feature engine must use IDENTICAL formulas to `src/feature_engine.py` |
| DC-04 | Renko uses **bid price** (matching training convention) |
| DC-05 | Z-score window = 1000, warmup = 30 ticks |
| DC-06 | Micro-buffer = `deque(maxlen=100)`, NEVER reset at brick boundaries |
| DC-07 | Susceptibility: raw division first, THEN z-score |
| DC-08 | Models are `.keras` format, require TensorFlow ≥ 2.15 |
| DC-09 | Models loaded from `BrickOfTicks_Trader/models/fold_1,2,3/model.keras` |
| DC-10 | No MetaTrader5 Python package dependency anywhere in the bridge code |

---

## 10. Success Criteria (Paper Trading Gate)

The Local Socket Bridge is considered **successful** when:

1. **Connection Stability**: EA and Python maintain stable socket connection for 8+ consecutive trading hours.
2. **Tick Fidelity**: 100% of OnTick() events received by Python (verified via tick count comparison).
3. **Feature Parity**: Live z-scores are within 1e-4 of offline computation on the same tick data.
4. **Trade Execution**: At least 5 trades placed and confirmed by the EA, with correct SL/TP distances (= brick_size).
5. **Win Rate**: Observed WR after 30+ trades is within 20% of the backtest expectation (> 72%).
6. **No Crashes**: Python process runs for 5 consecutive trading days without an unhandled exception.

---

## 11. Out of Scope (v1.0)

- Multi-symbol support
- ONNX inference acceleration  
- Web dashboard / monitoring UI
- Position sizing (Kelly criterion)
- Live retraining or feature drift detection
- Automated broker reconnection (manual restart acceptable in v1.0)
- The `K=0.00295` model retraining — this PRD covers Phase 1 live testing with cross-scale models
