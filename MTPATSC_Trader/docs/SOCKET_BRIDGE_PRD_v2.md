# BrickOfTicks — Local Socket Bridge
# Product Requirements Document v2.0
# **[UPDATED]** — All sections revised per deployment audit findings

> **Status**: v2.0  
> **Platform**: macOS (No Windows, No VPS)  
> **Approach**: MQL5 EA ↔ Python TCP Socket (localhost only)  
> **Brick Factor**: `K = 0.00295` (standardized — see Section 1.3)

---

## 1. Background & Context

### 1.1 Why the Socket Bridge?

The `MetaTrader5` Python library (the original BrickOfTicks execution plan) is a Windows-only DLL wrapper. It cannot run natively on macOS. The Socket Bridge splits the system into two processes on the same machine communicating over local TCP:

| Component | Runs In | Responsibility |
|---|---|---|
| `TickSender.mq5` | MT5 Terminal (Mac native via Wine) | Stream L1 ticks, execute trade commands |
| `BridgeEngine.py` | Native macOS Python | Process ticks, run models, send signals |

### 1.2 Forensic & Feasibility Context

The BrickOfTicks system has been exhaustively validated:

- **Predictive Edge**: 90.3% Win Rate (3-fold ensemble, `Pred_OS >= 1.4`) on 2024 holdout
- **Root Cause of Original Live Failure**: Spread-to-brick ratio was 14.8% at `K=0.00118`, consuming 30% of TP margin → negative expectancy
- **Resolution**: `K=0.00295` reduces spread burden to **5.9%**, delivering **+0.747 net expectancy per trade**
- **Model Validity**: Models statistically valid. Feature parity verified offline
- **Platform Blocker**: macOS cannot run `MetaTrader5` Python library

### 1.3 **[UPDATED]** Brick Size: K=0.00295 — Canonical Formula

> [!IMPORTANT]
> **DEPRECATION NOTICE**: The original multiplier `K=0.00118` (brick_size = open × 0.00118) is **UNPROFITABLE** and must not be used. At XAUUSD spreads of ~0.84 pts, the spread-to-brick ratio was 14.8%, causing systematic negative expectancy. All documents, code, and configurations must use `K=0.00295`.

**Formula (single source of truth)**:
```
brick_size = today_daily_open_price × 0.00295
```

**Example at Gold $2400**:
- `brick_size = 2400 × 0.00295 = 7.08 pts` (not 2.84 pts)
- Spread burden: 0.84 / 7.08 = **5.9%** (vs 14.8% at K=0.00118)
- Net TP at 1:1 RR: `1.0 − 0.059 = 0.941` bricks
- Net expectancy at 90.3% WR: `0.903 × 0.941 − 0.097 × 1.059 = **+0.747** bricks/trade`

**Why 0.00295?** Determined by a systematic feasibility study (brick_study/) evaluating K multipliers from 0.00118 to 0.00354. K=0.00295 (2.5× original) represents the optimal trade-off:
- Spread burden below 6% (profitable threshold)
- Trade frequency ~93 trades/year (sufficient for statistical confidence)
- Naive continuation WR improved from ~45% to 50.22%

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    macOS Machine (localhost)                     │
│                                                                 │
│  ┌──────────────────────────────┐                               │
│  │  MT5 Terminal (Mac/Wine)      │                              │
│  │  ┌────────────────────────┐  │                               │
│  │  │  TickSender.mq5 (EA)   │  │  TCP :9000 (ticks → Python)  │
│  │  │  OnTick → socket send  │◄─┼──────────────────────────┐   │
│  │  │  OnTimer → cmd listen  │  │  TCP :9001 (cmds ← Python)│  │
│  │  └────────────────────────┘  │                           │   │
│  └──────────────────────────────┘                           │   │
│                                                             │   │
│  ┌──────────────────────────────────────────────────────────▼─┐ │
│  │                   BridgeEngine.py                           │ │
│  │  TickReceiver → RenkoBuilder(K=0.00295) → FeatureEngine    │ │
│  │       → InferenceBuffer → EnsemblePredictor(Pred_OS≥1.4)  │ │
│  │       → CommandSender → MT5 execution                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. MQL5 Component Requirements

### 3.1 TickSender EA (`TickSender.mq5`)

**FR-MQL-01**: On every `OnTick()`, format and send over TCP port 9000.

**FR-MQL-02**: Use MQL5 native socket API (`SocketCreate`, `SocketConnect`). Requires MT5 Build ≥ 2400.

**FR-MQL-03**: On `OnInit()`, send `DAYOPEN` message first, then send history batch (`HTICK` × N), then `HDONE`. Only switch to live `TICK` mode after `HDONE` is sent.

**FR-MQL-04**: Listen on port 9001 for Python commands. Parse in `OnTimer()` every 100ms.

**FR-MQL-05**: Execute received commands via `OrderSend()`. Send `CONFIRM` back on port 9000 regardless of success/failure.

**FR-MQL-06**: If Python's port 9000 server is not yet up, retry `SocketConnect` every 2 seconds.

**FR-MQL-07**: On `OnDeinit()`, close all socket handles gracefully.

### 3.2 Message Protocol

> [!WARNING]
> Socket binding to `localhost` (127.0.0.1) is critical. Do NOT bind to 0.0.0.0 or external IPs. This would expose the command socket (port 9001) to network access, allowing anyone on the network to execute trades. Always use `localhost` for both EA connection and Python server binding.

**Port 9000 — MT5 → Python (Tick Channel)**

| Message | Format | Trigger |
|---|---|---|
| Day Open | `DAYOPEN\|<time_msc>\|<price>` | `OnInit()`, first |
| History Tick | `HTICK\|<time_msc>\|<bid>\|<ask>\|<bid_vol>\|<ask_vol>` | History batch |
| History Done | `HDONE\|<count>` | After last HTICK |
| Live Tick | `TICK\|<time_msc>\|<bid>\|<ask>\|<bid_vol>\|<ask_vol>` | Every `OnTick()` |
| Heartbeat | `HEARTBEAT\|<time_msc>` | If no tick for 500ms |
| Confirm | `CONFIRM\|<req_id>\|<ticket>\|OK` or `CONFIRM\|<req_id>\|0\|ERROR\|<code>` | After order |

**Port 9001 — Python → MT5 (Command Channel)**

| Message | Format |
|---|---|
| Buy | `BUY\|<price>\|<sl>\|<tp>\|<volume>\|<req_id>` |
| Sell | `SELL\|<price>\|<sl>\|<tp>\|<volume>\|<req_id>` |
| Close | `CLOSE\|<ticket>\|<req_id>` |
| Modify SL | `MODIFYSL\|<ticket>\|<new_sl>\|<req_id>` |

**Encoding**: UTF-8, newline-terminated (`\n`). All floats: 5 decimal places dot-decimal. Timestamps: ms since Unix epoch (int64).

---

## 4. Python Component Requirements

### 4.1 Tick Receiver (`bridge/tick_receiver.py`)

**FR-PY-01**: Bind TCP server to `localhost:9000`. Accept one persistent connection from EA.

**FR-PY-02**: Parse all message types. Route to appropriate handlers.

**FR-PY-03**: `TICK` and `HTICK` messages pushed to thread-safe `queue.Queue(maxsize=10000)` and `list` respectively.

**FR-PY-04**: `CONFIRM` messages pushed to `confirm_queue` for CommandSender.

**FR-PY-05**: Log `WARNING` if no message received for 10s. Log `ERROR` if 30s. Enter degraded mode if 60s.

### 4.2 Command Sender (`bridge/command_sender.py`)

**FR-PY-06**: Connect to `localhost:9001`. Send formatted command strings.

**FR-PY-07**: After sending each command, wait for matching `CONFIRM` from `confirm_queue`.

**FR-PY-08**: **[NEW]** Confirmation timeout: **5 seconds**. If no `CONFIRM` arrives within 5s, log `ERROR: COMMAND TIMEOUT (req_id=<id>)`. DO NOT update state. DO NOT retry automatically. Operator must manually inspect MT5 terminal.

**FR-PY-09**: Auto-reconnect: if `sendall()` raises `BrokenPipeError`, attempt reconnect once with 2s delay. If reconnect fails, log `CRITICAL` and enter degraded mode (no new trades).

### 4.3 **[UPDATED]** Renko Builder (`bridge/renko.py`)

**FR-PY-RENKO-01**: Brick size formula — **strictly enforced**:
```python
brick_size = today_open_price * 0.00295   # K=0.00295 ONLY
# Example: 2400 * 0.00295 = 7.08 pts
# DEPRECATED: brick_size = open * 0.00118  (DO NOT USE — unprofitable)
# DEPRECATED: brick_size = open * 0.0018   (DO NOT USE — original bot spec is obsolete)
```

**FR-PY-RENKO-02**: `today_open_price` comes from the EA's `DAYOPEN` message received at startup. It is the D1 bar open for the current broker session day.

**FR-PY-RENKO-03**: Reversals require **2× brick_size** movement. `while` loop handles gap-fills.

**FR-PY-RENKO-04**: Feed **bid price** to `update_tick()` — matching training pipeline convention.

### 4.4 **[UPDATED]** Feature Engine (`bridge/feature_engine.py`)

Port from `src/feature_engine.py`. The following critical rules must be preserved exactly:

**FR-PY-FEAT-01**: Process EVERY tick — no sampling, no skipping. Z-score windows depend on continuous updates.

**FR-PY-FEAT-02**: OFI uses **weak inequalities** (`>=`, `<=`), not strict:
```python
e_k = (
    (1 if dBid >= 0 else 0) * bid_vol_k          # bid lift/refresh
  - (1 if dBid <= 0 else 0) * bid_vol_{k-1}      # bid cancel/pull
  - (1 if dAsk <= 0 else 0) * ask_vol_k           # ask cancel/pull
  + (1 if dAsk >= 0 else 0) * ask_vol_{k-1}       # ask offer/refresh
)
```

**FR-PY-FEAT-03**: Susceptibility = raw division FIRST, then z-score:
```python
susc_raw = raw_ofi / (depth_raw + 1e-8)   # Divide raws
z_susc = z_susc_tracker.update(susc_raw)  # THEN z-score
# NEVER: z_susc = z_ofi / z_depth  (this is WRONG)
```

**FR-PY-FEAT-04**: **[UPDATED]** Volume Fallback — Full Specification:

When `bid_vol <= 0` OR `ask_vol <= 0` (broker does not supply tick volume):
```python
# Price-direction proxy (validated: maintains 88.25% WR in ablation test)
prev_mid = (prev_bid + prev_ask) / 2.0
curr_mid = (bid + ask) / 2.0

raw_ofi  = 1.0  if curr_mid > prev_mid  else \
          -1.0  if curr_mid < prev_mid  else 0.0
depth_raw = 0.0   # No volume data available
susc_raw  = 0.0   # Cannot compute susceptibility without depth
# z-score trackers receive 0.0 for depth and susc — handled gracefully
```

This fallback is sourced from the ablation study in Phase 8 of the training pipeline (see `Resources/implementation.md` Phase 8 Volume Mitigation Results).

**FR-PY-FEAT-05**: Z-score window = 1000 ticks, warmup threshold = 30. Return 0.0 below warmup. Use incremental Welford formula (O(1)) at full window.

### 4.5 **[UPDATED]** Inference Configuration

**FR-PY-INFER-01**: **[UPDATED]** Signal thresholds — **unified for K=0.00295**:
```python
PROB_WIN_THRESHOLD = 0.5    # Standard — unchanged
PRED_OS_THRESHOLD  = 1.4    # Calibrated on K=0.00295 holdout → 90.3% WR
ENSEMBLE_VOTING    = "majority"   # >= 2 of 3 models must signal
```

> [!NOTE]
> **Why 1.4?** The original thresholds (1.60/1.70/1.80 per fold) were calibrated on `K=0.00118` data. The K=0.00295 holdout feasibility study (`src/backtest_holdout_K295.py`) showed that a unified threshold of **1.4** maximizes trade frequency and Win Rate (90.3%) under the new brick scale. At threshold 1.3, WR is 87.5% with 136 trades/year. At 1.4, WR is 90.3% with 93 trades/year. The 1.4 value is the operational optimum.

**FR-PY-INFER-02**: Model paths (priority order):
1. `BrickOfTicks_Trader/models/fold_N/model.keras` (primary — already present)
2. `outputs/exec/cv/fold_N/model.keras` (fallback from training output)
3. On startup, verify all 3 models load or exit with `CRITICAL` log

**FR-PY-INFER-03**: **[UPDATED — BAITING DISABLED]**

> [!CAUTION]
> **BAITING STRATEGY IS DISABLED.** The baiting strategy (placing a REVERSE trade when all 3 models predict `prob_win < 0.2` AND `pred_os < 0.7`) showed high WR in the offline backtest (88.75%). However, the forensic execution analysis proved that under **execution pricing** (bid/ask spread applied at entry and exit), baiting trades collapse to sub-50% WR because the reversal entry is always at the ask for a SELL signal — the worst possible execution price. Until execution pricing is re-validated for baiting specifically, this strategy is disabled.
>
> **No code path in `bridge/ensemble.py` should return `action = -1`.**

### 4.6 **[UPDATED]** Break-Even Logic

**FR-PY-BE-01**: Complete break-even specification:

```
TRIGGER: When price moves >= 0.3125 × brick_size in the favorable direction from entry

  BUY:  current_bid >= entry_price + (0.3125 × brick_size)
  SELL: current_ask <= entry_price - (0.3125 × brick_size)

ACTION: Send MODIFYSL command → new_sl = entry_price (lock in zero loss)

FREQUENCY: Check on EVERY incoming tick after position is opened

ONE-TIME: Once triggered, set state.be_triggered = True. Never revert even if price pulls back.
```

**Rationale**: `0.3125 = 5/16`. This value means the trade has moved halfway between entry and TP before break-even activates, providing sufficient buffer against random noise while protecting against trend reversals before TP is hit.

**FR-PY-BE-02**: State machine:
```
IDLE → [trade opened] → MONITORING → [BE trigger] → BE_ACTIVE → [TP or SL hit] → IDLE
                                     [SL hit]    → IDLE (without BE)
```

### 4.7 **[NEW]** Daily Rollover Logic

**FR-PY-ROLL-01**: Rollover detection — Python detects a new trading day via a new `DAYOPEN` message from the EA. The EA sends `DAYOPEN` on `OnInit()` and again each time `iOpen(_Symbol, PERIOD_D1, 0)` changes (checked in `OnTimer()`).

**FR-PY-ROLL-02**: Rollover action sequence:
1. Log: `ROLLOVER: Previous brick_size=<X>, new day_open=<Y>`
2. Compute: `new_brick_size = new_day_open × 0.00295`
3. Call `renko.update_brick_size(new_brick_size)` — affects only NEW bricks
4. Call `feature_engine.update_brick_size(new_brick_size)` — affects Progress calculation
5. Update `state.session_date` and reset `state.daily_pnl = 0.0`
6. Log: `ROLLOVER: New brick_size=<new_brick_size>`

**FR-PY-ROLL-03**: Open position handling during rollover:
- If a position is open at rollover: **DO NOT CLOSE**
- Keep the OLD brick_size for SL/TP management of that trade (stored in state)
- New trades after rollover use the NEW brick_size
- The old `active_sl` and `active_tp` in state remain unchanged

**FR-PY-ROLL-04**: Broker timezone: Most XAUUSD brokers use GMT+2 (or GMT+3 during DST). The rollover time is broker-dependent. Python does not need to track time — it simply reacts to the `DAYOPEN` message.

### 4.8 **[NEW]** Trading Logic & Monitoring

**FR-PY-TRADE-01**: Log spread at entry for all trades
- Compute: `entry_spread = entry_ask - entry_bid` (or bid - ask for SHORT)
- Log to `trades.csv` as `entry_spread_pts` column
- Post-session analysis: flag trades where `entry_spread > 0.60 pts`
- Phase 2 enhancement: add pre-entry spread filter

---

## 5. Warmup Protocol

**FR-WARM-01**: EA sends history batch at startup. Python waits for `HDONE` before entering live mode.

**FR-WARM-02**: Python replays all `HTICK` messages through `FeatureEngine` and `RenkoBuilder` to establish state. `InferenceBuffer` fills with brick snapshots.

**FR-WARM-03**: Warmup gate: `≥ 5,000 ticks processed` AND `≥ 10 bricks formed`. If gate not met from history, Python enters **Warmup Live Mode** — processes ticks but suppresses trade execution until gate is met.

**FR-WARM-04**: During warmup, no trades are placed regardless of any signal.

**FR-WARM-05**: Log warmup result: `WARMUP DONE: ticks=<N>, bricks=<M>, z_window_fill=<pct>%`

---

## 6. **[UPDATED]** Error Recovery Policy

### 6.1 Socket Disconnection Recovery

**FR-ERR-01**: Exponential backoff retry schedule (Python → MT5 side):
```
Attempt 1: wait 1s
Attempt 2: wait 2s
Attempt 3: wait 4s
...
Attempt 8: wait 60s (cap)
Max retries: 10
After 10 failures: Python exits with exit code 2 (not 0)
```

**FR-ERR-02**: Tick queue buffering: maintain `queue.Queue(maxsize=10000)`. If queue is full, drop oldest ticks and log `WARNING: TICK BUFFER OVERFLOW`.

**FR-ERR-03**: If position is open and disconnect lasts > 60s: log `CRITICAL: OPEN POSITION — SOCKET DOWN > 60s`. Do NOT auto-close. Operator must manually check MT5 terminal.

### 6.2 Degraded Mode

**FR-ERR-04**: Python enters **Degraded Mode** when:
- No ticks received for 30+ seconds
- Socket reconnection fails (all 10 attempts exhausted)
- Command confirmation timed out twice in a row

In Degraded Mode:
- Accept and process ticks (if any resume)
- Do NOT open new trades
- Do NOT close existing trades automatically
- Log `DEGRADED MODE ACTIVE` on every 60s heartbeat

**FR-ERR-05**: Degraded Mode exits when:
- Normal tick stream resumes (3 ticks received within 5s)
- Log `DEGRADED MODE CLEARED — resuming normal operation`

### 6.3 MT5 Terminal Crash

**FR-ERR-06**: Python detects terminal crash via socket timeout (no ticks for 30s). Enter Degraded Mode. Attempt reconnect every 60s.

**FR-ERR-07**: If the EA crashes but MT5 is still running: operator must reattach EA to the chart manually. Python will reconnect automatically on next `SocketConnect` attempt.

### 6.4 Command Timeout

**FR-ERR-08**: 5-second timeout on all trade commands. If timeout fires:
- Log `ERROR: COMMAND TIMEOUT — req_id=<id>, command=<BUY/SELL>`
- Do NOT update `state.active_ticket`
- Do NOT send another order (prevent duplicates)
- Log `ACTION REQUIRED: Check MT5 terminal manually for pending orders`

---

## 7. **[UPDATED]** Latency Targets

> [!NOTE]
> These are realistic targets measured on M1 MacBook Pro. Intel Mac users may see 1.5–2× slower values for TensorFlow inference.

| Stage | Target (p95) | Acceptable |
|---|---|---|
| Socket receive → feature vector | < 15ms | < 30ms |
| Feature vector → brick detection | < 5ms | < 10ms |
| Brick close → model inference (3 models) | < 80ms | < 150ms |
| Inference → command sent | < 10ms | < 20ms |
| Command sent → MT5 execution (broker) | < 50ms | < 100ms |
| **Total: Brick close → fill** | **< 160ms** | **< 310ms** |

**Impact**: At K=0.00295, bricks last ~133 minutes on average. A 310ms latency represents 0.04% of the brick's lifetime — negligible for this strategy.

---

## 8. **[NEW]** Phase -1: Data Analytics & Broker Data Audit

Before live trading, we must verify that data received from the MT5 broker (via socket ticks) is consistent with the offline training data (Dukascopy parquet ticks). Differences could degrade model predictions.

### 8.1 What to Audit

| Dimension | Training Data Source | Live Data Source | Risk if Different |
|---|---|---|---|
| Spread distribution | Dukascopy raw parquet | Broker raw ticks | Feature `z_Spread` distribution shifts |
| Tick frequency (velocity) | ~3-8 ticks/sec | Broker-dependent | Feature `z_Vel` distribution shifts |
| Volume (bid_vol/ask_vol) | Dukascopy tick volumes | Broker-dependent | OFI/Susc features may degrade |
| Bid/ask convention | Dukascopy bid | Broker bid | Renko price level differences |
| Time resolution | Millisecond | Millisecond | Possible rounding artifacts |

### 8.2 Audit Process

1. Collect 1 full trading session of raw ticks via socket bridge (without trading)
2. Export to parquet: `bridge/logs/live_ticks_audit_<date>.parquet`
3. Run `bridge/data_audit.py` to compare distributions

### 8.3 Volume Fallback Validation

If the broker does NOT provide `bid_vol`/`ask_vol` (common with retail XAUUSD CFD brokers):
- All ticks will trigger the **Volume Fallback mode** (see FR-PY-FEAT-04)
- Must verify that the fallback (`sign(Δmid)`) produces consistent z_OFI distributions
- Ablation study confirmed: Tick Direction encoding yields **88.25% WR** (only 1.5% below full-volume model)
- This is acceptable for Phase 1 live testing

See full audit specification in `SOCKET_BRIDGE_IMPL_v2.md` Phase -1.

---

## 9. **[NEW]** Failure Mode Analysis

### 9.1 MT5 Terminal Crash
- **Symptom**: No ticks for > 30s
- **Impact**: Python enters Degraded Mode, no new trades
- **Mitigation**: Auto-reconnect every 60s
- **Recovery**: Restart MT5, reattach EA — Python auto-reconnects

### 9.2 Python Process Killed
- **Symptom**: EA sends ticks to closed socket → `SocketSend` returns error
- **Impact**: EA OnTick() blocks briefly on failed send
- **Mitigation**: EA detects send failure, closes socket, waits for new connection with 2s retry
- **Recovery**: Restart `python bridge/main.py` → state.json prevents duplicate trades

### 9.3 Broker Disconnection During Trade
- **Symptom**: CONFIRM timeout after order sent
- **Impact**: Uncertain if order executed
- **Mitigation**: Log CRITICAL alert. Operator checks MT5 terminal manually
- **Recovery**: If order went through, manually update `state.json` with ticket number

### 9.4 Spread Spike During Entry
- **Symptom**: Execution spread 3× normal (common during news events)
- **Impact**: Effective RR degrades from 1:1 to ~0.7:1 for that trade
- **Mitigation** (Phase 1): Accept and log. Do not cancel.
- **Future (Phase 2)**: Add spread filter — skip entry if `live_spread > 0.60 pts`

### 9.5 Warmup Insufficient
- **Symptom**: < 10 bricks formed from history batch
- **Impact**: Inference Buffer returns None → no trades for first few hours
- **Mitigation**: Warmup Live Mode — process ticks live until gate is met
- **Recovery**: Automatic — no operator action needed

---

## 10. **[NEW]** Pre-Deployment Checklist

### Hardware
- [ ] Stable internet connection (Ethernet preferred over WiFi for tick streaming)
- [ ] Minimum 8GB RAM, 4-core CPU
- [ ] 10GB free disk space (for rotating logs and tick archives)

### Software
- [ ] macOS 12.0+ (Monterey or later)
- [ ] Python 3.9+ with TensorFlow ≥ 2.15
- [ ] MT5 terminal installed and logged into **demo account**
- [ ] MetaEditor installed (for compiling `TickSender.mq5`)
- [ ] MT5 Build ≥ 2400 (for `SocketCreate` API)

### Configuration
- [ ] Broker provides XAUUSD with live spreads on demo
- [ ] Ports 9000 and 9001 not blocked by macOS firewall (`System Settings → Network → Firewall`)
- [ ] Python bridge has write permission to `bridge/logs/`
- [ ] `bridge/logs/state.json` either absent (fresh start) or valid JSON

### Models & Data
- [ ] All 3 fold models present: `BrickOfTicks_Trader/models/fold_1,2,3/model.keras`
- [ ] `K=0.00295` holdout tensors generated (`outputs/tensors_holdout_K295/`)
- [ ] Feature engine parity test passed (Phase 4.2 in Tasks)

### Pre-Flight Test
- [ ] Start bridge: `python bridge/main.py` → wait for WARMUP DONE
- [ ] Verify brick_size in log: `brick_size = <day_open> * 0.00295`
  - Example: if day_open = 2400.00, expect brick_size = 7.08
- [ ] Verify first brick forms correctly (open/close/high/low in log)
- [ ] Run `python bridge/dry_run.py` — verify all 3 models load
- [ ] Run feature engine unit tests: `python -m pytest bridge/tests/test_feature_bridge.py`
- [ ] Attach EA to XAUUSD M1 chart, run `nc -l 9000` — verify tick stream arrives
- [ ] Send test command: `echo "BUY|2400.10|2393.00|2407.20|0.01|test001" | nc 127.0.0.1 9001` — verify `CONFIRM` in port 9000 output

---

## 11. Success Criteria (5-Day Paper Trading Gate)

> [!IMPORTANT]
> All criteria must be met before transitioning to any live-money deployment.

| # | Criterion | Measurement Method | Target |
|---|---|---|---|
| 1 | Uptime | `bridge.log` downtime events | > 95% of market hours |
| 2 | Tick capture rate | EA send count vs Python receive count | > 99% |
| 3 | Feature parity | Compare offline vs live features on same 500 ticks | < 0.1% deviation |
| 4 | Prediction parity | Compare offline vs live predictions on same tensors | 100% match |
| 5 | Trade execution | Signals resulting in confirmed MT5 order | 100% (0 timeouts) |
| 6 | Win rate | Closed trades (minimum 10) | > 72% |
| 7 | SL/TP correctness | `abs(sl_dist - brick_size) < 0.01` for all trades | 100% |
| 8 | No unhandled exceptions | `bridge.log` CRITICAL/EXCEPTION count | 0 |

---

## 12. Out of Scope (v1.0)

- Multi-symbol support
- ONNX inference acceleration
- Web dashboard / monitoring UI
- Position sizing beyond fixed `LOT_SIZE = 0.01`
- Baiting/reversal strategy (disabled — see FR-PY-INFER-03)
- Automated model retraining
- The K=0.00295 model retraining — Phase 1 uses cross-scale models already validated
