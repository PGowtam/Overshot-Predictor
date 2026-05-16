# BrickOfTicks — Local Socket Bridge
# Task Breakdown v2.0
# **[UPDATED]** — All critical issues resolved

> Track: `[ ]` todo · `[/]` in progress · `[x]` done  
> **Rule**: No phase begins until ALL previous phase tasks are `[x]`.  
> **K = 0.00295** | **Pred_OS ≥ 1.4** | **Baiting: DISABLED**

---

## **[NEW]** Phase -1: Broker Data Audit

> Run this phase BEFORE any live connection. Requires ~60 min of socket tick collection.

### -1.1 Tick Collection
- [ ] Attach `TickSender.mq5` to chart in listen-only mode (no command socket yet)
- [ ] Run `python bridge/data_audit.py --collect --duration=60` to capture 1 session of ticks
- [ ] Verify output file: `bridge/logs/audit/live_ticks_<date>.parquet` exists and has > 5000 rows
- [ ] Export tick count from EA log (Experts tab) for comparison

### -1.2 Distribution Analysis
- [ ] Run `python bridge/data_audit.py --analyze` against `Data/Raw/Ticks/2023/` training parquets
- [ ] **Spread check**: Assert `live_spread_mean` within 20% of `train_spread_mean`
  - Fail action: Log broker name and spread stats; investigate feed quality before proceeding
- [ ] **Velocity check**: Assert `live_median_dt_ms` within 50% of `train_median_dt_ms`
  - Fail action: Log warning; proceed with caution (velocity features may drift)
- [ ] **Volume check**: Record `live_vol_pct` (percentage of ticks with bid_vol > 0)
  - If `live_vol_pct < 50%`: **Volume Fallback is required** — document and proceed to -1.3
  - If `live_vol_pct >= 50%`: Full volume mode confirmed

### -1.3 Volume Fallback Validation (if fallback required)
- [ ] Run `python bridge/data_audit.py --validate-fallback`
- [ ] Assert `proxy_ofi_pos_ratio` is within 10% of 0.50 (balanced signal)
  - Fail action: **DO NOT PROCEED** — investigate broker data feed
- [ ] Confirm understanding: Volume fallback yields **88.25% WR** (vs 90.3% full volume) — acceptable for Phase 1
- [ ] Document broker volume availability in `bridge/logs/audit/broker_profile.json`

### -1.4 Audit Gate
- [ ] Spread: PASS
- [ ] Velocity: PASS or WARNING (proceed with note)
- [ ] Volume fallback: VALIDATED (if applicable)
- [ ] Save audit report: `bridge/logs/audit/audit_report_<date>.md`

---

## Phase 0: MQL5 EA (`TickSender.mq5`)

### 0.1 Environment Check
- [ ] Confirm MT5 is installed on Mac (broker installer: ICMarkets, Pepperstone, or similar)
- [ ] Confirm MT5 Build ≥ 2400: Help → About → check build number
- [ ] Open MetaEditor: verify `SocketCreate` appears in autocomplete
- [ ] Create `mql5/` directory: `mkdir -p BrickOfTicks_Trader/mql5/`

### 0.2 EA Core Implementation
- [ ] Create `mql5/TickSender.mq5` in MetaEditor
- [ ] Implement `OnInit()`: connect tick_socket to `127.0.0.1:9000` with 10-retry loop (2s each)
- [ ] Implement `OnInit()`: send `DAYOPEN|<time_msc>|<d1_open>` immediately after connect
- [ ] Implement `SendHistory(5000)`: call `CopyTicks()`, send all as `HTICK|...` messages
- [ ] Implement `OnInit()`: send `HDONE|<count>` after history batch
- [ ] Implement `OnInit()`: create cmd_server socket, `SocketBind(:9001)`, `SocketListen(1)`
- [ ] Implement `OnTick()`: format and send `TICK|<time_msc>|<bid>|<ask>|<vol>|<vol>\n`
- [ ] Implement `OnTick()`: detect daily rollover (date change), send new `DAYOPEN` message
- [ ] Implement `OnTimer(100ms)`: `SocketAccept(cmd_server)`, read command lines
- [ ] Implement `ProcessCommand()`: handle BUY, SELL, MODIFYSL, CLOSE with `OrderSend()`
- [ ] Implement `ProcessCommand()`: send `CONFIRM|<req_id>|<ticket>|OK` or `ERROR` on tick_socket
- [ ] Implement `OnDeinit()`: close all socket handles

### 0.3 Phase 0 Verification Gate
- [ ] EA compiles without warnings in MetaEditor
- [ ] Attach EA to XAUUSD M1 chart on demo account
- [ ] Run `nc -l 9000` on Mac terminal — verify first message is `DAYOPEN|...`
- [ ] Verify `HTICK` messages arrive (≥ 100 lines before `HDONE`)
- [ ] Verify `HDONE|<count>` arrives and count matches EA log
- [ ] Verify live `TICK` messages arrive continuously after `HDONE`
- [ ] Send test command: `echo "BUY|2400.10|2393.00|2407.20|0.01|test001" | nc 127.0.0.1 9001`
- [ ] Verify `CONFIRM|test001|<ticket>|OK` appears on the `nc -l 9000` terminal
- [ ] Verify order appears in MT5 Positions tab

---

## Phase 1: Tick Receiver (`bridge/tick_receiver.py`)

### 1.1 Implementation
- [ ] Create `bridge/tick_receiver.py`
- [ ] `TickReceiver.__init__()`: initialize `tick_queue(maxsize=10000)`, `history_ticks=[]`, `history_done=Event()`, `confirm_queue=Queue()`, `day_open_price=None`
- [ ] `start()`: bind TCP server to `localhost:9000`, spawn accept thread (daemon)
- [ ] `_accept()`: call `srv.accept()`, loop reading 4096-byte chunks, split on `\n`
- [ ] `_dispatch('TICK')`: parse 6 fields, push dict to tick_queue; if full drop oldest + log WARNING
- [ ] `_dispatch('HTICK')`: append to `history_ticks` list
- [ ] `_dispatch('HDONE')`: set `history_done` event, log count
- [ ] `_dispatch('DAYOPEN')`: store `day_open_price`, log value
- [ ] `_dispatch('CONFIRM')`: push to `confirm_queue`
- [ ] `_dispatch('HEARTBEAT')`: no-op

### 1.2 Verification
- [ ] Unit test: feed `TICK|1714900800123|2400.12|2400.14|3.5|2.1\n` → assert all 5 fields correct types
- [ ] Unit test: feed 200 `HTICK` + `HDONE|200` → assert `len(history_ticks)==200` and `history_done.is_set()`
- [ ] Unit test: feed `DAYOPEN|...|2398.50` → assert `day_open_price == 2398.50`
- [ ] Thread safety: two threads push 500 ticks each → `tick_queue.qsize() <= 1000` (no crash)
- [ ] Overflow test: fill queue to 10000, push one more → oldest dropped, WARNING logged

---

## Phase 2: Command Sender (`bridge/command_sender.py`)

### 2.1 Implementation
- [ ] Create `bridge/command_sender.py`
- [ ] `CommandSender.__init__(confirm_queue, port=9001)`: store references
- [ ] `connect()`: TCP connect to `localhost:9001`, log success
- [ ] `_send(msg_type, fields)`: generate UUID req_id, format pipe-delimited line, `sendall()`
- [ ] `_send()`: on `BrokenPipeError` → reconnect once (2s delay) then retry
- [ ] **[NEW]** `_await_confirm(req_id)`: poll `confirm_queue` with 0.1s timeout, match by req_id, put unmatched back; timeout = 5.0s hard limit
- [ ] On timeout: log `ERROR: COMMAND TIMEOUT req_id=<id>` + `ACTION REQUIRED: Check MT5 terminal`; return `None`
- [ ] `buy(price, sl, tp, volume=0.01)`: `_send('BUY', ...)` → `_await_confirm()`
- [ ] `sell(price, sl, tp, volume=0.01)`: same
- [ ] `modify_sl(ticket, new_sl)`: `_send('MODIFYSL', ...)` → `_await_confirm()`
- [ ] `close(ticket)`: `_send('CLOSE', ...)` → `_await_confirm()`

### 2.2 Verification
- [ ] Unit test: `buy(2400.10, 2393.00, 2407.20)` → assert message format `BUY|2400.10000|...`
- [ ] Integration test: send BUY to live EA on demo → assert `CONFIRM|...|OK` received within 5s
- [ ] **[NEW]** Timeout test: disconnect EA, call `buy()` → assert returns `None` within 5.5s, logs ERROR
- [ ] Reconnect test: drop connection, call `buy()` → assert reconnect attempted, command retried

---

## Phase 3: Renko Builder (K=0.00295 — MANDATORY)

### 3.1 Implementation
- [ ] Create `bridge/renko.py` — port from `bot_implementation.md` Phase 2
- [ ] **[UPDATED]** Set `K_MULTIPLIER = 0.00295` as module-level constant (not 0.0018, not 0.00118)
- [ ] Add deprecation comment: `# DEPRECATED: 0.00118 (unprofitable), 0.0018 (old bot spec)`
- [ ] `BrickEvent` namedtuple: `(open, close, high, low, uptrend, timestamp, brick_size, sequence)`
- [ ] `RenkoBuilder.__init__(day_open_price)`: compute `self.brick_size = day_open_price * K_MULTIPLIER`
- [ ] `update_tick(bid_price, time_ms)`: UP/DOWN while-loop with 2× reversal rule, gap-fills
- [ ] `update_brick_size(new_day_open)`: update `self.brick_size = new_day_open * K_MULTIPLIER`
- [ ] Sequence tracking: binary string, `maxlen=100`

### 3.2 Verification
- [ ] Assert `K_MULTIPLIER == 0.00295` in unit test
- [ ] Example test: `day_open=2400`, `brick_size` should be `7.08 ±0.001`
- [ ] Manual test: UP bricks at 2407.08, 2414.16; reversal requires reaching 2400.00 (2× below 2414.16)
- [ ] Gap-fill test: price jumps from 2400 to 2435.4 → exactly 5 bricks emitted (5 × 7.08)
- [ ] Extreme volatility test: 50 ticks arrive with large jumps → assert gap-fill bricks sequence has contiguous open/close levels
- [ ] **[UPDATED]** Parity test: feed bid prices from `Data/Processed/XAUUSD_Holdout_K00295.csv` → brick count and open/close prices match CSV (not the old K=0.00118 CSV)

---

## Phase 4: Feature Engine Port

### 4.1 Implementation
- [ ] Copy `src/feature_engine.py` → `bridge/feature_engine.py`
- [ ] Verify `LiveFeatureEngine` class (not batch version) is present
- [ ] Verify `RollingZScore`: window=1000, warmup=30, Welford O(1)
- [ ] **[UPDATED]** Verify volume fallback is fully implemented:
  - `if bid_vol <= 0 or ask_vol <= 0:` → `raw_ofi = sign(Δmid)`, `depth_raw = 0.0`, `susc_raw = 0.0`
- [ ] Verify susceptibility: `susc_raw = raw_ofi / (depth_raw + 1e-8)` BEFORE z-score
- [ ] Verify OFI weak inequalities: `dBid >= 0` and `dBid <= 0` (not strict)

### 4.2 Verification
- [ ] Parity test: load 200 ticks from `outputs/holdout_K295/features/` → max feature error < 1e-4
- [ ] First tick: assert returns `[0.0] * 9`
- [ ] **[NEW]** Volume fallback test: inject tick with `bid_vol=0, ask_vol=0`, price up → assert `raw_ofi == 1.0`
- [ ] Susceptibility guard: tick with `depth_raw=0` → no crash, no NaN
- [ ] Weak inequality test: tick where `dBid == 0` (price unchanged, volume changes) → OFI ≠ 0 (volume refresh captured)
- [ ] Z-score sliding window test: push 1500 ticks, assert incremental mean/var exactly matches `np.mean()` and `np.var()` of the last 1000 ticks

---

## Phase 5: Buffer & Ensemble

### 5.1 Buffer Port
- [ ] Copy `inference/buffer.py` → `bridge/buffer.py`
- [ ] Assert `micro_buffer = deque(maxlen=100)` — NOT 5000 (Phase 1 uses 100 for model compatibility)
- [ ] Assert `on_brick_close()` rewrites Flag_Curr (idx 6) and Decay (idx 8)
- [ ] Assert zero-padding at front when < 100 ticks
- [ ] Assert returns `None` when < 10 bricks in history

### 5.2 Ensemble Port
- [ ] Copy `inference/ensemble.py` → `bridge/ensemble.py`
- [ ] **[UPDATED]** Set `PRED_OS_THRESHOLD = 1.4` (unified for K=0.00295)
  - Add comment: `# Validated on K=0.00295 holdout: 90.3% WR, 93 trades/year`
  - Remove per-fold thresholds (1.60, 1.70, 1.80 are obsolete for K=0.00118)
- [ ] **[UPDATED — BAITING REMOVED]** Remove all code paths that return `action=-1`
  - Delete `is_bait` calculation
  - Delete baiting threshold constants
  - Simplify return: `action = 1 if votes >= VOTE_THRESHOLD else 0`
- [ ] Model paths: primary `BrickOfTicks_Trader/models/fold_N/model.keras`, fallback `outputs/exec/cv/fold_N/model.keras`
- [ ] On startup: if any model fails to load → log CRITICAL + `sys.exit(1)`

### 5.3 Verification
- [ ] Assert `PRED_OS_THRESHOLD == 1.4` in unit test
- [ ] **[UPDATED]** Confirm threshold produces correct WR: load `outputs/tensors_holdout_K295/holdout_*.npy` → assert WR at threshold 1.4 is within 5% of 90.3%
- [ ] **[NEW]** Baiting disabled: assert no code path in `predict()` ever returns `action=-1`
  - Test: mock 3 models with `prob_win=0.05, pred_os=0.3` → assert `action == 0` (SKIP, not -1)
- [ ] Voting: 3/3=ENTER, 2/3=ENTER, 1/3=SKIP, 0/3=SKIP

---

## Phase 6: Risk, State & Logger

### 6.1 State Manager
- [ ] Create `bridge/state.py`
- [ ] Implement `DEFAULT_STATE` with all 12 fields (see IMPL_v2 Phase 6)
- [ ] `load()`: read `bridge/logs/state.json`, merge with DEFAULT_STATE (supports new fields)
- [ ] `save()`: atomic write (write to `.tmp` file, then `os.rename()`) to prevent corruption on crash
- [ ] `update(key, value)`: update field + call `save()` immediately
- [ ] `get(key)`: return field value

### 6.2 Risk Manager
- [ ] Create `bridge/risk.py`
- [ ] `check_daily_limit(daily_pnl_pts, brick_size)`: return `False` if `daily_pnl_pts < -(5 × brick_size)`
  - Rationale: 5 losses at 1× brick_size = 5 SL hits (conservative daily halt)
- [ ] `check_position_open(state)`: return `False` if `state.get('active_ticket') != 0`
- [ ] **[UPDATED]** `check_be_trigger(tick, state)`: `True` if price moved `>= 0.3125 × active_brick_size` favorably
  - Add comment: `# 0.3125 = 5/16 — chosen to balance premature exits vs protection`
  - BUY: `tick['bid'] >= entry + 0.3125 × bs`
  - SELL: `tick['ask'] <= entry - 0.3125 × bs`

### 6.3 Trade Logger
- [ ] Create `bridge/trade_logger.py`
- [ ] CSV columns: `timestamp, brick_dir, fold1_pw, fold1_os, fold2_pw, fold2_os, fold3_pw, fold3_os, votes, action, entry, sl, tp, ticket, outcome, pnl_pts, entry_spread_pts`
- [ ] `log_signal(brick, decision)`: write inference row (outcome=PENDING initially)
- [ ] `log_order(ticket, entry, sl, tp, direction)`: update row with order details
- [ ] `log_outcome(ticket, outcome, pnl_pts)`: update row when trade closes
- [ ] `generate_session_report()`: compute WR, expectancy, trade count; compare to backtest baseline (90.3% WR); save `.md` summary

### 6.4 Verification
- [ ] StateManager: write state → kill process → reload → all 12 fields recovered correctly
- [ ] Atomic save test: truncate `.tmp` during write → original JSON intact
- [ ] Risk limit: `daily_pnl=-40, brick_size=7.08` → `check_daily_limit()` returns `False` (−40 < −35.4)
- [ ] BE trigger: `entry=2400, brick_size=7.08, direction=1, bid=2402.22` → `0.3125×7.08=2.2125 → True`
- [ ] Trade logger: 10 signals → CSV has 10 rows, all columns populated
- [ ] Logging format test: verify trades.csv header matches exact columns and `entry_spread_pts` is recorded correctly

---

## Phase 7: BridgeEngine Main Loop (`bridge/main.py`)

### 7.1 Core Implementation
- [ ] `BridgeEngine.__init__()`: initialize all components
- [ ] `start()`: start receiver → load models → connect sender → wait DAYOPEN (30s timeout)
- [ ] `_warmup()`: wait `history_done` event (30s timeout) → replay all `history_ticks` via `_process_tick(..., is_warmup=True)` → log result
- [ ] Warmup gate: `brick_count >= 10` AND `len(z_ofi.deque) >= 1000` → set `state.warmup_done = True`
- [ ] Warmup Live Mode fallback:
  - If after replaying history: brick_count < 10 OR z_ofi.deque < 1000
  - Continue processing live ticks with `is_warmup=True` (no trades)
  - Gate check every 10 bricks: once met → set warmup_done=True → start trading
  - Log: "WARMUP LIVE MODE — brick_count=X, z_ofi_len=Y, target=(10, 1000)"
- [ ] `_run()`: `tick_queue.get(timeout=60)` loop; if `Empty` → log WARNING + check degraded mode
- [ ] `_process_tick()`: feature_engine → buffer.append → renko.update → for each new brick: `buffer.on_brick_close()` → if tensors + not warmup → `_on_signal()`
- [ ] `_on_signal()`: risk checks → ensemble.predict → if action==1 → compute SL/TP → send buy/sell → update state
- [ ] `_check_be()`: on every tick, check if break-even trigger met → send MODIFYSL command

### 7.2 **[NEW]** Rollover Implementation
- [ ] **[UPDATED]** Split rollover into 3 sub-tasks:
  - **Detect**: monitor `receiver.day_open_price` for changes; compare to `state.session_date`
  - **Compute**: `new_brick_size = new_day_open * 0.00295`; log old vs new values
  - **Apply**: call `renko.update_brick_size(new_day_open)`; call `feature_engine.update_brick_size(new_brick_size)`; reset `state.daily_pnl = 0.0`; update `state.session_date`
- [ ] Open position during rollover: verify state keeps `active_sl`, `active_tp`, `active_brick_size` unchanged

### 7.3 **[NEW]** Degraded Mode Implementation
- [ ] `_enter_degraded_mode()`: set `state.degraded_mode = True`; log CRITICAL; suspend new trades
- [ ] `_exit_degraded_mode()`: set `state.degraded_mode = False`; log INFO; resume normal operation
- [ ] Degraded trigger: 3 consecutive `tick_queue.get(timeout=60)` timeouts
- [ ] Degraded exit: 3 ticks received within 5 seconds
- [ ] Reconnect loop in degraded mode: exponential backoff (1,2,4,8,...,60s), max 10 attempts; after 10 → `sys.exit(2)`
- [ ] Graceful shutdown: Ctrl+C → save state → log session summary → `sys.exit(0)`

### 7.4 **[NEW]** Latency Profiling (Verification)
- [ ] Run `python tests/test_latency.py` with all 3 models loaded
- [ ] Log p50, p95, p99 inference latencies
- [ ] Assert p95 inference < 150ms (acceptable target)
- [ ] If p95 > 150ms: document Mac hardware specs and proceed with WARNING (bricks are ~133 min — latency is not critical)

### 7.5 Integration Verification Gate
- [ ] `BridgeEngine.start()` completes without error — "Bridge started" in log
- [ ] After warmup: `brick_count >= 10`, `len(z_ofi.deque) >= 1000`, log shows "WARMUP DONE"
- [ ] 30-minute dry run during market hours: ticks streaming, bricks forming, inference running
- [ ] Verify `trades.csv` gets signal rows even before any trade is taken
- [ ] First trade signal: verify `abs(sl_dist - brick_size) < 0.01` and `abs(tp_dist - brick_size) < 0.01`
- [ ] SKIP signal: verify no order sent when `votes < 2`

---

## Phase 8: 5-Day Paper Trading Validation

### 8.1 Pre-Session Setup
- [ ] Confirm demo account has XAUUSD with live spreads (not fixed demo spreads)
- [ ] Set `LOT_SIZE = 0.01` (minimum risk)
- [ ] Attach `TickSender.mq5` to XAUUSD M1 chart on demo
- [ ] Start `python bridge/main.py` → verify "Bridge Started" and "WARMUP DONE" in log

### 8.2 Day-by-Day Monitoring
- [ ] **Day 1**: Tick stream confirmed live; at least 1 brick closes; at least 1 signal row in `trades.csv`
- [ ] **Day 2**: Review all signal rows — confirm SL/TP correct (`|sl_dist - brick_size| < 0.01`)
- [ ] **Day 3**: Zero unhandled exceptions in `bridge.log`; check for TIMEOUT or DEGRADED events
- [ ] **Day 4**: Generate interim report: current WR on completed trades (if ≥ 5)
- [ ] **Day 5**: Run `generate_session_report()` → review against backtest baseline

### 8.3 **[UPDATED]** Acceptance Criteria (Measurable)
- [ ] **Uptime**: > 95% of market hours (downtime logged; broker outages excluded)
- [ ] **Tick count verification**: 
  - Extract EA tick send count from MT5 Experts log: `<date> <count> ticks sent`
  - Extract Python tick receive count: `grep "ticks received" bridge.log`
  - Compute: capture_rate = python_count / ea_count
  - Assert: capture_rate > 0.99 (>99% of ticks captured)
- [ ] **Feature parity**: on 500 ticks also saved offline → live feature vectors within 0.1% of offline
- [ ] **Prediction parity**: same 500 ticks as tensors → live predictions match offline 100%
- [ ] **Trade execution**: 0 COMMAND TIMEOUT events across all 5 days
- [ ] **Win rate**: on ≥ 10 completed trades → WR > 72% (within 20% of 90.3% backtest)
- [ ] **SL/TP correctness**: 100% of trades have `sl_dist = tp_dist = brick_size ±0.01`
- [ ] **No unhandled exceptions**: CRITICAL/EXCEPTION count = 0 in `bridge.log`

### 8.4 Crash Recovery Test
- [ ] Simulate crash: `kill -9 <pid>` mid-session with open position
- [ ] Restart: `python bridge/main.py` → verify `state.json` loaded, active_ticket recognized
- [ ] Assert: no duplicate trades opened; existing SL/TP untouched

---

## Quick Reference: v1 → v2 Changes

| Issue | v1 (Incorrect) | v2 (Fixed) |
|---|---|---|
| Brick factor | Mixed (0.0018, 0.00118, 0.00295) | **K=0.00295 everywhere** |
| Pred_OS threshold | Mixed (1.3, 1.6, 1.8, 1.4) | **1.4 unified** with rationale |
| Volume fallback | Mentioned but formula missing | **Full formula specified** with ablation citation |
| Baiting strategy | Referenced as active | **REMOVED** — forensic audit shows sub-50% under execution pricing |
| Break-even | "0.3125 × brick_size" no rationale | **5/16 formula explained** with state machine |
| Day rollover | Vague single task | **3 sub-tasks** (detect/compute/apply) + open position handling |
| Error recovery | "auto-reconnect" — no detail | **Exponential backoff**, degraded mode, exit codes |
| Latency targets | `< 5ms` unrealistic | **Realistic: p95 < 150ms inference** |
| Model paths | Inconsistent | **Primary + fallback paths** with startup validation |
| Confirm timeout | Not specified | **5s hard timeout** with NO-RETRY policy |
| Success criteria | Vague ("no crashes") | **8 measurable criteria** with measurement method |
| Baiting tasks | Active | **Replaced**: assert `action=-1` never returned |
| Data audit | Missing | **New Phase -1** with broker drift analysis and fallback validation |
