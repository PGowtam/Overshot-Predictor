# BrickOfTicks — Local Socket Bridge
# Task Breakdown

> Track: `[ ]` todo · `[/]` in progress · `[x]` done  
> **Rule**: No phase begins until ALL previous phase tasks are `[x]`.

---

## Phase 0: MQL5 Expert Advisor (`TickSender.mq5`)

### 0.1 Environment Check
- [ ] Confirm MT5 is installed on Mac (via broker's native installer e.g. ICMarkets, Pepperstone)
- [ ] Confirm MT5 Build number ≥ 2400 (required for `SocketCreate` API)
- [ ] Confirm `SocketCreate`, `SocketConnect`, `SocketBind`, `SocketListen`, `SocketAccept` are available in MetaEditor Autocomplete
- [ ] Create `mql5/` directory in project root for EA file

### 0.2 MQL5 EA Core
- [ ] Create `mql5/TickSender.mq5` file in MetaEditor
- [ ] Implement `OnInit()`: create tick_socket, connect to `127.0.0.1:9000`, send `DAYOPEN` message
- [ ] Implement `OnInit()`: create cmd_server, bind to port 9001, call `SocketListen()`
- [ ] Implement `OnTick()`: format `TICK|<time_msc>|<bid>|<ask>|<vol>|<vol>` and send
- [ ] Implement `OnTimer()`: accept pending client on port 9001, read command lines
- [ ] Implement `OnDeinit()`: close all socket handles

### 0.3 MQL5 EA History (Warmup Support)
- [ ] Implement `SendHistory(count)`: call `CopyTicks()` for last N ticks
- [ ] Format each as `HTICK|...` and send over tick_socket
- [ ] Send `HDONE|<count>` after all history ticks sent
- [ ] Call `SendHistory(5000)` from `OnInit()` before switching to live mode

### 0.4 MQL5 Command Processor
- [ ] Implement `ProcessCommand(line)`: parse pipe-delimited string
- [ ] Handle `BUY`: fill `MqlTradeRequest`, call `OrderSend()`
- [ ] Handle `SELL`: same as BUY with `ORDER_TYPE_SELL`
- [ ] Handle `MODIFYSL`: use `TRADE_ACTION_SLTP`
- [ ] Handle `CLOSE`: opposite market order
- [ ] Send `CONFIRM|<req_id>|<ticket>|OK` or `CONFIRM|...|ERROR|<code>` back on tick_socket
- [ ] Use `MAGIC_NUMBER = 314159` on all orders

### 0.5 Phase 0 Verification
- [ ] EA compiles without warnings in MetaEditor
- [ ] Attach EA to XAUUSD M1 chart on demo account
- [ ] Run `nc -l 9000` on Mac terminal — verify tick messages arriving
- [ ] Verify `DAYOPEN` message arrives first, then `HTICK` batch, then `HDONE`
- [ ] Verify live `TICK` messages arrive after `HDONE`
- [ ] Send `BUY|2400.10|2393.00|2407.20|0.01|req001` via `nc 127.0.0.1 9001` — verify order placed on demo
- [ ] Verify `CONFIRM|req001|<ticket>|OK` received on port 9000

---

## Phase 1: Python Tick Receiver

### 1.1 Implementation
- [ ] Create `bridge/tick_receiver.py`
- [ ] `TickReceiver.__init__()`: initialize tick_queue, history_ticks list, history_done Event, confirm_queue
- [ ] `TickReceiver.start()`: bind to port 9000, start accept thread
- [ ] `_accept_loop()`: accept connection, read lines, call `_dispatch()`
- [ ] `_dispatch()`: route `TICK`, `HTICK`, `HDONE`, `DAYOPEN`, `CONFIRM` messages
- [ ] `TICK` parser: convert to `{time_msc, bid, ask, bid_vol, ask_vol}` dict, push to tick_queue
- [ ] `HTICK` parser: append to `self.history_ticks`
- [ ] `HDONE` handler: set `self.history_done` event, log count
- [ ] `DAYOPEN` handler: store `self.day_open_price`
- [ ] `CONFIRM` handler: push to `self.confirm_queue`

### 1.2 Verification
- [ ] Unit test: feed `TICK|1714900800123|2400.12|2400.14|3.5|2.1\n` string to `_dispatch()` → assert dict fields correct
- [ ] Unit test: feed 100 `HTICK` lines + `HDONE|100` → assert `len(history_ticks)==100` and `history_done.is_set()`
- [ ] Integration test: run `nc` to send 10 tick lines → assert `tick_queue.qsize() == 10`
- [ ] Thread safety test: 2 producer threads pushing ticks → no race conditions

---

## Phase 2: Python Command Sender

### 2.1 Implementation
- [ ] Create `bridge/command_sender.py`
- [ ] `CommandSender.__init__()`: store port, timeout
- [ ] `CommandSender.connect()`: TCP connect to `127.0.0.1:9001`
- [ ] `_send()`: generate UUID request_id, format message, `conn.sendall()`
- [ ] `buy(price, sl, tp, volume)`: call `_send('BUY', ...)`
- [ ] `sell(price, sl, tp, volume)`: call `_send('SELL', ...)`
- [ ] `modify_sl(ticket, new_sl)`: call `_send('MODIFYSL', ...)`
- [ ] `close(ticket)`: call `_send('CLOSE', ...)`
- [ ] Auto-reconnect: if `sendall()` fails, reconnect once then retry

### 2.2 Verification
- [ ] Unit test: `buy(2400.10, 2393.00, 2407.20)` → assert correct format string
- [ ] Integration test with EA: send buy command → verify `CONFIRM|...|OK` received
- [ ] Error handling: EA returns error code → no crash, logs `ERROR`
- [ ] Timeout test: kill EA → verify 5s timeout fires, logs `COMMAND TIMEOUT`

---

## Phase 3: Renko Builder (K=0.00295)

### 3.1 Implementation
- [ ] Create `bridge/renko.py` — port from `bot_implementation.md` Phase 2 spec
- [ ] `BrickEvent` namedtuple: `(open, close, high, low, uptrend, timestamp, brick_size, sequence)`
- [ ] `RenkoBuilder.__init__(brick_size, start_price)`: initialize state
- [ ] `update_tick(price, timestamp_ms)`: UP/DOWN while-loop with 2× reversal rule
- [ ] Sequence tracking: binary string `maxlen=100`
- [ ] `update_brick_size(new_size)`: for daily rollover
- [ ] **CRITICAL**: brick_size set externally as `day_open * 0.00295` — never computed inside

### 3.2 Verification
- [ ] Manual test: `brick_size=7.0, start=2400.0`, feed `[2407.0, 2414.0, 2399.0]`
  - First brick: UP at 2407 (open=2400, close=2407)
  - Second brick: UP at 2414
  - Reversal: 2399 is 15pt below 2414 → needs 14pt (2×7) → triggers at 2400 → DOWN brick at (open=2407, close=2400) ✓
- [ ] Gap fill: `price=2430, start=2400, brick_size=7` → 4 UP bricks emitted
- [ ] Parity test: feed bid prices from `Data/Processed/XAUUSD_Holdout_K00295.csv` → brick count and prices match CSV

---

## Phase 4: Feature Engine Port

### 4.1 Implementation
- [ ] Copy `src/feature_engine.py` to `bridge/feature_engine.py`
- [ ] Verify `LiveFeatureEngine` class is present (not the batch version)
- [ ] Verify `RollingZScore` class: window=1000, warmup=30, Welford O(1)
- [ ] Verify volume fallback: `bid_vol <= 0` → price-direction proxy for OFI
- [ ] Verify `compute_vector()` accepts `(bid, ask, bid_vol, ask_vol, time_ms)`
- [ ] Verify `on_new_brick(brick)` updates context correctly

### 4.2 Verification
- [ ] Parity test: load 200 ticks from `outputs/holdout_K295/features/` parquet
- [ ] Feed through `LiveFeatureEngine` → compare to stored feature vectors (max error < 1e-4)
- [ ] First tick test: assert returns `[0.0] * 9`
- [ ] Susceptibility test: feed tick with `depth_raw=0` → no crash (1e-8 guard)

---

## Phase 5: Buffer & Ensemble Port

### 5.1 Buffer
- [ ] Copy `inference/buffer.py` to `bridge/buffer.py`
- [ ] Confirm `micro_buffer = deque(maxlen=100)` — stores `(9D_vector, brick_id)` tuples
- [ ] Confirm `on_brick_close()` rewrites Flag_Curr (idx 6) and Decay (idx 8)
- [ ] Confirm zero-padding at front when < 100 ticks in buffer
- [ ] Confirm returns `None` when < 10 bricks in history
- [ ] Confirm output shapes: `(1, 10, 100, 9)` and `(1, 10, 3)`

### 5.2 Ensemble
- [ ] Copy `inference/ensemble.py` to `bridge/ensemble.py`
- [ ] Update: `PRED_OS_THRESHOLD = 1.4` (validated on K=0.00295 holdout → 90.3% WR)
- [ ] Confirm models loaded from `BrickOfTicks_Trader/models/fold_1,2,3/model.keras`
- [ ] Confirm voting: `>= 2 of 3` models signal → `action=1`
- [ ] Confirm baiting: ALL 3 models `prob_win < 0.2` AND `pred_os < 0.7` → `action=-1`

### 5.3 Verification
- [ ] Load `outputs/tensors_holdout_K295/holdout_micro.npy` and `holdout_macro.npy`
- [ ] Run first 20 samples through bridge ensemble → assert WR matches `backtest_holdout_K295.py` output
- [ ] Voting test: mock 3 models → 3/3, 2/3, 1/3, 0/3 → correct actions

---

## Phase 6: Risk, State & Logger

### 6.1 State Manager
- [ ] Create `bridge/state.py`
- [ ] `StateManager.load()`: read `bridge/logs/state.json`, fall back to DEFAULT_STATE if missing
- [ ] `StateManager.save()`: atomic write (write to temp, rename) to prevent corruption
- [ ] `StateManager.update(key, value)`: update field and save
- [ ] Default state fields: `last_tick_msc`, `active_ticket`, `active_direction`, `active_entry`, `active_sl`, `active_tp`, `be_triggered`, `daily_pnl`, `brick_count`, `session_date`, `warmup_done`

### 6.2 Risk Manager
- [ ] Create `bridge/risk.py`
- [ ] `check_daily_limit(daily_pnl_pts, brick_size)`: returns False if `daily_pnl_pts < -(5 × brick_size)`
- [ ] `check_position_open(state)`: returns False if `active_ticket != 0`
- [ ] `check_be_trigger(current_price, entry, direction, brick_size)`: returns True if price moved `0.3125 × brick_size` favorably

### 6.3 Trade Logger
- [ ] Create `bridge/trade_logger.py`
- [ ] `log_signal(brick, decision)`: write row to `trades.csv` with full inference details
- [ ] `log_order(req_id, ticket, entry, sl, tp, direction)`: write order confirmation row
- [ ] `generate_session_report()`: compute WR, expectancy, trade count from CSV; print to log and save `.md` report

### 6.4 Verification
- [ ] StateManager: write state → delete process → reload → all fields recovered
- [ ] Risk: `daily_pnl=-40, brick_size=7` → `check_daily_limit()` returns False (−40 < −35)
- [ ] Trade logger: 10 signals logged → CSV has 10 rows with all columns populated

---

## Phase 7: BridgeEngine Main Loop

### 7.1 Implementation
- [ ] Create `bridge/main.py` with `BridgeEngine` class
- [ ] `start()`: initialize components, start receiver, load models, connect sender, wait for `DAYOPEN`
- [ ] `_wait_for_day_open()`: poll `receiver.day_open_price` with 30s timeout
- [ ] `_warmup()`: wait for `history_done`, replay all history ticks via `_process_tick(..., is_warmup=True)`
- [ ] Log warmup result: brick count, z-score window fill, snapshot count
- [ ] `_run()`: `tick_queue.get(timeout=60)` loop
- [ ] `_process_tick(tick, is_warmup)`: feature_engine → buffer.append → renko.update → on brick_close → optionally `_on_signal`
- [ ] `_on_signal(brick, tensors)`: risk checks → ensemble.predict → buy/sell/bait command
- [ ] Break-even monitoring: on every tick after a trade, check `risk.check_be_trigger()` → send `modify_sl` if triggered
- [ ] Daily rollover: detect new `session_date` in state → recompute brick_size from new `DAYOPEN` message
- [ ] Graceful shutdown: `Ctrl+C` → save state → close sockets → print session summary

### 7.2 Verification
- [ ] `BridgeEngine.start()` runs without error, logs "Warmup complete"
- [ ] After warmup: `len(buffer.snapshots) >= 10`, `len(z_ofi.deque) >= 1000`
- [ ] Run 30 min during market hours: verify tick stream, brick closes, inference logs
- [ ] Verify first trade signal: check SL = entry ± brick_size, TP = entry ± brick_size
- [ ] Verify `trades.csv` has correct columns and values

---

## Phase 8: Paper Trading (5-Day Validation)

### 8.1 Session Preparation
- [ ] Verify demo account has XAUUSD available with live spreads
- [ ] Set `LOT_SIZE = 0.01` in `bridge/ensemble.py` or settings
- [ ] Attach `TickSender.mq5` to XAUUSD M1 chart
- [ ] Start `python bridge/main.py` — verify "Bridge Started" log

### 8.2 Day-by-Day Monitoring
- [ ] **Day 1**: Confirm tick stream live, at least 1 brick close, at least 1 signal logged
- [ ] **Day 2**: Review `trades.csv` — confirm SL/TP correct on all taken trades
- [ ] **Day 3**: Verify no unhandled exceptions in `bridge.log`
- [ ] **Day 4**: Check WR on trades taken so far (interim check)
- [ ] **Day 5**: Generate `session_report.md` — compare to backtest baseline

### 8.3 Final Acceptance Criteria
- [ ] Socket connection stable for 5 consecutive trading days
- [ ] Zero unhandled Python exceptions
- [ ] All taken trades have `SL distance = TP distance = brick_size`
- [ ] Win rate on ≥ 10 completed trades > 72%
- [ ] Expectancy per trade > 0 pts
- [ ] State file survives simulated crash-and-restart with no duplicate trades

---

## Quick Reference: Architecture-Level Changes vs. Original Bot

| Item | Original Bot (`bot_prd.md`) | **Socket Bridge (This)** |
|---|---|---|
| MT5 Integration | `import MetaTrader5` (Windows) | MQL5 EA ↔ TCP Socket |
| Platform | Windows VPS | **macOS Local** |
| Brick Factor | `0.0018` | **`0.00295`** |
| Pred_OS Threshold | 1.60/1.70/1.80 | **1.4 (unified, validated)** |
| Tick Fetch | `mt5.copy_ticks_from()` | **History via `HTICK` batch** |
| Order Placement | `mt5.order_send()` | **Socket command BUY/SELL** |
| Account Info | `mt5.account_info()` | **Local PnL tracking** |
| Risk Check | Equity vs Balance (MT5) | **Local brick-size-based rule** |
| Warmup | `copy_ticks_from` 5000 ticks | **EA sends `HTICK` batch** |
