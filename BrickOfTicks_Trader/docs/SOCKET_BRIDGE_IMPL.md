# BrickOfTicks — Local Socket Bridge
# Implementation Plan

> **Platform**: macOS | **Approach**: MQL5 EA ↔ Python TCP Socket | **Brick Size**: K=0.00295

---

## Project Structure

```
BrickOfTicks_Trader/
├── docs/
│   ├── SOCKET_BRIDGE_PRD.md        # Product Requirements (this system)
│   ├── SOCKET_BRIDGE_IMPL.md       # This file
│   └── SOCKET_BRIDGE_TASKS.md      # Task tracker
├── bridge/
│   ├── __init__.py
│   ├── main.py                     # Entry point — BridgeEngine orchestrator
│   ├── tick_receiver.py            # TCP server port 9000 — parses tick messages
│   ├── command_sender.py           # TCP client port 9001 — sends trade commands
│   ├── renko.py                    # Renko builder (K=0.00295)
│   ├── feature_engine.py           # 9D feature vector + RollingZScore (port from src/)
│   ├── buffer.py                   # Micro-buffer + tensor assembly
│   ├── ensemble.py                 # 3-fold ensemble loader + voting
│   ├── risk.py                     # Daily drawdown guard
│   ├── state.py                    # JSON state persistence
│   ├── trade_logger.py             # CSV trade log + session report
│   └── logs/
│       ├── state.json
│       ├── trades.csv
│       └── bridge.log
├── mql5/
│   └── TickSender.mq5              # MQL5 Expert Advisor
├── models/                         # Pre-trained fold models (already present)
│   ├── fold_1/model.keras
│   ├── fold_2/model.keras
│   └── fold_3/model.keras
└── tests/
    ├── test_protocol.py            # Socket protocol parsing tests
    ├── test_renko_bridge.py        # Renko parity tests
    ├── test_feature_bridge.py      # Feature parity tests
    └── test_ensemble_bridge.py     # Inference parity tests
```

---

## Phase 0: MQL5 Expert Advisor (`TickSender.mq5`)

### Objective
Build the MQL5 EA that runs inside MT5 (Mac), streams every L1 tick to Python via socket, and executes trade commands received from Python.

### Key Implementation

```mql5
// Dual-socket architecture:
// Socket 1 (client → port 9000): EA sends ticks to Python
// Socket 2 (server on port 9001): EA receives commands from Python

int tick_socket;     // Client — connects to Python's port 9000 server
int cmd_server;      // Server — listens on port 9001 for Python commands
int cmd_client;      // Accepted connection from Python

int OnInit() {
    tick_socket = SocketCreate();
    SocketConnect(tick_socket, "127.0.0.1", 9000, 5000);
    
    // Send DAYOPEN on init
    string day_open_msg = StringFormat("DAYOPEN|%lld|%.5f\n",
        TimeTradeServer() * 1000LL, iOpen(_Symbol, PERIOD_D1, 0));
    SocketSend(tick_socket, StringToCharArray(day_open_msg), ...);
    
    // Also send historical ticks for warmup
    SendHistory(5000);
    
    cmd_server = SocketCreate();
    SocketBind(cmd_server, 9001);
    SocketListen(cmd_server, 1);
    
    EventSetMillisecondTimer(100);
    return INIT_SUCCEEDED;
}

void OnTick() {
    MqlTick tick;
    SymbolInfoTick(_Symbol, tick);
    string msg = StringFormat("TICK|%lld|%.5f|%.5f|%.2f|%.2f\n",
        tick.time_msc, tick.bid, tick.ask,
        tick.volume_real, tick.volume_real);  // bid/ask vol approx
    SocketSend(tick_socket, StringToCharArray(msg), ...);
}

void OnTimer() {
    // Accept any pending command connections
    if (cmd_client == INVALID_HANDLE) {
        cmd_client = SocketAccept(cmd_server, 0);
    }
    // Read and execute any commands
    if (cmd_client != INVALID_HANDLE) {
        string line = ReadSocketLine(cmd_client);
        if (line != "") ProcessCommand(line);
    }
}
```

### History Send (Warmup Support)
```mql5
void SendHistory(int count) {
    MqlTick ticks[];
    int copied = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, 0, count);
    for (int i = 0; i < copied; i++) {
        string msg = StringFormat("HTICK|%lld|%.5f|%.5f|%.2f|%.2f\n", ...);
        SocketSend(tick_socket, ...);
    }
    string done = StringFormat("HDONE|%d\n", copied);
    SocketSend(tick_socket, ...);
}
```

### Command Processor
```mql5
void ProcessCommand(string line) {
    string parts[];
    StringSplit(line, '|', parts);
    string type = parts[0];
    
    if (type == "BUY" || type == "SELL") {
        MqlTradeRequest req = {};
        req.action = TRADE_ACTION_DEAL;
        req.symbol = _Symbol;
        req.price  = StringToDouble(parts[1]);
        req.sl     = StringToDouble(parts[2]);
        req.tp     = StringToDouble(parts[3]);
        req.volume = StringToDouble(parts[4]);
        req.type   = (type == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
        req.magic  = 314159;
        MqlTradeResult res = {};
        OrderSend(req, res);
        
        // Send confirmation back to Python on port 9000
        string confirm = StringFormat("CONFIRM|%s|%d|%s\n",
            parts[5], res.order,
            (res.retcode == TRADE_RETCODE_DONE) ? "OK" : "ERROR");
        SocketSend(tick_socket, ...);
    }
}
```

### Verification (Phase 0 Gate)
- [ ] EA compiles without errors in MT5 MetaEditor
- [ ] EA connects to Python's port 9000 and 9001 on attach
- [ ] Tick messages received by Python's `nc -l 9000` test listener
- [ ] `DAYOPEN` message sent correctly on attach
- [ ] `HDONE` message received after history batch
- [ ] Command `BUY|2400.10|2392.90|2407.30|0.01|req001` executes on demo

---

## Phase 1: Python Tick Receiver & Protocol Parser

### File: `bridge/tick_receiver.py`

```python
import socket, threading, queue, logging

class TickReceiver:
    def __init__(self, port=9000):
        self.port = port
        self.tick_queue = queue.Queue(maxsize=10000)
        self.history_ticks = []
        self.history_done = threading.Event()
        self._running = False
        
    def start(self):
        """Start listening server on port 9000."""
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(('127.0.0.1', self.port))
        self._server.listen(1)
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()
        
    def _accept_loop(self):
        conn, _ = self._server.accept()
        buffer = ''
        while self._running:
            data = conn.recv(4096).decode('utf-8')
            if not data: break
            buffer += data
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                self._dispatch(line.strip())
                
    def _dispatch(self, line):
        if not line: return
        parts = line.split('|')
        msg_type = parts[0]
        
        if msg_type == 'TICK':
            tick = {
                'time_msc': int(parts[1]),
                'bid': float(parts[2]),
                'ask': float(parts[3]),
                'bid_vol': float(parts[4]),
                'ask_vol': float(parts[5])
            }
            self.tick_queue.put(tick)
            
        elif msg_type == 'HTICK':
            self.history_ticks.append({
                'time_msc': int(parts[1]),
                'bid': float(parts[2]), 'ask': float(parts[3]),
                'bid_vol': float(parts[4]), 'ask_vol': float(parts[5])
            })
            
        elif msg_type == 'HDONE':
            self.history_done.set()
            
        elif msg_type == 'DAYOPEN':
            self.day_open_price = float(parts[2])
            
        elif msg_type == 'CONFIRM':
            # Push to a confirm queue for CommandSender
            self.confirm_queue.put({'req_id': parts[1],
                                    'ticket': int(parts[2]),
                                    'status': parts[3]})
```

### Verification (Phase 1 Gate)
- [ ] `TickReceiver.start()` binds to port 9000 without error
- [ ] All 5 message types parsed without KeyError or ValueError
- [ ] History ticks accumulate, `history_done` event fires on `HDONE`
- [ ] `tick_queue` is thread-safe (producer/consumer test)

---

## Phase 2: Python Command Sender

### File: `bridge/command_sender.py`

```python
import socket, uuid, logging

class CommandSender:
    def __init__(self, port=9001, timeout=5.0):
        self.port = port
        self.timeout = timeout
        self._conn = None
        
    def connect(self):
        self._conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._conn.connect(('127.0.0.1', self.port))
        self._conn.settimeout(self.timeout)
        
    def _send(self, msg_type, fields):
        req_id = str(uuid.uuid4())[:8]
        line = '|'.join([msg_type] + [str(f) for f in fields] + [req_id]) + '\n'
        self._conn.sendall(line.encode('utf-8'))
        return req_id
        
    def buy(self, price, sl, tp, volume=0.01):
        return self._send('BUY', [f'{price:.5f}', f'{sl:.5f}', f'{tp:.5f}', f'{volume:.2f}'])
        
    def sell(self, price, sl, tp, volume=0.01):
        return self._send('SELL', [f'{price:.5f}', f'{sl:.5f}', f'{tp:.5f}', f'{volume:.2f}'])
        
    def modify_sl(self, ticket, new_sl):
        return self._send('MODIFYSL', [ticket, f'{new_sl:.5f}'])
        
    def close(self, ticket):
        return self._send('CLOSE', [ticket])
```

### Verification (Phase 2 Gate)
- [ ] `CommandSender.buy()` sends correctly formatted string to EA
- [ ] EA receives command, executes on demo, sends `CONFIRM|...|OK`
- [ ] `CONFIRM|...|ERROR` handled gracefully (no crash)
- [ ] 5-second timeout fires if EA unresponsive

---

## Phase 3: Renko Builder (K=0.00295)

### File: `bridge/renko.py`

Port directly from `src/feature_engine.py` + training `create_renko_dynamic_ticks.py`. **One critical change**:

```python
# OLD (original bot):
brick_size = open_price * 0.0018

# NEW (K=0.00295 architecture):
brick_size = open_price * 0.00295
```

The `RenkoBuilder` class logic is identical to `bot_implementation.md` Phase 2.

### Verification (Phase 3 Gate)
- [ ] Feed 1000 bid prices from `Data/Processed/XAUUSD_Holdout_K00295.csv`
- [ ] Brick count and open/close prices match the CSV exactly
- [ ] Gap-fill test: price jump of 3× brick_size emits 3 bricks

---

## Phase 4: Feature Engine Port

### File: `bridge/feature_engine.py`

Direct copy of `src/feature_engine.py` (the live version, not the batch version). No changes needed. This is the most critical component — it must be identical to training.

### Verification (Phase 4 Gate)
- [ ] Feed 500 ticks from `outputs/holdout_K295/features/tick_vectors/brick_1.npy`
- [ ] Live feature engine output matches stored vectors (max error < 1e-4)
- [ ] Volume fallback: ticks with `bid_vol=0` use price-direction proxy

---

## Phase 5: Buffer + Ensemble (Port)

### Files: `bridge/buffer.py`, `bridge/ensemble.py`

Port directly from `inference/buffer.py` and `inference/ensemble.py`.

**Updated threshold** in `bridge/ensemble.py`:
```python
PRED_OS_THRESHOLD = 1.4   # Updated from 1.6/1.7/1.8 — validated on K=0.00295 holdout
PROB_WIN_THRESHOLD = 0.5  # Standard
VOTE_THRESHOLD = 2        # >= 2 of 3 models
```

### Verification (Phase 5 Gate)
- [ ] Load tensors from `outputs/tensors_holdout_K295/holdout_micro.npy`
- [ ] Ensemble output matches `src/backtest_holdout_K295.py` results for first 20 samples
- [ ] Voting logic: 3/3=ENTER, 2/3=ENTER, 1/3=SKIP

---

## Phase 6: Risk, State & Logger

### Files: `bridge/risk.py`, `bridge/state.py`, `bridge/trade_logger.py`

**`bridge/state.py`** — JSON persistence:
```python
DEFAULT_STATE = {
    "last_tick_msc": 0,
    "active_ticket": 0,
    "active_direction": 0,       # 1=BUY, -1=SELL
    "active_entry": 0.0,
    "active_sl": 0.0,
    "active_tp": 0.0,
    "be_triggered": False,
    "daily_pnl": 0.0,
    "brick_count": 0,
    "session_date": "",
    "warmup_done": False
}
```

**`bridge/risk.py`** — Since we have no direct account access, risk is tracked locally:
```python
class RiskManager:
    def check_daily_limit(self, daily_pnl_pts: float, brick_size: float) -> bool:
        # Approximate: 3% daily loss = 5 losing trades at 1× brick_size
        return daily_pnl_pts > -(5 * brick_size)
```

**`bridge/trade_logger.py`** — CSV log with columns:
`timestamp, brick_dir, fold1_pw, fold1_os, fold2_pw, fold2_os, fold3_pw, fold3_os, votes, action, entry, sl, tp, ticket, outcome`

---

## Phase 7: BridgeEngine Main Loop

### File: `bridge/main.py`

```python
class BridgeEngine:
    def __init__(self):
        self.receiver = TickReceiver(port=9000)
        self.sender = CommandSender(port=9001)
        self.renko = None           # Initialized after DAYOPEN received
        self.feature_engine = LiveFeatureEngine()
        self.buffer = InferenceBuffer()
        self.ensemble = EnsemblePredictor('models/')
        self.risk = RiskManager()
        self.state = StateManager('bridge/logs/state.json')
        self.logger_csv = TradeLogger('bridge/logs/trades.csv')
        
    def start(self):
        # 1. Start receiver (listens on port 9000)
        self.receiver.start()
        # 2. Load models
        self.ensemble.load()
        # 3. Connect command sender to EA
        self.sender.connect()
        # 4. Wait for DAYOPEN message
        self._wait_for_day_open()
        # 5. Initialize Renko with brick_size = day_open * 0.00295
        self.renko = RenkoBuilder(self.receiver.day_open_price * 0.00295,
                                   self.receiver.day_open_price)
        # 6. Warmup: replay history ticks
        self._warmup()
        # 7. Start main loop
        self._run()
        
    def _warmup(self):
        self.receiver.history_done.wait(timeout=30)
        for tick in self.receiver.history_ticks:
            self._process_tick(tick, is_warmup=True)
        logging.info(f"Warmup done: {self.renko.brick_count} bricks, "
                     f"buffer={len(self.buffer.snapshots)} snapshots")
        self.state.update('warmup_done', True)
        
    def _run(self):
        while True:
            try:
                tick = self.receiver.tick_queue.get(timeout=60)
                self._process_tick(tick, is_warmup=False)
            except queue.Empty:
                logging.warning("No tick for 60s — possible disconnect")
                
    def _process_tick(self, tick, is_warmup=False):
        feat_vec = self.feature_engine.compute_vector(
            tick['bid'], tick['ask'], tick['bid_vol'],
            tick['ask_vol'], tick['time_msc'])
        self.buffer.append_tick(feat_vec, self.renko.brick_count)
        
        new_bricks = self.renko.update_tick(tick['bid'], tick['time_msc'])
        for brick in new_bricks:
            self.feature_engine.on_new_brick(brick)
            result = self.buffer.on_brick_close(brick)
            if result and not is_warmup:
                self._on_signal(brick, result)
                
    def _on_signal(self, brick, tensors):
        if not self.risk.check_daily_limit(...): return
        if self.state.get('active_ticket'): return  # Position open
        
        micro, macro = tensors
        decision = self.ensemble.predict(micro, macro)
        self.logger_csv.log_signal(brick, decision)
        
        if decision['action'] == 1:   # ENTER
            entry = brick.close
            dist = brick.brick_size
            if brick.uptrend:
                sl, tp, direction = entry - dist, entry + dist, 'BUY'
            else:
                sl, tp, direction = entry + dist, entry - dist, 'SELL'
            
            req_id = getattr(self.sender, direction.lower())(entry, sl, tp)
            # (Confirmation handled async via confirm_queue)
            
        elif decision['action'] == -1:  # BAIT (Reverse)
            # Reversed direction
            entry = brick.close
            dist = brick.brick_size
            if brick.uptrend:   # UP brick → bet DOWN
                sl, tp = entry + dist, entry - dist
                req_id = self.sender.sell(entry, sl, tp)
            else:
                sl, tp = entry - dist, entry + dist
                req_id = self.sender.buy(entry, sl, tp)
```

### Verification (Phase 7 Gate)
- [ ] `BridgeEngine.start()` completes without error
- [ ] Warmup processes history ticks and logs brick count
- [ ] Main loop runs for 30 minutes without exception
- [ ] At least 1 brick close event triggers ensemble inference
- [ ] Signal logged to `bridge/logs/trades.csv`

---

## Phase 8: Paper Trading Validation

### Objective
Run for 5 consecutive trading days on a demo account. Collect all trade signals (both taken and skipped).

### Key Metrics to Monitor

| Metric | Target |
|---|---|
| Win Rate (taken trades) | > 72% (within 20% of 90.3% backtest) |
| Trade Frequency | 80-100 trades/year (~2-3 per week) |
| Expectancy | > 0 pts per trade |
| Max Consecutive Losses | < 5 |
| Socket Stability | 0 reconnection failures |

### Session Report Generation
After each session, `bridge/trade_logger.py` auto-generates a markdown summary comparing:
- Observed WR vs Backtest WR (90.3%)
- Observed trade frequency vs expected
- Feature distribution stats (to detect drift)

---

## Critical Invariants (Must Never Be Violated)

| # | Rule |
|---|---|
| 1 | `brick_size = day_open * 0.00295` — never hardcoded, never ATR-based |
| 2 | Feature engine processes EVERY tick — no skipping |
| 3 | Micro-buffer NEVER resets at brick boundaries |
| 4 | Susceptibility: `raw_ofi / (raw_depth + 1e-8)` first, then z-score |
| 5 | OFI uses weak inequalities (`>=`, `<=`) |
| 6 | Z-score window = 1000, warmup = 30 ticks |
| 7 | Renko uses **bid price** (not ask, not mid) |
| 8 | Models called with `training=False` |
| 9 | No trades during warmup |
| 10 | State saved after every trade event |
| 11 | `Pred_OS >= 1.4` threshold (validated on K=0.00295 holdout) |
| 12 | No `import MetaTrader5` anywhere in bridge code |
