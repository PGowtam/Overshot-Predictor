# BrickOfTicks — Local Socket Bridge
# Implementation Plan v2.0
# **[UPDATED]** — All sections revised per deployment audit

> **Platform**: macOS | **K = 0.00295** | **Pred_OS threshold = 1.4** | **Baiting: DISABLED**

---

## Directory Structure

```
BrickOfTicks_Trader/
├── docs/
│   ├── SOCKET_BRIDGE_PRD_v2.md
│   ├── SOCKET_BRIDGE_IMPL_v2.md     # This file
│   └── SOCKET_BRIDGE_TASKS_v2.md
├── bridge/
│   ├── main.py                      # BridgeEngine orchestrator
│   ├── tick_receiver.py             # TCP server :9000
│   ├── command_sender.py            # TCP client :9001
│   ├── renko.py                     # RenkoBuilder K=0.00295
│   ├── feature_engine.py            # 9D features + RollingZScore
│   ├── buffer.py                    # Micro-buffer + tensor assembly
│   ├── ensemble.py                  # 3-fold loader + voting (NO BAITING)
│   ├── risk.py                      # Daily drawdown guard
│   ├── state.py                     # JSON persistence
│   ├── trade_logger.py              # CSV log + session report
│   ├── data_audit.py                # **[NEW]** Broker data drift analysis
│   └── logs/
│       ├── state.json
│       ├── trades.csv
│       ├── bridge.log
│       └── audit/                   # Raw tick captures for analysis
├── mql5/
│   └── TickSender.mq5
├── models/
│   ├── fold_1/model.keras
│   ├── fold_2/model.keras
│   └── fold_3/model.keras
└── tests/
    ├── test_protocol.py
    ├── test_renko_bridge.py
    ├── test_feature_bridge.py
    ├── test_ensemble_bridge.py
    └── test_latency.py              # **[NEW]** Latency profiling
```

---

## **[NEW]** Phase -1: Broker Data Audit

### Objective
Before any live trading, verify that ticks from the MT5 broker match the statistical profile of the Dukascopy training data. Differences in spread, velocity, or volume availability will degrade model predictions if unchecked.

### Implementation: `bridge/data_audit.py`

```python
"""
Run BEFORE Phase 7 (paper trading).
Collects 1 session of raw socket ticks, compares to training parquet.
"""
import pandas as pd, numpy as np, json
from pathlib import Path

def collect_session_ticks(receiver, duration_minutes=60):
    """Collect raw ticks from socket for audit_duration."""
    ticks = []
    deadline = time.time() + duration_minutes * 60
    while time.time() < deadline:
        tick = receiver.tick_queue.get(timeout=10)
        ticks.append(tick)
    return pd.DataFrame(ticks)

def load_training_sample(n=50000):
    """Load random sample from Dukascopy training parquet files."""
    parquet_dir = Path("Data/Raw/Ticks/2023")
    files = list(parquet_dir.rglob("*.parquet"))[:5]
    dfs = [pd.read_parquet(f).sample(min(10000, len(pd.read_parquet(f)))) for f in files]
    return pd.concat(dfs).head(n)

def audit_distributions(live_df, train_df):
    results = {}
    
    # 1. Spread distribution
    live_spread = live_df['ask'] - live_df['bid']
    train_spread = train_df['ask'] - train_df['bid']
    results['spread'] = {
        'live_mean': live_spread.mean(), 'train_mean': train_spread.mean(),
        'live_std':  live_spread.std(),  'train_std':  train_spread.std(),
        'drift_pct': abs(live_spread.mean() - train_spread.mean()) / train_spread.mean() * 100
    }
    
    # 2. Tick velocity
    live_dt = live_df['time_msc'].diff().dropna()
    train_dt = train_df['time_msc'].diff().dropna()
    results['velocity'] = {
        'live_median_dt_ms': live_dt.median(), 'train_median_dt_ms': train_dt.median(),
        'drift_pct': abs(live_dt.median() - train_dt.median()) / train_dt.median() * 100
    }
    
    # 3. Volume availability
    live_has_vol = (live_df['bid_vol'] > 0).mean()
    train_has_vol = (train_df['bid_vol'] > 0).mean()
    results['volume'] = {
        'live_vol_pct': live_has_vol * 100,
        'train_vol_pct': train_has_vol * 100,
        'fallback_required': live_has_vol < 0.5
    }
    
    # 4. Verdict
    spread_ok  = results['spread']['drift_pct'] < 20
    vel_ok     = results['velocity']['drift_pct'] < 50
    fallback   = results['volume']['fallback_required']
    
    results['verdict'] = {
        'spread_ok':  spread_ok,
        'velocity_ok': vel_ok,
        'volume_fallback_required': fallback,
        'PROCEED': spread_ok and vel_ok,
        'NOTES': "Volume fallback active — expect 88.25% WR (vs 90.3%)" if fallback else "Full volume mode"
    }
    return results

def validate_fallback_if_needed(live_df):
    """If broker has no volume, verify price-direction proxy matches training OFI sign distribution."""
    if (live_df['bid_vol'] > 0).mean() > 0.5:
        return {"skip": "Volume available — fallback not needed"}
    
    # Compute proxy OFI from live ticks
    mid = (live_df['bid'] + live_df['ask']) / 2
    proxy_ofi = np.sign(mid.diff().fillna(0))
    
    # Proxy OFI should have ~50% +1 and ~50% -1 (zero-mean process)
    pos_ratio = (proxy_ofi > 0).mean()
    neg_ratio = (proxy_ofi < 0).mean()
    balanced  = abs(pos_ratio - 0.5) < 0.10
    
    return {
        "pos_ratio": pos_ratio, "neg_ratio": neg_ratio,
        "balanced": balanced,
        "PASS": balanced,
        "note": "Fallback OFI is balanced — suitable for inference" if balanced
                else "WARNING: Proxy OFI is skewed — investigate broker data"
    }
```

### Acceptance Criteria
| Check | Pass | Fail Action |
|---|---|---|
| Spread drift < 20% | Proceed | Investigate broker feed quality |
| Velocity drift < 50% | Proceed | Log warning, proceed with caution |
| Volume available > 50% | Full mode | Activate volume fallback |
| Volume fallback OFI balanced | Proceed | Do NOT trade — investigate |

---

## Phase 0: MQL5 EA (`TickSender.mq5`)

### Key Code

```mql5
#property strict
int    tick_socket   = INVALID_HANDLE;
int    cmd_server    = INVALID_HANDLE;
int    cmd_client    = INVALID_HANDLE;
string last_day_open = "";

int OnInit() {
    // 1. Connect to Python's tick server
    tick_socket = SocketCreate();
    for (int i = 0; i < 10; i++) {
        if (SocketConnect(tick_socket, "127.0.0.1", 9000, 2000)) break;
        Sleep(2000);
    }
    
    // 2. Send DAYOPEN
    double day_open = iOpen(_Symbol, PERIOD_D1, 0);
    SendMsg(StringFormat("DAYOPEN|%lld|%.5f", GetTickCount64(), day_open));
    last_day_open = TimeToString(TimeCurrent(), TIME_DATE);
    
    // 3. Send history for warmup
    SendHistory(5000);
    
    // 4. Open command server
    cmd_server = SocketCreate();
    SocketBind(cmd_server, 9001);
    SocketListen(cmd_server, 1);
    
    EventSetMillisecondTimer(100);
    return INIT_SUCCEEDED;
}

void OnTick() {
    MqlTick t;
    SymbolInfoTick(_Symbol, t);
    string msg = StringFormat("TICK|%lld|%.5f|%.5f|%.2f|%.2f\n",
        t.time_msc, t.bid, t.ask, t.volume_real, t.volume_real);
    SendMsg(msg);
    
    // Daily rollover detection
    string today = TimeToString(TimeCurrent(), TIME_DATE);
    if (today != last_day_open) {
        double new_open = iOpen(_Symbol, PERIOD_D1, 0);
        SendMsg(StringFormat("DAYOPEN|%lld|%.5f\n", t.time_msc, new_open));
        last_day_open = today;
    }
}

void OnTimer() {
    // Accept command client if not connected
    if (cmd_client == INVALID_HANDLE)
        cmd_client = SocketAccept(cmd_server, 0);
    
    // Read commands
    if (cmd_client != INVALID_HANDLE) {
        string line = ReadLine(cmd_client);
        if (line != "") ProcessCommand(line);
    }
}

void ProcessCommand(string line) {
    string parts[];  int n = StringSplit(line, '|', parts);
    string req_id   = (n > 0) ? parts[n-1] : "unknown";
    
    if (parts[0] == "BUY" || parts[0] == "SELL") {
        MqlTradeRequest req = {}; MqlTradeResult res = {};
        req.action   = TRADE_ACTION_DEAL;
        req.symbol   = _Symbol;
        req.price    = StringToDouble(parts[1]);
        req.sl       = StringToDouble(parts[2]);
        req.tp       = StringToDouble(parts[3]);
        req.volume   = StringToDouble(parts[4]);
        req.type     = (parts[0]=="BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
        req.magic    = 314159;
        req.deviation= 20;
        req.type_filling = ORDER_FILLING_IOC;
        OrderSend(req, res);
        
        string ok = (res.retcode == TRADE_RETCODE_DONE) ? "OK" :
                    StringFormat("ERROR|%d", res.retcode);
        SendMsg(StringFormat("CONFIRM|%s|%d|%s\n", req_id, res.order, ok));
    }
    else if (parts[0] == "MODIFYSL") {
        // TRADE_ACTION_SLTP
        MqlTradeRequest req = {}; MqlTradeResult res = {};
        req.action = TRADE_ACTION_SLTP;
        req.position = (ulong)StringToInteger(parts[1]);
        req.sl       = StringToDouble(parts[2]);
        OrderSend(req, res);
        SendMsg(StringFormat("CONFIRM|%s|%s|%s\n", req_id, parts[1],
            (res.retcode==TRADE_RETCODE_DONE)?"OK":"ERROR"));
    }
}
```

---

## Phase 1: Tick Receiver (`bridge/tick_receiver.py`)

```python
import socket, threading, queue, logging

class TickReceiver:
    def __init__(self, port=9000):
        self.tick_queue    = queue.Queue(maxsize=10000)
        self.history_ticks = []
        self.history_done  = threading.Event()
        self.confirm_queue = queue.Queue()
        self.day_open_price = None
        self._port = port

    def start(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', self._port)); srv.listen(1)
        logging.info(f"TickReceiver listening on :{self._port}")
        threading.Thread(target=self._accept, args=(srv,), daemon=True).start()

    def _accept(self, srv):
        conn, addr = srv.accept()
        logging.info(f"EA connected from {addr}")
        buf = ''
        while True:
            data = conn.recv(4096).decode('utf-8', errors='replace')
            if not data: break
            buf += data
            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                self._dispatch(line.strip())

    def _dispatch(self, line):
        if not line: return
        p = line.split('|')
        t = p[0]
        if t == 'TICK':
            tick = {'time_msc': int(p[1]), 'bid': float(p[2]),
                    'ask': float(p[3]), 'bid_vol': float(p[4]), 'ask_vol': float(p[5])}
            try: self.tick_queue.put_nowait(tick)
            except queue.Full:
                self.tick_queue.get_nowait()  # Drop oldest
                self.tick_queue.put_nowait(tick)
                logging.warning("TICK BUFFER OVERFLOW — dropping oldest tick")
        elif t == 'HTICK':
            self.history_ticks.append({
                'time_msc': int(p[1]), 'bid': float(p[2]), 'ask': float(p[3]),
                'bid_vol': float(p[4]), 'ask_vol': float(p[5])})
        elif t == 'HDONE':
            logging.info(f"History batch complete: {p[1]} ticks")
            self.history_done.set()
        elif t == 'DAYOPEN':
            self.day_open_price = float(p[2])
            logging.info(f"Day open price: {self.day_open_price}")
        elif t == 'CONFIRM':
            self.confirm_queue.put({'req_id': p[1], 'ticket': int(p[2]), 'status': p[3]})
        elif t == 'HEARTBEAT':
            pass  # Ignore, just prevents stall detection
```

---

## Phase 2: Command Sender (`bridge/command_sender.py`)

```python
import socket, uuid, logging, queue, time

class CommandSender:
    CONFIRM_TIMEOUT = 5.0  # seconds — HARD REQUIREMENT per FR-PY-08

    def __init__(self, confirm_queue: queue.Queue, port=9001):
        self._port = port
        self._conn = None
        self._confirms = confirm_queue

    def connect(self):
        self._conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._conn.connect(('127.0.0.1', self._port))
        logging.info("CommandSender connected to EA on :9001")

    def _send(self, msg_type, fields):
        req_id = str(uuid.uuid4())[:8]
        line = '|'.join([msg_type] + [str(f) for f in fields] + [req_id]) + '\n'
        try:
            self._conn.sendall(line.encode('utf-8'))
        except BrokenPipeError:
            logging.error("CommandSender: broken pipe — reconnecting")
            time.sleep(2)
            self.connect()
            self._conn.sendall(line.encode('utf-8'))
        return req_id

    def _await_confirm(self, req_id):
        """Wait up to CONFIRM_TIMEOUT for matching CONFIRM message."""
        deadline = time.time() + self.CONFIRM_TIMEOUT
        while time.time() < deadline:
            try:
                conf = self._confirms.get(timeout=0.1)
                if conf['req_id'] == req_id:
                    return conf
                self._confirms.put(conf)  # Not ours — put back
            except queue.Empty:
                pass
        logging.error(f"COMMAND TIMEOUT — req_id={req_id}. Check MT5 terminal manually.")
        return None

    def buy(self, price, sl, tp, volume=0.01):
        req = self._send('BUY', [f'{price:.5f}', f'{sl:.5f}', f'{tp:.5f}', f'{volume:.2f}'])
        return self._await_confirm(req)

    def sell(self, price, sl, tp, volume=0.01):
        req = self._send('SELL', [f'{price:.5f}', f'{sl:.5f}', f'{tp:.5f}', f'{volume:.2f}'])
        return self._await_confirm(req)

    def modify_sl(self, ticket, new_sl):
        req = self._send('MODIFYSL', [ticket, f'{new_sl:.5f}'])
        return self._await_confirm(req)
```

---

## Phase 3: Renko Builder — K=0.00295 Only

```python
# bridge/renko.py — CRITICAL: only one formula is allowed
K_MULTIPLIER = 0.00295   # Validated: 5.9% spread burden, +0.747 expectancy

class RenkoBuilder:
    def __init__(self, day_open_price: float):
        self.brick_size  = day_open_price * K_MULTIPLIER
        self.start_price = day_open_price
        self.current_price = day_open_price
        self.uptrend = 0
        self.history = []
        self.sequence = ''
        self.brick_count = 0

    def update_brick_size(self, new_day_open: float):
        """Call on daily rollover. Does NOT affect open bricks."""
        old = self.brick_size
        self.brick_size = new_day_open * K_MULTIPLIER
        logging.info(f"Brick size updated: {old:.4f} → {self.brick_size:.4f}")
```

---

## Phase 4: Feature Engine Port

Direct copy of `src/feature_engine.py` → `bridge/feature_engine.py`. No algorithmic changes.

**Volume fallback — full implementation** (sourced from `Resources/implementation.md` + ablation study):

```python
def compute_vector(self, bid, ask, bid_vol, ask_vol, time_ms):
    mid = (bid + ask) / 2.0
    if self.prev_bid is None:
        self._init_prev(bid, ask, bid_vol, ask_vol, time_ms)
        return [0.0] * 9

    # Volume Fallback (FR-PY-FEAT-04)
    if bid_vol <= 0 or ask_vol <= 0:
        prev_mid = (self.prev_bid + self.prev_ask) / 2.0
        raw_ofi   = 1.0 if mid > prev_mid else (-1.0 if mid < prev_mid else 0.0)
        depth_raw = 0.0
        susc_raw  = 0.0
        # Note: z-score trackers receive 0.0 for depth/susc — handled by sigma guard
    else:
        dBid = bid - self.prev_bid
        dAsk = ask - self.prev_ask
        raw_ofi = (
            (1 if dBid >= 0 else 0) * bid_vol
          - (1 if dBid <= 0 else 0) * self.prev_bid_vol
          - (1 if dAsk <= 0 else 0) * ask_vol
          + (1 if dAsk >= 0 else 0) * self.prev_ask_vol
        )
        depth_raw = bid_vol + ask_vol
        susc_raw  = raw_ofi / (depth_raw + 1e-8)  # Raw division FIRST
    # ... rest of compute_vector unchanged
```

---

## Phase 5: Ensemble — Baiting REMOVED

```python
# bridge/ensemble.py
# **[UPDATED]** — Baiting strategy REMOVED per forensic audit

PROB_WIN_THRESHOLD = 0.5
PRED_OS_THRESHOLD  = 1.4   # Calibrated on K=0.00295 holdout
VOTE_THRESHOLD     = 2     # >= 2 of 3

class EnsemblePredictor:
    def predict(self, micro, macro):
        votes, details = 0, []
        for model in self.models:
            preds = model([micro, macro], training=False)
            pw = float(preds[0].numpy().flatten()[0])
            po = float(preds[1].numpy().flatten()[0])
            signal = (pw >= PROB_WIN_THRESHOLD) and (po >= PRED_OS_THRESHOLD)
            votes += int(signal)
            details.append({'prob_win': pw, 'pred_os': po, 'signal': signal})

        # Standard signal only — NO BAITING (action=-1 path removed)
        action = 1 if votes >= VOTE_THRESHOLD else 0
        return {'action': action, 'votes': votes, 'details': details}
        # action: 1=ENTER, 0=SKIP. -1 (REVERSE/BAIT) is NEVER returned.
```

---

## Phase 6: State Manager — Full Schema

```python
DEFAULT_STATE = {
    "schema_version":   2,
    "last_tick_msc":    0,
    "active_ticket":    0,         # 0 = no position
    "active_direction": 0,         # 1=BUY, -1=SELL
    "active_entry":     0.0,
    "active_sl":        0.0,
    "active_tp":        0.0,
    "active_brick_size":0.0,       # Brick size at time of entry
    "be_triggered":     False,
    "daily_pnl":        0.0,       # Points, reset on rollover
    "brick_count":      0,
    "session_date":     "",        # YYYY-MM-DD broker date
    "warmup_done":      False,
    "degraded_mode":    False
}
```

---

## Phase 7: BridgeEngine Main Loop

### Degraded Mode State Machine

```
NORMAL ──[no ticks 30s]──► DEGRADED ──[3 ticks in 5s]──► NORMAL
   │                            │
   │                     [10 reconnect fails]
   │                            │
   └────────────────────► EXIT (code 2)
```

### Main Loop with Latency Profiling

```python
def _process_tick(self, tick, is_warmup=False):
    t0 = time.perf_counter()

    feat_vec = self.feature_engine.compute_vector(
        tick['bid'], tick['ask'], tick['bid_vol'], tick['ask_vol'], tick['time_msc'])
    self.buffer.append_tick(feat_vec, self.renko.brick_count)

    new_bricks = self.renko.update_tick(tick['bid'], tick['time_msc'])

    for brick in new_bricks:
        self.feature_engine.on_new_brick(brick)
        tensors = self.buffer.on_brick_close(brick)
        if tensors and not is_warmup:
            t1 = time.perf_counter()
            self._on_signal(brick, tensors)
            t2 = time.perf_counter()
            inference_ms = (t2 - t1) * 1000
            if inference_ms > 150:
                logging.warning(f"SLOW INFERENCE: {inference_ms:.0f}ms (target <80ms)")

    # Break-even check on every tick
    if self.state.get('active_ticket') and not self.state.get('be_triggered'):
        self._check_be(tick)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    if elapsed_ms > 30:
        logging.warning(f"SLOW TICK PROCESSING: {elapsed_ms:.0f}ms")

def _check_be(self, tick):
    entry = self.state.get('active_entry')
    bs    = self.state.get('active_brick_size')
    direction = self.state.get('active_direction')
    trigger_dist = 0.3125 * bs    # 5/16 of brick_size

    if direction == 1 and tick['bid'] >= entry + trigger_dist:
        self._trigger_be(entry)
    elif direction == -1 and tick['ask'] <= entry - trigger_dist:
        self._trigger_be(entry)

def _trigger_be(self, entry_price):
    ticket = self.state.get('active_ticket')
    conf = self.sender.modify_sl(ticket, entry_price)
    if conf and conf['status'] == 'OK':
        self.state.update('be_triggered', True)
        self.state.update('active_sl', entry_price)
        logging.info(f"BREAK-EVEN triggered: SL moved to {entry_price:.5f}")
```

### Daily Rollover Handler

```python
def _on_day_open(self, new_price):
    """Called when TickReceiver.day_open_price changes (new DAYOPEN message)."""
    logging.info(f"ROLLOVER detected: new_day_open={new_price}")
    self.renko.update_brick_size(new_price)
    self.feature_engine.update_brick_size(new_price * self.renko.K_MULTIPLIER)
    self.state.update('session_date', str(datetime.date.today()))
    self.state.update('daily_pnl', 0.0)
    # DO NOT close open positions during rollover
    logging.info(f"ROLLOVER complete: new brick_size={self.renko.brick_size:.4f}")
```

---

## **[NEW]** Phase 8: Latency Profiling (`tests/test_latency.py`)

```python
"""
Run during Phase 7 verification to measure actual latency on target Mac.
"""
def profile_inference_latency(model_dir, n_samples=100):
    ensemble = EnsemblePredictor(model_dir)
    ensemble.load()
    
    micro = np.random.randn(1, 10, 100, 9).astype(np.float32)
    macro = np.random.randn(1, 10, 3).astype(np.float32)
    
    # Warmup
    for _ in range(5): ensemble.predict(micro, macro)
    
    # Measure
    latencies = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        ensemble.predict(micro, macro)
        latencies.append((time.perf_counter() - t0) * 1000)
    
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    print(f"Inference latency — p50: {p50:.1f}ms  p95: {p95:.1f}ms  p99: {p99:.1f}ms")
    assert p95 < 150, f"FAIL: p95 inference latency {p95:.1f}ms > 150ms target"
    return {'p50': p50, 'p95': p95, 'p99': p99}
```

---

## Critical Invariants

| # | Rule | Verification |
|---|---|---|
| 1 | `brick_size = day_open * 0.00295` only | Assert K constant in renko.py |
| 2 | Feature engine processes EVERY tick | Assert tick_count == z_ofi deque length |
| 3 | Micro-buffer NEVER resets at brick boundaries | Assert buffer has multi-brick IDs |
| 4 | Susceptibility: raw division FIRST, then z-score | Code review: `susc_raw = ofi/(depth+1e-8)` |
| 5 | OFI uses weak inequalities (`>=`, `<=`) | Code review |
| 6 | Z-score window = 1000, warmup = 30 | Assert `self.window == 1000` |
| 7 | Renko uses bid price | Assert `update_tick(tick['bid'], ...)` |
| 8 | Models called with `training=False` | Code review |
| 9 | No trades during warmup | Assert order_count == 0 after warmup |
| 10 | State saved after every trade event | Assert `state.save()` in all state-change paths |
| 11 | `Pred_OS >= 1.4` threshold | Assert constant in ensemble.py |
| 12 | `action=-1` (baiting) NEVER returned | Assert no code path returns -1 |
| 13 | CONFIRM timeout = 5s | Assert `CONFIRM_TIMEOUT = 5.0` in command_sender.py |
| 14 | No `import MetaTrader5` in bridge/ | `grep -r "import MetaTrader5" bridge/` returns empty |
