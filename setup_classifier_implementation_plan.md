# SetupClassifier — Complete Implementation Plan
## Multi-Timeframe Price Action Trade Setup Classifier for BrickOfTicks

> **System Codename**: `SetupClassifier`
> **Objective**: After each Renko brick closes, analyse the price action *within* that brick using adaptive multi-timeframe candle representations, and output a probability distribution over 5 classes: T1 (1:1 Continuation), T2 (1:2 Pullback), T3 (1:2 Reversal), T4 (1:3 Deep Reversal), T0 (No Trade).
> **Foundation**: Tick data only. No volume. No microstructure. Broker-agnostic by design.

---

## Table of Contents

1. [System Overview & Design Decisions](#1-system-overview--design-decisions)
2. [Trade Geometry — Exact Definitions](#2-trade-geometry--exact-definitions)
3. [The Core Problem — Variable Brick Duration](#3-the-core-problem--variable-brick-duration)
4. [Data Foundation](#4-data-foundation)
5. [Phase 1 — Label Generation](#5-phase-1--label-generation)
6. [Phase 2 — Feature Engineering](#6-phase-2--feature-engineering)
7. [Phase 3 — The Adaptive Multi-Timeframe Candle Representation](#7-phase-3--the-adaptive-multi-timeframe-candle-representation)
8. [Phase 4 — Tensor Construction](#8-phase-4--tensor-construction)
9. [Phase 5 — Model Architecture](#9-phase-5--model-architecture)
10. [Phase 6 — Training](#10-phase-6--training)
11. [Phase 7 — Calibration & Threshold Sweep](#11-phase-7--calibration--threshold-sweep)
12. [Phase 8 — Evaluation](#12-phase-8--evaluation)
13. [File Map & Artifacts](#13-file-map--artifacts)
14. [Research Foundation](#14-research-foundation)
15. [Critical Invariants](#15-critical-invariants)

---

## 1. System Overview & Design Decisions

### 1.1 Why a Probability Distribution (Not a Single Recommendation)

The model outputs a **5-class softmax probability vector** `[P(T0), P(T1), P(T2), P(T3), P(T4)]`, not a single winning class. This is the correct design for three reasons:

1. Multiple setups can be simultaneously valid. A brick with a perfect pullback structure may also show reversal pressure — surfacing both probabilities lets the trader or downstream system make a risk-weighted decision.
2. The "no trade" class (T0) acts as a learned uncertainty absorber. Without it, the model is forced to assign a trade type to every brick regardless of pattern quality.
3. The probability vector enables confidence-gating downstream: only execute T3 if P(T3) > 0.65 AND P(T0) < 0.15, for example.

### 1.2 Why This System Is Independent of BrickOfTicks

BrickOfTicks predicts whether a brick's momentum will continue for one more brick (binary WIN/LOSS). SetupClassifier predicts *which structural setup* the current price action supports across multiple RR profiles. They answer orthogonal questions and can be stacked: BrickOfTicks filters for valid momentum, SetupClassifier selects the best RR structure for execution.

### 1.3 Broker Agnosticism Strategy

Since volume and microstructure features are broker-specific, all features are derived exclusively from:
- **Bid price only** (the execution price, consistent across all brokers for the retail L1 environment)
- **Timestamp** (for candle construction and duration)
- **Ask price** (for spread as a regime quality signal, but no microstructure ratios)

This makes the feature space transferable across demo → live → different broker environments.

---

## 2. Trade Geometry — Exact Definitions

Let `K` = brick_size, `C` = brick close price, `O` = brick open price, `D` = direction (+1 = bullish, -1 = bearish).

### Type 1 — 1:1 Continuation (Market Order at Close)
```
Entry:  C  (market order at brick close)
TP:     C + D * K
SL:     C - D * K
RR:     1:1
```
WIN condition: bid hits TP before SL
y_mag range: [1.0, ∞)

### Type 2 — 1:2 Pullback Continuation (Limit Order at Brick Open)
```
Entry:  O  (limit order at the OPEN of the just-closed brick)
TP:     O + D * 2K  (two bricks beyond entry in continuation direction)
SL:     O - D * K   (one brick against, i.e., below entry for LONG)
RR:     1:2
```
WIN condition: bid/ask reaches TP after entry fills, before SL
Note: The limit may not fill. Unfilled limits count as "no trade taken" and are excluded from the label set for T2.

### Type 3 — 1:2 Standard Reversal (Market Order at Close)
```
Entry:  C  (market order at brick close)
TP:     C - D * 2K  (two bricks in REVERSE direction)
SL:     C + D * K   (one brick in the original direction)
RR:     1:2
```
WIN condition: a reversal brick closes AND price reached two brick sizes in the reverse direction before hitting SL. TP triggers when price has moved 2K in reverse from entry (not necessarily when a reversal brick *closes*, but when the price *level* 2K away is touched).

### Type 4 — 1:3 Deep Reversal (Market Order at Close, Reverse Target)
```
Entry:  C  (market order at brick close)
TP:     C - D * 3K  (three bricks in REVERSE direction from entry)
SL:     C + D * K   (one brick in the original direction)
RR:     1:3
```
WIN condition: price touches TP level (3K in reverse from entry) before SL.
Note: This is the deepest, rarest setup. Expect the fewest labels.

### Trade Geometry Summary Table
| Type | Entry | TP Distance | SL Distance | RR | Direction |
|------|-------|------------|------------|-----|-----------|
| T1   | Market @ Close | 1K continuation | 1K reversal | 1:1 | Same as brick |
| T2   | Limit @ Open | 2K continuation | 1K reversal | 1:2 | Same as brick |
| T3   | Market @ Close | 2K reversal | 1K continuation | 1:2 | Opposite brick |
| T4   | Market @ Close | 3K reversal | 1K continuation | 1:3 | Opposite brick |

---

## 3. The Core Problem — Variable Brick Duration

This is the central engineering challenge of the entire system. A single brick can take 1 minute or 100 minutes. Fixed timeframe candle analysis (e.g., always use M1 + M5 + M15) breaks when:
- A 90-minute brick contains 90 M1 candles but a 2-minute brick contains only 2.
- Short bricks have almost no M15 candles, making the feature vector degenerate.
- The model cannot learn consistent patterns if the same "number of candles" represents wildly different time spans.

### Solution: Adaptive Normalised Candle Segments (ANCS)

Instead of fixed timeframes, we slice the intra-brick tick sequence into **N equal-proportion segments** based on the brick's own duration. This is the key innovation.

**Definition**: Given a brick of duration T seconds and N_ticks total ticks, divide the ticks into S segments of equal *tick count* (not equal time). Within each segment, compute (O, H, L, C, duration_fraction) to form a mini-candle.

Why tick-count segments rather than time segments:
- Tick count is proportional to market activity within the brick, not wall-clock time.
- A slow overnight brick with 50 ticks in 90 minutes and a fast session brick with 50 ticks in 2 minutes have the same ANCS representation.
- The model learns from the *shape* of the price action within the brick, not its absolute duration.

We use S = 10 segments (10 mini-candles per brick). This is the "micro" representation.

We additionally compute S = 5 segments for a "macro view" of the same brick (coarser granularity).

And we look back at the last 5 completed bricks' ANCS representations for context.

This gives us a true multi-resolution, multi-timeframe view without any fixed-timeframe dependency.

---

## 4. Data Foundation

### 4.1 Input Data (Already Available)
- L1 tick data: `Data/Raw/Ticks/{year}/{month}/{day}.parquet`
- Schema: `timestamp` (ms UTC), `bid`, `ask`, `bid_vol` (ignored), `ask_vol` (ignored)
- Renko bricks: constructed fresh via `RenkoBuilder` at K=0.00118 (same as BrickOfTicks)
- Coverage: 2020–2024 (5 years, ~30,000+ bricks)

### 4.2 What We Need to Compute
For each Renko brick:
1. All ticks from the previous brick close to this brick close → the "intra-brick tick sequence"
2. The ANCS representation (10-segment and 5-segment micro-candles)
3. Price action features derived from those segments (see Phase 2)
4. The label: which of T0/T1/T2/T3/T4 actually worked out for this brick (see Phase 1)

---

## 5. Phase 1 — Label Generation

**Script**: `src/sc_label_generator.py`
**Output**: `outputs/setup_classifier/labels.parquet`

### 5.1 Algorithm

For every Renko brick, we simultaneously simulate all four trade types using the exact bid/ask execution model. Each trade type gets its own label independently.

```
For each brick i:
    C = brick.close
    O = brick.open
    K = brick.brick_size
    D = +1 if brick.uptrend else -1

    # Simulate T1: Market @ C, TP = C + D*K, SL = C - D*K
    t1_result = scan_ticks(entry=C, tp=C + D*K, sl=C - D*K,
                           entry_type='market', future_ticks)

    # Simulate T2: Limit @ O, TP = O + D*2K, SL = O - D*K
    t2_result = scan_ticks(entry=O, tp=O + D*2*K, sl=O - D*K,
                           entry_type='limit', future_ticks)

    # Simulate T3: Market @ C, TP = C - D*2K, SL = C + D*K
    t3_result = scan_ticks(entry=C, tp=C - D*2*K, sl=C + D*K,
                           entry_type='market', future_ticks)

    # Simulate T4: Limit @ O, TP = O - D*3K, SL = O + D*K
    t4_result = scan_ticks(entry=O, tp=O - D*3*K, sl=O + D*K,
                           entry_type='limit', future_ticks)
```

### 5.2 The `scan_ticks` Function

```python
def scan_ticks(entry, tp, sl, entry_type, future_ticks, direction):
    """
    Returns: (win: bool, y_mag: float, filled: bool)
    
    For limit orders: scan until entry is touched (fill) or opposite extreme 
    (never filled → filled=False, skip label).
    For market orders: filled=True always.
    
    Exit scanning uses bid for LONG TP/SL, ask for SHORT TP/SL.
    """
    filled = (entry_type == 'market')
    
    for tick in future_ticks:
        scan_price = tick['bid'] if direction == 1 else tick['ask']
        
        # Check limit fill (for T2 and T4)
        if not filled:
            fill_price = tick['ask'] if direction == 1 else tick['bid']
            if direction == 1 and fill_price <= entry:
                filled = True
            elif direction == -1 and fill_price >= entry:
                filled = True
            # If we've gone past SL before filling, order never fills
            if direction == 1 and scan_price <= sl:
                return (False, 0.0, False)  # Never filled
            if direction == -1 and scan_price >= sl:
                return (False, 0.0, False)
            continue
        
        # Exit scanning (filled)
        if direction == 1:
            if scan_price >= tp:
                return (True, abs(tp - entry) / brick_size, True)
            if scan_price <= sl:
                return (False, abs(scan_price - entry) / brick_size, True)
        else:
            if scan_price <= tp:
                return (True, abs(tp - entry) / brick_size, True)
            if scan_price >= sl:
                return (False, abs(scan_price - entry) / brick_size, True)
    
    return (False, 0.0, True)  # Force-closed
```

### 5.3 Multi-Label to Single Class Assignment

Each brick now has four binary outcomes: (t1_win, t2_win, t3_win, t4_win). We need a single 5-class label.

**Assignment rule** (priority-ordered to resolve conflicts):
```
if t2_win AND t2_filled:    label = 2   # Highest RR continuation
elif t4_win AND t4_filled:  label = 4   # Highest RR reversal
elif t1_win:                label = 1   # Standard continuation
elif t3_win:                label = 3   # Standard reversal
else:                       label = 0   # No setup worked
```

Priority reasoning: Higher-RR setups are prioritised because they represent a stronger structural signal. If a T2 wins, a T1 would also have won (the limit is above the market order entry for LONGs) — but the better setup was T2.

**Also save all four binary labels separately** in the parquet for ablation studies.

### 5.4 Expected Label Distribution
Based on the structural base rates:
- T0 (no trade): ~30–40% (majority class — this is fine, T0 is genuinely the most common outcome)
- T1: ~25–35% (1:1 at 50% base rate, model will filter)
- T2: ~15–20% (pullback must occur first)
- T3: ~10–15% (reversal setups rarer)
- T4: ~5–10% (deep reversal, rarest)

### 5.5 Output Schema
```
brick_id, date, direction, C, O, K,
t1_win, t1_y_mag,
t2_win, t2_y_mag, t2_filled,
t3_win, t3_y_mag,
t4_win, t4_y_mag, t4_filled,
label (0-4),
exclude_flag, brick_duration_seconds
```

---

## 6. Phase 2 — Feature Engineering

**Script**: `src/sc_feature_engine.py`
**Output**: `outputs/setup_classifier/features/`

This phase computes the raw feature vectors. All features are derived from bid/ask prices and timestamps only — zero volume, zero microstructure ratios.

### 6.1 The Adaptive Normalised Candle Segments (ANCS)

For the ticks within a brick (from previous brick close to this brick close):

```python
def compute_ancs(intra_brick_ticks, n_segments=10):
    """
    Split tick sequence into n_segments equal-tick-count segments.
    For each segment compute: O, H, L, C, duration_fraction, tick_count_fraction
    Returns: (n_segments, 6) array
    """
    N = len(intra_brick_ticks)
    seg_size = max(1, N // n_segments)
    
    segments = []
    brick_open_price = intra_brick_ticks[0]['bid']
    brick_start_time = intra_brick_ticks[0]['time_msc']
    brick_end_time = intra_brick_ticks[-1]['time_msc']
    brick_duration = max(1, brick_end_time - brick_start_time)
    
    for i in range(n_segments):
        start = i * seg_size
        end = min((i + 1) * seg_size, N) if i < n_segments - 1 else N
        seg_ticks = intra_brick_ticks[start:end]
        
        if len(seg_ticks) == 0:
            segments.append([0, 0, 0, 0, 0, 0])
            continue
        
        prices = [t['bid'] for t in seg_ticks]
        seg_open = prices[0]
        seg_high = max(prices)
        seg_low = min(prices)
        seg_close = prices[-1]
        
        # Normalise all prices by brick_size (scale-invariant)
        seg_o = (seg_open - brick_open_price) / brick_size
        seg_h = (seg_high - brick_open_price) / brick_size
        seg_l = (seg_low - brick_open_price) / brick_size
        seg_c = (seg_close - brick_open_price) / brick_size
        
        # Temporal position of this segment within the brick
        seg_time_start = seg_ticks[0]['time_msc']
        seg_time_end = seg_ticks[-1]['time_msc']
        duration_frac = (seg_time_end - brick_start_time) / brick_duration
        
        # Tick density of this segment (are ticks clustered here?)
        tick_frac = len(seg_ticks) / N
        
        segments.append([seg_o, seg_h, seg_l, seg_c, duration_frac, tick_frac])
    
    return np.array(segments, dtype=np.float32)  # (10, 6)
```

### 6.2 Classical Candlestick Pattern Features (Rule-Based, Broker-Agnostic)

Computed from the full intra-brick price sequence (not segments):

```python
def compute_candle_features(intra_brick_ticks, brick_size, direction):
    """
    Returns a 15-element vector of normalised candle shape features.
    All features are normalised by brick_size — fully scale-invariant.
    """
    prices = [t['bid'] for t in intra_brick_ticks]
    O = prices[0]
    H = max(prices)
    L = min(prices)
    C = prices[-1]
    
    body = abs(C - O)
    upper_wick = H - max(O, C)
    lower_wick = min(O, C) - L
    full_range = H - L
    
    # Normalise by brick_size throughout
    features = [
        body / brick_size,                    # Body size
        upper_wick / brick_size,              # Upper wick
        lower_wick / brick_size,              # Lower wick
        full_range / brick_size,              # Full range (volatility of brick)
        (C - O) / brick_size,                 # Signed body (direction of candle)
        upper_wick / (full_range + 1e-8),     # Upper wick ratio (pin bar signal)
        lower_wick / (full_range + 1e-8),     # Lower wick ratio
        body / (full_range + 1e-8),           # Body ratio (indecision signal)
        
        # Where did price close within its range? [0=bottom, 1=top]
        (C - L) / (full_range + 1e-8),
        
        # Where did price open within its range?
        (O - L) / (full_range + 1e-8),
        
        # Momentum: did price finish above or below midpoint?
        1.0 if C > (H + L) / 2 else 0.0,
        
        # Rejection: did price test the direction extreme and pull back?
        (H - C) / brick_size if direction == 1 else (C - L) / brick_size,
        
        # Engulf signal: did intra-brick range exceed the brick_size?
        min(full_range / brick_size, 3.0),
        
        # Open proximity to extremes (measures if candle gapped)
        abs(O - L) / brick_size,
        abs(H - O) / brick_size,
    ]
    
    return np.array(features, dtype=np.float32)  # (15,)
```

### 6.3 Momentum & Structure Features

```python
def compute_momentum_features(intra_brick_ticks, brick_size):
    """
    Time-aware momentum features. 25-element vector.
    """
    prices = [t['bid'] for t in intra_brick_ticks]
    times = [t['time_msc'] for t in intra_brick_ticks]
    N = len(prices)
    
    # Split into thirds: early / mid / late phase of the brick
    t1 = prices[:N//3]
    t2 = prices[N//3:2*N//3]
    t3 = prices[2*N//3:]
    
    def phase_stats(phase):
        if len(phase) < 2:
            return [0, 0, 0, 0]
        return [
            (phase[-1] - phase[0]) / brick_size,   # Net move
            max(phase) / brick_size,                # Relative high
            min(phase) / brick_size,                # Relative low
            np.std(phase) / brick_size,             # Volatility
        ]
    
    early = phase_stats(t1)
    mid = phase_stats(t2)
    late = phase_stats(t3)
    
    # Acceleration: did momentum increase or decrease through the brick?
    early_move = abs(early[0])
    late_move = abs(late[0])
    acceleration = (late_move - early_move)  # Positive = accelerating
    
    # Tick velocity profile (are ticks arriving faster at end?)
    if N > 10:
        dt_early = np.mean(np.diff(times[:N//3])) if N//3 > 1 else 1000
        dt_late = np.mean(np.diff(times[2*N//3:])) if N - 2*N//3 > 1 else 1000
        velocity_ratio = dt_early / (dt_late + 1e-3)  # >1 = speeding up
    else:
        velocity_ratio = 1.0
    
    # Spread dynamics (closing spread vs opening spread)
    spreads = [t['ask'] - t['bid'] for t in intra_brick_ticks]
    spread_open = np.mean(spreads[:max(1, N//10)])
    spread_close = np.mean(spreads[max(0, 9*N//10):])
    spread_trend = (spread_close - spread_open) / (spread_open + 1e-8)  # >0 = widening
    
    # Price path complexity: how much did price oscillate?
    direction_changes = sum(
        1 for i in range(1, len(prices)-1)
        if (prices[i] - prices[i-1]) * (prices[i+1] - prices[i]) < 0
    )
    path_complexity = direction_changes / max(1, N)
    
    # Log duration (already proven useful in BrickOfTicks)
    duration_ms = times[-1] - times[0]
    log_duration = np.log1p(duration_ms / 1000)  # log(seconds + 1)
    
    features = (early + mid + late +
                [acceleration, velocity_ratio, spread_trend, 
                 path_complexity, log_duration,
                 spread_open / brick_size,
                 spread_close / brick_size,
                 len(prices) / 100.0])  # Tick count (normalised)
    
    return np.array(features[:25], dtype=np.float32)  # (25,)
```

### 6.4 Inter-Brick Context Features (Last 5 Bricks)

```python
def compute_context_features(last_5_bricks):
    """
    Structural context from the last 5 completed bricks.
    15-element vector.
    """
    if len(last_5_bricks) < 5:
        return np.zeros(15, dtype=np.float32)
    
    directions = [b['direction'] for b in last_5_bricks]
    durations = [b['duration_seconds'] for b in last_5_bricks]
    sizes = [b['brick_size'] for b in last_5_bricks]
    
    # Trend consistency: how many of last 5 are same direction?
    same_dir = sum(1 for d in directions if d == directions[-1])
    trend_consistency = same_dir / 5.0  # [0.2, 1.0]
    
    # Duration trend: is the market speeding up or slowing down?
    dur_trend = (durations[-1] - durations[0]) / (durations[0] + 1)
    
    # Shannon entropy of direction sequence (from EXP-08 research)
    p_up = sum(1 for d in directions if d == 1) / 5.0
    if p_up in (0, 1):
        entropy = 0.0
    else:
        entropy = -(p_up * np.log2(p_up) + (1-p_up) * np.log2(1-p_up))
    
    # Alternation rate (reversal frequency)
    alternations = sum(
        1 for i in range(1, 5) if directions[i] != directions[i-1]
    )
    alternation_rate = alternations / 4.0
    
    # Brick size stability
    size_cv = np.std(sizes) / (np.mean(sizes) + 1e-8)
    
    return np.array([
        trend_consistency,
        dur_trend,
        entropy,
        alternation_rate,
        size_cv,
        *[float(d) for d in directions],          # 5 direction values
        *[np.log1p(dur) for dur in durations[-3:]] # Last 3 log-durations
    ], dtype=np.float32)  # (15,)
```

### 6.5 Total Feature Vector Per Brick

| Component | Shape | Description |
|-----------|-------|-------------|
| ANCS Fine (10 segments) | (10, 6) | 10 mini-candles at equal-tick resolution |
| ANCS Coarse (5 segments) | (5, 6) | 5 mini-candles at coarser resolution |
| Candle shape features | (15,) | Normalised body/wick/range structure |
| Momentum features | (25,) | Phase momentum, spread dynamics, complexity |
| Context features | (15,) | Last 5 bricks structural context |

---

## 7. Phase 3 — The Adaptive Multi-Timeframe Candle Representation

**This phase is the intellectual core of the system.**

### 7.1 The Three-View Representation

For every brick, we construct three parallel views:

**View A — Fine ANCS (10 segments)**: captures the granular tick-by-tick price action structure. Equivalent to a "M1 candle sequence" for a long brick or a "tick chart" for a short brick. This is where pin bars, engulfing patterns, and momentum breaks are visible.

**View B — Coarse ANCS (5 segments)**: captures the structural shape of the brick at a higher level. Equivalent to a "M5 or M15 view". This is where the overall directional bias and major reversal structures are visible.

**View C — History Context**: the last 5 bricks' coarse ANCS stacked. Equivalent to a "H1 or H4 context view". This is where trend structure, swing highs/lows, and multi-brick patterns (double tops, ranges) are visible.

This three-view design is inspired by the ConvLSTM2D spatiotemporal approach (Springer, 2025) and the multi-timeframe prediction approach (Medium, 2026) but adapted to be duration-invariant through the tick-proportion normalisation.

### 7.2 Why Not Fixed Timeframes

The Springer ConvLSTM2D paper uses fixed 1-hour bars. The PeerJ CNN paper uses 15-minute bars. Both fail for our use case because:
- Our bricks take 1–100 minutes — a 15-minute candle captures 100% of a short brick but only 15% of a long brick.
- The feature vector would have wildly different "completeness" across bricks.
- The model would learn a spurious correlation between brick duration and candle count.

The ANCS approach solves this by making the representation duration-invariant by construction.

### 7.3 The Gramian Angular Field Option (Future Extension)

The ResearchGate CNN-LSTM paper (2022) achieved 90–93% accuracy using Gramian Angular Field (GAF) image encoding of candlestick patterns. For a future version, the ANCS sequences can be encoded as GAF images and processed by a 2D CNN. This is flagged as a Phase 2 research extension, not part of the initial implementation.

---

## 8. Phase 4 — Tensor Construction

**Script**: `src/sc_tensor_builder.py`
**Output**: `outputs/setup_classifier/tensors/`

### 8.1 Input Tensors per Sample

For each brick i (with i ≥ 6, so context is available):

```python
{
    # Primary feature inputs
    'ancs_fine':    np.array shape (10, 6),   # View A
    'ancs_coarse':  np.array shape (5, 6),    # View B
    'history':      np.array shape (5, 5, 6), # View C: last 5 bricks' coarse ANCS
    
    # Scalar feature inputs (concatenated into one vector)
    'scalars':      np.array shape (55,),     # candle(15) + momentum(25) + context(15)
    
    # Labels
    'label':        int in {0, 1, 2, 3, 4},
    'label_t1':     float (0 or 1),           # Raw binary for ablation
    'label_t2':     float (0 or 1),
    'label_t3':     float (0 or 1),
    'label_t4':     float (0 or 1),
}
```

### 8.2 Walk-Forward Splits

| Split | Date Range | Purpose |
|-------|-----------|---------|
| Train | Jan 2020 – Dec 2022 | Model fitting |
| Val | Jan 2023 – Jun 2023 | Calibration, hyperparameter selection |
| Test | Jul 2023 – Dec 2023 | OOS evaluation |
| Holdout | Jan 2024 – Dec 2024 | Final pristine validation |

### 8.3 Class Weighting

Given the expected imbalance (T0 ~35%, T4 ~7%), compute class weights:
```python
class_weights = {
    0: total / (5 * count_T0),
    1: total / (5 * count_T1),
    2: total / (5 * count_T2),
    3: total / (5 * count_T3),
    4: total / (5 * count_T4),
}
```
Apply as sample weights during training, not as loss modification.

---

## 9. Phase 5 — Model Architecture

**Script**: `src/sc_model.py`

### 9.1 Architecture Overview

The model has three parallel encoding branches that process the three views, then fuses them into a 5-class softmax head.

```
┌─────────────────────────────────────────────────────────┐
│                   SetupClassifier                        │
├──────────────┬──────────────┬────────────────────────────┤
│  Branch A    │  Branch B    │  Branch C                  │
│  Fine ANCS   │  Coarse ANCS │  History Context           │
│  (10, 6)     │  (5, 6)      │  (5, 5, 6)                 │
│              │              │                            │
│  Conv1D(32)  │  Conv1D(16)  │  TimeDistributed           │
│  Conv1D(32)  │              │  Conv1D(16)                │
│  Flatten     │  Flatten     │  LSTM(16)                  │
│  Dense(32)   │  Dense(16)   │                            │
│  (32,)       │  (16,)       │  (16,)                     │
└──────┬───────┴──────┬───────┴──────┬─────────────────────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
              Concatenate (64,)
                      │
              + Scalars (55,)
                      │
              Concatenate (119,)
                      │
              Dense(128, relu, L2)
              Dropout(0.3)
              Dense(64, relu, L2)
              Dropout(0.2)
                      │
              Dense(5, softmax)  ← P(T0), P(T1), P(T2), P(T3), P(T4)
```

### 9.2 Full Architecture Code

```python
def build_setup_classifier():
    
    # ── Inputs ──────────────────────────────────────────────────
    ancs_fine_input    = Input(shape=(10, 6), name='ancs_fine')
    ancs_coarse_input  = Input(shape=(5, 6),  name='ancs_coarse')
    history_input      = Input(shape=(5, 5, 6), name='history')
    scalar_input       = Input(shape=(55,),   name='scalars')
    
    # ── Branch A: Fine ANCS ─────────────────────────────────────
    # Two Conv1D layers to capture both local (single-segment) 
    # and short-range (3-segment) patterns
    a = Conv1D(32, kernel_size=1, activation='relu',
               padding='causal', kernel_regularizer=l2(1e-4))(ancs_fine_input)
    a = Conv1D(32, kernel_size=3, activation='relu',
               padding='causal', kernel_regularizer=l2(1e-4))(a)
    a = Flatten()(a)
    a = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(a)
    a = Dropout(0.3)(a)
    
    # ── Branch B: Coarse ANCS ───────────────────────────────────
    b = Conv1D(16, kernel_size=1, activation='relu',
               padding='causal', kernel_regularizer=l2(1e-4))(ancs_coarse_input)
    b = Flatten()(b)
    b = Dense(16, activation='relu', kernel_regularizer=l2(1e-4))(b)
    b = Dropout(0.3)(b)
    
    # ── Branch C: History Context ────────────────────────────────
    # TimeDistributed CNN over 5 historical bricks
    c = TimeDistributed(
        Conv1D(16, kernel_size=1, activation='relu', padding='causal')
    )(history_input)  # (batch, 5, 5, 16)
    c = TimeDistributed(Flatten())(c)    # (batch, 5, 80)
    c = LSTM(16, return_sequences=False,
             kernel_regularizer=l2(1e-4))(c)
    c = Dropout(0.2)(c)
    
    # ── Fusion ───────────────────────────────────────────────────
    fused = Concatenate()([a, b, c])         # (64,)
    fused = Concatenate()([fused, scalar_input])  # (119,)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(fused)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.2)(x)
    
    # ── Output: 5-class softmax ──────────────────────────────────
    output = Dense(5, activation='softmax', name='setup_probs')(x)
    
    model = Model(
        inputs=[ancs_fine_input, ancs_coarse_input, history_input, scalar_input],
        outputs=output,
        name='setup_classifier'
    )
    return model


def compile_setup_classifier(model):
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc')]
    )
    return model
```

### 9.3 Why This Architecture

- **Separate branches for each view**: the three resolution levels contain structurally different information. Mixing them before encoding would lose the resolution-specific patterns.
- **Conv1D (not LSTM) for intra-brick views**: within a single brick, the "sequence" of 10 or 5 ANCS segments is short enough that CNN filters capture the relevant local patterns (pin bar = one segment with extreme wick, engulfing = body of segment 10 > body of segment 1). LSTM would be overkill for 10 time steps and would increase training time.
- **LSTM for history branch**: the sequence of 5 historical bricks *is* a temporal sequence in the standard sense — trend structure requires remembering what happened 3 bricks ago in context of 1 brick ago. LSTM is appropriate here.
- **No TimeDistributed submodel**: avoids the macOS Metal deadlock (learned from BrickOfTicks training bug). All TimeDistributed wrappers use plain layers.
- **Top-2 accuracy metric**: since T2 wins imply T1 wins (and T4 wins imply T3 wins), top-2 accuracy is a more meaningful training signal than top-1 accuracy alone.

---

## 10. Phase 6 — Training

**Script**: `src/sc_train.py`

### 10.1 Training Configuration

```python
BATCH_SIZE = 64
MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20
LR_REDUCE_PATIENCE = 10
LR_REDUCE_FACTOR = 0.5
MIN_LR = 1e-6
```

### 10.2 Loss Function

`sparse_categorical_crossentropy` with class weights applied as sample_weight. **Do not use label smoothing** — the labels are programmatically derived (not human-annotated) and are exact. Smoothing would introduce artificial uncertainty.

### 10.3 Callbacks

```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
    ModelCheckpoint(filepath=MODEL_PATH, monitor='val_loss', save_best_only=True),
    CSVLogger(str(LOG_PATH)),
]
```

### 10.4 Expected Training Signal Check (Before Full Training)

Before investing in full training, run a signal existence check:

```python
# Train a logistic regression baseline on the scalar features alone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_scalars, y_train)
report = classification_report(y_val, lr.predict(X_val_scalars))
print(report)
```

**GO condition**: macro F1 > 0.25 (above random 0.20 for 5 classes)
**ABORT condition**: macro F1 < 0.22 — signals the features carry no discriminative information, root-cause before training the neural network.

---

## 11. Phase 7 — Calibration & Threshold Sweep

**Script**: `src/sc_calibrate.py`

### 11.1 Per-Class Confidence Thresholding

After training, sweep a confidence threshold for each class independently:

```python
for setup_class in [1, 2, 3, 4]:
    for theta in np.arange(0.3, 0.9, 0.05):
        mask = (predicted_class == setup_class) & (prob[:, setup_class] >= theta)
        win_rate = y_true[mask] == setup_class
        n_trades = mask.sum()
        ev = compute_ev(setup_class, win_rate, n_trades)
```

Where `compute_ev` accounts for the RR of each setup:
```
T1 EV = WR * 1.0 - (1 - WR) * 1.0    (1:1)
T2 EV = WR * 2.0 - (1 - WR) * 1.0    (1:2)
T3 EV = WR * 2.0 - (1 - WR) * 1.0    (1:2)
T4 EV = WR * 3.0 - (1 - WR) * 1.0    (1:3)
```

The optimal threshold for each class maximises `EV × sqrt(n_trades)` (balances precision and frequency).

### 11.2 Output Config

```json
{
  "T1_threshold": 0.60,
  "T2_threshold": 0.55,
  "T3_threshold": 0.65,
  "T4_threshold": 0.70,
  "T0_veto_threshold": 0.40
}
```

The T0 veto: if P(T0) > 0.40, do not trade regardless of which other class has the highest probability.

---

## 12. Phase 8 — Evaluation

**Script**: `src/sc_evaluate.py`

### 12.1 Metrics to Report

| Metric | Description |
|--------|-------------|
| Per-class precision | Of all T2 predictions, what fraction were actual T2 wins? |
| Per-class recall | Of all true T2 setups, what fraction did the model catch? |
| Per-class EV | `precision * RR - (1 - precision) * 1.0` for each setup type |
| Combined EV | Total R earned per brick observed |
| Top-2 accuracy | Is the correct class in the top-2 predictions? |
| Confusion matrix | Full 5×5 matrix across all setup types |
| Monthly breakdown | Per-month WR and EV for regime stability analysis |

### 12.2 Acceptance Criteria

| Setup | Min Precision | Min Trades (6-month OOS) |
|-------|--------------|--------------------------|
| T1 | 55% | 100 |
| T2 | 45% | 50 |
| T3 | 45% | 50 |
| T4 | 38% | 20 |

Note: Lower thresholds for T2/T3/T4 because their higher RR means positive EV at lower win rates. T2 at 45% WR is: 0.45×2 - 0.55×1 = +0.35R per trade. T4 at 38% WR is: 0.38×3 - 0.62×1 = +0.52R per trade.

---

## 13. File Map & Artifacts

```
SetupClassifier/
├── src/
│   ├── sc_label_generator.py       # Phase 1: Multi-type label scanning
│   ├── sc_feature_engine.py        # Phase 2: ANCS + candle + momentum features
│   ├── sc_tensor_builder.py        # Phase 4: Walk-forward tensor compilation
│   ├── sc_model.py                 # Phase 5: Three-branch architecture
│   ├── sc_train.py                 # Phase 6: Training loop
│   ├── sc_calibrate.py             # Phase 7: Per-class threshold sweep
│   └── sc_evaluate.py             # Phase 8: OOS evaluation
│
├── outputs/setup_classifier/
│   ├── labels.parquet              # All 4 binary labels + 5-class label
│   ├── features/                   # Per-brick ANCS + scalar vectors
│   ├── tensors/                    # Train/val/test/holdout tensor splits
│   ├── model.keras                 # Trained SetupClassifier
│   ├── config.json                 # Calibrated per-class thresholds
│   ├── training_log.csv            # Epoch-by-epoch loss history
│   └── evaluation_report.json     # Full OOS metrics
```

---

## 14. Research Foundation

This design synthesises findings from the following work:

| Research | Finding Applied |
|----------|----------------|
| Springer ConvLSTM2D (2025) | Spatiotemporal dependencies need separate encoding paths before fusion |
| PeerJ CNN on 15m candles (2025) | CNN effectively captures candlestick pattern shape; sliding window approach validated |
| ResearchGate GAF-CNN (2022) | 90–93% pattern recognition using image encoding (flagged for Phase 2 upgrade) |
| arXiv LSTM pattern detection (2018) | LSTM outperforms CNN for sequence-level pattern generalisation |
| ScienceDirect info-driven bars (2025) | Volume/dollar/range bars outperform fixed time bars — validates our tick-proportion ANCS approach |
| arXiv multi-timeframe crypto (2025) | Confidence scoring + inter-network agreement → position sizing. Directly informs our T0 veto + per-class threshold design |

---

## 15. Critical Invariants

These rules must never be violated:

1. **No lookahead in features**: the ANCS for brick i uses only ticks from brick i's open to brick i's close. The label for brick i uses ticks from brick i's close onwards. No tick used in features can be used in labels.

2. **Normalise by brick_size, never by price**: all candle features divide by the current brick_size. This makes features scale-invariant across the 2020–2024 price range ($1,500–$2,700 for XAUUSD).

3. **Bid for LONG exits, Ask for SHORT exits**: the label scanner must use execution-realistic pricing, identical to the Phase 9 BrickOfTicks approach.

4. **Limit fill validation**: T2 and T4 labels are only generated for bricks where the limit order actually filled. Unfilled orders are excluded from those classes entirely (not labelled as T0).

5. **No volume features**: zero `bid_vol` / `ask_vol` usage anywhere in the feature pipeline. The system must produce identical features when these columns are null.

6. **Walk-forward only**: training data never overlaps with validation or test data chronologically. The context window (last 5 bricks) does not span split boundaries — bricks at the start of each split use zero-padded context if needed.

7. **No TimeDistributed(Model)**: all TimeDistributed wrappers use plain Keras layers to avoid the macOS Metal deadlock documented in BrickOfTicks Phase 6.
