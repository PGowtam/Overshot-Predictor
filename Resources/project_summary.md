# BrickOfTicks: Complete Project Summary

> **One-shot reference document** — Everything discovered, built, and validated across 9 Phases and 2 Iterations.

---

## 1. The Core Idea

We predict the **outcome of the next Renko brick** (WIN or LOSS) by reading the L1 order book microstructure at the moment the brick closes. A Renko chart converts raw XAUUSD ticks into discrete directional events (bricks). Each brick close is a decision point: *will price continue (WIN) or reverse (LOSS)?*

The model doesn't predict *price*. It predicts whether the **current momentum regime** as encoded in the tick-by-tick order flow will sustain for at least one more brick.

---

## 2. Data Foundation

### 2.1 Renko CSV
- **30,978 bricks** spanning Jan 2020 – Dec 2024, dynamically sized via ATR (mean brick size ≈ $2.11).
- Columns: `date`, `open`, `high`, `low`, `close`, `uptrend`, `brick_size`, `sequence`, `outcome`.
- The `outcome` column is the broker's own WIN/LOSS determination (bid-price based). We do **not** use it as our training label.

### 2.2 L1 Tick Data
- Source: `Data/Raw/Ticks/{year}/{month}/{day}.parquet`
- Schema: `timestamp` (ns UTC), `bid`, `bid_vol`, `ask`, `ask_vol`
- Coverage: 2020–2026, ~billions of ticks.
- This is **indicative L1** data (top-of-book). No trade prints, no depth levels beyond L1.

### 2.3 Why "Indicative" Matters
OTC forex has no central order book. L1 data represents the broker's published best bid/ask. Features like OFI and Depth measure the broker's risk appetite, not a true exchange book. This is a deliberate design choice — we exploit the broker's quoting behavior as a predictive signal.

---

## 3. Label Generation (Phase 1)

### 3.1 The Hybrid Overshoot Algorithm
For each brick, we scan future ticks to compute:
- **`y_class`**: Binary (1 = WIN, 0 = LOSS). Determined by whether price hits TP (1× brick_size in the trend direction) before SL (1× brick_size against the trend).
- **`y_mag`**: Continuous. Measures how far price traveled in the favorable direction, normalized by brick_size.

**Two-phase scan:**
1. **Pre-TP** (SL-bounded): Track `peak = max(mid)` [LONG] or `min(mid)` [SHORT]. If SL hit first → LOSS, `y_mag = peak_distance / brick_size < 1.0`.
2. **Post-TP** (Trailing): After TP hit, track extension with a 1-brick-size trailing reversal. `y_mag = peak_distance / brick_size ≥ 1.0`.

This creates a **natural boundary at y_mag ≈ 1.0**: LOSS bricks always have `y_mag < 1.0`, WIN bricks always have `y_mag ≥ 1.0`.

### 3.2 Execution Pricing (Phase 9 Evolution)
The original scan used `mid = (bid + ask) / 2` for all ticks. Phase 9 introduced **execution-realistic pricing**:
- **LONG exits** → scan with `bid` (what you receive when selling)
- **SHORT exits** → scan with `ask` (what you pay when buying back)

This eliminated a directional asymmetry (LONG 57.5% vs SHORT 42.7% → balanced ~49/49%) and produced the honest ground truth labels used for all final models.

### 3.3 Key Statistics
| Metric | Mid-Price Labels | Execution-Priced Labels |
|---|---|---|
| Total Bricks | 30,978 | 30,978 |
| Resolved | 30,563 (98.66%) | ~30,571 |
| LONG WR | 57.5% | ~48.7% |
| SHORT WR | 42.7% | ~37.5% |
| Overall WR | 50.1% | ~43.1% |
| Excluded (tick gaps) | 415 (1.34%) | ~407 |

---

## 4. Feature Engineering (Phase 2)

### 4.1 The 9-Dimensional Tick Vector
Every single tick in the dataset produces a 9-element feature vector:

| Index | Feature | Description | Processing |
|---|---|---|---|
| 0 | `z_OFI` | Order Flow Imbalance | Weak-inequality OFI → Rolling 1000-tick z-score |
| 1 | `z_Depth` | Total book depth (bid_vol + ask_vol) | Rolling z-score |
| 2 | `z_Susc` | Susceptibility = raw_OFI / raw_Depth | Divide RAW first, then z-score |
| 3 | `z_Vel` | Tick velocity = 1/Δt | Rolling z-score |
| 4 | `z_Spread` | Spread = ask − bid | Rolling z-score |
| 5 | `Progress` | (mid − brick_open) / brick_size | Sawtooth, resets per brick |
| 6 | `Flag_Curr` | 1 if tick is within current brick | Binary |
| 7 | `Flag_Zone` | 1 if mid passed previous brick boundary | Binary |
| 8 | `Decay` | (current_brick − tick_brick) / buffer_depth | Positional encoding |

### 4.2 Z-Score Implementation
- **Window**: 1000 ticks (rolling, not cumulative).
- **Warmup**: Returns 0.0 when window has < 30 values.
- **Algorithm**: O(1) incremental Welford sliding window:
  ```
  μ_new = μ_old + (x_new − x_old) / N
  M2_new = M2_old + (x_new − x_old) × ((x_new − μ_new) + (x_old − μ_old))
  ```
- **Critical rule**: Susceptibility divides RAW OFI/Depth first, then z-scores the ratio. Never divide two z-scores.

### 4.3 The 3-Dimensional Macro Vector
Per brick:
| Index | Feature | Formula |
|---|---|---|
| 0 | `log_dur` | log(duration_seconds + 1) |
| 1 | `direction` | +1 (uptrend) or −1 (downtrend) |
| 2 | `z_size` | (brick_size − mean_50) / std_50 |

### 4.4 Signal Validation
Point-biserial correlation at brick close:
- **OFI**: r = +0.099 (p = 4.9e-67) ✅
- **Spread**: r = −0.097 (p = 7.3e-64) ✅
- **Velocity**: r = +0.001 (not significant)

OFI and spread carry **strong linear signal** at the brick-close timescale.

---

## 5. Buffer Simulation (Phase 3)

### 5.1 Micro-Buffer
- `deque(maxlen=100)` of 9D tick vectors.
- **Never resets** at brick boundaries — continuous across the entire dataset.
- At each brick close, snapshot the buffer → `(100, 9)` array.
- If buffer has < 100 ticks, zero-pad the oldest positions.
- After snapshotting, `Flag_Curr` (index 6) and `Decay` (index 8) are rewritten relative to the current brick.

### 5.2 Macro-History
- Last 10 macro-vectors stacked → `(10, 3)`.
- For bricks before 10 have formed, zero-pad oldest slots.

---

## 6. Tensor Construction (Phase 4)

### 6.1 Final Tensor Shape
For each brick `i ≥ 10`:
- **Micro Tensor**: `(10, 100, 9)` — last 10 brick snapshots stacked.
- **Macro Tensor**: `(10, 3)` — last 10 macro-vectors.

### 6.2 Walk-Forward Splits (original)
| Split | Date Range | Purpose |
|---|---|---|
| Train | Jan 2020 – Dec 2022 | Model fitting |
| Val | Jan 2023 – Jun 2023 | Threshold calibration |
| Test | Jul 2023 – Dec 2023 | Out-of-sample evaluation |
| Holdout | Jan 2024 – Dec 2024 | Pristine final validation |

### 6.3 Training Exclusions
1. `exclude_flag = True` → removed from ALL splits (tick gaps).
2. `duration < 2s` → removed from training only (fast bricks, noisy).
3. `chain_depth > 5` → `sample_weight = 0.5` in training (consecutive fast-brick chains).

---

## 7. Model Architecture (Phase 5)

### 7.1 Dual-Head CNN+LSTM
```
Inputs:
  Micro: (Batch, 10, 100, 9)
  Macro: (Batch, 10, 3)

TimeDistributed CNN Encoder (applied to each of 10 bricks):
  → 3 Parallel Conv1D branches (k=1, k=3, k=5), 16 filters each, causal padding, LeakyReLU(0.1)
  → Concatenate → (100, 48)
  → MaxPool1D(pool_size=4) → (25, 48)
  → Flatten → Dense(32, relu, L2=1e-4) → Dropout(0.3)
  → Output: (32,) per brick → (Batch, 10, 32)

Fusion:
  → Concatenate CNN output + Macro input → (Batch, 10, 35)

LSTM:
  → LSTM(32, L2=1e-4) → Dropout(0.3) → (Batch, 32)

Head A (prob_win): Dense(1, sigmoid) — probability of WIN
Head B (pred_os):  Dense(1, relu)    — predicted overshoot magnitude
```

### 7.2 Training Configuration
- **Optimizer**: Adam(lr=1e-3)
- **Loss**: `1.0 × BCE(prob_win, y_class) + 0.3 × Huber(pred_os, y_mag, δ=1.0)`
- **Head B**: Trained on ALL samples (WIN and LOSS). LOSS bricks naturally have `y_mag < 1.0`, serving as the anchor.
- **Regularization**: Dropout(0.3), L2(1e-4), EarlyStopping(patience=15), ReduceLROnPlateau(factor=0.5, patience=8).

### 7.3 Why Dual Heads?
Head A predicts *if* the trade wins. Head B predicts *how much* it overshoots. The combination enables a two-threshold filter: only enter trades where the model is confident (Prob_Win ≥ 0.5) AND predicts strong overshoot (Pred_OS ≥ threshold). This dramatically increases precision.

---

## 8. Threshold Calibration (Phase 7)

### 8.1 Operational Thresholds
| Parameter | Value | Meaning |
|---|---|---|
| `Prob_Win_threshold` | 0.50 | Minimum probability from Head A |
| `Pred_OS_threshold` | 1.3 – 1.8 | Minimum predicted overshoot from Head B |

### 8.2 The Pred_OS Filter is the Key Insight
The Pred_OS threshold is what transforms a ~77% accuracy classifier into a 87%+ precision trading system. By only entering trades where the model predicts the price will extend well beyond the TP level, we select for high-conviction setups.

Higher thresholds → fewer trades, higher win rate:
- `Pred_OS ≥ 1.3` → ~86% WR, ~1,500 trades on holdout
- `Pred_OS ≥ 1.8` → ~95% WR, ~350 trades on holdout

---

## 9. Cross-Validation (Iteration 2)

### 9.1 3-Fold Expanding Window
To eliminate single-split overfitting, we trained 3 independent models on expanding time windows:

| Fold | Train | Val | Test | Test WR | Calibrated OS |
|---|---|---|---|---|---|
| 1 | 2020–2021 | H1 2022 | H2 2022 | **93.41%** | 1.60 |
| 2 | 2020–H1 2022 | H2 2022 | H1 2023 | **89.36%** | 1.70 |
| 3 | 2020–2022 | H1 2023 | H2 2023 | **92.68%** | 1.80 |

### 9.2 Majority Voting Ensemble
On the pristine **2024 Holdout** (unseen by any fold):
- **Ensemble Win Rate**: `91.02%` (412 trades)
- Rule: Enter trade only if ≥ 2 of 3 fold models agree.

Each individual fold model also performed well independently on the holdout:
- Fold 1: 90.26% (462 trades)
- Fold 2: 88.94% (416 trades)
- Fold 3: 94.65% (355 trades)

---

## 10. Key Design Decisions & Lessons

### What Worked
1. **OFI with weak inequalities** — captures order refreshes when price is static.
2. **Susceptibility = raw OFI / raw Depth before z-scoring** — captures how the book absorbs flow.
3. **Continuous micro-buffer (never reset)** — preserves cross-brick context.
4. **Dual-head architecture** — Pred_OS as a confidence filter is the single biggest precision booster.
5. **Execution pricing** — honest labels removed directional bias.
6. **MaxPool1D(4) instead of GlobalAvgPool** — preserves temporal position within the tick window.

### What Didn't Work / Was Irrelevant
1. **Volume features** (raw bid_vol, ask_vol) — book sizes carry almost no additional signal beyond OFI/Depth.
2. **Tick velocity** — arrival rate alone doesn't predict outcome.
3. **Cumulative Welford z-score** — rolling window is strictly better.
4. **Lee-Ready / Trade Imbalance** — invalid for OTC L1 data (no trade prints).

### Critical Invariants (Never Violate These)
1. **No lookahead bias**: Feature vector at tick `k` uses zero info from `k+1` onwards.
2. **Rolling z-score window = 1000 ticks**, warmup = 30 ticks.
3. **y_class from hybrid algorithm**, NOT from CSV outcome column.
4. **Susceptibility: divide RAW, then z-score**. Never divide two z-scores.
5. **Micro-buffer never resets at brick boundaries**.

---

## 11. Artifacts & File Map

### Models
| Path | Description |
|---|---|
| `outputs/exec/model.keras` | Phase 9 execution-priced model (best single model) |
| `outputs/exec/config.json` | Thresholds for the single exec model |
| `outputs/exec/cv/fold_1/model.keras` | CV Fold 1 model |
| `outputs/exec/cv/fold_2/model.keras` | CV Fold 2 model |
| `outputs/exec/cv/fold_3/model.keras` | CV Fold 3 model |
| `outputs/exec/cv/fold_N/config.json` | Per-fold calibrated thresholds |

### Data / Tensors
| Path | Description |
|---|---|
| `outputs/exec/labels.parquet` | Execution-priced labels (30,978 rows) |
| `outputs/exec/tensors/` | Train/Val/Test tensors for Phase 9 model |
| `outputs/exec/cv/fold_N/tensors/` | Per-fold tensor datasets |
| `outputs/features/` | Cached 9D tick vectors, macro vectors |
| `outputs/features/snapshots/` | (100,9) snapshots per brick |

### Source Code
| File | Phase | Description |
|---|---|---|
| `src/label_generator.py` | 1 | Hybrid overshoot scan |
| `src/feature_engine.py` | 2 | 9D tick vector + 3D macro |
| `src/buffer_sim.py` | 3 | Micro-buffer simulation |
| `src/tensor_builder.py` | 4 | Tensor construction |
| `src/model.py` | 5 | CNN+LSTM architecture |
| `src/train.py` | 6 | Training loop |
| `src/calibrate.py` | 7 | Threshold calibration |
| `src/evaluate.py` | 8 | Test set evaluation |
| `src/phase9_pipeline.py` | 9 | Execution-price pipeline |
| `src/cv_tensor_builder.py` | IT2 | Cross-validation tensor builder |
| `src/cv_train.py` | IT2 | Cross-validation training |
| `src/cv_evaluate.py` | IT2 | Cross-validation evaluation |

---

## 12. What's Next: Production Trading Bot

The trained models and calibrated thresholds are ready to be handed to an MT5 execution engine. The execution engine must:

1. **Stream L1 ticks** from MT5 using `copy_ticks_from` (gap-less).
2. **Build Renko bricks** from bid price (matching training's renko construction).
3. **Compute 9D feature vectors** for every tick using the identical z-score windows and formulas.
4. **Maintain a continuous `deque(maxlen=100)` micro-buffer** — never reset between bricks.
5. **On each brick close**: Snapshot the buffer, stack last 10 snapshots + macro vectors, run inference through the loaded `.keras` model.
6. **Apply dual-threshold filter**: Only trade if `Prob_Win ≥ 0.5` AND `Pred_OS ≥ threshold`.
7. **Place orders** with SL = 1× brick_size and TP = 1× brick_size.

See `Resources/mt5_bot_specification.md` for the complete production specification.
