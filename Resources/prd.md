# BrickOfTicks Training System — Product Requirements Document

> **Scope**: Training pipeline only. Execution engine, live trading, and broker integration are out of scope for this phase.

---

## 1. Objective

Build a complete offline training pipeline that:
1. Generates regression labels (`y_mag`) by scanning L1 tick data after each Renko brick close
2. Computes the 9-dimensional tick feature vector and 3-dimensional macro vector for every brick
3. Constructs the hierarchical tensor inputs: Micro Tensor `(N, 10, 100, 9)` + Macro Tensor `(N, 10, 3)`
4. Trains a CNN+LSTM dual-head model with walk-forward time splits
5. Calibrates execution thresholds on the validation set
6. Evaluates final performance on the held-out test set

**Success means**: A trained model file + calibrated thresholds that can be handed to a future execution engine.

---

## 2. Data Inputs

### 2.1 Renko CSV
- **File**: `Data/Raw/renko_with_tick_outcomes_no_be_XAUUSD20-24.csv`
- **Rows**: 30,978 bricks (2020–2024)
- **Columns**: `date`, `open`, `high`, `low`, `close`, `volume`, `uptrend`, `brick_size`, `sequence`, `outcome`
- **Key facts**: `volume` is always 1.0 (OTC). `outcome` is binary WIN/LOSS. `brick_size` is ATR-based dynamic (mean ~2.11).

### 2.2 L1 Tick Data
- **Location**: `Data/Raw/Ticks/{year}/{month}/{day}.parquet`
- **Schema**: `timestamp` (ns, UTC), `bid`, `bid_vol`, `ask`, `ask_vol`
- **Coverage**: 2020–2026 (training uses 2020–2023)

### 2.3 What's Missing
The CSV has no `y_mag` column. This **must be computed** by scanning L1 ticks forward from each brick's close timestamp to find peak price before TP/SL is hit. This is a hard prerequisite for Head B training.

---

## 3. Output Artifacts

| Artifact | Description |
|---|---|
| `labels.parquet` | Renko CSV enriched with `y_class` (similar to the outcomes column in the Renko CSV), `y_mag`, `duration_seconds`, `brick_id` |
| Feature cache (`.npz` or `.parquet`) | Pre-computed 9D tick vectors and 3D macro vectors per brick |
| Tensor dataset (`.npz`) | Final `(N, 10, 100, 9)` micro tensors + `(N, 10, 3)` macro tensors + labels |
| `model.keras` | Trained dual-head CNN+LSTM model weights |
| `config.json` | Calibrated thresholds (`Prob_Win_threshold`, `Pred_OS_threshold`) + z-score window params |
| Evaluation report | Metrics on test set: WR, precision-recall, Head B correlation, confusion matrix |

---

## 4. Functional Requirements

### 4.1 Label Generation (FR-LG)

**FR-LG-01**: For each brick, compute `y_mag` using the **hybrid overshoot** algorithm:
```python
def calculate_true_overshoot(entry, brick_size, is_long, future_ticks):
    tp = entry + brick_size if is_long else entry - brick_size
    sl = entry - brick_size if is_long else entry + brick_size
    peak = entry  # rolling max (LONG) or min (SHORT)
    tp_hit = False

    for tick in future_ticks:
        mid = (tick.bid + tick.ask) / 2

        # Track extension in favorable direction
        if is_long:
            peak = max(peak, mid)
        else:
            peak = min(peak, mid)

        # Phase 1: Before TP — reversal check = SL (fixed level)
        if not tp_hit:
            if (is_long and mid >= tp) or (not is_long and mid <= tp):
                tp_hit = True
                continue
            if (is_long and mid <= sl) or (not is_long and mid >= sl):
                break  # SL hit → done

        # Phase 2: After TP — reversal = 1 brick_size trailing from peak
        else:
            if (is_long and mid <= peak - brick_size):
                break
            if (not is_long and mid >= peak + brick_size):
                break

    return abs(peak - entry) / brick_size
```
- **Pre-TP**: Reversal check is the fixed SL level. If SL hit → stop. `y_mag ∈ [0, ~1.0)` for LOSS bricks.
- **Post-TP**: Switches to dynamic 1-brick-size trailing reversal from rolling peak. `y_mag ∈ [1.0, ∞)` for WIN bricks.
- Creates a **natural boundary at y_mag ≈ 1.0** mirroring WIN/LOSS split.
- All tick prices use **mid-price**: `mid = (bid + ask) / 2`. The original CSV used bid-only pricing — mid-price is used here to match the feature pipeline's price convention.
- `y_class` is derived **from the hybrid algorithm**: `y_class = 1 if tp_hit else 0`. NOT from the CSV `outcome` column.

**FR-LG-02**: The CSV `outcome` column is retained as a **validation reference**. Report mismatch rate between `y_class` (from mid-price hybrid) and CSV `outcome` (from bid-price). Expected ~5–10% mismatch near TP/SL boundary due to half-spread offset. If mismatch > 15%, investigate data integrity. The mismatch is **systematic, not random**: LONGs gain WINs (mid reaches TP sooner), SHORTs lose WINs (mid reaches TP later). For mismatched bricks, plot y_mag histogram and assert >80% have `y_mag ∈ [0.85, 1.15]` — confirming boundary effect, not data corruption.

#### Phase 1 Findings (Empirical)

- **Natural boundary confirmed**: LOSS max=0.999998, WIN min=1.000191 — clean separation at y_mag ≈ 1.0.
- **30,978 bricks**: 30,563 resolved (98.66%), 415 excluded (tick gaps). WIN=15,302 (50.1%), LOSS=15,261 (49.9%).
- **y_mag distributions**: LOSS mean=0.353, std=0.295. WIN mean=2.086, std=1.175.
- **Mismatch rate**: 8.0% (2,431/30,563) — within expected 5–10%.
- **Boundary clustering**: Only 33.5% of mismatches have y_mag ∈ [0.85, 1.15], not the predicted >80%. This is because the CSV used a fundamentally different reversal algorithm (not just bid-only pricing), so mismatches can occur at any y_mag level.
- **Directional consistency**: 98.1% of mismatches follow the predicted mid-vs-bid pattern — this is the true validation of data integrity:
  - LONG mismatches: 1,277/1,307 (97.7%) are CSV=LOSS→algo=WIN (mid reaches TP sooner)
  - SHORT mismatches: 1,107/1,124 (98.5%) are CSV=WIN→algo=LOSS (mid reaches TP later)
- **Conclusion**: The directional consistency check (>90% threshold, actual 98.1%) replaces the boundary clustering check as the primary mismatch validation.

**FR-LG-03**: Compute `duration_seconds` = time between current brick close and next brick close (from tick stream or CSV timestamps).

**FR-LG-04**: Flag bricks where L1 tick data has gaps (no ticks found, or neither SL, TP, nor trailing reversal triggered). These bricks have invalid labels (`y_class` and `y_mag` are meaningless) and must be **excluded from ALL splits** (train, val, test, holdout).

### 4.1.1 Signal Existence Checkpoints (FR-SC)

**FR-SC-01** (After Phase 1): Compute point-biserial correlation between raw feature values at brick close and `y_class`:
- Extract last tick before each brick close: raw OFI, raw Velocity, raw Spread
- Compute Pearson r with `y_class` for each
- **RED** if all |r| < 0.02: features carry no linear signal at this timescale — log warning, proceed with caution
- **GREEN** if any |r| > 0.03: linear signal exists, proceed with confidence
- **AMBER** if 0.02–0.03: signal is weak, CNN+LSTM must find non-linear patterns

**FR-SC-02** (After Phase 4): Train a simple logistic regression baseline on mean feature values:
- For each brick: compute mean of each of 9 features across last 10 ticks in its buffer
- Train LogisticRegression on these 9 means to predict y_class (same train/val split)
- Compare val accuracy against majority-class baseline (~50%)
- **RED** if logistic regression < 52%: features carry almost no separable signal
- **GREEN** if > 55%: strong signal, deep model should perform well
- **AMBER** if 52–55%: weak signal, deep model's job is to amplify it

### 4.2 Feature Engineering (FR-FE)

**FR-FE-01**: Compute OFI using **weak inequalities** (`>=` and `<=`):
```
e_k = I(dBid>=0)*q^B_k - I(dBid<=0)*q^B_{k-1} - I(dAsk<=0)*q^A_k + I(dAsk>=0)*q^A_{k-1}
```

**FR-FE-02**: Compute Susceptibility by dividing **raw values first**, then z-scoring:
```
S_raw = e_k / (D_k + 1e-8)
z_Susc = (S_raw - μ_1000) / σ_1000
```

**FR-FE-03**: All 5 z-scored features use a **rolling window of 1000 ticks** (not cumulative Welford).

**FR-FE-04**: Z-score online update uses sliding-window incremental formula:
```
μ_new = μ_old + (x_new - x_old) / N
M2_new = M2_old + (x_new - x_old) * ((x_new - μ_new) + (x_old - μ_old))
σ = sqrt(M2_new / (N - 1))
```

**FR-FE-05**: For ticks before the deque reaches N≥30, all z-scored features are set to 0.

**FR-FE-06**: Progress, Flag_Curr, Flag_Zone, Decay computed as defined in the implementation plan (no z-scoring).

**FR-FE-07**: Macro-vector per brick: `[log(duration_s + 1), direction(±1), z_Size]` where `z_Size = (brick_size - μ_50_bricks) / σ_50_bricks`.

### 4.3 Buffer Simulation (FR-BF)

**FR-BF-01**: Simulate a `deque(maxlen=100)` Micro-Buffer. Append each tick's 9D vector. **Never reset at brick boundaries.**

**FR-BF-02**: At each brick close, snapshot the current Micro-Buffer state as a `(100, 9)` array. If buffer has fewer than 100 ticks, zero-pad the oldest positions.

**FR-BF-03**: Maintain a Macro-History of the last 10 brick snapshots. For bricks before 10 have formed, zero-pad the oldest brick slots.

### 4.4 Tensor Construction (FR-TC)

**FR-TC-01**: For each brick (after the first 10), construct:
- Micro Tensor: `(10, 100, 9)` — last 10 brick snapshots
- Macro Tensor: `(10, 3)` — last 10 macro-vectors

**FR-TC-02**: Training samples begin at the 11th brick in each walk-forward split (first 10 used for context only).

### 4.5 Model Architecture (FR-ML)

**FR-ML-01**: TimeDistributed CNN block:
- 3 parallel Conv1D branches (kernels 1, 3, 5; 16 filters each; `padding='causal'`; `LeakyReLU(0.1)`)
- Concat along feature axis → `(100, 48)`
- `MaxPool1D(pool_size=4)` → `(25, 48)` — NOT GlobalAvgPool
- `Flatten() → Dense(32, relu) → Dropout(0.3)` → `(32,)` per brick
- Applied via `TimeDistributed` → output: `(Batch, 10, 32)`

**FR-ML-02**: Fusion: `Concatenate([cnn_embedding(32), macro_vector(3)])` → `(35,)` per brick → `(Batch, 10, 35)`

**FR-ML-03**: LSTM: `LSTM(units=32, return_sequences=False) → Dropout(0.3)` → `(Batch, 32)`

**FR-ML-04**: Dual heads from shared LSTM output:
- **Head A**: `Dense(1, activation='sigmoid')` → `Prob_Win`
- **Head B**: `Dense(1, activation='relu')` → `Pred_OS`

### 4.6 Training (FR-TR)

**FR-TR-01**: Walk-forward split by date (NO random shuffling):
- Train: Jan 2020 – Dec 2022
- Validation: Jan 2023 – Jun 2023
- Test: Jul 2023 – Dec 2023
- Holdout: Jan 2024 – Dec 2024 (out-of-sample paper validation, never used for any decision)

**FR-TR-02**: Loss: `L = 1.0 × BCE(Prob_Win, y_class) + 0.3 × Huber(Pred_OS, y_mag, δ=1.0)`

**FR-TR-03**: Head B is trained on ALL samples (WIN and LOSS). LOSS bricks have `y_mag ∈ [0, ~1.0)`, WIN bricks have `y_mag ∈ [1.0, ∞)`. No masking.

**FR-TR-04**: Exclude bricks with `duration < 2s` from training only. Keep in val/test.

**FR-TR-05**: `sample_weight = 0.5` for bricks in consecutive fast-brick chains of depth > 5.

**FR-TR-06**: Regularization: `Dropout(0.3)`, `L2(1e-4)`, early stopping `patience=15`, `ReduceLROnPlateau(factor=0.5, patience=8)`.

**FR-TR-07**: Overfitting diagnostic: if `val_loss > 1.5 × train_loss` after epoch 20, flag and increase Dropout to 0.4.

### 4.7 Threshold Calibration (FR-CAL)

**FR-CAL-01**: On validation set, plot precision-recall curve for Head A. Select `Prob_Win_threshold` where precision ≥ 0.60.

**FR-CAL-02**: On validation set, plot Pred_OS distributions for WIN vs LOSS. Select `Pred_OS_threshold` where WIN distribution dominates. Starting value: 1.1.

**FR-CAL-03**: Save calibrated thresholds to `config.json`.

### 4.8 Evaluation (FR-EV)

**FR-EV-01**: On test set (Jul–Dec 2023), compute:
- Win rate of model-filtered bricks (target ≥60%)
- Head B Pearson r with actual `y_mag` on WIN samples (target ≥0.30)
- Pred_OS distribution check: >70% of WIN predictions ≥ 1.0

**FR-EV-02**: Volume Feature Mitigation Workflow (sequential, stop early if signal confirmed):
1. **Step 1 — Ablation**: Train model with all 9 features, then retrain with only 6 non-volume features (`z_Vel`, `z_Spread`, `Progress`, `Flag_Curr`, `Flag_Zone`, `Decay`). Compare test WR. If volume features add <1% WR, proceed to Step 2.
2. **Step 2 — Tick Direction Encoding**: Replace `z_OFI` with `tick_direction = sign(mid_k - mid_{k-1})` accumulated over rolling window. Retrain and compare.
3. **Step 3 — Volume Ratio**: Replace raw `bid_vol`/`ask_vol` in OFI/Depth/Susc with `bid_vol / (bid_vol + ask_vol)` ratio. Retrain and compare.
4. **Step 4 — Feature Importance**: After selecting the best feature set, compute gradient-based feature attribution or permutation importance. Document which features contribute most to predictions.

**FR-EV-03**: Generate confusion matrix, precision-recall curves, and per-month breakdown.

**FR-EV-04**: On holdout set (2024), run model with calibrated thresholds. Report WR as out-of-sample validation.

**FR-EV-05**: Holdout Failure Remediation (triggered if holdout WR < 55%):
1. **Diagnose**: Plot monthly holdout WR, compare 2024 feature distributions vs 2020–2022, check y_class balance drift
2. **Expanding window retrain**: Train = 2020–2023, Val = H1 2024, Test = H2 2024. If WR recovers → regime-dependent
3. **Feature audit**: Re-run volume mitigation workflow on expanded dataset, compare feature importance rankings
4. **Architecture simplification**: Remove Head B, train pure classifier; or reduce to LSTM-only
5. **Pivot decision**: If no variant achieves > 55% out-of-sample, conclude L1 indicative data insufficient at Renko timescales

---

## 5. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NF-01 | No look-ahead bias: feature vector at tick k uses zero info from k+1 onwards |
| NF-02 | Train/inference reproducibility: model output for identical tensor is identical |
| NF-03 | All z-scored features use rolling 1000-tick window (not cumulative) |
| NF-04 | Feature pipeline must process the full 2020–2023 dataset in < 2 hours on local hardware |
| NF-05 | Model training must complete in < 4 hours with early stopping |

---

## 6. Design Constraints

| ID | Constraint |
|---|---|
| DC-01 | No Lee-Ready or Trade Imbalance features — invalid for OTC L1 data |
| DC-02 | Susceptibility: divide RAW values first, then z-score. Never divide two z-scores. |
| DC-03 | MaxPool1D, not GlobalAvgPool — temporal position must be preserved |
| DC-04 | Raw (100,9) snapshots stored — never pre-compute CNN embeddings |
| DC-05 | Pred_OS measured from ENTRY price, not from TP level |
| DC-06 | y_mag uses hybrid overshoot: LOSS bricks `y_mag ∈ [0, ~1.0)`, WIN bricks `y_mag ∈ [1.0, ∞)`. No masking of Head B. |
| DC-07 | All tick prices use mid-price `(bid + ask) / 2`. y_class is derived from hybrid algorithm, NOT from CSV outcome. |

---

## 7. Out of Scope (v1.0 Training)

- Execution engine, warmup protocol, order placement
- Hard rule parameters (Sequence Agreement, fast-brick routing) — deferred to execution engine
- Live spread guard, broker integration
- State persistence for live sessions (`state.json`)
- Multi-instrument support
- Automated retraining pipeline
- Sub-10ms latency optimization
- Expanding window cross-validation (3-fold) — planned for iteration 2 after first end-to-end pass

### Planned Post-Pipeline Phase

**Phase 9: Market Realism Recalibration** — After Phases 1–8 complete with mid-price scanning, recalibrate the pipeline using execution-realistic pricing (bid for LONG exits, ask for SHORT exits). Two-pronged approach:

1. **Option A (Quick Check)**: Re-evaluate the existing mid-price model against execution-priced labels. Answers: *"If we deployed this model today, what's the real WR?"*
2. **Option B (Full Re-run)**: Retrain a completely new model on execution-priced labels. Re-run features → buffers → tensors → training → calibration. Answers: *"Does the model learn better patterns when trained on realistic labels?"*

**Comprehensive Benchmark**: Compare mid-price model, mid-price model on exec labels, and exec-price model across all metrics (WR, Sharpe, MaxDD, directional balance). Determine which pricing yields a genuine, tradeable edge.

Motivated by the finding that mid-price scanning introduces a directional asymmetry (LONG 57.5% WIN vs SHORT 42.7%) that doesn't reflect real execution.
