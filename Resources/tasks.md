# BrickOfTicks Training Pipeline — Task Breakdown

> Track progress by marking: `[ ]` todo, `[/]` in progress, `[x]` done

---

## Phase 0: Project Setup

- [x] Create project directory structure (`src/`, `tests/`, `outputs/`, `outputs/features/`, `outputs/tensors/`, `outputs/plots/`)
- [x] Create Python virtual environment in `.venv`
- [x] Install core dependencies: `numpy`, `pandas`, `pyarrow`, `tensorflow`/`keras`, `scikit-learn`, `matplotlib`, `scipy`
- [x] Verify data availability:
  - [x] Confirm `Data/Raw/renko_with_tick_outcomes_no_be_XAUUSD20-24.csv` loads correctly (30,978 rows)
  - [x] Confirm tick parquets load (`Data/Raw/Ticks/2020/01/01.parquet` — schema: timestamp, bid, bid_vol, ask, ask_vol)
  - [x] Spot-check tick data for 3 random dates across 2020, 2021, 2022, 2023
- [x] Load Renko CSV and print summary stats: date range, row count, WIN/LOSS ratio, brick_size distribution

---

## Phase 1: Label Generation (`src/label_generator.py`)

### 1.1 Core label generator
- [x] Write function `load_ticks_for_date(date) -> DataFrame` that loads the correct parquet file for a given date
- [x] Write function `load_ticks_range(start_time, end_time) -> DataFrame` that loads and concatenates tick files spanning multiple days
- [x] Write function `calculate_true_overshoot(entry, brick_size, is_long, future_ticks) -> y_mag`:
  - [x] Compute TP and SL levels
  - [x] Phase 1 (pre-TP): track rolling peak, check reversal against fixed SL level
  - [x] Phase 2 (post-TP): switch to dynamic 1-brick-size trailing reversal from peak
  - [x] Return `y_mag = abs(peak - entry) / brick_size`
  - [x] LOSS bricks: y_mag ∈ [0, ~1.0), WIN bricks: y_mag ∈ [1.0, ∞)
  - [x] Handle edge case: tick data ends before resolution → return None (exclude)
- [x] Write function `compute_duration(brick_close_time, next_brick_close_time) -> float` (seconds)

### 1.2 Batch processing
- [x] Write main function `generate_all_labels(renko_csv, tick_dir) -> DataFrame`:
  - [x] Iterate through all 30,978 bricks
  - [x] For each brick: load ticks after close, call `calculate_true_overshoot`
  - [x] Derive `y_class` from hybrid algorithm: `y_class = 1 if tp_hit else 0` (NOT from CSV outcome)
  - [x] Compute `csv_outcome_match = (y_class == (1 if CSV.outcome == 'WIN' else 0))`
  - [x] Assign global `brick_id` (0 to N-1)
  - [x] Handle edge cases: last brick in dataset, missing tick files, overnight gaps
- [x] Add progress reporting (print every 1000 bricks)
- [x] Add `exclude_flag` column for bricks where neither SL/TP/reversal resolved

### 1.3 Validation
- [x] Assert `y_mag >= 0.0` for all rows
- [x] Assert `y_mag < 1.0` for all `y_class == 0` rows
- [x] Assert `y_mag >= 1.0` for all `y_class == 1` rows
- [x] Report CSV outcome mismatch rate: `sum(~csv_outcome_match) / total` (expect 5–10%)
- [x] If mismatch > 15%, investigate: print mismatched bricks' y_mag values (should cluster near 1.0)
- [x] Directional consistency check: assert >90% of mismatches follow mid-vs-bid pattern (actual: 98.1%)
- [x] Print y_mag distribution stats: mean, median, std, min, max for WIN and LOSS separately
- [x] Count and print excluded labels (tick data gaps)
- [x] Save to `outputs/labels.parquet`

### 1.4 Unit tests (`tests/test_label_generator.py`)
- [x] Test LONG LOSS: price goes up 0.5 bricks then hits SL → y_mag = 0.5
- [x] Test LONG LOSS: price never rises, immediately hits SL → y_mag = 0.0
- [x] Test LONG WIN: price hits TP then extends 1 more brick before trailing reversal → y_mag = 2.0
- [x] Test LONG WIN: price hits TP exactly, immediately retraces → y_mag = 1.0
- [x] Test SHORT: mirror of LONG tests
- [x] Test edge case: tick data ends before resolution → returns None

---

## Phase 1.5: Signal Existence Checkpoint 1 (`src/signal_check.py`)

### 1.5.1 Raw feature extraction
- [x] Load `outputs/labels.parquet`
- [x] For each non-excluded brick, load last tick before brick close
- [x] Compute raw features at that tick:
  - [x] `raw_ofi = e_k` (OFI using previous tick for delta)
  - [x] `raw_velocity = 1 / (t_k - t_{k-1} + 1e-3)`
  - [x] `raw_spread = ask - bid`

### 1.5.2 Correlation analysis
- [x] Compute point-biserial correlation: `pearsonr(raw_ofi, y_class)`
- [x] Compute point-biserial correlation: `pearsonr(raw_velocity, y_class)`
- [x] Compute point-biserial correlation: `pearsonr(raw_spread, y_class)`
- [x] Print p-values alongside correlations

### 1.5.3 Decision gate
- [x] If all |r| < 0.02: print "🔴 RED — No linear signal detected. Proceed with caution."
- [x] If any |r| > 0.03: print "🟢 GREEN — Signal confirmed."
- [x] If 0.02–0.03: print "🟡 AMBER — Weak signal. CNN+LSTM must find non-linear patterns."
- [x] Save results to `outputs/signal_check_1.json`

---

## Phase 2: Feature Engineering (`src/feature_engine.py`)

### 2.1 Z-score normalization
- [ ] Implement `RollingZScore` class:
  - [ ] `__init__(window=1000)`: create deque, init μ, M2
  - [ ] `update(x_new) -> z_value`: O(1) incremental update when full, recompute when filling
  - [ ] Return 0.0 when deque has fewer than 30 values
  - [ ] Handle edge case: σ = 0 (constant values) → return 0.0

### 2.2 OFI computation
- [ ] Implement `compute_ofi(bid_k, bid_km1, ask_k, ask_km1, bid_vol_k, bid_vol_km1, ask_vol_k, ask_vol_km1) -> float`:
  - [ ] Use WEAK inequalities (`>=` and `<=`)
  - [ ] Verify it fires when `dBid == 0` but `bid_vol` changes

### 2.3 All raw feature computations
- [ ] Implement `compute_depth(bid_vol, ask_vol) -> float`
- [ ] Implement `compute_susceptibility(ofi_raw, depth_raw) -> float` (divide RAW, add 1e-8 to denominator)
- [ ] Implement `compute_velocity(t_k, t_km1) -> float` (1 / (gap_ms + 1e-3))
- [ ] Implement `compute_spread(ask, bid) -> float`
- [ ] Implement `compute_progress(mid, brick_open, brick_size) -> float`
- [ ] Implement `compute_flag_curr(tick_brick_id, current_brick_id) -> int`
- [ ] Implement `compute_flag_zone(mid, prev_brick_open, prev_brick_size) -> int`
- [ ] Implement `compute_decay(current_brick_id, tick_brick_id, max_depth) -> float`

### 2.4 Macro-vector computation
- [ ] Implement `compute_macro_vector(duration_s, is_uptrend, brick_size, brick_size_history) -> np.array(3)`:
  - [ ] `log_dur = log(duration_s + 1)`
  - [ ] `direction = +1 if uptrend else -1`
  - [ ] `z_size = (brick_size - mean(last_50)) / std(last_50)` (handle < 50 bricks case)

### 2.5 Full pipeline
- [ ] Write `process_all_ticks(labels_df, tick_dir) -> (tick_features_per_brick, macro_vectors)`:
  - [ ] Process ticks chronologically across all bricks
  - [ ] Maintain 5 `RollingZScore` instances (OFI, Depth, Susc, Vel, Spread)
  - [ ] Track brick boundaries using timestamps from labels
  - [ ] For each tick: compute all 9 features, store in per-brick list
  - [ ] At each brick close: compute macro-vector
  - [ ] Save per-brick tick vectors to `outputs/features/tick_vectors_{brick_id}.npy`
  - [ ] Save all macro-vectors to `outputs/features/macro_vectors.npy`

### 2.6 Validation
- [ ] Assert no NaN or Inf in z_Susc across entire dataset
- [ ] Assert z_OFI non-zero when bid static but bid_vol changes (find such cases in data)
- [ ] Print feature statistics per feature (mean, std, min, max)
- [ ] Verify Progress resets at brick boundaries (check 5 random bricks)
- [ ] Save `outputs/features/brick_metadata.parquet`

### 2.7 Unit tests (`tests/test_feature_engine.py`)
- [ ] Test `RollingZScore` with known values (verify against numpy.mean/std)
- [ ] Test OFI weak inequality: dBid=0, bid_vol changes → e_k ≠ 0
- [ ] Test OFI weak inequality: dBid=0, same bid_vol → e_k = 0
- [ ] Test Susceptibility: divide raw, not z-scores
- [ ] Test z-score returns 0.0 when N < 30
- [ ] Test z-score handles σ=0 gracefully

---

## Phase 3: Buffer Simulation (`src/buffer_sim.py`)

### 3.1 Micro-Buffer simulation
- [ ] Implement `simulate_micro_buffer(all_tick_vectors_per_brick) -> list[np.array(100, 9)]`:
  - [ ] Maintain `deque(maxlen=100)`
  - [ ] For each brick: append all tick vectors, then snapshot
  - [ ] Zero-pad at front if buffer has < 100 ticks
  - [ ] NEVER reset the buffer between bricks

### 3.2 Save snapshots
- [ ] Save each snapshot to `outputs/features/snapshots/snapshot_{brick_id}.npy`
- [ ] Save metadata to `outputs/features/buffer_metadata.parquet`: brick_id, n_real_ticks, n_padded

### 3.3 Validation
- [ ] Assert every snapshot has shape `(100, 9)`
- [ ] Assert no NaN in any snapshot
- [ ] For 5 random fast bricks (duration < 10s): verify Flag_Curr count matches expected tick count
- [ ] Verify buffer continuity: last N ticks of brick i match first N ticks of brick i+1 (for 10 random pairs)

### 3.4 Unit tests (`tests/test_buffer_sim.py`)
- [ ] Test with exactly 100 ticks → no padding
- [ ] Test with 50 ticks → 50 zeros at front
- [ ] Test with 150 ticks over 2 bricks → buffer rolls over correctly
- [ ] Test continuity across 3 bricks

---

## Phase 4: Tensor Construction (`src/tensor_builder.py`)

### 4.1 Tensor assembly
- [ ] Write `build_tensors(snapshots, macro_vectors, labels) -> dict`:
  - [ ] For each brick i ≥ 10: stack last 10 snapshots → `(10, 100, 9)`
  - [ ] Stack last 10 macro-vectors → `(10, 3)`
  - [ ] Pair with (y_class, y_mag, duration, date, brick_id)

### 4.2 Walk-forward split
- [ ] Implement split assignment:
  - [ ] Train: date < 2023-01-01
  - [ ] Val: 2023-01-01 ≤ date < 2023-07-01
  - [ ] Test: 2023-07-01 ≤ date < 2024-01-01
  - [ ] Holdout: date ≥ 2024-01-01

### 4.3 Training exclusions
- [ ] Drop `exclude_flag = True` bricks from ALL splits (invalid labels)
- [ ] Remove bricks with `duration < 2s` from training set only (valid labels, but spillover-dominated)
- [ ] Compute fast-brick chain depth per brick
- [ ] Assign `sample_weight = 0.5` for `chain_depth > 5`, else `1.0`
- [ ] Keep all non-excluded bricks in val/test

### 4.4 Save tensors
- [ ] Save train/val/test/holdout splits as `.npy` files:
  - [ ] `{split}_micro.npy`, `{split}_macro.npy`
  - [ ] `{split}_y_class.npy`, `{split}_y_mag.npy`
  - [ ] `train_weights.npy`
- [ ] Save split metadata: brick counts, date ranges, class balance

### 4.5 Validation
- [ ] Assert zero date overlap between train/val/test/holdout
- [ ] Assert all micro tensors shape: `(10, 100, 9)`
- [ ] Assert all macro tensors shape: `(10, 3)`
- [ ] Assert no NaN in any tensor
- [ ] Print split sizes and WIN/LOSS ratio per split
- [ ] Assert `exclude_flag` bricks NOT in any split
- [ ] Assert `duration < 2s` bricks NOT in training but present in val/test

### 4.6 Unit tests (`tests/test_tensor_builder.py`)
- [ ] Test split assignment with known dates
- [ ] Test chain depth calculation
- [ ] Test that brick_id < 10 produces no tensor (not enough context)
- [ ] Test sample weight assignment

---

## Phase 4.5: Signal Existence Checkpoint 2 (`src/signal_check.py`)

### 4.5.1 Feature aggregation
- [ ] Load train and val tensors from Phase 4
- [ ] For each brick: compute mean of each of 9 features across last 10 ticks of most recent snapshot
- [ ] Result: 9D feature vector per brick

### 4.5.2 Logistic regression baseline
- [ ] Train `sklearn.linear_model.LogisticRegression(C=1.0)` on training 9D features → y_class
- [ ] Predict on validation set
- [ ] Compute accuracy and AUC
- [ ] Compare against majority-class baseline accuracy

### 4.5.3 Decision gate
- [ ] If accuracy < 52%: print "🔴 RED — Features carry almost no separable signal."
- [ ] If accuracy > 55%: print "🟢 GREEN — Strong signal, deep model should amplify."
- [ ] If 52–55%: print "🟡 AMBER — Weak signal, proceed but manage expectations."
- [ ] Save results to `outputs/signal_check_2.json`

---

## Phase 5: Model Architecture (`src/model.py`)

### 5.1 Build model
- [ ] Implement `build_model() -> keras.Model`:
  - [ ] CNN block with 3 parallel Conv1D branches (k=1,3,5; 16 filters; causal; LeakyReLU)
  - [ ] MaxPool1D(4) — NOT GlobalAvgPool
  - [ ] Flatten → Dense(32, relu) → Dropout(0.3)
  - [ ] TimeDistributed wrapper
  - [ ] Fusion: Concatenate CNN output (32) with macro input (3)
  - [ ] LSTM(32, return_sequences=False) → Dropout(0.3)
  - [ ] Head A: Dense(1, sigmoid) named 'prob_win'
  - [ ] Head B: Dense(1, relu) named 'pred_os'
  - [ ] L2(1e-4) on Dense and LSTM kernels

### 5.2 Compile model
- [ ] Implement `compile_model(model)`:
  - [ ] Head A loss: BinaryCrossentropy
  - [ ] Head B loss: Huber(delta=1.0)
  - [ ] Loss weights: {'prob_win': 1.0, 'pred_os': 0.3}
  - [ ] Optimizer: Adam(lr=1e-3)
  - [ ] Metrics: Head A accuracy, Head B MAE

### 5.3 Validation
- [ ] Print `model.summary()` — verify ≈48K params
- [ ] Forward pass with random input: `(2, 10, 100, 9)` + `(2, 10, 3)`
- [ ] Assert Head A output shape: `(2, 1)`, values ∈ [0, 1]
- [ ] Assert Head B output shape: `(2, 1)`, values ≥ 0

### 5.4 Unit tests (`tests/test_model.py`)
- [ ] Test model builds without error
- [ ] Test forward pass produces correct output shapes
- [ ] Test Head A activation is sigmoid (output bounded)
- [ ] Test Head B activation is relu (output non-negative)
- [ ] Test that MaxPool1D is used (not GlobalAvgPool) — check layer names

---

## Phase 6: Training (`src/train.py`)

### 6.1 Training script
- [ ] Load train/val tensors from `outputs/tensors/`
- [ ] Load sample weights for training
- [ ] Build and compile model
- [ ] Set up callbacks:
  - [ ] EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
  - [ ] ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8)
  - [ ] ModelCheckpoint('outputs/model.keras', save_best_only=True)
  - [ ] CSVLogger('outputs/training_log.csv')
- [ ] Train with `model.fit()`, batch_size=64, max_epochs=200
- [ ] Log warnings if val_loss > 1.5 × train_loss after epoch 20

### 6.2 Post-training diagnostics
- [ ] Plot training vs validation loss curves
- [ ] Print Head A prediction variance on validation set
- [ ] Print Head B prediction variance on validation set
- [ ] If Head B std < 0.05: log WARNING about potential fallback needed
- [ ] Print mean Pred_OS on LOSS samples from training set (target < 0.5)
- [ ] Save loss curve plot to `outputs/plots/loss_curves.png`

### 6.3 Validation
- [ ] Assert training converges (final loss < initial loss)
- [ ] Assert model file saved to `outputs/model.keras`
- [ ] Assert training log saved to `outputs/training_log.csv`

---

## Phase 7: Threshold Calibration (`src/calibrate.py`)

### 7.1 Generate validation predictions
- [ ] Load trained model from `outputs/model.keras`
- [ ] Load validation tensors
- [ ] Generate predictions: `Prob_Win`, `Pred_OS` for all val bricks

### 7.2 Calibrate Head A threshold
- [ ] Compute precision-recall curve
- [ ] Find Prob_Win_threshold where precision ≥ 0.60
- [ ] Plot precision-recall curve → `outputs/plots/pr_curve.png`

### 7.3 Calibrate Head B threshold
- [ ] Separate Pred_OS predictions by actual WIN vs LOSS
- [ ] Plot overlapping distributions → `outputs/plots/pred_os_dist.png`
- [ ] Find Pred_OS_threshold where WIN distribution clearly dominates LOSS
- [ ] Start from 1.1, adjust based on distributions

### 7.4 Save config
- [ ] Write `outputs/config.json` with calibrated thresholds
- [ ] Print selected thresholds and their justification metrics

---

## Phase 8: Evaluation (`src/evaluate.py`)

### 8.1 Model evaluation on test set
- [ ] Load model, config, and test tensors
- [ ] Generate predictions on test set
- [ ] Apply calibrated thresholds: filter bricks passing both
- [ ] Compute win rate on filtered bricks (target ≥ 60%)
- [ ] Compute Head B Pearson r with actual y_mag on WIN samples (target ≥ 0.30)
- [ ] Compute Pred_OS > 1.0 ratio on WIN predictions (target ≥ 70%)

### 8.2 Volume Feature Mitigation Workflow

#### Step 1 — Ablation Test
- [ ] Rebuild feature pipeline with only 6 non-volume features: `z_Vel`, `z_Spread`, `Progress`, `Flag_Curr`, `Flag_Zone`, `Decay`
- [ ] Rebuild tensors from 6-feature snapshots `(10, 100, 6)` — update model input shape
- [ ] Retrain model with identical config
- [ ] Compare test WR against 9-feature baseline
- [ ] Record: ΔWR from ablation. If volume adds <1% WR → proceed to Step 2. If ≥1% → skip to Step 4.

#### Step 2 — Tick Direction Encoding
- [ ] Implement `tick_direction = sign(mid_k - mid_{k-1})` as z-scored replacement for `z_OFI`
- [ ] Keep `z_Depth` computed from raw volumes, recompute `z_Susc = tick_direction / (Depth + 1e-8)` then z-score
- [ ] Rebuild tensors (still 9 features, but 3 changed)
- [ ] Retrain and compare test WR against baseline and Step 1

#### Step 3 — Volume Ratio Reformulation
- [ ] Replace raw `bid_vol`/`ask_vol` with `vol_ratio = bid_vol / (bid_vol + ask_vol + 1e-8)`
- [ ] Recompute OFI using vol_ratio instead of raw volumes
- [ ] Recompute Depth using vol_ratio sum
- [ ] Recompute Susceptibility
- [ ] Rebuild tensors, retrain, compare

#### Step 4 — Feature Importance Analysis
- [ ] Using the best model from Steps 1–3, compute permutation importance per feature
- [ ] For each of 9 features: zero-out that channel across all test tensors, measure Prob_Win / Pred_OS change
- [ ] Rank features by impact
- [ ] Drop any features with zero measured importance
- [ ] Save feature importance plot → `outputs/plots/feature_importance.png`
- [ ] Document best feature set in `outputs/evaluation_report.md`

### 8.3 Holdout evaluation (2024)
- [ ] Load holdout tensors (2024 data)
- [ ] Run model with calibrated thresholds
- [ ] Compute holdout WR (target ≥ 55%)
- [ ] Print monthly WR breakdown for 2024
- [ ] Apply decision gate:
  - [ ] ≥ 58%: Deploy to paper trading
  - [ ] 55–58%: Investigate per-month breakdown
  - [ ] 50–55%: Trigger remediation
  - [ ] < 50%: Do NOT deploy

### 8.4 Holdout Failure Remediation (if WR < 55%)
- [ ] Step 1 — Diagnose:
  - [ ] Plot monthly holdout WR
  - [ ] Compare 2024 feature distributions vs 2020–2022 (histogram overlay)
  - [ ] Check y_class balance drift between training and holdout
- [ ] Step 2 — Expanding window retrain:
  - [ ] Retrain with Train = 2020–2023, Val = H1 2024, Test = H2 2024
  - [ ] Compare WR against original model
  - [ ] If WR recovers: model is regime-dependent, needs periodic retraining
- [ ] Step 3 — Feature audit:
  - [ ] Re-run volume mitigation workflow on expanded dataset
  - [ ] Compare feature importance rankings: original vs expanded
  - [ ] If rankings change dramatically: feature-outcome relationship is non-stationary
- [ ] Step 4 — Architecture simplification:
  - [ ] Remove Head B, train pure classifier
  - [ ] Or reduce to LSTM-only (fewer params, harder to overfit)
  - [ ] If simple LSTM can't beat 52%: features don't predict at this timescale
- [ ] Step 5 — Pivot decision:
  - [ ] If no variant > 55%: conclude L1 indicative data insufficient at Renko timescales
  - [ ] Document findings in `outputs/evaluation_report.md`

### 8.5 Generate reports
- [ ] Confusion matrix (model-filtered trades) → `outputs/plots/confusion_matrix.png`
- [ ] Precision-recall curve on test set → `outputs/plots/test_pr_curve.png`
- [ ] Pred_OS distribution (test set WIN vs LOSS) → `outputs/plots/test_pred_os.png`
- [ ] Monthly WR breakdown (Jul–Dec 2023) → `outputs/plots/monthly_wr.png`
- [ ] Prob_Win histogram → `outputs/plots/prob_win_hist.png`
- [ ] Pred_OS histogram → `outputs/plots/pred_os_hist.png`
- [ ] Volume ablation comparison table (Steps 1–3)
- [ ] Feature importance chart (Step 4)
- [ ] Holdout (2024) report

### 8.6 Final summary
- [ ] Print all key metrics in a summary table
- [ ] Save evaluation report to `outputs/evaluation_report.md`
- [ ] Declare PASS or FAIL for each acceptance criterion

---

## Iteration 2: Expanding Window Cross-Validation (Deferred)

_To be executed after the first end-to-end pass succeeds._

### CV.1 3-Fold expanding window
- [ ] Fold 1: Train 2020–2021, Val H1 2022, Test H2 2022
- [ ] Fold 2: Train 2020–2022, Val H1 2023, Test H2 2023
- [ ] Fold 3: Train 2020–2023, Val H1 2024, Test H2 2024

### CV.2 Aggregation
- [ ] Report mean ± std WR across all 3 folds
- [ ] If all 3 folds WR ≥ 58%: model has genuine cross-regime generalization
- [ ] Document results in `outputs/cross_validation_report.md`
