# Tasks: SetupClassifier (MTPATSC)

## Phase 1 & 2: C++ Feature Engine (Heavy Lifting)
- `[x]` Define C++ `FeatureRow_MTPATSC` struct (ANCS Fine, ANCS Coarse, Context, Scalars).
- `[x]` Implement `scan_ticks` logic in C++ with Bid/Ask spread logic and limit fill verification.
- `[x]` Implement Multi-Label Priority Assignment (T2 > T4 > T1 > T3 > T0) in C++.
- `[x]` Compute 10-segment Fine ANCS and 5-segment Coarse ANCS natively in C++.
- `[x]` Calculate 15 Candle features and 25 Momentum features (accelerations, velocities, etc).
- `[x]` Add `boundary_proximity` calculation for conditional label smoothing.

## Phase 3 & 4: Python Tensor Builder & Scaling
- `[x]` Create `src/sc_tensor_builder.py` to ingest the C++ parquets.
- `[x]` Reshape C++ output into distinct branches: `ancs_fine (10, 6)`, `ancs_coarse (5, 6)`, `history (5, 5, 6)`.
- `[x]` Implement `RobustScaler` fitting on the Train split and transform Test/Eval splits. Save `scalar_scaler.pkl`.
- `[x]` Calculate Class Weights dynamically based on T0-T4 distribution in Train set.

## Phase 5 & 6: Keras Model & Dual-Objective Training
- `[x]` Construct multi-branch Keras model (`Conv1D` for ANCS, `TimeDistributed(Conv1D) -> LSTM` for Context).
- `[x]` Configure Dual-Objective Loss: Primary 5-class Softmax (CrossEntropy * 0.40) + Auxiliary 4-class Sigmoids (BCE * 0.15 each).
- `[x]` Implement Conditional Label Smoothing logic using `boundary_proximity`.
- `[x]` Train Model using EarlyStopping, ReduceLROnPlateau, and Walk-Forward chronological splits.

## Phase 7 & 8: Calibration & Evaluation
- `[x]` Write `src/sc_calibrate.py` to sweep probability thresholds for each class to maximize `EV * sqrt(n_trades)`.
- `[x]` Evaluate Limit Fill Sensitivity (simulate 70% to 100% fill rates on T2/T4) to find the breakeven point.
- `[x]` Output final `evaluation_report.json` with Top-2 Accuracy, EV distributions, and monthly regime stability.
