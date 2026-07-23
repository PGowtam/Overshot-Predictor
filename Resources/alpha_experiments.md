# BrickOfTicks — Alpha Frontier: Definitive Experiment Catalogue

> **Baseline:** Volume Fallback Model — **93.12% OOS Win Rate**, 160 trades (6 months), +0.86R EV/trade at `Prob_Win ≥ 0.6, Pred_OS ≥ 1.7`, 1:1 TP:SL.
>
> **Objective:** Push WR toward theoretical certainty, maximize confident trade frequency, and discover orthogonal alpha sources within the BrickOfTicks microstructure framework.
>
> **Constraint:** All experiments are implementable using the existing offline pipeline (parquet tick data, label generator, feature engine, tensor builder, model training). No live data or MT5 connectivity required. **New scripts only** — no modifications to verified source code.

---

# I. ARCHITECTURAL EXPERIMENTS

Fundamental changes to model topology, inference timing, or state representation.

---

## EXP-01 · Multi-Resolution Renko State Confluence

**Category:** Architecture · Confirmation Gate
**Tags:** `▲ Win Rate` `⬡ Architecture`

### The Hypothesis

Renko bricks at different scales encode different regimes of the same underlying order flow. A Base brick (≈ $2.11) captures micro-momentum transitions. A Macro brick (2–3× Base) captures the structural trend regime. When both engines simultaneously predict the same direction with high confidence, the signal is drawn from two statistically independent vantage points of the same price process — a geometric proof of directional conviction.

### The Mechanic

1. After each anchor reset, instantiate two parallel `RenkoBuilder` objects: `base_engine` (brick_size = ATR) and `macro_engine` (brick_size = 2× ATR). Feed identical ticks to both.
2. At each Base brick close, snapshot the current micro-buffer (100 ticks, 9D) and run inference through the existing dual-head CNN+LSTM. Record `(Prob_Win_base, Pred_OS_base)`.
3. Simultaneously, query `macro_engine.state`: compute partial progress of the in-progress macro brick = `(mid − macro_open) / macro_size`. This is a continuous scalar `[0, 1]` encoding "how much of the macro move has already printed."
4. **Macro State Vector (4D):** At each Base brick close:
   - `macro_progress` — fractional completion of the current Macro brick
   - `macro_direction` — current Macro trend (+1 / −1 / 0)
   - `macro_streak` — consecutive Macro bricks in the same direction
   - `macro_alignment` — binary: `1` if Base direction == Macro direction
5. **Fusion:** Append the 4D Macro State to the existing 3D macro-vector, creating a **(10, 7) macro tensor**. The LSTM now sees both intra-brick dynamics AND the higher-resolution structural context.
6. Train a second lightweight "macro head" model (or use simple logistic regression) that takes: the macro progress scalar + last 10 macro-brick outcomes. This predicts `P(macro brick = WIN | current state)`.
7. **Confluence gate:** Only enter a base-scale trade if `(Prob_Win_base ≥ θ_base) AND (P_macro ≥ θ_macro) AND both predictions share the same direction`. Sweep `θ_macro` from 0.5 → 0.8 on the validation set.
8. **Label Correlation Study (do first):** Before any model training, stratify all Base bricks by `macro_alignment` and compute WR conditionally. If aligned bricks have a structurally higher WR, the signal is real.

### The Expected Edge

This is architecturally equivalent to requiring two independent hypothesis tests to reject the null simultaneously. If base-only WR is 93% and macro confirmation is ~70% correlated, the joint precision climbs toward the **96–98% regime**. The primary cost is reduced frequency — expect 30–50% fewer signals — but each surviving signal carries dramatically lower false-positive probability.

| Metric | Estimate |
| :--- | :--- |
| WR impact | +2–4% |
| Frequency impact | −30–50% |
| Complexity | 3/5 |
| Implementable offline | ✅ |

---

## EXP-02 · Intra-Brick Early-Exit Inference

**Category:** Architecture · Inference Timing
**Tags:** `▲ Frequency` `⬡ Architecture`

### The Hypothesis

Currently the model only infers at brick close. But the micro-buffer holds 100 ticks of continuous state — at 50% brick progress, the buffer already contains the causal structure of the current move. The information about the next brick outcome may crystallise well before the current brick closes. If we can detect the moment of maximum predictive certainty mid-brick, we can enter earlier, capturing more of the favorable move.

### The Mechanic

1. At each tick during an active brick, snapshot the current micro-buffer at 5 progress thresholds: `p ∈ {0.25, 0.40, 0.55, 0.70, 0.85}` of brick completion (Progress feature encodes this).
2. Run the trained model on each partial snapshot. Record the time-series of `(Prob_Win(t), Pred_OS(t))` across the 5 checkpoints for every brick in the dataset.
3. Compute the "**confidence emergence time**" — the first progress level at which the model's `Prob_Win` first crosses θ and `Pred_OS` first exceeds the calibrated threshold. Plot its distribution across WIN bricks.
4. Test the hypothesis: for the subset of WIN bricks where confidence emerges at `p < 0.7`, does early entry at `p` (vs. entry at `p=1.0`) provide a better execution price (closer to theoretical TP)? Measure `y_mag` improvement on early vs. late entries.
5. If early-entry improves EV: retrain a lightweight "confidence emergence detector" — a small LSTM that takes the `Prob_Win` time series as input and predicts the optimal entry time within the brick.

### The Expected Edge

Two compounding benefits: (1) earlier execution price → larger `y_mag` per trade → higher EV at 1:1 RR. (2) Some bricks that would trigger a full-close signal may begin their reversal before close — early exit saves capital. This doesn't increase trade count but improves the risk-adjusted return on each existing signal.

| Metric | Estimate |
| :--- | :--- |
| EV impact | Structural improvement |
| WR impact | Neutral to +1% |
| Complexity | 4/5 |
| Requires mid-brick inference | ⚠️ |

---

## EXP-03 · Latent Regime "Mixture of Experts" (MoE)

**Category:** Architecture · Regime-Aware Routing
**Tags:** `▲ Frequency` `▲ Win Rate` `⬡ Architecture`

### The Hypothesis

The current model assumes the 9D tick vector operates in a single, continuous market environment. It averages out distinct microstructural regimes. But the order book behaves fundamentally differently during a toxic institutional sweep versus passive market-making absorption. A global model rejects valid signals in high-noise regimes because it applies low-noise logic to them.

### The Mechanic

1. **Unsupervised Phase:** Train a Variational Autoencoder (VAE) on the 100-tick micro-buffers without labels. Force the VAE to cluster the raw L1 flow into *N* distinct hidden states (e.g., *N=4*: High-Toxicity, Mean-Reverting, Vacuum, Spoof-Heavy).
2. **Supervised Phase:** Implement a Mixture of Experts (MoE) architecture. Train 4 separate CNN+LSTM models, one for each regime.
3. **Live Routing:** At brick close, the routing network assigns the current micro-buffer to one of the 4 regimes and heavily weights that specific expert's `Prob_Win` and `Pred_OS` outputs.
4. **Validation:** Compare per-regime WR for the MoE vs the global model. Measure whether the MoE unlocks high-confidence signals in volatile regimes that the global model previously filtered out.

### The Expected Edge

Massive increase in trade frequency. By isolating market regimes, the model finds high-confidence signals in volatile environments that the global model threw out due to conflicting data. Each expert specializes in its regime's dynamics rather than compromising across all regimes.

| Metric | Estimate |
| :--- | :--- |
| Frequency impact | +30–60% |
| WR impact | +1–3% |
| Complexity | 5/5 |
| Training cost | 4× (one model per regime) |

---

## EXP-04 · Real-Time Survival Analysis — Time-to-Absorption Decay

**Category:** Architecture · Auxiliary Head
**Tags:** `▲ Win Rate` `⬡ Architecture`

### The Hypothesis

Renko removes the time dimension from price, which is its greatest strength. However, the *rate* at which time was removed contains alpha. If two identical Renko bricks form with the exact same OFI and Depth, but one took 400 milliseconds and the other took 14 seconds, the institutional urgency is completely different.

### The Mechanic

1. Add an auxiliary regression head to the LSTM (**Head C**).
2. Train this head to predict the **real-time duration in milliseconds** of the next brick.
3. Use a **Survival Analysis loss function** (Cox Proportional Hazards or log-normal AFT) instead of standard MSE, treating brick completion as the "death" event. This properly handles the censored, right-skewed nature of duration data.
4. **Filter:** Require `Pred_Duration` to be under a specific threshold (calibrated on validation set) to validate the trade. High `Prob_Win` + low `Pred_Duration` = the model is confident AND expects the next brick to form fast.
5. The existing macro-vector already contains `log(duration + 1)` — Head C makes this a first-class prediction target rather than an input feature, forcing the network to explicitly model the temporal dynamics of brick formation.

### The Expected Edge

High `Prob_Win` signals are often invalidated by sudden market shifts if the trade takes too long to play out. By filtering for predicted high-velocity absorption, you minimize time-in-market exposure, drastically reducing the chance of an exogenous shock ruining a perfectly modeled setup.

| Metric | Estimate |
| :--- | :--- |
| WR impact | +1–3% |
| Risk-adjusted improvement | Significant |
| Complexity | 3/5 |
| Requires architecture change | Head C addition only |

---

## EXP-05 · Dual-Model Ensemble with Orthogonality Test

**Category:** Architecture · Ensemble Engineering
**Tags:** `▲ Win Rate` `⬡ Architecture`

### The Hypothesis

The 3-fold CV ensemble achieves 91.02% WR (412 trades) vs. individual fold best of 94.65% (355 trades). Majority voting is a blunt instrument — it treats all fold disagreements equally. But the 3 fold models have structurally different training windows (2021, 2022, 2023 market regimes). Their errors may be partially orthogonal. If Fold A fails on regime X and Fold B succeeds, a confidence-weighted ensemble that assigns weights based on regime similarity should outperform naive voting.

### The Mechanic

1. **Regime Fingerprint (5D):** For each fold's test window, compute: `[mean_brick_duration, std_brick_duration, mean_spread, mean_OFI_magnitude, trend_ratio (up/total bricks)]`.
2. For the current inference window (trailing 20 bricks), compute the same 5D fingerprint. Compute **cosine similarity** between current conditions and each fold's test fingerprint.
3. **Similarity-Weighted Ensemble:**
   ```
   ensemble_score = Σ (sim_k × Prob_Win_k) / Σ sim_k
   ```
   This soft-weights fold models by how similar today's market is to the conditions where that fold proved itself.
4. **Orthogonality-Boosted Variant:** Identify which pairs of fold models have the lowest error overlap (correlation of binary prediction errors on holdout). Use only the most orthogonal pair when all 3 disagree.
5. Compare: (1) majority vote, (2) uniform average, (3) similarity-weighted, (4) orthogonality-boosted. Metric: `WR × √(trade_count)` to penalize precision-only strategies.

### The Expected Edge

The Condorcet Jury Theorem states that if individual predictors have `p > 0.5` accuracy and their errors are independent, an ensemble converges to certainty. Maximising error independence through regime-aware weighting is a mathematically sound path to pushing ensemble WR above 95% without retraining.

| Metric | Estimate |
| :--- | :--- |
| WR impact | +1–3% via smarter weighting |
| Frequency impact | Neutral |
| Complexity | 4/5 |
| No retraining needed | ✅ |

---

# II. FEATURE ENGINEERING EXPERIMENTS

New information channels derived from existing tick data, injected as features or gates.

---

## EXP-06 · OFI Autocorrelation Decay Signature

**Category:** Features · Second-Order Statistics
**Tags:** `▲ Win Rate` `Novel Feature Dimension`

### The Hypothesis

The CNN encoder treats each tick's `z_OFI` as an independent signal. But OFI at the microstructure level exhibits short-term positive autocorrelation (order flow momentum) followed by negative autocorrelation (mean reversion from market makers absorbing flow). The *shape* of the OFI autocorrelation function (ACF) over the 100-tick buffer may distinguish "genuine momentum bricks" (strong positive ACF decay) from "noise bricks" (flat or oscillating ACF). This is signal the CNN cannot currently extract because it lacks a global temporal summary.

### The Mechanic

1. For each brick snapshot, extract the 100-tick `z_OFI` sequence. Compute autocorrelation at lags `τ ∈ {1, 2, 3, 5, 8, 13, 21}` (Fibonacci lags to span micro → meso timescales). This yields a **7D OFI-ACF vector**.
2. Compute the analogous ACF vector for `z_Spread` (spread autocorrelation captures market maker quote strategy). Another **7D vector**.
3. Augment the existing 3D macro-vector to a **17D macro-vector** (7 OFI-ACF + 7 Spread-ACF + 3 existing). Retrain the LSTM fusion stage only (freeze CNN weights) to evaluate marginal contribution.
4. **Statistical Test:** Do bricks with monotonically decaying positive OFI-ACF (`ρ_1 > ρ_2 > ... > 0`) have significantly higher WIN rates than bricks with oscillating ACF? Run chi-squared across quartiles.
5. If validated, add the ACF vectors as a third input head or as additional features in the LSTM fusion layer. Sweep contribution weight in ablation study.

### The Expected Edge

The ACF structure is a **second-order statistic** — it compresses information about the temporal structure of order flow that point-by-point CNN processing cannot capture. This is a genuinely new feature dimension, with zero data collection cost.

| Metric | Estimate |
| :--- | :--- |
| WR impact | Potentially high (+2–5%) |
| New data required | None — computed from existing buffer |
| Complexity | 4/5 |

---

## EXP-07 · Cross-Brick OFI Momentum Persistence Scoring

**Category:** Features · Flow Continuity
**Tags:** `▲ Win Rate` `▲ Frequency`

### The Hypothesis

The model's micro-buffer is continuous (never resets at brick boundaries), but the LSTM processes brick-level summaries. This means the model cannot directly measure whether the directional OFI impulse that formed the *previous* brick is still alive at the *current* brick's close. If OFI momentum persists across the boundary, continuation is structurally more likely. If OFI has already reversed at the boundary, the brick was a terminal move.

### The Mechanic

1. At each brick close, compute the mean `z_OFI` of the last 20 ticks of the *previous* brick and the first 20 ticks of the *current* brick.
2. **OFI Persistence Score:** `persistence = sign(mean_OFI_prev) × mean_OFI_curr`. Positive = flow continues across boundary. Negative = flow reversed.
3. **OFI Gradient:** `gradient = mean_OFI_curr - mean_OFI_prev`. Positive gradient = accelerating flow momentum.
4. Add `persistence` and `gradient` to the macro-vector. Test conditional WR across persistence quartiles.
5. **Interaction Term:** Test `persistence × Prob_Win` as a gating signal — high persistence + high Prob_Win should be the highest-conviction combination.

### The Expected Edge

This directly measures whether the *cause* of the brick (directional order flow) is still active or has already exhausted. The current model sees tick-level OFI but cannot reason about cross-brick flow continuity. This is the first feature that explicitly captures the **causal chain** across brick boundaries.

| Metric | Estimate |
| :--- | :--- |
| WR impact | +1–3% |
| Frequency impact | +5–10% (unlocks marginal trades with strong persistence) |
| Complexity | 2/5 |

---

## EXP-08 · Sequence Entropy as a Regime Detector

**Category:** Features · Information Theory
**Tags:** `▲ Win Rate` `Regime Classification`

### The Hypothesis

The Renko `sequence` field (binary string of `1`s and `0`s encoding the last 100 brick directions) contains **information-theoretic structure** that the current model ignores. A sequence like `1111111111` (pure trend) has zero Shannon entropy. A sequence like `1010101010` (alternating) has maximum entropy. The model receives brick directions implicitly through the LSTM, but the *global entropy* is a potent meta-feature the LSTM cannot trivially extract.

### The Mechanic

1. **Entropy Computation:** At each brick close, compute Shannon entropy of the last N bricks (sweep N = 10, 20, 50):
   ```python
   from collections import Counter
   from math import log2
   
   def sequence_entropy(seq: str, window: int = 20) -> float:
       chunk = seq[-window:]
       if len(chunk) < 2:
           return 0.0
       counts = Counter(chunk)
       total = len(chunk)
       return -sum((c/total) * log2(c/total) for c in counts.values())
   ```
2. **Conditional WR Study:** Stratify model-filtered trades by entropy buckets. Test: does WR at `entropy < 0.5` significantly exceed WR at `entropy > 0.8`?
3. **Feature Integration:** Add `sequence_entropy(window=20)` as the 4th element of the macro-vector → **(10, 4) macro tensor**.
4. **Alternative — Run-Length Encoding:** Mean run length of last 20 bricks. Run-length > 5 → strong trend. Run-length ≈ 1 → maximum chop.

### The Expected Edge

Base rate analysis showed `pre-streak = 0` (reversals) had the lowest 2RR WR (14.9%) while `streak ≥ 7` had the highest (20.3%). Entropy formalizes this — bricks in low-entropy regimes should have 5–15% higher continuation WR.

| Metric | Estimate |
| :--- | :--- |
| WR impact | +1–3% |
| Complexity | 1/5 |
| Zero new data | ✅ |

---

## EXP-09 · Tick Arrival Process — The Hawkes Process Intensity Feature

**Category:** Features · Point Process Theory
**Tags:** `▲ Win Rate` `Novel Feature Dimension`

### The Hypothesis

In tick-driven markets, the **temporal clustering of tick arrivals** is itself informative. During strong momentum, ticks cluster in bursts (self-exciting behavior). During consolidation, ticks arrive at a more uniform Poisson rate. This is formalized by the **Hawkes process**: a self-exciting point process where the intensity increases after each event and decays exponentially. The kernel parameters (`μ` = baseline intensity, `α` = self-excitation, `β` = decay rate) directly quantify the degree of momentum self-reinforcement.

### The Mechanic

1. **Hawkes Intensity Estimation:** At each brick close, estimate instantaneous Hawkes intensity using the last 200 tick timestamps:
   ```python
   def hawkes_intensity(timestamps_ms, alpha=0.5, beta=0.001):
       t_now = timestamps_ms[-1]
       intensity = 0.0
       for t in timestamps_ms[:-1]:
           intensity += alpha * math.exp(-beta * (t_now - t))
       return intensity
   ```
2. **Branching Ratio:** `n* = α/β`. If `n* > 1` → **supercritical** (tick arrivals accelerating). If `n* < 1` → **subcritical** (returning to baseline). Bricks closing during supercritical phases ride genuine momentum self-excitation.
3. **Integration:** Add to macro-vector: `hawkes_intensity_zscore` + `branching_ratio`.
4. **Conditional WR Study:** Stratify by branching ratio quartiles.

### The Expected Edge

This is the most mathematically rigorous measure of whether momentum is *self-sustaining* versus *decaying*. Captures information invisible to `z_Vel` — not just speed, but whether each tick is triggering more ticks. Literature: Bacry et al. 2015, Rambaldi et al. 2017.

| Metric | Estimate |
| :--- | :--- |
| WR impact | +2–5% |
| Complexity | 4/5 |
| Novel theoretical grounding | ✅ |

---

## EXP-10 · Intra-Brick Velocity Wavelet — The Momentum Fingerprint

**Category:** Features · Shape Descriptors
**Tags:** `▲ Win Rate` `Novel Feature Dimension`

### The Hypothesis

Not all bricks that form in the same duration have the same microstructure quality. A 30-second brick where all movement happens in the first 5 seconds (front-loaded impulse) has fundamentally different continuation dynamics than one where price steadily marches for 30 seconds (constant velocity). The current model sees `z_Vel` per tick but cannot distinguish velocity *profiles*.

### The Mechanic

1. For each brick, extract the raw velocity sequence. Normalize to 20 bins (evenly spaced in time).
2. **Wavelet Decomposition (Haar, db2):** Single-level DWT producing approximation (10 values) and detail (10 values) coefficients.
3. **Summary Statistics (5D):**
   - `vel_skew` — Front-loaded vs back-loaded momentum
   - `vel_kurtosis` — One sharp burst vs distributed acceleration
   - `detail_energy` — L2 norm of detail coefficients (turbulence)
   - `vel_slope` — Linear regression slope (accelerating vs decelerating)
   - `vel_max_ratio` — Position (0–1) of maximum velocity (impulse vs climax)
4. Append 5D fingerprint to macro-vector → **(10, 8) macro tensor**.

### The Expected Edge

Front-loaded impulse bricks represent aggressive institutional positioning. Back-loaded bricks represent retail chasing. Velocity *shape* is a second-order refinement of duration. Expect same-duration bricks stratified into +/− 3–5% WR buckets.

| Metric | Estimate |
| :--- | :--- |
| WR impact | +2–4% |
| Complexity | 4/5 |
| Invisible to current model | ✅ |

---

# III. CALIBRATION & GATING EXPERIMENTS

Post-hoc corrections, threshold engineering, and hard pre-filters that require no retraining.

---

## EXP-11 · Adversarial Confidence Calibration — Temperature Scaling

**Category:** Calibration · Probability Correction
**Tags:** `▲ Frequency` `Zero Cost`

### The Hypothesis

Neural network sigmoid outputs are notoriously **miscalibrated**. A model that says "80% confident" might empirically be right 70% or 90% of the time. If our model is *underconfident*, we're leaving money on the table by requiring thresholds that are stricter than necessary. **Proper calibration turns the output into a true probability**, enabling mathematically optimal threshold selection.

### The Mechanic

1. **Reliability Diagram:** Bin model outputs into 20 bins. For each bin compute empirical WR. Plot predicted probability vs empirical WR.
2. **Expected Calibration Error (ECE):**
   ```
   ECE = Σ (|bin_n| / N) × |accuracy_n − confidence_n|
   ```
3. **Temperature Scaling:** Learn a single scalar `T` on the validation set:
   ```python
   calibrated_prob = sigmoid(logit(raw_prob) / T)
   ```
   Optimize `T` to minimize NLL. No retraining needed.
4. **Platt Scaling Alternative:** 2-parameter logistic regression for asymmetric miscalibration.
5. **Re-Threshold After Calibration:** With calibrated probabilities, re-sweep thresholds. Optimal threshold may shift — potentially more trades at the same true WR.

### The Expected Edge

If underconfident (predicting 0.55 when true probability is 0.65), calibration allows lowering `Prob_Win_threshold` from 0.6 to 0.5, capturing more trades at the same true confidence. Also enables **Kelly-criterion position sizing**: `EV = P_cal × TP − (1 − P_cal) × SL`.

| Metric | Estimate |
| :--- | :--- |
| Frequency impact | +10–20% trades |
| WR impact | 0% (probability corrected, not improved) |
| Complexity | **1/5** |
| Zero retraining | ✅ |

---

## EXP-12 · Spread Microstructure as a Real-Time Risk Signal

**Category:** Gating · Veto Layer
**Tags:** `▲ Win Rate` `▲ Frequency`

### The Hypothesis

Spread carries `r = −0.097` correlation with `y_class`. But this is a point-in-time measurement. The *dynamics* of spread during the brick — whether it was contracting (market makers gaining conviction), expanding (uncertainty rising), or stable — carry a much stronger signal. A widening spread in the final 10 ticks is a market maker withdrawing liquidity, a leading indicator of adverse price action. Forensics confirmed: **losing high-confidence trades had 4.3× higher z_Spread** than winners.

### The Mechanic

1. From the 100-tick micro-buffer at brick close, extract the `z_Spread` time series. Fit a linear regression of `z_Spread` on tick index for the last 25 ticks: slope `β_spread = dSpread/dt`.
2. Categorise into 3 spread-dynamic states: **CONTRACTING** (`β_spread < −threshold`), **STABLE** (`|β_spread| ≤ threshold`), **EXPANDING** (`β_spread > threshold`). Sweep threshold.
3. Test: is `y_class` significantly different across spread-dynamic states? Hypothesis: CONTRACTING bricks have WR ≥ 5% higher than EXPANDING bricks.
4. **New Feature:** Add `β_spread` as a 10th feature to the tick vector, or as an additional macro-vector scalar.
5. **Interaction Term:** `z_OFI × β_spread`. Negative values (positive OFI + contracting spread) = market makers simultaneously absorbing flow and tightening quotes = highest conviction. This is a genuine second-order feature the CNN doesn't encode.
6. **Hard Veto:** Refuse to trade EXPANDING spread bricks regardless of model output.
7. **Spread Regime Classification (Extended):**
   - `TIGHT` — spread < 25th percentile of trailing window
   - `NORMAL` — 25th to 75th
   - `WIDE` — 75th to 95th
   - `TOXIC` — > 95th percentile

### The Expected Edge

If EXPANDING spread veto eliminates 5% of trades but those 5% had 60% WR (vs 93% average), filtered WR climbs to ~94.5% with minimal frequency loss. The `z_OFI × β_spread` interaction term may be the **single strongest predictive tick-level feature** not currently in the model.

| Metric | Estimate |
| :--- | :--- |
| WR impact | +1–4% |
| New feature dimension | ✅ |
| Complexity | 2/5 |

---

## EXP-13 · Volatility-Regime Conditional Gating

**Category:** Gating · Adaptive Thresholds
**Tags:** `▲ Win Rate` `▲ Frequency`

### The Hypothesis

A fixed `Pred_OS ≥ 1.7` threshold assumes signal quality is stationary across regimes. But during low-volatility sessions (narrow spreads, slow bricks), the model may achieve 95%+ precision at lower thresholds. During high-volatility events (news, London/NY overlap), the same output values may be less reliable. **Conditioning thresholds on the current volatility regime should recover trades lost to over-filtering in quiet markets.**

### The Mechanic

1. **Realised Volatility Proxy:** At brick close, compute `RV = rolling 20-brick standard deviation of log(brick_duration)`. This is a direct measure of regime stability already derivable from the existing macro-vector.
2. **Cluster into 3 regimes** (K-means or quantile-based): LOW (bottom 33%), MID (middle 33%), HIGH (top 33%).
3. For each regime, run a **separate precision-recall calibration sweep** on `(Prob_Win, Pred_OS)`. Derive regime-specific optimal thresholds: `(θ_prob_low, θ_OS_low)`, `(θ_prob_mid, ...)`, `(θ_prob_high, ...)`.
4. At inference time: compute current RV → classify regime → apply regime-specific thresholds. This is a **3-row lookup table**, adding zero latency.
5. Measure improvement: frequency + WR of adaptive-threshold strategy vs fixed thresholds across each regime slice.

### The Expected Edge

In LOW volatility: lower thresholds unlock trades currently excluded by over-conservative filtering (+20–40% frequency without WR sacrifice). In HIGH volatility: tighter thresholds reduce false positives. Net effect: a volatility-aware edge operating closer to the Pareto frontier of WR × frequency.

| Metric | Estimate |
| :--- | :--- |
| Frequency impact | +20–40% in low-vol |
| WR impact | +1–2% via reduced high-vol FPs |
| Complexity | 2/5 |
| Trivial inference cost | ✅ |

---

# IV. SIGNAL DIVERSITY & FREQUENCY EXPERIMENTS

Methods to increase the number of actionable signals without diluting precision.

---

## EXP-14 · Baiting Strategy Inversion + Symmetry Exploit

**Category:** Signal Diversity · Dual-Lobe Trading
**Tags:** `▲ Frequency (4–6×)` `▲ Win Rate`

### The Hypothesis

The model's failure mode is not random — it concentrates in a specific region of output space (`Prob_Win < 0.2, Pred_OS < 0.7`). This region already yields **88.75% win rate on reversal trades** in 2024 holdout (1,120 trades). The existing "baiting" result is being left on the floor. The true exploit is treating the model as a **directional classifier with two high-confidence lobes**: the WIN lobe (top-right of output space) and the LOSS lobe (bottom-left), each independently tradeable.

### The Mechanic

1. Define a symmetric output space partition:
   - `Prob_Win ∈ (0, 0.2] ∩ Pred_OS ∈ (0, 0.7]` → **REVERSAL signal**
   - `Prob_Win ∈ [0.6, 1.0] ∩ Pred_OS ∈ [1.7, ∞)` → **STANDARD signal**
2. **Map the output space** as a 2D heatmap on the 2024 holdout: x=Prob_Win, y=Pred_OS, color = empirical WR in each cell. Identify exact boundary contours of both high-precision lobes.
3. Fit a **logistic boundary classifier** (or GMM) to precisely delineate the two lobes. Replace hard threshold rules with continuous "lobe membership probability" score.
4. **For the REVERSAL signal:** trade direction is the *opposite* of the primary model's predicted direction. If Renko brick was uptrend, bet DOWN. Entry, TP, SL sizing identical.
5. Backtest the combined strategy (STANDARD + REVERSAL) with the constraint that both signals cannot fire simultaneously. Track combined EV, drawdown, and frequency.

### The Expected Edge

Standard signals: ~160 trades / 6 months. Reversal signals: ~1,120 trades / 6 months in the broader window. Even at conservative filtering, this is a **5–7× increase** in actionable signal count with no new training data required. The edge is already proven in holdout; formalising the boundary is a calibration exercise, not a research gamble.

> **CAUTION:** The earlier feasibility study explicitly recommended "Kill the Baiting Strategy" due to choppy-market risks. This experiment must prove that the LOSS lobe is genuine by testing on the *exact* execution-priced labels and incorporating the spread veto from EXP-12. If `z_Spread > 2.0` during a reversal signal, the signal is almost certainly chop, not a trapped counterparty.

| Metric | Estimate |
| :--- | :--- |
| Frequency impact | **+4–6×** |
| WR on reversals | ~88–93% |
| Complexity | 2/5 |
| Risk | Must validate with spread veto |

---

## EXP-15 · Adverse Selection "Baiting" — The Retail Trap Network

**Category:** Signal Diversity · Adversarial Modeling
**Tags:** `▲ Win Rate` `▲ Frequency`

### The Hypothesis

The most lucrative trades occur when someone else is *forced* to liquidate. We want to predict not just when we will win, but when the counterparty is demonstrably trapped. High-probability continuations often follow a specific structural anomaly: price moving *against* the dominant Order Flow Imbalance, luring retail into a false breakout before institutional flow slams the door.

### The Mechanic

1. **Create a specialized label subset — "The Trap":** These are bricks where the standard model's continuation prediction fails spectacularly (immediate 1-brick reversal causing a rapid SL trigger). Extract the feature signatures of these traps.
2. **Train a secondary, adversarial "Trap Network"** strictly to identify these localized liquidity traps using the **divergence between `z_Vel` and `z_OFI`** — specifically: high velocity + low OFI = fast price movement on thin flow = retail chasing.
3. **Execution Logic:** Only trigger a trade when the Primary Network predicts a WIN *and* the Trap Network predicts the opposing side is about to be stopped out.
4. **Cross-validation:** The Trap Network must be validated on completely separate data from the Primary Network. Use walk-forward splits to prevent information leakage.

### The Expected Edge

This pushes win rate closer to the theoretical limit by only taking trades backed by forced liquidation. It transforms "uncertainty" in the primary model into a high-confidence reversal signal. The `z_Vel / z_OFI` divergence is a well-studied market microstructure indicator of toxic flow.

| Metric | Estimate |
| :--- | :--- |
| WR impact | +2–5% on filtered signals |
| Frequency impact | Reduction (highly selective) |
| Complexity | 4/5 |
| Novel concept | ✅ |

---

## EXP-16 · Conditional Markov Chain on Brick Sequences — The Memory Model

**Category:** Signal Diversity · Pattern-Based Boosting
**Tags:** `▲ Win Rate` `Ensemble Orthogonality`

### The Hypothesis

Renko sequences are not i.i.d. The probability of continuation depends on the **specific pattern** of the last K bricks, not just the count. `11110` has a very different continuation probability than `10110`. The LSTM implicitly learns these patterns, but we can build an **explicit conditional probability table** from 30,000 bricks and use it for Bayesian fusion.

### The Mechanic

1. **K-th Order Markov Model:** For each `K ∈ {3, 4, 5, 6, 7, 8}`, build:
   ```
   P(next=1 | last_K = pattern) = count(pattern + "1") / count(pattern + *)
   ```
   K=8 gives 256 patterns, ~117 samples each (sufficient).
2. **Transition Probability Feature:** At each brick close, look up the K-order Markov continuation probability. Single scalar `[0, 1]`.
3. **Bayesian Fusion:**
   ```
   P_fused = (P_nn × P_markov) / (P_nn × P_markov + (1−P_nn) × (1−P_markov))
   ```
   Sharper when both agree, uncertain when they disagree.
4. **Wilson Score CIs:** For each K-order pattern, compute 95% confidence intervals. Patterns with CI lower bound > 0.6 are "structural setups." Patterns with CI upper bound < 0.4 are structural reversals.

### The Expected Edge

The Markov model learns from a completely different information source (sequence structure) than the CNN+LSTM (tick microstructure). Their agreement is a very strong compound signal. Specific 5–8 brick patterns with 70%+ continuation probability would be structural alpha — independent of order flow.

| Metric | Estimate |
| :--- | :--- |
| WR impact | +1–3% |
| Frequency impact | Neutral to +5–15% on targeted patterns |
| Complexity | 2/5 |

---

# V. RISK-REWARD ENGINEERING

Methods to increase expected value per trade without affecting win rate.

---

## EXP-17 · Adaptive Risk-Reward via Pred_OS Targeting

**Category:** Risk-Reward · TP Optimization
**Tags:** `▲ EV (+20–50%)` `WR Unchanged`

### The Hypothesis

We use fixed 1:1 TP:SL, but `Pred_OS` already predicts the expected magnitude. A brick with `Pred_OS = 3.2` can travel 3.2× brick_size — yet we cap profit at 1×. We're systematically leaving money on the table. **Dynamically setting TP as a function of Pred_OS** captures more from high-conviction setups.

### The Mechanic

1. **Optimal TP Mapping:** For each `Pred_OS` bucket (1.0–1.5, 1.5–2.0, ..., 3.0+), compute the empirical `y_mag` CDF. Find the TP multiplier maximizing:
   ```
   EV(tp_mult) = P(y_mag ≥ tp_mult) × tp_mult − P(y_mag < tp_mult) × 1.0
   ```
2. **Monte Carlo Validation:** Simulate 10,000 sessions on the test set with dynamic TP. Compare total PnL against fixed 1:1 and fixed 1:2.
3. **Implementation:**
   ```python
   def dynamic_tp(pred_os: float, brick_size: float) -> float:
       tp_multiplier = min(max(pred_os * 0.7, 1.0), 3.0)
       return brick_size * tp_multiplier
   ```
   The `0.7` haircut accounts for model noise (Pearson r ≈ 0 but Spearman ρ may be significant).
4. **Kelly Sizing:** With dynamic TP, each trade has different RR. Apply Kelly:
   ```
   f* = (p × b − q) / b   where b = TP/SL, p = calibrated_prob, q = 1−p
   ```
5. **Spearman ρ Test (do first):** Before implementing, compute Spearman rank correlation between `Pred_OS` and `y_mag` on the test set. If ρ > 0.1 (even though Pearson r ≈ 0), the ordinal ranking is sufficient for dynamic TP to work.

### The Expected Edge

If average winner increases from +1.0R to +1.3R while maintaining 93% WR, net EV jumps from +0.86R to **+1.14R per trade** — a 32% profitability increase with zero change to trade selection.

| Metric | Estimate |
| :--- | :--- |
| WR impact | 0% (unchanged) |
| EV impact | **+20–50%** |
| Complexity | 2/5 |

---

## EXP-00A · Imbalanced Focal Loss Training (Replaces Undersampling)

**Category:** Loss Engineering · Class Imbalance
**Tags:** `▲ Calibration` `Highest Priority`

### The Hypothesis
Majority Class Undersampling destroys the model's calibration to the true market prior (35% win rate). A model trained on 50/50 data will predict mathematically invalid real-world probabilities. Instead of altering the dataset, we must alter the penalty using **Focal Loss**. Focal Loss dynamically scales down the loss based on prediction confidence, preventing the overwhelming majority class (failed breakouts) from dominating the gradients without destroying the true prior.

### The Mechanic
1. Restore the original imbalanced training set (35/65).
2. Replace `BinaryCrossentropy` with Focal Loss: $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$.
3. Set $\gamma = 2.0$ to force the model to focus on hard-to-classify winning breakouts.
4. Train the model and evaluate. Max probability output of 0.45 is acceptable—it means the model is perfectly calibrated and indicating the best setup has a 45% chance of success.

---

## EXP-00B · Target Discretization (y_mag Bucketization)

**Category:** Target Engineering · Regression Fix
**Tags:** `▲ Calibration` `Highest Priority`

### The Hypothesis
Financial time-series data is characterized by fat tails and extreme noise. The continuous target (`y_mag`) is highly right-skewed. Using Huber loss on this data forces the network to simply predict the median of the distribution to minimize errors (flatlining at ~0.79). Discretizing the target into bins converts the noisy regression problem into a stable classification problem, allowing the model to look for broader momentum regime signatures.

### The Mechanic
1. Convert `y_mag` into categorical bins:
   - Bin 0: < 0.5 R (Immediate failure)
   - Bin 1: 0.5 R to 1.0 R (Struggle)
   - Bin 2: 1.0 R to 2.0 R (Standard Win)
   - Bin 3: > 2.0 R (Runner)
2. Replace the `Pred_OS` regression head with a `Dense(4, activation='softmax')` multi-class classification head.
3. Use `CategoricalCrossentropy` loss.

---

## EXP-00C · Temporal Attention / Causal TCN (Replaces CNN+LSTM)

**Category:** Architecture · Temporal Resolution
**Tags:** `▲ Win Rate` `Highest Priority`

### The Hypothesis
The current CNN + MaxPool on micro-tick data destroys valuable temporal information. MaxPool is translation-invariant (it blurs exactly *when* an event occurred). Furthermore, LSTMs struggle with long, noisy sequences (100 ticks) because their hidden state gets diluted. We need an architecture that preserves exact temporal resolution and causality.

### The Mechanic
1. **Attention Mechanism:** Replace the LSTM with a Transformer/Attention layer that can explicitly "pay attention" to the exact moment a high-spread, high-velocity order flow imbalance occurred without losing its sequence position.
2. **Causal TCN:** Replace the CNN+MaxPool block with a Temporal Convolutional Network (TCN) using causal, dilated convolutions (no pooling). This preserves the exact temporal hierarchy while expanding the receptive field.

---

# VI. MASTER PRIORITIZATION MATRIX

| Priority | ID | Experiment | Effort | WR Impact | EV / Freq Impact | Category |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | EXP-00A | Imbalanced Focal Loss Training | 🟢 1/5 | Calibration | True Prior Restored | Loss Eng. |
| **2** | EXP-00B | Target Discretization (Buckets) | 🟢 2/5 | Calibration | Unlocks Pred_OS | Target Eng. |
| **3** | EXP-00C | Temporal Attention / TCN | 🔴 5/5 | High | Structural | Architecture |
| **4** | EXP-11 | Confidence Calibration | 🟢 1/5 | 0% | +10–20% freq | Calibration |
| **5** | EXP-12 | Spread Microstructure Veto | 🟢 2/5 | +1–4% | Minor freq loss | Gating |
| **6** | EXP-08 | Sequence Entropy | 🟢 1/5 | +1–3% | Neutral | Feature |
| **7** | EXP-13 | Volatility-Regime Gating | 🟢 2/5 | +1–2% | +20–40% in low-vol | Gating |
| **8** | EXP-17 | Adaptive TP via Pred_OS | 🟡 2/5 | 0% | **+20–50% EV** | Risk-Reward |
| **9** | EXP-16 | Markov Chain Sequences | 🟡 2/5 | +1–3% | +5–15% targeted | Signal |
| **10** | EXP-07 | Cross-Brick OFI Persistence | 🟡 2/5 | +1–3% | +5–10% freq | Feature |
| **11** | EXP-14 | Baiting Inversion (Dual-Lobe) | 🟡 2/5 | − | **+4–6× freq** | Signal |
| **12** | EXP-01 | Multi-Resolution Confluence | 🟡 3/5 | +2–4% | −30–50% freq | Architecture |
| **13** | EXP-05 | Ensemble Orthogonality | 🟡 4/5 | +1–3% | Neutral | Architecture |
| **14** | EXP-04 | Survival Analysis (Head C) | 🟡 3/5 | +1–3% | Risk-adjusted | Architecture |
| **15** | EXP-02 | Early-Exit Inference | 🔴 4/5 | +1% | Structural EV | Architecture |
| **16** | EXP-06 | OFI Autocorrelation Decay | 🔴 4/5 | +2–5% | Novel dimension | Feature |
| **17** | EXP-09 | Hawkes Process Intensity | 🔴 4/5 | +2–5% | Novel dimension | Feature |
| **18** | EXP-10 | Velocity Wavelet Fingerprint | 🔴 4/5 | +2–4% | Neutral | Feature |
| **19** | EXP-15 | Trap Network (Adversarial) | 🔴 4/5 | +2–5% | Highly selective | Signal |
| **20** | EXP-03 | Mixture of Experts (MoE) | 🔴 5/5 | +1–3% | +30–60% freq | Architecture |

### Recommended Execution Phases

**Phase 0 — Core Fixes (Immediate):**
`EXP-00A` → `EXP-00B` → `EXP-00C`
Address the fundamental calibration and structural bottlenecks identified by ML expert review.

**Phase A — Quick Wins (Week 1):**
`EXP-11` → `EXP-12` → `EXP-08` → `EXP-13`
Nearly free. No retraining. Potential cumulative impact: +2–6% WR, +20–40% frequency.

**Phase B — EV Engineering (Week 2):**
`EXP-17` → `EXP-16` → `EXP-07`
Moderate effort. Major EV uplift. Potential: +20–50% EV per trade.

**Phase C — Signal Multiplication (Week 3):**
`EXP-14` → `EXP-01` → `EXP-05`
Requires careful validation. Potential: 4–6× signal count via dual-lobe, +2–4% WR via multi-resolution.

**Phase D — Deep Research (Ongoing):**
`EXP-06` → `EXP-09` → `EXP-10` → `EXP-15` → `EXP-03`
High ceiling, high effort. Novel feature dimensions and architectural changes.

---

## Implementation Protocol

All experiments must follow the established codebase pattern:

1. **New scripts only** — no modifications to existing verified source code
2. **Output to dedicated directories** — `outputs/experiments/{exp_id}/`
3. **Conditional WR analysis first** — before any model retraining, validate the signal statistically with chi-squared or Mann-Whitney U tests
4. **Walk-forward evaluation** — always evaluate on the July–Dec 2023 test set with the same `y_class` labels
5. **Document results** — each experiment generates a `{exp_id}_report.json` with metrics and a `{exp_id}_analysis.md` with findings
6. **Abort criteria** — if the conditional WR study shows < 1% stratification between target groups, the feature is noise — do not proceed to model integration
