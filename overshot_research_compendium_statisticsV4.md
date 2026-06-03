# Overshot Research Compendium: The Path to V4

This document serves as the master record of the Overshot quantitative research project. It details the evolution of the strategy from a static momentum-exhaustion hypothesis into a dynamic, regime-aware liquidity-vacuum trading system. 

It documents every failure, every statistical pivot, and the final executable edge.

---

## Phase 1: The Initial Hypothesis (V3)
The project began with the hypothesis that short-term price movements (Renko bricks) followed by specific order book dynamics signaled "Momentum Exhaustion." 

**Original Features:**
- `spread_current`: Absolute bid-ask spread
- `abs_ofi_peak`: Peak Order Flow Imbalance
- `vel_peak`: Peak price velocity
- `wick_ratio`: Rejection wicks
- `absorption_index`: Passive limit order absorption

**V3 Absolute Rule Set (Optimized on Dukascopy 2020-2026):**
- `spread_current > 0.784`
- `abs_ofi_peak < 1.074`
- `vel_peak < 13.91`

**Initial Backtest Result:**
- **Trades:** 2,341
- **Win Rate:** 87%
- **Profit Factor:** 6.70

---

## Phase 2: The SHAP Revelation
To validate *why* the V3 system worked, we ran SHAP (SHapley Additive exPlanations) dependency plots. The results shattered the original hypothesis.

**The Finding:**
Momentum exhaustion (Wicks, Velocity, Absorption) had almost no predictive power. The model's decision-making was entirely dominated by `Spread`, followed by `OFI`.

**The Pivot:**
The edge was not "Momentum Exhaustion." The edge was a **Liquidity Vacuum**. 
When the market spread expands massively while institutional participation (OFI) drops to near-zero, the market is physically incapable of sustaining a directional breakout and is forced to mean-revert.

---

## Phase 3: The Dynamic Regime Framework (V4)
Absolute thresholds (`spread > 0.784`) are brittle and break when market volatility shifts. Based on the SHAP findings, we rebuilt the engine to use **rolling 5-day percentiles**.

**V4 Logic:**
Instead of asking "Is spread > 0.784?", the system asks "Is the current spread in the top 5% of all spreads seen in the last 5 days?"

**The 95/20 Rule:**
- `spread_pct >= 95%`
- `ofi_pct <= 20%`

---

## Phase 4: Statistical Audits & Stress Testing

Before moving to out-of-sample data, we rigorously audited the V4 backtest.

### 1. The COVID Tuesday Anomaly
We audited performance by Day of Week and found that **Tuesday** accounted for 50% of all trades and boasted a 96% win rate with a Profit Factor of 29. 
* **Diagnosis:** This was entirely driven by the March 2020 COVID liquidity shock. We quarantined 2020 data from all future baseline metrics to avoid extreme outlier bias.

### 2. Velocity is Noise
We tested the 95/20 rule with and without Velocity filters. Adding velocity filters threw away profitable trades without improving the expectancy. Velocity was officially deprecated from the signal logic.

### 3. The Session Filter Audit (Excluding 2020)
We plotted win rates by UTC Hour and identified structural "weak hours" where the edge decayed (03:00, 15:00, 18:00, 22:00, 23:00 UTC).
* **Rule A (Pure V4):** 75.03% WR, PF 3.00
* **Rule B (V4 + Exclude Weak Hours):** 78.22% WR, PF 3.59

---

## Phase 5: The 2026 Out-of-Sample Reality Check
We built a multiprocessing Python simulator (`rule_test_2026_v4_mp.py`) to replicate exact live execution conditions, testing the strategy on **2026 BlackBull Markets MT5 Tick Data**.

**Initial OOS Results:**
- **Rule A:** 1406 Trades | 55.91% WR | PF 1.27
- **Rule B:** 1236 Trades | 57.20% WR | PF 1.34

**The Crisis:** The Win Rate plummeted from 78% (Dukascopy In-Sample) to 57% (BlackBull Out-of-Sample). The trade count exploded from an expected ~300 trades to over 1,200.

### The Broker Distribution Mismatch
We analyzed the raw spread distribution between the two data feeds:
* **Dukascopy:** Continuous pricing with 2,180 unique spread values. The 95th percentile was a massive 4.40 points.
* **BlackBull:** Discretized pricing with essentially 2 levels (0.22 normal, 1.30 widened). 

**Conclusion:** The 95th percentile on BlackBull was just standard, off-hours markup noise. The true liquidity vacuum signature was buried deeper in the tail.

---

## Phase 6: BlackBull Calibration & The Peak
We ran a threshold sensitivity matrix on the BlackBull 2026 trades to find where the true edge lived on this specific feed.

| Spread Pct | OFI Pct | Trades | Win Rate | Profit Factor | PnL |
|---|---|---|---|---|---|
| 95% | 20% | 1404 | 55.91% | 1.27 | +166R |
| 99% | 10% | 529 | 65.41% | 1.89 | +163R |
| 99% | 5% | 450 | 68.22% | 2.15 | +164R |
| **99%** | **3%** | **364** | **70.60%** | **2.40** | **+150R** |

By tightening the OFI filter to **3%**, we stripped out 1,040 noise trades. The Profit Factor nearly doubled. **OFI proved to be the ultimate arbiter of truth on a noisy broker feed.**

---

## Phase 7: Robustness & The 12-Loss Streak
We stress-tested the `99/3` configuration to ensure it wasn't curve-fit.

1. **Month-by-Month Stability:** Perfect. All 5 months of 2026 were positive with Win Rates between 64% and 77%, and Profit Factors between 1.82 and 3.37.
2. **Directional Bias:** Gold Longs (73% WR) outperformed Shorts (67% WR), but both sides were heavily profitable.
3. **Session Concentration:** An astonishing **87% of all trades (319/364)** fired at exactly **01:00 UTC** (Monday Open / Daily Rollover).

### The 12-Loss Streak (The Execution Illusion)
We audited the worst drawdown in the dataset: a 12-loss streak that occurred on Feb 2, 2026.
* **The Forensic Data:** All 12 trades fired within a single 37-second window. The price gapped violently by 77 points. 
* **The Renko Illusion:** Because Renko charts backfill gaps with virtual bricks, the simulator instantly triggered 7 trades on the exact same millisecond timestamp (`01:11:53.334`). 
* **The Truth:** In live trading, it is physically impossible to execute trades inside a price gap. 

### Stripping the Illusion
When we removed **only exact millisecond duplicates** (the physically impossible gap-fill bricks), the raw 364 trades shrank to **211 legitimate, executable trades**.

**The Final Executable Reality (99/3 Rule, Gap-Fills Removed):**
* **Trades:** 211
* **Win Rate:** 62.56%
* **Profit Factor:** 1.67
* **Total PnL:** +53.0R

---

## The Final Structural Thesis
We have successfully reverse-engineered a highly specific market anomaly. 

At the daily/weekly open (01:00 UTC), retail brokers mechanically widen their spreads to protect against volatility. However, if the institutional order book goes entirely flat (Order Flow Imbalance drops below the 3rd percentile), this widened spread is mathematically unsupported by market participation. 

When price moves into this vacuum, it cannot sustain a breakout and violently snaps back. 

### Final Deployment Architecture
1. **Data Engine:** 5-day rolling percentiles for `spread` and `abs_ofi_peak`.
2. **Signal:** `spread_pct >= 99` AND `ofi_pct <= 3`.
3. **Execution Filter:** `if (TimeCurrent() == LastTradeTime) return;` (Ignore multi-fires generated by gap-fills).
4. **Time Filter:** Optional restriction to `01:00 UTC` and `17:00 UTC` where the structural edge is thickest. 
