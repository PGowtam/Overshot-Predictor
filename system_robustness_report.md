# Future Architecture: Next-Generation System Robustness

While our current dual-head CNN+LSTM model achieves a phenomenal 90%+ holdout accuracy on Level 1 Gold data, the jump from "profitable backtest" to "profitable live execution" requires a shift from predictive accuracy to **systemic robustness**. 

Below is a detailed research report on the quantitative, architectural, and executional upgrades we can implement to push this system to institutional grade.

---

## 1. Data Dimension Upgrade: L2 Limit Order Book (LOB)

Currently, the Feature Engine relies on **Level 1 (Top of Book)** data (Bid, Ask, Bid Volume, Ask Volume). The limitation of L1 data is that Market Makers often spoof or quickly pull liquidity at the top node to trigger panic.

### The Upgrade: Real "Depth of Market" (L2) Encoding
Instead of just OFI at the top level, we ingest the top 10 levels of the Bid/Ask book.
*   **Volume Imbalance Convolution**: Apply a 1D convolution across the book depth. This allows the CNN to see "liquidity walls" resting at $2,400.50 that aren't visible in L1 data.
*   **Order Book Imbalance (OBI)**: Compute a weighted average of volume across all 10 levels, heavily prioritizing ticks that are "hitting the thick side" of the book.
*   **Research Paper**: *Nousi et al. (2019)* demonstrated that CNNs fed with L2 Limit Order Book images significantly outperform L1-only models in predicting mid-price movements.

## 2. Advanced Label Engineering: Adaptive Targets

Right now, Phase 1 (`label_generator.py`) generates `WIN/LOSS` labels based on a fixed ratio: `TP = 1 Brick, SL = 1 Brick`. 

### The Upgrade: ATR-Adjusted Targets
Gold's volatility expands and contracts. A $4.00 stop loss during the Asian session might be perfectly safe, but during the NY session, $4.00 is just noise.
*   **The Math**: Instead of $TP/SL = Brick\_Size$, use $TP = 1.2 \times ATR_{14}$ and $SL = 1.0 \times ATR_{14}$.
*   **The Impact**: The model stops trying to predict "1 brick moves" in high volatility, and starts predicting genuine momentum breakouts, reducing the number of trades stopped out by spread noise (whipsaws).

## 3. Dynamic Position Sizing (The Kelly Criterion)

Our current system is binary: If `Prob_Win > 0.5` and `Pred_OS > 1.6` $\rightarrow$ Trade 1.0 Lot. 

### The Upgrade: Output-Scaled Lot Sizing
We have two continuous outputs (`Prob_Win` ranging from 0.0 to 1.0, and `Pred_OS` ranging from 0.0 to X.X). We should use the Kelly Criterion to optimize capital growth.
$$ f^* = W - \frac{1-W}{R} $$
Where $W$ is the predicted probability of the win, and $R$ is the risk/reward ratio.
*   **Rule Engine**: 
    - If `Prob_Win = 0.60`, trade `0.10 Lots` (Micro position for marginal edge).
    - If `Prob_Win = 0.95` and `Pred_OS > 2.0`, trade `1.00 Lots` (Max size for sniper setup).
*   **Why it works**: You maximize returns during "hot streaks" (high confidence environments) and automatically scale down to preserve capital when the model is uncertain.

## 4. Regime Classification (Hidden Markov Models)

A single neural network can suffer from "catastrophic forgetting" if market behavior fundamentally shifts (e.g., transitioning from a high-inflation Fed tightening regime to a low-volatility easing regime).

### The Upgrade: The "Router" Architecture
We deploy a **Hidden Markov Model (HMM)** before the Neural Network that analyzes macro conditions (Rolling 1-day Volatility, Daily OFI Mean, VIX index).
*   The HMM classifies the market into `State 0 (Ranging)`, `State 1 (Trending)`, or `State 2 (News Chaos)`.
*   You train 3 separate CNN+LSTM models, one for each state. 
*   **Execution**: The HMM routes the current live tick to the correct expert model. This completely eliminates the need to find a "one-size-fits-all" threshold.

## 5. Execution Logic: Probability of Informed Trading (VPIN)

Spread and slippage are mostly caused by "Adverse Selection" — Market Makers widen the spread when they suspect "Informed Traders" (institutions) are aggressively buying.

### The Upgrade: Volume-Synchronized Probability of Informed Trading (VPIN)
*   **The concept**: Calculate the order flow toxicity. If VPIN spikes, it means the big players are sweeping the book, and liquidity is about to dry up.
*   **The Defense**: If `VPIN > 0.8`, the bot temporarily switches its `Pred_OS` threshold from $1.6 \rightarrow 2.5$. It demands an infinitely higher statistical edge to enter the market when the market is "toxic."

## 6. MLOps: Online Learning Framework

The model we validated scored 91% on 2024 data, but markets eventually drift. Retraining the entire 2020-2024 dataset from scratch every month is computationally heavy.

### The Upgrade: Continuous "Warm" Retraining
*   At the end of every trading week (Friday night), a script scrapes the MT5 ticks for the last 5 days.
*   It generates labels for those 5 days.
*   It takes the `model.keras` weights, unlocks the final Dense layers (keeping the CNN frozen), and trains for exactly **1 epoch** on the fresh data with a tiny learning rate ($1e^{-5}$).
*   **Result**: The core physics (CNN filters) stay permanent, but the decision boundaries (Dense layers) "slide" dynamically to match that specific week's regime.

---

### Priority Execution Roadmap
If you plan to scale this bot from a personal project to an institutional-grade system, execute these upgrades in this exact order:

1. **Dynamic Kelly Position Sizing** (Highest immediate impact on equity curve).
2. **Online Learning Pipeline** (Highest impact on 2026/2027 survival).
3. **ATR-Adjusted Labels** (Requires full pipeline modification).
4. **L2 Order Book Data** (Requires buying expensive historical data from tickdata.com or similar).
