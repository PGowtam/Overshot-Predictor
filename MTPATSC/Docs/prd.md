# Product Requirements Document (PRD): SetupClassifier (MTPATSC)

## 1. Product Objective
The `SetupClassifier` (Multi-Timeframe Price Action Trade Setup Classifier) is designed to evaluate the internal micro-structure of a Renko brick (using ANCS - Adaptive Normalised Candle Segments) to output a 5-class probability distribution. This distribution determines the mathematical likelihood of success across four strict structural trade setups (T1, T2, T3, T4) or a "No Trade" (T0) scenario.

## 2. Core Requirements
- **Input Foundation**: Tick data only (Bid, Ask, Timestamp). Zero volume dependencies.
- **Broker Agnosticism**: All features must be fully scale-invariant (normalised by `brick_size`) to transfer seamlessly across live and demo environments.
- **Execution-Realistic Labelling**: Win/Loss states must be calculated using exact Bid/Ask crossing mechanics to simulate spread decay and limit-order queue positioning.
- **Duration-Invariant Features**: Because Renko bricks vary massively in duration (1 min to 100+ mins), fixed time-candles (M1, M5) are useless. The engine must compute equal-tick-count segments (ANCS) to ensure temporal normalisation.
- **Dual-Objective Training**: The model will train a 5-class Softmax head alongside 4 auxiliary Binary Cross-Entropy heads to combat class starvation (e.g., T2 stealing T1's labels in strong momentum).

## 3. Execution Mechanics & Trade Geometry
Let `K` = brick_size (e.g., 0.00118).
Let `C` = brick close price.
Let `O` = brick open price.
Let `D` = direction of the just-closed brick (+1 for Bullish, -1 for Bearish).

### Type 1 (T1) — 1:1 Continuation
- **Setup**: Momentum continuation at market.
- **Buy Order (D = +1)**: Enter MARKET BUY at `C`. Take Profit at `C + K`. Stop Loss at `C - K`.
- **Sell Order (D = -1)**: Enter MARKET SELL at `C`. Take Profit at `C - K`. Stop Loss at `C + K`.

### Type 2 (T2) — 1:2 Pullback Continuation
- **Setup**: Waiting for a 1-brick pullback before continuing the trend.
- **Buy Order (D = +1)**: Place LIMIT BUY at `O` (bottom of the bullish brick). Take Profit at `O + 2K`. Stop Loss at `O - K`.
- **Sell Order (D = -1)**: Place LIMIT SELL at `O` (top of the bearish brick). Take Profit at `O - 2K`. Stop Loss at `O + K`.
- *Requirement*: Limit must be physically triggered. If price never touches `O`, trade is skipped (no fill).

### Type 3 (T3) — 1:2 Standard Reversal
- **Setup**: Fading the immediate close of the brick, betting on a full reversal.
- **Buy Order (D = -1)**: (Brick was Bearish). Enter MARKET BUY at `C`. Take Profit at `C + 2K`. Stop Loss at `C - K`.
- **Sell Order (D = +1)**: (Brick was Bullish). Enter MARKET SELL at `C`. Take Profit at `C - 2K`. Stop Loss at `C + K`.

### Type 4 (T4) — 1:3 Deep Reversal
- **Setup**: Betting on a massive, immediate reversal from the brick close.
- **Buy Order (D = -1)**: (Brick was Bearish). Enter MARKET BUY at `C`. Take Profit at `C + 3K`. Stop Loss at `C - K`.
- **Sell Order (D = +1)**: (Brick was Bullish). Enter MARKET SELL at `C`. Take Profit at `C - 3K`. Stop Loss at `C + K`.

## 4. Advanced Quantitative Constraints
1. **Limit Fill Sensitivity**: Must evaluate EV across varying fill rates (70% - 100%) to ensure T2 edge survives queue priority misses.
2. **Conditional Label Smoothing**: Near-boundary hits (e.g. hitting TP by 0.1 pips) must receive label smoothing, while clear 10K sweeps remain strictly 1-hot.
3. **Pre-Fusion Normalisation**: Scalar arrays must be normalised via `RobustScaler` before concatenation in Keras to prevent outlier values (`log_duration`) from blowing up the Dense layers.
