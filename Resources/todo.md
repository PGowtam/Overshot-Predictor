# BrickOfTicks — Global To-Do List

## Upcoming Tests
- [ ] **2026 Holdout Evaluation (Live Engine Simulation):** Test the Volume Fallback model using the newly discovered optimal thresholds from EXP-11 (`Prob_Win ≥ 0.52`, `Pred_OS ≥ 1.0`) on the completely unseen 2026 tick data. Instead of raw offline tensor validation, we will try to get these ticks from MT5 and pipe them through our `BrickOfTicks_Trader` (the live bridge engine). We need to figure out a mechanism to simulate/inject these historical MT5 ticks into the live trader to analyze its actual behavior and execution logic out-of-sample.

## Alpha Experiments Pending
- [ ] **EXP-12:** Spread Microstructure Veto (Gating based on widening spread)
- [ ] **EXP-08:** Sequence Entropy (Regime detection)
- [ ] **EXP-13:** Volatility-Regime Gating
- [ ] **EXP-17:** Adaptive TP via Pred_OS (Risk-Reward engineering)
- [ ] **EXP-16:** Markov Chain Sequences
- [ ] **EXP-07:** Cross-Brick OFI Persistence
- [ ] **EXP-14:** Baiting Inversion (Dual-Lobe)
- [ ] **EXP-01:** Multi-Resolution Confluence
- [ ] **EXP-05:** Ensemble Orthogonality
- [ ] **EXP-04:** Survival Analysis (Head C)
- [ ] **EXP-02:** Early-Exit Inference
- [ ] **EXP-06:** OFI Autocorrelation Decay
- [ ] **EXP-09:** Hawkes Process Intensity
- [ ] **EXP-10:** Velocity Wavelet Fingerprint
- [ ] **EXP-15:** Trap Network (Adversarial)
- [ ] **EXP-03:** Mixture of Experts (MoE)
