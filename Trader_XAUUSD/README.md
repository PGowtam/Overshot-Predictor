# MT5 Renko Weighted Ensemble Trader

This directory contains the production-ready trading system that connects to MetaTrader 5 and executes the 5-Agent Weighted Ensemble strategy.

## Directory Structure
- `config/`: Configuration settings (Symbol, Risk, Paths).
- `data/`: Data ingestion (Gap-less Tick Stream) and Feature Engineering.
- `models/`: Inference engines (Ensemble, CatBoost, HMM).
- `execution/`: Order management and Orbit Engine (Trading Loop).
- `utils/`: Logging and State persistence.

## Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **MetaTrader 5**: Ensure MT5 is installed and running. Enable "Algo Trading" in the terminal.
3. **Model Paths**: Ensure the `Models` directory is correctly pointed to in `config/definitions.py`. By default it expects `Trader/Models` (or mapped to parent).
   - If models are missing, copying them from `Reinforcement/Model` and `Reinforcement/RL_Agent_Final/models` to `Trader/Models` is recommended.

## Running
Run the main entry point:
```bash
python main.py
```

## Features
- **Gap-Less execution**: Uses `copy_ticks_from` to ensure no data is lost during processing latency.
- **Real-time Feature Engineering**: Replicates the 21-dim vector construction from the training phase.
- **Intra-Brick Management**: Monitors price tick-by-tick for Break-Even triggers within a forming brick.
- **Safety**: Daily Risk Limit (-3%) and State Persistence for crash recovery.
