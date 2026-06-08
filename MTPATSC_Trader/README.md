# BrickOfTicks MT5 Trader

This is the production execution engine for the gold-overshot strategy.

## 🚀 Setup Instructions (Windows)

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd Overshot/BrickOfTicks_Trader
   ```

2. **Environment Setup**:
   - Install Python 3.10+ (ensure it's added to PATH).
   - Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **MetaTrader 5 Link**:
   - Ensure MetaTrader 5 (XAUUSD Broker) is open and logged in.
   - Auto-Trading must be enabled in the MT5 Terminal.

## 🛠️ Implementation Progress

The project is currently in the **Scaffolding** phase.
Refer to `Resources/bot_implementation.md` and `Resources/bot_tasks.md` in the root folder for step-by-step development instructions.

- [x] Phase 0: Project Scaffolding & Model Deployment (Done)
- [ ] Phase 1: RollingZScore Implementation (Next)

## 🎯 Configuration
The optimized thresholds are already set in `config/settings.py`:
- **Standard**: Prob_Win >= 0.7, Pred_OS >= 1.2
- **Baiting**: Prob_Win < 0.2, Pred_OS < 0.7
