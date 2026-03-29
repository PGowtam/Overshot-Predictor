from execution.orbit import OrbitEngine
import sys
import os

def main():
    print("Welcome to MT5 Renko Trader (RL Ensemble)")
    
    # Optional: Parse args for mode (Live/Paper/Backtest)
    
    engine = OrbitEngine()
    engine.run()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()
