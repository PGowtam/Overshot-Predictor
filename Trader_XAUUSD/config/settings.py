import logging

# Symbol Settings
SYMBOL = "AUDJPY" # XAUUSD.m or similar depending on broker
TIMEFRAME = 1 # M1 (Int value for MT5 TIMEFRAME_M1 is 1)
TIMEZONE_OFFSET = 0 # Hours to shift from UTC (e.g., UTC+2 for Broker Time)

# Trading Params
LOT_SIZE = 0.2
MAGIC_NUMBER = 123456
DEVIATION = 20 # Slippage points

# Risk Management
MAX_DAILY_DRAWDOWN_PCT = 0.03 # 3%
RISK_PER_TRADE_PCT = 0.005 # 0.5% (Not always used if fixed lot, but good for calc)

# Renko Params
# Default starting values, usually overwritten by Daily Optimizer
DEFAULT_BRICK_SIZE = 1.0 
DEFAULT_OFFSET = 0.0
# T6 Factor: 0.00236 / 2 = 0.00118
BRICK_SIZE_FACTOR = 0.00118

# Ensemble Weights (Optimized)
WEIGHTS = {
    'ppo': 0.3735,
    'dqn': 1.3229,
    'qrdqn': 1.0548,
    'recurrent': 0.6007,
    'transformer': 1.2186
}
VOTE_THRESHOLD = 4.2094

# Logging
LOG_LEVEL = logging.INFO
LOG_FILE = "trader.log"
