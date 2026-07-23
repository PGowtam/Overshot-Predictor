import numpy as np

class DynamicRiskEngine:
    """
    Replaces the naive 1:1 Renko SL/TP with context-aware institutional risk management.
    """
    def __init__(self, atr_period=14):
        self.atr_period = atr_period
        self.history = []
        self.current_atr = 1.0

    def update_atr(self, high, low, close_prev):
        tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
        self.history.append(tr)
        if len(self.history) > self.atr_period:
            self.history.pop(0)
        self.current_atr = np.mean(self.history)

    def calculate_stops(self, entry_price, direction, volatility_modifier=1.5):
        """
        Volatility-adjusted SL and Asymmetric RR.
        Instead of 1:1, we aim for a 1:1.5 or 1:2 profile based on ATR.
        """
        sl_dist = self.current_atr * volatility_modifier
        tp_dist = sl_dist * 1.5 # Asymmetric RR

        sl = entry_price - sl_dist if direction == 1 else entry_price + sl_dist
        tp = entry_price + tp_dist if direction == 1 else entry_price - tp_dist
        return sl, tp

    def calculate_trailing_stop(self, current_price, current_sl, direction, activation_dist):
        """Dynamic trailing exit to lock in profits without choking the trade."""
        if direction == 1:
            # Trailing stop only moves up
            new_sl = current_price - (self.current_atr * 1.0)
            if new_sl > current_sl and current_price > (current_sl + activation_dist):
                return new_sl
        else:
            new_sl = current_price + (self.current_atr * 1.0)
            if new_sl < current_sl and current_price < (current_sl - activation_dist):
                return new_sl
        return current_sl

if __name__ == "__main__":
    print("Dynamic Risk Module defined.")
