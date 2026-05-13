class RegimeMetaFilter:
    """
    Acts as a higher-level governor. Even if the models say "TRADE",
    this filter can veto based on macroeconomic or microstructural hostility.
    """
    def __init__(self):
        self.consecutive_losses = 0

    def evaluate_regime(self, tick_velocity, spread, volatility, time_of_day_mock=None):
        """
        Determines the current regime. Returns (is_tradable, regime_label)
        """
        # Veto highly illiquid or hyper-volatile regimes
        if spread > 1.5:
            return False, "HOSTILE_SPREAD"

        if tick_velocity < 0.1: # Ticks are too slow, market is dead
            return False, "DEAD_MARKET"

        if volatility > 2.0: # News shock
            return False, "NEWS_SHOCK"

        if self.consecutive_losses > 3:
            # Circuit breaker to prevent tilt/regime mismatch destruction
            return False, "CIRCUIT_BREAKER"

        return True, "FAVORABLE"

    def register_trade_result(self, is_win):
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

if __name__ == "__main__":
    print("Regime Meta-Filter defined.")
