class ExecutionRouter:
    """
    Determines HOW to execute a signal based on spread and volatility regimes.
    """
    def __init__(self):
        pass

    def route_signal(self, signal_confidence, current_spread, spread_threshold, vol_regime):
        """
        Decides whether to use Market, Limit, Iceberg, or Veto the trade entirely.
        """
        # VETO: If spread is blowing out (e.g. news event), NEVER execute.
        if current_spread > spread_threshold * 2.0:
            return "VETO", "Spread Explosion"

        # HYBRID: High confidence + High Volatility -> Use Market to guarantee entry, accept slippage
        if signal_confidence > 0.8 and vol_regime in ["HIGH_VOL_TREND", "MOMENTUM"]:
            return "MARKET", "High Conviction Momentum"

        # PASSIVE: Medium confidence or Mean Reverting regime -> Use Limit to avoid spread cost
        if vol_regime == "LOW_VOL_CHOP":
            return "LIMIT", "Mean Reversion Capture"

        # DEFAULT: Standard Market order for normal conditions
        return "MARKET", "Standard Execution"

if __name__ == "__main__":
    print("Execution Architectures defined.")
