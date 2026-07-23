import numpy as np

class AdvancedMicrostructure:
    """
    Simulates high-fidelity market microstructure:
    - Liquidity exhaustion
    - Spread dynamics (widening during vol)
    - Stop hunts (price briefly wicking through levels)
    - Order queue prioritization
    """
    def __init__(self, base_spread=0.2, depth_decay=0.05):
        self.base_spread = base_spread
        self.depth_decay = depth_decay
        self.current_volatility = 0.01

    def update_volatility(self, true_range):
        """Update local volatility based on recent price movement."""
        self.current_volatility = 0.8 * self.current_volatility + 0.2 * true_range

    def get_dynamic_spread(self):
        """Spread widens as volatility increases."""
        return self.base_spread + (self.current_volatility * 2.0) + max(0, np.random.normal(0, 0.05))

    def simulate_fill(self, requested_price, direction, order_type, size=1.0, is_stop=False):
        """
        Simulates the reality of getting filled.
        Returns (filled_price, latency_ms, status)
        """
        spread = self.get_dynamic_spread()
        latency = np.random.exponential(30) # Base python-MT5 latency

        # Stop hunts: Stop orders often suffer massive slippage due to liquidity vacuums
        if is_stop:
            latency += np.random.exponential(100) # Stop orders delayed in queue
            slippage = np.random.exponential(self.current_volatility * 3.0)
            fill_price = requested_price + slippage if direction == 1 else requested_price - slippage
            return fill_price, latency, "FILLED"

        if order_type == "MARKET":
            # Market orders sweep the book
            # The larger the size, the further we sweep (depth_decay)
            slippage = np.random.exponential(self.current_volatility) + (size * self.depth_decay)
            fill_price = requested_price + slippage if direction == 1 else requested_price - slippage
            return fill_price, latency, "FILLED"

        elif order_type == "LIMIT":
            # Limit orders suffer from adverse selection
            # We assume a limit order only fills if price moves *through* it by at least half the spread
            penetration_required = spread * 0.5

            # This logic will be called iteratively by the pipeline to check if price crossed
            return requested_price, latency, "PENDING"

        return requested_price, latency, "REJECTED"

if __name__ == "__main__":
    print("Microstructure Simulator defined.")
