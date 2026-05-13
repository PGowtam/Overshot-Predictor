import numpy as np

class ExecutionSimulator:
    def __init__(self, base_latency_ms: int = 30):
        self.base_latency_ms = base_latency_ms

    def simulate_market_order(self, direction: int, signal_time_msc: int, signal_price: float, tick_stream: list, current_idx: int, regime) -> dict:
        """
        Simulates the reality of executing a market order.
        Accounts for:
        1. Python-to-Broker Latency
        2. Liquidity depth (will we slip through the spread?)
        """
        # 1. Determine actual execution time
        latency = self.base_latency_ms + np.random.exponential(15) # Base + Network jitter
        exec_time_msc = signal_time_msc + latency

        # 2. Fast-forward the tick stream to find the actual price at execution time
        exec_tick = tick_stream[current_idx]
        for i in range(current_idx, len(tick_stream)):
            if tick_stream[i]["time_msc"] >= exec_time_msc:
                exec_tick = tick_stream[i]
                break

        # 3. Determine Execution Price
        # Buy at Ask, Sell at Bid
        base_exec_price = exec_tick["ask"] if direction == 1 else exec_tick["bid"]

        # 4. Simulate Slippage (Liquidity Exhaustion)
        # If liquidity is thin, we sweep the book and slip
        is_deep_liquidity = np.random.random() < regime.liquidity_prob
        slippage_points = 0.0

        if not is_deep_liquidity:
            # We slip further into the spread direction
            slippage_points = np.abs(np.random.normal(0.1, 0.2)) # Average 0.1 pts slippage
            if direction == 1:
                base_exec_price += slippage_points
            else:
                base_exec_price -= slippage_points

        return {
            "status": "FILLED",
            "exec_time": exec_tick["time_msc"],
            "requested_price": signal_price,
            "actual_price": base_exec_price,
            "slippage": abs(base_exec_price - signal_price),
            "latency_ms": latency
        }

    def simulate_limit_order(self, direction: int, limit_price: float, tick_stream: list, start_idx: int) -> dict:
        """
        Simulates the fallback limit order.
        Adverse Selection Engine: We only get filled if price retraces to us.
        """
        for i in range(start_idx, len(tick_stream)):
            tick = tick_stream[i]

            # To fill a Buy Limit, the Ask must drop to/below our limit
            if direction == 1 and tick["ask"] <= limit_price:
                return {
                    "status": "FILLED",
                    "exec_time": tick["time_msc"],
                    "actual_price": limit_price,
                    "ticks_to_fill": i - start_idx
                }
            # To fill a Sell Limit, the Bid must rise to/above our limit
            elif direction == -1 and tick["bid"] >= limit_price:
                return {
                    "status": "FILLED",
                    "exec_time": tick["time_msc"],
                    "actual_price": limit_price,
                    "ticks_to_fill": i - start_idx
                }

        return {"status": "MISSED"} # Price never retraced

if __name__ == "__main__":
    print("Execution Simulator defined.")
