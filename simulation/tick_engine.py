import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Iterator

@dataclass
class MarketRegime:
    name: str
    base_volatility: float  # Base tick variance
    spread_mean: float      # Mean spread in points
    spread_std: float       # Spread variance
    liquidity_prob: float   # Probability of deep liquidity (no slippage)
    momentum_prob: float    # Probability of sequential directional ticks
    jump_prob: float        # Probability of flash spike

REGIMES = {
    "NORMAL": MarketRegime("NORMAL", 0.05, 0.20, 0.05, 0.80, 0.10, 0.001),
    "HIGH_VOL_TREND": MarketRegime("HIGH_VOL_TREND", 0.15, 0.40, 0.15, 0.40, 0.35, 0.02),
    "LOW_VOL_CHOP": MarketRegime("LOW_VOL_CHOP", 0.02, 0.15, 0.02, 0.90, 0.05, 0.0001),
    "NEWS_SHOCK": MarketRegime("NEWS_SHOCK", 0.50, 1.50, 0.50, 0.10, 0.60, 0.10)
}

class SyntheticTickEngine:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.current_price = 2000.00  # Base XAUUSD price
        self.timestamp_ms = 1600000000000

    def stream_ticks(self, num_ticks: int, regime: MarketRegime) -> Iterator[dict]:
        """Generates a realistic stream of L1 order book updates."""

        # Pre-generate some random variables for speed
        vol_noise = self.rng.normal(0, regime.base_volatility, num_ticks)
        spread_noise = np.abs(self.rng.normal(regime.spread_mean, regime.spread_std, num_ticks))
        time_gaps = self.rng.exponential(250, num_ticks) # Average 250ms per tick

        jump_flags = self.rng.random(num_ticks) < regime.jump_prob
        momentum_flags = self.rng.random(num_ticks) < regime.momentum_prob

        current_momentum_dir = 1 if self.rng.random() > 0.5 else -1

        for i in range(num_ticks):
            self.timestamp_ms += int(time_gaps[i])

            # Base price update
            price_change = vol_noise[i]

            # Apply momentum
            if momentum_flags[i]:
                price_change += current_momentum_dir * (regime.base_volatility * 2)
            else:
                # 5% chance to flip momentum direction when not actively in a momentum burst
                if self.rng.random() < 0.05:
                    current_momentum_dir *= -1

            # Apply jumps (Flash spikes/sweeps)
            if jump_flags[i]:
                price_change += current_momentum_dir * (regime.base_volatility * 10)

            self.current_price += price_change

            # Spread logic
            current_spread = max(0.01, spread_noise[i]) # Never negative spread

            # In a sweep, spread widens drastically
            if jump_flags[i]:
                current_spread *= 3.0

            bid = self.current_price - (current_spread / 2)
            ask = self.current_price + (current_spread / 2)

            # Volume logic (synthetic OTC representation)
            bid_vol = self.rng.integers(1, 15) if self.rng.random() > 0.1 else 0.0
            ask_vol = self.rng.integers(1, 15) if self.rng.random() > 0.1 else 0.0

            yield {
                "time_msc": self.timestamp_ms,
                "bid": round(bid, 3),
                "ask": round(ask, 3),
                "bid_vol": float(bid_vol),
                "ask_vol": float(ask_vol),
                "spread": current_spread,
                "regime": regime.name
            }

if __name__ == '__main__':
    engine = SyntheticTickEngine()
    print("Testing Tick Engine (NORMAL regime)...")
    stream = engine.stream_ticks(5, REGIMES["NORMAL"])
    for t in stream:
        print(t)
