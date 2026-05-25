"""
BrickOfTicks Socket Bridge — Renko Builder
==========================================
Constructs Renko bricks from live tick data with exact K=0.00295 scaling.

K_MULTIPLIER must strictly be 0.00295.
DEPRECATED: 0.00118 (unprofitable spread-to-brick ratio)
DEPRECATED: 0.0018 (old bot spec)
"""

import logging
from typing import List, NamedTuple

logger = logging.getLogger(__name__)

# CRITICAL: only one formula is allowed
K_MULTIPLIER = 0.00295   # Validated: 5.9% spread burden, +0.747 expectancy

class BrickEvent(NamedTuple):
    open: float
    close: float
    high: float
    low: float
    uptrend: int
    timestamp: int
    brick_size: float
    sequence: str

class RenkoBuilder:
    def __init__(self, day_open_price: float):
        """
        Initialize the Renko builder with the current day's open price.
        """
        self.brick_size = day_open_price * K_MULTIPLIER
        self.current_price = day_open_price
        self.uptrend = 0
        self.sequence = ''
        self.brick_count = 0
        
        logger.info(f"RenkoBuilder initialized: day_open={day_open_price:.5f}, brick_size={self.brick_size:.5f}")

    def update_brick_size(self, new_brick_size: float, new_day_open: float = None):
        """
        Call on daily rollover to rescale the brick size.
        Does NOT affect already formed open bricks.
        """
        old = self.brick_size
        self.brick_size = new_brick_size
        logger.info(f"Brick size updated: {old:.5f} → {self.brick_size:.5f}")
        
        if self.brick_count == 0 and new_day_open is not None:
            self.current_price = new_day_open
            logger.info(f"Startup correction: current_price snapped to true day_open {new_day_open:.5f}")

    def update_tick(self, bid_price: float, time_ms: int) -> List[BrickEvent]:
        """
        Process a new tick (bid price) and return a list of newly formed bricks (if any).
        Handles gap fills and 2x reversal rules.
        """
        new_bricks = []

        if self.uptrend == 0:
            if bid_price >= self.current_price + self.brick_size:
                while bid_price >= self.current_price + self.brick_size:
                    self.current_price += self.brick_size
                    self.uptrend = 1
                    self.brick_count += 1
                    self._append_sequence('1')
                    new_bricks.append(BrickEvent(
                        open=self.current_price - self.brick_size,
                        close=self.current_price,
                        high=self.current_price,
                        low=self.current_price - self.brick_size,
                        uptrend=1,
                        timestamp=time_ms,
                        brick_size=self.brick_size,
                        sequence=self.sequence
                    ))
            elif bid_price <= self.current_price - self.brick_size:
                while bid_price <= self.current_price - self.brick_size:
                    self.current_price -= self.brick_size
                    self.uptrend = -1
                    self.brick_count += 1
                    self._append_sequence('0')
                    new_bricks.append(BrickEvent(
                        open=self.current_price + self.brick_size,
                        close=self.current_price,
                        high=self.current_price + self.brick_size,
                        low=self.current_price,
                        uptrend=-1,
                        timestamp=time_ms,
                        brick_size=self.brick_size,
                        sequence=self.sequence
                    ))
        elif self.uptrend == 1:
            if bid_price >= self.current_price + self.brick_size:
                while bid_price >= self.current_price + self.brick_size:
                    self.current_price += self.brick_size
                    self.brick_count += 1
                    self._append_sequence('1')
                    new_bricks.append(BrickEvent(
                        open=self.current_price - self.brick_size,
                        close=self.current_price,
                        high=self.current_price,
                        low=self.current_price - self.brick_size,
                        uptrend=1,
                        timestamp=time_ms,
                        brick_size=self.brick_size,
                        sequence=self.sequence
                    ))
            elif bid_price <= self.current_price - 2 * self.brick_size:
                # Reversal down
                self.current_price -= 2 * self.brick_size
                self.uptrend = -1
                self.brick_count += 1
                self._append_sequence('0')
                new_bricks.append(BrickEvent(
                    open=self.current_price + self.brick_size,
                    close=self.current_price,
                    high=self.current_price + self.brick_size,
                    low=self.current_price,
                    uptrend=-1,
                    timestamp=time_ms,
                    brick_size=self.brick_size,
                    sequence=self.sequence
                ))
                # Gap fill remaining
                while bid_price <= self.current_price - self.brick_size:
                    self.current_price -= self.brick_size
                    self.brick_count += 1
                    self._append_sequence('0')
                    new_bricks.append(BrickEvent(
                        open=self.current_price + self.brick_size,
                        close=self.current_price,
                        high=self.current_price + self.brick_size,
                        low=self.current_price,
                        uptrend=-1,
                        timestamp=time_ms,
                        brick_size=self.brick_size,
                        sequence=self.sequence
                    ))
        else: # self.uptrend == -1
            if bid_price <= self.current_price - self.brick_size:
                while bid_price <= self.current_price - self.brick_size:
                    self.current_price -= self.brick_size
                    self.brick_count += 1
                    self._append_sequence('0')
                    new_bricks.append(BrickEvent(
                        open=self.current_price + self.brick_size,
                        close=self.current_price,
                        high=self.current_price + self.brick_size,
                        low=self.current_price,
                        uptrend=-1,
                        timestamp=time_ms,
                        brick_size=self.brick_size,
                        sequence=self.sequence
                    ))
            elif bid_price >= self.current_price + 2 * self.brick_size:
                # Reversal up
                self.current_price += 2 * self.brick_size
                self.uptrend = 1
                self.brick_count += 1
                self._append_sequence('1')
                new_bricks.append(BrickEvent(
                    open=self.current_price - self.brick_size,
                    close=self.current_price,
                    high=self.current_price,
                    low=self.current_price - self.brick_size,
                    uptrend=1,
                    timestamp=time_ms,
                    brick_size=self.brick_size,
                    sequence=self.sequence
                ))
                # Gap fill remaining
                while bid_price >= self.current_price + self.brick_size:
                    self.current_price += self.brick_size
                    self.brick_count += 1
                    self._append_sequence('1')
                    new_bricks.append(BrickEvent(
                        open=self.current_price - self.brick_size,
                        close=self.current_price,
                        high=self.current_price,
                        low=self.current_price - self.brick_size,
                        uptrend=1,
                        timestamp=time_ms,
                        brick_size=self.brick_size,
                        sequence=self.sequence
                    ))
                    
        return new_bricks

    def _append_sequence(self, bit: str):
        self.sequence += bit
        if len(self.sequence) > 100:
            self.sequence = self.sequence[-100:]
