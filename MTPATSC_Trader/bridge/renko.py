"""
MTPATSC Trader — Renko Builder
===============================
Constructs Renko bricks from live tick data with K=0.00118 scaling.
Collects intra-brick ticks for ANCS feature computation at brick close.

K_MULTIPLIER = 0.00118 (matches MTPATSC training pipeline).
"""

import logging
from typing import List, NamedTuple, Optional

logger = logging.getLogger(__name__)

# CRITICAL: must match training pipeline exactly
K_MULTIPLIER = 0.00118


class BrickEvent(NamedTuple):
    open: float
    close: float
    high: float
    low: float
    uptrend: int       # 1 = UP, -1 = DOWN
    timestamp: int     # milliseconds since epoch
    brick_size: float
    sequence: str
    intra_ticks: list  # List of {'bid': float, 'ask': float, 'time_msc': int}


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
        self.current_ticks = []  # Accumulates ticks for current forming brick

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

    def update_tick(self, bid: float, time_ms: int, ask: float = None) -> List[BrickEvent]:
        """
        Process a new tick (bid price) and return a list of newly formed bricks (if any).
        Handles gap fills and 2x reversal rules.
        Also collects intra-brick ticks for feature extraction.
        """
        # Store tick for intra-brick feature computation
        tick_record = {'bid': bid, 'time_msc': time_ms}
        if ask is not None:
            tick_record['ask'] = ask
        else:
            tick_record['ask'] = bid  # Fallback if no ask provided
        self.current_ticks.append(tick_record)

        new_bricks = []
        price = bid

        if self.uptrend == 0:
            if price >= self.current_price + self.brick_size:
                while price >= self.current_price + self.brick_size:
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
                        sequence=self.sequence,
                        intra_ticks=list(self.current_ticks)
                    ))
                    # Reset tick buffer, keep current tick as start of next brick
                    self.current_ticks = [tick_record.copy()]
            elif price <= self.current_price - self.brick_size:
                while price <= self.current_price - self.brick_size:
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
                        sequence=self.sequence,
                        intra_ticks=list(self.current_ticks)
                    ))
                    self.current_ticks = [tick_record.copy()]
        elif self.uptrend == 1:
            if price >= self.current_price + self.brick_size:
                while price >= self.current_price + self.brick_size:
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
                        sequence=self.sequence,
                        intra_ticks=list(self.current_ticks)
                    ))
                    self.current_ticks = [tick_record.copy()]
            elif price <= self.current_price - 2 * self.brick_size:
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
                    sequence=self.sequence,
                    intra_ticks=list(self.current_ticks)
                ))
                self.current_ticks = [tick_record.copy()]
                # Gap fill remaining
                while price <= self.current_price - self.brick_size:
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
                        sequence=self.sequence,
                        intra_ticks=list(self.current_ticks)
                    ))
                    self.current_ticks = [tick_record.copy()]
        else:  # self.uptrend == -1
            if price <= self.current_price - self.brick_size:
                while price <= self.current_price - self.brick_size:
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
                        sequence=self.sequence,
                        intra_ticks=list(self.current_ticks)
                    ))
                    self.current_ticks = [tick_record.copy()]
            elif price >= self.current_price + 2 * self.brick_size:
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
                    sequence=self.sequence,
                    intra_ticks=list(self.current_ticks)
                ))
                self.current_ticks = [tick_record.copy()]
                # Gap fill remaining
                while price >= self.current_price + self.brick_size:
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
                        sequence=self.sequence,
                        intra_ticks=list(self.current_ticks)
                    ))
                    self.current_ticks = [tick_record.copy()]

        return new_bricks

    def _append_sequence(self, bit: str):
        self.sequence += bit
        if len(self.sequence) > 100:
            self.sequence = self.sequence[-100:]
