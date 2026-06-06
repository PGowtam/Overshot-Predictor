import logging

logger = logging.getLogger(__name__)

class EMA:
    def __init__(self, period):
        self.period = period
        self.alpha = 2.0 / (period + 1)
        self.value = None
        
    def update(self, price):
        if self.value is None:
            self.value = price
        else:
            self.value = (price * self.alpha) + (self.value * (1.0 - self.alpha))
        return self.value

class Swing:
    def __init__(self, is_high, bar, index):
        self.is_high = is_high
        self.bar = bar
        self.index = index
        self.price = bar.high if is_high else bar.low

class StructureDetector:
    def __init__(self, left_bars=2, right_bars=2):
        self.left = left_bars
        self.right = right_bars
        self.history = []
        self.swings = []
        self.current_idx = -1

    def update(self, completed_bar):
        self.current_idx += 1
        self.history.append((self.current_idx, completed_bar))
        
        # Keep just enough history for the fractal
        window_size = self.left + self.right + 1
        if len(self.history) > window_size:
            self.history.pop(0)

        if len(self.history) == window_size:
            center_idx_in_hist = self.left
            center_abs_idx, center_bar = self.history[center_idx_in_hist]
            
            is_swing_high = True
            is_swing_low = True
            
            for i, (abs_idx, bar) in enumerate(self.history):
                if i == center_idx_in_hist:
                    continue
                if bar.high >= center_bar.high:
                    is_swing_high = False
                if bar.low <= center_bar.low:
                    is_swing_low = False
                    
            if is_swing_high:
                s = Swing(True, center_bar, center_abs_idx)
                self.swings.append(s)
                return s
            elif is_swing_low:
                s = Swing(False, center_bar, center_abs_idx)
                self.swings.append(s)
                return s
                
        return None

class FVG:
    def __init__(self, is_bullish, top, bottom, creation_time_msc):
        self.is_bullish = is_bullish
        self.top = top
        self.bottom = bottom
        self.midpoint = (top + bottom) / 2.0
        self.creation_time_msc = creation_time_msc
        self.mitigated = False
        
class FVGDetector:
    def __init__(self):
        self.history = []
        
    def update(self, completed_bar):
        self.history.append(completed_bar)
        if len(self.history) > 3:
            self.history.pop(0)
            
        if len(self.history) == 3:
            bar1, bar2, bar3 = self.history
            
            # Bullish FVG: Bar 1 High < Bar 3 Low. Gap is Bar 1 High to Bar 3 Low.
            if bar1.high < bar3.low:
                # The gap is strictly between the high of bar 1 and the low of bar 3
                return FVG(True, bar3.low, bar1.high, bar3.time_msc)
                
            # Bearish FVG: Bar 1 Low > Bar 3 High. Gap is Bar 3 High to Bar 1 Low.
            if bar1.low > bar3.high:
                return FVG(False, bar1.low, bar3.high, bar3.time_msc)
                
        return None

class OrderBlock:
    def __init__(self, is_bullish, high, low, creation_time_msc):
        self.is_bullish = is_bullish
        self.high = high
        self.low = low
        self.midpoint = (high + low) / 2.0
        self.creation_time_msc = creation_time_msc
        self.mitigated = False

def find_order_block(bars_history, start_idx_backwards, is_bullish_impulse):
    """
    Finds the order block (last opposite candle) before an impulse.
    is_bullish_impulse: True if we are looking for a bullish OB (last bearish candle).
    """
    if not bars_history or start_idx_backwards < 0 or start_idx_backwards >= len(bars_history):
        return None
        
    for i in range(start_idx_backwards, max(-1, start_idx_backwards - 10), -1):
        bar = bars_history[i]
        is_bearish_candle = bar.close < bar.open
        is_bullish_candle = bar.close > bar.open
        
        if is_bullish_impulse and is_bearish_candle:
            # Bullish OB is the last bearish candle.
            return OrderBlock(True, bar.high, bar.low, bar.time_msc)
        elif not is_bullish_impulse and is_bullish_candle:
            # Bearish OB is the last bullish candle.
            return OrderBlock(False, bar.high, bar.low, bar.time_msc)
            
    return None
