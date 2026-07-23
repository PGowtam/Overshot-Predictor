import pytz
from datetime import timedelta

from datetime import datetime, timezone, timedelta
from ict_core import EMA
import logging

logger = logging.getLogger(__name__)

class Bar:
    __slots__ = ['time_msc', 'open', 'high', 'low', 'close', 'volume']
    def __init__(self, time_msc, open_p):
        self.time_msc = time_msc
        self.open = open_p
        self.high = open_p
        self.low = open_p
        self.close = open_p
        self.volume = 0.0

class SessionState:
    def __init__(self):
        self.in_asian = False
        self.in_london = False
        self.in_ny = False
        self.in_london_sb = False # Silver Bullet
        self.in_ny_sb = False
        
        self.ash = None
        self.asl = None
        self.london_h = None
        self.london_l = None
        self.pdh = None
        self.pdl = None
        
        self.current_asian_h = -1
        self.current_asian_l = 999999
        self.current_london_h = -1
        self.current_london_l = 999999
        self.current_day_h = -1
        self.current_day_l = 999999

class ICTDataBuilder:
    def __init__(self):
        self.ny_tz = pytz.timezone('America/New_York')
        
        self.current_m5 = None
        self.current_m15 = None
        self.current_h4 = None
        self.current_d1 = None
        
        self.m5_history = []
        self.m15_history = []
        
        # EMAs
        self.ema_h4_20 = EMA(20)
        self.ema_d1_50 = EMA(50)
        self.ema_d1_200 = EMA(200)
        
        self.session = SessionState()
        self.last_trading_day = None
        
        self.cached_h1_start_ms = -1
        self.cached_ny_hour = -1
        self.cached_trade_date_str = ""

    def process_tick(self, time_msc, bid, vol):
        """
        Updates bars and session state based on incoming tick.
        Returns (m5_completed, m15_completed) which are the bar objects if a bar just completed, else None.
        """
        # UTC alignments are the same for M5/M15 as NY
        m5_start_ms = time_msc - (time_msc % (5 * 60 * 1000))
        m15_start_ms = time_msc - (time_msc % (15 * 60 * 1000))
        
        m5_completed = None
        if self.current_m5 is None:
            self.current_m5 = Bar(m5_start_ms, bid)
        elif m5_start_ms > self.current_m5.time_msc:
            m5_completed = self.current_m5
            self.m5_history.append(self.current_m5)
            self.current_m5 = Bar(m5_start_ms, bid)
            # Memory mgmt
            if len(self.m5_history) > 1000:
                self.m5_history.pop(0)
                
        if bid > self.current_m5.high: self.current_m5.high = bid
        if bid < self.current_m5.low: self.current_m5.low = bid
        self.current_m5.close = bid
        self.current_m5.volume += vol
        
        m15_completed = None
        if self.current_m15 is None:
            self.current_m15 = Bar(m15_start_ms, bid)
        elif m15_start_ms > self.current_m15.time_msc:
            m15_completed = self.current_m15
            self.m15_history.append(self.current_m15)
            self.current_m15 = Bar(m15_start_ms, bid)
            if len(self.m15_history) > 500:
                self.m15_history.pop(0)
                
        if bid > self.current_m15.high: self.current_m15.high = bid
        if bid < self.current_m15.low: self.current_m15.low = bid
        self.current_m15.close = bid
        self.current_m15.volume += vol

        # NY Session tracking with caching for performance
        h1_start_ms = time_msc - (time_msc % 3600000)
        if h1_start_ms != self.cached_h1_start_ms:
            dt_ny = datetime.fromtimestamp(time_msc / 1000.0, tz=timezone.utc).astimezone(self.ny_tz)
            self.cached_h1_start_ms = h1_start_ms
            self.cached_ny_hour = dt_ny.hour
            self.cached_trade_date_str = f"{dt_ny.year}-{dt_ny.month}-{dt_ny.day}" if self.cached_ny_hour < 17 else f"{dt_ny.year}-{dt_ny.month}-{dt_ny.day+1}(approx)"
            
        hour = self.cached_ny_hour
        current_trade_day_str = self.cached_trade_date_str
        
        # D1 Bar Aggregation
        if self.last_trading_day != current_trade_day_str:
            if self.current_d1 is not None:
                self.ema_d1_50.update(self.current_d1.close)
                self.ema_d1_200.update(self.current_d1.close)
            self.current_d1 = Bar(time_msc, bid)
        else:
            if self.current_d1 is not None:
                if bid > self.current_d1.high: self.current_d1.high = bid
                if bid < self.current_d1.low: self.current_d1.low = bid
                self.current_d1.close = bid
                self.current_d1.volume += vol
                
        # H4 Bar Aggregation
        h4_start_hour = hour - ((hour - 17) % 4)
        if h4_start_hour < 0: h4_start_hour += 24
        current_h4_str = f"{current_trade_day_str}_{h4_start_hour}"
        
        if not hasattr(self, 'last_h4_str') or getattr(self, 'last_h4_str') != current_h4_str:
            if self.current_h4 is not None:
                self.ema_h4_20.update(self.current_h4.close)
            self.current_h4 = Bar(time_msc, bid)
            self.last_h4_str = current_h4_str
        else:
            if self.current_h4 is not None:
                if bid > self.current_h4.high: self.current_h4.high = bid
                if bid < self.current_h4.low: self.current_h4.low = bid
                self.current_h4.close = bid
                self.current_h4.volume += vol
        
        if self.last_trading_day != current_trade_day_str:
            if self.last_trading_day is not None:
                # Rollover happened
                self.session.pdh = self.session.current_day_h
                self.session.pdl = self.session.current_day_l
            
            self.session.current_day_h = bid
            self.session.current_day_l = bid
            self.last_trading_day = current_trade_day_str
            
            # Reset session highs/lows
            self.session.current_asian_h = -1
            self.session.current_asian_l = 999999
            self.session.current_london_h = -1
            self.session.current_london_l = 999999

        # Update daily extreme
        if bid > self.session.current_day_h: self.session.current_day_h = bid
        if bid < self.session.current_day_l: self.session.current_day_l = bid

        # Asian Session (20:00 - 01:00)
        self.session.in_asian = (hour >= 20 or hour < 1)
        if self.session.in_asian:
            if bid > self.session.current_asian_h: self.session.current_asian_h = bid
            if bid < self.session.current_asian_l: self.session.current_asian_l = bid
        elif hour == 1 and self.session.current_asian_h != -1:
            # Asian closed, lock levels
            self.session.ash = self.session.current_asian_h
            self.session.asl = self.session.current_asian_l

        # London Session (02:00 - 05:00)
        self.session.in_london = (2 <= hour < 5)
        self.session.in_london_sb = (3 <= hour < 4)
        if self.session.in_london:
            if bid > self.session.current_london_h: self.session.current_london_h = bid
            if bid < self.session.current_london_l: self.session.current_london_l = bid
        elif hour == 5 and self.session.current_london_h != -1:
            self.session.london_h = self.session.current_london_h
            self.session.london_l = self.session.current_london_l

        # NY Session (07:00 - 10:00)
        self.session.in_ny = (7 <= hour < 10)
        self.session.in_ny_sb = (10 <= hour < 11)

        return m5_completed, m15_completed
