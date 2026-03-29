import time
import numpy as np
from collections import deque
from datetime import datetime, timedelta
import pandas as pd
import os
from config.settings import DEFAULT_BRICK_SIZE, DEFAULT_OFFSET, SYMBOL, TIMEZONE_OFFSET, BRICK_SIZE_FACTOR
from data.connector import MT5Connector
from data.tick_buffer import TickStream
from data.renko import RenkoBuilder
from data.features import FeatureEngineer
from models.meta_wrapper import MetaTraderWrapper
from models.predictors import OutcomePredictor
from execution.orders import OrderExecutor
from execution.risk import RiskManager
from utils.logger import logger
from utils.state import StateManager
import MetaTrader5 as mt5

class OrbitEngine:
    def __init__(self):
        # Components
        self.connector = MT5Connector()
        self.state = StateManager()
        self.clock = None # Initialized in start() after connection
        
        # Load Optimization params from state or default
        saved_brick = self.state.get("optimization", {}).get("brick_size", 0.0)
        self.brick_size = saved_brick 
        self.offset = self.state.get("optimization", {}).get("grid_offset", DEFAULT_OFFSET)
        
        self.renko = None 
        
        self.features = FeatureEngineer()
        
        # --- META SYSTEMS INTEGRATION ---
        self.ensemble = MetaTraderWrapper()
        self.ensemble.load_all()
        
        self.predictors = OutcomePredictor()
        self.predictors.load()
        
        self.orders = OrderExecutor()
        self.risk = RiskManager(self.state)
        
        # Runtime State
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        self.current_date = datetime.utcnow().date() # Initialize Date Tracker
        
        # Transformer Buffer (Stack of 10 Observations)
        self.obs_stack = deque(maxlen=10)
        
        # M1 Data Buffer (for Indicators)
        # We accumulate ticks and resample to M1
        self.tick_accumulator = []
        self.m1_buffer = pd.DataFrame() # Holds recent M1 bars
        
    def start(self):
        if not self.connector.connect():
            return
            
        # Initialize TickStream NOW, when MT5 is connected
        self.clock = TickStream()
            
        # Optimization & Warmup Logic delegated to session init
        self._initialize_session()
        
    def _initialize_session(self, force_history_fetch=True):
        """
        Full Session Initialization:
        1. Fetch History (if needed)
        2. Optimize Renko Params (Size, Anchor)
        3. Find Startup Candle (First hit of Anchor)
        4. Replay History to build State
        """
        import MetaTrader5 as mt5
        from data.renko_optimizer import RenkoOptimizer
        
        logger.info("Initializing Session...")
        
        history_df = None
        if force_history_fetch:
            logger.info("Fetching M1 History for Optimization (7 Days)...")
            days_back = 7
            ticks = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 1440 * days_back)
            
            if ticks is None or len(ticks) == 0:
                logger.warning("History Fetch Failed. Using Default Fallback.")
                # Fallback: Current Price as Anchor
                start_price = mt5.symbol_info_tick(SYMBOL).ask
                self.brick_size = start_price * BRICK_SIZE_FACTOR
                # self.renko = RenkoBuilder(self.brick_size, start_price) # Will happen below if we fake DF
                # Use tiny fake DF to trigger fallback logic below or just handle here?
                # Handling here for safety
                self.renko = RenkoBuilder(self.brick_size, start_price)
                return
            else:
                # Convert to DF
                history_df = pd.DataFrame(ticks)
                history_df['time'] = history_df['time'] + (TIMEZONE_OFFSET * 3600)
                history_df['date'] = pd.to_datetime(history_df['time'], unit='s')
                history_df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        if history_df is None or history_df.empty:
             return

        # Optimize
        optimizer = RenkoOptimizer()
        best_bs, best_anchor, _ = optimizer.optimize(history_df)
        
        self.brick_size = best_bs
        
        # --- ANCHOR SEARCH LOGIC ---
        # We must find the first candle in history where the price crossed the anchor.
        # Anchor is a price level.
        # Condition: Low <= Anchor <= High
        
        scan_idx = -1
        # Convert necessary columns to numpy for speed
        n_highs = history_df['high'].values
        n_lows = history_df['low'].values
        
        # Simple Loop (Numpy optimization possible but loop is fine for 10k rows once)
        for i in range(len(history_df)):
            if n_lows[i] <= best_anchor <= n_highs[i]:
                scan_idx = i
                break
                
        if scan_idx == -1:
            logger.warning(f"Optimization Anchor {best_anchor} not found in history range! Defaulting to start of history.")
            scan_idx = 0
            start_price = history_df.iloc[0]['close']
        else:
            logger.info(f"Anchor {best_anchor} found at Index {scan_idx} ({history_df.iloc[scan_idx]['date']})")
            start_price = best_anchor
            
        # Initialize M1 Buffer
        # We keep data starting from scan_idx to replay
        replay_df = history_df.iloc[scan_idx:].copy()
        
        # Initialize Renko
        # Start exactly at Anchor Price
        self.renko = RenkoBuilder(self.brick_size, start_price)
        
        # Replay
        logger.info(f"Replaying Renko State ({len(replay_df)} bars)...")
        replay_df.set_index('date', inplace=True)
        
        # IMPORTANT: M1 Buffer needs to be populated for Indicators
        # We start with the replay data as our initial buffer
        # But we might need pre-roll for indicators (e.g. SMA50 needs 50 bars before).
        # So we should slice M1 buffer from scan_idx - 1000 effectively.
        buffer_start = max(0, scan_idx - 1000)
        self.m1_buffer = history_df.iloc[buffer_start:].copy()
        self.m1_buffer.set_index('date', inplace=True)
        
        # Renko Replay Loop (Only on Replay Segment)
        count = 0
        for idx, row in replay_df.iterrows():
            ts_ms = int(row.name.timestamp() * 1000)
            
            p_open = row['open']
            p_close = row['close']
            p_high = row['high']
            p_low = row['low']
            
            if p_close >= p_open:
                prices = [p_low, p_high, p_close]
            else:
                prices = [p_high, p_low, p_close]
                
            for p in prices:
                self.renko.update_tick(p, ts_ms)
            count += 1
            
        logger.info(f"Replay Complete. Generated {len(self.renko.history)} bricks.")
        
        # Fill Transformer Stack
        # Ideally we'd replay inference too to fill stack correctly.
        # For now, we fill with zeros or try to calculate last N states.
        # Recalculating last 10 states is better.
        
        self.obs_stack.clear()
        
        if len(self.renko.history) > 10:
             # Try to backfill stack
             pass
             
        # Standard Fill (Zeros if empty)
        # Fix: Meta-Agent uses 30-dim obs.
        dummy_obs = np.zeros(30, dtype=np.float32)
        while len(self.obs_stack) < 10:
            self.obs_stack.append(dummy_obs)
            
        self._save_renko_snapshot()
            
        logger.info(f"Orbit Started. Brick: {self.brick_size:.4f}")
        
    def _save_renko_snapshot(self):
        """
        Saves the current Renko history (after warmup) to a CSV file.
        """
        try:
            if not self.renko or not self.renko.history:
                logger.warning("No Renko history to save.")
                return

            # Directory
            save_dir = "renkos"
            os.makedirs(save_dir, exist_ok=True)
            
            # Filename based on current time
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"renko_{timestamp_str}.csv"
            filepath = os.path.join(save_dir, filename)
            
            # Convert to DataFrame
            # Renko history is list of NewBrickEvent namedtuples
            df = pd.DataFrame(self.renko.history)
            
            # Add readable date (from SHIFTED timestamps)
            if 'timestamp' in df.columns and not df.empty:
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Save
            df.to_csv(filepath, index=False)
            logger.info(f"Saved Warmup Renko Snapshot to: {filepath} ({len(df)} bricks)")
            
        except Exception as e:
            logger.error(f"Failed to save Renko snapshot: {e}")
        
    def get_normalized_time_left(self):
        """
        Calculates normalized time left in the trading day [0, 1].
        """
        # UTC+Offset (Broker Time)
        now = datetime.utcnow() + timedelta(hours=TIMEZONE_OFFSET)
        
        # Session End: 23:59:00
        session_end = now.replace(hour=23, minute=59, second=0, microsecond=0)
        
        # If we are past 23:59, maybe it's next day? Or wait for reset?
        # Assuming we run 24h, session resets at 00:00.
        # Simple distance: Seconds until 23:59 today.
        
        time_left_sec = (session_end - now).total_seconds()
        
        # Total seconds in day (86400) or trading session?
        # Predictor trained on full day decay usually.
        # Normalize by 24h
        norm = time_left_sec / 86400.0
        
        # Clip
        return max(0.0, min(1.0, norm))


    def check_new_day(self):
        """
        Checks if the UTC day has changed.
        If yes, resets PnL and Re-initializes Renko to match Training Environment.
        """
        now = datetime.utcnow()
        today = now.date()
        
        if self.current_date is None:
            self.current_date = today
            return
            
        if today > self.current_date:
            logger.info(f"Daily Reset Detected: {self.current_date} -> {today}")
            
            # 1. Reset PnL
            self.state.update("daily_pnl", 0.0)
            self.state.update("risk_limit_hit_warned", False)
            logger.info("Daily PnL reset to 0.0")
            
            # 2. Reset Renko
            # To be precise, we want the Open Price of the New Day.
            # If we call this exactly at 00:00:01, current tick price is close to Open.
            # Ideally, fetch the M1 bar for 00:00 to get exact Open.
            
            # 2. Reset Renko & Session
            # Call full initialization to re-optimize and sync with new day dynamics
            self._initialize_session(force_history_fetch=True)
            
            # Update Date
            self.current_date = today
            logger.info("Daily Reset Complete. System Ready.")

    def _check_active_trades_closure(self):
        """
        Iterates over all active tickets and checks their status.
        Removes closed tickets and updates PnL.
        """
        active_tickets = self.state.get("active_tickets", [])
        if not active_tickets:
             return
             
        import MetaTrader5 as mt5
        
        # We need a copy to iterate safely while modifying
        still_active = []
        details_map = self.state.get("active_ticket_details", {})
        dirty = False
        
        # Optimization: Fetch all history deals once for today
        from_time = datetime.now() - timedelta(hours=24)
        deals = mt5.history_deals_get(date_from=from_time, date_to=datetime.now() + timedelta(minutes=1))
        
        closed_deals_map = {}
        if deals:
            for d in deals:
                if d.entry == mt5.DEAL_ENTRY_OUT:
                    closed_deals_map[d.position_id] = d
        
        for ticket in active_tickets:
            # Check if Open
            positions = mt5.positions_get(ticket=ticket)
            
            if positions:
                still_active.append(ticket)
            else:
                # CLOSED
                dirty = True
                
                # Find Deal - ROBUST METHOD: Fetch specific deal by Position ID
                # This ignores time range issues completely
                my_deal = None
                
                # 1. Try Cached Bulk Lookup (Optimization)
                my_deal = closed_deals_map.get(ticket)
                
                # 2. If not found, force specific lookup (Robustness)
                if not my_deal:
                    history = mt5.history_deals_get(position=ticket)
                    if history:
                        for h in history:
                            if h.entry == mt5.DEAL_ENTRY_OUT:
                                my_deal = h
                                break
                
                if my_deal:
                    # Determine Outcome from Cached Details
                    t_info = details_map.get(str(ticket), {})
                    entry_price = t_info.get('entry', 0.0)
                    direction = t_info.get('direction', 0)
                    close_price = my_deal.price
                    
                    be_threshold = self.brick_size * 0.1
                    win_threshold = self.brick_size * 0.8
                    
                    price_diff = (close_price - entry_price) * direction
                    
                    unit_pnl = 0.0
                    outcome_str = "BE"
                    
                    if abs(price_diff) < be_threshold or my_deal.reason == mt5.DEAL_REASON_SL:
                        if abs(close_price - entry_price) < be_threshold:
                            unit_pnl = 0.0
                            outcome_str = "BE"
                        elif price_diff <= -win_threshold:
                            unit_pnl = -0.5
                            outcome_str = "LOSS"
                        else:
                            if price_diff < 0: unit_pnl = -0.5; outcome_str = "LOSS"
                            else: unit_pnl = 0.5; outcome_str = "WIN"
                    else:
                        if price_diff >= win_threshold:
                            unit_pnl = 0.5
                            outcome_str = "WIN"
                        elif price_diff <= -win_threshold:
                            unit_pnl = -0.5
                            outcome_str = "LOSS"
                        else:
                            unit_pnl = 0.0
                            outcome_str = "BE"
                            
                    logger.info(f"Trade {ticket} Closed. Outcome: {outcome_str} (PnL: {unit_pnl})")
                    
                    # Update PnL
                    current_daily = self.state.get("daily_pnl", 0.0)
                    self.state.update("daily_pnl", current_daily + unit_pnl)
                    
                    # Remove from details
                    if str(ticket) in details_map:
                        del details_map[str(ticket)]
                        
                else:
                    logger.warning(f"Trade {ticket} closed but Deal not found based on history lookup.")
                    # Still remove it from active
                    if str(ticket) in details_map:
                        del details_map[str(ticket)]
                        
        if dirty:
            self.state.update("active_tickets", still_active)
            self.state.update("active_ticket_details", details_map)
            
            # Sync scalar 'active_ticket' to the last one (or 0) for display compatibility
            if still_active:
                latest = still_active[-1]
                self.state.update("active_ticket", latest)
            else:
                self.state.update("active_ticket", 0)
                self.state.update("active_entry_price", 0.0)

    def pulse(self):
        """
        Single heartbeat of the loop.
        """
        # 0. Daily Reset Check
        self.check_new_day()
        
        # 2. Check Active Trade Outcome (MULTI-TRADE)
        # We MUST do this BEFORE Risk Check (or even if Risk Limit hit) to manage existing trades.
        self._check_active_trades_closure()
        
        # 1. Risk Check
        if not self.risk.check_daily_limit():
            # Stop Trading Triggered
            # We skip: Pending Checks, Tick Processing, Renko Updates (Signal Generation)
            # But we return TRUE to keep the loop alive so check_new_day() can eventually fire.
            
            # Sleep briefly to avoid CPU spin since we aren't waiting on socket ticks
            time.sleep(1)
            return True

        # 3. Check Pending Orders (Limit Fallback Monitoring)
        pending_ticket = self.state.get("pending_ticket")
        if pending_ticket:
            import MetaTrader5 as mt5
            # Is it still pending?
            orders = mt5.orders_get(ticket=pending_ticket)
            
            if not orders:
                # It's gone from Pending. Did it fill?
                # Check Positions
                positions = mt5.positions_get(ticket=pending_ticket)
                if positions:
                    logger.info(f"Limit Order {pending_ticket} FILLED!")
                    # Promote to Active
                    pos = positions[0]
                    self.state.update("active_ticket", pending_ticket)
                    self.state.update("active_entry_price", pos.price_open)
                    dir = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
                    self.state.update("active_direction", dir)
                    
                    # Clear Pending State
                    self.state.update("pending_ticket", 0)
                    self.state.update("pending_be_level", 0.0)
                else:
                    # Cancelled or Expired?
                    logger.info(f"Limit Order {pending_ticket} expired or cancelled externally.")
                    self.state.update("pending_ticket", 0)
            else:
                # Still Pending. Check Price for Invalidation (Runs Away).
                be_level = self.state.get("pending_be_level", 0.0)
                direction = self.state.get("pending_direction", 0)
                
                cancel_needed = False
                if direction == 1: # Buy Limit
                    # If Bid rises ABOVE BE Level (Run away profit)
                    curr_bid = mt5.symbol_info_tick(SYMBOL).bid
                    if curr_bid > be_level:
                        logger.warning(f"Pending Buy Missed Move ({curr_bid} > {be_level}). Cancelling.")
                        cancel_needed = True
                elif direction == -1: # Sell Limit
                    # If Ask drops BELOW BE Level (Run away profit)
                    curr_ask = mt5.symbol_info_tick(SYMBOL).ask
                    if curr_ask < be_level:
                        logger.warning(f"Pending Sell Missed Move ({curr_ask} < {be_level}). Cancelling.")
                        cancel_needed = True
                        
                if cancel_needed:
                    if self.orders.cancel_order(pending_ticket):
                        self.state.update("pending_ticket", 0)
            
        # 4. Fetch Ticks (Gap-less)
        new_ticks = self.clock.fetch()
        if not new_ticks:
            time.sleep(0.001) 
            return True
            
        # 4. Process Ticks
        for t in new_ticks:
            price = t['bid'] 
            # Apply Timezone Offset to Live Ticks
            ts = t['time_msc'] + (TIMEZONE_OFFSET * 3600 * 1000)
            
            # Apply Offset to tick Dict for M1 Buffer
            t_shifted = t.copy()
            t_shifted['time'] = t['time'] + (TIMEZONE_OFFSET * 3600)
            
            # Update M1 Accumulator
            self.update_m1_buffer(t_shifted)
            
            # A. Update Renko
            new_bricks = self.renko.update_tick(price, ts)
            
            # B. Intra-Brick Logic (BE Check)
            be_price = self.renko.get_be_price()
            if be_price:
                 should_trigger = False
                 if self.renko.uptrend == 1 and price >= be_price:
                     should_trigger = True
                 elif self.renko.uptrend == -1 and price <= be_price:
                     should_trigger = True
                     
                 if should_trigger:
                     self.orders.move_sl_to_entry(SYMBOL)
            
            # C. New Brick Handling
            for brick in new_bricks:
                logger.info(f"New Brick: {brick}")
                self.process_signal(brick)
                
        return True
        
    def update_m1_buffer(self, tick):
        """
        Maintains the self.m1_buffer DataFrame in real-time.
        """
        ts_sec = tick['time'] # Epoch seconds
        # Tick time is UTC (usually). History DF was converted to Broker Time (UTC+Offset).
        # We need to maintain consistency.
        # History: 'date' = pd.to_datetime(history_df['time'], unit='s') where 'time' was adjusted.
        
        # Here `tick['time']` is raw from MT5.
        # We need to shift it for the index.
        adjusted_ts = ts_sec + (TIMEZONE_OFFSET * 3600)
        dt = pd.to_datetime(adjusted_ts, unit='s')
        
        # Floor to minute
        dt_floored = dt.floor('min')
        
        # Fix: Ensure indices are compatible (Timestamp vs DatetimeIndex)
        if hasattr(self.m1_buffer.index, 'tz'):
             # If buffer is tz-aware, make sure dt_floored is too? 
             # Usually history is tz-naive (after conversion).
             pass
             
        # Check if we have a bar for this minute
        if dt_floored not in self.m1_buffer.index:
            # Create new bar
            new_row = pd.DataFrame([{
                'open': tick['bid'],
                'high': tick['bid'],
                'low': tick['bid'],
                'close': tick['bid'],
                'volume': tick['volume'] 
            }], index=[dt_floored])
            self.m1_buffer = pd.concat([self.m1_buffer, new_row])
            
            # Keep buffer size manageable (e.g. 1000 bars)
            if len(self.m1_buffer) > 2000:
                self.m1_buffer = self.m1_buffer.iloc[-1000:]
        else:
            # Update current bar
            # LOC usage for safety
            self.m1_buffer.loc[dt_floored, 'high'] = max(self.m1_buffer.loc[dt_floored, 'high'], tick['bid'])
            self.m1_buffer.loc[dt_floored, 'low'] = min(self.m1_buffer.loc[dt_floored, 'low'], tick['bid'])
            self.m1_buffer.loc[dt_floored, 'close'] = tick['bid']
            self.m1_buffer.loc[dt_floored, 'volume'] += tick['volume']
            
    def process_signal(self, brick):
        # 1. Feature Engineering
        prev = self.renko.history[-2] if len(self.renko.history) > 1 else brick
        b_dict = brick._asdict() 
        p_dict = prev._asdict() # Contains 'sequence' now
        
        # Pass the M1 Buffer!
        ind_dict = self.features.get_indicators(self.m1_buffer)
        
        # DEBUG: Always log indicators to catch static data
        logger.info(f"Ind Dict: {ind_dict}")
        if not self.m1_buffer.empty:
             logger.info(f"M1 Buffer Last: {self.m1_buffer.index[-1]}")
        else:
             logger.warning("M1 Buffer Empty!")
        
        preds = self.predictors.predict(brick, self.renko.history[:-1], ind_dict) 
        
        # DYNAMIC TIME LEFT and PNL from State
        time_left = self.get_normalized_time_left()
        pnl = self.state.get("daily_pnl", 0.0)
        # Fix: Clip PnL to match Training Limits [-5, 5]
        pnl = max(-5.0, min(5.0, pnl))
        
        obs = self.features.calculate_state(
            b_dict, p_dict, self.m1_buffer,
            preds, 
            pnl,
            time_left,
            renko_history=self.renko.history,
            brick_size_val=self.brick_size
        )
        
        # --- FEATURE MASKING FIX (V2 Adaptation) ---
        # FeatureEngineer V2 outputs 30 dims (no masking needed for structure)
        # But we must ensure Orbit Logic doesn't break if it expects 21.
        # Orbit expects obs to be passed to ensemble.predict.
        # Our MetaWrapper .predict(obs) handles the 30-dim vector.
        # So we do NOT need to mask [3:7] anymore.
        # Removing masking logic to preserve V2 integrity.
        # obs[3:7] = 0.0
        
        # DEBUG: Log Full State Vector
        logger.info(f"State Vector: {obs}")
        # ---------------------------
        # ---------------------------
        
        # Update Stack
        self.obs_stack.append(obs)
        # Prepare Stack for Transformer (1, 10, 21)
        stack_arr = np.array(self.obs_stack)
        if len(stack_arr) < 10:
             padding = np.zeros((10 - len(stack_arr), 21))
             stack_arr = np.vstack([padding, stack_arr])
        
        # 2. Latency Check (Catch-Up Mode)
        # Check if brick is "Live" or "History"
        # We compare brick timestamp (adjusted) to System Time (adjusted same way if needed)
        # Simplest: Compare to current UTC timestamp + Offset
        
        system_time_ms = time.time() * 1000
        # If brick has offset applied, we apply same to system? 
        # In pulse: ts = t['time_msc'] + (TIMEZONE_OFFSET * 3600 * 1000)
        # brick.timestamp from pulse ts.
        # So we shift system time too.
        current_adjusted_ms = system_time_ms + (TIMEZONE_OFFSET * 3600 * 1000)
        
        latency_ms = current_adjusted_ms - brick.timestamp
        is_catchup = latency_ms > 60000 # 60 Seconds lag
        
        # DEBUG LOGS
        # logger.info(f"Brick TS: {brick.timestamp}, Sys: {current_adjusted_ms}, Latency: {latency_ms}")
        
        if is_catchup:
             if len(self.renko.history) % 10 == 0:
                 # Fix: brick has no .date attribute. Use timestamp conversion.
                 b_date = datetime.fromtimestamp(brick.timestamp / 1000.0)
                 logger.info(f"Catching up... Brick {b_date} (Latency: {latency_ms/1000:.1f}s)")
             
             # Skip Inference and Execution
             # But we MUST update the stack (done above)
             return
        
        # DEBUG INDICATORS
        # logger.info(f"Inds: {ind_dict}")
             
        # 3. Inference
        action, self.lstm_states, score = self.ensemble.predict(
            obs, 
            lstm_states=self.lstm_states, 
            episode_starts=self.episode_starts,
            obs_stack=stack_arr 
        )
        self.episode_starts = np.array([False])
        
        logger.info(f"Ensemble Vote: {score:.4f} -> Action: {action}")
        
        # 3. Execution (1:1 Ratio)
        if action == 1:
            # MULTI-TRADE LOGIC ----------------------
            # Check Active Tickets List
            active_tickets = self.state.get("active_tickets", [])
            
            # Just-In-Time Closure Check
            # Before skipping, let's force a check on all active tickets to see if they closed.
            if active_tickets:
                self._check_active_trades_closure()
                # Reload list after check
                active_tickets = self.state.get("active_tickets", [])

            # Check for Pending Orders
            if self.state.get("pending_ticket", 0) != 0:
                 logger.info("Signal Skipped: Pending Limit Order active.")
                 return
                 
            # Entry Targets
            entry = brick.close
            dist = self.brick_size
            
            if brick.uptrend:
                sl = entry - dist
                tp = entry + dist
                direction = 1
            else:
                sl = entry + dist
                tp = entry - dist
                direction = -1
                
            # Filter Logic: "Enter if TP != Current TP"
            if active_tickets:
                # Get Latest Trade details
                details_map = self.state.get("active_ticket_details", {})
                latest_ticket = active_tickets[-1]
                latest_info = details_map.get(str(latest_ticket)) # JSON keys are strings
                
                if latest_info:
                    # Robustness: Check tolerance?
                    # TP is double. Use small epsilon or exact match (ticks are discrete usually)
                    # MT5 prices are rounded. State stores floats.
                    # Best: abs(latest_tp - new_tp) < 1e-5
                    last_tp = latest_info.get('tp', 0.0)
                    
                    if abs(last_tp - tp) < 1e-4:
                        logger.info(f"Signal Skipped: Duplicate Trade Intent (TP {tp:.5f} already active on {latest_ticket}).")
                        return
                    else:
                        logger.info(f"Concurrent Entry Triggered: New TP {tp:.5f} != Last TP {last_tp:.5f}")
            
            # ----------------------------------------

            # SLIPPAGE CHECK
            if direction == 1:
                # Buy
                current_price = mt5.symbol_info_tick(SYMBOL).ask
                slippage = current_price - entry
                # Max Slippage: 8% of Brick
                max_slip = self.brick_size * 0.08
                
                # BE Level Logic
                be_level = entry + (self.brick_size * 0.3125)
                
                if slippage > max_slip:
                    logger.warning(f"High Slippage ({slippage:.5f} > {max_slip:.5f}). switch to Limit.")
                    
                    current_bid = mt5.symbol_info_tick(SYMBOL).bid
                    if current_bid > be_level:
                         logger.warning(f"Limit Order Skipped: Price {current_bid} ran away past {be_level}")
                         return
                         
                    ticket = self.orders.send_limit_order(direction, entry, sl, tp)
                    if ticket:
                        self.state.update("pending_ticket", ticket)
                        self.state.update("pending_be_level", be_level)
                        self.state.update("pending_direction", direction)
                    return
            else:
                # Sell
                current_price = mt5.symbol_info_tick(SYMBOL).bid
                slippage = entry - current_price 
                max_slip = self.brick_size * 0.08
                
                be_level = entry - (self.brick_size * 0.3125)
                
                if slippage > max_slip:
                     logger.warning(f"High Slippage ({slippage:.5f} > {max_slip:.5f}). switch to Limit.")
                     current_ask = mt5.symbol_info_tick(SYMBOL).ask
                     if current_ask < be_level:
                          logger.warning(f"Limit Order Skipped: Price {current_ask} ran away past {be_level}")
                          return
                          
                     ticket = self.orders.send_limit_order(direction, entry, sl, tp)
                     if ticket:
                        self.state.update("pending_ticket", ticket)
                        self.state.update("pending_be_level", be_level)
                        self.state.update("pending_direction", direction)
                     return
                
            ticket = self.orders.send_market_order(direction, sl, tp)
            
            if ticket:
                # Add to Multi-Trade State
                active_tickets = self.state.get("active_tickets", [])
                active_tickets.append(ticket)
                self.state.update("active_tickets", active_tickets)
                
                details_map = self.state.get("active_ticket_details", {})
                details_map[str(ticket)] = {
                    "entry": entry,
                    "direction": direction,
                    "tp": tp,
                    "sl": sl
                }
                self.state.update("active_ticket_details", details_map)
                
                # For backward compatibility / display, update the 'scalar' active (pointing to latest)
                self.state.update("active_ticket", ticket)
                self.state.update("active_entry_price", entry)
                self.state.update("active_direction", direction)

    def run(self):
        self.start()
        try:
            while True:
                if not self.pulse():
                    break
        except KeyboardInterrupt:
            logger.info("Orbit Stopped by User")
            self.connector.shutdown()

