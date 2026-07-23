"""
BrickOfTicks MT5 Trading Bot
Phase 7, 8, 13: Main loop with Path Optimization, Warmup, and Daily Sync
"""
import time
import os
import MetaTrader5 as mt5

from BrickOfTicks_Trader.config.settings import (
    SYMBOL, BRICK_SIZE_FACTOR, WARMUP_TICKS, MODELS_DIR, MAGIC_NUMBER,
    PATH_LOOKBACK_DAYS
)
from BrickOfTicks_Trader.data.connector import MT5Connector
from BrickOfTicks_Trader.data.tick_stream import TickStream
from BrickOfTicks_Trader.data.renko import RenkoBuilder
from BrickOfTicks_Trader.data.feature_engine import LiveFeatureEngine
from BrickOfTicks_Trader.data.path_optimizer import PathOptimizer

from BrickOfTicks_Trader.inference.buffer import InferenceBuffer
from BrickOfTicks_Trader.inference.ensemble import EnsemblePredictor

from BrickOfTicks_Trader.execution.orders import OrderExecutor
from BrickOfTicks_Trader.execution.risk import RiskManager

from BrickOfTicks_Trader.utils.logger import logger
from BrickOfTicks_Trader.utils.state import state
from BrickOfTicks_Trader.execution.sync import DailySynchronizer
from BrickOfTicks_Trader.config.settings import STATE_FILE


class OrbitEngine:
    """The central trading engine orchestrating data handling, ML inference, and execution."""
    def __init__(self):
        self.connector = MT5Connector()
        self.stream = None
        
        self.renko = None
        self.features = LiveFeatureEngine()
        self.buffer = InferenceBuffer()
        
        self.ensemble = EnsemblePredictor(str(MODELS_DIR))
        self.orders = OrderExecutor()
        self.risk = RiskManager()
        self.sync = DailySynchronizer()
        self.optimizer = PathOptimizer()
        
        self.is_running = False
        self.status = "INIT"  # INIT, REPLAY, WARMUP, READY

    def initialize(self) -> bool:
        """Connect to MT5, setup starting states, load models."""
        if not self.connector.connect():
            return False
            
        # Get starting ask format
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            logger.error(f"Cannot initialize OrbitEngine: Unable to get tick for {SYMBOL}")
            return False
            
        # Dynamic brick size baseline: ask * factor
        brick_size = tick.ask * BRICK_SIZE_FACTOR
        logger.info(f"Initial dynamic brick size calculated: {brick_size:.4f}")
        
        # We will initialize RenkoBuilder inside _path_warmup using the optimized anchor
        self.renko = None
        self.brick_size = brick_size
        
        # Load Predictors
        try:
            self.ensemble.load()
        except Exception as e:
            logger.error(f"Failed to load ensemble models: {e}")
            return False
            
        self.stream = TickStream()  # Inits with current tick time_msc
        
        return True

    def _path_warmup(self) -> bool:
        """
        Phase 13: Path-Optimized Warmup.
        
        Always performs a cold start:
        1. Fetch multi-day tick history (5 days with fallback)
        2. Run PathOptimizer to find the best Renko starting anchor
        3. Replay all ticks through the engine (features, renko, buffer)
        4. Check integrity gate (>= 5000 ticks AND >= 10 bricks)
        
        Returns True if warmup succeeded and status should be READY.
        """
        logger.info("Starting Path-Optimized Warmup (Phase 13)...")
        
        # 1. Fetch multi-day history
        ticks, num_days = self.stream.fetch_multi_day_history(max_days=PATH_LOOKBACK_DAYS)
        
        if ticks is None or len(ticks) == 0:
            logger.error("Path Warmup: No historical ticks available.")
            return False
        
        logger.info(f"Path Warmup: Got {len(ticks)} ticks across {num_days} trading days.")
        
        # 2. Get current brick size
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            logger.error("Path Warmup: Cannot get current tick for brick size.")
            return False
            
        self.brick_size = tick.ask * BRICK_SIZE_FACTOR
        
        # 3. Run Path Optimizer
        best_price, best_idx, best_profit = self.optimizer.find_optimal_anchor(
            ticks, self.brick_size
        )
        
        if best_price is None:
            logger.warning("Path Warmup: Optimizer failed. Using first tick price as fallback.")
            best_price = float(ticks[0]['bid'])
            best_idx = 0
        
        logger.info(
            f"Path Warmup: Optimal anchor = {best_price:.2f}, "
            f"Profit = {best_profit:.2f}, Starting replay..."
        )
        
        # 4. Initialize fresh ML pipeline with the optimized anchor
        self.renko = RenkoBuilder(brick_size=self.brick_size, start_price=best_price)
        self.features = LiveFeatureEngine()
        self.buffer = InferenceBuffer()
        
        # 5. Replay ALL ticks from the optimized start index forward
        processed_ticks = 0
        bricks_formed = 0
        
        # We replay from best_idx to use the exact anchor point
        # but we feed ALL ticks to the feature engine for z-score saturation
        for i, t in enumerate(ticks):
            try:
                bid = float(t['bid'])
                ask = float(t['ask'])
                bid_v = float(t['volume_real']) if 'volume_real' in t.dtype.names else 0.0
                ask_v = bid_v
                ts_msc = int(t['time_msc'])
            except (IndexError, AttributeError):
                bid, ask, bid_v, ask_v, ts_msc = t.bid, t.ask, t.volume, t.volume, t.time_msc

            # Feed feature engine with every tick for z-score warmup
            vec_9d = self.features.compute_vector(bid, ask, bid_v, ask_v, ts_msc)
            
            # Only feed renko from the anchor index forward
            if i >= best_idx:
                new_bricks = self.renko.update_tick(bid, ts_msc)
                
                for b in new_bricks:
                    bricks_formed += 1
                    self.features.on_new_brick(b)
                    self.buffer.on_brick_close(b)
                
                self.buffer.append_tick(vec_9d, self.features.current_brick_id)
            
            processed_ticks += 1
        
        # 6. Integrity Gate
        snapshots = len(self.buffer.snapshots)
        z_depth = len(self.features.z_ofi.deque)
        
        logger.info(
            f"Path Warmup Complete: {processed_ticks} ticks, "
            f"{bricks_formed} bricks, {snapshots} snapshots, "
            f"Z-score depth: {z_depth}"
        )
        
        if snapshots >= 10 and processed_ticks >= 5000:
            logger.info("Path Warmup: Integrity Gate PASSED. Status -> READY.")
            
            # Save state checkpoint after successful warmup
            state.save_internal_state(self.features, self.buffer, self.renko)
            
            return True
        else:
            logger.warning(
                f"Path Warmup: Integrity Gate FAILED "
                f"(snapshots={snapshots}, ticks={processed_ticks}). "
                f"Entering WARMUP lock."
            )
            return False

    def run(self):
        """Phase 8 & 13: Main loop with forced cold-start path optimization."""
        if not self.initialize():
            logger.error("Initialization failed. Exiting.")
            return
            
        self.status = "REPLAY"
        
        # Phase 13: Always cold start with path optimization
        # (No pickle bypass — recalculate best path on every startup)
        logger.info("Phase 13: Forced cold start. Running path optimization...")
        
        if self._path_warmup():
            self.status = "READY"
        else:
            logger.warning("Path warmup incomplete. Entering WARMUP lock state.")
            self.status = "WARMUP"
            # If renko is still None after failed warmup, create with current price
            if self.renko is None:
                tick = mt5.symbol_info_tick(SYMBOL)
                self.renko = RenkoBuilder(
                    brick_size=tick.ask * BRICK_SIZE_FACTOR,
                    start_price=tick.bid
                )
        
        self.is_running = True
        logger.info(f"Bot is LIVE and looping ticks... Status: {self.status}")
        
        while self.is_running:
            # 1. Phase 11: Daily Synchronization
            new_brick_size = self.sync.check_and_sync()
            if new_brick_size is not None:
                # Daily rollover detected — re-run path optimization
                logger.info("Daily rollover detected. Re-running path optimization...")
                
                if self.renko:
                    self.renko.update_brick_size(new_brick_size)
                self.features.update_brick_size(new_brick_size)
                self.brick_size = new_brick_size
                logger.info(f"Propagated new brick size {new_brick_size:.4f} to components.")
                
                # Re-run path optimization for the new day
                if self._path_warmup():
                    self.status = "READY"
                    logger.info("Daily path re-optimization complete. Status: READY.")
                else:
                    logger.warning("Daily path re-optimization failed. Continuing with current state.")
                
            # 2. Risk checkpoint
            if not self.risk.check_daily_limit():
                logger.error("Daily limit breached. Shutting down trading.")
                break
                
            # Disconnect protection
            if not self.connector.check_connection():
                time.sleep(5)
                continue
                
            new_ticks = self.stream.fetch()
            for t in new_ticks:
                try:
                    bid = float(t['bid'])
                    ask = float(t['ask'])
                    vol = float(t['volume']) if 'volume_real' not in t else float(t['volume_real'])
                    ts_msc = int(t['time_msc'])
                except:
                    bid, ask, vol, ts_msc = t.bid, t.ask, t.volume, t.time_msc
                
                # Update features
                vec_9d = self.features.compute_vector(bid, ask, vol, vol, ts_msc)
                


                # Update renko
                new_bricks = self.renko.update_tick(bid, ts_msc)
                
                for b in new_bricks:
                    self.features.on_new_brick(b)
                    
                    # Safety Audit: Log Brick OHLC
                    logger.info(
                        f"BRICK CLOSED | ID: {b.sequence[-5:]} | "
                        f"Type: {'UP' if b.uptrend else 'DOWN'} | "
                        f"O: {b.open:.2f} H: {b.high:.2f} L: {b.low:.2f} C: {b.close:.2f}"
                    )
                    
                    # On close, get tensors ready
                    tensors = self.buffer.on_brick_close(b)
                    
                    # Persistence Checkpoint
                    state.save_internal_state(self.features, self.buffer, self.renko)
                    
                    # If in WARMUP lock, check if we escaped
                    if self.status == "WARMUP":
                        if len(self.buffer.snapshots) >= 10 and len(self.features.z_ofi.deque) >= 5000:
                            self.status = "READY"
                            logger.info("Live WARMUP complete. Status -> READY.")
                            
                    if tensors is not None:
                        micro_t, macro_t = tensors
                        
                        # Phase 5: Inference (Run even if not trading to log "Safety" data)
                        prediction = self.ensemble.predict(micro_t, macro_t)
                        
                        # Safety Audit: Log Model Brain State
                        for d in prediction["details"]:
                            logger.info(
                                f"  Fold {d['fold']} | Prob: {d['prob_win']:.4f} | "
                                f"Pred: {d['pred_os']:.4f} | Signal: {d['signal']}"
                            )
                        
                        if self.status == "READY":
                            act = prediction["action"]
                            ttype = prediction["trade_type"]
                            
                            if act != 0:
                                logger.info(f"SIGNAL TRIGGERED: Action={act}, Type={ttype}, Votes={prediction['votes']}")
                                
                                # Slippage verification before market order
                                current_ask = mt5.symbol_info_tick(SYMBOL).ask
                                current_bid = mt5.symbol_info_tick(SYMBOL).bid
                                
                                # Baiting inverts direction. UP brick = 1, DOWN = -1. If Trade=Baiting (-1), dir becomes -1 * dir
                                brick_dir = 1 if b.uptrend else -1
                                trade_dir = brick_dir * act
                                
                                exec_price = current_ask if trade_dir == 1 else current_bid
                                
                                if self.risk.check_slippage(exec_price, b.close, self.renko.brick_size):
                                    # Use pending order as fallback
                                    sl = exec_price - self.renko.brick_size if trade_dir == 1 else exec_price + self.renko.brick_size
                                    tp = exec_price + self.renko.brick_size if trade_dir == 1 else exec_price - self.renko.brick_size
                                    
                                    # Assuming standard limit pricing (can adjust based on strategy)
                                    limit_price = b.close # target the missed level
                                    self.orders.send_limit_order(trade_dir, limit_price, sl, tp, "LimitFallback")
                                else:
                                    # Normal Market Entry
                                    sl = exec_price - self.renko.brick_size if trade_dir == 1 else exec_price + self.renko.brick_size
                                    tp = exec_price + self.renko.brick_size if trade_dir == 1 else exec_price - self.renko.brick_size
                                    
                                    ticket = self.orders.send_market_order(trade_dir, sl, tp, f"Bot_{ttype}")
                                    
                                    if ticket:
                                        # Save ticket to state to track
                                        state.update("last_ticket", ticket)
                                        
                # Always append to buffer, continuous tracking
                self.buffer.append_tick(vec_9d, self.features.current_brick_id)
                
            time.sleep(0.01) # 10ms poll yield
            
        self.connector.shutdown()

if __name__ == "__main__":
    engine = OrbitEngine()
    try:
        engine.run()
    except KeyboardInterrupt:
        logger.info("Bot manually interrupted. Shutting down gracefully.")
        engine.is_running = False
        engine.connector.shutdown()
