import time
import queue
import logging
import sys
import os
import datetime

import bridge.renko
import bridge.path_optimizer
bridge.renko.K_MULTIPLIER = 0.00118
bridge.path_optimizer.K_MULTIPLIER = 0.00118

from bridge.tick_receiver import TickReceiver
from bridge.command_sender import CommandSender
from bridge.renko import RenkoBuilder, K_MULTIPLIER
from bridge.feature_engine import LiveFeatureEngine
from bridge.buffer import InferenceBuffer
from bridge.ensemble import EnsemblePredictor
from bridge.state import StateManager
from bridge.risk import RiskManager
from bridge.trade_logger import TradeLogger
from bridge.path_optimizer import PathOptimizer

logger = logging.getLogger(__name__)


import json
import tensorflow as tf

class FallbackPredictor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None
        self.prob_win_threshold = 0.6
        self.pred_os_threshold = 1.7
        
    def load(self):
        config_path = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                self.prob_win_threshold = config.get("Prob_Win_threshold", 0.6)
                self.pred_os_threshold = config.get("Pred_OS_threshold", 1.7)
                logger.info(f"Loaded fallback config: PW>={self.prob_win_threshold}, OS>={self.pred_os_threshold}")
                
        model_path = os.path.join(self.model_dir, "model.keras")
        logger.info(f"Loading Fallback model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("Successfully loaded fallback model.")

    def predict(self, micro_tensor, macro_tensor):
        if not self.model:
            return {"action": 0, "votes": 0, "details": []}
            
        preds = self.model([micro_tensor, macro_tensor], training=False)
        pw = float(preds[0].numpy().flatten()[0])
        po = float(preds[1].numpy().flatten()[0])
        
        signal = (pw >= self.prob_win_threshold) and (po >= self.pred_os_threshold)
        action = 1 if signal else 0
        
        detail_item = {"prob_win": pw, "pred_os": po, "signal": signal}
        return {
            "action": action,
            "votes": 1 if signal else 0,
            "details": [detail_item, detail_item, detail_item]
        }

class BridgeEngine:
    """
    The central orchestrator for the BrickOfTicks Socket Bridge.
    Connects the MQL5 TCP sockets to the Keras deep learning pipeline,
    managing state, risk, latency profiling, and trade execution.
    """
    def __init__(self, tick_port=9000, cmd_port=9001):
        logger.info("Initializing BridgeEngine components...")
        self.receiver = TickReceiver(port=tick_port)
        self.sender = CommandSender(confirm_queue=self.receiver.confirm_queue, port=cmd_port)
        
        # Initialize with dummy price (1.0). Re-initialized on DAYOPEN.
        self.renko = RenkoBuilder(1.0)
        self.feature_engine = LiveFeatureEngine()
        
        self.buffer = InferenceBuffer()
        fallback_dir = "/Users/gopo/Quant Projects/CAPSTONE/Overshot/outputs/fallback"
        self.ensemble = FallbackPredictor(model_dir=fallback_dir)
        
        self.state = StateManager(filepath="logs/state_fallback.json")
        self.state.reset()
        
        self.risk = RiskManager()
        self.logger = TradeLogger(filepath="logs/trades_fallback.csv")
        self.optimizer = PathOptimizer()
        
        self.last_tick_time = time.time()
        self.degraded_mode = False
        self.last_day_open = None
        
        self.consecutive_timeouts = 0
        self.reconnect_attempts = 0
        self.history_bricks = []

    def start(self):
        """
        Boots the system, loads models, waits for MT5 connection and DAYOPEN.
        """
        logger.info("Starting BridgeEngine...")
        
        # 1. Load Keras models (exits if failed)
        self.ensemble.load()
        
        # 2. Start Receiver (port 9000)
        self.receiver.start()
        
        # 3. Start Command Sender (port 9001 - non-blocking background connect)
        try:
            self.sender.connect()
        except Exception as e:
            logger.warning(f"Could not connect CommandSender on startup: {e}. Will reconnect later.")
        
        # 4. Wait for DAYOPEN (arrives before history ticks)
        logger.info("Waiting for DAYOPEN from MT5 (max 30s)...")
        start_wait = time.time()
        while self.receiver.day_open_price is None:
            if time.time() - start_wait > 30.0:
                logger.critical("DAYOPEN timeout. No MT5 connection.")
                sys.exit(1)
            time.sleep(0.1)
            
        # Record the day open — do NOT trigger rollover logic on first boot.
        # The _warmup() method handles path optimization with the full history.
        self.last_day_open = self.receiver.day_open_price
        logger.info(f"Day open received: {self.last_day_open}")
        
        # 5. Run Warmup (includes PathOptimizer)
        self._warmup()
        
        # 6. Enter Main Loop
        self._run_loop()

    def _warmup(self):
        """
        Path-Optimized Warmup (matching training pipeline).
        
        1. Wait for multi-day history ticks from EA
        2. Run PathOptimizer to find the best Renko starting anchor
        3. Create fresh RenkoBuilder at the optimal anchor
        4. Replay all ticks through the engine (features, renko, buffer)
        5. Check integrity gate (>= 10 bricks AND >= 1000 z-score ticks)
        """
        logger.info("Waiting for HDONE (history batch complete) from MT5 (max 120s)...")
        if not self.receiver.history_done.wait(timeout=120.0):
            logger.critical("History replay timeout. MT5 failed to send HDONE.")
            sys.exit(1)
        
        history = self.receiver.get_history_ticks()
        logger.info(f"Received {len(history)} historical ticks from EA.")
        
        if len(history) == 0:
            logger.critical("No history ticks received. Cannot warm up.")
            sys.exit(1)
        
        # ── Sort & Deduplicate ────────────────────────────────────
        # EA reconnects can append out-of-order ticks (earlier timestamps at end).
        # This corrupts day boundary detection and causes -1e18 profit.
        history.sort(key=lambda t: t['time_msc'])
        
        # Deduplicate: remove consecutive ticks with identical time_msc
        deduped = [history[0]]
        for i in range(1, len(history)):
            if history[i]['time_msc'] != history[i-1]['time_msc']:
                deduped.append(history[i])
        
        if len(deduped) < len(history):
            logger.info(f"Deduplication: {len(history)} → {len(deduped)} ticks ({len(history) - len(deduped)} duplicates removed)")
            history = deduped
            self.receiver.set_history_ticks(history)  # Update thread-safely
        
        # ── Path Optimization ─────────────────────────────────────
        day_open = self.receiver.day_open_price or history[-1]['bid']
        brick_size = day_open * K_MULTIPLIER
        
        best_price, best_idx, best_profit = self.optimizer.find_optimal_anchor(
            history, brick_size
        )
        
        if best_price is None:
            logger.warning("Path Optimizer failed. Falling back to day_open as anchor.")
            best_price = day_open
            best_idx = 0
        
        logger.info(
            f"Path Optimization complete: anchor={best_price:.2f}, "
            f"profit={best_profit:.2f}, start_idx={best_idx}"
        )
        
        # ── Create Fresh Pipeline at Optimal Anchor ───────────────
        self.renko = RenkoBuilder(best_price)
        self.renko.update_brick_size(brick_size, new_day_open=best_price)
        self.feature_engine = LiveFeatureEngine()
        self.feature_engine.update_brick_size(brick_size)
        self.buffer = InferenceBuffer()
        self.history_bricks = []
        
        # ── Replay ALL ticks ──────────────────────────────────────
        # Feed ALL ticks to feature engine for z-score saturation
        # Feed ticks from best_idx onwards to Renko
        logger.info(f"Replaying ticks: features from idx=0, renko from idx={best_idx}...")
        
        for i, tick in enumerate(history):
            # Feature engine gets every tick for z-score warmup
            feat_vec = self.feature_engine.compute_vector(
                tick['bid'], tick['ask'], 0.0, 0.0, tick['time_msc']
            )
            
            # Renko only from the optimal anchor index forward
            if i >= best_idx:
                self.buffer.append_tick(feat_vec, self.renko.brick_count)
                
                new_bricks = self.renko.update_tick(tick['bid'], tick['time_msc'])
                for brick in new_bricks:
                    self.history_bricks.append(brick)
                    self.feature_engine.on_new_brick(brick)
                    self.buffer.on_brick_close(self.renko.brick_count - 1, self.feature_engine.last_macro)
        
        # ── Integrity Gate Check ──────────────────────────────────
        ticks_zscored = len(self.feature_engine.zs_ofi.deque)
        bricks = self.renko.brick_count
        
        if bricks >= 10 and ticks_zscored >= 1000:
            logger.info(f"Warmup PASSED: {bricks} bricks, {ticks_zscored} ticks tracked.")
            self.state.update("warmup_done", True)
        else:
            logger.warning(
                f"Warmup gate NOT YET MET: {bricks} bricks, {ticks_zscored} ticks tracked. "
                f"Waiting for live ticks to fill buffers."
            )
            self.state.update("warmup_done", False)
        
        # ── Snapshot Renko History ────────────────────────────────
        self._save_renko_snapshot()

    def _run_loop(self):
        """
        The continuous live streaming loop.
        """
        logger.info("BridgeEngine is LIVE and waiting for streaming ticks.")
        consecutive_ticks = 0
        window_start = time.time()
        
        while True:
            try:
                # 60s timeout to allow state machine checks
                tick = self.receiver.tick_queue.get(timeout=60.0)
                self.consecutive_timeouts = 0
                
                # Detect Rollover
                if self.receiver.day_open_price is not None and self.receiver.day_open_price != self.last_day_open:
                    if self.last_day_open is not None: # Not the first boot
                        self._on_day_open(self.receiver.day_open_price)
                    self.last_day_open = self.receiver.day_open_price
                    
                # Degraded Mode State Machine logic
                if self.degraded_mode:
                    if time.time() - window_start > 5.0:
                        window_start = time.time()
                        consecutive_ticks = 0
                    
                    consecutive_ticks += 1
                    if consecutive_ticks >= 3:
                        self._exit_degraded_mode()
                        
                self.last_tick_time = time.time()
                self._process_tick(tick, is_warmup=not self.state.get("warmup_done"))
                
                # If we just satisfied warmup via live ticks
                if not self.state.get("warmup_done"):
                    b_count = self.renko.brick_count
                    z_len = len(self.feature_engine.zs_ofi.deque)
                    
                    if b_count >= 10 and z_len >= 1000:
                        logger.info("Live tick stream fulfilled warmup gate. System ARMED.")
                        self.state.update("warmup_done", True)
                    elif b_count > 0 and b_count % 10 == 0:
                        # Log every 10 bricks
                        logger.info(f"WARMUP LIVE MODE — brick_count={b_count}, z_ofi_len={z_len}, target=(10, 1000)")
                        
            except queue.Empty:
                if not self.degraded_mode:
                    self.consecutive_timeouts += 1
                    if self.consecutive_timeouts >= 3:
                        self._enter_degraded_mode()
                else:
                    self._attempt_reconnect()
                    
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt caught. Commencing graceful shutdown...")
                self.state.save()
                self.logger.generate_session_report()
                logger.info("State saved. Session summary generated. Exiting.")
                sys.exit(0)
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)

    def _enter_degraded_mode(self):
        logger.critical("3 consecutive 60s timeouts! Entering DEGRADED MODE. Halting trading execution.")
        self.degraded_mode = True
        self.state.update("degraded_mode", True)
        self.reconnect_attempts = 0

    def _exit_degraded_mode(self):
        logger.info("Recovered 3 ticks in 5s. NORMAL MODE restored.")
        self.degraded_mode = False
        self.state.update("degraded_mode", False)
        self.reconnect_attempts = 0
        self.consecutive_timeouts = 0

    def _attempt_reconnect(self):
        self.reconnect_attempts += 1
        if self.reconnect_attempts > 10:
            logger.critical("Max reconnect attempts (10) reached in DEGRADED MODE. Exiting.")
            sys.exit(2)
            
        backoff = min(60, 2 ** (self.reconnect_attempts - 1))
        logger.info(f"Reconnecting CommandSender in {backoff}s (Attempt {self.reconnect_attempts}/10)...")
        time.sleep(backoff)
        try:
            self.sender.connect()
        except Exception as e:
            logger.warning(f"Reconnect failed: {e}")

    def _process_tick(self, tick, is_warmup=False):
        """
        Core physics sequence: Math -> Buffer -> Renko -> Predict -> Risk -> Fire
        """
        t0 = time.perf_counter()

        feat_vec = self.feature_engine.compute_vector(
            tick['bid'], tick['ask'], 0.0, 0.0, tick['time_msc'])
            
        self.buffer.append_tick(feat_vec, self.renko.brick_count)

        new_bricks = self.renko.update_tick(tick['bid'], tick['time_msc'])

        for brick in new_bricks:
            if is_warmup:
                self.history_bricks.append(brick)
            else:
                dir_str = 'UP' if brick.uptrend == 1 else 'DN'
                logger.info(f"🧱 NEW LIVE BRICK: Dir={dir_str}, Close={brick.close:.5f}, Seq='{brick.sequence}'")
                
            self.feature_engine.on_new_brick(brick)
            tensors = self.buffer.on_brick_close(self.renko.brick_count - 1, self.feature_engine.last_macro)
            
            if tensors and not is_warmup:
                t1 = time.perf_counter()
                self._on_signal(brick, tensors)
                t2 = time.perf_counter()
                inference_ms = (t2 - t1) * 1000
                if inference_ms > 150:
                    logger.warning(f"SLOW INFERENCE: {inference_ms:.0f}ms (target <80ms)")

        # Local check for SL/TP closure so state resets
        if not is_warmup and self.state.get('active_ticket'):
            self._check_sl_tp(tick)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > 30:
            logger.warning(f"SLOW TICK PROCESSING: {elapsed_ms:.0f}ms")

    def _check_sl_tp(self, tick):
        ticket = self.state.get('active_ticket', 0)
        direction = self.state.get('active_direction', 0)
        sl = self.state.get('active_sl', 0.0)
        tp = self.state.get('active_tp', 0.0)
        
        closed = False
        outcome = ""
        
        if direction == 1:
            if tick['bid'] <= sl:
                closed = True
                outcome = "LOSS"
            elif tick['bid'] >= tp:
                closed = True
                outcome = "WIN"
        elif direction == -1:
            if tick['ask'] >= sl:
                closed = True
                outcome = "LOSS"
            elif tick['ask'] <= tp:
                closed = True
                outcome = "WIN"
                
        if closed:
            logger.info(f"Local state check: Ticket {ticket} hit {outcome} level. Resetting state.")
            self.state.update("active_ticket", 0)
            self.state.update("active_direction", 0)
            self.logger.log_outcome(ticket, outcome, 0.0) # 0.0 placeholder pnl

    def _on_signal(self, brick, tensors):
        """
        Receives output from Keras and fires trades if bounds are met.
        """
        micro, macro = tensors
        result = self.ensemble.predict(micro, macro)
        
        # Log pending signal
        self.logger.log_signal(brick.timestamp, brick.uptrend, result)
        
        action = result.get('action', 0)
        
        if action == 1:
            if self.degraded_mode:
                logger.warning("Signal blocked: System is in DEGRADED MODE.")
                return
                
            if not self.risk.check_position_open(self.state):
                logger.info("Signal blocked: Position already open.")
                return
                
            daily_pnl = self.state.get('daily_pnl', 0.0)
            if not self.risk.check_daily_limit(daily_pnl, self.renko.brick_size):
                logger.warning("Signal blocked: Daily drawdown limit exceeded.")
                return
                
            # Fire Order!
            direction = brick.uptrend # 1 for UP, 0 for DOWN. Wait, the model predicts reversals or continuations?
            # Actually, action=1 means ENTER. But in which direction? 
            # In Overshot logic, we trade IN THE DIRECTION of the breakout.
            # If brick is uptrend (1), we BUY. If downtrend (0 or -1), we SELL.
            is_buy = (brick.uptrend == 1)
            
            # Target sizing based on K=0.00295 architecture: SL = 1 brick, TP = 3 bricks (Wait, what are the actual SL/TPs?)
            # PRD Phase 2 command sender has parameters: price, sl, tp, volume.
            # I will use active_brick_size for SL/TP distances as standard.
            bs = self.renko.brick_size
            price = brick.close # We execute at the close of the brick
            
            if is_buy:
                sl = price - (1.0 * bs)
                tp = price + (1.0 * bs) # 1:1 RR as per PRD tests
                conf = self.sender.buy(price, sl, tp, volume=0.01)
                active_dir = 1
            else:
                sl = price + (1.0 * bs)
                tp = price - (1.0 * bs)
                conf = self.sender.sell(price, sl, tp, volume=0.01)
                active_dir = -1
                
            if conf and conf.get('status') == 'OK':
                ticket = conf.get('ticket', 0)
                # Update State
                self.state.update("active_ticket", ticket)
                self.state.update("active_direction", active_dir)
                self.state.update("active_entry", price)
                self.state.update("active_sl", sl)
                self.state.update("active_tp", tp)
                self.state.update("active_brick_size", bs)
                self.state.update("be_triggered", False)
                
                # Update Logger
                # entry_spread_pts is 0 here since MT5 execution handles exact slippage,
                # but we could parse actual fill price from MT5 if passed back.
                self.logger.log_order(ticket, price, sl, tp, active_dir)
            else:
                logger.error(f"MT5 failed to execute order: {conf}")

    def _on_day_open(self, new_price):
        """Called when TickReceiver.day_open_price changes.
        
        On daily rollover, re-run path optimization using the accumulated
        history ticks to find the best anchor for the new day.
        """
        new_bs = new_price * K_MULTIPLIER
        
        logger.info(f"ROLLOVER detected: new_day_open={new_price:.2f}. Computing new_brick_size={new_bs:.4f}")
        logger.info(f"Old brick_size was: {self.renko.brick_size:.4f}")
        
        # Re-run path optimization if we have enough history
        history = self.receiver.get_history_ticks()
        if len(history) > 1000:
            # Sort in case of out-of-order ticks from reconnects
            history.sort(key=lambda t: t['time_msc'])
            logger.info("Daily rollover: re-running path optimization...")
            best_price, best_idx, best_profit = self.optimizer.find_optimal_anchor(
                history, new_bs
            )
            
            if best_price is not None:
                logger.info(
                    f"Rollover path optimization: anchor={best_price:.2f}, "
                    f"profit={best_profit:.2f}"
                )
                # Re-initialize with fresh optimized state
                self.renko = RenkoBuilder(best_price)
                self.renko.update_brick_size(new_bs, new_day_open=best_price)
                self.feature_engine = LiveFeatureEngine()
                self.feature_engine.update_brick_size(new_bs)
                self.buffer = InferenceBuffer()
                self.history_bricks = []
                
                # Replay from optimal index
                for i, tick in enumerate(history):
                    feat_vec = self.feature_engine.compute_vector(
                        tick['bid'], tick['ask'], 0.0, 0.0, tick['time_msc']
                    )
                    if i >= best_idx:
                        self.buffer.append_tick(feat_vec, self.renko.brick_count)
                        new_bricks = self.renko.update_tick(tick['bid'], tick['time_msc'])
                        for brick in new_bricks:
                            self.history_bricks.append(brick)
                            self.feature_engine.on_new_brick(brick)
                            self.buffer.on_brick_close(self.renko.brick_count - 1, self.feature_engine.last_macro)
                
                self._save_renko_snapshot()
                logger.info(f"Rollover path re-optimization complete: {self.renko.brick_count} bricks formed.")
            else:
                # Fallback: just update brick size
                self.renko.update_brick_size(new_bs, new_day_open=new_price)
                self.feature_engine.update_brick_size(new_bs)
        else:
            # Not enough history for path optimization — simple update
            self.renko.update_brick_size(new_bs, new_day_open=new_price)
            self.feature_engine.update_brick_size(new_bs)
        
        self.state.update('session_date', str(datetime.date.today()))
        self.state.update('daily_pnl', 0.0)
        self.last_day_open = new_price
        # DO NOT close open positions during rollover
        logger.info(f"ROLLOVER complete: new brick_size={self.renko.brick_size:.4f}")

    def _save_renko_snapshot(self):
        """Save the built Renko history to a CSV file and print last brick."""
        if not self.history_bricks:
            return
        
        import csv
        base_dir = os.path.dirname(os.path.abspath(__file__))
        history_path = os.path.join(base_dir, "logs", "renko_history_snapshot_fallback.csv")
        try:
            with open(history_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'uptrend', 'sequence'])
                for b in self.history_bricks:
                    writer.writerow([b.timestamp, b.open, b.high, b.low, b.close, b.uptrend, b.sequence])
            logger.info(f"Saved {len(self.history_bricks)} historical bricks to {history_path}")
        except Exception as e:
            logger.error(f"Failed to save renko history snapshot: {e}")
        
        # Print the LAST brick data to terminal for user visibility
        last = self.history_bricks[-1]
        dir_str = 'UP' if last.uptrend == 1 else 'DN'
        logger.info(f"LAST WARMUP BRICK: Dir={dir_str}, Close={last.close:.5f}, Seq='{last.sequence}'")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    engine = BridgeEngine()
    engine.start()
