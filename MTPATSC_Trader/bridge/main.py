"""
MTPATSC Trader — Bridge Engine (Main Orchestrator)
====================================================
Central orchestrator for the MTPATSC live trading system.
Connects MQL5 TCP sockets to the MTPATSC 5-class setup classifier.

Architecture:
  Tick → Renko (with intra-tick collection) → Feature Engine → Predictor → Trade Execution

Key differences from BrickOfTicks_Trader:
  - K_MULTIPLIER = 0.00118 (not 0.00295)
  - Features: ANCS/Candle/Momentum (not 9D z-score tick vectors)
  - Model: Single 5-class softmax (not 3-fold ensemble)
  - Trades: T1-T4 setup types with different R:R profiles
"""

import time
import queue
import logging
import sys
import os
import datetime

from bridge.tick_receiver import TickReceiver
from bridge.command_sender import CommandSender
from bridge.renko import RenkoBuilder, K_MULTIPLIER
from bridge.mtpatsc_feature_engine import MTPatscFeatureEngine
from bridge.mtpatsc_predictor import MTPatscPredictor
from bridge.state import StateManager
from bridge.risk import RiskManager
from bridge.trade_logger import TradeLogger
from bridge.path_optimizer import PathOptimizer

logger = logging.getLogger(__name__)


class BridgeEngine:
    """
    The central orchestrator for the MTPATSC Socket Bridge.
    Connects the MQL5 TCP sockets to the MTPATSC deep learning pipeline,
    managing state, risk, latency profiling, and multi-setup trade execution.
    """

    def __init__(self, tick_port=9000, cmd_port=9001):
        logger.info("Initializing MTPATSC BridgeEngine...")
        self.receiver = TickReceiver(port=tick_port)
        self.sender = CommandSender(confirm_queue=self.receiver.confirm_queue, port=cmd_port)

        # Initialize with dummy price (1.0). Re-initialized on DAYOPEN.
        self.renko = RenkoBuilder(1.0)

        # MTPATSC Feature Engine (replaces old feature_engine + buffer)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scaler_path = os.path.join(base_dir, "models", "scalar_scaler.pkl")
        self.feature_engine = MTPatscFeatureEngine(scaler_path=scaler_path)

        # MTPATSC Predictor (replaces old ensemble)
        self.predictor = MTPatscPredictor()

        self.state = StateManager()
        self.state.reset()

        self.risk = RiskManager()
        self.logger = TradeLogger()
        self.optimizer = PathOptimizer()

        self.last_tick_time = time.time()
        self.degraded_mode = False
        self.last_day_open = None

        self.consecutive_timeouts = 0
        self.reconnect_attempts = 0
        self.history_bricks = []

        # Track last live tick for spread computation
        self.last_bid = 0.0
        self.last_ask = 0.0

    def start(self):
        """
        Boots the system, loads model, waits for MT5 connection and DAYOPEN.
        """
        logger.info("Starting MTPATSC BridgeEngine...")

        # 1. Load MTPATSC model and thresholds
        self.predictor.load()

        # 2. Start Receiver (port 9000)
        self.receiver.start()

        # 3. Start Command Sender (port 9001 - non-blocking background connect)
        try:
            self.sender.connect()
        except Exception as e:
            logger.warning(f"Could not connect CommandSender on startup: {e}. Will reconnect later.")

        # 4. Wait for DAYOPEN
        logger.info("Waiting for DAYOPEN from MT5 (max 30s)...")
        start_wait = time.time()
        while self.receiver.day_open_price is None:
            if time.time() - start_wait > 30.0:
                logger.critical("DAYOPEN timeout. No MT5 connection.")
                sys.exit(1)
            time.sleep(0.1)

        self.last_day_open = self.receiver.day_open_price
        logger.info(f"Day open received: {self.last_day_open}")

        # 5. Run Warmup
        self._warmup()

        # 6. Enter Main Loop
        self._run_loop()

    def _warmup(self):
        """
        Path-Optimized Warmup:
        1. Wait for multi-day history ticks from EA
        2. Run PathOptimizer to find the best Renko starting anchor
        3. Create fresh RenkoBuilder at the optimal anchor
        4. Replay all ticks through the Renko engine
        5. Compute features at each brick close to populate history buffer
        6. Check integrity gate (>= 5 bricks for MTPATSC history tensor)
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

        # Sort & Deduplicate
        history.sort(key=lambda t: t['time_msc'])
        deduped = [history[0]]
        for i in range(1, len(history)):
            if history[i]['time_msc'] != history[i - 1]['time_msc']:
                deduped.append(history[i])
        if len(deduped) < len(history):
            logger.info(f"Deduplication: {len(history)} → {len(deduped)} ticks")
            history = deduped
            self.receiver.set_history_ticks(history)

        # Path Optimization
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

        # Create fresh pipeline at optimal anchor
        self.renko = RenkoBuilder(best_price)
        self.renko.update_brick_size(brick_size, new_day_open=best_price)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scaler_path = os.path.join(base_dir, "models", "scalar_scaler.pkl")
        self.feature_engine = MTPatscFeatureEngine(scaler_path=scaler_path)
        self.history_bricks = []

        # Replay ticks from best_idx onwards through Renko
        logger.info(f"Replaying {len(history) - best_idx} ticks from idx={best_idx}...")
        for i in range(best_idx, len(history)):
            tick = history[i]
            bid = tick['bid']
            ask = tick.get('ask', bid)
            time_msc = tick['time_msc']

            new_bricks = self.renko.update_tick(bid, time_msc, ask=ask)
            for brick in new_bricks:
                self.history_bricks.append(brick)
                # Compute features to populate rolling history buffer
                self.feature_engine.on_brick_close(brick)

        # Integrity Gate
        bricks = self.renko.brick_count
        history_ready = self.feature_engine.brick_count >= 5

        if bricks >= 5 and history_ready:
            logger.info(f"Warmup PASSED: {bricks} bricks formed, feature history populated.")
            self.state.update("warmup_done", True)
        else:
            logger.warning(
                f"Warmup gate NOT YET MET: {bricks} bricks (need ≥5). "
                f"Waiting for live ticks to fill buffers."
            )
            self.state.update("warmup_done", False)

        # Snapshot
        self._save_renko_snapshot()

    def _run_loop(self):
        """
        The continuous live streaming loop.
        """
        logger.info("MTPATSC BridgeEngine is LIVE and waiting for streaming ticks.")
        consecutive_ticks = 0
        window_start = time.time()

        while True:
            try:
                tick = self.receiver.tick_queue.get(timeout=60.0)
                self.consecutive_timeouts = 0

                # Track last bid/ask for spread computation
                self.last_bid = tick['bid']
                self.last_ask = tick['ask']

                # Detect Rollover
                if self.receiver.day_open_price is not None and self.receiver.day_open_price != self.last_day_open:
                    if self.last_day_open is not None:
                        self._on_day_open(self.receiver.day_open_price)
                    self.last_day_open = self.receiver.day_open_price

                # Degraded Mode logic
                if self.degraded_mode:
                    if time.time() - window_start > 5.0:
                        window_start = time.time()
                        consecutive_ticks = 0
                    consecutive_ticks += 1
                    if consecutive_ticks >= 3:
                        self._exit_degraded_mode()

                self.last_tick_time = time.time()
                self._process_tick(tick, is_warmup=not self.state.get("warmup_done"))

                # Live warmup satisfaction
                if not self.state.get("warmup_done"):
                    if self.feature_engine.brick_count >= 5:
                        logger.info("Live tick stream fulfilled warmup gate. System ARMED.")
                        self.state.update("warmup_done", True)

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
        logger.critical("3 consecutive 60s timeouts! Entering DEGRADED MODE.")
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
            logger.critical("Max reconnect attempts (10) reached. Exiting.")
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
        Core sequence: Tick → Renko (with intra-tick accumulation) → Feature → Predict → Trade
        """
        t0 = time.perf_counter()

        bid = tick['bid']
        ask = tick['ask']
        time_msc = tick['time_msc']

        new_bricks = self.renko.update_tick(bid, time_msc, ask=ask)

        for brick in new_bricks:
            if is_warmup:
                self.history_bricks.append(brick)
                self.feature_engine.on_brick_close(brick)
            else:
                dir_str = 'UP' if brick.uptrend == 1 else 'DN'
                logger.info(f"🧱 NEW LIVE BRICK: Dir={dir_str}, Close={brick.close:.2f}, "
                            f"BS={brick.brick_size:.4f}, Ticks={len(brick.intra_ticks)}")

                # Compute features
                tensors = self.feature_engine.on_brick_close(brick)

                if tensors is not None:
                    t1 = time.perf_counter()
                    self._on_signal(brick, tensors)
                    t2 = time.perf_counter()
                    inference_ms = (t2 - t1) * 1000
                    if inference_ms > 150:
                        logger.warning(f"SLOW INFERENCE: {inference_ms:.0f}ms (target <80ms)")
                else:
                    logger.info("Feature engine returned None (history buffer filling).")

        # Break-even check on every tick
        # NOTE: Disabled for T1 (1:1 RR) — kept for future T3/T4 setups
        if self.state.get('active_ticket') and not self.state.get('be_triggered'):
            setup_type = self.state.get('active_setup_type', 'T1')
            if setup_type != 'T1':  # Only BE for non-T1 setups
                self._check_be(tick)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > 30:
            logger.warning(f"SLOW TICK PROCESSING: {elapsed_ms:.0f}ms")

    def _check_be(self, tick):
        if self.risk.check_be_trigger(tick, self.state):
            entry_price = self.state.get('active_entry')
            direction = self.state.get('active_direction')
            ticket = self.state.get('active_ticket')

            # Offset SL to cover spread (0.5 pts past entry)
            new_sl = entry_price + 0.5 if direction == 1 else entry_price - 0.5

            logger.info(f"Triggering Break-Even for ticket {ticket} to SL: {new_sl:.2f}")
            conf = self.sender.modify_sl(ticket, new_sl)

            if conf and conf['status'] == 'OK':
                self.state.update('be_triggered', True)
                self.state.update('active_sl', new_sl)
                logger.info(f"BREAK-EVEN success: SL moved to {new_sl:.2f}")
            else:
                logger.error(f"BREAK-EVEN failed for ticket {ticket}.")

    def _on_signal(self, brick, tensors):
        """
        Run MTPATSC prediction and fire trades if all conditions are met.
        """
        # Run prediction
        result = self.predictor.predict(tensors, brick.uptrend)

        # Log prediction
        probs = result['probs']
        logger.info(f"📊 MTPATSC: P(T0)={probs[0]:.3f} P(T1)={probs[1]:.3f} "
                     f"P(T2)={probs[2]:.3f} P(T3)={probs[3]:.3f} P(T4)={probs[4]:.3f} "
                     f"→ {result['reason']}")

        # Log signal to trade logger
        self.logger.log_signal(brick.timestamp, brick.uptrend, result)

        if result['action'] != 1:
            return

        # Risk checks
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

        # Spread check: skip if spread > 10% of brick size
        spread = self.last_ask - self.last_bid
        if not self.risk.check_spread(spread, self.renko.brick_size):
            logger.warning(f"Signal blocked: Spread {spread:.2f} exceeds 10% of brick {self.renko.brick_size:.4f}")
            return

        # Execution geometry
        setup_type = result['setup_type']
        trade_direction = result['direction']
        rr = result['rr']
        bs = self.renko.brick_size
        close_price = brick.close

        if setup_type == 1:  # T1 Continuation — market order at close_price ± spread
            if trade_direction == 1:  # BUY
                entry = close_price + spread
                sl = entry - bs
                tp = entry + bs
            else:  # SELL
                entry = close_price - spread
                sl = entry + bs
                tp = entry - bs

        elif setup_type == 3:  # T3 Reversal — market order against brick direction
            if trade_direction == 1:  # BUY (brick was DOWN)
                entry = self.last_ask
                tp = entry + 2.0 * bs
                sl = entry - bs
            else:  # SELL (brick was UP)
                entry = self.last_bid
                tp = entry - 2.0 * bs
                sl = entry + bs

        elif setup_type == 4:  # T4 Deep Reversal
            if trade_direction == 1:  # BUY (brick was DOWN)
                entry = self.last_ask
                tp = entry + 3.0 * bs
                sl = entry - bs
            else:  # SELL (brick was UP)
                entry = self.last_bid
                tp = entry - 3.0 * bs
                sl = entry + bs

        else:
            logger.warning(f"Unhandled setup_type {setup_type}. Skipping.")
            return

        # Fire order
        is_buy = (trade_direction == 1)
        logger.info(f"🔥 FIRING T{setup_type} {'BUY' if is_buy else 'SELL'}: "
                     f"entry={entry:.2f}, sl={sl:.2f}, tp={tp:.2f}, RR=1:{rr:.0f}")

        if is_buy:
            conf = self.sender.buy(entry, sl, tp, volume=0.01)
        else:
            conf = self.sender.sell(entry, sl, tp, volume=0.01)

        if conf and conf.get('status') == 'OK':
            ticket = conf.get('ticket', 0)
            self.state.update("active_ticket", ticket)
            self.state.update("active_direction", trade_direction)
            self.state.update("active_entry", entry)
            self.state.update("active_sl", sl)
            self.state.update("active_tp", tp)
            self.state.update("active_brick_size", bs)
            self.state.update("active_setup_type", f"T{setup_type}")
            self.state.update("active_rr", rr)
            self.state.update("be_triggered", False)

            self.logger.log_order(ticket, entry, sl, tp, trade_direction)
            logger.info(f"✅ T{setup_type} order confirmed: ticket={ticket}")
        else:
            logger.error(f"❌ MT5 failed to execute T{setup_type} order: {conf}")

    def _on_day_open(self, new_price):
        """Called when TickReceiver.day_open_price changes (daily rollover)."""
        new_bs = new_price * K_MULTIPLIER

        logger.info(f"ROLLOVER detected: new_day_open={new_price:.2f}, new_brick_size={new_bs:.4f}")

        history = self.receiver.get_history_ticks()
        if len(history) > 1000:
            history.sort(key=lambda t: t['time_msc'])
            logger.info("Daily rollover: re-running path optimization...")
            best_price, best_idx, best_profit = self.optimizer.find_optimal_anchor(
                history, new_bs
            )

            if best_price is not None:
                logger.info(f"Rollover path optimization: anchor={best_price:.2f}, profit={best_profit:.2f}")
                self.renko = RenkoBuilder(best_price)
                self.renko.update_brick_size(new_bs, new_day_open=best_price)
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                scaler_path = os.path.join(base_dir, "models", "scalar_scaler.pkl")
                self.feature_engine = MTPatscFeatureEngine(scaler_path=scaler_path)
                self.history_bricks = []

                for i in range(best_idx, len(history)):
                    tick = history[i]
                    new_bricks = self.renko.update_tick(
                        tick['bid'], tick['time_msc'], ask=tick.get('ask', tick['bid']))
                    for brick in new_bricks:
                        self.history_bricks.append(brick)
                        self.feature_engine.on_brick_close(brick)

                self._save_renko_snapshot()
                logger.info(f"Rollover complete: {self.renko.brick_count} bricks formed.")
            else:
                self.renko.update_brick_size(new_bs, new_day_open=new_price)
        else:
            self.renko.update_brick_size(new_bs, new_day_open=new_price)

        self.state.update('session_date', str(datetime.date.today()))
        self.state.update('daily_pnl', 0.0)
        self.last_day_open = new_price
        logger.info(f"ROLLOVER complete: new brick_size={self.renko.brick_size:.4f}")

    def _save_renko_snapshot(self):
        """Save the built Renko history to a CSV file."""
        if not self.history_bricks:
            return

        import csv
        base_dir = os.path.dirname(os.path.abspath(__file__))
        history_path = os.path.join(base_dir, "logs", "renko_history_snapshot.csv")
        try:
            with open(history_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'uptrend', 'sequence'])
                for b in self.history_bricks:
                    writer.writerow([b.timestamp, b.open, b.high, b.low, b.close, b.uptrend, b.sequence])
            logger.info(f"Saved {len(self.history_bricks)} historical bricks to {history_path}")
        except Exception as e:
            logger.error(f"Failed to save renko history snapshot: {e}")

        last = self.history_bricks[-1]
        dir_str = 'UP' if last.uptrend == 1 else 'DN'
        logger.info(f"LAST WARMUP BRICK: Dir={dir_str}, Close={last.close:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    engine = BridgeEngine()
    engine.start()
