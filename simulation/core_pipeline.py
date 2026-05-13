import numpy as np
import tensorflow as tf
from simulation.tick_engine import SyntheticTickEngine, REGIMES
from simulation.order_book import ExecutionSimulator
from simulation.feature_engine import LiveFeatureEngine, InferenceBuffer
import sys
import os
sys.path.append(os.getcwd())
from src.model import compile_model

class SyntheticRenkoBuilder:
    def __init__(self, brick_size: float = 1.0):
        self.brick_size = brick_size
        self.current_price = None
        self.uptrend = True

    def update_tick(self, bid: float, time_msc: int) -> list:
        if self.current_price is None:
            self.current_price = bid
            return []

        bricks = []
        while True:
            dist = bid - self.current_price

            if self.uptrend:
                if dist >= self.brick_size:
                    self.current_price += self.brick_size
                    bricks.append({"dir": 1, "close": self.current_price, "ts": time_msc})
                elif dist <= -(self.brick_size * 2):
                    self.uptrend = False
                    self.current_price -= (self.brick_size * 2)
                    bricks.append({"dir": -1, "close": self.current_price, "ts": time_msc})
                else:
                    break
            else:
                if dist <= -self.brick_size:
                    self.current_price -= self.brick_size
                    bricks.append({"dir": -1, "close": self.current_price, "ts": time_msc})
                elif dist >= (self.brick_size * 2):
                    self.uptrend = True
                    self.current_price += (self.brick_size * 2)
                    bricks.append({"dir": 1, "close": self.current_price, "ts": time_msc})
                else:
                    break
        return bricks

class MonteCarloStressTester:
    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations
        self.tick_engine = SyntheticTickEngine()
        self.executor = ExecutionSimulator(base_latency_ms=50)

        # Load the actual model
        print("Loading real Keras model for inference...")
        model_path = "BrickOfTicks_Trader/models/fold_1/model.keras"
        raw_model = tf.keras.models.load_model(model_path, compile=False)
        self.model = compile_model(raw_model)

        # Thresholds from config
        self.prob_win_th = 0.5
        self.pred_os_th = 1.6

    def run_regime(self, regime_name: str, num_ticks=10000) -> dict:
        regime = REGIMES[regime_name]
        ticks = list(self.tick_engine.stream_ticks(num_ticks, regime))

        renko = SyntheticRenkoBuilder(brick_size=1.0)
        features = LiveFeatureEngine()
        buffer = InferenceBuffer()

        trades_taken = 0
        market_wins = 0; market_losses = 0
        limit_fills = 0; limit_wins = 0; limit_losses = 0
        total_slippage = 0.0
        active_trade = None

        for idx, tick in enumerate(ticks):

            # --- Check active trade ---
            if active_trade is not None:
                if active_trade["dir"] == 1:
                    if tick["bid"] >= active_trade["tp"]: active_trade["status"] = "WIN"
                    elif tick["bid"] <= active_trade["sl"]: active_trade["status"] = "LOSS"
                else:
                    if tick["ask"] <= active_trade["tp"]: active_trade["status"] = "WIN"
                    elif tick["ask"] >= active_trade["sl"]: active_trade["status"] = "LOSS"

                if active_trade["status"] != "OPEN":
                    if active_trade["type"] == "MARKET":
                        if active_trade["status"] == "WIN": market_wins += 1
                        else: market_losses += 1
                    elif active_trade["type"] == "LIMIT":
                        if active_trade["status"] == "WIN": limit_wins += 1
                        else: limit_losses += 1
                    active_trade = None

            # --- Process Features ---
            vec = features.compute_vector(tick)
            buffer.push_tick(vec)

            # --- Process Bricks ---
            new_bricks = renko.update_tick(tick["bid"], tick["time_msc"])
            for brick in new_bricks:
                features.on_new_brick(brick)
                tensors = buffer.on_brick_close(brick)

                if tensors is not None and active_trade is None:
                    # REAL INFERENCE
                    micro, macro = tensors
                    preds = self.model.predict([micro, macro], verbose=0)
                    prob_win = preds[0][0][0]
                    pred_os = preds[1][0][0]

                    if prob_win >= self.prob_win_th and pred_os >= self.pred_os_th:
                        signal_price = brick["close"]
                        direction = brick["dir"]

                        exec_result = self.executor.simulate_market_order(
                            direction, brick["ts"], signal_price, ticks, idx, regime)
                        trades_taken += 1

                        if exec_result["slippage"] > 0.08:
                            limit_result = self.executor.simulate_limit_order(
                                direction, signal_price, ticks, idx)
                            if limit_result["status"] == "FILLED":
                                limit_fills += 1
                                active_trade = {
                                    "dir": direction, "entry": limit_result["actual_price"],
                                    "sl": limit_result["actual_price"] - direction,
                                    "tp": limit_result["actual_price"] + direction,
                                    "status": "OPEN", "type": "LIMIT"
                                }
                        else:
                            total_slippage += exec_result["slippage"]
                            active_trade = {
                                "dir": direction, "entry": exec_result["actual_price"],
                                "sl": exec_result["actual_price"] - direction,
                                "tp": exec_result["actual_price"] + direction,
                                "status": "OPEN", "type": "MARKET"
                            }

        return {
            "regime": regime_name, "trades": trades_taken,
            "market_wins": market_wins, "market_losses": market_losses,
            "limit_fills": limit_fills, "limit_wins": limit_wins, "limit_losses": limit_losses,
            "avg_slippage": total_slippage / (market_wins + market_losses) if (market_wins + market_losses) > 0 else 0
        }

if __name__ == '__main__':
    tester = MonteCarloStressTester()
    print("Testing core pipeline with real model...")
    res = tester.run_regime("NORMAL", num_ticks=2000)
    print(res)
