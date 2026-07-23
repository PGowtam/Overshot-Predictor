import json
import os
import pickle
from pathlib import Path
from BrickOfTicks_Trader.config.settings import STATE_FILE
from BrickOfTicks_Trader.config.settings import LOGS_DIR
from BrickOfTicks_Trader.utils.logger import logger

class StateManager:
    """Manages the bot's persistent state for crash recovery."""
    def __init__(self, filepath=None):
        self.filepath = Path(filepath or STATE_FILE)
        self.state = {
            "last_tick_msc": 0,
            "daily_pnl": 0.0,
            "active_ticket": 0,
            "brick_count": 0,
            "current_day": None
        }
        self.load()

    def load(self):
        """Load state from JSON file."""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.state.update(data)
                logger.info(f"State loaded from {self.filepath}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        else:
            logger.info("No existing state file found. Starting fresh.")
            self.save()

    def save(self):
        """Save current state to JSON file."""
        try:
            # Ensure directory exists
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, 'w') as f:
                json.dump(self.state, f, indent=2)
            # logger.debug(f"State saved to {self.filepath}")
            pass
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def update(self, key, value):
        """Update a specific key in the state and save."""
        self.state[key] = value
        self.save()

    def get(self, key, default=None):
        """Get a value from the state."""
        return self.state.get(key, default)

    def save_internal_state(self, features, buffer, renko):
        """Phase 12: Serialize the core ML arrays and deques to a fast binary pickle."""
        pkl_path = Path(LOGS_DIR) / "internal_state.pkl"
        internal_data = {
            "renko": {
                "current_price": renko.current_price,
                "uptrend": renko.uptrend,
                "sequence": renko.sequence,
                "brick_size": renko.brick_size,
                "history_len": len(renko.history) # We don't save full history to save space, but we could
            },
            "features": {
                "z_ofi": list(features.z_ofi.deque),
                "z_depth": list(features.z_depth.deque),
                "z_susc": list(features.z_susc.deque),
                "z_vel": list(features.z_vel.deque),
                "z_spread": list(features.z_spread.deque),
                "prev_bid": features.prev_bid,
                "prev_ask": features.prev_ask,
                "prev_mid": features.prev_mid,
                "prev_time_ms": features.prev_time_ms,
                "current_brick_open": features.current_brick_open,
                "current_brick_size": features.current_brick_size,
                "current_brick_id": features.current_brick_id
            },
            "buffer": {
                "micro_buffer": list(buffer.micro_buffer),
                "macro_history": list(buffer.macro_history),
                "snapshots": list(buffer.snapshots),
                "brick_size_history": buffer.brick_size_history,
                "current_brick_id": buffer.current_brick_id
            }
        }
        try:
            with open(pkl_path, 'wb') as f:
                pickle.dump(internal_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to save internal state subset: {e}")

    def load_internal_state(self, features, buffer, renko) -> bool:
        """Phase 12: Load and apply the internal ML state to the instances."""
        pkl_path = Path(LOGS_DIR) / "internal_state.pkl"
        if not pkl_path.exists():
            return False
            
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                
            # Restore Renko
            rd = data["renko"]
            renko.current_price = rd["current_price"]
            renko.uptrend = rd["uptrend"]
            renko.sequence = rd["sequence"]
            renko.brick_size = rd["brick_size"]
            renko.history = rd.get("history", [])
            
            # Restore Features
            fd = data["features"]
            features.z_ofi.deque.extend(fd["z_ofi"])
            features.z_depth.deque.extend(fd["z_depth"])
            features.z_susc.deque.extend(fd["z_susc"])
            features.z_vel.deque.extend(fd["z_vel"])
            features.z_spread.deque.extend(fd["z_spread"])
            
            # Need to recalculate Welford means incrementally or just let them act as if full
            for z, name in zip(
                [features.z_ofi, features.z_depth, features.z_susc, features.z_vel, features.z_spread],
                ["z_ofi", "z_depth", "z_susc", "z_vel", "z_spread"]
            ):
                # Helper calculation to prime M2 and mean from the loaded deque state
                # Wait, our ZScore handles 'warmup' which computes mean/M2 if appending. 
                # Since we bypass append and directly extend deque, we must manually prime it.
                if len(z.deque) > 0:
                    arr = list(z.deque)
                    N = len(arr)
                    z.mean = sum(arr) / N
                    z.M2 = sum((x - z.mean) ** 2 for x in arr)
            
            features.prev_bid = fd["prev_bid"]
            features.prev_ask = fd["prev_ask"]
            features.prev_mid = fd["prev_mid"]
            features.prev_time_ms = fd.get("prev_time_ms")
            features.current_brick_open = fd["current_brick_open"]
            features.current_brick_size = fd["current_brick_size"]
            features.current_brick_id = fd["current_brick_id"]
            
            # Restore Buffer
            bd = data["buffer"]
            buffer.micro_buffer.extend(bd["micro_buffer"])
            buffer.macro_history.extend(bd["macro_history"])
            buffer.snapshots.extend(bd["snapshots"])
            buffer.brick_size_history = bd["brick_size_history"]
            buffer.current_brick_id = bd["current_brick_id"]
            
            logger.info("Internal ML state perfectly restored from pickle.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load internal state subset: {e}")
            return False

# Global State Manager instance
state = StateManager()
