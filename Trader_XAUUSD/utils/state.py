import json
import os
from datetime import datetime
from config.definitions import LOGS_DIR

class StateManager:
    def __init__(self, filename="daily_state.json"):
        self.filepath = os.path.join(LOGS_DIR, filename)
        self.state = {
            "last_processed_time_msc": 0,
            "current_day": None,
            "daily_pnl": 0.0,
            "trades_today": 0,
            "active_ticket": 0,
            "active_entry_price": 0.0,
            "active_direction": 0,
            "pending_ticket": 0,
            "pending_be_level": 0.0,
            "pending_direction": 0,
            "optimization": {
                "brick_size": 0.0,
                "grid_offset": 0.0
            }
        }
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.state.update(data)
            except Exception as e:
                print(f"Error loading state: {e}")

    def save(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            print(f"Error saving state: {e}")

    def update(self, key, value):
        self.state[key] = value
        self.save()

    def get(self, key, default=None):
        return self.state.get(key, default)
    
    def reset_daily(self, current_date_str):
        """Reset daily counters using explicit Server Date."""
        self.state["daily_pnl"] = 0.0
        self.state["trades_today"] = 0
        self.state["current_day"] = current_date_str
        self.save()
