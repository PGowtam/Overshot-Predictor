"""
MTPATSC Trader — State Manager
================================
Manages atomic persistence of the bridge state for crash recovery.
"""

import os
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_STATE = {
    "schema_version":   3,
    "last_tick_msc":    0,
    "active_ticket":    0,         # 0 = no position
    "active_direction": 0,         # 1=BUY, -1=SELL
    "active_entry":     0.0,
    "active_sl":        0.0,
    "active_tp":        0.0,
    "active_brick_size":0.0,       # Brick size at time of entry
    "active_setup_type":"",        # "T1", "T2", "T3", "T4"
    "active_rr":        0.0,       # Risk:reward ratio of the trade
    "be_triggered":     False,
    "daily_pnl":        0.0,       # Points, reset on rollover
    "brick_count":      0,
    "session_date":     "",        # YYYY-MM-DD broker date
    "warmup_done":      False,
    "degraded_mode":    False
}

class StateManager:
    """
    Manages atomic persistence of the bridge state.
    """
    def __init__(self, filepath="logs/state.json"):
        self.filepath = filepath
        self.tmp_filepath = filepath + ".tmp"
        self._state = DEFAULT_STATE.copy()

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)

    def load(self):
        """
        Load state from disk. Merges with DEFAULT_STATE to handle schema upgrades safely.
        """
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    disk_state = json.load(f)

                # Merge logic: preserve defaults for missing keys
                for k, v in disk_state.items():
                    if k in self._state:
                        self._state[k] = v

                logger.info(f"Loaded existing state from {self.filepath}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}. Using DEFAULT_STATE.")
        else:
            logger.info("No existing state found. Initializing new state.")
            self.save()

    def reset(self):
        """
        Reset state to defaults and persist. Called on every fresh run
        to prevent stale data (e.g. tickets from a different instrument)
        from carrying over.
        """
        self._state = DEFAULT_STATE.copy()
        self.save()
        logger.info("State reset to defaults for fresh run.")

    def save(self):
        """
        Atomically save state to disk to prevent corruption on crash.
        """
        try:
            with open(self.tmp_filepath, 'w') as f:
                json.dump(self._state, f, indent=4)
            # Atomic rename replaces the old file reliably
            os.replace(self.tmp_filepath, self.filepath)
        except Exception as e:
            logger.error(f"Atomic save failed: {e}")

    def update(self, key: str, value: Any):
        """
        Update a field and immediately persist.
        """
        if key not in self._state:
            logger.warning(f"Attempting to set unknown state key: {key}")

        self._state[key] = value
        self.save()

    def get(self, key: str, default=None) -> Any:
        return self._state.get(key, default)
