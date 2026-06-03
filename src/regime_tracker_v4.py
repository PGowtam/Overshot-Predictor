"""
Phase 1: Regime Distribution Engine
===================================
Tracks rolling 5-day market history to convert absolute signals into 
dynamic relative percentiles.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class RegimeTrackerV4:
    def __init__(self, lookback_days=5):
        self.lookback_days = lookback_days
        self.histories = {
            'spread_current': np.array([]),
            'abs_ofi_peak': np.array([]),
            'vel_peak': np.array([]),
            'wick_ratio': np.array([]),
            'absorption_index': np.array([])
        }
        
    def refresh(self, current_day, historical_df):
        """
        Loads the previous N days of history and pre-sorts for lightning fast
        percentile lookups during live ticks or backtesting.
        historical_df must contain: 'utc_day' and the feature columns.
        """
        start_day = current_day - timedelta(days=self.lookback_days)
        mask = (historical_df['utc_day'] >= start_day) & (historical_df['utc_day'] < current_day)
        window_df = historical_df[mask]
        
        for k in self.histories.keys():
            if k in window_df.columns:
                vals = window_df[k].dropna().values
                self.histories[k] = np.sort(vals)
                
    def is_ready(self, min_samples=100, required_features=None):
        """
        Check if tracker has enough data. If required_features is specified,
        only those features are checked. Otherwise checks all non-empty histories.
        """
        if required_features:
            return all(
                len(self.histories.get(f, [])) >= min_samples 
                for f in required_features
            )
        # Default: check only features that have been populated
        populated = {k: v for k, v in self.histories.items() if len(v) > 0}
        if not populated:
            return False
        return all(len(v) >= min_samples for v in populated.values())
        
    def get_percentile(self, feature_name, value):
        hist = self.histories.get(feature_name)
        if hist is None or len(hist) == 0:
            return np.nan
            
        # np.searchsorted is O(log N), perfect for high-frequency live pricing
        idx = np.searchsorted(hist, value, side='right')
        return (idx / len(hist)) * 100.0
        
    def compute_all_percentiles(self, features_dict):
        pcts = {}
        for k, v in features_dict.items():
            if k in self.histories:
                pcts[f"{k}_pct"] = self.get_percentile(k, v)
        return pcts
