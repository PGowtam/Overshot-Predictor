import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from datetime import datetime
from config.definitions import M_CATBOOST_BINARY, M_CATBOOST_MULTI, M_BINARY_SCALER, M_MULTI_SCALER
from utils.logger import logger
import os

class OutcomePredictor:
    def __init__(self):
        self.bin_model = None
        self.bin_scaler = None
        self.multi_model = None
        self.multi_scaler = None
        
    def load(self):
        try:
            # Binary Model
            if os.path.exists(M_CATBOOST_BINARY):
                self.bin_model = joblib.load(M_CATBOOST_BINARY)
                logger.info("Binary Outcome Model loaded.")
            else:
                logger.warning(f"Binary Model not found at {M_CATBOOST_BINARY}")

            if os.path.exists(M_BINARY_SCALER):
                self.bin_scaler = joblib.load(M_BINARY_SCALER)
                logger.info("Binary Scaler loaded.")
            else:
                logger.warning(f"Binary Scaler not found at {M_BINARY_SCALER}")

            # Multi Model
            if os.path.exists(M_CATBOOST_MULTI):
                try:
                    self.multi_model = joblib.load(M_CATBOOST_MULTI)
                except:
                     self.multi_model = CatBoostClassifier()
                     self.multi_model.load_model(M_CATBOOST_MULTI)
                logger.info("Multi-class Outcome Model loaded.")
            else:
                logger.warning(f"Multi Model not found at {M_CATBOOST_MULTI}")

            if os.path.exists(M_MULTI_SCALER):
                self.multi_scaler = joblib.load(M_MULTI_SCALER)
                logger.info("Multi Scaler loaded.")
            else:
                logger.warning(f"Multi Scaler not found at {M_MULTI_SCALER}")
                
            return True
        except Exception as e:
            logger.error(f"Error loading outcome models: {e}")
            return False
            
    def _extract_predictor_features(self, brick, prev_bricks, indicators_dict):
        """
        Reconstructs the exact feature set used in training.
        
        Needed Features:
         ['hour', 'day_of_week', 'duration_log' (BinaryOnly),
          'seq_len', 'seq_ones_ratio', 'uptrend_float',
          'brick_size',
          'RSI_14', 'MFI_14', 'ATR_14', 'BB_WIDTH', 'DIST_SMA50', 
          'RETURN_1M', 'RETURN_15M', 'RETURN_60M', 'RVOL_20',
          'brick_size_lag_1', 'uptrend_lag_1',
          'brick_size_lag_2', 'uptrend_lag_2']
        """
        # Parse timestamp
        # brick is namedtuple or object with timestamp
        # Assuming brick.timestamp is datetime or int ms
        if hasattr(brick, 'timestamp') and isinstance(brick.timestamp, (int, float, np.integer, np.floating)):
           dt = pd.to_datetime(brick.timestamp, unit='ms')
        else:
           # CRITICAL: Feature extraction requires valid server timestamp for 'Hour' feature.
           # Fallback to local time is dangerous due to timezone shift.
           raise ValueError(f"Feature Extraction Error: Brick missing valid timestamp. Got: {getattr(brick, 'timestamp', 'None')}")

        hour = dt.hour
        day = dt.dayofweek
        
        # Duration Log
        # We need prev_brick timestamp
        duration_log = 0.0
        if prev_bricks and len(prev_bricks) > 0:
            prev = prev_bricks[-1]
            if hasattr(prev, 'timestamp'):
                ms_diff = brick.timestamp - prev.timestamp
                duration_sec = ms_diff / 1000.0
                duration_log = np.log1p(duration_sec)
        
        # Sequence
        # Check if brick has 'sequence' field (added in RenkoBuilder)
        if hasattr(brick, 'sequence') and brick.sequence:
            seq = brick.sequence
        else:
            seq = ""
            
        seq_len = len(seq)
        if seq_len > 0:
            seq_ones = seq.count('1') / seq_len
        else:
            seq_ones = 0.0
        
        uptrend_float = 1.0 if brick.uptrend else 0.0
        
        # Lags
        # Need history access
        bs_1 = 0.0
        up_1 = 0.0
        bs_2 = 0.0
        up_2 = 0.0
        
        if prev_bricks:
            if len(prev_bricks) >= 1:
                p1 = prev_bricks[-1]
                bs_1 = abs(p1.close - p1.open) # approx brick size
                up_1 = 1.0 if p1.uptrend else 0.0
            if len(prev_bricks) >= 2:
                p2 = prev_bricks[-2]
                bs_2 = abs(p2.close - p2.open)
                up_2 = 1.0 if p2.uptrend else 0.0
                
        # Indicators
        # indicators_dict should provide these
        rsi = indicators_dict.get('RSI_14', 50.0)
        mfi = indicators_dict.get('MFI_14', 50.0)
        atr = indicators_dict.get('ATR_14', 0.0)
        bb_w = indicators_dict.get('BB_WIDTH', 0.0)
        dist_sma = indicators_dict.get('DIST_SMA50', 0.0)
        ret_1 = indicators_dict.get('RETURN_1M', 0.0)
        ret_15 = indicators_dict.get('RETURN_15M', 0.0)
        ret_60 = indicators_dict.get('RETURN_60M', 0.0)
        rvol = indicators_dict.get('RVOL_20', 1.0)
        
        # Determine Brick Size (Current)
        bs = abs(brick.close - brick.open)
        
        # Construct Dictionary
        feats = {
            'hour': hour,
            'day_of_week': day,
            'duration_log': duration_log,
            'seq_len': seq_len,
            'seq_ones_ratio': seq_ones,
            'uptrend_float': uptrend_float,
            'brick_size': bs,
            'RSI_14': rsi,
            'MFI_14': mfi,
            'ATR_14': atr,
            'BB_WIDTH': bb_w,
            'DIST_SMA50': dist_sma,
            'RETURN_1M': ret_1,
            'RETURN_15M': ret_15,
            'RETURN_60M': ret_60,
            'RVOL_20': rvol,
            'brick_size_lag_1': bs_1,
            'uptrend_lag_1': up_1,
            'brick_size_lag_2': bs_2,
            'uptrend_lag_2': up_2
        }
        return feats

    def predict(self, brick, prev_bricks, indicators_dict):
        """
        Full prediction pipeline: Extract -> Scale -> Predict
        """
        # If models missing, return safe defaults
        defaults = {'prob_bin_win': 0.5, 'prob_bin_loss': 0.5, 'prob_multi_win': 0.33, 'prob_multi_loss': 0.33}
        
        if not self.bin_model and not self.multi_model:
            return defaults
            
        # 1. Extract raw feature dict
        raw_feats = self._extract_predictor_features(brick, prev_bricks, indicators_dict)
        
        # 2. Prepare DataFrames (Single Row)
        # Binary needs 'duration_log'
        cols_binary = [
            'hour', 'day_of_week', 'duration_log',
            'seq_len', 'seq_ones_ratio', 'uptrend_float',
            'brick_size',
            'RSI_14', 'MFI_14', 'ATR_14', 'BB_WIDTH', 'DIST_SMA50', 
            'RETURN_1M', 'RETURN_15M', 'RETURN_60M', 'RVOL_20',
            'brick_size_lag_1', 'uptrend_lag_1',
            'brick_size_lag_2', 'uptrend_lag_2'
        ]
        
        # Multi needs NO 'duration_log'
        cols_multi = [c for c in cols_binary if c != 'duration_log']
        
        # 3. Predict Binary
        prob_bin_win = 0.5
        prob_bin_loss = 0.5
        
        if self.bin_model:
            df_bin = pd.DataFrame([raw_feats])[cols_binary]
            if self.bin_scaler:
                X_bin = pd.DataFrame(self.bin_scaler.transform(df_bin), columns=df_bin.columns)
            else:
                X_bin = df_bin
                
            try:
                # [Loss, Win]
                probs = self.bin_model.predict_proba(X_bin)[0]
                prob_bin_loss = probs[0]
                prob_bin_win = probs[1]
            except Exception as e:
                logger.error(f"Binary Predict Error: {e}")
                
        # 4. Predict Multi
        prob_multi_win = 0.33
        prob_multi_loss = 0.33
        
        if self.multi_model:
            df_multi = pd.DataFrame([raw_feats])[cols_multi]
            if self.multi_scaler:
                X_multi = pd.DataFrame(self.multi_scaler.transform(df_multi), columns=df_multi.columns)
            else:
                X_multi = df_multi
                
            try:
                # [Loss, Win, BE] map from training?
                # Train_Advanced: {'WIN': 1, 'LOSS': 0, 'BE': 2}
                # Classes order in predict_proba depends on sorted targets: 0, 1, 2 = LOSS, WIN, BE
                probs = self.multi_model.predict_proba(X_multi)[0]
                prob_multi_loss = probs[0]
                prob_multi_win = probs[1]
                # BE is probs[2]
            except Exception as e:
                logger.error(f"Multi Predict Error: {e}")

        return {
            'prob_bin_win': prob_bin_win, 
            'prob_bin_loss': prob_bin_loss,
            'prob_multi_win': prob_multi_win,
            'prob_multi_loss': prob_multi_loss
        } 
