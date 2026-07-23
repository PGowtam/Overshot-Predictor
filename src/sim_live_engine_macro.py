import sys
import pandas as pd
import logging
from pathlib import Path
import time
from datetime import timedelta

BASE_DIR = Path(__file__).resolve().parent.parent
TRADER_DIR = BASE_DIR / "BrickOfTicks_Trader"
sys.path.insert(0, str(TRADER_DIR))

import bridge.renko
import bridge.path_optimizer
bridge.renko.K_MULTIPLIER = 0.00118
bridge.path_optimizer.K_MULTIPLIER = 0.00118

from bridge.renko import RenkoBuilder
from bridge.feature_engine import LiveFeatureEngine
from bridge.path_optimizer import PathOptimizer

sys.path.insert(0, str(BASE_DIR / "src"))
from regime_tracker_v4 import RegimeTrackerV4, RegimeTrackerBrickV4
from sim_live_engine_rr_sweep import SimTrade_RR_Broker

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Sequential Macro Simulator (Dual Testing: 2000-Brick vs 100-Day)")
    
    parquet_path = BASE_DIR / "Data" / "xauusd_ticks_2026.parquet"
    logger.info(f"Loading tick data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df = df.sort_values('time_msc').reset_index(drop=True)
    df['utc_day'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True).dt.date
    
    days = sorted(df['utc_day'].unique())
    logger.info(f"Loaded {len(df)} ticks across {len(days)} trading days.")
    
    tracker_brick = RegimeTrackerBrickV4(lookback_bricks=5000)
    tracker_day = RegimeTrackerV4(lookback_days=100)
    
    broker_brick = SimTrade_RR_Broker()
    broker_day = SimTrade_RR_Broker()
    
    optimizer = PathOptimizer()
    
    last_trade_time_brick = -1
    last_trade_time_day = -1
    global_warmup_records = []
    
    t0 = time.time()
    
    for i, day in enumerate(days):
        day_ticks = df[df['utc_day'] == day].to_dict('records')
        if len(day_ticks) < 100:
            continue
            
        # Refresh 100-day calendar tracker
        if global_warmup_records:
            df_hist = pd.DataFrame(global_warmup_records)
            tracker_day.refresh(day, df_hist)
            
        start_day = day - timedelta(days=7)
        lookback_mask = (df['utc_day'] >= start_day) & (df['utc_day'] < day)
        lookback_ticks = df[lookback_mask].to_dict('records')
        
        day_open = day_ticks[0]['bid']
        brick_size = day_open * 0.00118
        
        if len(lookback_ticks) > 1000:
            best_price, best_idx, _ = optimizer.find_optimal_anchor(lookback_ticks, brick_size)
            if best_price is None: best_price = day_open; best_idx = 0
        else:
            best_price = day_open; best_idx = 0
            
        renko = RenkoBuilder(best_price)
        renko.update_brick_size(brick_size, new_day_open=best_price)
        feature_engine = LiveFeatureEngine()
        feature_engine.update_brick_size(brick_size)
        
        ofi_peak = 0.0
        
        # Warmup Renko grids
        for j, tick in enumerate(lookback_ticks):
            feat = feature_engine.compute_vector(tick['bid'], tick['ask'], 0.0, 0.0, tick['time_msc'])
            if feat is not None:
                if abs(feat[0]) > abs(ofi_peak): ofi_peak = feat[0]
                
            if j >= best_idx:
                new_bricks = renko.update_tick(tick['bid'], tick['time_msc'])
                for brick in new_bricks:
                    feature_engine.on_new_brick(brick)
                    ofi_peak = 0.0
                    
        # Process Live Day Ticks
        for tick in day_ticks:
            bid, ask, t_msc = tick['bid'], tick['ask'], tick['time_msc']
            
            feat = feature_engine.compute_vector(bid, ask, 0.0, 0.0, t_msc)
            if feat is not None:
                if abs(feat[0]) > abs(ofi_peak): 
                    ofi_peak = feat[0]
                    
            if broker_brick.active_positions: broker_brick.check_ticks(bid, ask, t_msc)
            if broker_day.active_positions: broker_day.check_ticks(bid, ask, t_msc)
                
            new_bricks = renko.update_tick(bid, t_msc)
            for brick in new_bricks:
                feature_engine.on_new_brick(brick)
                
                spread_current = tick['ask'] - tick['bid']
                abs_ofi = abs(ofi_peak)
                
                # Append to global records for the 100-day tracker
                global_warmup_records.append({
                    'utc_day': day,
                    'spread_current': spread_current,
                    'abs_ofi_peak': abs_ofi
                })
                
                # BRICK SIGNAL (Evaluates across all 105 days after 5000 bricks fill up)
                if tracker_brick.is_ready(min_samples=5000):
                    sp_pct_b = tracker_brick.get_percentile('spread_current', spread_current)
                    op_pct_b = tracker_brick.get_percentile('abs_ofi_peak', abs_ofi)
                    
                    if spread_current >= 1.0 and sp_pct_b >= 99 and op_pct_b <= 3:
                        if brick.timestamp != last_trade_time_brick:
                            last_trade_time_brick = brick.timestamp
                            is_buy = (brick.uptrend == 1)
                            bs = brick.brick_size
                            for rr in [1.0, 2.0, 3.0]:
                                price = bid if is_buy else ask
                                sl = price + bs if is_buy else price - bs
                                tp = price - (bs * rr) if is_buy else price + (bs * rr)
                                direction = -1 if is_buy else 1
                                broker_brick.execute(f"{rr}R", direction, price, sl, tp, bs, brick.timestamp)
                                
                tracker_brick.add_brick(spread_current, abs_ofi)
                
                # DAY SIGNAL (Evaluates ONLY on the last 5 days once 100 days fill up)
                if i >= 100 and tracker_day.is_ready(min_samples=1000):
                    sp_pct_d = tracker_day.get_percentile('spread_current', spread_current)
                    op_pct_d = tracker_day.get_percentile('abs_ofi_peak', abs_ofi)
                    
                    if spread_current >= 1.0 and sp_pct_d >= 99 and op_pct_d <= 3:
                        if brick.timestamp != last_trade_time_day:
                            last_trade_time_day = brick.timestamp
                            is_buy = (brick.uptrend == 1)
                            bs = brick.brick_size
                            for rr in [1.0, 2.0, 3.0]:
                                price = bid if is_buy else ask
                                sl = price + bs if is_buy else price - bs
                                tp = price - (bs * rr) if is_buy else price + (bs * rr)
                                direction = -1 if is_buy else 1
                                broker_day.execute(f"{rr}R", direction, price, sl, tp, bs, brick.timestamp)
                                
                ofi_peak = 0.0
                
        if broker_brick.active_positions:
            broker_brick.force_close_all(day_ticks[-1]['bid'], day_ticks[-1]['ask'], day_ticks[-1]['time_msc'])
        if broker_day.active_positions:
            broker_day.force_close_all(day_ticks[-1]['bid'], day_ticks[-1]['ask'], day_ticks[-1]['time_msc'])
            
        if (i+1) % 20 == 0:
            logger.info(f"Simulated {i+1}/{len(days)} days...")
            
    # Cleanup memory
    keep_days = min(110, len(days))
    if keep_days > 0:
        global_warmup_records = [r for r in global_warmup_records if r['utc_day'] >= days[-keep_days]]
            
    logger.info(f"Simulation completed in {time.time()-t0:.2f}s")
    
    # Analyze Brick Tracker
    logger.info("\n===========================================")
    logger.info("   5000-BRICK LOOKBACK (ALL 105 DAYS)   ")
    logger.info("===========================================")
    trades_df_b = pd.DataFrame(broker_brick.trade_log)
    if not trades_df_b.empty:
        for rr in ["1.0R", "2.0R", "3.0R"]:
            strat_df = trades_df_b[trades_df_b['rr_target'] == rr].copy()
            if strat_df.empty: continue
            resolved = strat_df[strat_df['outcome'].isin(['WIN', 'LOSS'])]
            total = len(resolved)
            wins = sum(resolved['outcome'] == 'WIN')
            wr = (wins / total * 100) if total > 0 else 0
            pf = resolved[resolved['pnl_r'] > 0]['pnl_r'].sum() / (abs(resolved[resolved['pnl_r'] < 0]['pnl_r'].sum()) + 1e-8)
            logger.info(f"--- {rr} TARGET --- | Trades: {total} | WR: {wr:.2f}% | PF: {pf:.2f} | PnL: {strat_df['pnl_r'].sum():+.2f}R")
    
    # Analyze Day Tracker
    logger.info("\n===========================================")
    logger.info("     100-DAY LOOKBACK (LAST 5 DAYS)     ")
    logger.info("===========================================")
    trades_df_d = pd.DataFrame(broker_day.trade_log)
    if not trades_df_d.empty:
        for rr in ["1.0R", "2.0R", "3.0R"]:
            strat_df = trades_df_d[trades_df_d['rr_target'] == rr].copy()
            if strat_df.empty: continue
            resolved = strat_df[strat_df['outcome'].isin(['WIN', 'LOSS'])]
            total = len(resolved)
            wins = sum(resolved['outcome'] == 'WIN')
            wr = (wins / total * 100) if total > 0 else 0
            pf = resolved[resolved['pnl_r'] > 0]['pnl_r'].sum() / (abs(resolved[resolved['pnl_r'] < 0]['pnl_r'].sum()) + 1e-8)
            logger.info(f"--- {rr} TARGET --- | Trades: {total} | WR: {wr:.2f}% | PF: {pf:.2f} | PnL: {strat_df['pnl_r'].sum():+.2f}R")
    else:
        logger.info("No trades generated on the last 5 days.")

if __name__ == "__main__":
    main()
