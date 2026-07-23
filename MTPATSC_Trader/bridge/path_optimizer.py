"""
BrickOfTicks Socket Bridge — Path Optimizer
============================================
Phase 13: Optimal Renko Anchor Discovery using Numba JIT.

Replicates the training pipeline's (create_renko_dynamic_ticks.py) optimization logic:
1. Given multi-day tick history, identify an "anchor day" (first available day)
2. Try candidate start prices across the anchor day's price range
3. For each candidate, build Renko bricks and simulate trades on the history window
4. Select the candidate that produced the highest historical profit

Constants (exact parity with training):
    K_MULTIPLIER = 0.00295  (brick_size = day_open * K)
    STEP_FACTOR  = 0.00295  (step_size  = anchor_open * STEP_FACTOR * 0.01)
    BE_TRIGGER   = 0.3125   (break-even at 5/16 of brick_size)
"""

import numpy as np
import logging
from numba import jit
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Constants (exact parity with training pipeline) ──────────────
K_MULTIPLIER = 0.00295
STEP_FACTOR  = 0.00295   # Matches current Renko config
BE_TRIGGER   = 0.3125


# ══════════════════════════════════════════════════════════════
# JIT-Compiled Hot Functions (matching training pipeline exactly)
# ══════════════════════════════════════════════════════════════

@jit(nopython=True)
def _build_renko_jit(timestamps, prices, start_index, start_price, brick_size, window_start_ts, stop_ts):
    """
    JIT-compiled Renko builder. Exact parity with generate_bricks_ticks_jit.
    Only emits bricks between window_start_ts and stop_ts.
    """
    n = len(timestamps)
    current_brick_price = start_price
    uptrend = 0  # 0=neutral, 1=up, -1=down

    out_dates = []
    out_closes = []
    out_uptrends = []

    for i in range(start_index, n):
        ts = timestamps[i]
        if ts >= stop_ts:
            break

        price = prices[i]

        if uptrend == 0:
            if price >= current_brick_price + brick_size:
                while price >= current_brick_price + brick_size:
                    current_brick_price += brick_size
                    if ts >= window_start_ts:
                        out_dates.append(ts)
                        out_closes.append(current_brick_price)
                        out_uptrends.append(1)
                uptrend = 1
            elif price <= current_brick_price - brick_size:
                while price <= current_brick_price - brick_size:
                    current_brick_price -= brick_size
                    if ts >= window_start_ts:
                        out_dates.append(ts)
                        out_closes.append(current_brick_price)
                        out_uptrends.append(-1)
                uptrend = -1
        elif uptrend == 1:
            if price >= current_brick_price + brick_size:
                while price >= current_brick_price + brick_size:
                    current_brick_price += brick_size
                    if ts >= window_start_ts:
                        out_dates.append(ts)
                        out_closes.append(current_brick_price)
                        out_uptrends.append(1)
            elif price <= current_brick_price - 2 * brick_size:
                current_brick_price -= 2 * brick_size
                if ts >= window_start_ts:
                    out_dates.append(ts)
                    out_closes.append(current_brick_price)
                    out_uptrends.append(-1)
                uptrend = -1
                while price <= current_brick_price - brick_size:
                    current_brick_price -= brick_size
                    if ts >= window_start_ts:
                        out_dates.append(ts)
                        out_closes.append(current_brick_price)
                        out_uptrends.append(-1)
        else:  # uptrend == -1
            if price <= current_brick_price - brick_size:
                while price <= current_brick_price - brick_size:
                    current_brick_price -= brick_size
                    if ts >= window_start_ts:
                        out_dates.append(ts)
                        out_closes.append(current_brick_price)
                        out_uptrends.append(-1)
            elif price >= current_brick_price + 2 * brick_size:
                current_brick_price += 2 * brick_size
                if ts >= window_start_ts:
                    out_dates.append(ts)
                    out_closes.append(current_brick_price)
                    out_uptrends.append(1)
                uptrend = 1
                while price >= current_brick_price + brick_size:
                    current_brick_price += brick_size
                    if ts >= window_start_ts:
                        out_dates.append(ts)
                        out_closes.append(current_brick_price)
                        out_uptrends.append(1)

    return out_dates, out_closes, out_uptrends


@jit(nopython=True)
def _simulate_profit_jit(b_dates, b_closes, b_uptrends, tick_ts, tick_prices, brick_size, be_trigger_frac):
    """
    JIT-compiled trade simulator. Exact parity with simulate_profit_ticks_jit.
    """
    n_bricks = len(b_dates)
    if n_bricks == 0:
        return 0.0

    daily_pnl = 0.0
    last_tick_idx = 0
    max_tick_idx = len(tick_ts)

    for i in range(n_bricks):
        entry_price = b_closes[i]
        brick_up = b_uptrends[i]
        start_time = b_dates[i]

        if brick_up == 1:
            tp_price = entry_price + brick_size
            sl_price = entry_price - brick_size
            be_trigger = entry_price + (be_trigger_frac * brick_size)
            trade_type = 1
        else:
            tp_price = entry_price - brick_size
            sl_price = entry_price + brick_size
            be_trigger = entry_price - (be_trigger_frac * brick_size)
            trade_type = -1

        if i < n_bricks - 1:
            end_time = b_dates[i + 1]
            if start_time == end_time:
                next_trend = b_uptrends[i + 1]
                if next_trend == brick_up:
                    daily_pnl += 1.0
                else:
                    daily_pnl -= 1.0
                continue
        else:
            end_time = tick_ts[-1] if max_tick_idx > 0 else start_time

        curr_idx = last_tick_idx
        while curr_idx < max_tick_idx and tick_ts[curr_idx] <= start_time:
            curr_idx += 1

        outcome = 0
        sl_moved_to_be = False

        scan_idx = curr_idx
        while scan_idx < max_tick_idx:
            ts = tick_ts[scan_idx]
            if ts >= end_time:
                break

            price = tick_prices[scan_idx]

            if trade_type == 1:
                current_sl = entry_price if sl_moved_to_be else sl_price
                if price <= current_sl:
                    outcome = -1 if not sl_moved_to_be else 0
                    break
                if price >= tp_price:
                    outcome = 1
                    break
                if price >= be_trigger:
                    sl_moved_to_be = True
            else:
                current_sl = entry_price if sl_moved_to_be else sl_price
                if price >= current_sl:
                    outcome = -1 if not sl_moved_to_be else 0
                    break
                if price <= tp_price:
                    outcome = 1
                    break
                if price <= be_trigger:
                    sl_moved_to_be = True

            scan_idx += 1

        last_tick_idx = scan_idx

        if outcome == 1:
            daily_pnl += 0.5
        elif outcome == -1:
            daily_pnl -= 0.5

    return daily_pnl


@jit(nopython=True)
def _optimize_candidates_jit(
    timestamps, prices, candidates,
    anchor_prices, anchor_start,
    brick_size, window_start_ts, target_start_ts,
    sim_ts, sim_prices, be_trigger_frac
):
    """
    JIT-compiled optimization loop. Tests all candidates in native code.
    """
    best_profit = -1e18
    best_cand = candidates[0]
    best_hit_idx = anchor_start
    n_cands = len(candidates)

    for c_idx in range(n_cands):
        cand = candidates[c_idx]

        # Find first tick in anchor day within brick_size of candidate
        hit_found = False
        hit_idx = anchor_start
        for j in range(len(anchor_prices)):
            if abs(anchor_prices[j] - cand) < brick_size:
                hit_idx = anchor_start + j
                hit_found = True
                break

        if not hit_found:
            continue

        # Build Renko
        b_dates, b_closes, b_uptrends = _build_renko_jit(
            timestamps, prices, hit_idx, cand, brick_size,
            window_start_ts, target_start_ts
        )

        if len(b_dates) == 0:
            continue

        # Convert to arrays for simulation
        bd = np.array(b_dates)
        bc = np.array(b_closes)
        bu = np.array(b_uptrends)

        # Simulate
        pnl = _simulate_profit_jit(bd, bc, bu, sim_ts, sim_prices, brick_size, be_trigger_frac)

        if pnl > best_profit:
            best_profit = pnl
            best_cand = cand
            best_hit_idx = hit_idx

    return best_cand, best_hit_idx, best_profit


# ══════════════════════════════════════════════════════════════
# PathOptimizer Class (orchestration layer)
# ══════════════════════════════════════════════════════════════

class PathOptimizer:
    """
    Finds the optimal Renko starting anchor price by simulating
    historical performance across candidate start prices.
    Uses Numba JIT compilation for maximum performance.
    """

    def __init__(self):
        # Warm up JIT on first construction (compile with tiny arrays)
        logger.info("PathOptimizer: Warming up JIT compiler...")
        _dummy_ts = np.array([1, 2, 3], dtype=np.int64)
        _dummy_p = np.array([100.0, 110.0, 120.0], dtype=np.float64)
        _build_renko_jit(_dummy_ts, _dummy_p, 0, 100.0, 10.0, 0, 999)
        _simulate_profit_jit(
            np.array([1], dtype=np.int64),
            np.array([110.0], dtype=np.float64),
            np.array([1], dtype=np.int64),
            _dummy_ts, _dummy_p, 10.0, 0.3125
        )
        _optimize_candidates_jit(
            _dummy_ts, _dummy_p, np.array([100.0]),
            _dummy_p, 0, 10.0, 0, 999,
            _dummy_ts, _dummy_p, 0.3125
        )
        logger.info("PathOptimizer: JIT compilation complete.")

    def find_optimal_anchor(self, history_ticks, brick_size: float):
        """
        Main entry point. Find the best starting price for the RenkoBuilder.

        Args:
            history_ticks: list of tick dicts with 'time_msc' and 'bid' keys
            brick_size: current brick size (day_open * K_MULTIPLIER)

        Returns:
            (best_start_price, best_start_idx, best_profit)
            or (None, None, None) if optimization fails
        """
        if not history_ticks or len(history_ticks) == 0:
            logger.error("PathOptimizer: No ticks provided.")
            return None, None, None

        # Pre-extract numpy arrays
        prices = np.array([t['bid'] for t in history_ticks], dtype=np.float64)
        time_msc = np.array([t['time_msc'] for t in history_ticks], dtype=np.int64)

        # Diagnostic: timestamp range
        first_ts = datetime.fromtimestamp(int(time_msc[0]) / 1000, tz=timezone.utc)
        last_ts = datetime.fromtimestamp(int(time_msc[-1]) / 1000, tz=timezone.utc)
        logger.info(
            f"PathOptimizer: {len(time_msc)} ticks, "
            f"range: {first_ts.strftime('%Y-%m-%d %H:%M')} → {last_ts.strftime('%Y-%m-%d %H:%M')} UTC, "
            f"price: {prices[0]:.2f} → {prices[-1]:.2f}"
        )

        # 1. Identify day boundaries
        day_boundaries = self._find_day_boundaries(time_msc)
        
        # Diagnostic: show all days found
        day_strs = [f"{b[2]}({b[1]-b[0]}t)" for b in day_boundaries]
        logger.info(f"PathOptimizer: Found {len(day_boundaries)} days: {', '.join(day_strs)}")

        if len(day_boundaries) < 2:
            logger.warning("PathOptimizer: Less than 2 days of data. Using first tick as anchor.")
            return float(prices[0]), 0, 0.0

        # 2. Anchor day = first, target = last
        anchor_day = day_boundaries[0]
        target_day = day_boundaries[-1]
        sim_days = day_boundaries[1:]

        anchor_start, anchor_end = anchor_day[0], anchor_day[1]
        anchor_prices = prices[anchor_start:anchor_end]
        anchor_high = float(anchor_prices.max())
        anchor_low = float(anchor_prices.min())
        anchor_open = float(anchor_prices[0])

        logger.info(
            f"PathOptimizer: Anchor day {anchor_day[2]} | "
            f"Open: {anchor_open:.2f}, High: {anchor_high:.2f}, Low: {anchor_low:.2f}"
        )

        # 3. Generate candidates (step_size uses STEP_FACTOR = 0.00295)
        step_size = anchor_open * STEP_FACTOR * 0.01
        if step_size <= 0:
            return float(prices[0]), 0, 0.0

        candidates = np.arange(anchor_low, anchor_high + step_size / 1000, step_size)[::-1]
        # Filter to within range
        candidates = candidates[(candidates >= anchor_low) & (candidates <= anchor_high)]

        logger.info(f"PathOptimizer: Testing {len(candidates)} candidates with Numba JIT...")

        # 4. Pre-slice simulation arrays
        sim_start_idx = sim_days[0][0]
        target_start_idx = target_day[0]

        sim_ts = time_msc[sim_start_idx:target_start_idx].copy()
        sim_prices_arr = prices[sim_start_idx:target_start_idx].copy()
        window_start_ts = int(time_msc[sim_start_idx])
        target_start_ts = int(time_msc[target_start_idx]) if target_start_idx < len(time_msc) else int(time_msc[-1])

        if len(sim_ts) == 0:
            logger.warning("PathOptimizer: No simulation ticks between anchor and target day.")
            return float(prices[0]), 0, 0.0

        # 5. Run JIT-compiled optimization
        best_cand, best_hit_idx, best_profit = _optimize_candidates_jit(
            time_msc, prices, candidates.astype(np.float64),
            anchor_prices.copy(), anchor_start,
            brick_size, window_start_ts, target_start_ts,
            sim_ts, sim_prices_arr, BE_TRIGGER
        )

        # Detect sentinel value — no candidate produced valid bricks
        if best_profit <= -1e17:
            logger.warning(
                f"PathOptimizer: No candidate produced valid bricks. "
                f"Falling back to target day open as anchor."
            )
            target_start = target_day[0]
            return float(prices[target_start]), int(target_start), 0.0

        logger.info(
            f"PathOptimizer: Best anchor = {best_cand:.2f} | "
            f"Historical PnL = {best_profit:.2f} | "
            f"Start index = {best_hit_idx} | "
            f"Candidates tested = {len(candidates)}"
        )

        return float(best_cand), int(best_hit_idx), float(best_profit)

    def _find_day_boundaries(self, time_msc_array):
        """Identify UTC day boundaries. Vectorized with numpy."""
        if len(time_msc_array) == 0:
            return []

        day_numbers = (time_msc_array // 1000 // 86400).astype(np.int64)
        changes = np.where(np.diff(day_numbers) != 0)[0] + 1

        boundaries = []
        prev_idx = 0
        for change_idx in changes:
            dn = day_numbers[prev_idx]
            date_str = datetime.fromtimestamp(int(dn) * 86400, tz=timezone.utc).strftime('%Y-%m-%d')
            boundaries.append((prev_idx, int(change_idx), date_str))
            prev_idx = int(change_idx)

        dn = day_numbers[prev_idx]
        date_str = datetime.fromtimestamp(int(dn) * 86400, tz=timezone.utc).strftime('%Y-%m-%d')
        boundaries.append((prev_idx, len(time_msc_array), date_str))

        return boundaries
