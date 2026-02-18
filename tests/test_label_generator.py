"""
Unit tests for Phase 1: Label Generator — calculate_true_overshoot.

Tests cover all scenarios from tasks.md 1.4:
- LONG LOSS: partial extension then SL hit
- LONG LOSS: immediate SL hit
- LONG WIN: TP hit + extension + trailing reversal
- LONG WIN: TP hit exactly, immediate retrace
- SHORT: mirror of all LONG tests
- Edge case: tick data exhausted before resolution
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from label_generator import calculate_true_overshoot


def _make_ticks(prices: list[tuple[float, float]]) -> pd.DataFrame:
    """Helper: create a tick DataFrame from a list of (bid, ask) tuples.

    Timestamps are synthetic (1 second apart, starting 2024-01-01 00:00:00).
    """
    ts = pd.date_range("2024-01-01", periods=len(prices), freq="1s", tz="UTC")
    bids = [p[0] for p in prices]
    asks = [p[1] for p in prices]
    return pd.DataFrame({
        "timestamp": ts,
        "bid": bids,
        "bid_vol": [0.001] * len(prices),
        "ask": asks,
        "ask_vol": [0.001] * len(prices),
    })


# ═══════════════════════════════════════════════════════════════
# LONG LOSS Tests
# ═══════════════════════════════════════════════════════════════

class TestLongLoss:
    """Test LONG direction where SL is hit (y_class = 0)."""

    def test_partial_extension_then_sl(self):
        """Price goes up 0.5 bricks then hits SL → y_mag ≈ 0.5."""
        entry = 100.0
        brick_size = 2.0
        # TP = 102, SL = 98
        # Spread = 0.2 → mid = bid + 0.1
        ticks = _make_ticks([
            (100.0, 100.2),   # mid=100.1 — slight up from entry
            (100.5, 100.7),   # mid=100.6
            (100.9, 101.1),   # mid=101.0  → peak = 101.0 (0.5 bricks up)
            (100.5, 100.7),   # mid=100.6  — retracing
            (100.0, 100.2),   # mid=100.1
            (97.8, 98.0),     # mid=97.9  → SL hit (mid <= 98)
        ])
        result = calculate_true_overshoot(entry, brick_size, True, ticks)

        assert result["y_class"] == 0, "Should be LOSS"
        assert result["tp_hit"] is False
        assert result["resolved"] is True
        assert abs(result["y_mag"] - 0.5) < 0.01, f"Expected y_mag ≈ 0.5, got {result['y_mag']}"

    def test_immediate_sl_hit(self):
        """Price never rises, immediately hits SL → y_mag ≈ 0.0."""
        entry = 100.0
        brick_size = 2.0
        # TP = 102, SL = 98
        ticks = _make_ticks([
            (99.5, 99.7),    # mid=99.6 — dropping
            (97.8, 98.0),    # mid=97.9 → SL hit
        ])
        result = calculate_true_overshoot(entry, brick_size, True, ticks)

        assert result["y_class"] == 0, "Should be LOSS"
        assert result["tp_hit"] is False
        assert result["resolved"] is True
        # Peak was at entry (100.0) since no tick went above. But first tick
        # mid=99.6, so peak stays at 100.0 (entry) → y_mag = 0.0
        assert result["y_mag"] < 0.1, f"Expected y_mag ≈ 0.0, got {result['y_mag']}"


# ═══════════════════════════════════════════════════════════════
# LONG WIN Tests
# ═══════════════════════════════════════════════════════════════

class TestLongWin:
    """Test LONG direction where TP is hit (y_class = 1)."""

    def test_tp_then_extends_one_brick(self):
        """Price hits TP then extends 1 more brick before trailing reversal → y_mag ≈ 2.0."""
        entry = 100.0
        brick_size = 2.0
        # TP = 102, SL = 98
        ticks = _make_ticks([
            (101.0, 101.2),   # mid=101.1  — climbing
            (101.9, 102.1),   # mid=102.0  → TP hit! (mid >= 102)
            (102.5, 102.7),   # mid=102.6  — extending past TP
            (103.5, 103.7),   # mid=103.6
            (103.9, 104.1),   # mid=104.0  → peak = 104.0 (2 bricks above entry)
            (103.0, 103.2),   # mid=103.1  — retracing but not 1 brick from peak
            (101.9, 102.1),   # mid=102.0  → trailing reversal! (104.0 - 102.0 = 2.0 = brick_size)
        ])
        result = calculate_true_overshoot(entry, brick_size, True, ticks)

        assert result["y_class"] == 1, "Should be WIN"
        assert result["tp_hit"] is True
        assert result["resolved"] is True
        assert abs(result["y_mag"] - 2.0) < 0.01, f"Expected y_mag ≈ 2.0, got {result['y_mag']}"

    def test_tp_hit_immediate_retrace(self):
        """Price hits TP exactly, immediately retraces → y_mag ≈ 1.0."""
        entry = 100.0
        brick_size = 2.0
        # TP = 102, SL = 98
        ticks = _make_ticks([
            (101.0, 101.2),   # mid=101.1
            (101.9, 102.1),   # mid=102.0  → TP hit
            (101.0, 101.2),   # mid=101.1  — immediate drop
            (99.9, 100.1),    # mid=100.0  → trailing reversal (102.0 - 100.0 = 2.0 >= brick_size)
        ])
        result = calculate_true_overshoot(entry, brick_size, True, ticks)

        assert result["y_class"] == 1, "Should be WIN"
        assert result["tp_hit"] is True
        assert result["resolved"] is True
        assert abs(result["y_mag"] - 1.0) < 0.01, f"Expected y_mag ≈ 1.0, got {result['y_mag']}"


# ═══════════════════════════════════════════════════════════════
# SHORT Tests (mirror of LONG)
# ═══════════════════════════════════════════════════════════════

class TestShortLoss:
    """Test SHORT direction where SL is hit (y_class = 0)."""

    def test_partial_extension_then_sl(self):
        """Price goes down 0.5 bricks then hits SL → y_mag ≈ 0.5."""
        entry = 100.0
        brick_size = 2.0
        # TP = 98 (SHORT: entry - brick_size), SL = 102 (entry + brick_size)
        ticks = _make_ticks([
            (99.8, 100.0),    # mid=99.9
            (99.3, 99.5),     # mid=99.4
            (98.9, 99.1),     # mid=99.0  → peak = 99.0 (0.5 bricks down)
            (99.5, 99.7),     # mid=99.6  — bouncing back
            (100.5, 100.7),   # mid=100.6
            (101.9, 102.1),   # mid=102.0 → SL hit (mid >= 102)
        ])
        result = calculate_true_overshoot(entry, brick_size, False, ticks)

        assert result["y_class"] == 0, "Should be LOSS"
        assert result["tp_hit"] is False
        assert result["resolved"] is True
        assert abs(result["y_mag"] - 0.5) < 0.01, f"Expected y_mag ≈ 0.5, got {result['y_mag']}"

    def test_immediate_sl_hit(self):
        """Price immediately hits SL → y_mag ≈ 0.0."""
        entry = 100.0
        brick_size = 2.0
        ticks = _make_ticks([
            (100.3, 100.5),   # mid=100.4 — rising
            (101.9, 102.1),   # mid=102.0 → SL hit
        ])
        result = calculate_true_overshoot(entry, brick_size, False, ticks)

        assert result["y_class"] == 0, "Should be LOSS"
        assert result["resolved"] is True
        assert result["y_mag"] < 0.1, f"Expected y_mag ≈ 0.0, got {result['y_mag']}"


class TestShortWin:
    """Test SHORT direction where TP is hit (y_class = 1)."""

    def test_tp_then_extends_one_brick(self):
        """SHORT: price drops to TP then extends 1 more brick → y_mag ≈ 2.0."""
        entry = 100.0
        brick_size = 2.0
        # TP = 98, SL = 102
        ticks = _make_ticks([
            (99.0, 99.2),     # mid=99.1 — dropping
            (97.9, 98.1),     # mid=98.0  → TP hit (mid <= 98)
            (97.0, 97.2),     # mid=97.1  — extending
            (95.9, 96.1),     # mid=96.0  → peak = 96.0 (2 bricks below entry)
            (96.5, 96.7),     # mid=96.6  — bouncing
            (97.9, 98.1),     # mid=98.0  → trailing reversal (96.0 + 2.0 = 98.0)
        ])
        result = calculate_true_overshoot(entry, brick_size, False, ticks)

        assert result["y_class"] == 1, "Should be WIN"
        assert result["tp_hit"] is True
        assert result["resolved"] is True
        assert abs(result["y_mag"] - 2.0) < 0.01, f"Expected y_mag ≈ 2.0, got {result['y_mag']}"

    def test_tp_hit_immediate_retrace(self):
        """SHORT: price hits TP exactly, immediately retraces → y_mag ≈ 1.0."""
        entry = 100.0
        brick_size = 2.0
        ticks = _make_ticks([
            (99.0, 99.2),     # mid=99.1
            (97.9, 98.1),     # mid=98.0  → TP hit
            (99.0, 99.2),     # mid=99.1  — bouncing up
            (99.9, 100.1),    # mid=100.0 → trailing reversal (98.0 + 2.0 = 100.0)
        ])
        result = calculate_true_overshoot(entry, brick_size, False, ticks)

        assert result["y_class"] == 1, "Should be WIN"
        assert result["tp_hit"] is True
        assert result["resolved"] is True
        assert abs(result["y_mag"] - 1.0) < 0.01, f"Expected y_mag ≈ 1.0, got {result['y_mag']}"


# ═══════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_ticks(self):
        """No tick data → unresolved, y_class/y_mag are None."""
        ticks = _make_ticks([])
        result = calculate_true_overshoot(100.0, 2.0, True, ticks)

        assert result["y_class"] is None
        assert result["y_mag"] is None
        assert result["resolved"] is False

    def test_ticks_end_before_resolution(self):
        """Tick data ends before SL/TP/reversal is triggered → resolved=False."""
        entry = 100.0
        brick_size = 2.0
        # Price meanders without hitting TP (102) or SL (98)
        ticks = _make_ticks([
            (100.0, 100.2),   # mid=100.1
            (100.5, 100.7),   # mid=100.6
            (100.3, 100.5),   # mid=100.4
            (100.8, 101.0),   # mid=100.9
        ])
        result = calculate_true_overshoot(entry, brick_size, True, ticks)

        assert result["resolved"] is False
        assert result["y_class"] is None, "Unresolved should have y_class=None"

    def test_y_mag_boundary_win_exactly_at_tp(self):
        """Price hits TP exactly on first check, then immediately triggers
        trailing reversal → y_mag should be exactly 1.0."""
        entry = 100.0
        brick_size = 2.0
        ticks = _make_ticks([
            (101.9, 102.1),   # mid=102.0 → TP hit
            (99.9, 100.1),    # mid=100.0 → trailing reversal (102 - 100 = 2 = brick_size)
        ])
        result = calculate_true_overshoot(entry, brick_size, True, ticks)

        assert result["y_class"] == 1
        assert result["y_mag"] == 1.0

    def test_large_extension(self):
        """Price extends significantly beyond TP → large y_mag."""
        entry = 100.0
        brick_size = 2.0
        ticks = _make_ticks([
            (101.9, 102.1),   # mid=102.0 → TP hit
            (105.0, 105.2),   # mid=105.1
            (107.0, 107.2),   # mid=107.1
            (109.0, 109.2),   # mid=109.1  → peak = 109.1 (~4.55 bricks)
            (106.9, 107.1),   # mid=107.0  → trailing reversal (109.1 - 107.0 > 2.0)
        ])
        result = calculate_true_overshoot(entry, brick_size, True, ticks)

        assert result["y_class"] == 1
        assert result["y_mag"] > 4.0, f"Expected y_mag > 4.0, got {result['y_mag']}"
        assert result["y_mag"] < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
