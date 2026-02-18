"""Unit tests for feature_engine.py (Phase 2.7)."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from feature_engine import (
    RollingZScore,
    compute_ofi,
    compute_depth,
    compute_susceptibility,
    compute_velocity,
    compute_spread,
    compute_progress,
    compute_flag_curr,
    compute_flag_zone,
    compute_decay,
    compute_macro_vector,
)


# ═══════════════════════════════════════════════════════════════
# RollingZScore Tests
# ═══════════════════════════════════════════════════════════════

class TestRollingZScore:
    """Tests for RollingZScore correctness (FR-FE-03, FR-FE-04, FR-FE-05)."""

    def test_returns_zero_when_n_lt_30(self):
        """FR-FE-05: z-score should return 0.0 when fewer than 30 values."""
        zs = RollingZScore(window=1000)
        for i in range(29):
            z = zs.update(float(i))
            assert z == 0.0, f"Expected 0.0 at N={i+1}, got {z}"

    def test_produces_value_at_n30(self):
        """After 30 values, z-score should produce non-zero values."""
        zs = RollingZScore(window=1000)
        for i in range(30):
            z = zs.update(float(i))
        # The 30th value should produce a non-zero z-score
        assert z != 0.0, "Expected non-zero z at N=30"

    def test_matches_numpy_during_filling(self):
        """Z-score during filling phase should match numpy computation."""
        zs = RollingZScore(window=1000)
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0,
                  5.0, 8.0, 9.0, 7.0, 9.0, 3.0, 2.0, 3.0, 8.0, 4.0,
                  6.0, 2.0, 6.0, 4.0, 3.0, 3.0, 8.0, 3.0, 2.0, 7.0]
        for v in values:
            z = zs.update(v)

        # Compare last z-score with numpy
        arr = np.array(values)
        expected_z = (values[-1] - arr.mean()) / arr.std(ddof=1)
        assert abs(z - expected_z) < 1e-6, f"Got {z}, expected {expected_z}"

    def test_matches_numpy_after_full_window(self):
        """Z-score after full window should match numpy on the last 1000 values."""
        zs = RollingZScore(window=100)  # Use smaller window for test speed
        np.random.seed(42)
        values = np.random.randn(200).tolist()

        for v in values:
            z = zs.update(v)

        # Compare with numpy on last 100 values
        window = values[-100:]
        arr = np.array(window)
        expected_z = (values[-1] - arr.mean()) / arr.std(ddof=1)
        assert abs(z - expected_z) < 1e-4, f"Got {z}, expected {expected_z}"

    def test_sigma_zero_returns_zero(self):
        """When σ=0 (all constant values), should return 0.0."""
        zs = RollingZScore(window=1000)
        for _ in range(50):
            z = zs.update(5.0)
        assert z == 0.0, f"Expected 0.0 for constant input, got {z}"

    def test_sigma_zero_after_full_window(self):
        """σ=0 after full window should also return 0.0."""
        zs = RollingZScore(window=50)
        for _ in range(60):
            z = zs.update(3.14)
        assert z == 0.0, f"Expected 0.0, got {z}"

    def test_incremental_vs_batch(self):
        """Incremental O(1) update should match batch recompute."""
        np.random.seed(123)
        values = np.random.randn(1500).tolist()

        zs = RollingZScore(window=1000)
        for v in values:
            z = zs.update(v)

        # Compare with batch numpy on last 1000
        window = values[-1000:]
        arr = np.array(window)
        expected_z = (values[-1] - arr.mean()) / arr.std(ddof=1)
        assert abs(z - expected_z) < 1e-3, f"Got {z}, expected {expected_z}"


# ═══════════════════════════════════════════════════════════════
# OFI Tests (FR-FE-01)
# ═══════════════════════════════════════════════════════════════

class TestOFI:
    """Tests for OFI weak inequality computation."""

    def test_weak_ineq_bid_static_vol_changes(self):
        """dBid=0 but bid_vol changes → e_k ≠ 0.

        When dBid=0: both I(dBid>=0) and I(dBid<=0) fire.
        e_k = 1*bid_vol_k - 1*bid_vol_{k-1} - ... = bid_vol_change + ask_vol_change
        """
        # bid unchanged, bid_vol increases from 100 to 150
        # ask unchanged, ask_vol unchanged at 100
        e = compute_ofi(
            bid_k=100.0, bid_km1=100.0,
            ask_k=101.0, ask_km1=101.0,
            bid_vol_k=150.0, bid_vol_km1=100.0,
            ask_vol_k=100.0, ask_vol_km1=100.0,
        )
        assert e != 0.0, f"Expected non-zero OFI when bid_vol changes, got {e}"
        assert e == 50.0, f"Expected 50.0, got {e}"

    def test_weak_ineq_bid_static_same_vol(self):
        """dBid=0, same bid_vol → bid contribution is 0."""
        # Everything static
        e = compute_ofi(
            bid_k=100.0, bid_km1=100.0,
            ask_k=101.0, ask_km1=101.0,
            bid_vol_k=100.0, bid_vol_km1=100.0,
            ask_vol_k=100.0, ask_vol_km1=100.0,
        )
        assert e == 0.0, f"Expected 0.0 when nothing changes, got {e}"

    def test_bid_up_ask_static(self):
        """Bid increases: I(dBid>=0)=1, I(dBid<=0)=0."""
        e = compute_ofi(
            bid_k=101.0, bid_km1=100.0,
            ask_k=102.0, ask_km1=102.0,
            bid_vol_k=50.0, bid_vol_km1=50.0,
            ask_vol_k=50.0, ask_vol_km1=50.0,
        )
        # dBid > 0: I(>=0)=1, I(<=0)=0 → bid_vol_k - 0 = 50
        # dAsk = 0: I(<=0)=1, I(>=0)=1 → -ask_vol_k + ask_vol_{k-1} = 0
        assert e == 50.0, f"Expected 50.0, got {e}"


# ═══════════════════════════════════════════════════════════════
# Susceptibility Test (FR-FE-02)
# ═══════════════════════════════════════════════════════════════

class TestSusceptibility:
    """Test that susceptibility divides RAW, not z-scored values."""

    def test_divide_raw_first(self):
        """S_raw = e_k / (D_k + 1e-8), then z-score S_raw."""
        ofi_raw = 10.0
        depth_raw = 200.0
        s = compute_susceptibility(ofi_raw, depth_raw)
        expected = 10.0 / (200.0 + 1e-8)
        assert abs(s - expected) < 1e-10, f"Got {s}, expected {expected}"

    def test_zero_depth_no_division_error(self):
        """Depth near zero should not cause division error."""
        s = compute_susceptibility(5.0, 0.0)
        expected = 5.0 / 1e-8
        assert abs(s - expected) < 1.0, f"Got {s}, expected ~{expected}"


# ═══════════════════════════════════════════════════════════════
# Other Feature Tests
# ═══════════════════════════════════════════════════════════════

class TestOtherFeatures:
    """Tests for other feature computations."""

    def test_depth(self):
        assert compute_depth(100.0, 200.0) == 300.0

    def test_velocity(self):
        # 100ms gap
        v = compute_velocity(200.0, 100.0)
        expected = 1.0 / (100.0 + 1e-3)
        assert abs(v - expected) < 1e-8

    def test_spread(self):
        assert compute_spread(101.5, 100.0) == 1.5

    def test_progress(self):
        # mid=105, brick_open=100, brick_size=10 → progress=0.5
        p = compute_progress(105.0, 100.0, 10.0)
        assert abs(p - 0.5) < 1e-10

    def test_flag_curr_same_brick(self):
        assert compute_flag_curr(5, 5) == 1

    def test_flag_curr_diff_brick(self):
        assert compute_flag_curr(3, 5) == 0

    def test_flag_zone_beyond(self):
        # mid=115, prev_open=100, prev_size=10 → |115-100|=15 >= 10 → 1
        assert compute_flag_zone(115.0, 100.0, 10.0) == 1

    def test_flag_zone_within(self):
        # mid=105, prev_open=100, prev_size=10 → |105-100|=5 < 10 → 0
        assert compute_flag_zone(105.0, 100.0, 10.0) == 0

    def test_decay(self):
        d = compute_decay(current_brick_id=10, tick_brick_id=5, max_depth=20)
        assert abs(d - 0.25) < 1e-10

    def test_decay_capped_at_1(self):
        d = compute_decay(current_brick_id=100, tick_brick_id=0, max_depth=20)
        assert d == 1.0


# ═══════════════════════════════════════════════════════════════
# Macro Vector Tests
# ═══════════════════════════════════════════════════════════════

class TestMacroVector:
    """Tests for macro-vector computation (FR-FE-07)."""

    def test_basic(self):
        mv = compute_macro_vector(
            duration_s=100.0, is_uptrend=True,
            brick_size=5.0, brick_size_history=[5.0, 5.0, 5.0]
        )
        assert mv.shape == (3,)
        assert abs(mv[0] - np.log(101.0)) < 1e-4  # log_dur
        assert mv[1] == 1.0  # direction
        # z_size with constant history → 0
        assert mv[2] == 0.0

    def test_downtrend(self):
        mv = compute_macro_vector(
            duration_s=60.0, is_uptrend=False,
            brick_size=3.0, brick_size_history=[3.0]
        )
        assert mv[1] == -1.0

    def test_z_size_with_variation(self):
        history = [4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 5.0]
        mv = compute_macro_vector(
            duration_s=100.0, is_uptrend=True,
            brick_size=8.0, brick_size_history=history
        )
        # z_size should be positive (8.0 > mean of history)
        assert mv[2] > 0, f"Expected positive z_size, got {mv[2]}"
