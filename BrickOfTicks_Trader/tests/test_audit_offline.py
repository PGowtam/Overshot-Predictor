"""
Test Suite: Data Audit — Offline Tests (Phase -1 Verification)
===============================================================
Tests the audit analysis logic using actual Dukascopy training parquets.
Does NOT require a live socket connection.

Covers:
  - Training data loading
  - Distribution analysis (spread, velocity, volume)
  - Volume fallback validation
  - Report generation
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridge.data_audit import (
    load_training_sample,
    audit_distributions,
    validate_fallback,
    generate_audit_report,
    TRAINING_TICK_DIR,
)


# ═══════════════════════════════════════════════════════════════════════
# Helper: Create synthetic tick DataFrames
# ═══════════════════════════════════════════════════════════════════════

def make_synthetic_ticks(n=1000, base_bid=2400.0, spread=0.10,
                         bid_vol=1.0, ask_vol=1.0, dt_ms=200):
    """Create a synthetic tick DataFrame matching the parquet schema."""
    np.random.seed(42)
    timestamps = np.arange(n) * dt_ms + 1714900800000  # ms since epoch
    bids = base_bid + np.cumsum(np.random.randn(n) * 0.01)
    asks = bids + spread
    bid_vols = np.full(n, bid_vol)
    ask_vols = np.full(n, ask_vol)

    return pd.DataFrame({
        'time_msc': timestamps.astype(int),
        'bid': bids,
        'ask': asks,
        'bid_vol': bid_vols,
        'ask_vol': ask_vols,
    })


def make_training_like_ticks(n=1000, base_bid=2400.0, spread=0.10, dt_ms=200):
    """Create ticks matching training data schema (has 'timestamp' column)."""
    np.random.seed(42)
    timestamps = pd.date_range('2023-01-03', periods=n, freq=f'{dt_ms}ms')
    bids = base_bid + np.cumsum(np.random.randn(n) * 0.01)
    asks = bids + spread
    bid_vols = np.full(n, 0.001)
    ask_vols = np.full(n, 0.001)

    return pd.DataFrame({
        'timestamp': timestamps,
        'bid': bids,
        'ask': asks,
        'bid_vol': bid_vols,
        'ask_vol': ask_vols,
    })


# ═══════════════════════════════════════════════════════════════════════
# Training Data Loading
# ═══════════════════════════════════════════════════════════════════════

class TestTrainingDataLoading:
    """Test that we can load Dukascopy training parquets."""

    def test_training_dir_exists(self):
        """Training tick directory exists in project."""
        assert TRAINING_TICK_DIR.exists(), (
            f"Training data directory not found: {TRAINING_TICK_DIR}")

    def test_2023_data_exists(self):
        """2023 training data directory exists."""
        dir_2023 = TRAINING_TICK_DIR / "2023"
        assert dir_2023.exists(), f"2023 data directory not found: {dir_2023}"

    def test_load_training_sample(self):
        """load_training_sample() returns valid DataFrame."""
        df = load_training_sample(year=2023, n_files=2, max_ticks=1000)

        assert len(df) > 0
        assert 'bid' in df.columns
        assert 'ask' in df.columns
        assert 'bid_vol' in df.columns or 'timestamp' in df.columns

    def test_training_schema(self):
        """Training data has expected columns."""
        df = load_training_sample(year=2023, n_files=1, max_ticks=100)

        # Must have bid and ask at minimum
        assert 'bid' in df.columns
        assert 'ask' in df.columns
        # Should have timestamp or time_msc
        assert 'timestamp' in df.columns or 'time_msc' in df.columns


# ═══════════════════════════════════════════════════════════════════════
# Distribution Analysis Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDistributionAnalysis:
    """Test audit_distributions() logic."""

    def test_identical_data_passes(self):
        """When live and training data are identical, all checks pass."""
        df = make_synthetic_ticks(n=500)
        results = audit_distributions(df, df)

        assert results['spread']['PASS'] is True
        assert results['spread']['drift_pct'] < 1.0  # Near zero
        assert results['velocity']['PASS'] is True
        assert results['verdict']['PROCEED'] is True

    def test_spread_drift_detection(self):
        """Spread drift > 20% is detected and flagged (relative spread)."""
        # Same base_bid so relative comparison is meaningful
        live_df = make_synthetic_ticks(n=500, spread=0.50, base_bid=2400.0)
        train_df = make_synthetic_ticks(n=500, spread=0.10, base_bid=2400.0)

        results = audit_distributions(live_df, train_df)

        # Live spread is 5× training spread (same price) → drift should be >20%
        assert results['spread']['PASS'] is False
        assert results['spread']['drift_pct'] > 20

    def test_spread_within_tolerance(self):
        """Spread drift < 20% passes (relative spread, same price level)."""
        live_df = make_synthetic_ticks(n=500, spread=0.11, base_bid=2400.0)
        train_df = make_synthetic_ticks(n=500, spread=0.10, base_bid=2400.0)

        results = audit_distributions(live_df, train_df)
        assert results['spread']['PASS'] is True

    def test_velocity_drift_detection(self):
        """Velocity drift > 50% is flagged."""
        live_df = make_synthetic_ticks(n=500, dt_ms=600)    # Slow feed
        train_df = make_synthetic_ticks(n=500, dt_ms=200)   # Fast feed

        results = audit_distributions(live_df, train_df)
        assert results['velocity']['drift_pct'] > 50

    def test_velocity_within_tolerance(self):
        """Velocity drift < 50% passes."""
        live_df = make_synthetic_ticks(n=500, dt_ms=250)    # 25% slower
        train_df = make_synthetic_ticks(n=500, dt_ms=200)

        results = audit_distributions(live_df, train_df)
        assert results['velocity']['PASS'] is True

    def test_volume_available(self):
        """Detects when volume data is available (>50%)."""
        df = make_synthetic_ticks(n=500, bid_vol=1.0)
        results = audit_distributions(df, df)

        assert results['volume']['fallback_required'] is False
        assert results['volume']['live_vol_pct'] > 50

    def test_volume_fallback_required(self):
        """Detects when volume fallback is required (<50%)."""
        live_df = make_synthetic_ticks(n=500, bid_vol=0.0, ask_vol=0.0)
        train_df = make_synthetic_ticks(n=500, bid_vol=1.0)

        results = audit_distributions(live_df, train_df)
        assert results['volume']['fallback_required'] is True
        assert results['volume']['live_vol_pct'] == 0.0

    def test_verdict_requires_both_checks(self):
        """PROCEED requires both spread AND velocity to pass."""
        # Spread passes (same base_bid and spread), velocity fails
        live_df = make_synthetic_ticks(n=500, spread=0.10, dt_ms=1000, base_bid=2400.0)
        train_df = make_synthetic_ticks(n=500, spread=0.10, dt_ms=200, base_bid=2400.0)

        results = audit_distributions(live_df, train_df)
        assert results['spread']['PASS'] is True
        # Velocity drift: |1000-200|/200 = 400% > 50%
        assert results['velocity']['PASS'] is False
        assert results['verdict']['PROCEED'] is False


# ═══════════════════════════════════════════════════════════════════════
# Volume Fallback Validation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestVolumeFallbackValidation:
    """Test validate_fallback() logic."""

    def test_skip_when_volume_available(self):
        """Skips validation when volume data is present."""
        df = make_synthetic_ticks(n=500, bid_vol=1.0)
        result = validate_fallback(df)

        assert result['skip'] is True
        assert result['PASS'] is True

    def test_balanced_proxy_ofi_passes(self):
        """Random walk produces balanced OFI proxy → PASS."""
        np.random.seed(42)
        n = 5000
        bids = 2400.0 + np.cumsum(np.random.randn(n) * 0.02)
        asks = bids + 0.10

        df = pd.DataFrame({
            'bid': bids,
            'ask': asks,
            'bid_vol': np.zeros(n),  # No volume
            'ask_vol': np.zeros(n),
        })

        result = validate_fallback(df)
        assert result['skip'] is False
        assert result['PASS'] is True
        assert result['balanced'] is True
        # pos_of_nonzero should be near 0.5
        assert abs(result['pos_of_nonzero'] - 0.5) < 0.10

    def test_skewed_ofi_fails(self):
        """Monotonically rising price → all positive OFI → skewed → FAIL."""
        n = 1000
        bids = np.linspace(2400, 2450, n)  # Always rising
        asks = bids + 0.10

        df = pd.DataFrame({
            'bid': bids,
            'ask': asks,
            'bid_vol': np.zeros(n),
            'ask_vol': np.zeros(n),
        })

        result = validate_fallback(df)
        assert result['skip'] is False
        assert result['PASS'] is False
        assert result['pos_of_nonzero'] > 0.90  # Almost all positive


# ═══════════════════════════════════════════════════════════════════════
# Report Generation Tests
# ═══════════════════════════════════════════════════════════════════════

class TestReportGeneration:
    """Test generate_audit_report() output."""

    def test_report_creates_files(self, tmp_path):
        """Report and JSON profile are created."""
        audit_results = {
            'spread': {'PASS': True, 'drift_pct': 5.0,
                       'live_mean_abs': 0.10, 'live_mean_bps': 4.17, 'live_std_bps': 0.83,
                       'live_median_abs': 0.09,
                       'train_mean_abs': 0.095, 'train_mean_bps': 3.96, 'train_std_bps': 0.75,
                       'train_median_abs': 0.09},
            'velocity': {'PASS': True, 'drift_pct': 10.0,
                         'live_median_dt_ms': 220, 'live_mean_dt_ms': 250,
                         'train_median_dt_ms': 200, 'train_mean_dt_ms': 230},
            'volume': {'live_vol_pct': 85.0, 'train_vol_pct': 90.0,
                       'fallback_required': False},
            'verdict': {'spread_ok': True, 'velocity_ok': True,
                        'volume_fallback_required': False,
                        'PROCEED': True, 'NOTES': 'Full volume mode'}
        }
        fallback_results = {'skip': True, 'reason': 'Volume available', 'PASS': True}

        output_path = tmp_path / "test_audit_report.md"
        result_path = generate_audit_report(
            audit_results, fallback_results, output_path=output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert '✅ PASS' in content
        assert 'Spread' in content
        assert 'Velocity' in content

    def test_report_shows_fail(self, tmp_path):
        """Report correctly shows FAIL status."""
        audit_results = {
            'spread': {'PASS': False, 'drift_pct': 45.0,
                       'live_mean_abs': 0.50, 'live_mean_bps': 20.83, 'live_std_bps': 4.17,
                       'live_median_abs': 0.48,
                       'train_mean_abs': 0.10, 'train_mean_bps': 4.17, 'train_std_bps': 0.83,
                       'train_median_abs': 0.09},
            'velocity': {'PASS': True, 'drift_pct': 5.0,
                         'live_median_dt_ms': 210, 'live_mean_dt_ms': 220,
                         'train_median_dt_ms': 200, 'train_mean_dt_ms': 210},
            'volume': {'live_vol_pct': 0.0, 'train_vol_pct': 90.0,
                       'fallback_required': True},
            'verdict': {'spread_ok': False, 'velocity_ok': True,
                        'volume_fallback_required': True,
                        'PROCEED': False, 'NOTES': 'Spread too wide'}
        }
        fallback_results = {'skip': False, 'PASS': True, 'balanced': True,
                            'total_ticks': 1000, 'pos_count': 480,
                            'neg_count': 470, 'zero_count': 50,
                            'pos_ratio': 0.48, 'neg_ratio': 0.47,
                            'zero_ratio': 0.05, 'pos_of_nonzero': 0.505,
                            'note': 'Balanced'}

        output_path = tmp_path / "test_audit_fail.md"
        generate_audit_report(audit_results, fallback_results, output_path=output_path)

        content = output_path.read_text()
        assert '❌ FAIL' in content


# ═══════════════════════════════════════════════════════════════════════
# Integration: Full Audit on Training Data (self-comparison)
# ═══════════════════════════════════════════════════════════════════════

class TestFullAuditIntegration:
    """Run audit on real training data — comparing it against itself."""

    def test_self_audit_passes(self):
        """Training data compared to itself should always pass."""
        try:
            df = load_training_sample(year=2023, n_files=2, max_ticks=5000)
        except FileNotFoundError:
            pytest.skip("Training data not available")

        results = audit_distributions(df, df)

        # Self-comparison should have near-zero drift
        assert results['spread']['drift_pct'] < 1.0
        assert results['velocity']['drift_pct'] < 1.0
        assert results['verdict']['PROCEED'] is True

    def test_cross_year_audit(self):
        """Compare 2023 vs 2024 training data — should be reasonably close."""
        try:
            df_2023 = load_training_sample(year=2023, n_files=2, max_ticks=5000)
            df_2024 = load_training_sample(year=2024, n_files=2, max_ticks=5000)
        except FileNotFoundError:
            pytest.skip("Training data not available for both years")

        results = audit_distributions(df_2024, df_2023)

        # Cross-year comparison: just verify it runs without error
        assert 'spread' in results
        assert 'velocity' in results
        assert 'verdict' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
