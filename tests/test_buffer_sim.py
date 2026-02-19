"""Unit tests for buffer_sim.py (Phase 3.4)."""

import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from buffer_sim import simulate_buffers, BUFFER_SIZE, IDX_FLAG_CURR, IDX_DECAY


def _make_test_env(tmp_path, brick_ticks: list[int]):
    """Create a minimal Phase 2 output for testing.

    Args:
        tmp_path: pytest tmp_path fixture
        brick_ticks: list of tick counts per brick, e.g. [100, 50, 150]
    """
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True)

    meta_rows = []
    for i, n in enumerate(brick_ticks):
        # Create random tick vectors (N, 9)
        vecs = np.random.randn(n, 9).astype(np.float32)
        # Set Flag_Curr=1, Decay=0 (as Phase 2 produces)
        vecs[:, IDX_FLAG_CURR] = 1.0
        vecs[:, IDX_DECAY] = 0.0
        np.save(feature_dir / f"tick_vectors_{i}.npy", vecs)
        meta_rows.append({
            "brick_id": i,
            "n_ticks": n,
            "start_time": None,
            "end_time": None,
        })

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_parquet(feature_dir / "brick_metadata.parquet", index=False)
    return feature_dir


class TestMicroBuffer:
    """Tests for micro-buffer simulation (FR-BF-01, FR-BF-02)."""

    def test_exactly_100_ticks_no_padding(self, tmp_path):
        """Single brick with 100 ticks → full buffer, no padding."""
        feature_dir = _make_test_env(tmp_path, [100])
        simulate_buffers(feature_dir)

        snap = np.load(feature_dir / "snapshots" / "snapshot_0.npy")
        assert snap.shape == (100, 9)

        # No padding → no zeros at front
        assert not np.all(snap[0] == 0)

        # All Flag_Curr should be 1 (all from current brick)
        assert np.sum(snap[:, IDX_FLAG_CURR] == 1.0) == 100

    def test_50_ticks_padding(self, tmp_path):
        """Single brick with 50 ticks → 50 zeros at front."""
        feature_dir = _make_test_env(tmp_path, [50])
        simulate_buffers(feature_dir)

        snap = np.load(feature_dir / "snapshots" / "snapshot_0.npy")
        assert snap.shape == (100, 9)

        # First 50 rows should be all zeros (padding)
        assert np.all(snap[:50] == 0)

        # Last 50 rows should have data
        assert not np.all(snap[50:] == 0)

    def test_buffer_rollover_150_ticks(self, tmp_path):
        """Two bricks: 120 ticks + 30 ticks → buffer rolls correctly."""
        feature_dir = _make_test_env(tmp_path, [120, 30])
        simulate_buffers(feature_dir)

        # Brick 0: 120 ticks → buffer has 100 (last 100 of 120)
        snap0 = np.load(feature_dir / "snapshots" / "snapshot_0.npy")
        assert snap0.shape == (100, 9)
        # All Flag_Curr=1 (all from brick 0)
        assert np.sum(snap0[:, IDX_FLAG_CURR] == 1.0) == 100

        # Brick 1: 30 new ticks → buffer has 100 (70 from brick 0 + 30 from brick 1)
        snap1 = np.load(feature_dir / "snapshots" / "snapshot_1.npy")
        assert snap1.shape == (100, 9)
        # Flag_Curr=1 count should equal 30 (current brick ticks)
        assert np.sum(snap1[:, IDX_FLAG_CURR] == 1.0) == 30
        # Flag_Curr=0 count should equal 70 (spillover from brick 0)
        assert np.sum(snap1[:, IDX_FLAG_CURR] == 0.0) == 70

    def test_continuity_across_3_bricks(self, tmp_path):
        """Three bricks: verify buffer contents carry over correctly."""
        feature_dir = _make_test_env(tmp_path, [60, 20, 15])
        simulate_buffers(feature_dir)

        snap0 = np.load(feature_dir / "snapshots" / "snapshot_0.npy")
        snap1 = np.load(feature_dir / "snapshots" / "snapshot_1.npy")
        snap2 = np.load(feature_dir / "snapshots" / "snapshot_2.npy")

        # All shapes (100, 9)
        assert snap0.shape == snap1.shape == snap2.shape == (100, 9)

        # Brick 0: 60 ticks → 40 padded + 60 real
        assert np.all(snap0[:40] == 0)

        # Brick 1: 60 + 20 = 80 real ticks → 20 padded
        assert np.all(snap1[:20] == 0)

        # Brick 2: 80 + 15 = 95 real ticks → 5 padded
        assert np.all(snap2[:5] == 0)

        # Continuity: z-scored features (cols 0-5) from brick 0's
        # last ticks should appear in brick 1's snapshot
        # In snap1, the first 60 real rows (after padding at pos 20..79) are from brick 0
        # Compare cols 0-5 and 7 (skipping Flag_Curr=6, Decay=8)
        compare_cols = [0, 1, 2, 3, 4, 5, 7]
        # snap0 real data is at rows 40:100 (60 ticks)
        # snap1 real data from brick 0 is at rows 20:80 (60 ticks from brick 0)
        assert np.allclose(
            snap0[40:, compare_cols],
            snap1[20:80, compare_cols],
            atol=1e-5
        )

    def test_decay_increases_with_brick_distance(self, tmp_path):
        """Decay should increase for ticks from older bricks."""
        feature_dir = _make_test_env(tmp_path, [40, 30, 20])
        simulate_buffers(feature_dir)

        snap2 = np.load(feature_dir / "snapshots" / "snapshot_2.npy")

        # Extract non-padded region (100 - 90 = 10 padded)
        # Total ticks: 40+30+20=90, buffer holds last 90
        # Decay for brick 0 ticks: (2-0)/100 = 0.02
        # Decay for brick 1 ticks: (2-1)/100 = 0.01
        # Decay for brick 2 ticks: (2-2)/100 = 0.0

        # Last 20 rows (brick 2) should have Decay=0
        assert np.allclose(snap2[80:, IDX_DECAY], 0.0)
        # Rows 50-79 (brick 1, 30 ticks) should have Decay=0.01
        assert np.allclose(snap2[50:80, IDX_DECAY], 0.01, atol=1e-6)
        # Rows 10-49 (brick 0, 40 ticks) should have Decay=0.02
        assert np.allclose(snap2[10:50, IDX_DECAY], 0.02, atol=1e-6)

    def test_zero_ticks_brick(self, tmp_path):
        """Brick with 0 ticks → snapshot inherits buffer from prior brick."""
        feature_dir = _make_test_env(tmp_path, [80, 0])
        simulate_buffers(feature_dir)

        snap0 = np.load(feature_dir / "snapshots" / "snapshot_0.npy")
        snap1 = np.load(feature_dir / "snapshots" / "snapshot_1.npy")

        # Brick 1 has 0 ticks, so buffer state is same as brick 0
        # But Flag_Curr is rewritten: all should be 0 (no ticks from brick 1)
        assert np.sum(snap1[:, IDX_FLAG_CURR] == 1.0) == 0
        # Decay should be rewritten: all from brick 0, distance = 1
        # Decay = (1-0)/100 = 0.01
        non_pad = snap1[20:]  # 80 real ticks, 20 padded
        assert np.allclose(non_pad[:, IDX_DECAY], 0.01, atol=1e-6)

    def test_metadata_saved(self, tmp_path):
        """Buffer metadata should be saved correctly."""
        feature_dir = _make_test_env(tmp_path, [50, 120, 10])
        simulate_buffers(feature_dir)

        meta = pd.read_parquet(feature_dir / "buffer_metadata.parquet")
        assert len(meta) == 3
        assert list(meta.columns) == ["brick_id", "n_real_ticks", "n_padded", "n_curr_brick_ticks"]

        # Brick 0: 50 real, 50 padded
        assert meta.iloc[0]["n_real_ticks"] == 50
        assert meta.iloc[0]["n_padded"] == 50

        # Brick 1: 50+120=170 → capped at 100 real, 0 padded
        assert meta.iloc[1]["n_real_ticks"] == 100
        assert meta.iloc[1]["n_padded"] == 0

        # Brick 2: 100+10=110 → capped at 100 real, 0 padded
        assert meta.iloc[2]["n_real_ticks"] == 100
        assert meta.iloc[2]["n_padded"] == 0
