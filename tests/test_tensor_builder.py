"""Unit tests for tensor_builder.py (Phase 4.6)."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensor_builder import (
    assign_split, compute_chain_depths, build_and_save_tensors,
    CONTEXT_BRICKS, FAST_BRICK_THRESHOLD, CHAIN_DEPTH_THRESHOLD,
    CHAIN_DURATION_THRESHOLD, SPLIT_DATES,
)


class TestSplitAssignment:
    """Test walk-forward split assignment (4.2)."""

    def test_train_date(self):
        d = pd.Timestamp("2022-06-15", tz="UTC")
        assert assign_split(d) == "train"

    def test_val_date(self):
        d = pd.Timestamp("2023-03-01", tz="UTC")
        assert assign_split(d) == "val"

    def test_test_date(self):
        d = pd.Timestamp("2023-09-01", tz="UTC")
        assert assign_split(d) == "test"

    def test_holdout_date(self):
        d = pd.Timestamp("2024-03-01", tz="UTC")
        assert assign_split(d) == "holdout"

    def test_boundary_val_start(self):
        """2023-01-01 is val, not train."""
        d = pd.Timestamp("2023-01-01", tz="UTC")
        assert assign_split(d) == "val"

    def test_boundary_test_start(self):
        d = pd.Timestamp("2023-07-01", tz="UTC")
        assert assign_split(d) == "test"

    def test_boundary_holdout_start(self):
        d = pd.Timestamp("2024-01-01", tz="UTC")
        assert assign_split(d) == "holdout"


class TestChainDepth:
    """Test fast-brick chain depth calculation (4.3)."""

    def test_no_fast_bricks(self):
        durations = np.array([100.0, 50.0, 200.0, 30.0])
        depths = compute_chain_depths(durations)
        assert list(depths) == [0, 0, 0, 0]

    def test_single_chain(self):
        # Bricks 1,2 are fast (< 10s), brick 3 follows
        durations = np.array([100.0, 5.0, 3.0, 50.0])
        depths = compute_chain_depths(durations)
        assert depths[0] == 0  # No prior
        assert depths[1] == 0  # Prior is 100s (not fast)
        assert depths[2] == 1  # Prior is 5s (fast), then 100s (stop)
        assert depths[3] == 2  # Prior 3s + 5s are fast, then 100s (stop)

    def test_long_chain(self):
        # 7 fast bricks in a row
        durations = np.array([100.0] + [2.0] * 7 + [50.0])
        depths = compute_chain_depths(durations)
        assert depths[0] == 0
        assert depths[7] == 6  # 6 prior fast bricks
        assert depths[8] == 7  # 7 prior fast bricks

    def test_chain_broken(self):
        durations = np.array([5.0, 3.0, 100.0, 2.0, 4.0])
        depths = compute_chain_depths(durations)
        assert depths[2] == 2  # 3s + 5s
        assert depths[3] == 0  # Prior is 100s (not fast)
        assert depths[4] == 1  # Prior is 2s (fast), then 100s (stop)


class TestTensorBuilder:
    """Integration tests for tensor construction (4.1)."""

    def _make_test_env(self, tmp_path, n_bricks=25):
        """Create minimal Phase 2/3 outputs for testing."""
        feature_dir = tmp_path / "features"
        snapshot_dir = feature_dir / "snapshots"
        snapshot_dir.mkdir(parents=True)

        # Create labels
        dates = pd.date_range("2022-06-01", periods=n_bricks, freq="D", tz="UTC")
        labels = pd.DataFrame({
            "brick_id": range(n_bricks),
            "date": dates,
            "y_class": [1.0 if i % 2 == 0 else 0.0 for i in range(n_bricks)],
            "y_mag": [0.5] * n_bricks,
            "duration_seconds": [100.0] * n_bricks,
            "exclude_flag": [False] * n_bricks,
        })
        labels.to_parquet(tmp_path / "labels.parquet", index=False)

        # Create macro vectors
        macros = np.random.randn(n_bricks, 3).astype(np.float32)
        np.save(feature_dir / "macro_vectors.npy", macros)

        # Create snapshots
        for i in range(n_bricks):
            snap = np.random.randn(100, 9).astype(np.float32)
            np.save(snapshot_dir / f"snapshot_{i}.npy", snap)

        return tmp_path

    def test_context_requirement(self, tmp_path, monkeypatch):
        """Bricks with index < 10 should produce no tensor."""
        env = self._make_test_env(tmp_path, n_bricks=15)
        # Monkeypatch paths
        import tensor_builder
        monkeypatch.setattr(tensor_builder, "OUTPUT_DIR", env)
        monkeypatch.setattr(tensor_builder, "FEATURE_DIR", env / "features")
        monkeypatch.setattr(tensor_builder, "SNAPSHOT_DIR", env / "features" / "snapshots")
        monkeypatch.setattr(tensor_builder, "TENSOR_DIR", env / "tensors")

        split_info = build_and_save_tensors()
        total = sum(v["n"] for v in split_info.values() if v["n"])
        # 15 bricks, first 10 are context → 5 samples
        assert total == 5

    def test_sample_weight_assignment(self, tmp_path, monkeypatch):
        """Bricks with chain_depth > 5 get weight 0.5."""
        env = self._make_test_env(tmp_path, n_bricks=25)
        # Make bricks 11-20 fast (duration < 10s) to create chains
        labels = pd.read_parquet(env / "labels.parquet")
        labels.loc[11:20, "duration_seconds"] = 3.0
        labels.to_parquet(env / "labels.parquet", index=False)

        import tensor_builder
        monkeypatch.setattr(tensor_builder, "OUTPUT_DIR", env)
        monkeypatch.setattr(tensor_builder, "FEATURE_DIR", env / "features")
        monkeypatch.setattr(tensor_builder, "SNAPSHOT_DIR", env / "features" / "snapshots")
        monkeypatch.setattr(tensor_builder, "TENSOR_DIR", env / "tensors")

        build_and_save_tensors()
        weights = np.load(env / "tensors" / "train_weights.npy")
        # Some weights should be 0.5 (chain_depth > 5)
        assert np.any(weights == 0.5)
        # Not all should be 0.5
        assert np.any(weights == 1.0)
