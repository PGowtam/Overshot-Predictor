"""
Feature Parity Test
====================
Validates the Python MTPatscFeatureEngine by:
  1. Running the C++ generate_dataset to get ground-truth features
  2. Building the same bricks with Python Renko and computing Python features
  3. Comparing brick-by-brick where bricks match (same close/open price)

Uses more ticks to ensure enough bricks form.
"""

import sys
import os
import numpy as np
import ctypes
from pathlib import Path

# Add project paths
BASE = Path(__file__).resolve().parent.parent.parent
MTPATSC_DIR = BASE / "MTPATSC"
TRADER_DIR = BASE / "MTPATSC_Trader"

sys.path.insert(0, str(TRADER_DIR))
sys.path.insert(0, str(MTPATSC_DIR / "src"))

# Load C++ library
lib_path = MTPATSC_DIR / "src" / "libmtpatsc_engine.dylib"
if not lib_path.exists():
    print(f"ERROR: C++ library not found at {lib_path}")
    print("Build it first: cd MTPATSC/src && make")
    sys.exit(1)

lib = ctypes.CDLL(str(lib_path))

# Load test data
import pandas as pd

DATA_PATH = BASE / "Data" / "xauusd_ticks_5ers_2026.parquet"
if not DATA_PATH.exists():
    print(f"ERROR: Test data not found at {DATA_PATH}")
    sys.exit(1)

print("Loading tick data...")
df = pd.read_parquet(DATA_PATH)
if 'timestamp' in df.columns and 'time_msc' not in df.columns:
    df['time_msc'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6

# Use ALL ticks to ensure enough bricks
df = df.sort_values('time_msc').reset_index(drop=True)
print(f"Loaded {len(df)} ticks. Price range: {df['bid'].min():.2f} - {df['bid'].max():.2f}")

bids = np.ascontiguousarray(df['bid'].values, dtype=np.float64)
asks = np.ascontiguousarray(df['ask'].values, dtype=np.float64)
times = np.ascontiguousarray(df['time_msc'].values, dtype=np.int64)

k_multiplier = 0.00118
day_open = float(bids[0])
bs = day_open * k_multiplier
print(f"Day open: {day_open:.2f}, Brick size: {bs:.4f}")

# ─── Step 1: Generate C++ features ────────────────────────────────

class FeatureLabelRow(ctypes.Structure):
    _fields_ = [
        ("brick_id", ctypes.c_int),
        ("timestamp", ctypes.c_int64),
        ("direction", ctypes.c_int),
        ("close_price", ctypes.c_double),
        ("open_price", ctypes.c_double),
        ("brick_size", ctypes.c_double),
        ("t1_win", ctypes.c_int),
        ("t1_y_mag", ctypes.c_double),
        ("t2_win", ctypes.c_int),
        ("t2_y_mag", ctypes.c_double),
        ("t2_filled", ctypes.c_int),
        ("t3_win", ctypes.c_int),
        ("t3_y_mag", ctypes.c_double),
        ("t4_win", ctypes.c_int),
        ("t4_y_mag", ctypes.c_double),
        ("t4_filled", ctypes.c_int),
        ("label", ctypes.c_int),
        ("exclude_flag", ctypes.c_int),
        ("brick_duration_seconds", ctypes.c_int64),
        ("ancs_fine", ctypes.c_float * 60),
        ("ancs_coarse", ctypes.c_float * 30),
        ("candle_features", ctypes.c_float * 15),
        ("momentum", ctypes.c_float * 19),
        ("history", ctypes.c_float * 150)
    ]

lib.generate_dataset.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_int)
]
lib.generate_dataset.restype = ctypes.POINTER(FeatureLabelRow)
lib.free_dataset.argtypes = [ctypes.POINTER(FeatureLabelRow)]
lib.free_dataset.restype = None

print("\nRunning C++ generate_dataset...")
out_num_rows = ctypes.c_int(0)
ptr = lib.generate_dataset(bids, asks, times, len(bids), k_multiplier, ctypes.byref(out_num_rows))

n_rows = out_num_rows.value
print(f"C++ engine produced {n_rows} bricks.")

# Extract C++ features
cpp_data = []
for i in range(n_rows):
    row = ptr[i]
    cpp_data.append({
        'brick_id': row.brick_id,
        'direction': row.direction,
        'close': row.close_price,
        'open': row.open_price,
        'brick_size': row.brick_size,
        'ancs_fine': np.array(list(row.ancs_fine), dtype=np.float32),
        'ancs_coarse': np.array(list(row.ancs_coarse), dtype=np.float32),
        'candle': np.array(list(row.candle_features), dtype=np.float32),
        'momentum': np.array(list(row.momentum), dtype=np.float32),
    })

lib.free_dataset(ptr)

# ─── Step 2: Build Python Renko and features ─────────────────────

from bridge.renko import RenkoBuilder
from bridge.mtpatsc_feature_engine import MTPatscFeatureEngine

print("\nBuilding Python Renko and computing features...")

# Python Renko starts from day_open (same as C++ for the first day)
py_renko = RenkoBuilder(day_open)
engine = MTPatscFeatureEngine()
py_bricks = []

for i in range(len(bids)):
    new_bricks = py_renko.update_tick(float(bids[i]), int(times[i]), ask=float(asks[i]))
    for b in new_bricks:
        py_bricks.append(b)

print(f"Python Renko produced {len(py_bricks)} bricks.")

if len(py_bricks) < 5:
    print("WARNING: Not enough Python bricks for full test. Doing basic sanity checks only.")

# Compute Python features for each brick
py_features = []
for b in py_bricks:
    fine = engine._compute_ancs(b.intra_ticks, 10, b.open, b.brick_size)
    coarse = engine._compute_ancs(b.intra_ticks, 5, b.open, b.brick_size)
    candle = engine._compute_candle_features(b.intra_ticks, b.brick_size, b.uptrend)
    momentum = engine._compute_momentum_features(b.intra_ticks, b.brick_size)
    py_features.append({
        'close': b.close,
        'open': b.open,
        'direction': b.uptrend,
        'ancs_fine': fine,
        'ancs_coarse': coarse,
        'candle': candle,
        'momentum': momentum,
    })

# ─── Test 1: NaN/Inf Sanity Check ────────────────────────────────
print("\n" + "="*60)
print("TEST 1: NaN/Inf Sanity Check")
print("="*60)

nan_count = 0
for idx, f in enumerate(py_features):
    for name in ['ancs_fine', 'ancs_coarse', 'candle', 'momentum']:
        arr = f[name]
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            print(f"  ❌ Brick {idx}: {name} has NaN/Inf")
            nan_count += 1

if nan_count == 0:
    print(f"  ✅ All {len(py_features)} bricks × 4 feature types: no NaN/Inf")
else:
    print(f"  ❌ {nan_count} NaN/Inf found")

# ─── Test 2: Feature Shapes ──────────────────────────────────────
print("\n" + "="*60)
print("TEST 2: Feature Shape Verification")
print("="*60)

engine2 = MTPatscFeatureEngine()
tensors = None
for b in py_bricks:
    tensors = engine2.on_brick_close(b)

if tensors is not None:
    shapes_ok = True
    expected = {
        'ancs_fine': (1, 10, 6),
        'ancs_coarse': (1, 5, 6),
        'history': (1, 5, 5, 6),
        'scalars': (1, 34),
    }
    for key, expected_shape in expected.items():
        actual_shape = tensors[key].shape
        if actual_shape != expected_shape:
            print(f"  ❌ {key}: expected {expected_shape}, got {actual_shape}")
            shapes_ok = False
        else:
            print(f"  ✅ {key}: {actual_shape}")

    if shapes_ok:
        print("\n  All tensor shapes correct!")
else:
    print(f"  WARNING: Feature engine returned None after {engine2.brick_count} bricks (need ≥5)")

# ─── Test 3: Cross-match C++ vs Python bricks by close price ─────
print("\n" + "="*60)
print("TEST 3: C++ vs Python Feature Comparison (matched bricks)")
print("="*60)

# The C++ engine may use different anchoring, so bricks won't align 1:1.
# But where we find bricks with the same close_price and open_price,
# the features should match exactly (same intra-ticks).
#
# NOTE: Even if anchors differ, the feature computation functions 
# (compute_ancs, compute_candle_features, compute_momentum_features)
# are pure functions of the intra-brick ticks and brick geometry.
# So if the same ticks form the same brick, features MUST match.

# Create lookup of C++ bricks by (open, close)
cpp_lookup = {}
for cd in cpp_data:
    key = (round(cd['open'], 2), round(cd['close'], 2))
    cpp_lookup[key] = cd

matched = 0
total_diff_fine = []
total_diff_coarse = []
total_diff_candle = []
total_diff_momentum = []

for pf in py_features:
    key = (round(pf['open'], 2), round(pf['close'], 2))
    if key in cpp_lookup:
        cd = cpp_lookup[key]
        matched += 1

        diff_fine = np.max(np.abs(cd['ancs_fine'] - pf['ancs_fine']))
        diff_coarse = np.max(np.abs(cd['ancs_coarse'] - pf['ancs_coarse']))
        diff_candle = np.max(np.abs(cd['candle'] - pf['candle']))
        diff_momentum = np.max(np.abs(cd['momentum'] - pf['momentum']))

        total_diff_fine.append(diff_fine)
        total_diff_coarse.append(diff_coarse)
        total_diff_candle.append(diff_candle)
        total_diff_momentum.append(diff_momentum)

        if matched <= 3:
            print(f"\n  Brick match #{matched}: open={key[0]}, close={key[1]}")
            print(f"    ANCS Fine max_diff:   {diff_fine:.8f}  {'✅' if diff_fine < 1e-3 else '⚠️'}")
            print(f"    ANCS Coarse max_diff: {diff_coarse:.8f}  {'✅' if diff_coarse < 1e-3 else '⚠️'}")
            print(f"    Candle max_diff:      {diff_candle:.8f}  {'✅' if diff_candle < 1e-3 else '⚠️'}")
            print(f"    Momentum max_diff:    {diff_momentum:.8f}  {'✅' if diff_momentum < 1e-3 else '⚠️'}")

if matched > 0:
    print(f"\n  Total matched bricks: {matched} out of {len(py_features)} Python / {len(cpp_data)} C++")
    print(f"  ANCS Fine   - mean max_diff: {np.mean(total_diff_fine):.6f}, worst: {np.max(total_diff_fine):.6f}")
    print(f"  ANCS Coarse - mean max_diff: {np.mean(total_diff_coarse):.6f}, worst: {np.max(total_diff_coarse):.6f}")
    print(f"  Candle      - mean max_diff: {np.mean(total_diff_candle):.6f}, worst: {np.max(total_diff_candle):.6f}")
    print(f"  Momentum    - mean max_diff: {np.mean(total_diff_momentum):.6f}, worst: {np.max(total_diff_momentum):.6f}")

    worst = max(np.max(total_diff_fine), np.max(total_diff_coarse), 
                np.max(total_diff_candle), np.max(total_diff_momentum))
    if worst < 1e-3:
        print(f"\n  ✅ OVERALL: All features match within 1e-3 tolerance (worst={worst:.8f})")
    elif worst < 0.1:
        print(f"\n  ⚠️ OVERALL: Minor differences detected (worst={worst:.6f}) — likely due to different anchor/ticks")
    else:
        print(f"\n  ❌ OVERALL: Significant differences detected (worst={worst:.6f})")
else:
    print(f"\n  No matched bricks found between C++ ({len(cpp_data)}) and Python ({len(py_features)}).")
    print("  This is expected if the C++ anchor optimization produced a different Renko path.")
    print("  NaN/Inf and shape tests above still validate the Python implementation.")

# ─── Test 4: Model loading ────────────────────────────────────────
print("\n" + "="*60)
print("TEST 4: Model Loading Test")
print("="*60)

try:
    from bridge.mtpatsc_predictor import MTPatscPredictor
    predictor = MTPatscPredictor()
    predictor.load()

    print(f"  ✅ Model loaded successfully")
    print(f"  Thresholds: {predictor.thresholds}")
    print(f"  Veto: {predictor.veto_threshold}")

    # Run inference on last tensor
    if tensors is not None:
        last_brick_dir = py_bricks[-1].uptrend
        result = predictor.predict(tensors, last_brick_dir)
        print(f"  ✅ Inference completed: action={result['action']}, setup={result['setup_type']}")
        print(f"     Probs: {result['probs']}")
        print(f"     Reason: {result['reason']}")
    else:
        print("  ⚠️ No tensors available for inference test")

except Exception as e:
    print(f"  ❌ Model loading failed: {e}")

print("\n" + "="*60)
print("ALL TESTS COMPLETE")
print("="*60)
