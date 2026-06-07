import os
import glob
import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("TensorBuilder")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "setup_classifier"
DATASET_DIR = OUTPUT_DIR / "dataset"
TENSOR_DIR = OUTPUT_DIR / "tensors"

def load_all_parquets(dataset_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(dataset_dir / "dataset_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No dataset parquet files found in {dataset_dir}")
    
    logger.info(f"Found {len(files)} parquet files. Loading...")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")
            
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} total rows.")
    return df

def main():
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    df = load_all_parquets(DATASET_DIR)
    
    # 2. Clean data
    # Drop rows with exclude_flag == 1
    initial_len = len(df)
    df = df[df['exclude_flag'] != 1].reset_index(drop=True)
    logger.info(f"Filtered out {initial_len - len(df)} rows with exclude_flag == 1. Remaining rows: {len(df)}")
    
    # Ensure date column is parsed correctly
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # 3. Create splits
    train_mask = df['date'] <= pd.to_datetime('2025-06-30').date()
    val_mask = (df['date'] >= pd.to_datetime('2025-07-01').date()) & (df['date'] <= pd.to_datetime('2025-10-31').date())
    test_mask = df['date'] >= pd.to_datetime('2025-11-01').date()
    
    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)
    
    logger.info(f"Train split size: {len(train_df)} (up to 2025-06-30)")
    logger.info(f"Val split size: {len(val_df)} (2025-07-01 to 2025-10-31)")
    logger.info(f"Test split size: {len(test_df)} (from 2025-11-01)")
    
    # Helper to parse and stack features
    def extract_tensors(split_df):
        if len(split_df) == 0:
            return {}
        
        # Multi-branch inputs
        ancs_fine = np.stack([np.stack(y) for y in split_df['ancs_fine']]).astype(np.float32)
        ancs_coarse = np.stack([np.stack(y) for y in split_df['ancs_coarse']]).astype(np.float32)
        history = np.stack([np.stack([np.stack(z) for z in y]) for y in split_df['history']]).astype(np.float32)
        
        # Concatenate candle_features (15-dim) and momentum (19-dim)
        candle = np.array(split_df['candle_features'].tolist(), dtype=np.float32)
        momentum = np.array(split_df['momentum'].tolist(), dtype=np.float32)
        scalars = np.hstack([candle, momentum])
        
        # Labels and targets
        labels = split_df['label'].values.astype(np.int32)
        t1_win = split_df['t1_win'].values.astype(np.float32)
        t2_win = split_df['t2_win'].values.astype(np.float32)
        t3_win = split_df['t3_win'].values.astype(np.float32)
        t4_win = split_df['t4_win'].values.astype(np.float32)
        
        # Boundary proximity calculation
        # t1 target = 1.0, t2 target = 2.0, t3 target = 2.0, t4 target = 3.0
        t1_prox = np.abs(split_df['t1_y_mag'].values - 1.0)
        t2_prox = np.abs(split_df['t2_y_mag'].values - 2.0)
        t3_prox = np.abs(split_df['t3_y_mag'].values - 2.0)
        t4_prox = np.abs(split_df['t4_y_mag'].values - 3.0)
        
        # Combine dynamically based on the assigned label class
        prox_dict = {1: t1_prox, 2: t2_prox, 3: t3_prox, 4: t4_prox}
        boundary_proximity = np.zeros(len(split_df), dtype=np.float32)
        for i, lbl in enumerate(labels):
            if lbl in prox_dict:
                boundary_proximity[i] = prox_dict[lbl][i]
            else:
                # For label 0 (T0), take the minimum proximity to any boundary
                boundary_proximity[i] = min(t1_prox[i], t2_prox[i], t3_prox[i], t4_prox[i])
                
        date_ints = split_df['date'].apply(lambda x: x.year * 100 + x.month).values.astype(np.int32)
        return {
            "ancs_fine": ancs_fine,
            "ancs_coarse": ancs_coarse,
            "history": history,
            "scalars": scalars,
            "label": labels,
            "t1_win": t1_win,
            "t2_win": t2_win,
            "t3_win": t3_win,
            "t4_win": t4_win,
            "boundary_proximity": boundary_proximity,
            "date_ints": date_ints
        }

    logger.info("Extracting raw tensors for splits...")
    train_tensors = extract_tensors(train_df)
    val_tensors = extract_tensors(val_df)
    test_tensors = extract_tensors(test_df)
    
    # 4. Fit and apply RobustScaler on scalar features
    logger.info("Fitting RobustScaler on Training scalar features...")
    scaler = RobustScaler()
    train_tensors['scalars'] = scaler.fit_transform(train_tensors['scalars']).astype(np.float32)
    
    if val_tensors:
        val_tensors['scalars'] = scaler.transform(val_tensors['scalars']).astype(np.float32)
    if test_tensors:
        test_tensors['scalars'] = scaler.transform(test_tensors['scalars']).astype(np.float32)
        
    scaler_path = OUTPUT_DIR / "scalar_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved fitted scaler to {scaler_path}")
    
    # 5. Compute Class Weights on training labels
    labels_train = train_tensors['label']
    unique_labels, counts = np.unique(labels_train, return_counts=True)
    total_samples = len(labels_train)
    num_classes = 5
    
    class_weights = {}
    for lbl, count in zip(unique_labels, counts):
        class_weights[int(lbl)] = float(total_samples / (num_classes * count))
        
    # Ensure all classes are represented, default to 1.0 if not found
    for i in range(num_classes):
        if i not in class_weights:
            class_weights[i] = 1.0
            
    weights_path = OUTPUT_DIR / "class_weights.json"
    with open(weights_path, "w") as f:
        json.dump(class_weights, f, indent=4)
    logger.info(f"Saved class weights to {weights_path}: {class_weights}")
    
    # 6. Save tensors as compressed .npz files
    def save_npz(split_name, tensors):
        if not tensors:
            return
        path = TENSOR_DIR / f"{split_name}.npz"
        np.savez_compressed(path, **tensors)
        logger.info(f"Saved {split_name} tensors to {path}")
        
    save_npz("train", train_tensors)
    save_npz("val", val_tensors)
    save_npz("test", test_tensors)
    
    logger.info("Tensor building phase completed successfully!")

if __name__ == "__main__":
    main()
