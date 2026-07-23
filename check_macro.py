import numpy as np
X_macro = np.load("/Users/gopo/Quant Projects/CAPSTONE/Overshot/outputs/exec_tensors_regime/test_macro.npy")
print("Local WR distribution:")
print(np.percentile(X_macro[:, -1, 6], [0, 25, 50, 75, 100]))
print("Z-size distribution:")
print(np.percentile(X_macro[:, -1, 2], [0, 25, 50, 75, 100]))
