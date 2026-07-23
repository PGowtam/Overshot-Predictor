import numpy as np

try:
    live_micro = np.load("live_micro_debug.npy")
    test_micro = np.load("/Users/gopo/Quant Projects/CAPSTONE/Overshot/outputs/exec_tensors_regime/test_micro.npy")
    
    print(f"Live Micro Shape: {live_micro.shape}")
    print(f"Test Micro Shape: {test_micro.shape}")
    
    # Compare first brick's micro features
    print("\nLive micro [0, -1, -1, :] (last tick of last brick):")
    print(live_micro[0, -1, -1, :])
    
    print("\nTest micro [0, -1, -1, :] (last tick of last brick):")
    print(test_micro[0, -1, -1, :])
    
except Exception as e:
    print(f"Error: {e}")
