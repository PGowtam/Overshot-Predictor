import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

def test_imports():
    print("Testing Imports...")
    try:
        from config import settings
        print("Config Loaded")
        from data.connector import MT5Connector
        print("Connector Loaded")
        from data.renko import RenkoBuilder
        print("Renko Loaded")
        from data.features_lib import regime # Test lib import
        print("Features Lib Loaded")
        from data.features import FeatureEngineer
        print("Feature Engineer Loaded")
        from models.ensemble import EnsembleAgent
        print("Ensemble Agent Loaded")
        from execution.orbit import OrbitEngine
        print("Orbit Engine Loaded")
        return True
    except Exception as e:
        print(f"Import Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_instantiation():
    print("\nTesting Instantiation...")
    try:
        from execution.orbit import OrbitEngine
        engine = OrbitEngine()
        print("OrbitEngine Instantiated Successfully")
        return True
    except Exception as e:
        print(f"Instantiation Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_imports():
        test_instantiation()
