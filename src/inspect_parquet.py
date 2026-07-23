import pandas as pd
import sys

def main():
    try:
        df = pd.read_parquet("outputs/sim_labels_v4/v4_percentiles_labels.parquet")
        print("Columns in sim_labels:")
        for col in df.columns:
            print(f" - {col}")
        print(f"\nRows: {len(df)}")
        print("\nFirst row sample:")
        print(df.iloc[0])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
