import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta, timezone

def fetch_ohlc_data():
    # Initialize connection to the MetaTrader 5 terminal
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return

    # Set the symbol and timeframe
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M1

    # Set the time range (UTC)
    # Current time in UTC
    utc_to = datetime.now(timezone.utc) + timedelta(hours=2) # Fetch 1 day forward
    # 5 days ago
    utc_from = utc_to - timedelta(days=5) 

    print(f"Fetching data from {utc_from} to {utc_to}...")

    # Copy rates from the defined time range
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)

    # Shutdown connection to the MetaTrader 5 terminal
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("No data received, error code =", mt5.last_error())
        return

    # Create DataFrame from the obtained data
    df = pd.DataFrame(rates)
    
    # Convert time in seconds into the datetime format
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Save to CSV
    csv_file = "xauusd_ohlc.csv"
    df.to_csv(csv_file, index=False)
    print(f"Data saved to {csv_file}")

    print(f"Fetched {len(df)} rows.")
    print("Head:")
    print(df.head())
    print("\nTail:")
    print(df.tail())

    return df

if __name__ == "__main__":
    fetch_ohlc_data()
